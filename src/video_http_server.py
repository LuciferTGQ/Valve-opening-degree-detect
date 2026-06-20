import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
import warnings
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import cgi


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from video_predictor import (
    FUSION_MODES,
    SUPPORTED_ALGORITHMS,
    estimate_stiction,
    predict_side_fusion,
    predict_video,
)


DEFAULT_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads", "videos")
DEFAULT_RESULT_DIR = os.path.join(PROJECT_ROOT, "output", "video_http")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
HTTP_ALGORITHMS = SUPPORTED_ALGORITHMS | FUSION_MODES


def _safe_filename(filename: str) -> str:
    """清理客户端文件名，避免路径穿越和特殊字符影响保存。"""
    name = os.path.basename(filename or "upload.mp4")
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._-") or "upload"
    ext = ext.lower() if ext.lower() in VIDEO_EXTENSIONS else ".mp4"
    return f"{stem}{ext}"


def _parse_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    return int(value)


def _first(params: dict[str, list[str]], key: str, default: Optional[str] = None) -> Optional[str]:
    values = params.get(key)
    return values[0] if values else default


class VideoPredictHTTPHandler(BaseHTTPRequestHandler):
    """接收手机 App 上传的视频，并调用逐帧阀门开度预测。"""

    upload_dir = DEFAULT_UPLOAD_DIR
    result_dir = DEFAULT_RESULT_DIR
    max_upload_bytes = 512 * 1024 * 1024

    server_version = "ValveVideoHTTP/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "valve-video-predictor",
                    "supported_algorithms": sorted(HTTP_ALGORITHMS),
                },
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "usage": "POST /predict-video with multipart field 'video' and form field 'algorithm'",
                "supported_algorithms": sorted(HTTP_ALGORITHMS),
            },
        )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/predict-video", "/predict_video", "/upload"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})
            return

        try:
            response = self._handle_predict_video(parsed.query)
            self._send_json(HTTPStatus.OK, response)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    def _handle_predict_video(self, query: str) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            raise ValueError("empty request body")
        if content_length > self.max_upload_bytes:
            limit_mb = self.max_upload_bytes / 1024 / 1024
            raise ValueError(f"uploaded video is too large, limit is {limit_mb:.0f} MB")

        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        params = parse_qs(query, keep_blank_values=True)
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            form = self._parse_multipart_form()
            video_path, form_params = self._save_multipart_video(form)
            params.update(form_params)
        else:
            video_path = self._save_raw_video(params, content_length)

        algorithm = (_first(params, "algorithm") or _first(params, "method") or "side_cv").lower().strip()
        if algorithm not in HTTP_ALGORITHMS:
            raise ValueError(f"unsupported algorithm: {algorithm}. Supported: {sorted(HTTP_ALGORITHMS)}")

        model_path = _first(params, "model_path") or _first(params, "model") or os.getenv("VALVE_MODEL_PATH")
        frame_stride = _parse_int(_first(params, "frame_stride"), 1)
        max_frames = _parse_int(_first(params, "max_frames"), None)
        strict = (_first(params, "strict", "false") or "false").lower() in {"1", "true", "yes"}
        fusion_low = float(_first(params, "fusion_low", "10") or "10")
        fusion_high = float(_first(params, "fusion_high", "70") or "70")
        fusion_transition = float(_first(params, "fusion_transition", "5") or "5")
        fusion_cv_probe_interval = int(_first(params, "fusion_cv_probe_interval", "15") or "15")
        fusion_probe_margin = float(_first(params, "fusion_probe_margin", "5") or "5")

        request_id = uuid.uuid4().hex
        csv_path = os.path.join(self.result_dir, f"{request_id}_{algorithm}.csv")
        plot_path = os.path.join(self.result_dir, f"{request_id}_{algorithm}.png")
        started = time.time()
        if algorithm in FUSION_MODES:
            results = predict_side_fusion(
                video_path=video_path,
                model_path=model_path,
                output_csv=csv_path,
                output_plot=plot_path,
                frame_stride=frame_stride or 1,
                max_frames=max_frames,
                strict=strict,
                verbose=True,
                low_threshold=fusion_low,
                high_threshold=fusion_high,
                transition_width=fusion_transition,
                cv_probe_interval=fusion_cv_probe_interval,
                probe_margin=fusion_probe_margin,
            )
        else:
            results = predict_video(
                video_path=video_path,
                algorithm=algorithm,
                model_path=model_path,
                output_csv=csv_path,
                output_plot=plot_path,
                frame_stride=frame_stride or 1,
                max_frames=max_frames,
                strict=strict,
                verbose=True,
            )
        elapsed_ms = round((time.time() - started) * 1000.0, 1)
        if algorithm in FUSION_MODES:
            failed = sum(1 for result in results if result.fusion_angle is None)
        else:
            failed = sum(1 for result in results if result.error)
        stiction = estimate_stiction(results)

        return {
            "ok": True,
            "request_id": request_id,
            "algorithm": algorithm,
            "video_path": video_path,
            "csv_path": csv_path,
            "plot_path": plot_path,
            "frame_count": len(results),
            "failed_count": failed,
            "stiction_level": stiction.level,
            "elapsed_ms": elapsed_ms,
            "results": [asdict(result) for result in results],
        }

    def _parse_multipart_form(self) -> cgi.FieldStorage:
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
            keep_blank_values=True,
        )

    def _save_multipart_video(self, form: cgi.FieldStorage) -> tuple[str, dict[str, list[str]]]:
        form_params: dict[str, list[str]] = {}
        file_item = None

        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                item = item[0]
            if getattr(item, "filename", None):
                if key in {"video", "file", "upload"} or file_item is None:
                    file_item = item
                continue
            form_params[key] = [item.value]

        if file_item is None:
            raise ValueError("multipart request must include a video file field named 'video' or 'file'")

        filename = _safe_filename(file_item.filename)
        save_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
        video_path = os.path.join(self.upload_dir, save_name)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file_item.file, f)

        if os.path.getsize(video_path) == 0:
            raise ValueError("uploaded video file is empty")
        return video_path, form_params

    def _save_raw_video(self, params: dict[str, list[str]], content_length: int) -> str:
        filename = _safe_filename(_first(params, "filename", "upload.mp4") or "upload.mp4")
        save_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
        video_path = os.path.join(self.upload_dir, save_name)

        remaining = content_length
        with open(video_path, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

        if os.path.getsize(video_path) == 0:
            raise ValueError("uploaded video file is empty")
        return video_path

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    upload_dir: str = DEFAULT_UPLOAD_DIR,
    result_dir: str = DEFAULT_RESULT_DIR,
    max_upload_mb: int = 512,
) -> None:
    """启动视频上传预测 HTTP 服务。"""
    handler = VideoPredictHTTPHandler
    handler.upload_dir = os.path.abspath(upload_dir)
    handler.result_dir = os.path.abspath(result_dir)
    handler.max_upload_bytes = max_upload_mb * 1024 * 1024

    server = ThreadingHTTPServer((host, port), handler)
    print(f"Valve video HTTP server listening on http://{host}:{port}")
    print("POST endpoint: /predict-video")
    print(f"Upload dir: {handler.upload_dir}")
    print(f"Result dir: {handler.result_dir}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP service for valve video prediction.")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8000, help="Listen port")
    parser.add_argument("--upload-dir", default=DEFAULT_UPLOAD_DIR, help="Directory for uploaded videos")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR, help="Directory for generated CSV files")
    parser.add_argument("--max-upload-mb", type=int, default=512, help="Maximum upload size in MB")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        upload_dir=args.upload_dir,
        result_dir=args.result_dir,
        max_upload_mb=args.max_upload_mb,
    )


if __name__ == "__main__":
    main()
