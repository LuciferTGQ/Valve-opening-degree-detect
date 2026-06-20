import argparse
import json
import os
import re
import shutil
import time
import uuid
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import cgi


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads", "demo_videos")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _safe_filename(filename: str) -> str:
    """清理客户端上传文件名，避免路径穿越和特殊字符问题。"""
    name = os.path.basename(filename or "upload.mp4")
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._-") or "upload"
    ext = ext.lower() if ext.lower() in VIDEO_EXTENSIONS else ".mp4"
    return f"{stem}{ext}"


class VideoUploadDemoHandler(BaseHTTPRequestHandler):
    """最小视频上传 Demo：接收 Unity 发来的视频并保存到本地。"""

    upload_dir = DEFAULT_UPLOAD_DIR
    max_upload_bytes = 512 * 1024 * 1024
    fake_delay_seconds = 3.0
    fake_frame_count = 10
    server_version = "ValveVideoUploadDemo/1.0"

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "video-upload-demo"})
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "usage": "POST /upload-video with multipart field 'video', or raw binary body",
                "upload_dir": os.path.abspath(self.upload_dir),
            },
        )

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/upload-video", "/upload", "/video"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})
            return

        try:
            saved_path = self._save_upload()
            time.sleep(self.fake_delay_seconds)
            fake_results = self._build_fake_results()
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "message": "video saved and fake processed",
                    "request_id": uuid.uuid4().hex,
                    "filename": os.path.basename(saved_path),
                    "path": saved_path,
                    "size_bytes": os.path.getsize(saved_path),
                    "processing_seconds": self.fake_delay_seconds,
                    "frame_count": len(fake_results),
                    "results": fake_results,
                },
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    def _save_upload(self) -> str:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            raise ValueError("empty request body")
        if content_length > self.max_upload_bytes:
            limit_mb = self.max_upload_bytes / 1024 / 1024
            raise ValueError(f"uploaded video is too large, limit is {limit_mb:.0f} MB")

        os.makedirs(self.upload_dir, exist_ok=True)
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            return self._save_multipart()
        return self._save_raw_binary(content_length)

    def _save_multipart(self) -> str:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
            keep_blank_values=True,
        )

        file_item = None
        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                item = item[0]
            if getattr(item, "filename", None):
                if key in {"video", "file", "upload"} or file_item is None:
                    file_item = item

        if file_item is None:
            raise ValueError("multipart request must include a video file field named 'video' or 'file'")

        filename = _safe_filename(file_item.filename)
        save_path = self._new_save_path(filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file_item.file, f)
        self._ensure_non_empty(save_path)
        return save_path

    def _save_raw_binary(self, content_length: int) -> str:
        query = parse_qs(urlparse(self.path).query)
        filename = _safe_filename((query.get("filename") or ["upload.mp4"])[0])
        save_path = self._new_save_path(filename)

        remaining = content_length
        with open(save_path, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

        self._ensure_non_empty(save_path)
        return save_path

    def _new_save_path(self, filename: str) -> str:
        prefix = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        return os.path.abspath(os.path.join(self.upload_dir, f"{prefix}_{filename}"))

    @staticmethod
    def _ensure_non_empty(path: str) -> None:
        if os.path.getsize(path) == 0:
            raise ValueError("uploaded video file is empty")

    def _build_fake_results(self) -> list[dict[str, Any]]:
        """构造假的逐帧阀门开度结果，用于 Unity 端联调解析逻辑。"""
        results = []
        for index in range(self.fake_frame_count):
            results.append(
                {
                    "frame_index": index,
                    "timestamp_ms": round(index * 200.0, 1),
                    "angle": round(20.0 + index * 2.5, 1),
                    "algorithm": "fake_demo",
                    "error": None,
                }
            )
        return results

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def run_server(
    host: str,
    port: int,
    upload_dir: str,
    max_upload_mb: int,
    fake_delay_seconds: float,
    fake_frame_count: int,
) -> None:
    VideoUploadDemoHandler.upload_dir = os.path.abspath(upload_dir)
    VideoUploadDemoHandler.max_upload_bytes = max_upload_mb * 1024 * 1024
    VideoUploadDemoHandler.fake_delay_seconds = max(0.0, fake_delay_seconds)
    VideoUploadDemoHandler.fake_frame_count = max(0, fake_frame_count)
    server = ThreadingHTTPServer((host, port), VideoUploadDemoHandler)
    print(f"Video upload demo listening on http://{host}:{port}")
    print("POST endpoint: /upload-video")
    print(f"Upload dir: {VideoUploadDemoHandler.upload_dir}")
    print(f"Fake delay: {VideoUploadDemoHandler.fake_delay_seconds}s")
    print(f"Fake frame count: {VideoUploadDemoHandler.fake_frame_count}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal demo server for receiving Unity video uploads.")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8000, help="Listen port")
    parser.add_argument("--upload-dir", default=DEFAULT_UPLOAD_DIR, help="Directory for uploaded videos")
    parser.add_argument("--max-upload-mb", type=int, default=512, help="Maximum upload size in MB")
    parser.add_argument("--fake-delay", type=float, default=3.0, help="Seconds to wait before returning fake results")
    parser.add_argument("--fake-frame-count", type=int, default=10, help="Number of fake frame results to return")
    args = parser.parse_args()

    run_server(
        args.host,
        args.port,
        args.upload_dir,
        args.max_upload_mb,
        args.fake_delay,
        args.fake_frame_count,
    )


if __name__ == "__main__":
    main()
