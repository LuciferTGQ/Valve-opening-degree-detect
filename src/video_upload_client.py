import argparse
import json
import mimetypes
import os
import uuid
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _multipart_body(
    video_path: str,
    algorithm: Optional[str],
    frame_stride: int,
    max_frames: Optional[int],
    fusion_low: float,
    fusion_high: float,
    fusion_transition: float,
    fusion_cv_probe_interval: int,
    fusion_probe_margin: float,
) -> tuple[bytes, str]:
    """构造与 Unity WWWForm 等价的 multipart/form-data 请求体。"""
    boundary = f"----ValveVideoBoundary{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def add_text(name: str, value: str) -> None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    if algorithm:
        add_text("algorithm", algorithm)
    add_text("frame_stride", str(frame_stride))
    if max_frames is not None:
        add_text("max_frames", str(max_frames))
    if algorithm == "side_fusion":
        add_text("fusion_low", str(fusion_low))
        add_text("fusion_high", str(fusion_high))
        add_text("fusion_transition", str(fusion_transition))
        add_text("fusion_cv_probe_interval", str(fusion_cv_probe_interval))
        add_text("fusion_probe_margin", str(fusion_probe_margin))

    filename = os.path.basename(video_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="video"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(video_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), boundary


def upload_video(
    url: str,
    video_path: str,
    algorithm: Optional[str] = None,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    timeout: float = 300.0,
    fusion_low: float = 10.0,
    fusion_high: float = 70.0,
    fusion_transition: float = 5.0,
    fusion_cv_probe_interval: int = 15,
    fusion_probe_margin: float = 5.0,
) -> dict:
    """上传视频并返回服务端 JSON，可用于模拟 Unity 上传行为。"""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    if frame_stride <= 0:
        raise ValueError("frame_stride must be greater than 0")

    body, boundary = _multipart_body(
        video_path,
        algorithm,
        frame_stride,
        max_frames,
        fusion_low,
        fusion_high,
        fusion_transition,
        fusion_cv_probe_interval,
        fusion_probe_margin,
    )
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"unable to connect to {url}: {exc.reason}") from exc

    return json.loads(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a video to the valve HTTP service.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/upload-video", help="Upload endpoint")
    parser.add_argument("--video", required=True, help="Local video path")
    parser.add_argument(
        "--algorithm",
        choices=["side_cv", "side_cnn", "top_cv", "top_cnn", "side_fusion"],
        default=None,
        help="Required by /predict-video; omit for the fake upload demo",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every N frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum sampled frames")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds")
    parser.add_argument("--fusion-low", type=float, default=10.0, help="Low-angle CV threshold")
    parser.add_argument("--fusion-high", type=float, default=70.0, help="High-angle CV threshold")
    parser.add_argument("--fusion-transition", type=float, default=5.0, help="Fusion transition width")
    parser.add_argument("--fusion-cv-probe-interval", type=int, default=15, help="CV probe interval in CNN mode")
    parser.add_argument("--fusion-probe-margin", type=float, default=5.0, help="CV probe threshold margin")
    parser.add_argument("--output-json", default=None, help="Optional path for saving response JSON")
    args = parser.parse_args()

    result = upload_video(
        url=args.url,
        video_path=args.video,
        algorithm=args.algorithm,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        timeout=args.timeout,
        fusion_low=args.fusion_low,
        fusion_high=args.fusion_high,
        fusion_transition=args.fusion_transition,
        fusion_cv_probe_interval=args.fusion_cv_probe_interval,
        fusion_probe_margin=args.fusion_probe_margin,
    )
    formatted = json.dumps(result, ensure_ascii=False, indent=2)
    print(formatted)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(formatted)
            f.write("\n")


if __name__ == "__main__":
    main()
