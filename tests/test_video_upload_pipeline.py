import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.video_http_server import VideoPredictHTTPHandler
from src.video_upload_client import upload_video
from src.video_upload_demo import VideoUploadDemoHandler


def _create_test_video(path: str, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create the test MP4 video")

    for index in range(frame_count):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[35:80, 35:80] = (0, 0, 255)
        frame[35:80, 80:125] = (0, 255, 0)
        cv2.putText(frame, str(index), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


class _RunningServer:
    def __init__(self, handler):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class VideoUploadPipelineTest(unittest.TestCase):
    def test_fake_upload_demo_returns_frame_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "unity_demo.mp4")
            _create_test_video(video_path)

            VideoUploadDemoHandler.upload_dir = os.path.join(tmpdir, "uploads")
            VideoUploadDemoHandler.fake_delay_seconds = 0.01
            VideoUploadDemoHandler.fake_frame_count = 4

            with _RunningServer(VideoUploadDemoHandler) as base_url:
                response = upload_video(f"{base_url}/upload-video", video_path, timeout=30)

            self.assertTrue(response["ok"])
            self.assertTrue(os.path.isfile(response["path"]))
            self.assertEqual(response["frame_count"], 4)
            self.assertEqual(response["results"][0]["algorithm"], "fake_demo")

    def test_predict_upload_runs_top_cv_and_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "unity_predict.mp4")
            _create_test_video(video_path)

            VideoPredictHTTPHandler.upload_dir = os.path.join(tmpdir, "uploads")
            VideoPredictHTTPHandler.result_dir = os.path.join(tmpdir, "results")

            with _RunningServer(VideoPredictHTTPHandler) as base_url:
                response = upload_video(
                    f"{base_url}/predict-video",
                    video_path,
                    algorithm="top_cv",
                    max_frames=2,
                    timeout=60,
                )

            self.assertTrue(response["ok"])
            self.assertEqual(response["algorithm"], "top_cv")
            self.assertEqual(response["frame_count"], 2)
            self.assertEqual(response["failed_count"], 0)
            self.assertIn(response["stiction_level"], {1, 2, 3})
            self.assertTrue(os.path.isfile(response["csv_path"]))
            self.assertTrue(os.path.isfile(response["plot_path"]))
            self.assertTrue(all(item["angle"] is not None for item in response["results"]))


if __name__ == "__main__":
    unittest.main()
