"""Browser-based dual-camera preview for headless teleoperation hosts.

The display owns no camera devices.  Teleoperation publishes the exact RGB
arrays that it records, and an HTTP server exposes the newest frames as two
MJPEG streams.  This keeps the preview synchronized with collection while
allowing the server itself to run without a desktop session.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

from flask import Flask, Response, abort, render_template_string
from werkzeug.serving import BaseWSGIServer, make_server

if TYPE_CHECKING:
    import numpy as np


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Teleoperation cameras</title>
  <style>
    body { margin: 0; background: #101114; color: #f4f4f5; font-family: sans-serif; }
    header { padding: 16px 24px; font-size: 18px; font-weight: 600; }
    main { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; padding: 0 24px 24px; }
    section { background: #1b1d22; border-radius: 8px; overflow: hidden; }
    h2 { margin: 0; padding: 10px 12px; font-size: 14px; font-weight: 500; }
    img { display: block; width: 100%; height: auto; background: #000; }
  </style>
</head>
<body>
  <header>Teleoperation live preview</header>
  <main>{% for camera in cameras %}
    <section><h2>{{ camera }}</h2><img src="{{ url_for('stream', camera=camera) }}" alt="{{ camera }} live stream"></section>
  {% endfor %}</main>
</body>
</html>"""


class DualCameraDisplay:
    """Serve the latest frames from the teleop loop to a web browser.

    Bind to ``127.0.0.1`` on a robot server and view it through SSH local port
    forwarding.  This avoids exposing an unauthenticated camera feed on the
    network.  ``publish`` does not encode or copy frames, so it has negligible
    impact on the collection control loop.
    """

    def __init__(
        self,
        camera_names: tuple[str, str] = ("agent_view", "wrist"),
        *,
        enabled: bool = True,
        host: str = "127.0.0.1",
        port: int = 8765,
        fps: float = 10,
        jpeg_quality: int = 80,
    ) -> None:
        if len(camera_names) != 2 or len(set(camera_names)) != 2:
            raise ValueError("camera_names must contain exactly two unique names")
        if not 0 < port < 65536:
            raise ValueError("port must be in the range 1..65535")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in the range 1..100")

        self.enabled = enabled
        self.camera_names = camera_names
        self.host = host
        self.port = port
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._frames: dict[str, np.ndarray | None] = {name: None for name in camera_names}
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._server: BaseWSGIServer | None = None
        self._thread: threading.Thread | None = None

        app = Flask(__name__)

        @app.get("/")
        def index() -> str:
            return render_template_string(_PAGE, cameras=self.camera_names)

        @app.get("/stream/<camera>")
        def stream(camera: str) -> Response:
            if camera not in self._frames:
                abort(404)
            return Response(
                self._mjpeg(camera),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/healthz")
        def healthz() -> tuple[dict[str, bool], int]:
            with self._lock:
                ready = all(frame is not None for frame in self._frames.values())
            return {"ready": ready}, 200 if ready else 503

        self.app = app

    def start(self) -> None:
        """Start the local HTTP server in a daemon thread."""
        if not self.enabled:
            return
        if self._server is not None:
            return
        server = make_server(self.host, self.port, self.app, threaded=True)
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="teleop-camera-display",
            daemon=True,
        )
        self._thread.start()
        print(f"Camera preview: http://{self.host}:{self.port}")

    def publish(self, frames: Mapping[str, np.ndarray]) -> None:
        """Publish the two RGB frames captured for the current collection tick."""
        if not self.enabled:
            return
        if set(frames) != set(self.camera_names):
            raise ValueError(f"Expected frames for {self.camera_names}, got {sorted(frames)}")
        with self._new_frame:
            self._frames.update(frames)
            self._new_frame.notify_all()

    def stop(self) -> None:
        """Stop serving previews. Safe to call more than once."""
        if not self.enabled:
            return
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _mjpeg(self, camera: str):
        interval = 1.0 / self.fps
        last_frame: np.ndarray | None = None
        while True:
            started = time.monotonic()
            with self._new_frame:
                if self._frames[camera] is last_frame:
                    self._new_frame.wait(timeout=1.0)
                frame = self._frames[camera]
            if frame is None:
                continue
            last_frame = frame
            jpeg = self._encode_jpeg(frame)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for the camera preview") from error
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype.name != "uint8":
            raise ValueError("Preview frames must be uint8 RGB HWC arrays")
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode preview frame")
        return encoded.tobytes()
