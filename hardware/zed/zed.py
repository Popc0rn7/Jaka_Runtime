"""Minimal RGB-only ZED 2i camera driver using OpenCV and Linux V4L2.

The ZED 2i is exposed as a UVC side-by-side video device.  This module uses
only ``/dev/video*``; it deliberately has no dependency on the ZED SDK, CUDA,
depth, or tracking APIs.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

Eye = Literal["left", "right", "stereo"]


class ZedCamera:
    """Read RGB frames from the ZED 2i V4L2/UVC stream.

    ``read()`` returns an HWC, ``uint8`` RGB array, ready for
    :class:`src.data_collector.LeRobotDataCollector`.  The native ZED UVC
    stream is side-by-side; the default ``eye="left"`` returns one camera.

    Args:
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 1344,
        height: int = 376,
        fps: int = 30,
        fourcc: str = "",
        eye: Eye = "left",
        output_width: int | None = None,
        output_height: int | None = None,
        startup_timeout: float = 3.0,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        if eye not in ("left", "right", "stereo"):
            raise ValueError(f"eye must be 'left', 'right', or 'stereo', got {eye!r}")
        if bool(output_width) != bool(output_height):
            raise ValueError("output_width and output_height must be provided together")
        if output_width is not None and (
            output_width <= 0 or output_height is None or output_height <= 0
        ):
            raise ValueError("output_width and output_height must be positive")
        if fourcc and len(fourcc) != 4:
            raise ValueError("fourcc must be empty or exactly four characters")
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")

        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.eye = eye
        self.output_width = output_width
        self.output_height = output_height
        self.startup_timeout = startup_timeout
        self._cv2 = None
        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_ready = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._worker_error: Exception | None = None

    def start(self) -> None:
        """Open the device and start the latest-frame capture thread.

        Startup waits for the first valid RGB frame. After that, :meth:`read`
        only reads an in-memory reference and never waits for V4L2.
        """
        if self._capture is not None:
            if self._thread is not None and self._thread.is_alive():
                return
            self.stop()
        if not Path(self.device).exists():
            raise FileNotFoundError(
                f"V4L2 device not found: {self.device}. Run `v4l2-ctl --list-devices`."
            )

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required; install the project's dependencies."
            ) from error

        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open V4L2 device: {self.device}")

        # Set pixel format first: V4L2 may reset the negotiated dimensions when
        # the format changes.
        if self.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        # Best effort only: some OpenCV/V4L2 builds ignore this property. The
        # background loop still continuously drains the driver queue.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stop_event.clear()
        self._frame_ready.clear()
        with self._frame_lock:
            self._latest_frame = None
            self._worker_error = None
        self._cv2 = cv2
        self._capture = capture
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(capture, cv2),
            name="zed-v4l2-capture",
            daemon=True,
        )
        self._thread.start()

        if not self._frame_ready.wait(timeout=self.startup_timeout):
            self.stop()
            raise RuntimeError(
                f"Timed out after {self.startup_timeout:.1f}s waiting for a frame from {self.device}"
            )
        with self._frame_lock:
            worker_error = self._worker_error
            has_frame = self._latest_frame is not None
        if worker_error is not None or not has_frame:
            self.stop()
            detail = str(worker_error) if worker_error is not None else "no frame received"
            raise RuntimeError(f"Could not start camera {self.device}: {detail}")

    def read(self) -> np.ndarray:
        """Immediately return the newest uint8 RGB HWC frame in memory."""
        if self._capture is None or self._thread is None:
            raise RuntimeError("Camera is not started. Call start() before read().")

        with self._frame_lock:
            frame_rgb = self._latest_frame
            worker_error = self._worker_error
        if worker_error is not None:
            raise RuntimeError(f"Camera capture stopped: {worker_error}") from worker_error
        if frame_rgb is None:
            raise RuntimeError("No camera frame is available")
        # The worker replaces this array reference for every frame and never
        # mutates an array after publishing it, so returning it requires no copy.
        return frame_rgb

    def _capture_loop(self, capture: object, cv2: object) -> None:
        """Continuously drain V4L2 and publish only the most recent RGB frame."""
        try:
            while not self._stop_event.is_set():
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    if self._stop_event.is_set():
                        break
                    raise RuntimeError(f"Failed to read a frame from {self.device}")
                frame_rgb = self._prepare_frame(frame_bgr, cv2)
                with self._frame_lock:
                    self._latest_frame = frame_rgb
                self._frame_ready.set()
        except Exception as error:
            with self._frame_lock:
                self._worker_error = error
            self._frame_ready.set()

    def _prepare_frame(self, frame_bgr: np.ndarray, cv2: object) -> np.ndarray:
        """Crop, resize, and convert one captured frame to RGB."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise RuntimeError(
                f"Expected a three-channel color frame, got shape {frame_bgr.shape}"
            )

        if self.eye != "stereo":
            frame_width = frame_bgr.shape[1]
            if frame_width % 2:
                raise RuntimeError(
                    f"Expected a side-by-side frame with even width, got {frame_width}"
                )
            midpoint = frame_width // 2
            frame_bgr = (
                frame_bgr[:, :midpoint]
                if self.eye == "left"
                else frame_bgr[:, midpoint:]
            )

        if self.output_width is not None and self.output_height is not None:
            frame_bgr = cv2.resize(
                frame_bgr,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        """Stop background capture and release V4L2. Safe to call repeatedly."""
        self._stop_event.set()
        thread = self._thread
        capture = self._capture

        # Normally read() returns within one camera period. If it does not,
        # release the device to unblock the worker before the final join.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive() and capture is not None:
                capture.release()
                thread.join(timeout=1.0)
        if capture is not None:
            capture.release()

        self._thread = None
        self._capture = None
        self._cv2 = None
        with self._frame_lock:
            self._latest_frame = None
            self._worker_error = None
        self._frame_ready.clear()

    close = stop

    def __enter__(self) -> "ZedCamera":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()
