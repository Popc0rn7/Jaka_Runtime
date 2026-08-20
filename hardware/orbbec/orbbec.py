"""Minimal RGB-only Orbbec Gemini camera driver using pyorbbecsdk.

The Gemini 336L exposes color, depth, left/right IR and IMU through the Orbbec
SDK.  This module deliberately enables the **color stream only**; it never
requests depth, IR, point clouds, or D2C alignment.

Unlike :mod:`hardware.zed`, this driver goes through the vendor SDK rather than
raw V4L2: an Orbbec camera registers several ``/dev/video*`` nodes (color, depth,
IR), so selecting the right one by path is fragile, and the SDK addresses the
camera by serial number instead.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Color formats this driver knows how to convert to RGB. An empty
# ``color_format`` lets the SDK pick the device default (MJPG on the Gemini 336L).
SUPPORTED_COLOR_FORMATS = ("MJPG", "YUYV", "RGB", "BGR")

# Per-frame wait budget. A timeout returns None and is retried, which is normal
# at stream start; it is not treated as a failure.
FRAME_TIMEOUT_MS = 1000


class OrbbecCamera:
    """Read RGB frames from an Orbbec camera's color stream.

    ``read()`` returns an HWC, ``uint8`` RGB array, ready for
    :class:`src.data_collector.LeRobotDataCollector`.

    Frames are returned at the native capture resolution. Each camera records
    its own stream as-is; cropping and downscaling to the training resolution
    belong to dataset export, not to this driver.

    Args:
        serial_number: Select a specific camera; empty picks the first one found.
        width: Requested color width.
        height: Requested color height.
        fps: Requested color frame rate.
        color_format: Requested pixel format, one of
            :data:`SUPPORTED_COLOR_FORMATS`; empty keeps the device default.
        startup_timeout: Seconds :meth:`start` waits for the first color frame.
    """

    def __init__(
        self,
        serial_number: str = "",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        color_format: str = "",
        startup_timeout: float = 3.0,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        if color_format and color_format not in SUPPORTED_COLOR_FORMATS:
            raise ValueError(
                f"color_format must be empty or one of {SUPPORTED_COLOR_FORMATS}, got {color_format!r}"
            )
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")

        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.color_format = color_format
        self.startup_timeout = startup_timeout
        self._cv2 = None
        self._np = None
        self._context = None
        self._pipeline = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_ready = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._worker_error: Exception | None = None

    def start(self) -> None:
        """Open the camera and start the latest-frame capture thread.

        Startup waits for the first valid RGB frame. After that, :meth:`read`
        only reads an in-memory reference and never waits for the SDK.
        """
        if self._pipeline is not None:
            if self._thread is not None and self._thread.is_alive():
                return
            self.stop()

        try:
            import cv2
            import numpy as np
            from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline
        except ImportError as error:
            raise RuntimeError(
                "pyorbbecsdk and OpenCV are required; install the project's dependencies."
            ) from error

        # The Context must stay referenced for as long as the camera is used: if
        # it is garbage collected, the DeviceList's internal manager is freed and
        # the SDK raises `NULL pointer passed for argument "deviceMgr"`.
        context = Context()
        device_list = context.query_devices()
        if device_list.get_count() == 0:
            raise RuntimeError(
                "No Orbbec device found. Check the USB connection, and make sure no "
                "other process (such as Orbbec Viewer) holds the camera open — the "
                "device can only be opened by one process at a time."
            )
        device = (
            device_list.get_device_by_serial_number(self.serial_number)
            if self.serial_number
            else device_list.get_device_by_index(0)
        )

        pipeline = Pipeline(device)
        requested_format = (
            getattr(OBFormat, self.color_format)
            if self.color_format
            else OBFormat.UNKNOWN_FORMAT
        )
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            # A zero/UNKNOWN argument means "any"; passing all four pins the profile.
            profile = profiles.get_video_stream_profile(
                self.width, self.height, requested_format, self.fps
            )
        except Exception as error:
            raise RuntimeError(
                f"Camera does not offer a {self.width}x{self.height} @ {self.fps}fps color "
                f"profile with format {self.color_format or 'any'}. Check the supported "
                "modes in Orbbec Viewer."
            ) from error

        config = Config()
        config.enable_stream(profile)

        self._stop_event.clear()
        self._frame_ready.clear()
        with self._frame_lock:
            self._latest_frame = None
            self._worker_error = None
        pipeline.start(config)
        self._cv2 = cv2
        self._np = np
        self._context = context
        self._pipeline = pipeline
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(pipeline, cv2, np),
            name="orbbec-color-capture",
            daemon=True,
        )
        self._thread.start()

        if not self._frame_ready.wait(timeout=self.startup_timeout):
            self.stop()
            raise RuntimeError(
                f"Timed out after {self.startup_timeout:.1f}s waiting for a color frame"
            )
        with self._frame_lock:
            worker_error = self._worker_error
            has_frame = self._latest_frame is not None
        if worker_error is not None or not has_frame:
            self.stop()
            detail = str(worker_error) if worker_error is not None else "no frame received"
            raise RuntimeError(f"Could not start the Orbbec camera: {detail}")

    def read(self) -> np.ndarray:
        """Immediately return the newest uint8 RGB HWC frame in memory."""
        if self._pipeline is None or self._thread is None:
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

    def _capture_loop(self, pipeline: object, cv2: object, np: object) -> None:
        """Continuously drain the pipeline and publish only the most recent frame."""
        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
                if frames is None:
                    continue  # Wait timed out; retry rather than fail.
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                frame_rgb = self._prepare_frame(color_frame, cv2, np)
                with self._frame_lock:
                    self._latest_frame = frame_rgb
                self._frame_ready.set()
        except Exception as error:
            with self._frame_lock:
                self._worker_error = error
            self._frame_ready.set()

    def _prepare_frame(self, color_frame: object, cv2: object, np: object) -> np.ndarray:
        """Decode one color frame to an RGB array at its native resolution."""
        width = color_frame.get_width()
        height = color_frame.get_height()
        format_name = color_frame.get_format().name
        data = np.asanyarray(color_frame.get_data())

        if format_name == "MJPG":
            frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError("Failed to decode an MJPG color frame")
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if format_name == "YUYV":
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUYV)
        if format_name == "RGB":
            return data.reshape(height, width, 3)
        if format_name == "BGR":
            return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
        raise RuntimeError(
            f"Unsupported color format {format_name}; set color_format to one of "
            f"{SUPPORTED_COLOR_FORMATS}"
        )

    def stop(self) -> None:
        """Stop background capture and release the camera. Safe to call repeatedly."""
        self._stop_event.set()
        thread = self._thread
        pipeline = self._pipeline

        # Join before stopping the pipeline: the worker must not call
        # wait_for_frames() on a stopped pipeline. The worker wakes up within
        # FRAME_TIMEOUT_MS, so this join normally returns immediately.
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=FRAME_TIMEOUT_MS / 1000 + 1.0)
        if pipeline is not None:
            pipeline.stop()

        self._thread = None
        self._pipeline = None
        self._context = None
        self._cv2 = None
        self._np = None
        with self._frame_lock:
            self._latest_frame = None
            self._worker_error = None
        self._frame_ready.clear()

    close = stop

    def __enter__(self) -> "OrbbecCamera":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()
