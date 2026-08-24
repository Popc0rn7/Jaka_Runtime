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
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Color formats this driver knows how to convert to RGB. An empty
# ``color_format`` lets the SDK pick the device default (MJPG on the Gemini 336L).
SUPPORTED_COLOR_FORMATS = ("MJPG", "YUYV", "RGB", "BGR")

# Per-frame wait budget. A timeout returns None and is retried, which is normal
# at stream start; it is not treated as a failure.
FRAME_TIMEOUT_MS = 1000

# The stream counts as stable once this many consecutive frame gaps stay within
# MAX_STABLE_GAP_PERIODS nominal frame periods. A cold start delivers one frame
# and then stalls for roughly two seconds, so waiting for a single frame would
# hand the caller a stream that cannot yet sustain its nominal rate.
STABLE_FRAMES = 3
MAX_STABLE_GAP_PERIODS = 3


class OrbbecCamera:
    """Read RGB frames from an Orbbec camera's color stream.

    ``read()`` returns an HWC, ``uint8`` RGB array, ready for
    :class:`src.data_collector.LeRobotDataCollector`.

    When ``output_width`` and ``output_height`` are set, frames are center-
    cropped to the requested aspect ratio and resized before being returned.
    Otherwise, frames are returned at the native capture resolution.

    Args:
        device: Select a specific camera; empty picks the first one found.
        width: Requested color width.
        height: Requested color height.
        fps: Requested color frame rate.
        color_format: Requested pixel format, one of
            :data:`SUPPORTED_COLOR_FORMATS`; empty keeps the device default.
        output_width: Optional RGB output width. Must be paired with
            ``output_height``.
        output_height: Optional RGB output height. Must be paired with
            ``output_width``.
        rotate_180: Rotate the color image by 180 degrees before it is
            returned. Use this when the camera is mounted upside down.
        startup_timeout: Seconds :meth:`start` waits for the first color frame.
    """

    def __init__(
        self,
        device: str = "",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        color_format: str = "",
        output_width: int | None = None,
        output_height: int | None = None,
        rotate_180: bool = False,
        startup_timeout: float = 3.0,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        if color_format and color_format not in SUPPORTED_COLOR_FORMATS:
            raise ValueError(
                f"color_format must be empty or one of {SUPPORTED_COLOR_FORMATS}, got {color_format!r}"
            )
        if bool(output_width) != bool(output_height):
            raise ValueError("output_width and output_height must be provided together")
        if output_width is not None and (
            output_width <= 0 or output_height is None or output_height <= 0
        ):
            raise ValueError("output_width and output_height must be positive")
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")

        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.color_format = color_format
        self.output_width = output_width
        self.output_height = output_height
        self.rotate_180 = rotate_180
        self.startup_timeout = startup_timeout
        self._cv2 = None
        self._np = None
        self._context = None
        self._pipeline = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stream_ready = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._worker_error: Exception | None = None

    def start(self) -> None:
        """Open the camera and start the latest-frame capture thread.

        Startup waits until the stream actually sustains ``fps`` — not merely
        until the first frame arrives — so that :meth:`read` never hands the
        caller duplicate frames from a cold start. After that, :meth:`read` only
        reads an in-memory reference and never waits for the SDK.
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
        selected_device = (
            device_list.get_device_by_serial_number(self.device)
            if self.device
            else device_list.get_device_by_index(0)
        )

        pipeline = Pipeline(selected_device)
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
        self._stream_ready.clear()
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

        if not self._stream_ready.wait(timeout=self.startup_timeout):
            self.stop()
            raise RuntimeError(
                f"Timed out after {self.startup_timeout:.1f}s waiting for a stable "
                f"{self.fps}fps color stream"
            )
        with self._frame_lock:
            worker_error = self._worker_error
            has_frame = self._latest_frame is not None
        if worker_error is not None or not has_frame:
            self.stop()
            detail = (
                str(worker_error) if worker_error is not None else "no frame received"
            )
            raise RuntimeError(f"Could not start the Orbbec camera: {detail}")

    def read(self) -> np.ndarray:
        """Immediately return the newest uint8 RGB HWC frame in memory."""
        if self._pipeline is None or self._thread is None:
            raise RuntimeError("Camera is not started. Call start() before read().")

        with self._frame_lock:
            frame_rgb = self._latest_frame
            worker_error = self._worker_error
        if worker_error is not None:
            raise RuntimeError(
                f"Camera capture stopped: {worker_error}"
            ) from worker_error
        if frame_rgb is None:
            raise RuntimeError("No camera frame is available")
        # The worker replaces this array reference for every frame and never
        # mutates an array after publishing it, so returning it requires no copy.
        return frame_rgb

    def _capture_loop(self, pipeline: object, cv2: object, np: object) -> None:
        """Continuously drain the pipeline and publish only the most recent frame."""
        max_gap = MAX_STABLE_GAP_PERIODS / self.fps
        # Both counters belong to this thread alone, so they need no lock.
        last_arrival: float | None = None
        stable_gaps = 0
        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
                if frames is None:
                    continue  # Wait timed out; retry rather than fail.
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                frame_rgb = self._prepare_frame(color_frame, cv2, np)
                arrival = time.perf_counter()
                stable_gaps = (
                    stable_gaps + 1
                    if last_arrival is not None and arrival - last_arrival <= max_gap
                    else 0
                )
                last_arrival = arrival
                with self._frame_lock:
                    self._latest_frame = frame_rgb
                if stable_gaps >= STABLE_FRAMES:
                    self._stream_ready.set()
        except Exception as error:
            with self._frame_lock:
                self._worker_error = error
            # Unblock start() instead of making it wait out the full timeout.
            self._stream_ready.set()

    def _prepare_frame(
        self, color_frame: object, cv2: object, np: object
    ) -> np.ndarray:
        """Decode, optionally resize, and return one RGB color frame."""
        width = color_frame.get_width()
        height = color_frame.get_height()
        format_name = color_frame.get_format().name
        data = np.asanyarray(color_frame.get_data())

        if format_name == "MJPG":
            frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError("Failed to decode an MJPG color frame")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        elif format_name == "YUYV":
            frame_rgb = cv2.cvtColor(
                data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUYV
            )
        elif format_name == "RGB":
            frame_rgb = data.reshape(height, width, 3)
        elif format_name == "BGR":
            frame_rgb = cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(
                f"Unsupported color format {format_name}; set color_format to one of "
                f"{SUPPORTED_COLOR_FORMATS}"
            )

        # Camera orientation belongs at the driver boundary so that previews
        # and recorded observations use the same canonical wrist-camera view.
        if self.rotate_180:
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)

        if self.output_width is None or self.output_height is None:
            return frame_rgb

        frame_height, frame_width = frame_rgb.shape[:2]
        target_ratio = self.output_width / self.output_height
        source_ratio = frame_width / frame_height
        if source_ratio > target_ratio:
            crop_width = round(frame_height * target_ratio)
            left = (frame_width - crop_width) // 2
            frame_rgb = frame_rgb[:, left : left + crop_width]
        elif source_ratio < target_ratio:
            crop_height = round(frame_width / target_ratio)
            top = (frame_height - crop_height) // 2
            frame_rgb = frame_rgb[top : top + crop_height, :]
        return cv2.resize(
            frame_rgb,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_AREA,
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
        self._stream_ready.clear()

    close = stop

    def __enter__(self) -> "OrbbecCamera":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()
