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
        device: str | None = None,
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

        if device is not None and not device.strip():
            device = None

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
        if self.device is None:
            self.device = self.discover_device()
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
            detail = (
                str(worker_error) if worker_error is not None else "no frame received"
            )
            raise RuntimeError(f"Could not start camera {self.device}: {detail}")

    @classmethod
    def discover_device(
        cls,
        *,
        video_root: Path = Path("/dev"),
        sysfs_root: Path = Path("/sys/class/video4linux"),
    ) -> str:
        """Return the first detectable ZED V4L2 node.

        ZED cameras publish a V4L2 ``name`` containing ``ZED``.  The V4L2
        sysfs entry links to its USB interface, whose parent device exposes
        Stereolabs' USB vendor ID (``2b03``).  Matching nodes are sorted by
        their ``videoN`` number for deterministic selection.
        """
        candidates: list[Path] = []
        nodes = sorted(
            sysfs_root.glob("video*"),
            key=lambda path: (
                not path.name.removeprefix("video").isdigit(),
                (
                    int(path.name.removeprefix("video"))
                    if path.name.removeprefix("video").isdigit()
                    else path.name
                ),
            ),
        )
        for node in nodes:
            name_path = node / "name"
            device_path = node / "device"
            if not name_path.is_file() or not device_path.exists():
                continue
            name = name_path.read_text().strip()
            if "zed" not in name.lower():
                continue
            usb_device = device_path.resolve()
            if cls._read_usb_property(usb_device, "idVendor") != "2b03":
                continue
            video_device = video_root / node.name
            if not video_device.exists():
                continue
            candidates.append(video_device)

        if candidates:
            return str(candidates[0])

        available = cls._describe_zed_devices(video_root, sysfs_root)
        raise RuntimeError(
            "No ZED camera found through V4L2. "
            f"Available ZED V4L2 devices: {available}."
        )

    @staticmethod
    def _read_usb_property(device: Path, property_name: str) -> str | None:
        """Read a USB attribute from a V4L2 interface or one of its parents."""
        for parent in (device, *device.parents):
            value_path = parent / property_name
            if value_path.is_file():
                value = value_path.read_text().strip()
                if value:
                    return value
        return None

    @classmethod
    def _describe_zed_devices(cls, video_root: Path, sysfs_root: Path) -> str:
        """Format detectable ZED nodes for discovery error messages."""
        found: list[str] = []
        for node in sorted(sysfs_root.glob("video*")):
            name_path = node / "name"
            device_path = node / "device"
            if not name_path.is_file() or not device_path.exists():
                continue
            if "zed" not in name_path.read_text().lower():
                continue
            usb_device = device_path.resolve()
            if cls._read_usb_property(usb_device, "idVendor") != "2b03":
                continue
            device = video_root / node.name
            found.append(str(device))
        return ", ".join(found) if found else "none"

    def read(self) -> np.ndarray:
        """Immediately return the newest uint8 RGB HWC frame in memory."""
        if self._capture is None or self._thread is None:
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
            # Crop excess area to the requested aspect ratio before resizing;
            # direct resizing would distort the camera image.
            frame_height, frame_width = frame_bgr.shape[:2]
            target_ratio = self.output_width / self.output_height
            source_ratio = frame_width / frame_height
            if source_ratio > target_ratio:
                crop_width = round(frame_height * target_ratio)
                left = (frame_width - crop_width) // 2
                frame_bgr = frame_bgr[:, left : left + crop_width]
            elif source_ratio < target_ratio:
                crop_height = round(frame_width / target_ratio)
                top = (frame_height - crop_height) // 2
                frame_bgr = frame_bgr[top : top + crop_height, :]
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
