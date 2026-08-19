"""LeRobot v3 dataset writer for the JAKA S5 teleoperation loop.

The collector deliberately has no dependency on the robot, gripper, or camera
drivers.  Teleoperation owns those devices; it calls :meth:`record_step` once
per control tick with the measured state, commanded action, and optional RGB
camera frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class LeRobotDataCollector:
    """Record complete teleoperation episodes in the LeRobot v3 format.

    Args:
        repo_id: LeRobot dataset identifier (for example ``"local/jaka_s5"``).
            It is metadata only; no data is uploaded automatically.
        root: Empty directory in which the local v3 dataset is created.
        fps: Teleoperation control frequency.
        state_names: Names for the measured robot state entries.
        action_names: Names for the commanded action entries.
        camera_shapes: Optional mapping from camera name to ``(height, width,
            channels)``. Frames are expected as uint8 RGB HWC arrays. A camera
            called ``wrist`` becomes ``observation.images.wrist``.
        task: Default natural-language task stored on each recorded frame.

    Example:
        collector = LeRobotDataCollector(
            repo_id="local/jaka_s5_pick",
            root="outputs/lerobot/jaka_s5_pick",
            state_names=[*(f"joint_{i}.pos" for i in range(6)), "gripper.pos"],
            action_names=[*(f"joint_{i}.pos" for i in range(6)), "gripper.pos"],
            camera_shapes={"front": (480, 640, 3)},
            task="Pick up the block",
        )
        # Call once in every teleop step.
        collector.record_step(state, action, {"front": rgb_frame})
        collector.save_episode()
        collector.finalize()
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        *,
        fps: int = 30,
        state_names: Sequence[str],
        action_names: Sequence[str],
        camera_shapes: Mapping[str, Sequence[int]] | None = None,
        task: str = "teleoperation task",
        robot_type: str = "jaka_s5",
        streaming_encoding: bool = True,
        encoder_queue_maxsize: int = 60,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if not state_names or not action_names:
            raise ValueError("state_names and action_names must not be empty")
        if not task.strip():
            raise ValueError("task must not be empty")

        self.root = Path(root)
        self.fps = fps
        self.state_names = tuple(state_names)
        self.action_names = tuple(action_names)
        self.task = task
        self.camera_shapes = self._normalize_camera_shapes(camera_shapes or {})
        self._closed = False

        # Delayed import keeps teleop importable on machines used only for robot
        # control. `uv sync` installs the required packages declared in pyproject.
        try:
            from lerobot.configs.video import RGBEncoderConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot v3 dataset dependencies are unavailable. Run `uv sync`."
            ) from exc

        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(
                f"Dataset root is not empty: {self.root}. Use a new directory "
                "to avoid overwriting an existing dataset."
            )

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=self.root,
            fps=fps,
            robot_type=robot_type,
            features=self._make_features(),
            use_videos=bool(self.camera_shapes),
            # H.264 is broadly supported by the FFmpeg builds bundled with PyAV.
            rgb_encoder=RGBEncoderConfig(vcodec="h264"),
            streaming_encoding=streaming_encoding and bool(self.camera_shapes),
            encoder_queue_maxsize=encoder_queue_maxsize,
        )

    def record_step(
        self,
        state: Sequence[float] | np.ndarray,
        action: Sequence[float] | np.ndarray,
        images: Mapping[str, np.ndarray] | None = None,
        *,
        task: str | None = None,
    ) -> None:
        """Append one synchronized teleoperation sample to the current episode.

        ``state`` should contain measured values (e.g. JAKA joint feedback and
        gripper feedback); ``action`` should contain the command sent in this
        tick. Images must be RGB ``uint8`` arrays in HWC layout.
        """
        self._ensure_open()
        frame: dict[str, Any] = {
            "observation.state": self._as_vector(state, self.state_names, "state"),
            "action": self._as_vector(action, self.action_names, "action"),
            "task": task if task is not None else self.task,
        }
        if not frame["task"].strip():
            raise ValueError("task must not be empty")

        supplied_images = images or {}
        if set(supplied_images) != set(self.camera_shapes):
            raise ValueError(
                "images must contain exactly the configured cameras: "
                f"expected {sorted(self.camera_shapes)}, got {sorted(supplied_images)}"
            )
        for name, expected_shape in self.camera_shapes.items():
            image = np.asarray(supplied_images[name])
            if image.shape != expected_shape:
                raise ValueError(
                    f"Camera '{name}' shape must be {expected_shape}, got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise TypeError(f"Camera '{name}' must be uint8 RGB, got {image.dtype}")
            frame[f"observation.images.{name}"] = image

        self.dataset.add_frame(frame)

    def save_episode(self) -> None:
        """Commit the current episode. Empty episodes are rejected."""
        self._ensure_open()
        if not self.dataset.has_pending_frames():
            raise RuntimeError("Cannot save an episode with no recorded frames")
        self.dataset.save_episode()

    def discard_episode(self) -> None:
        """Discard all frames collected since the last :meth:`save_episode`."""
        self._ensure_open()
        self.dataset.clear_episode_buffer(delete_images=True)

    def finalize(self) -> None:
        """Close parquet/video writers. Call after the final saved episode."""
        if self._closed:
            return
        if self.dataset.has_pending_frames():
            raise RuntimeError("Current episode has unsaved frames; call save_episode() or discard_episode()")
        self.dataset.finalize()
        self._closed = True

    def __enter__(self) -> "LeRobotDataCollector":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is not None and not self._closed:
            self.discard_episode()
        self.finalize()

    def _make_features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_names),),
                "names": list(self.state_names),
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.action_names),),
                "names": list(self.action_names),
            },
        }
        for name, shape in self.camera_shapes.items():
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            }
        return features

    @staticmethod
    def _normalize_camera_shapes(
        camera_shapes: Mapping[str, Sequence[int]],
    ) -> dict[str, tuple[int, int, int]]:
        normalized: dict[str, tuple[int, int, int]] = {}
        for name, shape in camera_shapes.items():
            if not name or "." in name:
                raise ValueError(f"Camera name must be non-empty and contain no '.': {name!r}")
            shape_tuple = tuple(shape)
            if len(shape_tuple) != 3 or any(not isinstance(dim, int) or dim <= 0 for dim in shape_tuple):
                raise ValueError(f"Camera '{name}' shape must be three positive integers, got {shape!r}")
            if shape_tuple[2] != 3:
                raise ValueError(f"Camera '{name}' must have 3 RGB channels, got {shape_tuple}")
            normalized[name] = shape_tuple
        return normalized

    @staticmethod
    def _as_vector(
        values: Sequence[float] | np.ndarray,
        names: Sequence[str],
        label: str,
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.shape != (len(names),):
            raise ValueError(f"{label} must have {len(names)} values, got shape {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{label} contains NaN or infinity")
        return vector

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Collector is finalized and cannot record more frames")
