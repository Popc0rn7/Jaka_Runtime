"""Official LeRobot v2.1 dataset writer for the JAKA S5 teleoperation loop.

The collector deliberately has no dependency on robot, gripper, or camera
drivers. It delegates all dataset validation, statistics, parquet metadata,
and video encoding to ``lerobot==0.3.3``, the release that creates the
official LeRobot v2.1 format while remaining compatible with NumPy < 2.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from types import MethodType

import numpy as np


def _encode_episode_videos_h264(
    dataset: Any,
    episode_index: int,
    *,
    encode_video_frames: Any,
    write_info: Any,
) -> None:
    """Encode one episode's camera frames as H.264 MP4 files."""
    for key in dataset.meta.video_keys:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        if video_path.is_file():
            continue
        image_dir = dataset._get_image_file_path(
            episode_index=episode_index,
            image_key=key,
            frame_index=0,
        ).parent
        encode_video_frames(image_dir, video_path, dataset.fps, vcodec="h264", overwrite=True)
        shutil.rmtree(image_dir)

    if dataset.meta.video_keys and episode_index == 0:
        dataset.meta.update_video_info()
        write_info(dataset.meta.info, dataset.meta.root)


class LeRobotDataCollector:
    """Record complete teleoperation episodes in official LeRobot v2.1 format."""

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

        # v2.1 writes frames asynchronously when configured, then validates
        # and encodes official episode/video artifacts on save. Its API has no
        # streaming-video encoder; retain this parameter for caller compatibility.
        del encoder_queue_maxsize
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.datasets.utils import write_info
            from lerobot.datasets.video_utils import encode_video_frames
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot v2.1 dataset dependencies are unavailable. Run `uv sync`."
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
            image_writer_threads=4 if streaming_encoding and self.camera_shapes else 0,
            batch_encoding_size=1,
        )
        self.dataset.encode_episode_videos = MethodType(
            lambda dataset, episode_index: _encode_episode_videos_h264(
                dataset,
                episode_index,
                encode_video_frames=encode_video_frames,
                write_info=write_info,
            ),
            self.dataset,
        )

    def record_step(
        self,
        state: Sequence[float] | np.ndarray,
        action: Sequence[float] | np.ndarray,
        images: Mapping[str, np.ndarray] | None = None,
        *,
        task: str | None = None,
    ) -> None:
        """Append one synchronized teleoperation sample to the current episode."""
        self._ensure_open()
        sample_task = task if task is not None else self.task
        if not sample_task.strip():
            raise ValueError("task must not be empty")
        frame: dict[str, Any] = {
            "observation.state": self._as_vector(state, self.state_names, "state"),
            "action": self._as_vector(action, self.action_names, "action"),
        }

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

        self.dataset.add_frame(frame, task=sample_task)

    def save_episode(self) -> None:
        """Validate, encode, and commit the current official v2.1 episode."""
        self._ensure_open()
        if not self.has_pending_frames():
            raise RuntimeError("Cannot save an episode with no recorded frames")
        self.dataset.save_episode()

    def discard_episode(self) -> None:
        """Discard frames and temporary images from the current episode."""
        self._ensure_open()
        self.dataset._wait_image_writer()
        self.dataset.clear_episode_buffer()

    def finalize(self) -> None:
        """Flush LeRobot's asynchronous image writer after the final episode."""
        if self._closed:
            return
        if self.has_pending_frames():
            raise RuntimeError("Current episode has unsaved frames; call save_episode() or discard_episode()")
        self.dataset.stop_image_writer()
        self._closed = True

    def has_pending_frames(self) -> bool:
        return bool(self.dataset.episode_buffer["size"])

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
