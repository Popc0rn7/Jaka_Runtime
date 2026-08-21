"""Replay one episode of a local LeRobot v2.1 or v3 dataset on JAKA S5.

Usage:
    uv run python script/replay.py \
        replay.dataset_root=data/demo/20260820_153529 \
        replay.episode=0 replay.confirm=true

Replay is per-episode on purpose. A v3 dataset packs every episode into the same
parquet file, so streaming the whole file would drive the arm straight from the
last pose of one episode to the first pose of the next within a couple of control
periods.
"""

import json
import os
import sys
import time
from pathlib import Path

import hydra
import pyarrow.parquet as pq
from omegaconf import DictConfig
from pyDHgripper import AG95

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from hardware.jaka_s5 import JOINT_COUNT, JakaS5

# Keep the operator in control of the transition into every episode.  These
# values intentionally mirror teleop.py's short pause before it starts moving.
RAMP_SECONDS = 2.0
START_DELAY_SECONDS = 2.0
ACTION_NAMES = [*(f"joint_{i}.pos" for i in range(JOINT_COUNT)), "gripper.target_pos"]


def read_episode_rows(dataset_root: Path) -> list[dict]:
    """Read episode metadata from either supported LeRobot on-disk layout."""
    v21_path = dataset_root / "meta" / "episodes.jsonl"
    if v21_path.is_file():
        return [json.loads(line) for line in v21_path.read_text(encoding="utf-8").splitlines() if line]

    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No LeRobot episode metadata found under {dataset_root / 'meta'}")

    rows: list[dict] = []
    for path in episode_files:
        table = pq.read_table(path)
        keep = [name for name in table.schema.names if not name.startswith("stats/")]
        rows.extend(table.select(keep).to_pylist())
    return rows


def load_episode(
    dataset_root: Path, episode_index: int
) -> tuple[int, str, list[list[float]]]:
    """Return (fps, task, actions) for one episode, without a Hugging Face cache."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot v3 dataset: {dataset_root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    # Names, not just shape: a 7-wide action in a different order would be
    # silently replayed onto the wrong joints.
    action_feature = info.get("features", {}).get("action", {})
    if action_feature.get("names") != ACTION_NAMES:
        raise ValueError(
            f"Replay expects action names {ACTION_NAMES}, got {action_feature.get('names')}"
        )

    episodes = read_episode_rows(dataset_root)
    available = sorted(row["episode_index"] for row in episodes)
    matches = [row for row in episodes if row["episode_index"] == episode_index]
    if not matches:
        raise ValueError(f"Episode {episode_index} not found; dataset has {available}")
    episode = matches[0]

    if info.get("codebase_version") == "v2.1":
        data_file = dataset_root / "data" / f"chunk-{episode_index // info['chunks_size']:03d}" / f"episode_{episode_index:06d}.parquet"
    else:
        data_file = dataset_root / info["data_path"].format(
            chunk_index=episode["data/chunk_index"],
            file_index=episode["data/file_index"],
        )
    if not data_file.is_file():
        raise FileNotFoundError(f"Missing data file: {data_file}")

    rows = pq.read_table(
        data_file, columns=["episode_index", "frame_index", "action"]
    ).to_pylist()
    actions = [
        row["action"]
        for row in sorted(
            (row for row in rows if row["episode_index"] == episode_index),
            key=lambda row: row["frame_index"],
        )
    ]
    if len(actions) != episode["length"]:
        raise ValueError(
            f"Episode {episode_index}: metadata claims {episode['length']} frames but "
            f"{data_file.name} holds {len(actions)}"
        )

    tasks = episode.get("tasks") or []
    return int(info["fps"]), tasks[0] if tasks else "", actions


def to_gripper_position(action: list[float], frame_index: int) -> int:
    """Convert the normalised gripper target to the AG95's 0-1000 range."""
    target = round(float(action[JOINT_COUNT]) * 1000)
    if not 0 <= target <= 1000:
        raise ValueError(
            f"Frame {frame_index} has an out-of-range gripper target: {action[JOINT_COUNT]}"
        )
    return target


def wait_for_operator(message: str) -> None:
    """Create an explicit, repeatable safety boundary before robot motion."""
    input(f"{message}\nPress Enter when the workspace is clear... ")


def ramp_to_episode_start(
    arm: JakaS5, gripper: AG95, first_action: list[float], fps: int
) -> int:
    """Slowly align the robot with an episode's initial command."""
    wait_for_operator("Ready to ramp to this episode's first pose.")
    print(
        f"Ramping starts in {START_DELAY_SECONDS:.0f} seconds. "
        "Keep the emergency stop accessible.",
        flush=True,
    )
    time.sleep(START_DELAY_SECONDS)

    gripper_target = to_gripper_position(first_action, 0)
    arm.JointCtrl(first_action[:JOINT_COUNT], step_num=round(RAMP_SECONDS * fps))
    gripper.set_pos(gripper_target)
    time.sleep(RAMP_SECONDS)
    print("Ramp complete.")
    return gripper_target


def replay_episode(
    arm: JakaS5,
    gripper: AG95,
    actions: list[list[float]],
    fps: int,
    episode_index: int,
) -> None:
    """Replay one already-ramped episode at its recorded control frequency."""
    gripper_target = ramp_to_episode_start(arm, gripper, actions[0], fps)
    wait_for_operator(
        f"Episode {episode_index} is in its start pose and ready to replay."
    )
    print(f"Replay starts in {START_DELAY_SECONDS:.0f} seconds...", flush=True)
    time.sleep(START_DELAY_SECONDS)

    dt = 1.0 / fps
    next_tick = time.perf_counter()
    for frame_index, action in enumerate(actions):
        arm.JointCtrl(action[:JOINT_COUNT], step_num=2)
        target = to_gripper_position(action, frame_index)
        if target != gripper_target:
            gripper.set_pos(target)
            gripper_target = target

        next_tick += dt
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()

    print(f"Replay of episode {episode_index} done.")


@hydra.main(version_base=None, config_path="../config", config_name="ultrahands")
def main(cfg: DictConfig):

    dataset_root = Path(cfg.replay.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = Path(root_dir) / dataset_root
    episode_index = int(cfg.replay.episode)

    fps, task, actions = load_episode(dataset_root, episode_index)
    print(f"Dataset : {dataset_root}")
    print(f"Episode : {episode_index} — {len(actions)} frames at {fps} Hz")
    print(f"Task    : {task}")
    if not actions:
        raise ValueError(f"Episode {episode_index} has no actions to replay")

    arm = JakaS5(ip="192.168.2.121", freq_hz=fps)
    gripper = AG95(port=cfg.gripper.port)
    try:
        arm.start()
        gripper.set_force(cfg.gripper.force)
        gripper.set_vel(cfg.gripper.velocity)

        replay_episode(arm, gripper, actions, fps, episode_index)
    finally:
        arm.stop()


if __name__ == "__main__":
    main()
