"""Replay one local LeRobot v3 episode on the JAKA S5 hardware.

Usage:
    uv run python script/replay.py \
        replay.dataset_root=data/demo/20260819_145248 replay.confirm=true
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


def load_actions(dataset_root: Path) -> tuple[int, list[list[float]]]:
    """Read the replayable action stream without requiring a Hugging Face cache."""
    info_path = dataset_root / "meta" / "info.json"
    data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    if not info_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Not a complete LeRobot v3 dataset: {dataset_root}")

    info = json.loads(info_path.read_text())
    action_feature = info.get("features", {}).get("action", {})
    if action_feature.get("shape") != [JOINT_COUNT + 1]:
        raise ValueError(
            "Replay expects action=[6 joint targets in rad, gripper target in [0, 1]], "
            f"got {action_feature}"
        )

    actions = pq.read_table(data_path, columns=["action"])["action"].to_pylist()
    if not actions:
        raise ValueError(f"Dataset has no action frames: {dataset_root}")
    return int(info["fps"]), actions


@hydra.main(version_base=None, config_path="../config", config_name="ultrahands")
def main(cfg: DictConfig):
    if not cfg.replay.confirm:
        raise RuntimeError(
            "Replay moves the real robot. Re-run with replay.confirm=true after "
            "checking the workspace and emergency stop."
        )

    dataset_root = Path(cfg.replay.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = Path(root_dir) / dataset_root
    fps, actions = load_actions(dataset_root)
    print(f"Loaded {len(actions)} actions at {fps} Hz from {dataset_root}")
    print("Replay starts in 3 seconds. Keep the emergency stop accessible.")
    time.sleep(3)

    arm = JakaS5(ip="192.168.2.121", freq_hz=fps)
    gripper = AG95(port=cfg.gripper.port)
    try:
        arm.start()
        gripper.set_force(cfg.gripper.force)
        gripper.set_vel(cfg.gripper.velocity)

        first_action = actions[0]
        first_gripper_target = round(float(first_action[JOINT_COUNT]) * 1000)
        if not 0 <= first_gripper_target <= 1000:
            raise ValueError(f"First frame has invalid gripper target: {first_action[JOINT_COUNT]}")
        print("Moving to the first replay action over 2 seconds...")
        arm.JointCtrl(first_action[:JOINT_COUNT], step_num=250)
        gripper.set_pos(first_gripper_target)
        time.sleep(2)

        previous_gripper_target = first_gripper_target
        dt = 1.0 / fps
        next_tick = time.perf_counter()
        for frame_index, action in enumerate(actions):
            joint_target = action[:JOINT_COUNT]
            gripper_target = round(float(action[JOINT_COUNT]) * 1000)
            if not 0 <= gripper_target <= 1000:
                raise ValueError(
                    f"Frame {frame_index} has invalid gripper target: {action[JOINT_COUNT]}"
                )

            arm.JointCtrl(joint_target, step_num=2)
            if gripper_target != previous_gripper_target:
                gripper.set_pos(gripper_target)
                previous_gripper_target = gripper_target

            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

        print("Replay done.")
    finally:
        arm.stop()


if __name__ == "__main__":
    main()
