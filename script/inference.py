"""Run an OpenPI policy on the JAKA S5.

The only command-line flag is ``--mock``.  It performs normal camera capture
and policy requests, but prints predicted actions instead of commanding the
robot or gripper.  All connection, observation, and control settings live in
``config/ultrahands.yaml``.
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import queue
import sys
import threading
import time
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from hardware.orbbec import OrbbecCamera
from hardware.zed import ZedCamera
from src.policy_client import OpenPIPolicyClient

# Keep mock mode independent of JAKA's native SDK, which is only imported when
# actual robot commands are requested.
JOINT_COUNT = 6
ACTION_DIM = JOINT_COUNT + 1
SAFETY_EPSILON = 1e-6
MOCK = False


class PolicyWorker:
    """Own the policy WebSocket in a background thread, one request at a time."""

    def __init__(self, policy: OpenPIPolicyClient) -> None:
        self._policy = policy
        self._requests: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._results: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._error: BaseException | None = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="policy")

    def start(self) -> None:
        self._thread.start()

    def request(self, observation: dict[str, Any]) -> bool:
        """Queue one inference request, without ever blocking control."""
        try:
            self._requests.put_nowait(observation)
        except queue.Full:
            return False
        return True

    def poll(self) -> np.ndarray | None:
        if self._error is not None:
            raise RuntimeError("Policy inference thread failed") from self._error
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stopped.set()

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                observation = self._requests.get(timeout=0.1)
                actions = self._policy.infer(observation, action_dim=ACTION_DIM)
                while not self._stopped.is_set():
                    try:
                        self._results.put(actions, timeout=0.1)
                        break
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except BaseException as error:
                self._error = error
                return


def parse_args() -> bool:
    """Parse the deliberately small public CLI before Hydra sees argv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Print policy actions without connecting to or moving the robot.",
    )
    args = parser.parse_args()
    return args.mock


def make_observation(
    client: OpenPIPolicyClient,
    cfg: DictConfig,
    joint_state: np.ndarray,
    gripper_state: float,
    agent_view: np.ndarray,
    wrist: np.ndarray,
) -> dict[str, Any]:
    """Create one checkpoint-compatible observation from a control tick."""
    state = np.asarray([*joint_state[:JOINT_COUNT], gripper_state], dtype=np.float32)
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid robot state for policy inference: {state}")
    return {
        str(cfg.policy.state_key): state,
        str(cfg.policy.agent_view_key): client.prepare_image(
            agent_view, int(cfg.policy.image_size)
        ),
        str(cfg.policy.wrist_key): client.prepare_image(
            wrist, int(cfg.policy.image_size)
        ),
        str(cfg.policy.prompt_key): str(cfg.policy.task),
    }


def set_gripper(gripper: Any, target: float) -> None:
    """Send a validated normalized gripper target to the AG95."""
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"Policy gripper target must be in [0, 1], got {target}")
    gripper.set_pos(round(target * 1000))


def validate_actions(
    actions: np.ndarray, current_joints: np.ndarray, cfg: DictConfig, *, fps: int
) -> None:
    """Reject out-of-range targets or abrupt joint changes before execution."""
    joint_min = np.asarray(cfg.safety.joint_min, dtype=np.float32)
    joint_max = np.asarray(cfg.safety.joint_max, dtype=np.float32)
    if joint_min.shape != (JOINT_COUNT,) or joint_max.shape != (JOINT_COUNT,):
        raise ValueError("safety.joint_min and safety.joint_max must each have 6 values")
    if np.any(joint_min > joint_max):
        raise ValueError("Each safety.joint_min value must not exceed joint_max")

    joints = actions[:, :JOINT_COUNT]
    out_of_range = (joints < joint_min - SAFETY_EPSILON) | (
        joints > joint_max + SAFETY_EPSILON
    )
    if np.any(out_of_range):
        action_index, joint_index = np.argwhere(out_of_range)[0]
        raise ValueError(
            f"Policy action {action_index} joint_{joint_index}={joints[action_index, joint_index]:.4f} "
            f"is outside [{joint_min[joint_index]:.4f}, {joint_max[joint_index]:.4f}]"
        )

    gripper_targets = actions[:, JOINT_COUNT]
    gripper_out_of_range = (gripper_targets < -SAFETY_EPSILON) | (
        gripper_targets > 1.0 + SAFETY_EPSILON
    )
    if np.any(gripper_out_of_range):
        action_index = int(np.flatnonzero(
            gripper_out_of_range
        )[0])
        raise ValueError(
            f"Policy action {action_index} gripper target "
            f"{gripper_targets[action_index]:.4f} is outside [0, 1]"
        )

    max_step = float(cfg.safety.max_joint_step)
    if max_step <= 0:
        raise ValueError("safety.max_joint_step must be positive")
    previous = np.vstack((np.asarray(current_joints[:JOINT_COUNT]), joints[:-1]))
    deltas = np.abs(joints - previous)
    step_too_large = deltas > max_step + SAFETY_EPSILON
    if np.any(step_too_large):
        action_index, joint_index = np.argwhere(step_too_large)[0]
        raise ValueError(
            f"Policy action {action_index} changes joint_{joint_index} by "
            f"{deltas[action_index, joint_index]:.4f} rad; maximum is {max_step:.4f} rad"
        )

    max_speed = float(cfg.safety.max_joint_speed)
    if max_speed <= 0:
        raise ValueError("safety.max_joint_speed must be positive")
    speeds = deltas * fps
    speed_too_high = speeds > max_speed + SAFETY_EPSILON
    if np.any(speed_too_high):
        action_index, joint_index = np.argwhere(speed_too_high)[0]
        raise ValueError(
            f"Policy action {action_index} commands joint_{joint_index} at "
            f"{speeds[action_index, joint_index]:.4f} rad/s; maximum is "
            f"{max_speed:.4f} rad/s"
        )


def execute_action(
    action: np.ndarray,
    arm: JakaS5 | None,
    gripper: Any | None,
    *,
    mock: bool,
    ramp_steps: int,
) -> None:
    """Execute exactly one buffered action; only the control thread calls this."""
    joints = action[:JOINT_COUNT].tolist()
    gripper_target = float(action[JOINT_COUNT])
    if mock:
        print(f"action: joints={joints}, gripper={gripper_target:.4f}")
        return
    assert arm is not None and gripper is not None
    arm.JointCtrl(joints, step_num=ramp_steps)
    set_gripper(gripper, gripper_target)


@hydra.main(version_base=None, config_path="../config", config_name="ultrahands")
def main(cfg: DictConfig) -> None:
    fps = int(cfg.robot.freq_hz)
    if fps <= 0:
        raise ValueError(f"robot.freq_hz must be positive, got {fps}")
    chunk_size = int(cfg.policy.actions_per_inference)
    refill_at = int(cfg.policy.refill_at_actions)
    if chunk_size <= 0 or not 0 <= refill_at < chunk_size:
        raise ValueError("policy requires 0 <= refill_at_actions < actions_per_inference")

    agent_camera = wrist_camera = arm = gripper = None
    policy = OpenPIPolicyClient(
        host=str(cfg.policy.host),
        port=int(cfg.policy.port),
        action_key=str(cfg.policy.action_key),
    )
    worker = PolicyWorker(policy)
    try:
        wrist_camera = OrbbecCamera(**cfg.orbbec)
        wrist_camera.start()
        agent_camera = ZedCamera(**cfg.zed)
        agent_camera.start()

        if not MOCK:
            from pyDHgripper import AG95
            from hardware.jaka_s5 import JakaS5

            arm = JakaS5(ip=str(cfg.robot.ip), freq_hz=fps)
            arm.start()
            gripper = AG95(port=str(cfg.gripper.port))
            gripper.set_force(int(cfg.gripper.force))
            gripper.set_vel(int(cfg.gripper.velocity))

        policy.reset()
        worker.start()
        print("Inference started. Press Ctrl-C to stop.", flush=True)
        gripper_state = 0.0
        action_buffer: deque[np.ndarray] = deque()
        last_command: np.ndarray | None = None
        pending_request = False
        next_tick = time.perf_counter()
        while True:
            next_tick += 1.0 / fps
            agent_view = agent_camera.read()
            wrist = wrist_camera.read()
            if MOCK:
                joint_state = np.zeros(JOINT_COUNT, dtype=np.float32)
            else:
                assert arm is not None
                joint_state = np.asarray(arm.get_joint_position(), dtype=np.float32)
                if joint_state.shape[0] < JOINT_COUNT:
                    raise RuntimeError("No complete JAKA joint feedback received")

            result = worker.poll()
            if result is not None:
                pending_request = False
                chunk = result[:chunk_size]
                # The chunk follows any already-buffered commands, not merely
                # the current feedback pose, so every queue transition is safe.
                reference = (
                    action_buffer[-1][:JOINT_COUNT]
                    if action_buffer
                    else (last_command if last_command is not None else joint_state)
                )
                validate_actions(chunk, reference, cfg, fps=fps)
                action_buffer.extend(chunk)

            if not pending_request and len(action_buffer) <= refill_at:
                observation = make_observation(
                    policy, cfg, joint_state, gripper_state, agent_view, wrist
                )
                pending_request = worker.request(observation)

            if action_buffer:
                action = action_buffer.popleft()
                execute_action(
                    action,
                    arm,
                    gripper,
                    mock=MOCK,
                    ramp_steps=2,
                )
                last_command = action[:JOINT_COUNT].copy()
                gripper_state = float(action[JOINT_COUNT])
            elif not MOCK and arm is not None and last_command is not None:
                # Policy latency exhausted the buffer: hold the last safe pose
                # instead of issuing an unvalidated or stale command.
                arm.JointCtrl(last_command.tolist(), step_num=2)

            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("Inference interrupted.")
    finally:
        worker.stop()
        for resource in (arm, wrist_camera, agent_camera):
            if resource is not None:
                resource.stop()


if __name__ == "__main__":
    MOCK = parse_args()
    # Hydra treats argv as configuration overrides.  This script intentionally
    # exposes no overrides: edit config/ultrahands.yaml for all non-mock setup.
    sys.argv = sys.argv[:1]
    main()
