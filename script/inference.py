"""Run an OpenPI policy on the JAKA S5.

The only command-line flag is ``--mock``.  It performs normal camera capture
and policy requests, but prints predicted actions instead of commanding the
robot or gripper.  All connection, observation, and control settings live in
``config/config.yaml``.
"""

from __future__ import annotations

import argparse
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
) -> np.ndarray:
    """Clip policy targets to the configured joint and gripper safety envelope."""
    safe_actions = np.asarray(actions, dtype=np.float32).copy()
    if safe_actions.ndim != 2 or safe_actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"Policy actions must have shape (action_horizon, {ACTION_DIM}), "
            f"got {safe_actions.shape}"
        )
    if not np.isfinite(safe_actions).all():
        raise ValueError("Policy actions contain NaN or infinity")
    safe_actions[:, JOINT_COUNT] = np.clip(safe_actions[:, JOINT_COUNT], 0.0, 1.0)
    joint_min = np.asarray(cfg.safety.joint_min, dtype=np.float32)
    joint_max = np.asarray(cfg.safety.joint_max, dtype=np.float32)
    if joint_min.shape != (JOINT_COUNT,) or joint_max.shape != (JOINT_COUNT,):
        raise ValueError(
            "safety.joint_min and safety.joint_max must each have 6 values"
        )
    if np.any(joint_min > joint_max):
        raise ValueError("Each safety.joint_min value must not exceed joint_max")

    max_step = float(cfg.safety.max_joint_step)
    if max_step <= 0:
        raise ValueError("safety.max_joint_step must be positive")
    max_speed = float(cfg.safety.max_joint_speed)
    if max_speed <= 0:
        raise ValueError("safety.max_joint_speed must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    max_delta = min(max_step, max_speed / fps)
    previous = np.asarray(current_joints[:JOINT_COUNT], dtype=np.float32)
    if previous.shape != (JOINT_COUNT,) or not np.isfinite(previous).all():
        raise ValueError("Current joint feedback must contain 6 finite values")
    previous = np.clip(previous, joint_min, joint_max)
    for target in safe_actions[:, :JOINT_COUNT]:
        target[:] = np.clip(target, joint_min, joint_max)
        target[:] = np.clip(target, previous - max_delta, previous + max_delta)
        previous = target
    return safe_actions


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


def ramp_to_policy_target(
    action: np.ndarray,
    arm: JakaS5 | None,
    gripper: Any | None,
    *,
    mock: bool,
    ramp_steps: int,
) -> None:
    """Move safely from the current pose to the first policy target."""
    if ramp_steps <= 0:
        raise ValueError("policy.ramp_steps must be positive")
    execute_action(action, arm, gripper, mock=mock, ramp_steps=ramp_steps)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    fps = int(cfg.jaka_s5.freq_hz)
    ramp_steps = int(cfg.policy.ramp_steps)

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

            arm = JakaS5(ip=str(cfg.jaka_s5.ip), freq_hz=fps)
            arm.start()
            gripper = AG95(port=str(cfg.dh_gripper.port))
            gripper.set_force(int(cfg.dh_gripper.force))
            gripper.set_vel(int(cfg.dh_gripper.velocity))

        policy.reset()
        worker.start()
        gripper_state = 0.0
        last_command: np.ndarray | None = None
        pending_request = False

        # Do not enter the control loop from an arbitrary startup pose.  First
        # obtain a policy target from the current observation, then ramp to it.
        print("Waiting for the first policy target...", flush=True)
        while True:
            agent_view = agent_camera.read()
            wrist = wrist_camera.read()
            if MOCK:
                joint_state = np.zeros(JOINT_COUNT, dtype=np.float32)
            else:
                assert arm is not None
                joint_state = np.asarray(arm.get_joint_position(), dtype=np.float32)
                if joint_state.shape[0] < JOINT_COUNT:
                    raise RuntimeError("No complete JAKA joint feedback received")

            if not pending_request:
                observation = make_observation(
                    policy, cfg, joint_state, gripper_state, agent_view, wrist
                )
                pending_request = worker.request(observation)

            result = worker.poll()
            if result is not None:
                first_action = validate_actions(result[:1], joint_state, cfg, fps=fps)[
                    0
                ]
                break
            time.sleep(0.01)

        if not MOCK:
            input("First policy target is ready. Press Enter to ramp to it... ")
        ramp_to_policy_target(
            first_action,
            arm,
            gripper,
            mock=MOCK,
            ramp_steps=ramp_steps,
        )
        last_command = first_action[:JOINT_COUNT].copy()
        gripper_state = float(first_action[JOINT_COUNT])
        pending_request = False
        print("Inference started. Press Ctrl-C to stop.", flush=True)

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
            executed_action = False
            if result is not None:
                pending_request = False
                action = validate_actions(
                    result[:1],
                    last_command if last_command is not None else joint_state,
                    cfg,
                    fps=fps,
                )[0]
                execute_action(
                    action,
                    arm,
                    gripper,
                    mock=MOCK,
                    ramp_steps=2,
                )
                last_command = action[:JOINT_COUNT].copy()
                gripper_state = float(action[JOINT_COUNT])
                executed_action = True

            if not pending_request and not executed_action:
                observation = make_observation(
                    policy, cfg, joint_state, gripper_state, agent_view, wrist
                )
                pending_request = worker.request(observation)

            if (
                not executed_action
                and not MOCK
                and arm is not None
                and last_command is not None
            ):
                # Hold the last safe pose while the one outstanding inference
                # request is in flight.
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
    # exposes no overrides: edit config/config.yaml for all non-mock setup.
    sys.argv = sys.argv[:1]
    main()
