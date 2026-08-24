from types import SimpleNamespace
import unittest

import numpy as np

from script.inference import (
    INITIAL_POSITION_RAMP_STEPS,
    initialize_robot,
    validate_actions,
)


class ValidateActionsTest(unittest.TestCase):
    def test_clips_joint_targets_to_ranges_and_per_tick_motion_limit(self) -> None:
        cfg = SimpleNamespace(
            safety=SimpleNamespace(
                joint_min=[-1.0] * 6,
                joint_max=[1.0] * 6,
                max_joint_step=0.1,
                max_joint_speed=3.0,
            )
        )
        actions = np.array(
            [[1.5, -1.5, 0.05, 0.0, 0.0, 0.0, 0.5]], dtype=np.float32
        )

        safe_actions = validate_actions(
            actions, np.zeros(6, dtype=np.float32), cfg, fps=30
        )

        np.testing.assert_allclose(
            safe_actions[0, :6], [0.1, -0.1, 0.05, 0.0, 0.0, 0.0]
        )

    def test_clips_gripper_targets_before_returning_safe_actions(self) -> None:
        cfg = SimpleNamespace(
            safety=SimpleNamespace(
                joint_min=[-1.0] * 6,
                joint_max=[1.0] * 6,
                max_joint_step=0.5,
                max_joint_speed=15.0,
            )
        )
        actions = np.array(
            [
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.005],
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.005],
            ],
            dtype=np.float32,
        )

        safe_actions = validate_actions(
            actions, np.zeros(6, dtype=np.float32), cfg, fps=30
        )

        np.testing.assert_allclose(safe_actions[:, 6], [0.0, 1.0])
        np.testing.assert_allclose(actions[:, 6], [-0.005, 1.005])


class InitializeRobotTest(unittest.TestCase):
    def test_moves_to_configured_initial_pose_and_closes_gripper(self) -> None:
        class Arm:
            def __init__(self) -> None:
                self.commands: list[tuple[list[float], int]] = []

            def JointCtrl(self, joints: list[float], step_num: int) -> None:
                self.commands.append((joints, step_num))

        class Gripper:
            def __init__(self) -> None:
                self.positions: list[int] = []

            def set_pos(self, position: int) -> None:
                self.positions.append(position)

        arm = Arm()
        gripper = Gripper()
        initialize_robot(
            arm,
            gripper,
            init_joint=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            mock=False,
            ramp_steps=INITIAL_POSITION_RAMP_STEPS,
            gripper_position_scale=900,
        )

        self.assertEqual(len(arm.commands), 1)
        joints, step_num = arm.commands[0]
        np.testing.assert_allclose(joints, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(step_num, 250)
        self.assertEqual(gripper.positions, [0])

    def test_rejects_an_invalid_initial_pose(self) -> None:
        with self.assertRaisesRegex(ValueError, "6 joint values"):
            initialize_robot(
                arm=None,
                gripper=None,
                init_joint=[0.0] * 5,
                mock=True,
                ramp_steps=INITIAL_POSITION_RAMP_STEPS,
            )


if __name__ == "__main__":
    unittest.main()
