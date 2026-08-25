import os
import sys
import time
import hydra
from pathlib import Path
from omegaconf import DictConfig
from pyDHgripper import AG95

# import modules from the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from hardware.jaka_s5 import JOINT_COUNT, JakaS5
from hardware.ultrahands import UltrahandsClient
from hardware.zed import ZedCamera
from hardware.orbbec import OrbbecCamera
from src.data_collector import LeRobotDataCollector
from src.gripper import normalized_to_position
from src.teleop_display import DualCameraDisplay


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    fps = int(cfg.jaka_s5.freq_hz)
    agent_shape = (int(cfg.zed.output_height), int(cfg.zed.output_width), 3)  # HWC
    wrist_shape = (
        int(cfg.orbbec.output_height),
        int(cfg.orbbec.output_width),
        3,
    )  # HWC

    wrist_camera = None
    agent_camera = None
    arm = None
    ultrahands = None
    collector = None
    display = None

    try:
        # 初始化 Orbbec Camera
        wrist_camera = OrbbecCamera(**cfg.orbbec)
        wrist_camera.start()

        # 初始化 Zed Camera
        agent_camera = ZedCamera(**cfg.zed)
        agent_camera.start()

        # 启动arm
        arm = JakaS5(ip=str(cfg.jaka_s5.ip), freq_hz=fps)
        arm.start()

        # 启动gripper
        gripper = AG95(port=str(cfg.dh_gripper.port))
        gripper.set_force(int(cfg.dh_gripper.force))
        gripper.set_vel(int(cfg.dh_gripper.velocity))

        # 初始化 Ultrahands Client
        ultrahands = UltrahandsClient(**cfg.client)
        ultrahands.start()

        # 初始化数据采集器。一个运行目录可保存多个 teleop episode。
        data_name = cfg.data_collect.name
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        data_path = Path(root_dir) / "data" / data_name / timestamp
        collector = LeRobotDataCollector(
            repo_id="local/" + data_name,
            root=data_path,
            fps=fps,
            state_names=[*(f"joint_{i}.pos" for i in range(6)), "gripper.target_pos"],
            action_names=[*(f"joint_{i}.pos" for i in range(6)), "gripper.target_pos"],
            task=cfg.data_collect.prompt,
            camera_shapes={"wrist_view": wrist_shape, "agent_view": agent_shape},
        )

        display = DualCameraDisplay(**cfg.display)
        display.start()

        episode_count = 0
        # Keep devices and the dataset open for the whole collection session.
        # Completing an episode returns here and prepares the next one.
        while True:
            gripper.set_pos(0)  # 夹爪初始状态为闭合
            ramp_to_ultrahands(arm, ultrahands)  # 缓慢移动到 Ultrahands 位置
            teleop(
                arm,
                gripper,
                ultrahands,
                agent_camera,
                wrist_camera,
                collector,
                display,
                fps,
                int(cfg.dh_gripper.position_scale),
            )
            episode_count += 1
            print(f"Episode {episode_count} saved. Ready for the next episode.")
    except KeyboardInterrupt:
        print("Collection interrupted.")
    finally:
        if collector is not None:
            if collector.has_pending_frames():
                collector.discard_episode()
            collector.finalize()

        for resource in (
            display,
            ultrahands,
            arm,
            wrist_camera,
            agent_camera,
        ):
            if resource is not None:
                resource.stop()


def ramp_to_ultrahands(arm: JakaS5, ultrahands: UltrahandsClient):
    print(
        "press X to start ramping to ultrahands position in 2 seconds...",
        end="",
        flush=True,
    )
    # Require a fresh press for every episode start.
    last_x = bool(ultrahands.input_report.stick_l1_vertical)
    while True:
        x_pressed = bool(ultrahands.input_report.stick_l1_vertical)
        if x_pressed and not last_x:
            break
        last_x = x_pressed
        time.sleep(0.01)

    joint_pos = ultrahands.input_report.angles
    if joint_pos is None or len(joint_pos) < JOINT_COUNT:
        raise RuntimeError("No Ultrahands joint angles received for ramping")
    arm.JointCtrl(joint_pos[:JOINT_COUNT], step_num=250)  # 2 seconds
    time.sleep(2.0)
    print("done.")


def teleop(
    arm: JakaS5,
    gripper: AG95,
    ultrahands: UltrahandsClient,
    agent_view_camera: ZedCamera,
    wrist_camera: OrbbecCamera,
    collector: LeRobotDataCollector,
    display: DualCameraDisplay,
    fps: int,
    gripper_position_scale: int,
):
    print("teleop started, press Y to stop...", end="", flush=True)

    # frequency config
    dt = 1.0 / fps
    next_tick = time.perf_counter()

    # gripper state
    gripper_open = False
    gripper_target = 0.0

    # teleop loop
    # Ignore controls held while transitioning from the previous episode.
    initial_report = ultrahands.input_report
    last_y = bool(initial_report.stick_l1_horizontal)
    last_rb = bool(initial_report.stick_l2_vertical)
    while True:
        next_tick += dt
        report = ultrahands.input_report

        # arm control
        angles = report.angles
        if angles is None or len(angles) < JOINT_COUNT:
            raise RuntimeError("No Ultrahands joint angles received")
        arm.JointCtrl(angles[:JOINT_COUNT], 2)

        # gripper control
        rb_pressed = bool(report.stick_l2_vertical)
        if rb_pressed and not last_rb:
            gripper_open = not gripper_open
            gripper_target = 1.0 if gripper_open else 0.0
            gripper.set_pos(
                normalized_to_position(gripper_target, gripper_position_scale)
            )
        last_rb = rb_pressed

        # capture images
        agent_view_image = agent_view_camera.read()
        wrist_image = wrist_camera.read()

        # collect data
        joint_state = arm.get_joint_position()
        # gripper_state = gripper.read_pos() / 1000.0  # sleep(0.08), may block
        gripper_state = gripper_target
        if joint_state is None or len(joint_state) < JOINT_COUNT:
            raise RuntimeError("No JAKA joint feedback received")
        collector.record_step(
            state=[
                *joint_state[:JOINT_COUNT],
                gripper_state,
            ],
            action=[
                *angles[:JOINT_COUNT],
                gripper_target,
            ],
            images={"wrist_view": wrist_image, "agent_view": agent_view_image},
        )
        display.publish({"wrist_view": wrist_image, "agent_view": agent_view_image})

        # check stop condition
        y_pressed = bool(report.stick_l1_horizontal)
        if y_pressed and not last_y:
            collector.save_episode()
            print("done.")
            break
        last_y = y_pressed

        # frequency control
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            print(f"Total:{sleep_s}ms")
            next_tick = time.perf_counter()


if __name__ == "__main__":
    main()
