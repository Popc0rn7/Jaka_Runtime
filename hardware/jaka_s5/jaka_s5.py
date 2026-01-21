import os
import ctypes

# load jaka shared libraries
jaka_dir = os.path.dirname(__file__)
ctypes.CDLL(os.path.join(jaka_dir, "libjakaAPI.so"), mode=ctypes.RTLD_GLOBAL)
from . import jkrc

ABS = 0
REL = 1
BLOCK = True
NONBLOCK = False
SPEED = 1
ACC = 1
TOL = 20


class JakaS5:

    def __init__(self, ip: str, freq_hz: int = 125):
        self.robot = jkrc.RC(ip)

        # 设置运动控制频率
        move_interval_ms = int(1000 / freq_hz)
        self.step_num = max(1, move_interval_ms // 8)  # 每步8ms，至少1步

    def start(self):
        """
        启动机器人：包含登录、上电、使能
        """
        self.robot.login()
        self.robot.power_on()
        self.robot.enable_robot()
        self.robot.servo_move_enable(True)

    def JointCtrl(self, joint_pos: list[float], step_num: int = 2):
        """
        关节运动控制 (MoveJ)
        :param joint_pos: 7个关节的弧度列表
        """
        self.robot.servo_j(
            joint_pos=joint_pos,
            move_mode=0,  # 0:绝对位置
            step_num=step_num,  # 在 step_num * 8ms 内到达目标位置
        )

    def stop(self):
        """
        关闭机器人：下使能、下电、断开连接
        """
        self.robot.servo_move_enable(False)
        self.robot.disable_robot()

    def get_joint_position(self):
        """
        获取当前机器人的位置信息
        """
        ret = self.robot.get_joint_position()
        return ret[1]
