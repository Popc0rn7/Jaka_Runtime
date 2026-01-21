from .lib import jkrc

ABS = 0
REL = 1
BLOCK = True
NONBLOCK = False
SPEED = 1
ACC = 1
TOL = 20

class JakaS5:

    def __init__(self, ip="192.168.2.106"):
        self.robot = jkrc.RC(ip)

    def start(self):
        """
        启动机器人：包含登录、上电、使能
        """
        self.robot.login()
        self.robot.power_on()
        self.robot.enable_robot()

    def JointCtrl(self,joint_pos: list[float]):
        """
        关节运动控制 (MoveJ)
        :param arm_id: 0 为左臂, 1 为右臂
        :param joint_pos: 7个关节的弧度列表
        :param is_block: 是否阻塞等待运动完成
        """
        self.robot.joint_move_extend(
            joint_pos, ABS, True, SPEED, ACC, TOL
        )
    def stop(self):
        """
        关闭机器人：下使能、下电、断开连接
        """
        (failed,) = self.robot.disable_robot()
        if failed:
            print(f"下使能失败: {failed}")
        (failed,) = self.robot.power_off()
        if failed:
            print(f"下电失败: {failed}")

    def get_info(self):
        """
        获取当前机器人的位置信息
        """
        ret = self.robot.get_joint_position()
        if ret[0]:
            print(f"获取关节位置失败: {ret[0]}")
        joint_pos = ret[1]
        ret = self.robot.get_tcp_position()
        if ret[0]:
            print(f"获取末端位置失败: {ret[0]}")
        tcp_pos = ret[1]
        return joint_pos, tcp_pos
