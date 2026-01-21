import os
import sys

# import modules from the root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from hardware.jaka_s5 import JakaS5

def main():
    robot = JakaS5(ip="192.168.2.121")
    robot.start()
    print(robot.get_joint_position())

if __name__ == "__main__":
    main()