from pyDHgripper import AG95

gripper = AG95(port='/dev/ttyUSB1')
print(gripper.read_state())