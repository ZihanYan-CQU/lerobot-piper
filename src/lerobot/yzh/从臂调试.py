import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V189, channel="can_follower")
robot = AgxArmFactory.create_arm(cfg)
end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot.connect()

fw = robot.get_firmware()
if fw is not None:
    print(fw)

robot.set_follower_mode()

time.sleep(0.5)
print("robotic arm is_ok =", robot.is_ok())
time.sleep(0.5)
print("effector is_ok =", end_effector.is_ok())

while not robot.enable():
    time.sleep(0.01)

# 设置臂的运动速度百分比
robot.set_speed_percent()

# 张开到 5cm，力 1N
end_effector.move_gripper_m(value=0.05, force=1.0)
time.sleep(1.0)

# 闭合（行程 0）
end_effector.move_gripper_m(value=0.0, force=1.0)