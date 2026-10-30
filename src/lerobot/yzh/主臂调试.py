import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V189, channel="can_master")
robot = AgxArmFactory.create_arm(cfg)
end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot.connect()
robot.set_follower_mode()
time.sleep(1)

# 设置臂的运动速度百分比
robot.set_speed_percent()

# 张开到 5cm，力 1N
end_effector.move_gripper_m(value=0.05, force=1.0)
time.sleep(1.0)

# 闭合（行程 0）
end_effector.move_gripper_m(value=0.0, force=1.0)
'''rating = robot.get_joint_assistance_rating()
if rating is not None:
    print(rating.msg)

param = end_effector.get_gripper_teaching_pendant_param()
if param is not None:
    print(
        param.msg.teaching_range_per,
        param.msg.max_range_config,
        param.msg.teaching_friction,
    )
    print(param.hz, param.timestamp)'''