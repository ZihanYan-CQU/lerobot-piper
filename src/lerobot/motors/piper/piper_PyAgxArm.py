import logging
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

def get_temp_color(temp: float, limit: float) -> str:
    if limit <= 0:
        return "white"
    ratio = temp / limit
    if ratio < 0.7:
        return "#00FF00"  # 绿色
    elif ratio < 0.85:
        t = (ratio - 0.7) / 0.15
        r = int(255 * t)
        return f"#{r:02X}FF00"  # 绿→黄
    elif ratio < 0.95:
        t = (ratio - 0.85) / 0.10
        g = int(255 * (1 - t))
        return f"#FF{g:02X}00"  # 黄→红
    else:
        return "#FF0000"  # 红色

def action_pos_dict_from_piper_read(raw: dict[str, Any]) -> dict[str, float]:
    """Convert `PiperMotorsBus.read()` output to LeRobot `joint_i.pos` (rad) and `gripper.pos` (m stroke)."""
    out: dict[str, float] = {}
    for i in range(1, 7):
        key = f"joint_{i}"
        out[f"{key}.pos"] = float(raw[key])
    out["gripper.pos"] = float(raw["gripper"])
    return out

def get_display_width(text: str) -> int:
    """计算文本在终端实际显示宽度，中文占2，英文/空格占1"""
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width

def visual_pad(text: str, target_width: int, align: str = "<") -> str:
    """
    按终端可视宽度补齐空格
    align: < 左对齐  > 右对齐  ^居中
    """
    current_w = get_display_width(text)
    pad_total = max(0, target_width - current_w)
    if align == "<":
        return text + " " * pad_total
    elif align == ">":
        return " " * pad_total + text
    else:
        left = pad_total // 2
        right = pad_total - left
        return " " * left + text + " " * right

def fmt_row(data):
    return " ".join(f"{v:^4}" for v in data)

def move_cursor_up(lines):
    """Move the cursor up by a specified number of lines."""
    print(f"\033[{lines}A", end="")

def move_cursor_down(lines):
    """Move the cursor down by a specified number of lines."""
    print(f"\033[{lines}B", end="")

def is_joint_target_reached(current_joints, target_joints, tolerance=0.02):
    """
    判断当前关节角是否到达目标位置。

    Args:
        current_joints: 当前关节角，例如 [0, 0.4, -0.4, 0, -0.4, 0]
        target_joints: 目标关节角
        tolerance: 单个关节允许的最大误差

    Returns:
        bool: 所有关节误差都不超过 tolerance 时返回 True
    """
    if len(current_joints) != len(target_joints):
        return False

    return all(
        abs(current - target) <= tolerance
        for current, target in zip(current_joints, target_joints)
    )

@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]

class PiperMotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self, 
                 config: PiperMotorsBusConfig):
        self.motors = config.motors
        self.cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.V189, channel=config.can_name)
        self.piper = AgxArmFactory.create_arm(self.cfg)
        self.end_effector = self.piper.init_effector(self.piper.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.piper.connect()

        # 录制数据集时改成0
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.LOG_LENGTH = 77
        self.LOG_START_LENGTH = self.LOG_LENGTH + 10*2 + 2
        self.LOG_END_LENGTH = self.LOG_LENGTH + 10*2 + 2
    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    机械臂动作控制
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def write(self, target_position:list):
    
        target_joint = target_position[:-1]
        gripper_range = target_position[-1]

        self.piper.move_j(target_joint)
        self.end_effector.move_gripper_m(value=abs(gripper_range), force=1.0) 

    def write_js(self, target_position:list):
        
        target_joint = target_position[:-1]
        gripper_range = target_position[-1]

        self.piper.move_js(target_joint)
        self.end_effector.move_gripper_m(value=abs(gripper_range), force=1.0)

    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    机械臂控制消息格式统一和单位统一
    action_pos_dict_from_piper_read: Convert `PiperMotorsBus.read()` output to LeRobot `joint_i.pos` (rad) and `gripper.pos` (m stroke).
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def read_action_positions(self,raw: dict[str, Any]) -> dict[str, float]:
        """Joint positions and gripper stroke in the same units as `write()` (rad / m)."""
        return action_pos_dict_from_piper_read(raw)

    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    温度监控
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def temperature_monitor(self) -> dict[str,list]:
            monitor_data = {
                "joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"],
                "motor_temp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 电机温度
                "driver_temp": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # 驱动器温度
            }
            try:
                for i in range(1,7):
                    status = self.piper.get_driver_states(i).msg
                    monitor_data["motor_temp"][i-1] = status.motor_temp
                    monitor_data["driver_temp"][i-1] = status.foc_temp
            except:
                pass
            return monitor_data

    def temperature_monitor_loop(
        self,
        events: dict,
        temperature_limit: float,
        check_interval: float,
        stop_event: threading.Event,
        on_overheat: Callable[[float, float], None] | None = None,
        display_temperature: bool = False,   # 是否持续在顶部显示温度
    ):
        """
        守护线程：周期检查关节温度，超限则失能机器人并请求主循环退出。
        """
        logging.info("Temperature monitor started.")
        root = None

        if display_temperature:
            try:
                import tkinter as tk
                root = tk.Tk()
                root.title("关节温度监控")
                root.attributes("-topmost", True)
                root.configure(bg='black')

                # 字体设置
                font = ("SimHei", 20)
                col_width = 10  # 每列宽度（字符数），确保对齐

                # 创建标签引用字典，用于后续更新
                temp_labels = {
                    'motor': [],
                    'driver': [],
                    'header': []
                }

                # 第一行：表头（关节名称）
                for col in range(6):  # 假设固定 6 个关节，可根据实际调整
                    lbl = tk.Label(root, text="", font=font, bg='black', fg='white',
                                width=col_width, anchor='center')
                    lbl.grid(row=0, column=col, padx=2, pady=2)
                    temp_labels['header'].append(lbl)

                # 第二行：电机温度
                for col in range(6):
                    lbl = tk.Label(root, text="0.0", font=font, bg='black', fg='green',
                                width=col_width, anchor='center')
                    lbl.grid(row=1, column=col, padx=2, pady=2)
                    temp_labels['motor'].append(lbl)

                # 第三行：驱动器温度
                for col in range(6):
                    lbl = tk.Label(root, text="0.0", font=font, bg='black', fg='green',
                                width=col_width, anchor='center')
                    lbl.grid(row=2, column=col, padx=2, pady=2)
                    temp_labels['driver'].append(lbl)

                root.update()
            except Exception as e:
                logging.warning(f"无法创建 GUI 窗口（{e}）。")

        while not stop_event.is_set():
            try:
                # 调用机器人自带的温度读取函数（并发安全）
                monitor_data = self.temperature_monitor()

                # 提取所有温度值（电机 + 驱动器）
                joint_names = monitor_data.get("joint_names", [])
                motor_temps = monitor_data.get("motor_temp", [])
                driver_temps = monitor_data.get("driver_temp", [])
                all_temps = motor_temps + driver_temps

                if display_temperature:
                    if joint_names and len(joint_names) == len(motor_temps) == len(driver_temps):
                                # 更新表头（关节名称）
                        for col, name in enumerate(joint_names):
                            temp_labels['header'][col].config(text=name)
                        # 更新电机温度
                        for col, temp in enumerate(motor_temps):
                            color = get_temp_color(temp, temperature_limit)
                            temp_labels['motor'][col].config(text=f"{temp:.1f}", fg=color)
                        # 更新驱动温度
                        for col, temp in enumerate(driver_temps):
                            color = get_temp_color(temp, temperature_limit)
                            temp_labels['driver'][col].config(text=f"{temp:.1f}", fg=color)

                    root.update()

                if all_temps:
                    max_temp = max(all_temps)
                    if max_temp > temperature_limit:
                        logging.error(
                            f"温度 {max_temp:.1f}°C 抵近阈值 {temperature_limit}°C，3秒后启动从臂失能程序..."
                        )
                        if on_overheat is not None:
                            try:
                                on_overheat(max_temp, temperature_limit)
                            except Exception as e:
                                logging.warning(f"发送温度超限回调失败: {e}")
                        time.sleep(3)
                        # 调用失能方法（根据实际机器人 API 调整）
                        self.safe_disconnect_follower(wait_time=10)
                        events["stop_recording"] = True   # 通知主循环退出
                        break
            except Exception as e:
                # 读取失败时记录警告，不终止线程
                logging.warning(f"读取温度失败: {e}")
            stop_event.wait(check_interval)

        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        
    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    读取机械臂助力等级和示教器助力等级
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''
    def get_assistance_rating(self) -> None:
        rating = self.piper.get_joint_assistance_rating()
        if rating is not None:
            print(f"│──────────{visual_pad(f'关节助力等级：{rating.msg}', self.LOG_LENGTH, '^')}──────────│")

        param = self.end_effector.get_gripper_teaching_pendant_param()
        if param is not None:
            print(f"│──────────{visual_pad(f'夹爪助力等级：{param.msg.teaching_friction}', self.LOG_LENGTH, '^')}──────────│")
        time.sleep(1)
        
    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    连接机械臂
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def connect_follower(self, enable:bool) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            
            if(enable):
                enable_flag = self.piper.get_joint_enable_status(255)
                while( not self.piper.enable()):
                    print(f"│──────────{visual_pad('从臂初始化中', self.LOG_LENGTH, '^')}──────────│")
                    time.sleep(0.1)
                self.end_effector.move_gripper_m(value=0.0, force=1.0)
                enable_log = visual_pad(str(enable_flag), 5, '^')
                print(f"│──────────{visual_pad(f'从臂使能状态：{enable_log}', self.LOG_LENGTH, '^')}──────────│")
            else:
                # move to safe disconnect position
                #self.piper.disable(255)
                self.electronic_emergency_stop()
                self.end_effector.disable_gripper()
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        #print(f"Returning response: {resp}")
        return resp

    def connect_leader(self) -> None:
        #time.sleep(0.5)
        #主臂示教器通过该函数得到的状态只有False
        #print("----------主臂示教器连接状态：", self.gripper.is_ok(),"----------")
        
        self.piper.set_follower_mode()
        time.sleep(1)

        # 设置臂的运动速度百分比
        self.piper.set_speed_percent()

        # 张开到 5cm，力 1N
        self.end_effector.move_gripper_m(value=0.05, force=1.0)
        time.sleep(1)

        # 闭合（行程 0）
        self.end_effector.move_gripper_m(value=0.0, force=1.0)
        time.sleep(1)

    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    回零
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''
    
    def apply_calibration_follower(self, log=True, arm_name="从臂") -> bool:
        """
        移动到初始位置，并等待从臂到达目标关节位置。
        """
        if log:
            print(f"│──────────{visual_pad(f'{arm_name}回零', self.LOG_LENGTH, '^')}──────────│")

        target_position = self.init_joint_position

        start_t = time.monotonic()
        while True:
            self.write(target_position=target_position)
            # 读取当前从臂位置
            current_position_dict = self.read_follower()
            current_position = [
                current_position_dict["joint_1"],
                current_position_dict["joint_2"],
                current_position_dict["joint_3"],
                current_position_dict["joint_4"],
                current_position_dict["joint_5"],
                current_position_dict["joint_6"],
                current_position_dict["gripper"],
            ]
            if is_joint_target_reached(
                current_position,
                target_position,
                tolerance=0.01,
            ):
                return True
            if time.monotonic() - start_t > 5.0:
                return False
            time.sleep(0.5)


    def apply_calibration_leader(self,log=True):
        """
            leader移动到初始位置
        """
        if(log):
            print(f"│──────────{visual_pad('主臂回零', self.LOG_LENGTH, '^')}──────────│")
        target_position = self.init_joint_position
        '''
        Piper 主臂在主从模式下，官方 SDK 文档明确说它主要发送的是控制帧，比如 0x155/0x156/0x157，而 get_leader_joint_angles() 读的也正是这些“主臂发送给从臂的目标值/控制值”，不是一个独立的实时电机位置反馈通道。

        没有这个time.sleep(1)的话，主臂最开始第一次调用回零时会报错
        因为主臂的所有读取状态的函数，实际上都是读取缓存中向从臂发送的控制消息，而这个控制消息，通常情况下是静默的（被动缓存），只有在主臂运动时才会主动发出
        在上电后初始激发CAN和进入Leader模式时，会向缓存中主动发送一帧控制消息
        这1秒真正提供的是：让后台CAN接收线程有机会在move_leader_to_home以前，捕获到至少一帧Leader发送的控制消息数据，进而使得get_leader_joint_angles和get_gripper_ctrl_states不会返回None

        record代码中：
        teleop.calibrate(log=False)
        act = teleop.get_action()
        print(act)
        robot.send_action(act)
        calibrate调用apply_calibration_leader，如果apply_calibration_leader调用move_leader_to_home，这会使得机械臂进入非leader和非follower的特殊模式，该模式下机械臂运动并不会发送控制消息
        因此get_leader_joint_angles和get_gripper_ctrl_states只能读取到机械臂回零之前所处位置的关节和夹爪状态，使得回零失败

        然而，如果调用的是move_leader_follower_to_home，此时的主臂会主动发送控制消息给从臂，使得缓存中的数据会持续更新
        这一下，直接同时解决了上述两个问题的痛点，不需要sleep，从臂也能正常回零！！！！！
        '''
        
        self.piper.move_leader_follower_to_home()
        time.sleep(4)
        '''start_t = time.monotonic()
        while True:
            ja = self.piper.get_joint_angles()
            gs = self.end_effector.get_gripper_status()

            if ja is not None and gs is not None:
                current_position = list(ja.msg)
                current_position.append(gs.msg.value)
                print(current_position)
                if is_joint_target_reached(
                    current_position,
                    target_position,
                    tolerance=0.01,
                ):
                    break
            if time.monotonic() - start_t > 5.0:
                break
            time.sleep(0.5)'''
        self.piper.restore_leader_drag_mode()
        time.sleep(1)
        self.piper.move_leader_follower_to_home()
        time.sleep(1)
        self.piper.restore_leader_drag_mode()

    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    读取机械臂关节和末端执行器消息
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def read_follower(self) -> dict[str, Any]:
        """
            - 机械臂关节消息,单位0.001度
            - 机械臂夹爪消息
        """
        joint_msg = self.piper.get_joint_angles().msg
        gripper_msg = self.end_effector.get_gripper_status().msg
        return {
            "joint_1": joint_msg[0],
            "joint_2": joint_msg[1],
            "joint_3": joint_msg[2],
            "joint_4": joint_msg[3],
            "joint_5": joint_msg[4],
            "joint_6": joint_msg[5],
            "gripper": gripper_msg.value
        }

    def try_read_leader(self) -> dict[str, Any] | None:
        """Return one complete leader control sample, or ``None`` while its cache is not ready.

        The Piper SDK explicitly permits both leader-control getters to return
        ``None`` immediately after a follower-to-leader mode switch.  Keeping
        that temporary state out of ``.msg`` access lets handover code wait for
        the first complete 0x155/0x156/0x157/0x159 control sample safely.
        """

        joint_state = self.piper.get_leader_joint_angles()
        gripper_state = self.end_effector.get_gripper_ctrl_states()
        if joint_state is None or gripper_state is None:
            return None

        joint_msg = joint_state.msg
        gripper_msg = gripper_state.msg
        return {
            "joint_1": joint_msg[0],
            "joint_2": joint_msg[1],
            "joint_3": joint_msg[2],
            "joint_4": joint_msg[3],
            "joint_5": joint_msg[4],
            "joint_6": joint_msg[5],
            "gripper": gripper_msg.value,
        }

    def read_leader(self) -> dict[str, Any]:
        """Read a complete leader control sample.

        Callers that can tolerate the mode-switch warm-up should use
        :meth:`try_read_leader` and retry when it returns ``None``.
        """

        sample = self.try_read_leader()
        if sample is None:
            raise RuntimeError("Piper leader control frames are not ready")
        return sample

    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    安全失能
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def safe_disconnect_follower(self,wait_time):
        """ 
            Move to safe disconnect position
        """
        print(f"│──────────{visual_pad('从臂回到安全失能位置', self.LOG_LENGTH, '^')}──────────│")
        self.write(target_position=self.safe_disable_position)
        time.sleep(3)
        '''print(f"│──────────{visual_pad(f'从臂在{wait_time}秒后失能！！！请注意扶好从臂避免失能下砸导致硬件损伤！', self.LOG_LENGTH, '^')}──────────│")
        time.sleep(wait_time)'''

        for i in range(int(wait_time), 0, -1):
            msg = f'从臂在{i}秒后失能！！！请注意扶好从臂避免失能下砸导致硬件损伤！'
            # 使用 end='\r' 让光标回到行首，flush=True 立即刷新输出
            print(f"│──────────{visual_pad(msg, self.LOG_LENGTH, '^')}──────────│", end='\r', flush=True)
            time.sleep(1)
        print(f"│──────────{visual_pad('从臂已失能', self.LOG_LENGTH, '^')}──────────│", end='\r', flush=True)
        # 循环结束后换行，避免后续输出接在倒计时行后
        print()
        self.connect_follower(enable=False)

    def safe_disconnect_leader(self):
        """ 
            Move to safe disconnect position
        """
        print(f"│──────────{visual_pad('主臂回到安全失能位置', self.LOG_LENGTH, '^')}──────────│")
        self.apply_calibration_leader()

    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    电子急停：将机械臂设置为急停状态。如果执行时机械臂关节处于抬起状态，机械臂会缓慢以恒定阻尼落下（不会立刻掉落，实测还是很快，阻尼不够大）
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def electronic_emergency_stop(self) -> None:
        self.piper.electronic_emergency_stop()
        time.sleep(3)
        self.piper.reset()

    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    联动模式切换
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def change_to_leader_mode(self) -> None:
        print(f"│──────────{visual_pad('主臂切换到主导臂零力拖动模式', self.LOG_LENGTH, '^')}──────────│")
        time.sleep(1)
        self.piper.set_leader_mode()

    def change_to_follower_mode(self) -> None:
        print(f"│──────────{visual_pad('从臂切换到跟随臂模式', self.LOG_LENGTH, '^')}──────────│")
        time.sleep(1)
        self.piper.set_follower_mode()

    '''
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    读取固件信息
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    '''

    def get_firware_version_follower(self) -> None:
        hardware_version = None
        software_version = None
        attempts = 0
        max_retries = 10

        while (hardware_version is None or software_version is None) and attempts < max_retries:
            firmware_info = self.piper.get_firmware()
            # 使用 .get() 避免键缺失导致 KeyError
            try:
                hardware_version = firmware_info.get('hardware_version')
                software_version = firmware_info.get('software_version')
            except:
                hardware_version = None
                software_version = None
            attempts += 1
            time.sleep(0.3)  # 等待一段时间再重试

        print(f"│──────────{visual_pad(f'从臂硬件版本: {hardware_version}', self.LOG_LENGTH, '^')}──────────│")
        print(f"│──────────{visual_pad(f'从臂软件版本: {software_version}', self.LOG_LENGTH, '^')}──────────│")

    def get_firware_version_leader(self) -> None:
        hardware_version = None
        software_version = None
        attempts = 0
        max_retries = 10

        while (hardware_version is None or software_version is None) and attempts < max_retries:
            firmware_info = self.piper.get_firmware()
            # 使用 .get() 避免键缺失导致 KeyError
            hardware_version = firmware_info.get('hardware_version')
            software_version = firmware_info.get('software_version')
            try:
                hardware_version = firmware_info.get('hardware_version')
                software_version = firmware_info.get('software_version')
            except:
                hardware_version = None
                software_version = None
            attempts += 1
            time.sleep(0.3)  # 等待一段时间再重试

        print(f"│──────────{visual_pad(f'主臂硬件版本: {hardware_version}', self.LOG_LENGTH, '^')}──────────│")
        print(f"│──────────{visual_pad(f'主臂软件版本: {software_version}', self.LOG_LENGTH, '^')}──────────│")
