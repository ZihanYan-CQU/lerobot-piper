import time
import sys
import os
sys.path.append(os.path.dirname(__file__))
import orvyn_device

def signed_u8(value):
    return value - 256 if value >= 128 else value

def signed_i16(lo, hi):
    value = lo | (hi << 8)
    return value - 65536 if value >= 32768 else value

class OrvynDevice:
    # 命令码常量
    CMD_STATUS = 0x30
    CMD_LED_PARAM = 0x10
    CMD_LED_COLOR = 0x11
    CMD_LED_CONFIG = 0x12
    CMD_SLEEP_CONFIG = 0x13
    CMD_SLEEP_NOW = 0x14
    CMD_BUZZER_PRESET = 0x20
    CMD_BUZZER_CUSTOM = 0x21
    CMD_QUOTA_SET = 0x31
    CMD_SCREEN_ROTATE = 0x32

    # 预设音效映射
    BUZZER_PRESETS = {
        "stop": 0, "beep": 1, "double-beep": 2,
        "success": 3, "failure": 4, "alarm": 5
    }
    # LED模式映射
    LED_MODES = {"static": 0, "blink": 1, "breath": 2}

    def __init__(self, path=None, timeout_ms=1500):
        self.path = path
        self.timeout_ms = timeout_ms
        self.transport = None
        self.client = None
        self._status_cache = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """建立设备连接"""
        self.transport = orvyn_device.HidTransport(
            path=self.path, timeout_ms=self.timeout_ms
        )
        self.client = orvyn_device.RuntimeClient(self.transport)

    def close(self):
        """关闭设备连接"""
        if self.transport:
            self.transport.close()
            self.transport = None
            self.client = None

    def _wait_response(self, expected_cmd, timeout_ms=None):
        """等待并匹配响应命令"""
        timeout = timeout_ms or self.timeout_ms
        deadline = time.monotonic() + timeout / 1000.0
        while True:
            msg = self.client.app_channel_read()
            if msg["command"] == (expected_cmd | 0x80):
                return self._parse_status(msg["data"])
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Command 0x{expected_cmd:02X} timeout")
            time.sleep(0.01)

    def _parse_status(self,data):
        if len(data) < 15:
            return {}
        mode_map = {0: "static", 1: "blink", 2: "breath"}
        period_ms = data[13] | (data[14] << 8)
        status = {
            "mode": data[0],
            "mode_str": mode_map.get(data[0], f"unknown ({data[0]})"),
            "brightness_pct": data[1],
            "colors": [
                (data[2], data[3], data[4]),
                (data[5], data[6], data[7]),
                (data[8], data[9], data[10]),
            ],
            "buzzer_active": bool(data[11]),
            "buzzer_seq_active": bool(data[12]),
            "period_ms": period_ms,
        }
        if len(data) >= 26:
            status["floor_colors"] = [
                (data[15], data[16], data[17]),
                (data[18], data[19], data[20]),
                (data[21], data[22], data[23]),
            ]
            status["base_brightness_pct"] = data[24]
            status["max_brightness_pct"] = data[25]
        else:
            status["floor_colors"] = [(255, 0, 0), (255, 128, 0), (0, 255, 0)]
            status["base_brightness_pct"] = 5
            status["max_brightness_pct"] = 30
        if len(data) >= 32:
            status["auto_sleep_enabled"] = bool(data[26])
            status["sleep_timeout_s"] = data[27] | (data[28] << 8)
            status["sleeping"] = bool(data[29])
            status["keep_5v_in_sleep"] = bool(data[30])
            status["saved_5v_power"] = bool(data[31])
        else:
            status["auto_sleep_enabled"] = False
            status["sleep_timeout_s"] = 0
            status["sleeping"] = False
            status["keep_5v_in_sleep"] = False
            status["saved_5v_power"] = False
        if len(data) >= 48:
            status["codex_5h_remaining_percent"] = data[32]
            status["codex_week_remaining_percent"] = data[33]
            status["display_rotation_index"] = data[34]
            status["display_ready"] = bool(data[35])
            status["sensor_status_temperature"] = signed_u8(data[36])
            status["sensor_status_humidity"] = signed_u8(data[37])
            status["sensor_status_light"] = signed_u8(data[38])
            status["temperature_milli_c"] = signed_i16(data[39], data[40])
            status["humidity_milli_percent"] = signed_i16(data[41], data[42])
            status["light_mlux"] = data[43] | (data[44] << 8) | (data[45] << 16) | (data[46] << 24)
            status["display_attached"] = bool(data[47])
            status["display_detected"] = bool(data[48]) if len(data) >= 52 else status["display_attached"]
            status["display_bus"] = data[49] if len(data) >= 52 else 255
            status["display_address"] = data[50] if len(data) >= 52 else 0
            status["display_status"] = signed_u8(data[51]) if len(data) >= 52 else 0
        else:
            status["codex_5h_remaining_percent"] = 255
            status["codex_week_remaining_percent"] = 255
            status["display_rotation_index"] = 0
            status["display_ready"] = False
            status["sensor_status_temperature"] = -1
            status["sensor_status_humidity"] = -1
            status["sensor_status_light"] = -1
            status["temperature_milli_c"] = 0
            status["humidity_milli_percent"] = 0
            status["light_mlux"] = 0
            status["display_attached"] = False
            status["display_detected"] = False
            status["display_bus"] = 255
            status["display_address"] = 0
            status["display_status"] = 0
        return status


    # -------------------------- 公共业务方法 --------------------------
    def get_status(self, refresh=True):
        """获取设备全量状态"""
        if refresh or not self._status_cache:
            self.client.app_channel_write(self.CMD_STATUS, b"")
            self._status_cache = self._wait_response(self.CMD_STATUS)
        return self._status_cache

    def set_led_color(self, colors):
        """设置LED颜色
        :param colors: 列表，1-3个RGB元组，不足3个自动补全
        """
        # 参数补全与校验
        colors = list(colors)
        if len(colors) < 3:
            while len(colors) < 3:
                colors.append(colors[-1])
        colors = colors[:3]
        # 组装payload
        payload = bytearray()
        for r, g, b in colors:
            payload.extend((r, g, b))
        self.client.app_channel_write(self.CMD_LED_COLOR, bytes(payload))
        self._wait_response(self.CMD_LED_COLOR)

    def set_led_mode(self, mode=None, brightness=None, period=None):
        """设置LED模式、亮度、周期"""
        curr = self.get_status()
        mode_val = self.LED_MODES[mode] if mode else curr["mode"]
        brightness_val = brightness if brightness is not None else curr["brightness_pct"]
        period_val = period if period is not None else curr["period_ms"] or 1000

        if not 0 <= brightness_val <= 100:
            raise ValueError("Brightness must be 0-100")

        payload = bytearray(4)
        payload[0] = mode_val
        payload[1] = period_val & 0xFF
        payload[2] = (period_val >> 8) & 0xFF
        payload[3] = brightness_val
        self.client.app_channel_write(self.CMD_LED_PARAM, bytes(payload))
        return self._wait_response(self.CMD_LED_PARAM)

    def set_led_config(self, base_colors=None, base_brightness=None, max_brightness=None):
        """设置LED持久化基础配置"""
        curr = self.get_status()
        colors = base_colors if base_colors else curr["floor_colors"]
        base_bri = base_brightness if base_brightness is not None else curr["base_brightness_pct"]
        max_bri = max_brightness if max_brightness is not None else curr["max_brightness_pct"]

        if base_bri > max_bri:
            raise ValueError("Base brightness cannot exceed max brightness")
        if not (0 <= base_bri <= 30 and 0 <= max_bri <= 30):
            raise ValueError("Brightness must be 0-30%")

        payload = bytearray()
        for r, g, b in colors:
            payload.extend((r, g, b))
        payload.extend((base_bri, max_bri))
        self.client.app_channel_write(self.CMD_LED_CONFIG, bytes(payload))
        return self._wait_response(self.CMD_LED_CONFIG)

    def buzzer_play(self, preset):
        """播放预设音效"""
        if preset not in self.BUZZER_PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        self.client.app_channel_write(
            self.CMD_BUZZER_PRESET, bytes([self.BUZZER_PRESETS[preset]])
        )
        self._wait_response(self.CMD_BUZZER_PRESET)

    def buzzer_custom(self, notes, loop=False):
        """播放自定义蜂鸣器序列
        :param notes: 列表，每组(频率Hz, 时长ms)，最多6组
        :param loop: 是否循环
        """
        if len(notes) > 6:
            notes = notes[:6]
        payload = bytearray(2 + len(notes)*8)
        payload[0] = len(notes)
        payload[1] = 1 if loop else 0
        for idx, (freq, dur) in enumerate(notes):
            offset = 2 + idx*8
            payload[offset:offset+4] = freq.to_bytes(4, byteorder="little")
            payload[offset+4:offset+8] = dur.to_bytes(4, byteorder="little")
        self.client.app_channel_write(self.CMD_BUZZER_CUSTOM, bytes(payload))
        self._wait_response(self.CMD_BUZZER_CUSTOM)

    def set_sleep_config(self, enable=None, timeout_s=None, keep_5v=None):
        """配置自动休眠"""
        curr = self.get_status()
        enabled = enable if enable is not None else curr["auto_sleep_enabled"]
        timeout = timeout_s if timeout_s is not None else curr["sleep_timeout_s"]
        keep5v = keep_5v if keep_5v is not None else curr["keep_5v_in_sleep"]

        if timeout < 5:
            raise ValueError("Sleep timeout must be at least 5 seconds")

        payload = bytearray(4)
        payload[0] = 1 if enabled else 0
        payload[1] = timeout & 0xFF
        payload[2] = (timeout >> 8) & 0xFF
        payload[3] = 1 if keep5v else 0
        self.client.app_channel_write(self.CMD_SLEEP_CONFIG, bytes(payload))
        return self._wait_response(self.CMD_SLEEP_CONFIG)

    def sleep_now(self):
        """立即进入休眠"""
        self.client.app_channel_write(self.CMD_SLEEP_NOW, b"")
        self._wait_response(self.CMD_SLEEP_NOW)

    def set_screen_rotate(self, angle):
        """设置屏幕旋转角度"""
        if angle not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0/90/180/270")
        payload = angle.to_bytes(2, byteorder="little")
        self.client.app_channel_write(self.CMD_SCREEN_ROTATE, payload)
        self._wait_response(self.CMD_SCREEN_ROTATE)

    def set_quota(self, five_hour, week):
        """手动设置Codex配额百分比"""
        if not (0 <= five_hour <= 100 and 0 <= week <= 100):
            raise ValueError("Quota values must be 0-100")
        self.client.app_channel_write(self.CMD_QUOTA_SET, bytes([five_hour, week]))
        self._wait_response(self.CMD_QUOTA_SET)

'''
dev = OrvynDevice(timeout_ms=2000)
dev.connect()
#dev.set_led_color([(255, 0, 0), (255, 127, 0), (0, 255, 0)])
#dev.set_led_config(base_brightness=0)
dev.set_led_color([(255, 0, 0), (0, 0, 0), (0, 255, 0)])
dev.set_sleep_config(enable=True, timeout_s=120, keep_5v=True)
dev.close()
'''