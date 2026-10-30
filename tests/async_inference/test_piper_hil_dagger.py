# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Hardware-free tests for Piper's DAgger takeover primitives."""

import math
import queue
import threading
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import lerobot.async_inference.client.piper_hil_dagger as dagger_module
from lerobot.async_inference.client.piper_hil_dagger import (
    PIPER_ACTION_KEYS,
    PIPER_EXPERT_CONTROL_IDS,
    PIPER_POSE_FEEDBACK_IDS,
    DaggerCorrectionRecorder,
    DaggerDatasetConfig,
    PiperCANBridgeDagger,
    PiperControlMode,
    PiperDaggerController,
    PiperDaggerHardwareConfig,
    PiperDaggerHardwareError,
    PiperHandoverAlignmentError,
)
from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features


def _pose(value: float) -> dict[str, float]:
    return dict.fromkeys(PIPER_ACTION_KEYS, value)


class _FakeBus:
    def __init__(
        self,
        name: str,
        events: list[str],
        pose: dict[str, float] | None = None,
        firmware: str = "S-V1.8-9",
    ):
        self.name = name
        self.events = events
        self.pose = _pose(0.0) if pose is None else dict(pose)
        self.writes: list[list[float]] = []
        self.init_joint_position = [0.0] * 7
        self.piper = SimpleNamespace(
            get_firmware=lambda: {"software_version": firmware},
            get_joint_enable_status=lambda _joint: True,
            enable=lambda: True,
            disconnect=lambda: self.events.append(f"{self.name}:disconnect"),
        )

    def change_to_follower_mode(self):
        self.events.append(f"{self.name}:follower")

    def change_to_leader_mode(self):
        self.events.append(f"{self.name}:leader")

    def apply_calibration_follower(self, log=True, arm_name="follower"):
        del log, arm_name
        self.pose = _pose(0.0)
        self.events.append(f"{self.name}:home_as_follower")
        return True

    def write_js(self, values):
        values = list(values)
        self.writes.append(values)
        self.events.append(f"{self.name}:write")

    def read_follower(self):
        return dict(self.pose)

    def read_leader(self):
        return dict(self.pose)

    @staticmethod
    def read_action_positions(raw):
        return dict(raw)


class _FakeBridge:
    def __init__(self, events: list[str], *, master_bus=None, follower_bus=None):
        self.events = events
        self.master_bus = master_bus
        self.follower_bus = follower_bus

    def start(self):
        self.events.append("bridge:start")

    def enable(self):
        self.events.append("bridge:enable")

    def disable(self):
        self.events.append("bridge:disable")

    def drain(self):
        self.events.append("bridge:drain")
        return 0

    def wait_for_follower_feedback(self, timeout_s):
        self.events.append(f"bridge:feedback:{timeout_s}")
        return 0x2A1

    def wait_for_master_feedback(self, timeout_s):
        self.events.append(f"bridge:master_feedback:{timeout_s}")
        return 0x2A1

    def read_master_feedback_pose(self, timeout_s):
        self.events.append("bridge:master_pose")
        return self.master_bus.read_follower() if self.master_bus is not None else _pose(0.0)

    def read_follower_feedback_pose(self, timeout_s):
        self.events.append("bridge:follower_pose")
        return self.follower_bus.read_follower() if self.follower_bus is not None else _pose(0.0)

    def wait_for_master_alignment(self, reference, config, *, trigger):
        self.events.append("bridge:master_alignment")
        trigger()
        pose = self.master_bus.read_leader() if self.master_bus is not None else _pose(0.0)
        joint_error = max(abs(pose[key] - reference[key]) for key in PIPER_ACTION_KEYS[:6])
        gripper_error = abs(pose["gripper.pos"] - reference["gripper.pos"])
        if joint_error > config.handover_joint_tolerance or gripper_error > config.handover_gripper_tolerance:
            raise PiperHandoverAlignmentError(f"gripper_error={gripper_error:.4f} m")
        return pose

    def close(self):
        self.events.append("bridge:close")


class _FakePiperRobot:
    name = "piper_follower"
    action_features = dict.fromkeys(PIPER_ACTION_KEYS, float)
    observation_features = dict.fromkeys(PIPER_ACTION_KEYS, float)

    def __init__(self, bus):
        self.bus = bus
        self.config = SimpleNamespace(teleop_joint_alpha=0.5, teleop_gripper_alpha=0.5)
        self._ema_cmd = None

    def get_observation(self):
        return dict(self.bus.pose)


def test_can_bridge_forwards_only_physical_standard_control_frames():
    valid = SimpleNamespace(
        is_rx=True,
        is_error_frame=False,
        is_extended_id=False,
        is_remote_frame=False,
        is_fd=False,
        arbitration_id=0x151,
    )
    assert PiperCANBridgeDagger.should_forward(valid)

    for update in (
        {"is_rx": False},
        {"is_rx": None},
        {"is_error_frame": True},
        {"is_extended_id": True},
        {"is_remote_frame": True},
        {"is_fd": True},
        {"arbitration_id": 0x2A1},
        {"arbitration_id": 0x150},
        {"arbitration_id": 0x15A},
    ):
        message = SimpleNamespace(**{**vars(valid), **update})
        assert not PiperCANBridgeDagger.should_forward(message)


@pytest.mark.parametrize("previously_enabled", [False, True])
def test_can_bridge_enable_drops_an_old_frame_already_taken_out_of_receive_queue(previously_enabled):
    received = queue.Queue()
    sent = queue.Queue()
    old_frame_waiting = threading.Event()
    release_old_frame = threading.Event()
    errors = []

    class PausedForwardLock:
        """Pause exactly between recv() and the worker's send decision."""

        def __init__(self):
            self.lock = threading.Lock()
            self.pause_once = True

        def __enter__(self):
            if threading.current_thread().name == "PiperDaggerCANBridge" and self.pause_once:
                self.pause_once = False
                old_frame_waiting.set()
                if not release_old_frame.wait(timeout=2):
                    errors.append("worker was not released")
            return self.lock.__enter__()

        def __exit__(self, *args):
            return self.lock.__exit__(*args)

    def recv(timeout):
        try:
            return received.get(timeout=timeout)
        except queue.Empty:
            return None

    def frame(value):
        return SimpleNamespace(arbitration_id=0x155, data=bytes([value]) * 8, is_rx=True)

    buses = {
        "can_master": SimpleNamespace(recv=recv),
        "can_follower": SimpleNamespace(send=lambda message, timeout: sent.put(message)),
    }
    bridge = PiperCANBridgeDagger(
        "can_master",
        "can_follower",
        can_module=SimpleNamespace(Message=SimpleNamespace),
        bus_factory=lambda **kwargs: buses[kwargs["channel"]],
    )
    bridge._forward_lock = PausedForwardLock()
    bridge.start()
    try:
        if previously_enabled:
            bridge.enable()
        received.put(frame(1))
        assert old_frame_waiting.wait(timeout=2)
        if previously_enabled:
            bridge.disable()
        # One stale frame is held by the worker, another is still in the bus.
        received.put(frame(2))
        bridge.enable()
        received.put(frame(3))
        release_old_frame.set()
        outgoing = sent.get(timeout=2)
        assert bytes(outgoing.data) == bytes([3]) * 8
        assert sent.empty()
        assert not errors
    finally:
        release_old_frame.set()
        bridge.close()


def test_can_bridge_disable_is_a_hard_boundary_for_inflight_send():
    class FakeMessage:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.is_rx = kwargs.get("is_rx", True)
            self.is_error_frame = kwargs.get("is_error_frame", False)
            self.is_extended_id = kwargs.get("is_extended_id", False)
            self.is_remote_frame = kwargs.get("is_remote_frame", False)
            self.is_fd = kwargs.get("is_fd", False)

    class ReceiveBus:
        def __init__(self):
            self.messages = queue.Queue()

        def recv(self, timeout):
            try:
                return self.messages.get(timeout=timeout)
            except queue.Empty:
                return None

        def shutdown(self):
            pass

    class SendBus:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.sent = []

        def send(self, message, timeout):
            del timeout
            self.entered.set()
            assert self.release.wait(timeout=1.0)
            self.sent.append(message)

        def shutdown(self):
            pass

    receive_bus = ReceiveBus()
    send_bus = SendBus()
    bus_by_channel = {"can_master": receive_bus, "can_follower": send_bus}
    can_module = SimpleNamespace(Message=FakeMessage)
    bridge = PiperCANBridgeDagger(
        "can_master",
        "can_follower",
        can_module=can_module,
        bus_factory=lambda **kwargs: bus_by_channel[kwargs["channel"]],
    )
    bridge.start()
    bridge.enable()
    receive_bus.messages.put(FakeMessage(arbitration_id=0x155, data=b"\x01" * 8, is_rx=True))
    assert send_bus.entered.wait(timeout=1.0)

    disabled = threading.Event()

    def disable():
        bridge.disable()
        disabled.set()

    worker = threading.Thread(target=disable)
    worker.start()
    time.sleep(0.02)
    assert not disabled.is_set()
    send_bus.release.set()
    worker.join(timeout=1.0)
    assert disabled.is_set()
    assert len(send_bus.sent) == 1

    receive_bus.messages.put(FakeMessage(arbitration_id=0x155, data=b"\x02" * 8, is_rx=True))
    time.sleep(0.03)
    assert len(send_bus.sent) == 1
    bridge.close()


@pytest.mark.parametrize("raw_width_um", [0, 12_000, 43_300, 70_000])
def test_can_bridge_forwards_absolute_gripper_width_without_offset(raw_width_um):
    class FakeMessage:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.is_rx = kwargs.get("is_rx", True)
            self.is_error_frame = kwargs.get("is_error_frame", False)
            self.is_extended_id = kwargs.get("is_extended_id", False)
            self.is_remote_frame = kwargs.get("is_remote_frame", False)
            self.is_fd = kwargs.get("is_fd", False)

    class ReceiveBus:
        def __init__(self):
            self.messages = queue.Queue()

        def recv(self, timeout):
            try:
                return self.messages.get(timeout=timeout)
            except queue.Empty:
                return None

        def shutdown(self):
            pass

    class SendBus:
        def __init__(self):
            self.sent = []
            self.sent_event = threading.Event()

        def send(self, message, timeout):
            del timeout
            self.sent.append(message)
            self.sent_event.set()

        def shutdown(self):
            pass

    receive_bus = ReceiveBus()
    send_bus = SendBus()
    bus_by_channel = {"can_master": receive_bus, "can_follower": send_bus}
    bridge = PiperCANBridgeDagger(
        "can_master",
        "can_follower",
        can_module=SimpleNamespace(Message=FakeMessage),
        bus_factory=lambda **kwargs: bus_by_channel[kwargs["channel"]],
    )
    bridge.start()
    bridge.enable()
    payload = bytearray(raw_width_um.to_bytes(4, "big", signed=True) + b"\x00\x32\x01\x00")
    receive_bus.messages.put(
        FakeMessage(
            arbitration_id=0x159,
            data=payload,
            is_rx=True,
        )
    )

    assert send_bus.sent_event.wait(timeout=1.0)
    forwarded_width_um = int.from_bytes(send_bus.sent[0].data[:4], "big", signed=True)
    assert forwarded_width_um == raw_width_um
    assert bytes(send_bus.sent[0].data) == bytes(payload)
    bridge.close()


def test_can_bridge_waits_for_fresh_feedback_on_master_channel():
    messages = queue.Queue()
    opened_channels: list[str] = []
    shutdown = threading.Event()

    class MonitorBus:
        def recv(self, timeout):
            try:
                return messages.get(timeout=timeout)
            except queue.Empty:
                return None

        def shutdown(self):
            shutdown.set()

    def bus_factory(**kwargs):
        opened_channels.append(kwargs["channel"])
        return MonitorBus()

    bridge = PiperCANBridgeDagger(
        "can_master",
        "can_follower",
        can_module=SimpleNamespace(),
        bus_factory=bus_factory,
    )
    messages.put(
        SimpleNamespace(
            arbitration_id=0x2A5,
            is_rx=True,
            is_error_frame=False,
        )
    )

    assert bridge.wait_for_master_feedback(0.1) == 0x2A5
    assert opened_channels == ["can_master"]
    assert shutdown.is_set()


def test_can_bridge_opens_monitor_before_mode_switch_and_waits_for_complete_expert_sample():
    messages = queue.Queue()
    events: list[str] = []

    class MonitorBus:
        def recv(self, timeout):
            try:
                return messages.get(timeout=timeout)
            except queue.Empty:
                return None

        def shutdown(self):
            events.append("monitor:shutdown")

    def bus_factory(**kwargs):
        assert kwargs["channel"] == "can_master"
        assert [entry["can_id"] for entry in kwargs["can_filters"]] == list(PIPER_EXPERT_CONTROL_IDS)
        events.append("monitor:open")
        return MonitorBus()

    bridge = PiperCANBridgeDagger(
        "can_master",
        "can_follower",
        can_module=SimpleNamespace(),
        bus_factory=bus_factory,
    )

    def switch_mode():
        events.append("master:leader")
        for arbitration_id in (0x159, 0x155, 0x157, 0x156):
            messages.put(
                SimpleNamespace(
                    arbitration_id=arbitration_id,
                    is_rx=True,
                    is_error_frame=False,
                    is_extended_id=False,
                    data=bytes(8),
                )
            )

    assert bridge.wait_for_master_alignment(
        _pose(0.0),
        PiperDaggerHardwareConfig(handover_alignment_timeout_s=0.1),
        trigger=switch_mode,
    ) == _pose(0.0)
    assert events == ["monitor:open", "master:leader", "monitor:shutdown"]


def test_can_bridge_reports_missing_expert_control_frames():
    messages = queue.Queue()

    class MonitorBus:
        def recv(self, timeout):
            try:
                return messages.get(timeout=timeout)
            except queue.Empty:
                return None

        def shutdown(self):
            pass

    bridge = PiperCANBridgeDagger(
        "can_master",
        "can_follower",
        receive_timeout_s=0.001,
        can_module=SimpleNamespace(),
        bus_factory=lambda **_kwargs: MonitorBus(),
    )
    for arbitration_id in (0x155, 0x156, 0x157):
        messages.put(
            SimpleNamespace(
                arbitration_id=arbitration_id,
                is_rx=True,
                is_error_frame=False,
                is_extended_id=False,
                data=bytes(8),
            )
        )

    with pytest.raises(PiperDaggerHardwareError, match=r"missing 0x159"):
        bridge.wait_for_master_alignment(
            _pose(0.0),
            PiperDaggerHardwareConfig(handover_alignment_timeout_s=0.01),
            trigger=lambda: None,
        )


def _pose_monitor_bridge(messages, *, channel="can_master", frame_ids=PIPER_EXPERT_CONTROL_IDS):
    class Monitor:
        def recv(self, timeout):
            try:
                return messages.get(timeout=timeout)
            except queue.Empty:
                return None

        def shutdown(self):
            pass

    def factory(**kwargs):
        assert kwargs["channel"] == channel
        assert [entry["can_id"] for entry in kwargs["can_filters"]] == list(frame_ids)
        return Monitor()

    return PiperCANBridgeDagger(
        "can_master",
        "can_follower",
        receive_timeout_s=0.001,
        can_module=SimpleNamespace(),
        bus_factory=factory,
    )


def _put_pose(messages, frame_ids, *, gripper_um=43_300, is_rx=True):
    for arbitration_id in frame_ids:
        # +1 degree, -2 degrees per pair exercises signed endian/radian decoding.
        payload = (1000).to_bytes(4, "big", signed=True) + (-2000).to_bytes(4, "big", signed=True)
        if arbitration_id == frame_ids[-1]:
            payload = gripper_um.to_bytes(4, "big", signed=True) + bytes(4)
        messages.put(SimpleNamespace(arbitration_id=arbitration_id, is_rx=is_rx, data=payload))


def _monitor_pose():
    return dict(zip(PIPER_ACTION_KEYS, [math.radians(1), math.radians(-2)] * 3 + [0.0433], strict=True))


@pytest.mark.parametrize("channel", ["can_master", "can_follower"])
def test_fresh_pose_requires_all_feedback_frames_and_decodes_physical_units(channel):
    messages = queue.Queue()
    bridge = _pose_monitor_bridge(messages, channel=channel, frame_ids=PIPER_POSE_FEEDBACK_IDS)
    _put_pose(messages, PIPER_POSE_FEEDBACK_IDS, gripper_um=0, is_rx=False)
    _put_pose(messages, PIPER_POSE_FEEDBACK_IDS)
    read = bridge.read_master_feedback_pose if channel == "can_master" else bridge.read_follower_feedback_pose
    assert read(0.05) == pytest.approx(_monitor_pose())


def test_takeover_monitor_rejects_local_echo_even_if_it_matches_the_reference():
    messages = queue.Queue()
    bridge = _pose_monitor_bridge(messages)

    def trigger():
        _put_pose(messages, PIPER_EXPERT_CONTROL_IDS, is_rx=False)
        _put_pose(messages, PIPER_EXPERT_CONTROL_IDS, gripper_um=0)

    with pytest.raises(PiperHandoverAlignmentError, match="gripper_error=0.0433 m"):
        bridge.wait_for_master_alignment(
            _monitor_pose(),
            PiperDaggerHardwareConfig(handover_alignment_timeout_s=0.02, handover_gripper_settle_s=0.005),
            trigger=trigger,
        )


def test_takeover_monitor_does_not_accept_brief_alignment_before_physical_closure():
    messages = queue.Queue()
    bridge = _pose_monitor_bridge(messages)

    def trigger():
        _put_pose(messages, PIPER_EXPERT_CONTROL_IDS)
        messages.put(SimpleNamespace(arbitration_id=0x159, is_rx=True, data=bytes(8)))

    with pytest.raises(PiperHandoverAlignmentError, match="last_leader_gripper=0.0000 m"):
        bridge.wait_for_master_alignment(
            _monitor_pose(),
            PiperDaggerHardwareConfig(handover_alignment_timeout_s=0.02, handover_gripper_settle_s=0.005),
            trigger=trigger,
        )


@pytest.mark.parametrize("status,zero", [(0x04, 0), (0x00, 1)])
def test_takeover_monitor_rejects_angle_mode_and_zeroing_commands(status, zero):
    messages = queue.Queue()
    bridge = _pose_monitor_bridge(messages)

    def trigger():
        data = bytes(6) + bytes([status, zero])
        messages.put(SimpleNamespace(arbitration_id=0x159, is_rx=True, data=data))

    with pytest.raises(PiperDaggerHardwareError, match="width-mode gripper frames without zeroing"):
        bridge.wait_for_master_alignment(_monitor_pose(), PiperDaggerHardwareConfig(), trigger=trigger)


def test_incomplete_physical_master_feedback_cannot_pass_alignment():
    messages = queue.Queue()
    bridge = _pose_monitor_bridge(messages, frame_ids=PIPER_POSE_FEEDBACK_IDS)
    messages.put(SimpleNamespace(arbitration_id=0x2A8, is_rx=True, data=bytes(8)))
    with pytest.raises(PiperDaggerHardwareError, match="missing 0x2A5, 0x2A6, 0x2A7"):
        bridge.read_master_feedback_pose(0.01)


def test_controller_mirrors_one_filtered_mit_command_and_orders_handover():
    events: list[str] = []
    follower_bus = _FakeBus("follower", events)
    master_bus = _FakeBus("master", events)
    bridge = _FakeBridge(events)
    robot = _FakePiperRobot(follower_bus)
    controller = PiperDaggerController(
        robot,
        PiperDaggerHardwareConfig(
            require_safe_firmware=True, feedback_timeout_s=0.25, handover_gripper_settle_s=0.0
        ),
        master_bus=master_bus,
        bridge=bridge,
    )

    follower_bus.writes.clear()
    master_bus.writes.clear()
    events.clear()
    performed = controller.send_same_action(_pose(2.0))
    assert performed == _pose(1.0)
    assert follower_bus.writes == [[1.0] * 7]
    assert master_bus.writes == [[1.0] * 7]
    assert events == ["follower:write", "master:write"]

    controller.pause_computer_output()
    events.clear()
    controller.enter_correction()
    assert events == [
        "bridge:disable",
        "bridge:drain",
        "master:follower",
        "bridge:follower_pose",
        "follower:write",
        "master:write",
        "bridge:master_pose",
        "bridge:master_alignment",
        "master:leader",
        "bridge:drain",
        "bridge:enable",
    ]
    assert controller.mode is PiperControlMode.CORRECTING
    with pytest.raises(PiperDaggerHardwareError, match="computer command rejected"):
        controller.send_same_action(_pose(3.0))

    follower_bus.pose = _pose(4.0)
    events.clear()
    corrected = controller.leave_correction()
    assert corrected == _pose(4.0)
    assert events == [
        "bridge:disable",
        "bridge:drain",
        "master:follower",
        "bridge:master_feedback:0.25",
        "bridge:feedback:0.25",
        "follower:write",
        "master:write",
    ]
    assert controller.mode is PiperControlMode.PAUSED
    controller.close()


def test_prepare_autonomy_reasserts_master_follower_mode_and_waits_for_fresh_feedback():
    events: list[str] = []
    follower_bus = _FakeBus("follower", events)
    master_bus = _FakeBus("master", events)
    bridge = _FakeBridge(events)
    robot = _FakePiperRobot(follower_bus)
    controller = PiperDaggerController(
        robot,
        PiperDaggerHardwareConfig(
            require_safe_firmware=True, feedback_timeout_s=0.25, handover_gripper_settle_s=0.0
        ),
        master_bus=master_bus,
        bridge=bridge,
    )

    events.clear()
    controller.prepare_autonomy()
    assert events == [
        "bridge:disable",
        "master:follower",
        "bridge:master_feedback:0.25",
        "bridge:feedback:0.25",
        "follower:write",
        "master:write",
    ]
    assert controller.mode is PiperControlMode.SHADOW

    events.clear()
    controller.pause_computer_output()
    controller.prepare_autonomy()
    assert events == [
        "bridge:disable",
        "master:follower",
        "bridge:follower_pose",
        "follower:write",
        "master:write",
        "bridge:master_pose",
        "bridge:disable",
        "master:follower",
        "bridge:master_feedback:0.25",
        "bridge:feedback:0.25",
        "follower:write",
        "master:write",
    ]
    controller.close()


@pytest.mark.parametrize("follower_first", [True, False])
def test_optional_follower_gains_apply_only_to_autonomy_and_never_to_master_or_handover(follower_first):
    events = []
    follower_bus = _FakeBus("follower", events)
    master_bus = _FakeBus("master", events)
    robot = _FakePiperRobot(follower_bus)
    robot.config.joint_control_mode = "mit"
    tuned_commands = []

    def tuned_write(values):
        tuned_commands.append(list(values))
        events.append("follower:mit")

    robot.write_joint_command = tuned_write
    controller = PiperDaggerController(
        robot,
        PiperDaggerHardwareConfig(
            follower_first=follower_first, feedback_timeout_s=0.25, handover_gripper_settle_s=0.0
        ),
        master_bus=master_bus,
        bridge=_FakeBridge(events),
    )
    assert tuned_commands == []  # Startup holds keep the native JS route.
    events.clear()
    controller.send_same_action(_pose(0.08))
    assert tuned_commands == [[0.04] * 7]  # Shared EMA runs exactly once.
    assert events == (["follower:mit", "master:write"] if follower_first else ["master:write", "follower:mit"])

    controller.pause_computer_output()
    controller.send_same_action(_pose(0.0))  # Paused reset commands use JS.
    controller.enter_correction()
    with pytest.raises(PiperDaggerHardwareError, match="computer command rejected"):
        controller.send_same_action(_pose(0.08))
    controller.leave_correction()
    assert len(tuned_commands) == 1

    controller.prepare_autonomy()
    controller.send_same_action(_pose(0.08))
    assert len(tuned_commands) == 2
    controller.close()


def test_controller_close_returns_master_home_before_disconnect():
    events: list[str] = []
    controller = PiperDaggerController(
        _FakePiperRobot(_FakeBus("follower", events)),
        PiperDaggerHardwareConfig(require_safe_firmware=True),
        master_bus=_FakeBus("master", events),
        bridge=_FakeBridge(events),
    )
    events.clear()

    controller.close()

    assert events == [
        "bridge:disable",
        "bridge:close",
        "master:follower",
        "bridge:master_feedback:1.0",
        "master:home_as_follower",
        "master:leader",
        "master:disconnect",
    ]
    assert controller.mode is PiperControlMode.CLOSED

    # Shutdown is idempotent: a second call cannot command another home move.
    controller.close()
    assert events.count("master:home_as_follower") == 1


def test_controller_close_restores_leader_mode_when_home_feedback_fails():
    events: list[str] = []

    class FeedbackFailingBridge(_FakeBridge):
        def wait_for_master_feedback(self, timeout_s):
            self.events.append(f"bridge:master_feedback:{timeout_s}")
            raise PiperDaggerHardwareError("no fresh master feedback")

    controller = PiperDaggerController(
        _FakePiperRobot(_FakeBus("follower", events)),
        PiperDaggerHardwareConfig(require_safe_firmware=True),
        master_bus=_FakeBus("master", events),
        bridge=FeedbackFailingBridge(events),
    )
    events.clear()

    with pytest.raises(PiperDaggerHardwareError, match="no fresh master feedback"):
        controller.close()

    assert events == [
        "bridge:disable",
        "bridge:close",
        "master:follower",
        "bridge:master_feedback:1.0",
        "master:leader",
        "master:disconnect",
    ]
    assert controller.mode is PiperControlMode.CLOSED


def test_controller_retries_transient_empty_leader_cache():
    events: list[str] = []

    class DelayedLeaderBus(_FakeBus):
        def __init__(self):
            super().__init__("master", events, pose=_pose(7.0))
            self.read_attempts = 0

        def try_read_leader(self):
            self.read_attempts += 1
            if self.read_attempts < 3:
                return None
            return dict(self.pose)

    follower_bus = _FakeBus("follower", events)
    master_bus = DelayedLeaderBus()
    controller = PiperDaggerController(
        _FakePiperRobot(follower_bus),
        PiperDaggerHardwareConfig(require_safe_firmware=True, feedback_timeout_s=0.1),
        master_bus=master_bus,
        bridge=_FakeBridge(events),
    )

    assert controller.read_expert_action() == _pose(7.0)
    assert master_bus.read_attempts == 3
    controller.close()


def test_pause_aligns_master_gripper_to_actual_follower_opening_before_takeover():
    events: list[str] = []

    class TrackingMasterBus(_FakeBus):
        def write_js(self, values):
            super().write_js(values)
            self.pose = dict(zip(PIPER_ACTION_KEYS, values, strict=True))

    follower_pose = _pose(0.0)
    follower_pose["gripper.pos"] = 0.04
    follower_bus = _FakeBus("follower", events, pose=follower_pose)
    master_bus = TrackingMasterBus("master", events, pose=follower_pose)
    controller = PiperDaggerController(
        _FakePiperRobot(follower_bus),
        PiperDaggerHardwareConfig(
            require_safe_firmware=True,
            handover_alignment_timeout_s=0.1,
            handover_gripper_tolerance=0.001,
            handover_gripper_settle_s=0.0,
        ),
        master_bus=master_bus,
        bridge=_FakeBridge(events, master_bus=master_bus, follower_bus=follower_bus),
    )

    # Simulate autonomous grasping: both arms received the same close command,
    # but the follower stopped on an object while the unloaded master closed.
    master_bus.pose["gripper.pos"] = 0.0
    follower_bus.writes.clear()
    master_bus.writes.clear()

    controller.pause_computer_output()

    assert controller.mode is PiperControlMode.PAUSED
    assert master_bus.pose["gripper.pos"] == pytest.approx(0.04)
    assert follower_bus.writes[0][-1] == pytest.approx(0.04)
    assert all(write[-1] == pytest.approx(0.04) for write in master_bus.writes)
    controller.close()


def _make_physical_handover(*, close_on_switches=0, close_by=0.0433, fail_restore=False):
    events: list[str] = []
    follower_pose = {**_pose(0.0), "gripper.pos": 0.0433}

    class PhysicalMasterBus(_FakeBus):
        leader_switches = 0
        is_leader = False

        def change_to_leader_mode(self):
            super().change_to_leader_mode()
            self.leader_switches += 1
            self.is_leader = True
            if self.leader_switches <= close_on_switches:
                self.pose["gripper.pos"] -= close_by

        def change_to_follower_mode(self):
            super().change_to_follower_mode()
            self.is_leader = False

        def write_js(self, values):
            # No unsupported attempt to drive the master while it is a leader.
            assert not self.is_leader
            super().write_js(values)
            if not (fail_restore and self.leader_switches):
                self.pose = dict(zip(PIPER_ACTION_KEYS, values, strict=True))

    follower_bus = _FakeBus("follower", events, pose=follower_pose)
    master_bus = PhysicalMasterBus("master", events, pose=follower_pose)
    controller = PiperDaggerController(
        _FakePiperRobot(follower_bus),
        PiperDaggerHardwareConfig(
            handover_alignment_timeout_s=0.03,
            handover_gripper_settle_s=0.0,
        ),
        master_bus=master_bus,
        bridge=_FakeBridge(events, master_bus=master_bus, follower_bus=follower_bus),
    )
    controller._mode = PiperControlMode.PAUSED
    events.clear()
    follower_bus.writes.clear()
    master_bus.writes.clear()
    return controller, master_bus, follower_bus, events


def test_takeover_realigns_physically_at_keypress_and_preserves_absolute_expert_action():
    controller, master, follower, events = _make_physical_handover()
    # Pendant drifted during a long pause after the previous alignment.
    master.pose["gripper.pos"] = 0.0
    controller.enter_correction()

    assert controller.mode is PiperControlMode.CORRECTING
    assert master.pose == follower.pose
    assert controller.read_expert_action() == master.pose
    assert events.index("master:write") < events.index("master:leader") < events.index("bridge:enable")
    controller.close()


def test_transient_physical_closure_gets_one_automatic_realignment_before_retry():
    controller, master, follower, events = _make_physical_handover(close_on_switches=1)
    controller.enter_correction()

    assert controller.mode is PiperControlMode.CORRECTING
    assert events.count("master:leader") == 2
    assert events.count("bridge:enable") == 1
    assert events[-1] == "bridge:enable"
    assert master.pose == follower.pose
    assert controller.read_expert_action()["gripper.pos"] == pytest.approx(0.0433)
    assert all(write[-1] == pytest.approx(0.0433) for write in master.writes + follower.writes)
    controller.close()


@pytest.mark.parametrize("close_by", [0.0045, 0.0433])
def test_persistent_physical_closure_is_not_hidden_and_master_is_restored(close_by):
    controller, master, follower, events = _make_physical_handover(close_on_switches=2, close_by=close_by)
    with pytest.raises(PiperHandoverAlignmentError, match=f"gripper_error={close_by:.4f} m"):
        controller.enter_correction()

    assert controller.mode is PiperControlMode.PAUSED
    assert events.count("master:leader") == 2
    assert "bridge:enable" not in events
    assert not master.is_leader
    assert master.pose == follower.pose
    assert follower.pose["gripper.pos"] == pytest.approx(0.0433)
    controller.close()


def test_failed_physical_restore_is_not_reported_as_recoverable_alignment_rejection():
    controller, master, _, events = _make_physical_handover(close_on_switches=2, fail_restore=True)
    with pytest.raises(PiperDaggerHardwareError, match="could not restore master alignment") as caught:
        controller.enter_correction()
    assert not isinstance(caught.value, PiperHandoverAlignmentError)
    assert "bridge:enable" not in events
    assert not master.is_leader
    controller.close()


def test_controller_rejects_mismatched_master_follower_firmware():
    events: list[str] = []
    robot = _FakePiperRobot(_FakeBus("follower", events, firmware="S-V1.8-9"))
    with pytest.raises(PiperDaggerHardwareError, match="firmware must match"):
        PiperDaggerController(
            robot,
            PiperDaggerHardwareConfig(),
            master_bus=_FakeBus("master", events, firmware="S-V1.9-0"),
            bridge=_FakeBridge(events),
        )


class _FakeDataset:
    def __init__(self, features):
        self.meta = SimpleNamespace(features=features)
        self.frames = []
        self.saved = []
        self.parallel_encoding = []
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def has_pending_frames(self):
        return bool(self.frames)

    def save_episode(self, *, parallel_encoding=True):
        self.parallel_encoding.append(parallel_encoding)
        self.saved.append(list(self.frames))
        self.frames.clear()

    def clear_episode_buffer(self):
        self.frames.clear()

    def finalize(self):
        self.finalized = True

    def push_to_hub(self, **kwargs):
        raise AssertionError(kwargs)


def test_correction_recorder_owns_recycled_images_before_reading_expert():
    robot = _FakePiperRobot(_FakeBus("follower", [], _pose(1.0)))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    robot.observation_features = {**robot.observation_features, "camera": image.shape}
    robot.get_observation = lambda: {**robot.bus.pose, "camera": image}

    def expert():
        image[:] = 255
        return _pose(9.0)

    config = DaggerDatasetConfig(use_videos=False)
    template = DaggerCorrectionRecorder(robot, expert, config)
    dataset = _FakeDataset(template._dataset_features())
    recorder = DaggerCorrectionRecorder(robot, expert, config, dataset=dataset)
    recorder.begin("recover the cube")
    recorder._record_one()
    recorder.end(save=True)
    recorder.close()

    frame = dataset.saved[0][0]
    assert not frame["observation.images.camera"].any()
    assert not np.shares_memory(frame["observation.images.camera"], image)
    np.testing.assert_array_equal(frame["action"], np.full(7, 9.0, dtype=np.float32))


@pytest.mark.parametrize("stall", ["none", "capture", "sleep"])
def test_correction_recorder_does_not_catch_up_after_stall(monkeypatch, stall):
    config = DaggerDatasetConfig(fps=30)
    recorder = DaggerCorrectionRecorder(_FakePiperRobot(_FakeBus("follower", [])), lambda: _pose(0), config)
    now = 0.0
    captures = []
    sleeps = []

    def capture():
        nonlocal now
        captures.append(now)
        if stall == "capture" and len(captures) == 1:
            now += 0.2
        if len(captures) == 8:
            recorder._shutdown.set()

    def sleep(duration):
        nonlocal now
        assert duration >= 0
        now += duration
        if stall == "sleep" and not sleeps:
            now += 0.2
        sleeps.append(duration)

    monkeypatch.setattr(dagger_module, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(dagger_module, "precise_sleep", sleep)
    monkeypatch.setattr(recorder, "_record_one", capture)
    recorder._active.set()
    recorder._run()

    assert len(captures) == 8
    assert np.all(np.diff(captures) >= 1 / config.fps - 1e-12)
    if stall == "none":
        np.testing.assert_allclose(captures, np.arange(8) / config.fps, atol=1e-12)


@pytest.mark.parametrize(
    "mismatch",
    [
        "fps",
        "missing",
        "extra",
        "action_shape",
        "action_dtype",
        "action_names",
        "state_names",
        "camera_shape",
        "camera_dtype",
        "intervention_dtype",
    ],
)
def test_correction_recorder_rejects_incompatible_metadata_before_resume(tmp_path, monkeypatch, mismatch):
    pytest.importorskip("datasets")
    from lerobot.datasets import dataset_metadata, lerobot_dataset

    robot = _FakePiperRobot(_FakeBus("follower", []))
    robot.observation_features = {**robot.observation_features, "camera": (4, 4, 3)}
    template = DaggerCorrectionRecorder(robot, lambda: _pose(0), DaggerDatasetConfig(use_videos=False))
    features = deepcopy({**template._dataset_features(), **DEFAULT_FEATURES})
    meta = SimpleNamespace(fps=30, features=features)
    if mismatch == "fps":
        meta.fps = 15
    elif mismatch == "missing":
        features.pop("intervention")
    elif mismatch == "extra":
        features["reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    elif mismatch == "action_shape":
        features["action"]["shape"] = (8,)
    elif mismatch == "action_dtype":
        features["action"]["dtype"] = "float64"
    elif mismatch == "action_names":
        features["action"]["names"] = list(reversed(features["action"]["names"]))
    elif mismatch == "state_names":
        features["observation.state"]["names"] = list(reversed(features["observation.state"]["names"]))
    elif mismatch == "camera_shape":
        features["observation.images.camera"]["shape"] = (8, 8, 3)
    elif mismatch == "camera_dtype":
        features["observation.images.camera"]["dtype"] = "video"
    elif mismatch == "intervention_dtype":
        features["intervention"]["dtype"] = "int64"

    monkeypatch.setattr(dataset_metadata, "LeRobotDatasetMetadata", lambda *args, **kwargs: meta)
    opened = []

    def resume(*args, **kwargs):
        opened.append(kwargs)
        return _FakeDataset(features)

    monkeypatch.setattr(lerobot_dataset.LeRobotDataset, "resume", resume)
    config = DaggerDatasetConfig(repo_id="test/remote_dagger", root=tmp_path, use_videos=False, resume=True)
    with pytest.raises(ValueError):
        DaggerCorrectionRecorder(robot, lambda: _pose(0), config)
    assert not opened


def test_correction_recorder_accepts_matching_metadata_and_list_shapes(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    from lerobot.datasets import dataset_metadata, lerobot_dataset

    robot = _FakePiperRobot(_FakeBus("follower", []))
    template = DaggerCorrectionRecorder(robot, lambda: _pose(0), DaggerDatasetConfig(use_videos=False))
    features = deepcopy({**template._dataset_features(), **DEFAULT_FEATURES})
    for feature in features.values():
        feature["shape"] = list(feature["shape"])
    meta = SimpleNamespace(fps=30, features=features)
    dataset = _FakeDataset(features)
    monkeypatch.setattr(dataset_metadata, "LeRobotDatasetMetadata", lambda *args, **kwargs: meta)
    opened = []

    def resume(*args, **kwargs):
        opened.append((args, kwargs))
        return dataset

    monkeypatch.setattr(lerobot_dataset.LeRobotDataset, "resume", resume)
    config = DaggerDatasetConfig(repo_id="test/remote_dagger", root=tmp_path, use_videos=False, resume=True)
    recorder = DaggerCorrectionRecorder(robot, lambda: _pose(0), config)
    assert recorder.dataset is dataset
    assert opened == [
        (
            (config.repo_id,),
            {"root": tmp_path, "image_writer_processes": 0, "image_writer_threads": 0},
        )
    ]
    recorder.close()


def test_correction_recorder_can_save_finalize_and_resume_real_dataset(tmp_path):
    pytest.importorskip("datasets")
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    robot = _FakePiperRobot(_FakeBus("follower", [], _pose(1.0)))
    config = DaggerDatasetConfig(
        repo_id="test/remote_dagger", root=tmp_path / "corrections", use_videos=False
    )
    recorder = DaggerCorrectionRecorder(robot, lambda: _pose(2.0), config)
    recorder.begin("recover the cube")
    for _ in range(4):
        recorder._record_one()
    recorder.end(save=True)
    recorder.close()

    resumed = DaggerCorrectionRecorder(robot, lambda: _pose(3.0), replace(config, resume=True))
    resumed.begin("recover the cube")
    for _ in range(4):
        resumed._record_one()
    resumed.end(save=True)
    resumed.close()

    meta = LeRobotDatasetMetadata(config.repo_id, root=config.root)
    assert meta.fps == 30
    assert meta.total_episodes == 2
    assert meta.total_frames == 8
    assert meta.features["action"]["names"] == list(PIPER_ACTION_KEYS)
    assert meta.features["intervention"]["dtype"] == "bool"


@pytest.mark.parametrize("save_failed_window", [False, True])
def test_capture_failure_is_published_before_capture_barrier_releases(save_failed_window):
    robot = _FakePiperRobot(_FakeBus("follower", [], _pose(1.0)))
    config = DaggerDatasetConfig(use_videos=False)
    template = DaggerCorrectionRecorder(robot, lambda: _pose(2.0), config)
    dataset = _FakeDataset(template._dataset_features())
    recorder = DaggerCorrectionRecorder(robot, lambda: _pose(2.0), config, dataset=dataset)
    recorder.begin("old correction")
    recorder._record_one()
    failed = threading.Event()
    continue_error = threading.Event()
    inner_lock = recorder._sample_lock

    class PreemptAfterFailedSample:
        def __enter__(self):
            return inner_lock.__enter__()

        def __exit__(self, error_type, error, traceback):
            inner_lock.__exit__(error_type, error, traceback)
            if error_type is not None and threading.current_thread().name == "PiperDaggerRecorder":
                failed.set()
                assert continue_error.wait(3.0)

    def failed_expert():
        raise OSError("old expert stream failure")

    recorder._sample_lock = PreemptAfterFailedSample()
    recorder.expert_action_reader = failed_expert
    recorder.start()
    try:
        assert failed.wait(1.0)
        recorder.pause_capture()
        recorder.finish_window(save=save_failed_window)
        if save_failed_window:
            assert recorder.saved_episodes == 0
            assert not dataset.saved
            assert not dataset.frames
            return
        recorder.expert_action_reader = lambda: _pose(3.0)
        recorder.begin("new correction")
        continue_error.set()
        deadline = time.monotonic() + 2.0
        while recorder.last_error is None and recorder.frames_in_window == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert recorder.last_error is None
        assert recorder.active
        assert dataset.frames
        assert all(frame["task"] == "new correction" for frame in dataset.frames)
    finally:
        continue_error.set()
        recorder.close()


def test_correction_recorder_saves_multiple_windows_with_expert_labels():
    events: list[str] = []
    follower_bus = _FakeBus("follower", events, _pose(1.0))
    robot = _FakePiperRobot(follower_bus)
    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    dataset = _FakeDataset(features)
    recorder = DaggerCorrectionRecorder(
        robot,
        lambda: _pose(9.0),
        DaggerDatasetConfig(repo_id="user/rollout_test", use_videos=False),
        dataset=dataset,
    )

    recorder.begin("recover the cube")
    recorder._record_one()
    recorder.pause_capture()
    assert recorder.finish_window(save=True) == 1
    assert recorder.saved_episodes == 1
    assert len(dataset.saved) == 1
    assert dataset.parallel_encoding == [False]
    frame = dataset.saved[0][0]
    np.testing.assert_array_equal(frame["observation.state"], np.ones(7, dtype=np.float32))
    np.testing.assert_array_equal(frame["action"], np.full(7, 9.0, dtype=np.float32))
    assert frame["task"] == "recover the cube"
    assert frame["intervention"].item()

    follower_bus.pose = _pose(2.0)
    recorder.begin("place the cube")
    recorder._record_one()
    recorder.pause_capture()
    assert recorder.finish_window(save=True) == 1
    assert recorder.saved_episodes == 2
    assert len(dataset.saved) == 2
    second_frame = dataset.saved[1][0]
    np.testing.assert_array_equal(second_frame["observation.state"], np.full(7, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(second_frame["action"], np.full(7, 9.0, dtype=np.float32))
    assert second_frame["task"] == "place the cube"


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_on_close", [False, True])
def test_recorder_discards_failed_save_and_always_finalizes(monkeypatch, error_type, fail_on_close):
    robot = _FakePiperRobot(_FakeBus("follower", []))
    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    dataset = _FakeDataset(features)
    # push_to_hub would raise in the fake: a failed save must never publish.
    recorder = DaggerCorrectionRecorder(
        robot,
        lambda: _pose(9.0),
        DaggerDatasetConfig(repo_id="user/rollout_test", use_videos=False, push_to_hub=True),
        dataset=dataset,
    )
    error = error_type("injected save failure")
    calls = []

    def fail_save(**kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(dataset, "save_episode", fail_save)
    recorder.begin("recover the cube")
    recorder._record_one()
    if fail_on_close:
        with pytest.raises(error_type) as caught:
            recorder.close(save_partial=True)
    else:
        recorder.pause_capture()
        with pytest.raises(error_type) as caught:
            recorder.finish_window(save=True)
        recorder.close(save_partial=True)
    assert caught.value is error
    assert recorder.last_error is error
    assert recorder.saved_episodes == 0
    assert recorder.frames_in_window == 0
    assert not recorder.active
    assert recorder._shutdown.is_set()
    assert not dataset.has_pending_frames()
    assert dataset.saved == []
    assert calls == [{"parallel_encoding": False}]
    assert dataset.finalized
