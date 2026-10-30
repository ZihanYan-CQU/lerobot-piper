import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.async_inference_local.control.robot_io import RobotIO


def test_piper_camera_wait_does_not_hold_motor_lock_and_snapshot_owns_image():
    entered, release = threading.Event(), threading.Event()
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)

    class Camera:
        def read_latest_with_timestamp(self, **kwargs):
            entered.set()
            assert release.wait(2)
            return pixels, time.perf_counter()

    robot = SimpleNamespace(
        name="piper_follower",
        action_features={"joint.pos": float},
        cameras={"front": Camera()},
        bus=SimpleNamespace(read_follower=lambda: None, read_action_positions=lambda _: {"joint.pos": 0}),
        send_action=lambda action: action,
    )
    io = RobotIO(robot)
    captures = []
    thread = threading.Thread(target=lambda: captures.append(io.capture(1)))
    thread.start()
    assert entered.wait(1)
    try:
        np.testing.assert_array_equal(io.send(np.array([2], dtype=np.float32)), [2])
    finally:
        release.set()
        thread.join(2)
    pixels[:] = 99
    assert captures[0][0]["front"].max() == 0
    assert not captures[0][0]["front"].flags.writeable


def test_faulted_piper_disconnect_latches_stop_without_homing_or_reset():
    events = []
    robot = SimpleNamespace(
        name="piper_follower",
        action_features={"joint.pos": float},
        cameras={"front": SimpleNamespace(is_connected=True, disconnect=lambda: events.append("camera"))},
        bus=SimpleNamespace(piper=SimpleNamespace(electronic_emergency_stop=lambda: events.append("stop"))),
        disconnect=lambda: events.append("home"),
    )
    RobotIO(robot).disconnect(faulted=True)
    assert events == ["stop", "camera"]


def piper(cameras, read=None):
    return SimpleNamespace(
        name="piper_follower",
        action_features={"joint.pos": float},
        cameras=cameras,
        bus=SimpleNamespace(
            read_follower=lambda: None,
            read_action_positions=read or (lambda _: {"joint.pos": 0}),
        ),
        send_action=lambda action: action,
    )


def test_legacy_camera_does_not_disable_other_cameras_timestamps():
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    timestamp = time.perf_counter() - 0.25

    def unsupported(**kwargs):
        raise NotImplementedError

    legacy = SimpleNamespace(read_latest_with_timestamp=unsupported, async_read=lambda: pixels)
    modern = SimpleNamespace(
        read_latest_with_timestamp=lambda **kwargs: (pixels, timestamp), async_read=lambda: pixels
    )
    io = RobotIO(piper({"legacy": legacy, "modern": modern}))
    for _ in range(2):
        _, _, source = io.capture(1)
        assert time.monotonic() - source >= 0.24


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), time.perf_counter() + 1000])
def test_invalid_camera_timestamp_cannot_be_redated_as_fresh(timestamp):
    camera = SimpleNamespace(
        read_latest_with_timestamp=lambda **kwargs: (np.zeros((4, 4, 3), dtype=np.uint8), timestamp)
    )
    with pytest.raises(ValueError, match="timestamp"):
        RobotIO(piper({"front": camera})).capture(1)


def test_camera_frame_is_owned_before_waiting_for_next_hardware_read():
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    camera = SimpleNamespace(read_latest_with_timestamp=lambda **kwargs: (pixels, time.perf_counter()))

    def read_pose(_):
        pixels[:] = 99  # Simulate an SDK buffer recycled during a subsequent I/O call.
        return {"joint.pos": 0}

    values, _, _ = RobotIO(piper({"front": camera}, read_pose)).capture(1)
    assert values["front"].max() == 0
    assert not values["front"].flags.writeable


def test_command_expiring_while_waiting_for_motor_lock_is_not_sent(monkeypatch):
    from lerobot.async_inference_local.control import robot_io

    clock = [10.0]
    monkeypatch.setattr(robot_io.time, "monotonic", lambda: clock[0])
    sent, errors = [], []
    robot = piper({})
    robot.send_action = lambda action: sent.append(action) or action
    io = RobotIO(robot)
    entered = threading.Event()

    def dispatch():
        entered.set()
        try:
            io.send(np.array([2], dtype=np.float32), deadline=10.1)
        except Exception as exc:
            errors.append(exc)

    with io._motor_lock:
        sender = threading.Thread(target=dispatch)
        sender.start()
        assert entered.wait(1)
        clock[0] = 10.2
    sender.join(1)
    assert not sent
    assert len(errors) == 1 and isinstance(errors[0], TimeoutError)


def test_fault_cleanup_attempts_all_cameras_even_if_one_fails():
    events = []

    def fail_close():
        events.append("failed camera")
        raise OSError("camera cleanup failed")

    robot = piper(
        {
            "first": SimpleNamespace(is_connected=True, disconnect=fail_close),
            "second": SimpleNamespace(is_connected=True, disconnect=lambda: events.append("second camera")),
        }
    )
    robot.bus.piper = SimpleNamespace(electronic_emergency_stop=lambda: events.append("stop"))
    robot._is_connected = True
    robot._ema_cmd = {"joint.pos": 2}
    with pytest.raises(ExceptionGroup, match="Piper"):
        RobotIO(robot).disconnect(faulted=True)
    assert events == ["stop", "failed camera", "second camera"]
    assert not robot._is_connected
    assert robot._ema_cmd is None
