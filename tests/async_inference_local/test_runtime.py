import logging
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference_local.runtime import deployment as runtime_module
from lerobot.async_inference_local.runtime.deployment import LocalDeployment, State
from lerobot.async_inference_local.shared.configs import InferenceConfig, LocalDeploymentConfig
from lerobot.async_inference_local.shared.types import ActionChunk, Prediction
from lerobot.policies.rtc import RTCConfig


class FakeRobot:
    name = "fake"
    observation_features = {"joint.pos": float}
    action_features = {"joint.pos": float}

    def __init__(self):
        self.is_connected = False
        self.position = 0.0
        self.sent = []
        self.closed = 0

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False
        self.closed += 1

    def get_observation(self):
        return {"joint.pos": self.position}

    def send_action(self, action):
        self.position = action["joint.pos"]
        self.sent.append((time.monotonic(), self.position))
        return action


class FakeEngine:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = []
        self.resets = []
        self.error = None

    def reset(self):
        self.resets.append(threading.current_thread().name)

    def warmup(self, observation, cancelled):
        return [0.001]

    def predict(self, request, latency_s):
        self.calls.append(request)
        self.entered.set()
        self.release.wait(3)
        if self.error:
            raise self.error
        value = 10.0 if request.task == "old" else 20.0
        tensor = torch.full((20, 1), value)
        return Prediction(
            ActionChunk(tensor, tensor, tensor.numpy(), request.observation.source_at), 4, 0.001
        )


def configuration(**overrides):
    values = {
        "robot": SimpleNamespace(),
        "pretrained_name_or_path": "unused",
        "task": "old",
        "inference": InferenceConfig(device="cpu", compile_model=False, warmup_runs=0),
        "actions_per_chunk": 20,
        "fps": 30,
        "interpolation_multiplier": 2,
        "observation_capture_fps": 120,
        "observation_max_age_s": 0.2,
        "action_max_age_s": 2,
        "startup_timeout_s": 2,
        "inference_timeout_s": 0.5,
        "stop_timeout_s": 1,
        "reset_duration_s": 0.02,
        "reset_fps": 100,
        "reset_settle_timeout_s": 0.1,
        "reset_settle_s": 0.0,
    }
    values.update(overrides)
    return LocalDeploymentConfig(**values)


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Condition did not become true")
        time.sleep(0.002)


def setup(**overrides):
    robot, engine = FakeRobot(), FakeEngine()
    runtime = LocalDeployment(configuration(**overrides), robot, engine_factory=lambda *args: engine)
    runtime.start()
    return runtime, robot, engine


def test_direct_deployment_runs_without_transport_and_only_worker_touches_policy():
    runtime, robot, engine = setup()
    try:
        assert runtime.state == State.READY
        assert not robot.sent
        runtime.resume()
        wait_for(lambda: len(robot.sent) >= 5)
        assert set(engine.resets) == {"LocalInference"}
        assert runtime.status()["accepted_chunks"] >= 2
        runtime.pause()
        count = len(robot.sent)
        time.sleep(0.06)
        assert len(robot.sent) == count
        assert runtime.failure is None
    finally:
        runtime.stop()
    assert robot.closed == 1


def test_pause_and_retarget_discard_a_result_already_inside_inference():
    runtime, robot, engine = setup()
    try:
        engine.release.clear()
        runtime.resume()
        assert engine.entered.wait(1)
        runtime.pause()
        runtime.set_task("new")
        runtime.resume()
        engine.release.set()
        wait_for(lambda: len(robot.sent) >= 4)
        assert engine.calls[0].task == "old"
        assert all(request.task == "new" for request in engine.calls[1:])
        # The first endpoint belongs to the new policy (20), not the late old result (10).
        assert robot.sent[1][1] == 20
        assert runtime.failure is None
    finally:
        engine.release.set()
        runtime.stop()


def test_reset_cancels_old_result_and_restores_initial_pose():
    runtime, robot, engine = setup()
    try:
        robot.position = 3
        engine.release.clear()
        runtime.resume()
        assert engine.entered.wait(1)
        runtime.reset()
        assert runtime.state == State.READY
        assert robot.position == pytest.approx(0)
        count = len(robot.sent)
        engine.release.set()
        time.sleep(0.06)
        assert len(robot.sent) == count
        assert runtime.schedule.remaining == 0
    finally:
        engine.release.set()
        runtime.stop()


def test_watchdog_stops_motion_while_inference_is_blocked():
    runtime, robot, engine = setup(inference_timeout_s=0.08)
    try:
        engine.release.clear()
        runtime.resume()
        assert engine.entered.wait(1)
        wait_for(lambda: runtime.failure is not None)
        assert "deadline" in runtime.failure
        assert not robot.sent
        engine.release.set()
        time.sleep(0.03)
        assert not robot.sent
    finally:
        engine.release.set()
        runtime.stop()


def test_worker_exception_wakes_owner_and_prevents_further_commands():
    runtime, robot, engine = setup()
    try:
        engine.error = ValueError("bad model output")
        runtime.resume()
        wait_for(lambda: runtime.failure is not None)
        assert "bad model output" in runtime.failure
        assert runtime.stop_event.is_set()
        assert not robot.sent
        with pytest.raises(RuntimeError):
            runtime.resume()
    finally:
        runtime.stop()


@pytest.mark.parametrize("worker", ["inference", "control"])
def test_worker_fault_revokes_motion_before_a_blocked_log_handler(worker):
    runtime, robot, engine = setup(stop_timeout_s=0.05)
    entered, release = threading.Event(), threading.Event()

    class BlockingHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR and record.threadName == f"Local{worker.title()}":
                entered.set()
                release.wait(2)

    handler = BlockingHandler()
    runtime_module.logger.addHandler(handler)
    try:
        runtime.resume()
        wait_for(lambda: len(robot.sent) >= 2)
        if worker == "inference":
            engine.release.clear()
            engine.entered.clear()
            assert engine.entered.wait(1)
            engine.error = ValueError("bad model output after an accepted chunk")
            engine.release.set()
        else:

            def fail_send(_):
                raise OSError("motor send failed")

            robot.send_action = fail_send
        assert entered.wait(1)
        assert runtime.stop_event.is_set(), "Motor revocation must precede log output"
        assert runtime.state == State.FAULT
        count = len(robot.sent)
        time.sleep(0.06)
        assert len(robot.sent) == count
        # A logging sink cannot retain the lifecycle or command lock.
        if worker == "control":
            release.set()  # Hardware workers must terminate before disconnect.
        runtime.stop()
        assert robot.closed == 1
    finally:
        release.set()
        engine.release.set()
        runtime_module.logger.removeHandler(handler)
        runtime.stop()


def test_stop_cancels_inflight_prediction_and_is_idempotent():
    runtime, robot, engine = setup()
    engine.release.clear()
    runtime.resume()
    assert engine.entered.wait(1)
    engine.release.set()
    runtime.stop()
    count = len(robot.sent)
    time.sleep(0.03)
    assert len(robot.sent) == count
    runtime.stop()
    assert robot.closed == 1


def test_stop_during_reset_does_not_resume_or_read_disconnected_hardware():
    runtime, robot, _ = setup(reset_duration_s=0.2)
    errors = []
    original_read = robot.get_observation

    def checked_read():
        assert robot.is_connected
        return original_read()

    robot.get_observation = checked_read
    robot.position = 3

    def reset():
        try:
            runtime.reset()
        except Exception as exc:
            errors.append(exc)

    resetter = threading.Thread(target=reset)
    resetter.start()
    wait_for(lambda: len(robot.sent) >= 1)
    runtime.stop()
    resetter.join(1)
    assert not resetter.is_alive()
    assert not errors
    assert runtime.state == State.STOPPED
    assert not robot.is_connected


def test_stale_result_is_not_admitted_and_empty_buffer_does_not_replay_actions():
    runtime, robot, engine = setup(action_max_age_s=0.21)
    try:
        engine.release.clear()
        runtime.resume()
        assert engine.entered.wait(1)
        time.sleep(0.23)
        # Fence all subsequent captures so the deliberately old result cannot be
        # hidden by an immediate fresh result in this test.
        with runtime._condition:
            runtime._valid_after = time.monotonic() + 10
            runtime._latest = None
        engine.release.set()
        wait_for(lambda: runtime.status()["rejected_chunks"] > 0)
        assert not robot.sent
    finally:
        engine.release.set()
        runtime.stop()


def test_startup_warmup_failure_disconnects_without_starting_control():
    robot, engine = FakeRobot(), FakeEngine()

    def fail_warmup(*args):
        raise ValueError("warmup failed")

    engine.warmup = fail_warmup
    runtime = LocalDeployment(configuration(), robot, engine_factory=lambda *args: engine)
    with pytest.raises(RuntimeError, match="warmup failed"):
        runtime.start()
    assert not robot.sent
    assert robot.closed == 1
    runtime.stop()
    assert robot.closed == 1


def test_capture_keeps_latest_observation_without_queuing_parallel_inferences():
    runtime, _, engine = setup()
    try:
        engine.release.clear()
        runtime.resume()
        assert engine.entered.wait(1)
        first_sequence = engine.calls[0].observation.sequence
        wait_for(lambda: runtime._sequence >= first_sequence + 4)
        latest_sequence = runtime._sequence
        assert len(engine.calls) == 1
        engine.release.set()
        wait_for(lambda: len(engine.calls) >= 2)
        assert engine.calls[1].observation.sequence >= latest_sequence
        assert runtime.failure is None
    finally:
        engine.release.set()
        runtime.stop()


def test_new_generation_waits_for_new_camera_frame_not_just_new_capture_call():
    runtime, robot, engine = setup()
    source = [time.monotonic()]
    captured = []

    def capture(max_age):
        captured.append(time.monotonic())
        return {"joint.pos": robot.position}, captured[-1], source[0]

    runtime.io.capture = capture
    try:
        runtime.resume()
        wait_for(lambda: len(captured) >= 3)
        assert not engine.calls
        source[0] = time.monotonic()
        wait_for(lambda: len(robot.sent) >= 1)
        assert engine.calls[0].observation.source_at >= runtime._valid_after
    finally:
        runtime.stop()


def test_stop_waits_for_connection_before_disconnecting_hardware():
    robot, engine = FakeRobot(), FakeEngine()
    entered, release, stopped, closing = (threading.Event() for _ in range(4))
    errors = []

    def connect():
        entered.set()
        assert release.wait(2)
        robot.is_connected = True

    original_close = robot.disconnect

    def disconnect():
        closing.set()
        original_close()

    robot.connect, robot.disconnect = connect, disconnect
    runtime = LocalDeployment(configuration(), robot, engine_factory=lambda *args: engine)

    def start():
        try:
            runtime.start()
        except RuntimeError as exc:
            errors.append(exc)

    def stop():
        stopped.set()
        runtime.stop()

    starter = threading.Thread(target=start)
    stopper = threading.Thread(target=stop)
    starter.start()
    assert entered.wait(1)
    stopper.start()
    assert stopped.wait(1)
    try:
        assert not closing.wait(0.05), "Disconnected while connect() was still initializing the robot"
    finally:
        release.set()
        starter.join(2)
        stopper.join(2)
    assert not starter.is_alive() and not stopper.is_alive()
    assert not robot.is_connected
    assert robot.closed == 1
    assert not robot.sent


def test_hardware_cleanup_does_not_wait_for_cancelled_gpu_call():
    runtime, robot, engine = setup(stop_timeout_s=0.5)
    engine.release.clear()
    runtime.resume()
    assert engine.entered.wait(1)
    stopper = threading.Thread(target=runtime.stop)
    stopper.start()
    try:
        wait_for(lambda: robot.closed == 1, timeout=0.2)
        assert not engine.release.is_set()
        assert not robot.sent
    finally:
        engine.release.set()
        stopper.join(2)


def test_guided_request_keeps_full_prefix_independently_of_guidance_mask_horizon():
    runtime, _, engine = setup(rtc=RTCConfig(execution_horizon=2))
    try:
        runtime.resume()
        wait_for(lambda: len(engine.calls) >= 2)
        assert len(engine.calls[1].prefix) > 2
    finally:
        runtime.stop()


def test_concurrent_stop_callers_wait_for_single_disconnect():
    runtime, robot, _ = setup()
    entered, release, returned = (threading.Event() for _ in range(3))
    original_close = robot.disconnect

    def disconnect():
        entered.set()
        assert release.wait(2)
        original_close()

    def stop_again():
        runtime.stop()
        returned.set()

    robot.disconnect = disconnect
    first, second = threading.Thread(target=runtime.stop), threading.Thread(target=stop_again)
    first.start()
    assert entered.wait(1)
    second.start()
    try:
        assert not returned.wait(0.05)
    finally:
        release.set()
        first.join(2)
        second.join(2)
    assert robot.closed == 1
    assert returned.is_set()
