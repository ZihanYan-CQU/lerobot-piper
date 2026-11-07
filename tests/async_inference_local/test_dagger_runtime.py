import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_piper_dagger import PIPER_ACTION_KEYS, _FakeDataset, _pose
from test_runtime import wait_for

from lerobot.async_inference_local.control.dagger_io import DaggerRobotIO
from lerobot.async_inference_local.control.piper_dagger import PiperControlMode, PiperHandoverAlignmentError
from lerobot.async_inference_local.recording.correction import DaggerCorrectionRecorder, DaggerDatasetConfig
from lerobot.async_inference_local.runtime.dagger import DaggerPhase, LocalDagger
from lerobot.async_inference_local.runtime.deployment import State
from lerobot.async_inference_local.shared.configs import InferenceConfig
from lerobot.async_inference_local.shared.dagger_configs import LocalDaggerConfig
from lerobot.async_inference_local.shared.types import ActionChunk, Prediction
from lerobot.light import NullStatusIndicator
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features


class Robot:
    name = "piper_follower"
    action_features = dict.fromkeys(PIPER_ACTION_KEYS, float)
    observation_features = action_features

    def __init__(self):
        self.pose = _pose(0.0)
        self.cameras = {}
        self.closed = threading.Event()
        self.events = []
        self._is_connected = False
        self.bus = SimpleNamespace(
            read_follower=lambda: dict(self.pose),
            read_action_positions=lambda p: p,
            piper=SimpleNamespace(electronic_emergency_stop=self.emergency_stop),
        )

    def connect(self):
        self._is_connected = True

    def disconnect(self):
        self.events.append("follower:home")
        self._is_connected = False
        self.closed.set()

    def emergency_stop(self):
        self.events.append("follower:stop")
        self.closed.set()

    def get_observation(self):
        assert not self.closed.is_set()
        return dict(self.pose)


class Controller:
    def __init__(self, robot, config):
        self.robot = robot
        self.mode = PiperControlMode.SHADOW
        self.enabled = False
        self.expert = _pose(0.8)
        self.master_stuck = False
        self.feedback_calls = 0
        self.feedback_error = None
        self.health_error = None
        self.alignment_error = None
        self.commands = []
        self.bridge = SimpleNamespace(disable=self.disable)

    def disable(self):
        self.enabled = False

    def seed_command(self, pose):
        self.seed = dict(pose)

    def prepare_autonomy(self):
        assert self.mode != PiperControlMode.CORRECTING
        self.mode = PiperControlMode.SHADOW
        return dict(self.robot.pose)

    def pause_computer_output(self):
        self.mode = PiperControlMode.PAUSED

    def enter_correction(self):
        assert self.mode == PiperControlMode.PAUSED
        if self.alignment_error:
            raise self.alignment_error
        self.mode = PiperControlMode.CORRECTING
        self.enabled = True

    def leave_correction(self):
        assert self.mode == PiperControlMode.CORRECTING
        self.disable()
        self.mode = PiperControlMode.PAUSED
        return dict(self.robot.pose)

    def abort_to_paused(self):
        assert not self.robot.closed.is_set()
        self.disable()
        self.mode = PiperControlMode.PAUSED
        return dict(self.robot.pose)

    def read_expert_action(self):
        assert not self.robot.closed.is_set()
        assert self.mode == PiperControlMode.CORRECTING
        return dict(self.expert)

    def assert_correction_healthy(self):
        if self.health_error:
            raise self.health_error
        assert self.enabled

    def send_same_action(self, pose):
        assert self.mode != PiperControlMode.CORRECTING
        assert not self.robot.closed.is_set()
        self.commands.append(dict(pose))
        self.robot.pose = dict(pose)
        return dict(pose)

    def send_planned_action(self, pose):
        return self.send_same_action(pose)

    def read_reset_feedback(self, timeout_s):
        assert timeout_s > 0
        self.feedback_calls += 1
        if self.feedback_error:
            raise self.feedback_error
        master = dict(self.robot.pose)
        if self.master_stuck:
            master["joint_1.pos"] += 0.1
        return {"master": master, "follower": dict(self.robot.pose)}

    def close(self):
        self.disable()
        self.mode = PiperControlMode.CLOSED
        self.robot.events.append("master:home")

    def emergency_close(self):
        self.disable()
        self.mode = PiperControlMode.CLOSED
        self.robot.events.append("master:stop")


class Engine:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.error = None
        self.calls = []
        self.owner_threads = []

    def warmup(self, *args):
        self.owner_threads.append(threading.current_thread().name)
        return []

    def reset(self):
        self.owner_threads.append(threading.current_thread().name)

    def predict(self, request, latency_s):
        self.owner_threads.append(threading.current_thread().name)
        self.calls.append(request)
        self.entered.set()
        assert self.release.wait(5)
        if self.error:
            raise self.error
        values = torch.full((20, 7), 0.5)
        values[:, -1] = 0.02
        return Prediction(
            ActionChunk(values, values, values.numpy(), request.observation.source_at), 4, 0.001
        )


def config(**overrides):
    values = {
        "robot": SimpleNamespace(type="piper_follower"),
        "pretrained_name_or_path": "unused",
        "task": "pick cube",
        "inference": InferenceConfig(device="cpu", compile_model=False, warmup_runs=0),
        "dataset": DaggerDatasetConfig(repo_id="user/corrections", use_videos=False, fps=60),
        "input_device": "prompt",
        "actions_per_chunk": 20,
        "observation_capture_fps": 120,
        "observation_max_age_s": 0.2,
        "startup_timeout_s": 2,
        "inference_timeout_s": 3,
        "stop_timeout_s": 0.1,
        "reset_duration_s": 0.01,
        "reset_fps": 200,
        "reset_settle_s": 0.01,
        "reset_settle_timeout_s": 0.15,
    }
    values.update(overrides)
    return LocalDaggerConfig(**values)


def setup(*, start=True, recorder_factory=None, **overrides):
    cfg = config(**overrides)
    robot, engine = Robot(), Engine()
    io = DaggerRobotIO(robot, cfg.hardware, controller_factory=Controller)
    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    dataset = _FakeDataset(features)
    runtime = LocalDagger(
        cfg,
        robot,
        engine_factory=lambda *a: engine,
        robot_io=io,
        recorder_factory=recorder_factory
        or (lambda *a, **kw: DaggerCorrectionRecorder(*a, dataset=dataset, **kw)),
        status_indicator=NullStatusIndicator(),
    )
    if start:
        runtime.start()
    return runtime, robot, engine, dataset


def correct(runtime):
    runtime.resume()
    runtime.pause()
    runtime.start_correction()
    wait_for(lambda: runtime.recorder.frames_in_window > 0)


def test_full_collection_records_only_raw_expert_corrections_and_resets_both_arms():
    runtime, robot, engine, dataset = setup()
    try:
        runtime.resume()
        wait_for(lambda: len(runtime.io.controller.commands) > 3)
        assert not dataset.frames and not dataset.saved
        runtime.pause()
        old_count = len(runtime.io.controller.commands)
        runtime.start_correction()
        wait_for(lambda: runtime.recorder.frames_in_window >= 2)
        assert len(runtime.io.controller.commands) == old_count
        runtime.finish_correction()
        assert runtime.state == State.READY and runtime.phase == DaggerPhase.PAUSED
        assert not runtime.io.controller.enabled
        assert runtime.io.controller.feedback_calls >= 2
        assert robot.pose["gripper.pos"] == pytest.approx(0.07)
        assert len(dataset.saved) == 1
        for frame in dataset.saved[0]:
            np.testing.assert_allclose(frame["action"], np.full(7, 0.8))
            assert frame["intervention"].item() and frame["task"] == "pick cube"
            assert frame["observation.state"][-1] < 0.07
        assert dataset.parallel_encoding == [False]
        assert set(engine.owner_threads) == {"LocalInference"}
    finally:
        runtime.stop()
    assert dataset.finalized
    assert robot.events == ["master:home", "follower:home"]


def test_late_inference_is_fenced_by_pause_and_human_takeover():
    runtime, _, engine, dataset = setup()
    try:
        engine.release.clear()
        runtime.resume()
        assert engine.entered.wait(1)
        runtime.pause()
        with pytest.raises(RuntimeError, match="READY"):
            runtime.resume()
        runtime.start_correction()
        engine.release.set()
        wait_for(lambda: runtime._pending is None)
        assert runtime.schedule.remaining == 0
        assert runtime.io.controller.commands == []
        wait_for(lambda: runtime.recorder.frames_in_window > 0)
        runtime.finish_correction()
        assert len(dataset.saved) == 1
    finally:
        engine.release.set()
        runtime.stop()


def test_target_counts_this_run_and_finishes_without_an_extra_reset():
    runtime, robot, _, dataset = setup(num_episodes=1)
    correct(runtime)
    count = len(runtime.io.controller.commands)
    runtime.finish_correction()
    assert runtime.stop_event.is_set()
    assert len(runtime.io.controller.commands) == count
    assert runtime.io.controller.feedback_calls == 0
    assert runtime.recorder.saved_episodes == 1
    runtime.stop()
    runtime.stop()
    assert len(dataset.saved) == 1
    assert robot.events == ["master:home", "follower:home"]


def test_discard_and_reset_do_not_count_a_correction():
    runtime, _, _, dataset = setup(num_episodes=1)
    try:
        correct(runtime)
        runtime.reset()
        assert runtime.state == State.READY
        assert not dataset.frames and not dataset.saved
        assert runtime.recorder.saved_episodes == 0
        assert not runtime.stop_event.is_set()
        correct(runtime)
        runtime.finish_correction()
        assert runtime.stop_event.is_set()
    finally:
        runtime.stop()


@pytest.mark.parametrize("save_partial", [False, True])
@pytest.mark.parametrize("fault", [False, True])
def test_stop_partial_policy_and_dual_arm_fault_cleanup(save_partial, fault):
    runtime, robot, _, dataset = setup(save_partial_on_stop=save_partial)
    correct(runtime)
    if fault:
        runtime._fault("inference", RuntimeError("GPU failure"))
    runtime.stop()
    assert len(dataset.saved) == int(save_partial and not fault)
    assert not dataset.frames
    assert dataset.finalized
    assert robot.events == (["master:stop", "follower:stop"] if fault else ["master:home", "follower:home"])


def test_alignment_rejection_leaves_pause_retryable_without_recording():
    runtime, _, _, dataset = setup()
    try:
        runtime.resume()
        runtime.pause()
        runtime.io.controller.alignment_error = PiperHandoverAlignmentError("unaligned")
        with pytest.raises(PiperHandoverAlignmentError):
            runtime.start_correction()
        assert runtime.phase == DaggerPhase.PAUSED and runtime.failure is None
        assert not runtime.recorder.active and not dataset.frames
        runtime.io.controller.alignment_error = None
        runtime.start_correction()
        assert runtime.phase == DaggerPhase.CORRECTING
    finally:
        runtime.stop()


def test_failed_save_requires_explicit_reset_and_is_never_retried(monkeypatch):
    runtime, _, _, dataset = setup(save_partial_on_stop=True)
    calls = []

    def fail_save(**kw):
        calls.append(kw)
        raise OSError("disk full")

    monkeypatch.setattr(dataset, "save_episode", fail_save)
    try:
        correct(runtime)
        with pytest.raises(OSError, match="disk full"):
            runtime.finish_correction()
        assert runtime.phase == DaggerPhase.FAULT
        assert runtime.io.controller.feedback_calls == 0
        assert runtime.failure is None
        with pytest.raises(RuntimeError):
            runtime.resume()
        with pytest.raises(RuntimeError):
            runtime.start_correction()
        runtime.reset()
        assert runtime.state == State.READY and runtime.status()["dagger_error"] is None
    finally:
        runtime.stop()
    assert len(calls) == 1 and runtime.recorder.saved_episodes == 0
    assert dataset.finalized


@pytest.mark.parametrize("source", ["recording", "bridge"])
def test_correction_watchdog_closes_gate_and_requires_recovery(monkeypatch, source):
    runtime, _, _, dataset = setup()
    try:
        correct(runtime)
        if source == "recording":

            def fail_frame(frame):
                raise OSError("image write rejected")

            monkeypatch.setattr(dataset, "add_frame", fail_frame)
        else:
            runtime.io.controller.health_error = RuntimeError("CAN bridge failed")
        wait_for(lambda: runtime.phase == DaggerPhase.FAULT)
        assert not runtime.recorder.active
        assert not runtime.io.controller.enabled
        assert runtime.failure is None
        runtime.reset()
        assert runtime.state == State.READY
        assert not dataset.frames and not dataset.saved
    finally:
        runtime.stop()


def test_stuck_master_prevents_gripper_release_and_reset_can_be_retried():
    runtime, robot, _, _ = setup()
    try:
        correct(runtime)
        robot.pose["gripper.pos"] = 0.0167
        runtime.io.controller.master_stuck = True
        with pytest.raises(TimeoutError, match="master.joint_1.pos"):
            runtime.reset()
        assert runtime.phase == DaggerPhase.FAULT and runtime.state == State.PAUSED
        assert not runtime.stop_event.is_set()
        assert all(
            command["gripper.pos"] == pytest.approx(0.0167) for command in runtime.io.controller.commands
        )
        runtime.io.controller.master_stuck = False
        runtime.reset()
        assert runtime.state == State.READY
        assert robot.pose["gripper.pos"] == pytest.approx(0.07)
    finally:
        runtime.stop()


def test_stop_closes_hardware_before_blocked_video_save_finishes(monkeypatch):
    runtime, robot, _, dataset = setup()
    entered, release = threading.Event(), threading.Event()
    save = dataset.save_episode
    errors = []

    def blocked_save(**kw):
        entered.set()
        assert release.wait(4)
        save(**kw)

    def stop():
        try:
            runtime.stop()
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(dataset, "save_episode", blocked_save)
    correct(runtime)
    assert runtime.submit_operator_event("advance")
    assert entered.wait(1)
    assert runtime.status()["saving"]
    assert not runtime.submit_operator_event("advance")
    assert not runtime.submit_operator_event("reset")
    assert runtime.submit_operator_event("stop")
    worker = threading.Thread(target=stop, daemon=True)
    worker.start()
    try:
        assert robot.closed.wait(0.5), "Hardware cleanup must not wait for encoding"
        assert worker.is_alive()
        assert runtime.state == State.STOPPED
        assert runtime.io.controller.feedback_calls == 0
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert dataset.finalized and len(dataset.saved) == 1
    assert runtime.io.controller.feedback_calls == 0


def test_repeated_advance_during_pause_cannot_enter_correction(monkeypatch):
    runtime, _, _, _ = setup()
    entered, release = threading.Event(), threading.Event()
    pause = runtime.io.controller.pause_computer_output

    def blocked_pause():
        entered.set()
        assert release.wait(3)
        pause()

    monkeypatch.setattr(runtime.io.controller, "pause_computer_output", blocked_pause)
    try:
        runtime.resume()
        assert runtime.submit_operator_event("advance")
        assert entered.wait(1)
        for _ in range(20):
            assert not runtime.submit_operator_event("advance")
        release.set()
        wait_for(lambda: not runtime._operator_pending)
        assert runtime.state == State.PAUSED and runtime.phase == DaggerPhase.PAUSED
        assert not runtime.recorder.active
    finally:
        release.set()
        runtime.stop()


def test_recording_initialization_failure_closes_both_arms_without_homing():
    def fail(*a, **kw):
        raise OSError("dataset root exists")

    runtime, robot, _, _ = setup(start=False, recorder_factory=fail)
    with pytest.raises(OSError, match="dataset root"):
        runtime.start()
    assert robot.events == ["master:stop", "follower:stop"]
    assert runtime.state == State.STOPPED


def test_task_changes_do_not_relabel_an_active_correction():
    runtime, _, _, dataset = setup()
    try:
        runtime.set_task("new task")
        correct(runtime)
        with pytest.raises(RuntimeError, match="READY"):
            runtime.set_task("wrong task")
        runtime.finish_correction()
        assert all(frame["task"] == "new task" for frame in dataset.saved[0])
    finally:
        runtime.stop()


def test_autostart_waits_until_recorder_and_watchdog_exist():
    runtime, _, _, _ = setup(autostart=True)
    try:
        assert runtime.state == State.RUNNING
        assert runtime.recorder is not None
        assert runtime._watchdog_thread is not None and runtime._watchdog_thread.is_alive()
        runtime.pause()
        runtime.start_correction()
        wait_for(lambda: runtime.recorder.frames_in_window > 0)
    finally:
        runtime.stop()


def test_queued_event_cannot_replay_after_an_external_phase_change(monkeypatch):
    runtime, _, _, _ = setup(start=False)
    entered, release = threading.Event(), threading.Event()
    original = runtime._operator_loop

    def blocked_worker():
        entered.set()
        assert release.wait(3)
        original()

    monkeypatch.setattr(runtime, "_operator_loop", blocked_worker)
    runtime.start()
    try:
        assert entered.wait(1)
        assert runtime.submit_operator_event("advance")
        runtime.resume()
        runtime.pause()
        release.set()
        wait_for(lambda: not runtime._operator_pending)
        assert runtime.phase == DaggerPhase.PAUSED
        assert not runtime.recorder.active
    finally:
        release.set()
        runtime.stop()


def test_stop_requested_mid_handover_closes_any_late_opened_gate(monkeypatch):
    runtime, robot, _, _ = setup()
    entered, release = threading.Event(), threading.Event()
    original = runtime.io.controller.enter_correction

    def blocked_enter():
        entered.set()
        assert release.wait(3)
        original()

    monkeypatch.setattr(runtime.io.controller, "enter_correction", blocked_enter)
    runtime.resume()
    runtime.pause()
    assert runtime.submit_operator_event("advance")
    assert entered.wait(1)
    runtime.request_stop()
    release.set()
    wait_for(lambda: not runtime._operator_pending)
    assert not runtime.io.controller.enabled
    assert not runtime.recorder.active
    runtime.stop()
    assert robot.closed.is_set()


def test_stop_during_reset_prevents_later_hardware_commands():
    runtime, robot, _, dataset = setup(reset_duration_s=2)
    correct(runtime)
    assert runtime.submit_operator_event("reset")
    wait_for(lambda: runtime.state == State.RESETTING)
    runtime.stop()
    assert robot.closed.is_set() and dataset.finalized
    runtime._operator_thread.join(1)
    assert not runtime._operator_thread.is_alive()
    assert runtime.state == State.STOPPED
    assert not dataset.saved
