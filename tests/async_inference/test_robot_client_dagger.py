# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Generation fencing and phase tests for the consolidated DAgger client."""

import logging
import sys
import threading
import time

import pytest
import torch

import lerobot.async_inference.client.robot_client_dagger as dagger_module
from lerobot.async_inference.client.piper_hil_dagger import (
    PiperDaggerHardwareError,
    PiperHandoverAlignmentError,
)
from lerobot.async_inference.client.robot_client_dagger import (
    DAggerPhase,
    RobotClientDagger,
    RobotClientDaggerConfig,
    _collection_progress_postfix,
    _refresh_collection_progress,
    _render_status,
)
from lerobot.async_inference.client.robot_client_deployment import RobotClientDeployment, RolloutState
from lerobot.async_inference.shared.helpers import TimedAction
from lerobot.async_inference.shared.rtc_remote import RemoteRTCActionChunk, encode_rollout_observation_id
from lerobot.light import NullStatusIndicator, StatusCue
from tests.async_inference.test_robot_client_deployment import (
    _assert_entrypoint_cleans_up_while_input_blocks,
)
from tests.mocks.mock_robot import MockRobotConfig


class _FakeController:
    def __init__(self):
        self.calls = []
        self.pose = {f"motor_{index}.pos": 0.0 for index in range(1, 4)}

    def prepare_autonomy(self):
        self.calls.append("prepare")
        return dict(self.pose)

    def pause_computer_output(self):
        self.calls.append("pause")

    def enter_correction(self):
        self.calls.append("enter_correction")

    def leave_correction(self):
        self.calls.append("leave_correction")
        self.pose = dict.fromkeys(self.pose, 4.0)
        return dict(self.pose)

    def abort_to_paused(self):
        self.calls.append("abort")
        return dict(self.pose)

    def send_same_action(self, action):
        performed = {key: float(value) for key, value in action.items()}
        self.pose = dict(performed)
        self.calls.append(("send", performed))
        return performed

    def read_expert_action(self):
        return dict(self.pose)

    def read_reset_feedback(self, timeout_s):
        return {"master": dict(self.pose), "follower": dict(self.pose)}

    def close(self):
        self.calls.append("close")


class _FakeRecorder:
    enabled = True

    def __init__(self):
        self.calls = []
        self.frames_in_window = 0
        self.saved_episodes = 0
        self.last_error = None
        self.pending = False

    def start(self):
        self.calls.append("start")

    def begin(self, task):
        self.calls.append(("begin", task))
        self.last_error = None
        self.frames_in_window = 3
        self.pending = True

    def pause_capture(self):
        self.calls.append("pause_capture")

    def finish_window(self, *, save):
        self.calls.append(("finish", save))
        frames = self.frames_in_window
        if save and self.pending and frames > 0:
            self.saved_episodes += 1
        self.pending = False
        self.frames_in_window = 0
        return frames

    def close(self, *, save_partial=False):
        self.calls.append(("close", save_partial))


class _FakeIndicator:
    def __init__(self):
        self.cues = []
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def show(self, cue):
        self.cues.append(cue)

    def close(self):
        self.closed = True


def _make_client(tmp_path, *, dagger_num_episodes=0, status_indicator=None):
    controller = _FakeController()
    recorder = _FakeRecorder()
    if status_indicator is None:
        status_indicator = NullStatusIndicator()
    client = RobotClientDagger(
        RobotClientDaggerConfig(
            policy_type="smolvla",
            pretrained_name_or_path="checkpoint",
            robot=MockRobotConfig(
                calibration_dir=tmp_path,
                random_values=False,
                static_values=[0.0, 0.0, 0.0],
            ),
            actions_per_chunk=8,
            server_address="localhost:9999",
            interpolation_multiplier=2,
            reset_duration_s=0,
            reset_settle_s=0,
            task="pick the cube",
            dagger_require_safe_firmware=False,
            dagger_num_episodes=dagger_num_episodes,
        ),
        dagger_controller=controller,
        correction_recorder=recorder,
        status_indicator=status_indicator,
    )
    return client, controller, recorder


def _chunk(source_observation_id, values):
    timestamp = time.time()
    return RemoteRTCActionChunk(
        source_observation_id=source_observation_id,
        chunk_start_step=0,
        observation_timestamp=timestamp,
        actions=[
            TimedAction(
                timestamp=timestamp + index / 30,
                timestep=index,
                action=torch.tensor([value, value, value]),
            )
            for index, value in enumerate(values)
        ],
        conditioned_delay=0,
        prefix_length=0,
    )


def test_dagger_inherits_paper_rtc_splice_default(tmp_path):
    client, _, _ = _make_client(tmp_path)
    assert "active_splice_bridge_steps" not in client.config.to_dict()


def test_dagger_prompt_cleans_up_on_control_failure_without_keyboard_input(tmp_path, monkeypatch):
    client, controller, recorder = _make_client(tmp_path)
    client.config.dagger_input_device = "prompt"
    _assert_entrypoint_cleans_up_while_input_blocks(client, dagger_module, monkeypatch)
    assert "abort" in controller.calls
    assert "close" in controller.calls
    assert ("close", False) in recorder.calls


def test_dagger_handled_dispatch_failure_preserves_manual_reset_recovery(tmp_path, monkeypatch):
    client, controller, _ = _make_client(tmp_path)
    faulted = threading.Event()
    original_set_fault = client._set_fault

    def fail(action):
        raise RuntimeError("recoverable command failure")

    def observe_fault(operation, exc):
        original_set_fault(operation, exc)
        faulted.set()

    assert client.start_rollout()
    source = encode_rollout_observation_id(client.rollout_generation, 0)
    assert client._handle_action_chunk(_chunk(source, [1.0]))
    monkeypatch.setattr(controller, "send_same_action", fail)
    monkeypatch.setattr(client, "_set_fault", observe_fault)
    monkeypatch.setattr(client, "_reset_robot_pose", lambda: True)
    client.start_barrier = threading.Barrier(1)
    worker = threading.Thread(target=client.control_loop)
    worker.start()
    try:
        assert faulted.wait(timeout=2)
        assert worker.is_alive()
        assert client.running
        assert client.dagger_phase is DAggerPhase.FAULT
        assert client.worker_error is None
        assert "pause" in controller.calls
        assert client.reset_rollout()
        assert client.rollout_state is RolloutState.READY
        assert client.dagger_phase is DAggerPhase.PAUSED
    finally:
        client.stop()
        worker.join(timeout=2)
        assert not worker.is_alive()


def test_rejected_absolute_handover_remains_paused_without_recording(tmp_path, monkeypatch, caplog):
    client, controller, recorder = _make_client(tmp_path)
    assert client.config.dagger_handover_gripper_tolerance == 0.001
    assert client.start_rollout()
    assert client.pause_autonomy()

    def rejected_handover():
        raise PiperHandoverAlignmentError("master closed after switching; restored follower alignment")

    monkeypatch.setattr(controller, "enter_correction", rejected_handover)
    with caplog.at_level(logging.WARNING):
        assert not client.start_correction()

    assert client.dagger_phase is DAggerPhase.PAUSED
    assert not any(isinstance(call, tuple) and call[0] == "begin" for call in recorder.calls)
    assert not client.operator_stop_event.is_set()
    assert "未采集数据，仍处于暂停" in caplog.text
    controller.calls.clear()
    assert client.control_loop_action() is None
    assert not any(isinstance(call, tuple) and call[0] == "send" for call in controller.calls)


def test_failed_handover_restore_still_enters_fault(tmp_path, monkeypatch):
    client, controller, recorder = _make_client(tmp_path)
    assert client.start_rollout()
    assert client.pause_autonomy()

    def failed_handover():
        raise PiperDaggerHardwareError("could not restore master alignment")

    monkeypatch.setattr(controller, "enter_correction", failed_handover)
    assert not client.start_correction()
    assert client.dagger_phase is DAggerPhase.FAULT
    assert not any(isinstance(call, tuple) and call[0] == "begin" for call in recorder.calls)


def test_pause_invalidates_old_chunk_and_correction_end_resets_rollout(tmp_path):
    client, controller, recorder = _make_client(tmp_path)
    assert client.start_rollout()
    generation_before_pause = client.rollout_generation
    old_source = encode_rollout_observation_id(generation_before_pause, 0)
    assert client._handle_action_chunk(_chunk(old_source, [0.0, 1.0]))

    assert client.pause_autonomy()
    paused_generation = client.rollout_generation
    assert paused_generation > generation_before_pause
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert client.action_schedule.qsize(-1) == 0
    assert not client._handle_action_chunk(_chunk(old_source, [99.0]))

    assert client.start_correction()
    assert client.dagger_phase is DAggerPhase.CORRECTING
    assert not client._request_observation_if_ready()
    assert client.stop_correction()
    assert ("finish", True) in recorder.calls
    assert recorder.saved_episodes == 1
    assert client.rollout_generation > paused_generation
    assert client.rollout_state.value == "ready"
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert controller.calls[:4] == ["prepare", "pause", "enter_correction", "leave_correction"]
    assert "abort" in controller.calls
    assert not client._request_observation_if_ready()
    client.stop()


def test_start_is_rejected_without_touching_hardware_while_rollout_is_active(tmp_path):
    client, controller, _ = _make_client(tmp_path)
    assert client.start_rollout()
    calls_before = list(controller.calls)
    assert not client.start_rollout("different task")
    assert controller.calls == calls_before
    client.stop()


@pytest.mark.parametrize("inflight", [False, True])
def test_esc_blocks_queued_and_interpolated_policy_commands(tmp_path, inflight):
    client, controller, _ = _make_client(tmp_path)
    try:
        assert client.start_rollout()
        source = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source, [1.0, 2.0]))
        if inflight:
            assert client.control_loop_action() is not None
        before = list(controller.calls)
        latest = client.latest_action
        assert client.submit_operator_event("stop")
        assert client.control_loop_action() is None
        assert client.control_loop_action() is None
        assert controller.calls == before
        assert client.latest_action == latest
    finally:
        client.stop()


@pytest.mark.parametrize(
    "gate", ["_request_observation_if_ready", "send_observation", "_handle_action_chunk"]
)
def test_esc_rejects_new_policy_requests_and_results(tmp_path, monkeypatch, gate):
    client, _, _ = _make_client(tmp_path)
    entered = []

    def parent_gate(self, *args, **kwargs):
        entered.append(gate)
        return True

    monkeypatch.setattr(RobotClientDeployment, gate, parent_gate)
    try:
        assert client.start_rollout()
        assert client.submit_operator_event("stop")
        assert not getattr(client, gate)(None)
        assert not entered
    finally:
        client.stop()


@pytest.mark.parametrize("operation", ["dispatch", "takeover"])
def test_esc_is_rechecked_after_waiting_for_command_lock(tmp_path, operation):
    client, controller, _ = _make_client(tmp_path)
    waiting = threading.Event()
    inner_lock = client._robot_command_lock
    worker = None
    results = []

    class ObservedCommandLock:
        def __enter__(self):
            if threading.current_thread() is worker:
                waiting.set()
            inner_lock.acquire()
            return self

        def __exit__(self, *exc):
            inner_lock.release()

    try:
        assert client.start_rollout()
        if operation == "dispatch":
            source = encode_rollout_observation_id(client.rollout_generation, 0)
            assert client._handle_action_chunk(_chunk(source, [1.0, 2.0]))
            action = client.control_loop_action
        else:
            assert client.pause_autonomy()
            action = client.start_correction
        client._robot_command_lock = ObservedCommandLock()
        before = list(controller.calls)
        with inner_lock:
            worker = threading.Thread(target=lambda: results.append(action()))
            worker.start()
            assert waiting.wait(1.0)
            assert client.submit_operator_event("stop")
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert results == ([None] if operation == "dispatch" else [False])
        assert controller.calls == before
    finally:
        if worker is not None:
            worker.join(timeout=2.0)
        client._robot_command_lock = inner_lock
        client.stop()


@pytest.mark.parametrize("stage", ["prepare", "handover", "begin"])
@pytest.mark.parametrize("stop_kind", ["esc", "shutdown"])
def test_stop_during_start_does_not_publish_autonomy_or_correction(tmp_path, monkeypatch, stage, stop_kind):
    client, controller, recorder = _make_client(tmp_path)
    try:
        if stage == "prepare":
            target, method = controller, "prepare_autonomy"
            start = client.start_rollout
        else:
            assert client.start_rollout()
            assert client.pause_autonomy()
            target, method = (controller, "enter_correction") if stage == "handover" else (recorder, "begin")
            start = client.start_correction
        original = getattr(target, method)

        def stop_during_operation(*args, **kwargs):
            result = original(*args, **kwargs)
            if stop_kind == "esc":
                assert client.submit_operator_event("stop")
            else:
                client.shutdown_event.set()
            return result

        monkeypatch.setattr(target, method, stop_during_operation)
        assert not start()
        assert client.dagger_phase is DAggerPhase.PAUSED
        assert not recorder.pending
        assert recorder.saved_episodes == 0
        if stage != "begin":
            assert not any(isinstance(call, tuple) and call[0] == "begin" for call in recorder.calls)
        if stage != "prepare":
            assert "abort" in controller.calls
    finally:
        client.stop()


@pytest.mark.parametrize("stop_kind", ["esc", "shutdown"])
def test_stop_during_rollout_setup_does_not_publish_autonomy(tmp_path, monkeypatch, stop_kind):
    client, _, recorder = _make_client(tmp_path)
    parent_start = RobotClientDeployment.start_rollout

    def stop_after_setup(policy_client, task=None):
        result = parent_start(policy_client, task)
        if stop_kind == "esc":
            policy_client.submit_operator_event("stop")
        else:
            policy_client.shutdown_event.set()
        return result

    monkeypatch.setattr(RobotClientDeployment, "start_rollout", stop_after_setup)
    try:
        assert not client.start_rollout()
        assert client.dagger_phase is DAggerPhase.PAUSED
        assert not recorder.pending
    finally:
        client.stop()


def test_cancelled_handover_rollback_failure_enters_fault(tmp_path, monkeypatch):
    client, controller, recorder = _make_client(tmp_path)
    try:
        assert client.start_rollout()
        assert client.pause_autonomy()
        original = controller.enter_correction

        def stop_during_handover():
            original()
            client.submit_operator_event("stop")

        def fail_rollback():
            raise PiperDaggerHardwareError("cancel rollback failed")

        monkeypatch.setattr(controller, "enter_correction", stop_during_handover)
        monkeypatch.setattr(controller, "abort_to_paused", fail_rollback)
        assert not client.start_correction()
        assert client.dagger_phase is DAggerPhase.FAULT
        assert "cancel rollback failed" in client.dagger_fault
        assert not recorder.pending
        assert recorder.saved_episodes == 0
    finally:
        client.stop()


def test_reset_during_correction_disables_takeover_and_discards_partial_window(tmp_path):
    client, controller, recorder = _make_client(tmp_path)
    assert client.start_rollout()
    assert client.pause_autonomy()
    assert client.start_correction()

    assert client.reset_rollout()
    abort_index = controller.calls.index("abort")
    # The inherited reset then sends a dual-arm hold/reset command only after
    # the correction bridge has been aborted.
    assert controller.calls[abort_index + 1][0] == "send"
    assert ("finish", False) in recorder.calls
    assert client.dagger_phase is DAggerPhase.PAUSED
    client.stop()


def test_two_corrections_save_two_episodes_in_one_client_run(tmp_path):
    client, _, recorder = _make_client(tmp_path)
    for expected_episodes in (1, 2):
        assert client.start_rollout()
        assert client.pause_autonomy()
        assert client.start_correction()
        assert client.stop_correction()
        assert recorder.saved_episodes == expected_episodes
        assert client.rollout_state.value == "ready"
        assert client.dagger_phase is DAggerPhase.PAUSED
    assert "episode=2/∞" in _render_status(client)
    assert not client.operator_stop_event.is_set()
    client.stop()


@pytest.mark.parametrize("episode_target", [0, None])
def test_zero_and_none_episode_targets_are_unlimited(tmp_path, episode_target):
    client, _, _ = _make_client(tmp_path, dagger_num_episodes=episode_target)
    assert client.dagger_episode_target is None
    assert not client.dagger_collection_complete
    client.stop()


def test_reset_discards_only_current_window_and_preserves_saved_episodes(tmp_path):
    client, _, recorder = _make_client(tmp_path)
    assert client.start_rollout()
    assert client.pause_autonomy()

    assert client.start_correction()
    assert client.stop_correction()
    assert recorder.saved_episodes == 1

    assert client.start_rollout()
    assert client.pause_autonomy()
    assert client.start_correction()
    assert recorder.pending
    assert client.reset_rollout()
    assert not recorder.pending
    assert recorder.saved_episodes == 1
    assert client.dagger_phase is DAggerPhase.PAUSED
    client.stop()


def test_episode_target_exits_through_safe_disconnect_without_duplicate_reset(tmp_path):
    indicator = _FakeIndicator()
    client, controller, recorder = _make_client(
        tmp_path,
        dagger_num_episodes=2,
        status_indicator=indicator,
    )
    assert client.start_rollout()
    assert client.pause_autonomy()

    assert client.start_correction()
    assert client.stop_correction()
    assert recorder.saved_episodes == 1
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert client.rollout_state.value == "ready"
    assert not client.operator_stop_event.is_set()
    reset_sends_before_final = sum(isinstance(call, tuple) and call[0] == "send" for call in controller.calls)

    assert client.start_rollout()
    assert client.pause_autonomy()
    assert client.start_correction()
    assert client.stop_correction()
    assert recorder.saved_episodes == 2
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert client.rollout_state.value == "running"
    assert client.operator_stop_event.is_set()
    assert client.running
    assert "close" not in controller.calls
    assert (
        sum(isinstance(call, tuple) and call[0] == "send" for call in controller.calls)
        == reset_sends_before_final
    )
    assert "episode=2/2" in _render_status(client)
    assert not client._request_observation_if_ready()
    assert indicator.cues[-1] is StatusCue.EXIT
    calls_at_exit = list(controller.calls)
    assert not client._handle_operator_event("advance")
    assert controller.calls == calls_at_exit

    # The CLI main thread observes operator_stop_event and performs this
    # orderly teardown in its finally block.  For Piper, proxy.disconnect()
    # reaches follower.disconnect() -> safe_disconnect_follower().
    client.stop()
    assert not client.running
    assert "close" in controller.calls
    assert not client._follower_robot.is_connected
    assert indicator.closed


def test_reset_from_correction_discards_unsaved_window_and_starts_no_policy(tmp_path):
    client, _, recorder = _make_client(tmp_path)
    assert client.start_rollout()
    assert client.pause_autonomy()
    assert client.start_correction()

    assert client.reset_rollout()
    assert recorder.saved_episodes == 0
    assert not recorder.pending
    assert ("finish", False) in recorder.calls
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert client.rollout_state.value == "ready"
    assert not client._request_observation_if_ready()

    assert client.start_rollout()
    assert client.pause_autonomy()
    assert client.start_correction()
    assert client.stop_correction()
    assert recorder.saved_episodes == 1
    client.stop()


def test_arrow_state_machine_and_status_cues(tmp_path):
    indicator = _FakeIndicator()
    client, controller, recorder = _make_client(tmp_path, status_indicator=indicator)

    assert client._handle_operator_event("advance")
    assert client.dagger_phase is DAggerPhase.AUTONOMOUS
    assert indicator.cues[-1] is StatusCue.AUTONOMOUS

    assert client._handle_operator_event("advance")
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert indicator.cues[-1] is StatusCue.PAUSED

    assert client._handle_operator_event("advance")
    assert client.dagger_phase is DAggerPhase.CORRECTING
    assert indicator.cues[-1] is StatusCue.CORRECTING

    assert client._handle_operator_event("advance")
    assert recorder.saved_episodes == 1
    assert client.rollout_state.value == "ready"
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert indicator.cues[-1] is StatusCue.RESET_REQUIRED

    calls_after_reset = list(controller.calls)
    assert not client._handle_operator_event("reset")
    assert controller.calls == calls_after_reset

    assert client._handle_operator_event("stop")
    assert client.operator_stop_event.is_set()
    assert indicator.cues[-1] is StatusCue.EXIT
    client.stop()
    assert indicator.closed


def test_operator_keys_emit_distinct_colored_transition_logs(tmp_path, monkeypatch):
    emitted = []

    def capture_log(text, *, play_sounds, color):
        emitted.append((text, play_sounds, color))

    monkeypatch.setattr(dagger_module, "log_say", capture_log)
    client, _, _ = _make_client(tmp_path)

    for _ in range(4):
        assert client._handle_operator_event("advance")
    assert client._handle_operator_event("stop")

    assert [entry[2] for entry in emitted] == [
        "bright_green",
        "bright_yellow",
        "bright_cyan",
        "bright_blue",
        "bright_red",
    ]
    assert "右方向键" in emitted[0][0]
    assert "开始自主推理" in emitted[0][0]
    assert "暂停自主推理" in emitted[1][0]
    assert "人工接管" in emitted[2][0]
    assert "保存 episode" in emitted[3][0]
    assert "Esc 键" in emitted[4][0]
    assert all(not entry[1] for entry in emitted)
    client.stop()


def test_episode_save_releases_transition_lock_and_esc_bypasses_operator_queue(tmp_path, monkeypatch):
    class BlockingRecorder(_FakeRecorder):
        def __init__(self):
            super().__init__()
            self.save_started = threading.Event()
            self.release_save = threading.Event()
            self.close_started = threading.Event()

        def finish_window(self, *, save):
            if save:
                self.save_started.set()
                assert self.release_save.wait(timeout=2.0)
            return super().finish_window(save=save)

        def close(self, *, save_partial=False):
            self.calls.append(("close", save_partial))
            self.close_started.set()
            assert self.release_save.wait(timeout=2.0)

    emitted = []
    monkeypatch.setattr(
        dagger_module,
        "log_say",
        lambda text, *, play_sounds, color: emitted.append((text, play_sounds, color)),
    )
    client, controller, _ = _make_client(tmp_path)
    recorder = BlockingRecorder()
    client.correction_recorder = recorder
    assert client.start_rollout()
    assert client.pause_autonomy()
    assert client.start_correction()

    worker = threading.Thread(target=client.stop_correction)
    worker.start()
    assert recorder.save_started.wait(timeout=1.0)
    assert client.dagger_save_in_progress

    assert not client.submit_operator_event("advance")
    assert client._operator_queue.empty()
    assert client.submit_operator_event("stop")
    assert client.operator_stop_event.is_set()
    assert client._operator_queue.empty()
    assert "Esc 键" in emitted[-1][0]

    # Dataset serialization must not monopolize the hardware transition lock.
    assert client._dagger_transition_lock.acquire(blocking=False)
    client._dagger_transition_lock.release()

    # Hardware shutdown proceeds before dataset finalization is allowed to
    # finish; only the final process teardown remains waiting on the save.
    shutdown = threading.Thread(target=client.stop)
    shutdown.start()
    assert recorder.close_started.wait(timeout=1.0)
    assert "close" in controller.calls
    assert not client.running
    assert shutdown.is_alive()

    recorder.release_save.set()
    worker.join(timeout=1.0)
    shutdown.join(timeout=1.0)
    assert not worker.is_alive()
    assert not shutdown.is_alive()
    assert not client.dagger_save_in_progress


def test_progress_marks_episode_save_as_distinct_phase(tmp_path):
    client, _, _ = _make_client(tmp_path)
    client._save_in_progress.set()

    postfix = _collection_progress_postfix(client, color=True)

    assert "episode 保存中" in postfix
    assert postfix.startswith(dagger_module._ANSI_BOLD_BRIGHT_MAGENTA)
    client._save_in_progress.clear()
    client.stop()


def test_left_key_emits_discard_reset_log(tmp_path, monkeypatch):
    emitted = []
    monkeypatch.setattr(
        dagger_module,
        "log_say",
        lambda text, *, play_sounds, color: emitted.append((text, play_sounds, color)),
    )
    client, _, _ = _make_client(tmp_path)
    assert client._handle_operator_event("advance")
    assert client._handle_operator_event("advance")
    assert client._handle_operator_event("reset")

    assert emitted[-1][2] == "bright_magenta"
    assert "左方向键" in emitted[-1][0]
    assert "复位" in emitted[-1][0]
    client.stop()


def test_environment_reset_prompt_uses_bold_warning_colors(tmp_path, monkeypatch):
    emitted = []
    client, _, _ = _make_client(tmp_path)
    monkeypatch.setattr(client.logger, "warning", emitted.append)
    monkeypatch.setattr(
        dagger_module,
        "_terminal_emphasis",
        lambda text, color, *, enabled=None: f"{color}{text}{dagger_module._ANSI_RESET}",
    )

    client._announce_environment_reset(discarded_correction=False)
    client._announce_environment_reset(discarded_correction=True)

    assert emitted[0].startswith(dagger_module._ANSI_BOLD_BRIGHT_YELLOW)
    assert "请手动恢复实验环境" in emitted[0]
    assert emitted[1].startswith(dagger_module._ANSI_BOLD_BRIGHT_MAGENTA)
    assert "当前未保存 correction 已丢弃" in emitted[1]
    client.stop()


def test_collection_progress_reports_episode_phase_and_live_correction_frames(tmp_path):
    class FakeProgress:
        total = 3
        n = 0

        def __init__(self):
            self.postfix = None
            self.refreshes = 0

        def set_postfix_str(self, value, *, refresh):
            assert not refresh
            self.postfix = value

        def refresh(self):
            self.refreshes += 1

    client, _, recorder = _make_client(tmp_path, dagger_num_episodes=3)
    assert client.start_rollout()
    assert client.pause_autonomy()
    assert client.start_correction()
    recorder.frames_in_window = 27
    recorder.saved_episodes = 1
    progress = FakeProgress()

    _refresh_collection_progress(progress, client)

    assert progress.n == 1
    assert progress.refreshes == 1
    assert progress.postfix == _collection_progress_postfix(client)
    assert "阶段=人工接管" in progress.postfix
    assert "correction=27 帧" in progress.postfix
    assert "episode=1/3" in progress.postfix
    client.stop()


def test_collection_progress_uses_distinct_bold_phase_colors(tmp_path):
    client, _, _ = _make_client(tmp_path)
    phase_colors = {
        DAggerPhase.AUTONOMOUS: dagger_module._ANSI_BOLD_BRIGHT_GREEN,
        DAggerPhase.PAUSED: dagger_module._ANSI_BOLD_BRIGHT_YELLOW,
        DAggerPhase.CORRECTING: dagger_module._ANSI_BOLD_BRIGHT_CYAN,
        DAggerPhase.FAULT: dagger_module._ANSI_BOLD_BRIGHT_RED,
    }

    for phase, expected_color in phase_colors.items():
        client._dagger_phase = phase
        postfix = _collection_progress_postfix(client, color=True)
        assert postfix.startswith(f"{expected_color}阶段=")
        assert f"{dagger_module._ANSI_RESET} | 当前 correction=" in postfix

    assert len(set(phase_colors.values())) == len(phase_colors)
    client.stop()


def test_collection_progress_uses_bold_high_visibility_style(tmp_path, monkeypatch):
    options = {}
    fake_progress = object()

    def fake_tqdm(**kwargs):
        options.update(kwargs)
        return fake_progress

    monkeypatch.setattr(dagger_module, "tqdm", fake_tqdm)
    client, _, _ = _make_client(tmp_path, dagger_num_episodes=3)

    assert dagger_module._make_collection_progress(client) is fake_progress
    assert "\033[1;96m" in options["desc"]
    assert options["colour"] == "#00ff66"
    assert options["dynamic_ncols"]
    client.stop()


def test_progress_logging_redirect_preserves_console_level():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    root_logger.handlers = [console_handler]
    try:
        with dagger_module._progress_safe_logging():
            redirected_handlers = [
                handler for handler in root_logger.handlers if handler is not console_handler
            ]
            assert len(redirected_handlers) == 1
            assert redirected_handlers[0].level == logging.INFO
    finally:
        root_logger.handlers = original_handlers


@pytest.mark.parametrize("abort_phase", [DAggerPhase.PAUSED, DAggerPhase.CORRECTING])
def test_left_arrow_resets_from_pause_or_correction(tmp_path, abort_phase):
    indicator = _FakeIndicator()
    client, _, recorder = _make_client(tmp_path, status_indicator=indicator)
    assert client._handle_operator_event("advance")
    assert client._handle_operator_event("advance")
    if abort_phase is DAggerPhase.CORRECTING:
        assert client._handle_operator_event("advance")

    assert client.dagger_phase is abort_phase
    assert client._handle_operator_event("reset")
    assert recorder.saved_episodes == 0
    assert not recorder.pending
    assert client.rollout_state.value == "ready"
    assert client.dagger_phase is DAggerPhase.PAUSED
    assert indicator.cues[-1] is StatusCue.RESET_REQUIRED
    client.stop()
