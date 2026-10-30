# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Reset admission, recorder failure, bounded wire size and nonblocking diagnostics."""

import logging
import pickle  # nosec
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from lerobot.async_inference.client.piper_hil_dagger import PiperControlMode, PiperDaggerController
from lerobot.async_inference.client.robot_client_dagger import DAggerPhase
from lerobot.async_inference.client.robot_client_deployment import RolloutState
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from lerobot.async_inference.shared.helpers import TimedAction, TimedObservation
from lerobot.async_inference.shared.rtc_remote import RemoteRTCActionChunk, RemoteRTCObservation, RTCActionRef
from tests.async_inference.test_policy_server_deployment import _FakePipeline, _FakePolicy
from tests.async_inference.test_robot_client_dagger import _make_client as make_dagger
from tests.async_inference.test_robot_client_deployment import _chunk, _make_client as make_deployment


@pytest.mark.parametrize("raises", [False, True])
def test_failed_reset_blocks_start_until_another_reset_succeeds(tmp_path, monkeypatch, raises):
    client = make_deployment(tmp_path)

    def fail():
        if raises:
            raise RuntimeError("reset driver failed")
        return False

    try:
        assert client.start_rollout()
        monkeypatch.setattr(client, "_reset_robot_pose", fail)
        assert not client.reset_rollout()
        assert client.rollout_state is RolloutState.RESET_FAILED
        assert client.reset_error
        assert client.running
        assert not client.start_rollout()
        monkeypatch.setattr(client, "_reset_robot_pose", lambda: True)
        assert client.reset_rollout()
        assert client.reset_error is None
        assert client.start_rollout()
    finally:
        client.stop()


@pytest.mark.parametrize("bad_feedback", [0.0, float("nan")])
def test_successful_reset_command_is_not_proof_of_arrival(tmp_path, monkeypatch, bad_feedback):
    client = make_deployment(tmp_path)
    client.config.reset_verify_timeout_s = 0.05
    client._configured_reset_pose = dict.fromkeys(client.robot.action_features, 1.0)
    monkeypatch.setattr(
        client,
        "_read_reset_poses_unlocked",
        lambda _: {"robot": dict.fromkeys(client.robot.action_features, bad_feedback)},
    )
    try:
        assert client.start_rollout()
        assert not client.reset_rollout()
        assert client.rollout_state is RolloutState.RESET_FAILED
        assert not client.start_rollout()
        monkeypatch.setattr(
            client, "_read_reset_poses_unlocked", lambda _: {"robot": dict(client._configured_reset_pose)}
        )
        assert client.reset_rollout()
        assert client.rollout_state is RolloutState.READY
    finally:
        client.stop()


def test_reset_requires_both_arms_and_a_stable_feedback_interval(tmp_path, monkeypatch):
    client, _, _ = make_dagger(tmp_path)
    client.config.reset_verify_timeout_s = 1.0
    client.config.reset_settle_s = 0.08
    target = dict.fromkeys(client.robot.action_features, 1.0)
    client._configured_reset_pose = target
    reading = threading.Event()
    master_arrived = threading.Event()
    results = []

    def feedback(timeout_s):
        reading.set()
        master = target if master_arrived.is_set() else dict.fromkeys(target, 0.0)
        return {"follower": dict(target), "master": dict(master)}

    monkeypatch.setattr(client.dagger_controller, "read_reset_feedback", feedback)
    worker = threading.Thread(target=lambda: results.append(client.reset_rollout()))
    worker.start()
    try:
        assert reading.wait(timeout=1)
        assert client.rollout_state is RolloutState.RESETTING
        assert not client.start_rollout()
        assert not client.submit_operator_event("advance")
        assert not client.reset_rollout()
        master_arrived.set()
        # Both arms being in tolerance for one sample is still insufficient.
        assert not client.start_rollout()
        worker.join(timeout=2)
        assert results == [True]
        assert client.rollout_state is RolloutState.READY
        assert client._operator_queue.empty()
        assert client.start_rollout()
    finally:
        master_arrived.set()
        worker.join(timeout=2)
        client.stop()


def test_operator_events_cannot_cross_reset_or_replay_after_a_transition(tmp_path, monkeypatch):
    client, controller, _ = make_dagger(tmp_path)
    started = threading.Event()
    release = threading.Event()
    results = []

    def abort():
        started.set()
        assert release.wait(timeout=2)

    assert client.start_rollout()
    assert client.pause_autonomy()
    assert client.submit_operator_event("advance")  # Queued before reset begins.
    old_event = client._operator_queue.get_nowait()
    monkeypatch.setattr(controller, "abort_to_paused", abort)
    worker = threading.Thread(target=lambda: results.append(client.reset_rollout()))
    worker.start()
    try:
        assert started.wait(timeout=1)
        # The reset gate must cover CAN handover too, before pose interpolation.
        assert client.rollout_state is RolloutState.RESETTING
        for _ in range(3):
            assert not client.submit_operator_event("advance")
            assert not client.start_rollout()
        release.set()
        worker.join(timeout=2)
        assert results == [True]
        assert not client._handle_operator_event(old_event.name, expected=old_event)
        assert client.rollout_state is RolloutState.READY
        # Rapid duplicate presses from one state cannot start then immediately pause.
        assert client.submit_operator_event("advance")
        assert client.submit_operator_event("advance")
        first = client._operator_queue.get_nowait()
        duplicate = client._operator_queue.get_nowait()
        assert client._handle_operator_event(first.name, expected=first)
        assert not client._handle_operator_event(duplicate.name, expected=duplicate)
        assert client.dagger_phase is DAggerPhase.AUTONOMOUS
    finally:
        release.set()
        worker.join(timeout=2)
        client.stop()


def test_escape_cancels_reset_verification_without_admitting_next_rollout(tmp_path, monkeypatch):
    client, controller, _ = make_dagger(tmp_path)
    reading = threading.Event()
    client._configured_reset_pose = dict.fromkeys(client.robot.action_features, 1.0)

    def feedback(timeout_s):
        reading.set()
        return {arm: dict.fromkeys(client.robot.action_features, 0.0) for arm in ("master", "follower")}

    monkeypatch.setattr(controller, "read_reset_feedback", feedback)
    results = []
    worker = threading.Thread(target=lambda: results.append(client.reset_rollout()))
    worker.start()
    try:
        assert reading.wait(timeout=1)
        assert client.submit_operator_event("stop")
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert results == [False]
        assert client.rollout_state is RolloutState.RESET_FAILED
        assert not client.start_rollout()
    finally:
        client.operator_stop_event.set()
        worker.join(timeout=2)
        client.stop()


@pytest.mark.parametrize("abort_fails", [False, True])
def test_recorder_failure_immediately_faults_and_discards_correction(tmp_path, monkeypatch, abort_fails):
    client, controller, recorder = make_dagger(tmp_path)
    try:
        assert client.start_rollout()
        assert client.pause_autonomy()
        assert client.start_correction()
        if abort_fails:

            def fail():
                raise RuntimeError("handover failed")

            monkeypatch.setattr(controller, "abort_to_paused", fail)
        recorder.last_error = RuntimeError("camera failed while recording")
        assert client.control_loop_action() is None
        assert client.dagger_phase is DAggerPhase.FAULT
        assert "camera failed" in client.dagger_fault
        assert ("finish", False) in recorder.calls
        assert recorder.saved_episodes == 0
        assert not recorder.pending
        assert not client._handle_operator_event("advance")
        assert client.running  # Manual reset recovery remains available.
        if not abort_fails:
            assert "abort" in controller.calls
            assert client.reset_rollout()
            assert client.start_rollout()
    finally:
        client.stop()


def test_reset_feedback_uses_new_physical_samples_from_both_can_channels():
    reads = []
    controller = PiperDaggerController.__new__(PiperDaggerController)
    controller._lock = threading.RLock()
    controller._closed = False
    controller._mode = PiperControlMode.PAUSED
    controller.config = SimpleNamespace(feedback_timeout_s=1.0)
    controller.bridge = SimpleNamespace(
        read_follower_feedback_pose=lambda timeout: reads.append("follower") or {"joint.pos": 1.0},
        read_master_feedback_pose=lambda timeout: reads.append("master") or {"joint.pos": 2.0},
    )
    assert controller.read_reset_feedback(0.1) == {
        "follower": {"joint.pos": 1.0},
        "master": {"joint.pos": 2.0},
    }
    assert reads == ["follower", "master"]


@pytest.mark.parametrize("mode", ["guided", "trained"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_action_chunk_wire_roundtrip_is_compact_and_lossless(mode, dtype):
    # The selected action rows share a much larger backing allocation.
    values = torch.arange(100 * 32, dtype=torch.float32).to(dtype).view(100, 32)[:50, :7]
    chunk = RemoteRTCActionChunk(
        17,
        5,
        123.0,
        [TimedAction(123.0 + i / 30, 5 + i, row) for i, row in enumerate(values)],
        conditioned_delay=3,
        prefix_length=10,
        inference_seconds=0.08,
        rtc_mode=mode,
        training_max_delay=4,
    )
    wire = pickle.dumps(chunk)  # nosec
    restored = pickle.loads(wire)  # nosec
    assert len(wire) < values.numel() * values.element_size() + 2500
    torch.testing.assert_close(torch.stack([action.action for action in restored.actions]), values)
    assert [(a.timestamp, a.timestep) for a in restored.actions] == [
        (a.timestamp, a.timestep) for a in chunk.actions
    ]
    assert {k: v for k, v in vars(restored).items() if k != "actions"} == {
        k: v for k, v in vars(chunk).items() if k != "actions"
    }
    assert values.untyped_storage().nbytes() == 100 * 32 * values.element_size()


def test_empty_action_chunk_wire_roundtrip():
    chunk = RemoteRTCActionChunk(1, 0, 0.0)
    assert pickle.loads(pickle.dumps(chunk)) == chunk  # nosec


@pytest.mark.parametrize("dagger", [False, True])
def test_chunk_logging_does_not_hold_the_control_lifecycle_lock(tmp_path, monkeypatch, dagger):
    client = make_dagger(tmp_path)[0] if dagger else make_deployment(tmp_path)
    client.config.debug_rtc_diagnostics = True
    logging_started = threading.Event()
    release_logging = threading.Event()
    dispatched = threading.Event()

    class SlowLogHandler(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("RTC MERGE LIVE"):
                logging_started.set()
                assert release_logging.wait(timeout=2)

    log = logging.Logger("slow-rtc-diagnostics", level=logging.DEBUG)
    log.addHandler(SlowLogHandler())
    monkeypatch.setattr(client, "logger", log)
    assert client.start_rollout()
    assert client._request_observation_if_ready()
    request = client._request_mailbox.get_nowait()
    receiver = threading.Thread(
        target=lambda: client._handle_action_chunk(_chunk(request.observation_id, [1.0]))
    )
    control = threading.Thread(target=lambda: (client.control_loop_action(), dispatched.set()))
    receiver.start()
    try:
        assert logging_started.wait(timeout=1)
        control.start()
        assert dispatched.wait(timeout=0.5), "diagnostic handler blocked action dispatch"
    finally:
        release_logging.set()
        receiver.join(timeout=2)
        if control.ident is not None:
            control.join(timeout=2)
        client.stop()


@pytest.mark.parametrize("diagnostics", [False, True])
def test_server_alignment_diagnostics_are_opt_in_debug(caplog, diagnostics):
    server = PolicyServerDeployment(PolicyServerDeploymentConfig(debug_rtc_diagnostics=diagnostics))
    server.policy = _FakePolicy("checkpoint")
    server.preprocessor = _FakePipeline()
    server.postprocessor = _FakePipeline()
    server.device = "cpu"
    server.actions_per_chunk = 20
    server.lerobot_features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}
    }
    overlap = torch.ones(3, 1)
    server._active_rollout_generation = 0
    server._chunk_cache.put(7, overlap, overlap)
    envelope = RemoteRTCObservation(
        TimedObservation(time.time(), 0, {"joint.pos": 0.0}),
        8,
        -1,
        action_refs=tuple(RTCActionRef(7, i) for i in range(3)),
    )
    try:
        with caplog.at_level(logging.DEBUG):
            server._predict_remote_chunk(envelope)
        records = [
            r
            for r in caplog.records
            if r.getMessage().startswith(("RTC ALIGN", "RTC CONDITIONED EDGE", "RTC SERVER PHASES"))
        ]
        assert bool(records) == diagnostics
        assert all(record.levelno == logging.DEBUG for record in records)
    finally:
        server.stop()
