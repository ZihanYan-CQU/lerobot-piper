# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the consolidated deployment robot client."""

import threading
import time
from dataclasses import replace

import pytest
import torch

import lerobot.async_inference.client.robot_client_deployment as deployment_client_module
from lerobot.async_inference.client.robot_client_deployment import (
    RobotClientDeployment,
    RobotClientDeploymentConfig,
    RolloutState,
    _InterpolatedClient,
)
from lerobot.async_inference.shared.helpers import TimedAction, TimedObservation
from lerobot.async_inference.shared.rtc_remote import (
    RemoteRTCActionChunk,
    RemoteRTCObservation,
    RTCActionRef,
    decode_rollout_generation,
    decode_rollout_sequence,
    encode_rollout_observation_id,
)
from tests.mocks.mock_robot import MockRobotConfig


def _make_client(
    tmp_path,
    *,
    task: str = "task A",
    actions_per_chunk: int = 8,
    adaptive_rtt_trigger: bool = True,
    rtt_trigger_margin_steps: int = 5,
    guided_late_action_tolerances: dict[str, float] | None = None,
) -> RobotClientDeployment:
    return RobotClientDeployment(
        RobotClientDeploymentConfig(
            policy_type="smolvla",
            pretrained_name_or_path="checkpoint",
            robot=MockRobotConfig(
                calibration_dir=tmp_path,
                random_values=False,
                static_values=[0.0, 0.0, 0.0],
            ),
            actions_per_chunk=actions_per_chunk,
            server_address="localhost:9999",
            interpolation_multiplier=2,
            adaptive_rtt_trigger=adaptive_rtt_trigger,
            rtt_trigger_margin_steps=rtt_trigger_margin_steps,
            guided_late_action_tolerances=guided_late_action_tolerances or {},
            reset_duration_s=0,
            reset_settle_s=0,
            task=task,
        )
    )


def _chunk(
    source_observation_id: int,
    values: list[float],
    *,
    start_step: int = 0,
    conditioned_delay: int = 0,
    prefix_length: int = 0,
) -> RemoteRTCActionChunk:
    timestamp = time.time()
    return RemoteRTCActionChunk(
        source_observation_id=source_observation_id,
        chunk_start_step=start_step,
        observation_timestamp=timestamp,
        actions=[
            TimedAction(
                timestamp=timestamp + index / 30,
                timestep=start_step + index,
                action=torch.tensor([value, value, value]),
            )
            for index, value in enumerate(values)
        ],
        conditioned_delay=conditioned_delay,
        prefix_length=prefix_length,
    )


def _observation(
    client: RobotClientDeployment,
    observation_id: int,
    state: float,
    *,
    has_future_actions: bool,
) -> RemoteRTCObservation:
    raw_observation = dict.fromkeys(client.robot.observation_features, state)
    refs = (RTCActionRef(observation_id - 1, 0),) if has_future_actions else ()
    return RemoteRTCObservation(
        observation=TimedObservation(
            timestamp=time.time(),
            timestep=0,
            observation=raw_observation,
        ),
        observation_id=observation_id,
        executed_action_step=-1,
        action_refs=refs,
    )


def test_deployment_defaults_to_unmodified_rtc_trajectory(tmp_path):
    client = _make_client(tmp_path)
    assert "active_splice_bridge_steps" not in client.config.to_dict()


@pytest.mark.parametrize("inflight", [False, True])
def test_chunk_replacement_preserves_every_new_policy_endpoint(tmp_path, inflight):
    client = _make_client(tmp_path)
    try:
        assert client.start_rollout()
        source = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source, [0.0, 2.0, 4.0, 6.0]))
        client.control_loop_action()
        if inflight:
            client.control_loop_action()
        first_mutable = 2 if inflight else 1
        incoming = [10.0, 20.0, 40.0, 42.0, 44.0, 46.0]
        assert client._handle_action_chunk(_chunk(source + 1, incoming))
        refs = client.action_schedule.snapshot_refs(client.latest_action, 8)
        expected = tuple(RTCActionRef(source + 1, i) for i in range(first_mutable, len(incoming)))
        assert refs == ((RTCActionRef(source, 1),) + expected if inflight else expected)
        if inflight:
            assert client.control_loop_action()["motor_1.pos"] == 2.0
        for value in incoming[first_mutable:]:
            client.control_loop_action()  # Ordinary interpolation midpoint.
            assert client.control_loop_action()["motor_1.pos"] == value
    finally:
        client.stop()


def test_reset_clears_every_episode_scoped_execution_state(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    generation_a = client.rollout_generation
    source_a = encode_rollout_observation_id(generation_a, 0)
    assert client._handle_action_chunk(_chunk(source_a, [0.0, 2.0, 4.0]))

    client.control_loop_action()  # policy endpoint 0
    client.control_loop_action()  # midpoint toward endpoint 1 (still in flight)
    assert client.latest_action == 0
    assert client.action_schedule.inflight() is not None

    with client._transport_lock:
        client._inflight_request_id = source_a
        client._inflight_requested_at = time.monotonic() - 0.25
        client._adaptive_rtt_steps = 7.5
        client._latest_request_rtt_seconds = 0.25
        client._rtt_updated_at = time.monotonic()
        client._consecutive_rtt_capacity_overruns = 3
        client._rtt_capacity_warning_emitted = True
        client._latest_latency_feedback = (source_a, 0.25)
    client.action_queue_size.extend([3, 2, 1])

    assert client.reset_rollout()

    assert client.rollout_state is RolloutState.READY
    assert client.rollout_generation > generation_a
    assert client.task == "task A"
    assert client.latest_action == -1
    assert client.action_schedule.inflight() is None
    assert client.action_schedule.qsize(-1) == 0
    assert client.action_interpolator.needs_new_action()
    assert client.action_interpolator._prev is None
    assert client.command_diagnostics_snapshot() is None
    assert list(client.action_queue_size) == [3, 2, 1]  # Process-level diagnostics survive reset.
    assert client.action_chunk_size == client.config.actions_per_chunk
    assert client._inflight_request_id is None
    assert client._inflight_requested_at is None
    assert client._request_timings == {}
    assert client._adaptive_rtt_steps == 7.5
    assert client._latest_request_rtt_seconds == 0.25
    assert client._consecutive_rtt_capacity_overruns == 0
    assert not client._rtt_capacity_warning_emitted
    assert client._latest_latency_feedback is None
    assert client._latest_received_chunk_id == -1
    assert client._starvation_started_at is None
    assert client._starvation_missed_command_ticks == 0
    assert client.running
    assert client.robot.is_connected

    # A fresh interpolator emits task B's first endpoint directly; it cannot
    # interpolate from task A's previous/in-flight action.
    assert client.start_rollout("task B")
    source_b = encode_rollout_observation_id(client.rollout_generation, 0)
    assert client._handle_action_chunk(_chunk(source_b, [10.0, 12.0]))
    assert client.control_loop_action() == {
        "motor_1.pos": 10.0,
        "motor_2.pos": 10.0,
        "motor_3.pos": 10.0,
    }
    client.stop()


def test_late_chunk_from_pre_reset_inference_cannot_mutate_new_rollout(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout("task A")
    old_source = encode_rollout_observation_id(client.rollout_generation, 9)

    assert client.reset_rollout()
    assert client.start_rollout("task B")
    generation_b = client.rollout_generation

    assert not client._handle_action_chunk(_chunk(old_source, [90.0, 91.0]))
    assert client.latest_action == -1
    assert client.action_schedule.qsize(-1) == 0
    assert client._latest_received_chunk_id == -1
    assert client._latest_latency_feedback is None

    new_source = encode_rollout_observation_id(generation_b, 0)
    assert client._handle_action_chunk(_chunk(new_source, [1.0, 2.0]))
    assert client.action_schedule.qsize(-1) == 2
    client.stop()


def test_observation_ids_are_monotonic_and_generation_scoped_across_reset(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    generation_a = client.rollout_generation
    assert client._request_observation_if_ready()
    request_a = client._request_mailbox.get_nowait()

    assert decode_rollout_generation(request_a.observation_id) == generation_a
    assert decode_rollout_sequence(request_a.observation_id) == 0

    assert client.reset_rollout()
    assert client.start_rollout("task B")
    generation_b = client.rollout_generation
    assert client._request_observation_if_ready()
    request_b = client._request_mailbox.get_nowait()

    assert generation_b > generation_a
    assert request_b.observation_id > request_a.observation_id
    assert decode_rollout_generation(request_b.observation_id) == generation_b
    assert decode_rollout_sequence(request_b.observation_id) == 0
    assert request_b.task == "task B"
    client.stop()


def test_empty_schedule_allows_only_one_concurrent_inference_request(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()

    worker_count = 8
    barrier = threading.Barrier(worker_count)
    results = []

    def request_at_once():
        barrier.wait()
        results.append(client._request_observation_if_ready())

    workers = [threading.Thread(target=request_at_once) for _ in range(worker_count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers)
    assert sum(results) == 1
    assert client._request_sequence == 1
    request = client._request_mailbox.get_nowait()
    assert client._inflight_request_id == request.observation_id

    # An empty schedule still cannot publish a second request until the first
    # inference returns.
    assert not client._request_observation_if_ready()
    assert client._request_sequence == 1

    assert not client._handle_action_chunk(_chunk(request.observation_id, []))
    assert client._inflight_request_id is None
    assert client._request_observation_if_ready()
    assert client._request_sequence == 2
    client.stop()


def test_starvation_diagnostics_span_empty_schedule_until_first_real_command(tmp_path):
    client = _make_client(tmp_path)
    client.config.debug_rtc_diagnostics = True
    assert client.start_rollout()

    assert client.control_loop_action() is None
    assert client._starvation_started_at is not None
    assert client._starvation_missed_command_ticks == 1

    source = encode_rollout_observation_id(client.rollout_generation, 0)
    assert client._handle_action_chunk(_chunk(source, [1.0, 2.0]))
    assert client.control_loop_action() == {
        "motor_1.pos": 1.0,
        "motor_2.pos": 1.0,
        "motor_3.pos": 1.0,
    }
    assert client._starvation_started_at is None
    assert client._starvation_missed_command_ticks == 0
    client.stop()


def test_adaptive_trigger_uses_full_rtt_and_keeps_single_flight_pipeline_continuous(monkeypatch, tmp_path):
    client = _make_client(tmp_path, actions_per_chunk=50)
    assert client.start_rollout()
    clock = [100.0]
    monkeypatch.setattr(deployment_client_module.time, "monotonic", lambda: clock[0])

    assert client._trigger_threshold() == 25
    assert client._request_observation_if_ready()
    first_request = client._request_mailbox.get_nowait()
    clock[0] += 0.79  # ceil(0.79 * 30 Hz) = 24 policy steps
    assert not client._handle_action_chunk(_chunk(first_request.observation_id, []))

    assert client._latest_request_rtt_seconds == pytest.approx(0.79)
    assert client._adaptive_rtt_steps == 24
    # max(RTT + margin, chunk - RTT + margin) = max(29, 31).
    # A steady-state result normally leaves about 26 actions here, so the next
    # request starts immediately while the request-id gate still permits only
    # one inference at a time.
    assert client._trigger_threshold() == 31

    # A faster result lowers the estimate gradually instead of moving the
    # request boundary back and forth with every RTT sample.
    assert client._request_observation_if_ready()
    second_request = client._request_mailbox.get_nowait()
    clock[0] += 0.39  # ceil(0.39 * 30 Hz) = 12 policy steps
    assert not client._handle_action_chunk(_chunk(second_request.observation_id, []))
    assert client._adaptive_rtt_steps == pytest.approx(21.6)
    assert client._trigger_threshold() == 33
    client.stop()


def test_adaptive_threshold_triggers_before_fixed_half_chunk_boundary(tmp_path):
    client = _make_client(tmp_path, actions_per_chunk=50)
    assert client.start_rollout()
    source = encode_rollout_observation_id(client.rollout_generation, 0)
    assert client._handle_action_chunk(_chunk(source, [0.0] * 50))

    with client.latest_action_lock:
        client.latest_action = 20
    with client._transport_lock:
        client._adaptive_rtt_steps = 24.0
        client._rtt_updated_at = time.monotonic()
    client._request_sequence = 1

    assert client.action_schedule.qsize(client.latest_action) == 29
    assert client._trigger_threshold() == 31
    assert client._trigger_threshold() > int(client.action_chunk_size * client.config.chunk_size_threshold)
    assert client._request_observation_if_ready()
    client.stop()


def test_new_chunk_merges_guided_prefix_from_real_live_step(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    generation = client.rollout_generation
    old_source = encode_rollout_observation_id(generation, 0)
    new_source = encode_rollout_observation_id(generation, 1)

    assert client._handle_action_chunk(_chunk(old_source, list(range(8))))
    assert client._handle_action_chunk(
        _chunk(
            new_source,
            [10.0 + index for index in range(8)],
            conditioned_delay=3,
            prefix_length=8,
        )
    )

    # Nothing has been consumed yet, so step 0 is the real live step.  The
    # server's guided prefix is the transition trajectory, not an old-schedule
    # ownership interval; all still-mutable rows are therefore replaced.  Raw
    # indices remain indices into the full server-side chunk for reconstruction.
    assert client.action_schedule.snapshot_refs(-1, 8) == (
        RTCActionRef(new_source, 0),
        RTCActionRef(new_source, 1),
        RTCActionRef(new_source, 2),
        RTCActionRef(new_source, 3),
        RTCActionRef(new_source, 4),
        RTCActionRef(new_source, 5),
        RTCActionRef(new_source, 6),
        RTCActionRef(new_source, 7),
    )
    client.stop()


def test_guided_lateness_uses_consumed_steps_not_wall_clock(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    generation = client.rollout_generation
    old_source = encode_rollout_observation_id(generation, 0)
    new_source = encode_rollout_observation_id(generation, 1)
    assert client._handle_action_chunk(_chunk(old_source, [0.0, 1.0, 2.0, 3.0]))
    client.control_loop_action()  # commit step 0; step 1 is the real live step

    delayed = _chunk(
        new_source,
        [10.0, 11.0, 12.0, 13.0],
        start_step=1,
        conditioned_delay=1,
        prefix_length=3,
    )
    delayed.observation_timestamp = time.time() - 5.0

    # The robot consumed zero rows from this chunk while it was in flight.  A
    # large wall RTT must not make the client pretend hundreds of actions ran
    # and discard an otherwise live guided trajectory.
    assert client._handle_action_chunk(delayed)
    assert client.action_schedule.snapshot_refs(0, 4)[0].source_observation_id == new_source
    client.stop()


@pytest.mark.parametrize("adaptive", [True, False])
def test_fast_result_can_replan_before_queue_reaches_threshold(tmp_path, adaptive):
    client = _make_client(tmp_path, actions_per_chunk=50, adaptive_rtt_trigger=adaptive)
    try:
        assert client.start_rollout()
        assert client._request_observation_if_ready()
        request = client._request_mailbox.get_nowait()
        assert not client._request_observation_if_ready()
        assert client._handle_action_chunk(_chunk(request.observation_id, [0.0] * 50))
        assert client.action_schedule.qsize(-1) > client._trigger_threshold()
        assert client._request_observation_if_ready() is adaptive
        # Even with immediate replanning, the next request owns the single slot.
        assert not client._request_observation_if_ready()
    finally:
        client.stop()


@pytest.mark.parametrize(
    "conflict", [None, "conditioned", "past", "inflight", "seam", "missing_history", "unset"]
)
def test_late_guided_chunk_requires_agreement_through_execution_boundary(tmp_path, conflict):
    tolerances = {} if conflict == "unset" else dict.fromkeys([f"motor_{i}.pos" for i in range(1, 4)], 0.05)
    client = _make_client(tmp_path, actions_per_chunk=12, guided_late_action_tolerances=tolerances)
    try:
        assert client.start_rollout()
        source = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source, list(range(12))))
        for _ in range(10):
            client.control_loop_action()
        assert client.latest_action == 4
        inflight = client.action_schedule.inflight()
        assert inflight.timed_action.get_timestep() == 5
        before = client.action_schedule.snapshot_refs(4, 12)

        # Consumed=6 exceeds d+3=4. The old rule discarded even this compatible
        # trajectory. An earlier conflicting path must still fail even when it
        # returns to the right anchor at step 5.
        values = [float(i) + 0.03 for i in range(12)]
        if conflict in {"conditioned", "past", "inflight", "seam"}:
            values[{"conditioned": 0, "past": 2, "inflight": 5, "seam": 6}[conflict]] += 1.0
        if conflict == "missing_history":
            client.action_schedule._committed_history.clear()
        late = _chunk(source + 1, values, conditioned_delay=1, prefix_length=12)
        assert client._handle_action_chunk(late) is (conflict is None)
        assert client.action_schedule.inflight() is inflight
        after = client.action_schedule.snapshot_refs(4, 12)
        if conflict is None:
            assert after[0] == before[0]
            assert after[1] == RTCActionRef(source + 1, 6)
        else:
            assert after == before
    finally:
        client.stop()


def test_late_guided_tolerance_uses_each_feature_unit(tmp_path):
    tolerances = {"motor_1.pos": 0.05, "motor_2.pos": 0.05, "motor_3.pos": 0.001}
    client = _make_client(tmp_path, guided_late_action_tolerances=tolerances)
    try:
        assert client.start_rollout()
        source = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source, [0.0] * 8))
        for _ in range(9):
            client.control_loop_action()
        late = _chunk(source + 1, [0.0] * 8, conditioned_delay=1, prefix_length=8)
        late.actions[2].get_action()[2] = 0.002
        assert not client._handle_action_chunk(late)
    finally:
        client.stop()


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_guided_feedback_uses_monotonic_snapshot_time_and_logs_phases(
    monkeypatch, tmp_path, caplog, timing_enabled
):
    client = _make_client(tmp_path)
    client.config.log_rtc_timing = timing_enabled
    try:
        assert client.start_rollout()
        clock = [100.0]
        monkeypatch.setattr(deployment_client_module.time, "monotonic", lambda: clock[0])
        assert client._request_observation_if_ready()
        request = client._request_mailbox.get_nowait()
        client._mark_request_timing(request.observation_id, "snapshot_at", 100.2)
        clock[0] = 100.7
        chunk = _chunk(request.observation_id, [0.0] * 8)
        chunk.observation_timestamp -= 300.0  # A wall-clock adjustment must not inflate d.
        chunk.inference_seconds = 0.3
        with caplog.at_level("INFO"):
            assert client._handle_action_chunk(chunk)
        assert client._latest_latency_feedback[1] == pytest.approx(0.5)
        assert client._latest_request_rtt_seconds == pytest.approx(0.7)
        assert ("snapshot_to_result_ms=500.0 outside_server_ms=200.0" in caplog.text) is timing_enabled
        assert ("accepted=True" in caplog.text) is timing_enabled
    finally:
        client.stop()


def test_single_inflight_gate_rejects_redundant_direct_send(monkeypatch, tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    generation = client.rollout_generation
    uploaded = []

    def collect_upload(self, envelope):
        del self
        uploaded.append(envelope)
        return True

    monkeypatch.setattr(_InterpolatedClient, "send_observation", collect_upload)

    initial_id = encode_rollout_observation_id(generation, 0)
    initial = _observation(client, initial_id, 0.0, has_future_actions=False)
    assert client.send_observation(initial)
    assert initial.observation.must_go
    assert client._handle_action_chunk(_chunk(initial_id, [0.0, 1.0]))
    # Similar joint states still need a new prediction to refill the queue.
    refill_id = encode_rollout_observation_id(generation, 1)
    refill = _observation(client, refill_id, 0.1, has_future_actions=True)
    assert client.send_observation(refill)
    assert refill.observation.must_go
    assert client._inflight_request_id == refill_id

    # A second observation cannot enter transport while the refill inference is
    # still live. The request-id gate rejects it before upload.
    redundant_id = encode_rollout_observation_id(generation, 2)
    redundant = _observation(client, redundant_id, 0.1, has_future_actions=True)
    assert not client.send_observation(redundant)
    assert not redundant.observation.must_go
    assert client._inflight_request_id == refill_id
    assert [envelope.observation_id for envelope in uploaded] == [initial_id, refill_id]

    # A result frees the request slot for the next refill immediately.
    assert client._handle_action_chunk(_chunk(refill_id, [2.0, 3.0]))
    assert client._inflight_request_id is None
    next_refill_id = encode_rollout_observation_id(generation, 3)
    next_refill = _observation(client, next_refill_id, 0.1, has_future_actions=True)
    assert client.send_observation(next_refill)
    assert next_refill.observation.must_go
    assert [envelope.observation_id for envelope in uploaded] == [
        initial_id,
        refill_id,
        next_refill_id,
    ]
    client.stop()


def test_failed_refill_upload_releases_only_its_own_request_slot(monkeypatch, tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    observation_id = encode_rollout_observation_id(client.rollout_generation, 0)
    envelope = _observation(client, observation_id, 0.0, has_future_actions=True)

    def fail_upload(self, sent_envelope):
        del self
        assert sent_envelope is envelope
        return False

    monkeypatch.setattr(_InterpolatedClient, "send_observation", fail_upload)
    assert not client.send_observation(envelope)
    assert client._inflight_request_id is None
    assert client._force_fresh_snapshot
    assert client._retired_request_id == observation_id

    # A late failure from an older request must not clear a newer request's
    # request slot or trigger another fresh snapshot.
    newer_id = encode_rollout_observation_id(client.rollout_generation, 1)
    with client._transport_lock:
        client._inflight_request_id = newer_id
        client._force_fresh_snapshot = False
    client._retire_failed_request(observation_id)
    assert client._inflight_request_id == newer_id
    assert not client._force_fresh_snapshot
    client.stop()


def test_late_old_generation_chunk_cannot_release_new_inflight_request(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout("task A")
    assert client._request_observation_if_ready()
    old_request = client._request_mailbox.get_nowait()

    assert client.reset_rollout()
    assert client.start_rollout("task B")
    assert client._request_observation_if_ready()
    new_request = client._request_mailbox.get_nowait()
    assert client._inflight_request_id == new_request.observation_id

    assert not client._handle_action_chunk(_chunk(old_request.observation_id, [90.0]))
    assert client._inflight_request_id == new_request.observation_id

    assert client._handle_action_chunk(_chunk(new_request.observation_id, [1.0]))
    assert client._inflight_request_id is None
    client.stop()


def test_capture_started_before_reset_is_dropped_after_generation_changes(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout("task A")
    assert client._request_observation_if_ready()

    capture_started = threading.Event()
    release_capture = threading.Event()
    capture_finished = threading.Event()
    uploads = []

    def delayed_capture():
        capture_started.set()
        assert release_capture.wait(timeout=2)
        capture_finished.set()
        return {
            "motor_1.pos": 0.0,
            "motor_2.pos": 0.0,
            "motor_3.pos": 0.0,
        }

    def collect_upload(envelope):
        uploads.append(envelope)
        return True

    client.robot.get_observation = delayed_capture
    client.send_observation = collect_upload
    client._reset_robot_pose = lambda: True
    client.start_barrier = threading.Barrier(1)
    sender = threading.Thread(target=client.observation_sender)
    sender.start()

    try:
        assert capture_started.wait(timeout=2)
        source_a = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source_a, [1.0]))
        action_done = threading.Event()

        def dispatch_during_capture():
            client.control_loop_action()
            action_done.set()

        action_thread = threading.Thread(target=dispatch_during_capture)
        action_thread.start()
        assert action_done.wait(timeout=0.2), "camera capture blocked the local action path"
        action_thread.join(timeout=2)

        assert client.reset_rollout()
        assert client.start_rollout("task B")
        release_capture.set()
        assert capture_finished.wait(timeout=2)
        time.sleep(0.05)
        assert uploads == []
    finally:
        release_capture.set()
        client.stop()
        sender.join(timeout=2)


def test_reset_waits_for_an_already_started_action_then_fences_all_later_actions(tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    source = encode_rollout_observation_id(client.rollout_generation, 0)
    assert client._handle_action_chunk(_chunk(source, [1.0, 2.0]))

    send_started = threading.Event()
    release_send = threading.Event()
    reset_done = threading.Event()
    commands = []

    def blocked_send(action):
        commands.append(action)
        send_started.set()
        assert release_send.wait(timeout=2)
        return action

    client.robot.send_action = blocked_send
    client._reset_robot_pose = lambda: True

    action_thread = threading.Thread(target=client.control_loop_action)
    action_thread.start()
    assert send_started.wait(timeout=2)

    def do_reset():
        assert client.reset_rollout()
        reset_done.set()

    reset_thread = threading.Thread(target=do_reset)
    reset_thread.start()
    assert not reset_done.wait(timeout=0.05)

    release_send.set()
    action_thread.join(timeout=2)
    reset_thread.join(timeout=2)
    assert reset_done.is_set()
    assert len(commands) == 1
    assert client.control_loop_action() is None
    assert len(commands) == 1
    client.stop()


def test_reset_does_not_wait_for_inflight_upload_and_upload_cannot_cross_boundary(monkeypatch, tmp_path):
    client = _make_client(tmp_path)
    assert client.start_rollout("task A")
    generation_a = client.rollout_generation
    observation_id = encode_rollout_observation_id(generation_a, 0)
    envelope = RemoteRTCObservation(
        observation=TimedObservation(
            timestamp=time.time(),
            timestep=0,
            observation={
                "motor_1.pos": 0.0,
                "motor_2.pos": 0.0,
                "motor_3.pos": 0.0,
            },
        ),
        observation_id=observation_id,
        executed_action_step=-1,
    )

    upload_started = threading.Event()
    release_upload = threading.Event()
    result = []

    def blocked_upload(self, sent_envelope):
        del self
        assert sent_envelope is envelope
        upload_started.set()
        assert release_upload.wait(timeout=2)
        return True

    monkeypatch.setattr(_InterpolatedClient, "send_observation", blocked_upload)
    client._reset_robot_pose = lambda: True
    upload_thread = threading.Thread(target=lambda: result.append(client.send_observation(envelope)))
    upload_thread.start()
    assert upload_started.wait(timeout=2)

    # The gate covers the RPC itself, not only the mailbox/camera stages.
    # The empty queue needs a refill, but the active upload owns the only slot.
    assert not client._request_observation_if_ready()

    # The RPC remains blocked, but reset is local and must complete immediately.
    assert client.reset_rollout()
    assert client.start_rollout("task B")
    assert client._inflight_request_id is None

    release_upload.set()
    upload_thread.join(timeout=2)
    assert result == [False]
    assert client._inflight_request_id is None
    client.stop()


def test_control_thread_survives_reset_and_runs_a_second_rollout(tmp_path):
    client = _make_client(tmp_path)
    client.start_barrier = threading.Barrier(1)
    client._reset_robot_pose = lambda: True
    client._request_observation_if_ready = lambda task=None: False
    control_thread = threading.Thread(target=client.control_loop)
    control_thread.start()

    def wait_for_step(step: int) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if client.latest_action == step:
                return
            time.sleep(0.01)
        raise AssertionError(f"client did not execute policy step {step}")

    try:
        assert client.start_rollout("task A")
        source_a = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source_a, [1.0]))
        wait_for_step(0)

        assert client.reset_rollout()
        assert client.rollout_state is RolloutState.READY
        assert control_thread.is_alive()

        assert client.start_rollout("task B")
        source_b = encode_rollout_observation_id(client.rollout_generation, 0)
        assert client._handle_action_chunk(_chunk(source_b, [7.0]))
        wait_for_step(0)
        assert control_thread.is_alive()
    finally:
        client.stop()
        control_thread.join(timeout=2)

    assert not control_thread.is_alive()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_request_timeout_must_be_finite_and_positive(tmp_path, timeout):
    client = _make_client(tmp_path)
    try:
        with pytest.raises(ValueError, match="request_timeout_s"):
            replace(client.config, request_timeout_s=timeout)
    finally:
        client.stop()


@pytest.mark.parametrize("failure_at", ["dispatch", "refill"])
def test_control_thread_failure_invalidates_rollout_and_signals_shutdown(tmp_path, monkeypatch, failure_at):
    client = _make_client(tmp_path)
    assert client.start_rollout()
    assert client._request_observation_if_ready()
    request = client._request_mailbox.get_nowait()
    response = _chunk(request.observation_id, [1.0, 2.0])
    assert client._handle_action_chunk(response)

    def fail(*args, **kwargs):
        raise RuntimeError(f"simulated {failure_at} failure")

    if failure_at == "dispatch":
        monkeypatch.setattr(client.robot, "send_action", fail)
    else:
        monkeypatch.setattr(client, "_request_observation_if_ready", fail)
    client.start_barrier = threading.Barrier(1)
    worker = threading.Thread(target=client.control_loop)
    worker.start()
    try:
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert not client.running
        assert client.rollout_state is RolloutState.FAULT
        assert f"simulated {failure_at} failure" in client.worker_error
        assert client.rollout_generation > decode_rollout_generation(response.source_observation_id)
        assert client.action_schedule.qsize(-1) == 0
        assert client.action_schedule.inflight() is None
        assert client._inflight_request_id is None
        assert client.control_loop_action() is None
        assert not client._handle_action_chunk(response)
        assert not client.start_rollout()
        assert not client.reset_rollout()
    finally:
        client.stop()
        worker.join(timeout=2)
    assert client.rollout_state is RolloutState.STOPPED
    assert not client.robot.is_connected


def _assert_entrypoint_cleans_up_while_input_blocks(client, module, monkeypatch):
    """Exercise the actual CLI owner, with fake network workers and real control failure."""

    input_started = threading.Event()
    release_input = threading.Event()
    input_finished = threading.Event()
    errors = []

    def blocked_input(prompt):
        input_started.set()
        try:
            assert release_input.wait(timeout=5)
            return "/start"
        finally:
            input_finished.set()

    def network_worker():
        client.start_barrier.wait(timeout=2)
        client.shutdown_event.wait(timeout=5)

    def fail_refill():
        assert input_started.wait(timeout=2)
        raise RuntimeError("simulated control failure with blocked stdin")

    def run():
        try:
            module.async_client.__wrapped__(client.config)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(module, type(client).__name__, lambda config: client)
    monkeypatch.setattr(client, "start", lambda: True)
    monkeypatch.setattr(client, "receive_actions", network_worker)
    monkeypatch.setattr(client, "observation_sender", network_worker)
    monkeypatch.setattr(client, "_request_observation_if_ready", fail_refill)
    monkeypatch.setattr("builtins.input", blocked_input)
    assert client.start_rollout()
    owner = threading.Thread(target=run)
    owner.start()
    try:
        assert input_started.wait(timeout=2)
        owner.join(timeout=3)
        assert not owner.is_alive(), "CLI cleanup was blocked waiting for user input"
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
        assert "simulated control failure" in str(errors[0])
        assert client.rollout_state is RolloutState.STOPPED
        assert not client.robot.is_connected
        assert "simulated control failure" in client.worker_error
    finally:
        release_input.set()
        client.shutdown_event.set()
        owner.join(timeout=3)
        client.stop()
        assert input_finished.wait(timeout=2)


def test_deployment_cleans_up_on_control_failure_without_keyboard_input(tmp_path, monkeypatch):
    client = _make_client(tmp_path)
    _assert_entrypoint_cleans_up_while_input_blocks(client, deployment_client_module, monkeypatch)
