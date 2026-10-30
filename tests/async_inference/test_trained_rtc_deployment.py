# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Trained deployment isolation and fresh-snapshot recovery, without hardware."""

import pickle  # nosec
import threading
import time
from dataclasses import replace
from queue import Empty, Queue
from types import SimpleNamespace

import grpc
import pytest
import torch

import lerobot.async_inference.shared.rtc_trained as trained_module
from lerobot.async_inference.client.robot_client_deployment import (
    RobotClientDeployment,
    RobotClientDeploymentConfig,
    _InterpolatedClient,
    _RTCClient,
)
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
)
from lerobot.async_inference.shared.helpers import TimedAction, TimedObservation
from lerobot.async_inference.shared.rtc_remote import (
    RemoteRTCActionChunk,
    RemoteRTCObservation,
    RemoteRTCRetry,
    RTCActionRef,
    RTCPrefixUnavailableError,
    encode_rollout_observation_id,
)
from lerobot.async_inference.shared.rtc_trained import (
    TrainedActionSchedule,
    TrainedRTCClientConfig,
    TrainedRTCClientRuntime,
    TrainedRTCServerConfig,
    TrainedRTCServerRuntime,
)
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.rtc import RTCConfig, RTCProcessor
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    batch_to_transition,
    create_transition,
)
from lerobot.transport import services_pb2
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.mocks.mock_robot import MockRobotConfig


class _Identity:
    steps = ()

    def __call__(self, value):
        return value

    def reset(self):
        pass


class _NativePrefixPolicy:
    def __init__(self):
        self.config = SimpleNamespace(
            rtc_training_max_delay=3, chunk_size=8, image_features={}, rtc_config=None
        )
        self.calls = []
        self.rtc_processor = None
        self.model = SimpleNamespace(rtc_processor=None)

    def predict_action_chunk(self, observation, *, inference_delay, prev_chunk_left_over):
        self.calls.append((observation, inference_delay, prev_chunk_left_over))
        actions = torch.arange(8, dtype=torch.float32).view(1, 8, 1).expand(1, 8, 3).clone()
        if inference_delay:
            actions[0, :inference_delay] = prev_chunk_left_over[:inference_delay]
        return actions

    def init_rtc_processor(self):
        self.rtc_processor = RTCProcessor(self.config.rtc_config, trained_mode_supported=True)
        self.model.rtc_processor = self.rtc_processor

    def reset(self):
        pass

    def supports_rtc(self):
        return True

    def eval(self):
        return self


def _request(prefix, *, observation_id=0, start=0, state=0.0):
    return RemoteRTCObservation(
        TimedObservation(time.time(), start, {f"motor_{i}.pos": state for i in range(1, 4)}),
        observation_id=observation_id,
        executed_action_step=start - 1,
        trained_prefix=prefix,
    )


def _chunk(request, *, delay=0, values=None, mode="trained"):
    if values is None:
        values = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 3).clone()
    start = request.observation.get_timestep()
    return RemoteRTCActionChunk(
        request.observation_id,
        start,
        request.observation.get_timestamp(),
        [TimedAction(time.time(), start + i, value) for i, value in enumerate(values)],
        conditioned_delay=delay,
        prefix_length=len(request.trained_prefix)
        if request.trained_prefix is not None
        else len(request.action_refs),
        rtc_mode=mode,
        training_max_delay=3 if mode == "trained" else 0,
    )


def _runtime(**kwargs):
    return TrainedRTCServerRuntime(
        config=kwargs.pop("config", TrainedRTCServerConfig()),
        policy=kwargs.pop("policy", _NativePrefixPolicy()),
        preprocessor=kwargs.pop("preprocessor", _Identity()),
        postprocessor=kwargs.pop("postprocessor", _Identity()),
        features={
            OBS_STATE: {"dtype": "float32", "shape": (3,), "names": [f"motor_{i}.pos" for i in range(1, 4)]}
        },
        device="cpu",
        fps=30,
        chunk_size=kwargs.pop("chunk_size", 8),
        queue_threshold=kwargs.pop("queue_threshold", 4),
        **kwargs,
    )


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    clients = []
    monkeypatch.setattr(_InterpolatedClient, "send_observation", lambda self, envelope: True)

    def make(mode="trained"):
        client = RobotClientDeployment(
            RobotClientDeploymentConfig(
                policy_type="pi05",
                pretrained_name_or_path="checkpoint",
                robot=MockRobotConfig(calibration_dir=tmp_path, random_values=False, static_values=[0.0] * 3),
                rtc_mode=mode,
                trained=TrainedRTCClientConfig(queue_threshold=4),
                actions_per_chunk=8,
                server_address="localhost:9999",
                interpolation_multiplier=2,
                adaptive_rtt_trigger=True,
                reset_duration_s=0,
                task="test",
            )
        )
        clients.append(client)
        assert client.start_rollout()
        return client

    yield make
    for client in clients:
        client.stop()


def _capture_request(client):
    assert client._request_observation_if_ready()
    capture = client._request_mailbox.get_nowait()
    snapshot = client._snapshot_execution_state()
    request = _request(
        snapshot.trained_prefix,
        observation_id=capture.observation_id,
        start=snapshot.executed_action_step + 1,
    )
    request = replace(request, action_refs=snapshot.action_refs)
    assert client.send_observation(request)
    return request


def test_trained_server_distinguishes_missing_prefix_from_empty_queue():
    runtime = _runtime()
    response = runtime.predict(_request(None))
    assert isinstance(response, RemoteRTCRetry)
    assert response.reason == "prefix_unavailable"
    assert runtime.policy.calls == []

    empty = runtime.predict(_request(torch.empty(0, 3)))
    assert isinstance(empty, RemoteRTCActionChunk)
    assert empty.conditioned_delay == 0
    assert runtime.policy.calls[-1][1:] == (0, None)

    prefix = torch.full((4, 3), 7.0)
    conditioned = runtime.predict(_request(prefix))
    assert conditioned.conditioned_delay == 3
    torch.testing.assert_close(torch.stack([a.action for a in conditioned.actions[:3]]), prefix[:3])


def test_trained_server_late_attempt_returns_retry_once(monkeypatch):
    runtime = _runtime()
    clock = iter((0.0, 0.2, 0.2))
    monkeypatch.setattr(trained_module.time, "perf_counter", lambda: next(clock))
    response = runtime.predict(_request(torch.ones(4, 3)))
    assert isinstance(response, RemoteRTCRetry)
    assert response.reason == "delay_exceeded"
    assert response.inference_seconds == 0.2
    assert len(runtime.policy.calls) == 1


@pytest.mark.parametrize("relative_enabled", [False, True])
def test_trained_prefix_roundtrips_normalization_and_current_state(relative_enabled):
    features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
    stats = {ACTION: {"mean": torch.tensor([0.5, 1.0, 1.5]), "std": torch.tensor([2.0, 3.0, 4.0])}}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    relative = RelativeActionsProcessorStep(enabled=relative_enabled)
    absolute = AbsoluteActionsProcessorStep(enabled=relative_enabled, relative_step=relative)

    class Preprocessor:
        steps = (relative, normalizer)

        def __call__(self, observation):
            relative(batch_to_transition(observation))
            return observation

    def postprocessor(actions):
        return absolute(unnormalizer(create_transition(action=actions)))[TransitionKey.ACTION]

    runtime = _runtime(preprocessor=Preprocessor(), postprocessor=postprocessor)
    prefix = torch.tensor([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [17.0, 18.0, 19.0]])
    for state in (10.0, 20.0):
        result = runtime.predict(_request(prefix, state=state))
        torch.testing.assert_close(torch.stack([a.action for a in result.actions[:3]]), prefix)
        expected = prefix - state if relative_enabled else prefix
        expected = (expected - stats[ACTION]["mean"]) / (stats[ACTION]["std"] + normalizer.eps)
        torch.testing.assert_close(runtime.policy.calls[-1][2], expected)


def test_trained_public_server_bypasses_guided_cache_latency_and_processor(monkeypatch):
    server = PolicyServerDeployment(PolicyServerDeploymentConfig(rtc=RTCConfig(enabled=True, mode="trained")))
    runtime = _runtime()
    server.policy = runtime.policy
    server.preprocessor = runtime.preprocessor
    server.postprocessor = runtime.postprocessor
    server._trained_runtime = runtime
    server._active_rollout_generation = 0

    def forbidden(*args, **kwargs):
        pytest.fail("trained called a guided runtime operation")

    monkeypatch.setattr(server, "_reconstruct_prefix", forbidden)
    monkeypatch.setattr(server, "_current_latency", forbidden)
    monkeypatch.setattr(server, "_record_latency", forbidden)
    try:
        server._configure_policy_rtc()
        assert type(server.policy.rtc_processor) is RTCProcessor
        result = server._predict_remote_chunk(_request(torch.ones(4, 3)))
        assert isinstance(result, RemoteRTCActionChunk)
        assert result.conditioned_delay == 3
        assert server._chunk_cache.get(0) is None
    finally:
        server.stop()


def test_trained_client_ignores_guided_heuristics_and_preserves_inflight_endpoint(make_client, monkeypatch):
    client = make_client()

    def forbidden(*args, **kwargs):
        pytest.fail("trained called guided merging")

    monkeypatch.setattr(_InterpolatedClient, "_handle_action_chunk", forbidden)
    client._adaptive_rtt_steps = 100.0
    assert type(client.action_schedule) is TrainedActionSchedule
    assert client._trigger_threshold() == 4
    first = _capture_request(client)
    assert first.action_refs == ()
    assert first.trained_prefix.shape == (0, 3)
    assert client._handle_action_chunk(_chunk(first))
    assert client._adaptive_rtt_steps == 100.0
    assert client._latest_latency_feedback is None

    # Reach endpoint 3, then start interpolating toward endpoint 4. The next
    # prefix includes endpoint 4 even though its first command has been sent.
    for _ in range(8):
        client.control_loop_action()
    assert client.latest_action == 3
    inflight = client.action_schedule.inflight()
    assert inflight.timed_action.get_timestep() == 4
    request = _capture_request(client)
    torch.testing.assert_close(request.trained_prefix[:, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    values = torch.full((8, 3), 40.0)
    values[:3] = request.trained_prefix[:3]
    assert client._handle_action_chunk(_chunk(request, delay=3, values=values))
    assert client.action_schedule.inflight() is inflight
    plan = client.action_schedule.snapshot_prefix(client.latest_action, 3)
    torch.testing.assert_close(plan[:, 0], torch.tensor([4.0, 5.0, 6.0, 40.0, 40.0, 40.0, 40.0, 40.0]))
    assert client.control_loop_action()["motor_1.pos"] == 4.0


@pytest.mark.parametrize("invalid", ["changed_prefix", "fake_cold", "wrong_steps"])
def test_trained_response_cannot_bypass_its_snapshot(invalid):
    runtime = TrainedRTCClientRuntime(TrainedRTCClientConfig(), 8, "cpu")
    request = _request(torch.ones(4, 3))
    runtime.remember_request(request)
    values = torch.ones(8, 3)
    chunk = _chunk(request, delay=3, values=values)
    if invalid == "changed_prefix":
        chunk.actions[0] = TimedAction(time.time(), 0, torch.zeros(3))
    elif invalid == "fake_cold":
        chunk.prefix_length = 0
        chunk.conditioned_delay = 0
    else:
        chunk.actions[0] = TimedAction(time.time(), 10, values[0])
    with pytest.raises(ValueError, match="snapshot|hard prefix"):
        runtime.accept(chunk, -1)
    assert runtime.schedule.qsize(-1) == 0


def test_trained_client_rejects_late_result_without_guided_tolerance(make_client):
    client = make_client()
    first = _capture_request(client)
    assert client._handle_action_chunk(_chunk(first))
    for _ in range(7):
        client.control_loop_action()
    request = _capture_request(client)
    values = torch.full((8, 3), 40.0)
    values[:2] = request.trained_prefix[:2]
    # Three endpoints become immutable. Guided permits +3 steps, trained must
    # reject even this one-step overrun and release the single-flight slot.
    for _ in range(6):
        client.control_loop_action()
    assert not client._handle_action_chunk(_chunk(request, delay=2, values=values))
    assert client._inflight_request_id is None
    assert client._force_fresh_snapshot
    assert client._trained_runtime.consecutive_rejections == 1
    fresh = _capture_request(client)
    assert fresh.observation_id > request.observation_id
    assert fresh.executed_action_step > request.executed_action_step
    assert len(fresh.trained_prefix) < len(request.trained_prefix)


@pytest.mark.parametrize("replan_on_result", [True, False])
def test_trained_fast_result_replans_without_waiting_for_half_chunk(make_client, replan_on_result):
    client = make_client()
    client.config.trained.replan_on_result = replan_on_result
    request = _capture_request(client)
    assert not client._request_observation_if_ready()
    assert client._handle_action_chunk(_chunk(request))
    assert client.action_schedule.qsize(-1) > client._trigger_threshold()
    assert client._request_observation_if_ready() is replan_on_result
    assert not client._request_observation_if_ready()


@pytest.mark.parametrize("mode", ["guided", "trained"])
def test_retry_recaptures_sensor_and_execution_prefix_and_fences_old_replies(make_client, mode):
    client = make_client(mode)
    first = _capture_request(client)
    assert client._handle_action_chunk(_chunk(first, mode=mode))
    for _ in range(7):
        client.control_loop_action()
    request = _capture_request(client)
    retry = RemoteRTCRetry(
        request.observation_id, request.observation.get_timestamp(), "delay_exceeded", mode
    )
    assert client._handle_snapshot_retry(retry)
    assert client._inflight_request_id is None
    client.control_loop_action()

    # Drive the actual asynchronous capture path, not a resend of `request`.
    captured = Queue()
    client.robot.get_observation = lambda: dict.fromkeys(client.robot.observation_features, 12.0)
    client.send_observation = lambda envelope: captured.put(envelope) or True
    client.start_barrier = threading.Barrier(1)
    worker = threading.Thread(target=client.observation_sender)
    worker.start()
    try:
        assert client._request_observation_if_ready()
        fresh = captured.get(timeout=2)
        assert fresh.observation_id > request.observation_id
        assert fresh.observation.get_timestamp() > request.observation.get_timestamp()
        assert fresh.observation.get_observation()["motor_1.pos"] == 12.0
        if mode == "trained":
            assert fresh.trained_prefix is not request.trained_prefix
            torch.testing.assert_close(
                fresh.trained_prefix, client._snapshot_execution_state().trained_prefix
            )
        else:
            assert fresh.action_refs == client._snapshot_execution_state().action_refs
        assert not client._handle_snapshot_retry(retry)
        assert client._inflight_request_id == fresh.observation_id
        # This transport test's mock must report the reset target for verification.
        client.robot.get_observation = lambda: dict.fromkeys(client.robot.observation_features, 0.0)
        assert client.reset_rollout()
        if mode == "trained":
            assert client._trained_runtime.consecutive_rejections == 0
            assert client._trained_runtime.schedule.qsize(-1) == 0
        assert not client._handle_snapshot_retry(replace(retry, source_observation_id=fresh.observation_id))
    finally:
        client.shutdown_event.set()
        worker.join(timeout=2)
        assert not worker.is_alive()


@pytest.mark.parametrize("failure", ["discard", "prefix", "exception"])
def test_server_never_reinfers_rejected_snapshot(monkeypatch, failure):
    server = PolicyServerDeployment(PolicyServerDeploymentConfig(action_wait_timeout=0))
    calls = []
    stop = threading.Event()

    def predict(request):
        calls.append(request.observation_id)
        if len(calls) == 1:
            if failure == "prefix":
                raise RTCPrefixUnavailableError("missing cache entry")
            if failure == "exception":
                raise RuntimeError("transient inference failure")
            return RemoteRTCRetry(request.observation_id, 0, "delay_exceeded")
        return _chunk(request, mode="guided")

    monkeypatch.setattr(server, "_predict_remote_chunk", predict)
    worker = threading.Thread(target=server._inference_loop, args=(stop,))
    worker.start()
    try:
        server._observation_mailbox.put(_request(None, observation_id=0))
        assert isinstance(server._action_mailbox.get(timeout=2), RemoteRTCRetry)
        with pytest.raises(Empty):
            server._action_mailbox.get(timeout=0.15)
        assert calls == [0]
        server._observation_mailbox.put(_request(None, observation_id=1, state=2.0))
        result = server._action_mailbox.get(timeout=2)
        assert isinstance(result, RemoteRTCActionChunk)
        assert result.source_observation_id == 1
        assert calls == [0, 1]
    finally:
        stop.set()
        worker.join(timeout=2)
        server.stop()
        assert not worker.is_alive()


def test_guided_cache_miss_is_not_silently_treated_as_cold_start():
    server = PolicyServerDeployment(PolicyServerDeploymentConfig())
    try:
        request = replace(_request(None), action_refs=(RTCActionRef(17, 0),))
        with pytest.raises(RTCPrefixUnavailableError):
            server._reconstruct_prefix(request)
        assert server._reconstruct_prefix(_request(None)) == (None, None)
    finally:
        server.stop()


def test_trained_setup_and_serialized_retry_recovers_through_public_endpoints(make_client, monkeypatch):
    client = make_client()
    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(
            rtc=RTCConfig(enabled=True, mode="trained", execution_horizon=1),
            action_wait_timeout=2,
            inference_latency=0,
        )
    )
    policy = _NativePrefixPolicy()

    def load(specs):
        server.policy = policy
        server.preprocessor = _Identity()
        server.postprocessor = _Identity()
        server.device = "cpu"
        server.actions_per_chunk = specs.actions_per_chunk
        server.lerobot_features = specs.lerobot_features
        return False

    monkeypatch.setattr(server, "_load_or_reuse_policy", load)
    monkeypatch.setattr(_InterpolatedClient, "send_observation", _RTCClient.send_observation)
    client.stub = SimpleNamespace(
        SendObservations=lambda stream, **kwargs: server.SendObservations(stream, None)
    )

    def receive():
        response = server.GetActions(services_pb2.Empty(), None)
        assert response.data
        return pickle.loads(response.data)  # nosec

    try:
        server.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(client.policy_setup)), None)
        assert server._trained_runtime.delay_steps == 3
        first = _capture_request(client)
        assert client._handle_action_chunk(receive())
        for _ in range(7):
            client.control_loop_action()

        native_predict = policy.predict_action_chunk

        def fail_once(observation, **kwargs):
            policy.predict_action_chunk = native_predict
            raise RuntimeError("simulated transient inference error")

        policy.predict_action_chunk = fail_once
        rejected = _capture_request(client)
        response = receive()
        assert isinstance(response, RemoteRTCRetry)
        assert response.rtc_mode == "trained"
        assert response.reason == "inference_error"
        assert client._handle_snapshot_retry(response)

        client.control_loop_action()
        fresh = _capture_request(client)
        response = receive()
        assert response.source_observation_id == fresh.observation_id > rejected.observation_id
        assert rejected.observation_id > first.observation_id
        assert response.prefix_length == len(fresh.trained_prefix)
        assert client._handle_action_chunk(response)
        assert client._trained_runtime.consecutive_rejections == 0
        assert server._chunk_cache.get(fresh.observation_id) is None
    finally:
        server.stop()


def test_trained_repeated_rejections_stop_client_and_release_request(make_client):
    client = make_client()
    maximum = client.config.trained.max_consecutive_rejections
    for attempt in range(maximum):
        request = _capture_request(client)
        retry = RemoteRTCRetry(
            request.observation_id, request.observation.get_timestamp(), "delay_exceeded", "trained"
        )
        if attempt == maximum - 1:
            with pytest.raises(RuntimeError, match="repeatedly rejected"):
                client._handle_snapshot_retry(retry)
        else:
            assert client._handle_snapshot_retry(retry)
            assert client.running
    assert not client.running
    assert client._inflight_request_id is None
    assert not client._trained_runtime._requests


@pytest.mark.parametrize("mode", ["guided", "trained"])
@pytest.mark.parametrize("lost_result", ["chunk", "retry"])
def test_lost_terminal_rpc_response_recovers_with_fresh_capture(make_client, monkeypatch, mode, lost_result):
    client = make_client(mode)
    first = _capture_request(client)
    assert client._handle_action_chunk(_chunk(first, mode=mode))
    for _ in range(7):
        client.control_loop_action()
    old = _capture_request(client)
    server = PolicyServerDeployment(PolicyServerDeploymentConfig(action_wait_timeout=0))
    retry = RemoteRTCRetry(old.observation_id, old.observation.timestamp, "delay_exceeded", mode)
    lost = _chunk(old, mode=mode) if lost_result == "chunk" else retry
    captured = Queue()
    monkeypatch.setattr(_InterpolatedClient, "send_observation", lambda self, req: captured.put(req) or True)
    client.robot.get_observation = lambda: dict.fromkeys(client.robot.observation_features, 12.0)
    client.start_barrier = threading.Barrier(1)
    worker = threading.Thread(target=client.observation_sender)
    worker.start()
    try:
        server._action_mailbox.put(lost)
        assert server.GetActions(services_pb2.Empty(), None).data  # Consumed, then lost in transit.
        assert not server.GetActions(services_pb2.Empty(), None).data
        assert not client._request_observation_if_ready()  # Still respect single-flight before timeout.
        for _ in range(2):
            client.control_loop_action()
        original_rtt = client._adaptive_rtt_steps
        with client._transport_lock:
            client._inflight_requested_at = time.monotonic() - client.config.request_timeout_s - 1

        assert client._request_observation_if_ready()
        fresh = captured.get(timeout=2)
        assert fresh.observation_id > old.observation_id
        assert fresh.executed_action_step > old.executed_action_step
        assert fresh.observation.timestamp > old.observation.timestamp
        assert fresh.observation.observation["motor_1.pos"] == 12.0
        snapshot = client._snapshot_execution_state()
        if mode == "trained":
            torch.testing.assert_close(fresh.trained_prefix, snapshot.trained_prefix)
            assert old.observation_id not in client._trained_runtime._requests
            assert client._trained_runtime.consecutive_rejections == 0
        else:
            assert fresh.action_refs == snapshot.action_refs
        assert old.observation_id not in client._request_timings
        assert client._adaptive_rtt_steps == original_rtt

        # Old completion, retry, upload and upload-failure callbacks cannot
        # clear or contaminate the replacement in the same rollout generation.
        assert not client._handle_action_chunk(_chunk(old, mode=mode))
        assert not client._handle_snapshot_retry(retry)
        assert not client.send_observation(old)
        client._retire_failed_request(old.observation_id)
        assert client._inflight_request_id == fresh.observation_id
        values = torch.full((8, 3), 40.0)
        delay = min(3, len(fresh.trained_prefix)) if mode == "trained" else 0
        if delay:
            values[:delay] = fresh.trained_prefix[:delay]
        assert client._handle_action_chunk(_chunk(fresh, delay=delay, values=values, mode=mode))
        assert client._inflight_request_id is None
        assert client.running
        assert not client.send_observation(old)  # Cannot resurrect an expired id even with an empty slot.
    finally:
        client.shutdown_event.set()
        worker.join(timeout=2)
        server.stop()
        assert not worker.is_alive()


@pytest.mark.parametrize("mode", ["guided", "trained"])
def test_capture_finishing_after_request_timeout_is_not_uploaded(make_client, monkeypatch, mode):
    client = make_client(mode)
    captured = Queue()
    started = threading.Event()
    release = threading.Event()
    capture_count = 0

    def capture():
        nonlocal capture_count
        capture_count += 1
        if capture_count == 1:
            started.set()
            assert release.wait(timeout=2)
        return dict.fromkeys(client.robot.observation_features, float(capture_count))

    client.robot.get_observation = capture
    monkeypatch.setattr(_InterpolatedClient, "send_observation", lambda self, req: captured.put(req) or True)
    assert client._request_observation_if_ready()
    old_id = client._inflight_request_id
    client.start_barrier = threading.Barrier(1)
    worker = threading.Thread(target=client.observation_sender)
    worker.start()
    try:
        assert started.wait(timeout=2)
        with client._transport_lock:
            client._inflight_requested_at = time.monotonic() - client.config.request_timeout_s - 1
        assert client._request_observation_if_ready()
        fresh_id = client._inflight_request_id
        release.set()
        fresh = captured.get(timeout=2)
        assert fresh.observation_id == fresh_id > old_id
        assert fresh.observation.observation["motor_1.pos"] == 2.0
        assert captured.empty()
    finally:
        release.set()
        client.shutdown_event.set()
        worker.join(timeout=2)
        assert not worker.is_alive()


@pytest.mark.parametrize("mode", ["guided", "trained"])
@pytest.mark.parametrize("upload_succeeds", [False, True])
def test_expired_upload_completion_cannot_release_replacement(
    make_client, monkeypatch, mode, upload_succeeds
):
    client = make_client(mode)
    started = threading.Event()
    release = threading.Event()
    result = []

    def upload(self, request):
        started.set()
        assert release.wait(timeout=2)
        return upload_succeeds

    monkeypatch.setattr(_InterpolatedClient, "send_observation", upload)
    # The direct-call compatibility path must also reserve the next sequence id.
    old = _request(
        torch.empty(0, 3) if mode == "trained" else None,
        observation_id=encode_rollout_observation_id(client.rollout_generation, 5),
    )
    worker = threading.Thread(target=lambda: result.append(client.send_observation(old)))
    worker.start()
    try:
        assert started.wait(timeout=2)
        with client._transport_lock:
            client._inflight_requested_at = time.monotonic() - client.config.request_timeout_s - 1
        assert client._request_observation_if_ready()
        fresh_id = client._inflight_request_id
        assert fresh_id > old.observation_id
        fresh_requested_at = client._inflight_requested_at
        release.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert result == [False]
        assert client._inflight_request_id == fresh_id
        assert client._inflight_requested_at == fresh_requested_at
        assert fresh_id in client._request_timings
        if mode == "trained":
            assert not client._trained_runtime._requests
    finally:
        release.set()
        worker.join(timeout=2)


def test_runtime_rpcs_have_finite_deadlines_and_poll_recovers(make_client, monkeypatch):
    client = make_client()
    monkeypatch.setattr(_InterpolatedClient, "send_observation", _RTCClient.send_observation)
    deadlines = []

    class LostResponse(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.DEADLINE_EXCEEDED

    def upload(stream, *, timeout):
        assert list(stream)
        deadlines.append(timeout)

    def poll(request, *, timeout):
        deadlines.append(timeout)
        if len(deadlines) == 2:
            raise LostResponse()
        client.shutdown_event.set()
        return services_pb2.Actions()

    client.stub = SimpleNamespace(SendObservations=upload, GetActions=poll)
    _capture_request(client)
    client.start_barrier = threading.Barrier(1)
    client.config.rpc_retry_backoff = 0
    client.receive_actions()
    assert deadlines == [client.config.request_timeout_s] * 3


@pytest.mark.parametrize(
    "change",
    [{"protocol_version": 1}, {"protocol_version": 2}, {"protocol_version": 3}, {"rtc_mode": "trained"}],
)
def test_deployment_rejects_protocol_or_mode_mismatch_before_loading(make_client, monkeypatch, change):
    client = make_client("guided")
    server = PolicyServerDeployment(PolicyServerDeploymentConfig())
    monkeypatch.setattr(
        server, "_load_or_reuse_policy", lambda _: pytest.fail("must validate before loading")
    )
    try:
        setup = replace(client.policy_setup, **change)
        with pytest.raises(ValueError, match="update both|must match"):
            server.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(setup)), None)
    finally:
        server.stop()


@pytest.mark.parametrize(
    "overrides",
    [{"config": TrainedRTCServerConfig(delay_steps=4)}, {"chunk_size": 5}, {"queue_threshold": 2}],
)
def test_trained_configuration_rejects_unsupported_commitment_budget(overrides):
    with pytest.raises(ValueError, match="trained"):
        _runtime(**overrides)
