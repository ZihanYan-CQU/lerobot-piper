# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Exercise startup on a real inference worker with serialized, in-process RPCs."""

import logging
import pickle  # nosec
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import grpc
import numpy as np
import pytest
import torch

import lerobot.async_inference.server.policy_server_deployment as server_module
from lerobot.async_inference.client.robot_client_deployment import (
    RobotClientDeployment,
    RobotClientDeploymentConfig,
    RolloutState,
)
from lerobot.async_inference.server.pi05_inference import PI05InferenceConfig
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
    _SlidingWindowLatencyTracker,
)
from lerobot.async_inference.shared.helpers import TimedObservation
from lerobot.async_inference.shared.image_transport import (
    IMAGE_TRANSPORT_MODES,
    encode_observation_images,
    prepare_remote_observation,
)
from lerobot.async_inference.shared.rtc_remote import (
    RemoteRTCActionChunk,
    RemoteRTCCalibrationRequest,
    RemoteRTCCalibrationResult,
    RemoteRTCObservation,
    encode_rollout_observation_id,
)
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc import RTCConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from tests.mocks.mock_robot import MockRobotConfig


class _Pipeline:
    steps = ()

    def __init__(self):
        self.threads = []
        self.reset_count = 0

    def __call__(self, value):
        self.threads.append(threading.get_ident())
        return value

    def reset(self):
        self.reset_count += 1


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = PI05Config(
            device="cpu",
            chunk_size=20,
            n_action_steps=20,
            image_resolution=(16, 16),
            tokenizer_max_length=8,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (3,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (3,))},
        )
        self.calls = []
        self.reset_count = 0
        self.rtc_processor = None
        self.fail = False
        self.entered = None
        self.release = None

    def supports_rtc(self):
        return True

    def init_rtc_processor(self):
        pass

    def reset(self):
        self.reset_count += 1

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, inference_delay, prev_chunk_left_over):
        assert not torch.is_inference_mode_enabled()
        assert batch["observation.state"].shape == (1, 3)
        self.calls.append((threading.get_ident(), inference_delay, prev_chunk_left_over is not None))
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(timeout=2)
        return torch.full((1, 20, 3), float("nan") if self.fail else 0.25)


class _ActionRPC:
    def __init__(self, stub):
        self.stub = stub

    def __call__(self, request, **kwargs):
        return self.stub._get_actions(request, **kwargs)

    def future(self, request, **kwargs):
        return self.stub.executor.submit(self.stub._get_actions, request, **kwargs)


class _Stub:
    """Keep protobuf streaming, pickle packing and worker queues in the test path."""

    def __init__(self, server):
        self.server = server
        self.responses = []
        self.requests = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.GetActions = _ActionRPC(self)

    def Ready(self, request):  # noqa: N802
        return self.server.Ready(request, SimpleNamespace(peer=lambda: "test-client"))

    def SendPolicyInstructions(self, request):  # noqa: N802
        return self.server.SendPolicyInstructions(request, None)

    def SendObservations(self, requests, timeout=None):  # noqa: N802
        del timeout
        self.requests.append(time.monotonic())
        return self.server.SendObservations(requests, None)

    def _get_actions(self, request, timeout=None):
        del timeout
        response = self.server.GetActions(request, None)
        if response.data:
            self.responses.append(pickle.loads(response.data))  # nosec
        return response


@pytest.fixture
def pair(tmp_path, monkeypatch, request):
    mode = getattr(request, "param", "guided")
    policy = _Policy()
    if mode == "trained":
        policy.config.rtc_training_max_delay = 4
    pre, post = _Pipeline(), _Pipeline()
    monkeypatch.setattr(
        server_module,
        "get_policy_class",
        lambda _: SimpleNamespace(
            config_class=SimpleNamespace(from_pretrained=lambda *a, **kw: policy.config),
            from_pretrained=lambda *a, **kw: policy,
        ),
    )
    monkeypatch.setattr(server_module, "make_pre_post_processors", lambda *a, **kw: (pre, post))
    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(
            action_wait_timeout=0.05,
            pi05_inference=PI05InferenceConfig(warmup_runs=2),
            rtc=RTCConfig(mode=mode),
        )
    )
    client = RobotClientDeployment(
        RobotClientDeploymentConfig(
            policy_type="pi05",
            pretrained_name_or_path="test-checkpoint",
            actions_per_chunk=20,
            robot=MockRobotConfig(calibration_dir=tmp_path, random_values=False, static_values=[0.0] * 3),
            resize_images_on_client=False,
            reset_duration_s=0,
            reset_settle_s=0,
            startup_calibration_runs=3,
            startup_calibration_timeout_s=2,
            task="test task",
            rtc_mode=mode,
        )
    )
    client.stub = _Stub(server)
    yield client, server, policy, pre, post
    client.stop()
    server.stop()
    client.stub.executor.shutdown(wait=True)


@pytest.mark.parametrize("pair", ["guided", "trained"], indirect=True)
@pytest.mark.parametrize("image_mode", IMAGE_TRANSPORT_MODES)
def test_real_grpc_calibration_preserves_negotiated_images_and_never_dispatches(
    pair, monkeypatch, image_mode
):
    client, server, policy, _, _ = pair
    image_key = "observation.images.top"
    policy.config.input_features[image_key] = PolicyFeature(FeatureType.VISUAL, (3, 16, 16))
    client.policy_setup.policy.lerobot_features[image_key] = {
        "dtype": "image",
        "shape": (24, 32, 3),
        "names": ["height", "width", "channels"],
    }
    client.policy_setup = replace(
        client.policy_setup, client_image_resize=True, image_transport_mode=image_mode
    )
    raw = {**client.robot.get_observation(), "top": np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)}
    monkeypatch.setattr(client.robot, "get_observation", lambda: raw.copy())
    monkeypatch.setattr(
        client.robot, "send_action", lambda _: pytest.fail("calibration dispatched an action")
    )
    images = []
    predict = policy.predict_action_chunk

    def capture(batch, **kwargs):
        images.append(batch[image_key].clone())
        return predict(batch, **kwargs)

    monkeypatch.setattr(policy, "predict_action_chunk", capture)
    rpc = grpc.server(ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server, rpc)
    port = rpc.add_insecure_port("127.0.0.1:0")
    rpc.start()
    in_process_stub = client.stub
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client.stub = services_pb2_grpc.AsyncInferenceStub(channel)
            assert client.start()
            assert "top" in client._image_transport_plan
            assert client._image_transport_plan["top"].mode == image_mode
            expected_raw = raw
            if image_mode == "openpi_uint8":
                expected_raw, _, _ = encode_observation_images(raw, client._image_transport_plan)
            expected = prepare_remote_observation(
                expected_raw,
                client.policy_config.lerobot_features,
                policy.config.image_features,
                client._image_transport_plan,
            )[image_key]
            assert len(images) == 9
            for actual in images[4:]:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert client.action_schedule.qsize(-1) == 0
            assert client.latest_action == -1
            assert len(server._latency_tracker) == 3
            if client.config.rtc_mode == "trained":
                assert client._adaptive_rtt_steps is None
                assert server._trained_runtime.delay_steps == 4
    finally:
        client.stub = in_process_stub
        rpc.stop(grace=None).wait(timeout=2)


@pytest.mark.parametrize("pair", ["guided", "trained"], indirect=True)
@pytest.mark.parametrize("debug_rtc,debug_cuda", [(False, False), (True, False), (False, True), (True, True)])
def test_phase_logging_is_opt_in_but_pi05_warmup_progress_is_always_visible(
    pair, monkeypatch, caplog, debug_rtc, debug_cuda
):
    client, server, policy, _, _ = pair
    server.config.debug_rtc_diagnostics = debug_rtc
    server.config.debug_cuda_sync = debug_cuda
    enabled = debug_rtc or debug_cuda
    if not enabled:
        monkeypatch.setattr(
            server, "_logged_diagnostic_phase", lambda *a, **kw: pytest.fail("phase tracing is disabled")
        )
    with caplog.at_level(logging.DEBUG):
        assert client.start()
    assert client.rollout_state is RolloutState.READY
    assert len(policy.calls) == 9
    assert len(server._latency_tracker) == 3
    assert server._current_latency() > 0
    assert client.action_schedule.qsize(-1) == 0
    messages = [record.getMessage() for record in caplog.records]
    for name in ("ordinary", "rtc"):
        for index in (1, 2):
            prefix = f"PI05 warmup {name} {index}/2"
            assert any(message.startswith(f"{prefix} starting (prefix=") for message in messages)
            assert any(message.startswith(f"{prefix}: ") and message.endswith(" ms") for message in messages)
    assert ("RTC PHASE begin phase=worker_initialize" in caplog.text) is enabled
    assert ("RTC PHASE begin phase=calibration " in caplog.text) is enabled
    assert ("RTC PHASE end phase=calibration " in caplog.text) is enabled
    assert ("Starting receiver" in caplog.text) is enabled
    if not enabled:
        assert "RTC PHASE" not in caplog.text
    assert "RTC calibration ready: initial RTT budget=" in caplog.text


def test_eager_full_pipeline_warmup_and_calibration_share_serving_thread_without_dispatch(pair, monkeypatch):
    client, server, policy, pre, post = pair
    send_action = client.robot.send_action
    monkeypatch.setattr(client.robot, "send_action", lambda _: pytest.fail("startup dispatched an action"))
    assert not policy.config.compile_model
    assert client.start()
    assert client.rollout_state is RolloutState.READY
    assert client.action_schedule.qsize(-1) == 0
    assert client.latest_action == -1
    assert client._latest_latency_feedback is None
    assert client._adaptive_rtt_steps is not None
    assert len(server._latency_tracker) == len(server._server_latency_tracker) == 3
    assert server._last_observation_id == server._latest_received_observation_id == -1
    assert server._chunk_cache.get(0) is None
    assert server._calibration_previous is None
    assert not server._calibration_open
    # Four server warmups, two real-observation warmups, three measured round trips.
    assert len(policy.calls) == len(pre.threads) == len(post.threads) == 9
    assert policy.calls[0][1:] == (0, False)
    assert policy.calls[2][1:] == (10, True)
    assert all(isinstance(result, RemoteRTCCalibrationResult) for result in client.stub.responses)
    budget = server._current_latency()

    assert client.start_rollout()
    source = encode_rollout_observation_id(client.rollout_generation, 0)
    request = RemoteRTCObservation(
        TimedObservation(time.time(), 0, {**client.robot.get_observation(), "task": client.task}),
        source,
        -1,
    )
    server.SendObservations(send_bytes_in_chunks(pickle.dumps(request), services_pb2.Observation), None)
    result = server._action_mailbox.get(timeout=2)
    assert isinstance(result, RemoteRTCActionChunk)
    assert result.source_observation_id == source
    assert len({call[0] for call in policy.calls}) == 1
    assert set(pre.threads + post.threads) == {server._inference_thread.ident}
    assert server._inference_thread.ident != threading.get_ident()
    assert server._current_latency() == budget  # First generation does not erase calibration.
    monkeypatch.setattr(client.robot, "send_action", send_action)
    assert client.reset_rollout()
    assert client._adaptive_rtt_steps is not None


def test_warmup_failure_blocks_readiness_and_leaves_no_actions(pair):
    client, server, policy, pre, post = pair
    policy.fail = True
    assert not client.start()
    assert client.rollout_state is RolloutState.FAULT
    assert not client.start_rollout()
    assert server.failed
    assert "non-finite" in server.failure_traceback
    assert server._action_mailbox.empty()
    assert len(server._latency_tracker) == len(server._server_latency_tracker) == 0
    assert pre.reset_count and post.reset_count


def test_setup_waits_for_worker_and_start_is_blocked_until_warmup_finishes(pair):
    client, server, policy, _, _ = pair
    policy.entered, policy.release = threading.Event(), threading.Event()
    results = []
    starter = threading.Thread(target=lambda: results.append(client.start()))
    starter.start()
    try:
        assert policy.entered.wait(timeout=1)
        assert not results
        assert client.rollout_state is RolloutState.STARTING
        assert not client.start_rollout()
        assert not client.reset_rollout()
    finally:
        policy.release.set()
        starter.join(timeout=3)
    assert results == [True]
    assert server._worker_ready.is_set()


def test_startup_timeout_does_not_admit_actions_or_a_second_worker(pair):
    client, server, policy, _, _ = pair
    policy.entered, policy.release = threading.Event(), threading.Event()
    server.config.pi05_inference.warmup_timeout_s = 0.05
    try:
        assert not client.start()
        assert client.rollout_state is RolloutState.FAULT
        assert not server.running
        assert not client.start_rollout()
    finally:
        policy.release.set()
        server._inference_thread.join(timeout=2)
    assert not server._inference_thread.is_alive()
    assert server._action_mailbox.empty()


def test_stop_during_handshake_cannot_be_undone_by_a_late_success(pair):
    client, _, policy, _, _ = pair
    policy.entered, policy.release = threading.Event(), threading.Event()
    results = []
    starter = threading.Thread(target=lambda: results.append(client.start()))
    starter.start()
    try:
        assert policy.entered.wait(timeout=1)
        client.stop()
    finally:
        policy.release.set()
        starter.join(timeout=3)
    assert results == [False]
    assert client.rollout_state is RolloutState.STOPPED
    assert not client.running
    assert not client.robot.is_connected
    assert not client.stub.requests


def test_calibration_cannot_be_replayed_after_ready_or_reconnection(pair):
    client, server, _, _, _ = pair
    assert client.start()
    old = RemoteRTCCalibrationRequest(client.policy_setup.calibration_token, 0)
    for request in (old, replace(old, token="another-connection")):
        with pytest.raises(ValueError, match="out-of-session"):
            server.SendObservations(
                send_bytes_in_chunks(pickle.dumps(request), services_pb2.Observation), None
            )
    assert len(server._latency_tracker) == 3
    server._reset_server()
    assert server._current_latency() == 0
    server.Ready(services_pb2.Empty(), SimpleNamespace(peer=lambda: "new-client"))
    with pytest.raises(ValueError, match="out-of-session"):
        server.SendObservations(send_bytes_in_chunks(pickle.dumps(old), services_pb2.Observation), None)


def test_calibration_failure_blocks_start_and_never_merges_the_result(pair, monkeypatch):
    client, server, _, _, _ = pair
    monkeypatch.setattr(server, "_calibrate", lambda _: (_ for _ in ()).throw(ValueError("broken pipeline")))
    assert not client.start()
    assert client.rollout_state is RolloutState.FAULT
    assert client.action_schedule.qsize(-1) == 0
    assert client._adaptive_rtt_steps is None
    assert not client.start_rollout()


def test_latency_ttl_and_real_spikes_remain_conservative(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    tracker = _SlidingWindowLatencyTracker(10, ttl_s=5)
    for _ in range(10):
        tracker.add(0.2)
    tracker.add(0.8)
    assert tracker.quantile(0.9) == 0.2
    assert tracker.estimate(0.9) == 0.8
    tracker.add(0.2)
    assert 0.2 < tracker.estimate(0.9) < 0.8
    now[0] += 6
    assert tracker.estimate(0.9) == 0
    assert len(tracker) == 0
    tracker.add(0.3)
    assert tracker.estimate(0.9) == 0.3


def test_spike_guard_cannot_outlive_the_rtt_window():
    tracker = _SlidingWindowLatencyTracker(10)
    for _ in range(10):
        tracker.add(0.4)
    tracker.add(20.0)  # A real compilation/transport stall, not startup warmup.
    assert tracker.estimate(0.9) == 20.0
    for _ in range(10):
        tracker.add(0.4)
    assert tracker.max() == 0.4
    assert tracker.estimate(0.9) == 0.4


def test_client_timing_expires_after_idle_and_reconnection(pair, monkeypatch):
    client, server, _, _, _ = pair
    assert client.start()
    assert client.start_rollout()
    assert client.reset_rollout()
    assert client._adaptive_rtt_steps is not None
    client._rtt_updated_at -= client.config.latency_history_ttl_s + 1
    client._trigger_threshold()
    assert client._adaptive_rtt_steps is None
    # Disabling startup measurement must not retain timings from the old connection.
    with client._transport_lock:
        client._update_rtt_estimate_locked(1.0, time.monotonic())
    client.config.startup_calibration_runs = 0
    assert client.start()
    assert client._adaptive_rtt_steps is None
    assert server._current_latency() == 0


def test_reconnection_excludes_background_receiver_until_calibration_finishes(pair, monkeypatch):
    client, _, policy, _, _ = pair
    assert client.start()
    client.start_barrier = threading.Barrier(1)
    entered = threading.Event()
    release_old_poll = threading.Event()
    polled_during_startup = threading.Event()
    original = client.stub._get_actions

    def get_actions(request, **kwargs):
        if threading.current_thread() is receiver:
            if not entered.is_set():
                entered.set()
                assert release_old_poll.wait(timeout=2)
                return services_pb2.Actions()
            if client._startup_in_progress.is_set():
                polled_during_startup.set()
        return original(request, **kwargs)

    monkeypatch.setattr(client.stub, "_get_actions", get_actions)
    receiver = threading.Thread(target=client.receive_actions)
    receiver.start()
    results = []
    policy.entered, policy.release = threading.Event(), threading.Event()
    starter = threading.Thread(target=lambda: results.append(client.start()))
    try:
        assert entered.wait(timeout=1)
        starter.start()
        assert client._startup_in_progress.wait(timeout=1)
        release_old_poll.set()
        assert policy.entered.wait(timeout=1)
        assert not polled_during_startup.wait(timeout=0.1)
        policy.release.set()
        starter.join(timeout=3)
        assert results == [True]
        assert client.rollout_state is RolloutState.READY
    finally:
        release_old_poll.set()
        policy.release.set()
        client.shutdown_event.set()
        starter.join(timeout=3)
        receiver.join(timeout=2)
    assert not starter.is_alive() and not receiver.is_alive()
