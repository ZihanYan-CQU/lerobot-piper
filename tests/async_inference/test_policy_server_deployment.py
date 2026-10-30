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

"""Tests for the consolidated deployment policy server."""

import logging
import pickle  # nosec
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import lerobot.async_inference.server.policy_server_deployment as deployment_server_module
from lerobot.async_inference.server.policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
    _LightweightGuidedRTCProcessor,
    _RTCServerConfig,
    _SlidingWindowLatencyTracker,
)
from lerobot.async_inference.shared.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.shared.rtc_remote import (
    DEPLOYMENT_RTC_PROTOCOL_VERSION,
    RemoteRTCActionChunk,
    RemoteRTCObservation,
    RemoteRTCPolicySetup,
    RemoteRTCRetry,
    RTCActionRef,
    encode_rollout_observation_id,
)
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc import RTCConfig, RTCProcessor
from lerobot.transport import services_pb2
from lerobot.transport.utils import send_bytes_in_chunks


class _ResetCounter:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _PolicyResetCounter(_ResetCounter):
    rtc_processor = None


class _Context:
    @staticmethod
    def peer():
        return "test-client"


class _FakePipeline(_ResetCounter):
    steps = []

    def __call__(self, value):
        return value


class _FakePolicy(_ResetCounter):
    def __init__(self, checkpoint: str):
        super().__init__()
        self.checkpoint = checkpoint
        self.device = None
        self.rtc_processor = None
        self.config = type(
            "Config",
            (),
            {
                "action_feature_names": ["joint.pos"],
                "image_features": {},
                "rtc_training_max_delay": 0,
                "rtc_config": None,
            },
        )()

    @classmethod
    def from_pretrained(cls, checkpoint):
        return cls(checkpoint)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def init_rtc_processor(self):
        return None

    def supports_rtc(self):
        return True

    def predict_action_chunk(self, observation, *, inference_delay, prev_chunk_left_over):
        del observation, inference_delay, prev_chunk_left_over
        return torch.zeros(1, 20, 1)


def _policy_setup(
    checkpoint: str = "checkpoint-a",
    *,
    actions_per_chunk: int = 20,
    rename_map: dict[str, str] | None = None,
) -> services_pb2.PolicySetup:
    setup = RemoteRTCPolicySetup(
        policy=RemotePolicyConfig(
            policy_type="smolvla",
            pretrained_name_or_path=checkpoint,
            lerobot_features={},
            actions_per_chunk=actions_per_chunk,
            device="cpu",
            rename_map={} if rename_map is None else rename_map,
        ),
        control_frequency_hz=30,
        action_feature_names=("joint.pos",),
        protocol_version=DEPLOYMENT_RTC_PROTOCOL_VERSION,
    )
    return services_pb2.PolicySetup(data=pickle.dumps(setup))


@pytest.mark.parametrize("active,remaining", [(False, 5.0), (True, 0.0)])
def test_expired_action_poll_does_not_consume_a_pending_result(active, remaining):
    server = PolicyServerDeployment(PolicyServerDeploymentConfig())
    result = RemoteRTCRetry(1, time.time(), "inference_error")
    context = SimpleNamespace(is_active=lambda: active, time_remaining=lambda: remaining)
    try:
        server._action_mailbox.put(result)
        assert not server.GetActions(services_pb2.Empty(), context).data
        assert pickle.loads(server.GetActions(services_pb2.Empty(), None).data) == result  # nosec
    finally:
        server.stop()


def test_action_poll_stops_waiting_at_client_deadline():
    server = PolicyServerDeployment(PolicyServerDeploymentConfig(action_wait_timeout=5))
    deadline = time.monotonic() + 0.05
    context = SimpleNamespace(
        is_active=lambda: time.monotonic() < deadline,
        time_remaining=lambda: max(0.0, deadline - time.monotonic()),
    )
    responses = []
    worker = threading.Thread(
        target=lambda: responses.append(server.GetActions(services_pb2.Empty(), context))
    )
    worker.start()
    try:
        worker.join(timeout=1)
        assert not worker.is_alive(), "expired RPC remained in the server's mailbox wait"
        assert len(responses) == 1 and not responses[0].data
        result = RemoteRTCRetry(2, time.time(), "inference_error")
        server._action_mailbox.put(result)
        assert pickle.loads(server.GetActions(services_pb2.Empty(), None).data) == result  # nosec
    finally:
        if worker.is_alive():
            server._action_mailbox.put_nowait(RemoteRTCRetry(0, 0.0, "test_cleanup"))
        worker.join(timeout=1)
        server.stop()


def test_deployment_defaults_to_paper_guidance_hyperparameters():
    config = PolicyServerDeploymentConfig(host="localhost", port=9999)
    assert config.rtc.prefix_attention_schedule is RTCAttentionSchedule.EXP
    assert config.rtc.max_guidance_weight == 5.0


@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("prefix_length,delay", [(1, 0), (3, 1), (3, 3)])
def test_lightweight_guidance_preserves_native_correction_without_parameter_graph(
    schedule, prefix_length, delay
):
    config = RTCConfig(prefix_attention_schedule=schedule, execution_horizon=3)
    native = RTCProcessor(config)
    optimized = _LightweightGuidedRTCProcessor(config)
    weight = torch.nn.Parameter(torch.tensor(0.5))
    x_t = torch.ones(1, 4, 2)
    previous = torch.full((prefix_length, 1), 0.1)
    grad_modes = []

    def denoise(value):
        grad_modes.append(torch.is_grad_enabled())
        return value * weight

    for time_value in [1.0, 0.5, 0.1]:
        kwargs = {
            "x_t": x_t,
            "prev_chunk_left_over": previous,
            "inference_delay": delay,
            "time": time_value,
            "original_denoise_step_partial": denoise,
        }
        with torch.no_grad():
            expected = native.denoise_step(**kwargs)
            actual = optimized.denoise_step(**kwargs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert grad_modes == [True, False] * 3
    assert weight.grad is None


@pytest.mark.parametrize("fast", [False, True])
def test_deployment_uses_lightweight_guidance_by_default(fast):
    class Policy:
        def __init__(self):
            self.config = type("Config", (), {"rtc_config": None, "rtc_training_max_delay": 0})()
            self.rtc_processor = None
            self.model = type("Model", (), {"rtc_processor": None})()

        def init_rtc_processor(self):
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            self.model.rtc_processor = self.rtc_processor

    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(host="localhost", port=9999, rtc_fast_guidance=fast)
    )
    server.policy = Policy()

    server._configure_policy_rtc()

    assert type(server.policy.rtc_processor) is _LightweightGuidedRTCProcessor
    assert server.policy.rtc_processor.fast is fast
    assert server.policy.model.rtc_processor is server.policy.rtc_processor
    server.stop()


def test_rtc_delay_max_forgets_samples_evicted_from_finite_buffer():
    tracker = _SlidingWindowLatencyTracker(maxlen=2)
    tracker.add(0.8)
    tracker.add(0.1)
    assert tracker.max() == 0.8

    tracker.add(0.2)

    assert tracker.max() == 0.2


def test_rtc_quantile_covers_cold_start_but_recovers_from_an_isolated_spike():
    tracker = _SlidingWindowLatencyTracker(maxlen=10)
    tracker.add(2.0)
    for _ in range(8):
        tracker.add(0.2)
    assert tracker.quantile(0.9) == 2.0  # Too few samples to discard the tail yet.
    tracker.add(0.2)
    assert tracker.quantile(0.9) == 0.2
    assert tracker.quantile(1.0) == 2.0
    # A sustained latency increase must raise the estimate, unlike a lone tail.
    tracker.add(0.8)
    tracker.add(0.8)
    assert tracker.quantile(0.9) == 0.8
    for _ in range(10):
        tracker.add(0.2)
    assert tracker.quantile(0.9) == 0.2


def test_rtc_latency_window_ignores_nonfinite_and_negative_samples():
    tracker = _SlidingWindowLatencyTracker(maxlen=2)
    tracker.add(0.2)
    for value in (float("nan"), float("inf"), -1.0):
        tracker.add(value)
    assert len(tracker) == 1
    assert tracker.quantile(0.9) == 0.2


def test_server_rtt_window_counts_each_request_once_and_survives_rollout_reset():
    server = PolicyServerDeployment(PolicyServerDeploymentConfig(action_wait_timeout=0))
    try:
        server._begin_rollout_generation(0)
        server._record_latency(0.1)
        assert server._current_latency() == 0.1  # Server-only bootstrap.
        for source in range(10):
            server._record_latency(0.01)
            envelope = RemoteRTCObservation(
                observation=TimedObservation(time.time(), 0, {}),
                observation_id=source + 1,
                executed_action_step=-1,
                latency_feedback_seconds=2.0 if source == 0 else 0.2,
                latency_feedback_source_id=source,
            )
            server.SendObservations(
                send_bytes_in_chunks(pickle.dumps(envelope), services_pb2.Observation), None
            )
        assert len(server._latency_tracker) == 10
        assert server._latency_tracker.quantile(0.9) == 0.2
        budget = server._current_latency()
        assert 0.2 < budget < 2.0  # A real spike decays gradually, independently of P90.
        server._record_latency(4.0)
        assert server._current_latency() == budget  # Do not mix server-only samples into RTT.

        duplicate_feedback = RemoteRTCObservation(
            observation=TimedObservation(time.time(), 0, {}),
            observation_id=11,
            executed_action_step=-1,
            latency_feedback_seconds=9.0,
            latency_feedback_source_id=9,
        )
        server.SendObservations(
            send_bytes_in_chunks(pickle.dumps(duplicate_feedback), services_pb2.Observation), None
        )
        assert len(server._latency_tracker) == 10
        assert server._current_latency() == budget
        server._begin_rollout_generation(1)
        assert server._current_latency() == budget
        assert len(server._latency_tracker) == len(server._server_latency_tracker) == 10
        server._reset_server()  # Reconnection invalidates even a warm model's old timings.
        assert server._current_latency() == 0.0
        assert len(server._latency_tracker) == len(server._server_latency_tracker) == 0
    finally:
        server.stop()


@pytest.mark.parametrize("quantile", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_guided_delay_quantile_is_rejected(quantile):
    with pytest.raises(ValueError, match="rtc_delay_quantile"):
        PolicyServerDeploymentConfig(rtc_delay_quantile=quantile)


def test_remote_guidance_does_not_hold_pad_a_short_real_overlap():
    class RecordingPolicy(_FakePolicy):
        def __init__(self):
            super().__init__("checkpoint-a")
            self.received_prefix = None

        def predict_action_chunk(self, observation, *, inference_delay, prev_chunk_left_over):
            del observation, inference_delay
            self.received_prefix = prev_chunk_left_over.detach().clone()
            return torch.zeros(1, 20, 1)

    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(host="localhost", port=9999, action_wait_timeout=0)
    )
    policy = RecordingPolicy()
    server.policy = policy
    server.preprocessor = _FakePipeline()
    server.postprocessor = _FakePipeline()
    server.device = "cpu"
    server.actions_per_chunk = 20
    server.lerobot_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["joint.pos"],
        }
    }

    real_overlap = torch.tensor([[1.0], [2.0], [3.0]])
    server._active_rollout_generation = 0
    server._chunk_cache.put(7, real_overlap, real_overlap)
    envelope = RemoteRTCObservation(
        observation=TimedObservation(time.time(), 0, {"joint.pos": 0.0}),
        observation_id=8,
        executed_action_step=-1,
        action_refs=tuple(RTCActionRef(7, index) for index in range(len(real_overlap))),
    )

    chunk = server._predict_remote_chunk(envelope)

    assert chunk is not None
    assert chunk.prefix_length == len(real_overlap)
    torch.testing.assert_close(policy.received_prefix, real_overlap)
    server.stop()


def test_new_generation_resets_server_episode_state_without_unloading_policy():
    server = PolicyServerDeployment(_RTCServerConfig(host="localhost", port=9999, action_wait_timeout=0))
    preprocessor = _ResetCounter()
    postprocessor = _ResetCounter()
    policy = _PolicyResetCounter()
    server.preprocessor = preprocessor
    server.postprocessor = postprocessor
    server.policy = policy

    server._chunk_cache.put(1, torch.ones(2, 3), torch.ones(2, 3))
    server._record_latency(0.2)
    server._last_latency_feedback_source = 7
    server._action_mailbox.put_nowait(
        RemoteRTCActionChunk(
            source_observation_id=1,
            chunk_start_step=0,
            observation_timestamp=time.time(),
            actions=[TimedAction(time.time(), 0, torch.zeros(3))],
        )
    )

    server._begin_rollout_generation(4)

    assert server.active_rollout_generation == 4
    assert server._chunk_cache.get(1) is None
    assert server._current_latency() == 0.2
    assert server._last_latency_feedback_source == 7
    assert server._action_mailbox.empty()
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1
    assert policy.reset_count == 1

    # More observations in the same rollout do not reset hidden state again.
    server._begin_rollout_generation(4)
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1
    assert policy.reset_count == 1
    assert server.policy is policy
    server.stop()


def test_stale_generation_observation_cannot_repopulate_new_latency_tracker():
    server = PolicyServerDeployment(_RTCServerConfig(host="localhost", port=9999, action_wait_timeout=0))
    new_id = encode_rollout_observation_id(4, 0)
    old_id = encode_rollout_observation_id(2, 99)
    server._latest_received_observation_id = new_id
    server._record_latency(0.1)

    stale = RemoteRTCObservation(
        observation=TimedObservation(time.time(), 0, {}),
        observation_id=old_id,
        executed_action_step=-1,
        latency_feedback_seconds=5.0,
        latency_feedback_source_id=old_id,
    )
    request = send_bytes_in_chunks(
        pickle.dumps(stale),
        services_pb2.Observation,
        silent=True,
    )

    server.SendObservations(request, None)

    assert server._current_latency() == 0.1
    assert server._last_latency_feedback_source == -1
    assert server._observation_mailbox.empty()
    server.stop()


def test_repeated_guided_rtt_overrun_warns_that_execution_horizon_is_too_short(caplog):
    config = PolicyServerDeploymentConfig(host="localhost", port=9999, action_wait_timeout=0)
    config.rtc.execution_horizon = 20
    server = PolicyServerDeployment(config)
    server.actions_per_chunk = 50

    with caplog.at_level(logging.WARNING):
        server._check_execution_horizon_feedback(1, 24 / 30)
        assert "guided conditioning is being prefix-limited" not in caplog.text
        server._check_execution_horizon_feedback(1, 25 / 30)

    assert "execution_horizon=20" in caplog.text
    assert "measured RTT=25 steps" in caplog.text
    assert "--rtc.execution_horizon=30" in caplog.text
    server.stop()


def test_reconnecting_same_model_reuses_warm_weights(monkeypatch):
    model_loads = []
    processor_builds = []

    class RecordingPolicy(_FakePolicy):
        @classmethod
        def from_pretrained(cls, checkpoint):
            model_loads.append(checkpoint)
            return cls(checkpoint)

    def build_processors(*args, **kwargs):
        processor_builds.append((args, kwargs))
        return _FakePipeline(), _FakePipeline()

    monkeypatch.setattr(deployment_server_module, "get_policy_class", lambda policy_type: RecordingPolicy)
    monkeypatch.setattr(deployment_server_module, "make_pre_post_processors", build_processors)
    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(host="localhost", port=9999, action_wait_timeout=0)
    )

    server.SendPolicyInstructions(_policy_setup(actions_per_chunk=20), _Context())
    first_policy = server.policy
    server.Ready(services_pb2.Empty(), _Context())
    server.SendPolicyInstructions(_policy_setup(actions_per_chunk=24), _Context())

    assert server.policy is first_policy
    assert server.actions_per_chunk == 24
    assert model_loads == ["checkpoint-a"]
    assert len(processor_builds) == 1
    assert server.model_loads == 1
    assert server.model_cache_hits == 1
    server.stop()


def test_cache_rebuilds_processors_for_new_rename_map_and_reloads_different_model(monkeypatch):
    model_loads = []
    processor_builds = []

    class RecordingPolicy(_FakePolicy):
        @classmethod
        def from_pretrained(cls, checkpoint):
            model_loads.append(checkpoint)
            return cls(checkpoint)

    def build_processors(*args, **kwargs):
        processor_builds.append((args, kwargs))
        return _FakePipeline(), _FakePipeline()

    monkeypatch.setattr(deployment_server_module, "get_policy_class", lambda policy_type: RecordingPolicy)
    monkeypatch.setattr(deployment_server_module, "make_pre_post_processors", build_processors)
    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(host="localhost", port=9999, action_wait_timeout=0)
    )

    server.SendPolicyInstructions(_policy_setup(), _Context())
    first_policy = server.policy
    server.Ready(services_pb2.Empty(), _Context())
    server.SendPolicyInstructions(_policy_setup(rename_map={"camera": "wrist"}), _Context())

    assert server.policy is first_policy
    assert model_loads == ["checkpoint-a"]
    assert len(processor_builds) == 2

    server.Ready(services_pb2.Empty(), _Context())
    server.SendPolicyInstructions(_policy_setup(checkpoint="checkpoint-b"), _Context())

    assert server.policy is not first_policy
    assert model_loads == ["checkpoint-a", "checkpoint-b"]
    assert len(processor_builds) == 3
    assert server.model_loads == 2
    assert server.model_cache_hits == 1
    server.stop()


def test_warm_policy_reuse_can_be_disabled(monkeypatch):
    model_loads = []

    class RecordingPolicy(_FakePolicy):
        @classmethod
        def from_pretrained(cls, checkpoint):
            model_loads.append(checkpoint)
            return cls(checkpoint)

    monkeypatch.setattr(deployment_server_module, "get_policy_class", lambda policy_type: RecordingPolicy)
    monkeypatch.setattr(
        deployment_server_module,
        "make_pre_post_processors",
        lambda *args, **kwargs: (_FakePipeline(), _FakePipeline()),
    )
    server = PolicyServerDeployment(
        PolicyServerDeploymentConfig(
            host="localhost",
            port=9999,
            action_wait_timeout=0,
            reuse_last_policy=False,
        )
    )

    server.SendPolicyInstructions(_policy_setup(), _Context())
    server.Ready(services_pb2.Empty(), _Context())
    server.SendPolicyInstructions(_policy_setup(), _Context())

    assert model_loads == ["checkpoint-a", "checkpoint-a"]
    assert server.model_loads == 2
    assert server.model_cache_hits == 0
    server.stop()
