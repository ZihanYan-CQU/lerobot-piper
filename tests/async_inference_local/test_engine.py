import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.async_inference_local.inference import engine as engine_module
from lerobot.async_inference_local.inference.engine import LocalPolicyEngine
from lerobot.async_inference_local.shared.types import ActionChunk, Observation, PlanSlice, Prefix, Request
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import RelativeActionsProcessorStep, create_transition, to_absolute_actions


def engine_fixture(*, trained=False, bad_prefix=False):
    engine = LocalPolicyEngine.__new__(LocalPolicyEngine)
    engine.config = SimpleNamespace(
        inference=SimpleNamespace(compile_model=False, warmup_runs=2),
        rtc=RTCConfig(mode="trained" if trained else "guided", execution_horizon=4),
        actions_per_chunk=6,
        fps=30,
        latency_margin_steps=1,
        task="pick",
    )
    engine.device = torch.device("cpu")
    engine.relative = RelativeActionsProcessorStep(enabled=True)
    engine.prefix_normalizer = None
    engine.trained_delay = 2
    engine.action_dim = 1
    calls = []
    output = torch.ones(1, 6, 1)

    class Processor:
        def __call__(self, batch):
            engine.relative(create_transition(observation=batch))
            return batch

        def reset(self):
            pass

    class Postprocessor:
        def __call__(self, value):
            return to_absolute_actions(value, engine.relative.get_cached_state(), [True])

        def reset(self):
            pass

    class Policy:
        rtc_processor = None

        def predict_action_chunk(self, batch, **kwargs):
            calls.append(kwargs)
            output.fill_(1)
            previous = kwargs["prev_chunk_left_over"]
            if trained and previous is not None and not bad_prefix:
                delay = kwargs["inference_delay"]
                output[0, :delay] = previous[:delay]
            return output

        def reset(self):
            pass

    engine.preprocessor = Processor()
    engine.postprocessor = Postprocessor()
    engine.policy = Policy()
    engine.prepare = lambda values, task: {"observation.state": torch.tensor([[values["joint.pos"]]])}
    return engine, calls, output


def request_with_prefix():
    raw = torch.ones(4, 1)
    absolute = raw + 10
    chunk = ActionChunk(raw, absolute, absolute.numpy(), time.monotonic())
    return Request(
        1,
        Observation(1, time.monotonic(), time.monotonic(), {"joint.pos": 20.0}),
        Prefix(0, (PlanSlice(chunk, 0, 4),)),
        "pick",
        time.monotonic(),
    )


def test_relative_prefix_reanchors_on_current_state_and_output_owns_model_buffers():
    engine, calls, output = engine_fixture()
    prediction = engine.predict(request_with_prefix(), 0.05)
    torch.testing.assert_close(calls[0]["prev_chunk_left_over"], torch.full((4, 1), -9.0))
    assert calls[0]["inference_delay"] == 3
    np.testing.assert_array_equal(prediction.chunk.cpu, np.full((6, 1), 21.0))
    output.fill_(999)
    torch.testing.assert_close(prediction.chunk.model, torch.ones(6, 1))
    np.testing.assert_array_equal(prediction.chunk.cpu, np.full((6, 1), 21.0))
    assert not prediction.chunk.cpu.flags.writeable


def test_trained_prefix_is_validated_after_unnormalization_and_reanchoring():
    engine, calls, _ = engine_fixture(trained=True)
    prediction = engine.predict(request_with_prefix(), 0.05)
    assert calls[0]["inference_delay"] == 2
    np.testing.assert_array_equal(prediction.chunk.cpu[:2], np.full((2, 1), 11.0))
    broken, _, _ = engine_fixture(trained=True, bad_prefix=True)
    with pytest.raises(ValueError, match="hard prefix"):
        broken.predict(request_with_prefix(), 0.05)


def test_warmup_covers_both_call_shapes_without_consuming_rng_or_dispatching():
    engine, calls, _ = engine_fixture()
    before = torch.random.get_rng_state().clone()
    samples = engine.warmup(request_with_prefix().observation, lambda: False)
    assert len(samples) == 2
    assert len(calls) == 4
    assert all(call["prev_chunk_left_over"] is None for call in calls[:2])
    assert all(call["prev_chunk_left_over"] is not None for call in calls[2:])
    assert torch.equal(before, torch.random.get_rng_state())


def test_warmup_cancellation_propagates_without_prediction():
    engine, calls, _ = engine_fixture()
    with pytest.raises(RuntimeError, match="cancelled"):
        engine.warmup(request_with_prefix().observation, lambda: True)
    assert not calls


def test_cuda_alias_is_resolved_before_selecting_device(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    def select(device):
        assert device == torch.device("cuda:1")
        selected.append(device)

    def stop_before_loading(_):
        raise RuntimeError("checkpoint sentinel")

    monkeypatch.setattr(torch.cuda, "set_device", select)
    monkeypatch.setattr(engine_module, "get_policy_class", stop_before_loading)
    with pytest.raises(RuntimeError, match="checkpoint sentinel"):
        LocalPolicyEngine(SimpleNamespace(inference=SimpleNamespace(device="cuda")), {}, [])
    assert selected == [torch.device("cuda:1")]
