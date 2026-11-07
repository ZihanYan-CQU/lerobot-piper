"""Single-inspection startup, immutable action coordinates and final receipt handling."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import lerobot.rl.rlt.features as feature_module
from lerobot.async_inference.server.rlt_runtime import RemoteRLTServer
from lerobot.async_inference.shared.helpers import RemotePolicyConfig
from lerobot.async_inference.shared.rlt_config import RLTServerConfig
from lerobot.async_inference.shared.rlt_remote import RLTFinished, RLTRequest, RLTSetup
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.rlt_actor import RLTActorPolicy
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.rl.rlt.features import FrozenRLTFeatures, checkpoint_fingerprint
from tests.policies.pi0_pi05.test_pi05_rlinf import tiny_networks as tiny_networks
from tests.policies.pi0_pi05.test_pi05_rlt_stage2 import actor_config, adapter, raw_batch
from tests.rl.test_rlt_rtc import SESSION, Coordinates, next_request, planner


@pytest.mark.parametrize("relative", [False, True])
def test_window_action_round_trip_with_owned_coordinate_snapshots(tiny_networks, relative):
    features = adapter()
    if not relative:
        for step in [*features.preprocessor.steps, *features.postprocessor.steps]:
            if isinstance(step, (AbsoluteActionsProcessorStep, RelativeActionsProcessorStep)):
                step.enabled = False
        features = FrozenRLTFeatures(
            features.policy, features.preprocessor, features.postprocessor, features.config
        )
    features.encode(raw_batch(5))
    coordinates = features.snapshot_coordinates()
    target = torch.tensor([[-0.2, 0.1, 0.3], [0.2, -0.1, 0.5]])
    absolute = features.executable_actions(target)
    torch.testing.assert_close(absolute, coordinates.decode(target), rtol=0, atol=0)
    for sent, normalized in zip(absolute, target, strict=True):
        torch.testing.assert_close(features.normalize_executed(sent), normalized, rtol=1e-5, atol=1e-6)
    features.encode(raw_batch(10))
    torch.testing.assert_close(coordinates.decode(target), absolute, rtol=0, atol=0)
    features.reset()
    with pytest.raises(RuntimeError, match="coordinates"):
        features.normalize_executed(absolute[0])
    with pytest.raises(RuntimeError, match="Encode"):
        features.executable_actions(target)


@pytest.mark.parametrize("truncated", [False, True])
def test_final_receipt_keeps_bootstrap_state_without_decoding_a_new_plan(tmp_path, monkeypatch, truncated):
    owner, features, replay = planner(tmp_path)
    first = owner.process(RLTRequest(SESSION, 1, 0, 0, {"value": 0}))

    def reject_decode(*args):
        raise AssertionError("An ended episode must not construct another motor plan")

    monkeypatch.setattr(Coordinates, "decode", reject_decode)
    result = owner.process(next_request(first, count=2, terminated=not truncated, truncated=truncated))
    assert isinstance(result, RLTFinished) and owner.pending is None
    assert len(replay) == 1 and features.encoded == [0, 2]
    row = replay._data[0]
    assert row.valid_steps == 2 and row.truncated == truncated and row.terminated != truncated
    assert row.action[2:].count_nonzero() == 0
    # Even truncation retains the actual next-state/reference for its TD bootstrap.
    torch.testing.assert_close(row.next_state["observation.rlt"], torch.full((4,), 0.002))
    assert len(list(tmp_path.glob("*.pt"))) == 1


def save_stage1(directory):
    original = adapter()
    original.policy.save_pretrained(directory)
    original.preprocessor.save_pretrained(directory)
    original.postprocessor.save_pretrained(directory)


def deployment_setup(saved, tmp_path, *, actor_checkpoint=None, warmstart=False, names=None):
    cfg = SimpleNamespace(
        rlt=RLTServerConfig(
            enabled=True,
            execution_horizon=2,
            prefix_steps=1,
            replay_dir=str(tmp_path),
            actor_checkpoint=str(actor_checkpoint) if actor_checkpoint else None,
            warmstart_stage2_actor=warmstart,
        ),
        fps=30,
        pi05_inference=SimpleNamespace(compile_model=False, compile_mode="default"),
    )
    server = SimpleNamespace(prefix="policy_server_deployment", config=cfg)
    setup = RLTSetup(
        SESSION,
        RemotePolicyConfig("pi05", str(saved), {}, 4, "cpu"),
        names or ("joint_1", "joint_2", "gripper"),
        cfg.fps,
        execution_horizon=2,
        prefix_steps=1,
        client_image_resize=False,
    )
    return server, setup


@pytest.mark.parametrize("actor_mode", ["none", "rtc", "stage2"])
def test_remote_startup_inspects_weights_once_and_preserves_actor_loading(
    tiny_networks, tmp_path, monkeypatch, actor_mode
):
    saved, actor_path = tmp_path / "stage1", tmp_path / "actor"
    save_stage1(saved)
    source_actor = None
    if actor_mode != "none":
        cfg = replace(
            actor_config(),
            feature_model_id=checkpoint_fingerprint(saved),
            rtc_prefix_steps=int(actor_mode == "rtc"),
            input_features={},
            hidden_dims=(8,),
        )
        source_actor = RLTActorPolicy(cfg)
        source_actor.save_pretrained(actor_path)
    calls = []
    fingerprint = feature_module.checkpoint_fingerprint

    def tracked(path, **kwargs):
        calls.append(str(path))
        return fingerprint(path, **kwargs)

    monkeypatch.setattr(feature_module, "checkpoint_fingerprint", tracked)
    server, setup = deployment_setup(
        saved,
        tmp_path,
        actor_checkpoint=actor_path if source_actor else None,
        warmstart=actor_mode == "stage2",
    )
    runtime = RemoteRLTServer(server, setup, None)
    try:
        assert calls == [str(saved)]
        assert runtime.features.config.feature_model_id == fingerprint(saved)
        if source_actor is not None:
            state = runtime.features.encode(raw_batch())
            state = {key: value[None] for key, value in state.items()}
            torch.testing.assert_close(
                runtime.planner.actor.predict_action_chunk(state),
                source_actor.predict_action_chunk(state),
                rtol=1e-5,
                atol=1e-6,
            )
    finally:
        runtime.close()


@pytest.mark.parametrize("bad", ["action_order", "actor_identity", "factory_identity"])
def test_single_inspection_still_rejects_incompatible_assets_before_model_load(
    tiny_networks, tmp_path, monkeypatch, bad
):
    saved = tmp_path / "stage1"
    save_stage1(saved)

    def reject_model_load(*args, **kwargs):
        raise AssertionError("Invalid model/action identities must fail before allocating the VLA")

    monkeypatch.setattr(PI05Policy, "from_pretrained", reject_model_load)
    if bad == "factory_identity":
        with pytest.raises(ValueError, match="weights/processors"):
            FrozenRLTFeatures.from_pretrained(
                saved, config_factory=lambda source, identity: actor_config(), device="cpu"
            )
        return
    actor_path = None
    names = ("joint_2", "joint_1", "gripper") if bad == "action_order" else None
    if bad == "actor_identity":
        actor_path = tmp_path / "actor"
        RLTActorPolicy(actor_config()).save_pretrained(actor_path)
    server, setup = deployment_setup(saved, tmp_path, actor_checkpoint=actor_path, names=names)
    with pytest.raises(ValueError, match="action order|different frozen"):
        RemoteRLTServer(server, setup, None)
