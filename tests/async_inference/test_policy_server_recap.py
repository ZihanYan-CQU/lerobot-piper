# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.async_inference.server.policy_server_dagger import PolicyServerDagger
from lerobot.async_inference.server.policy_server_recap import (
    PolicyServerRecap,
    PolicyServerRecapConfig,
    checkpoint_digest,
    checkpoint_files,
)
from lerobot.async_inference.shared.helpers import RemotePolicyConfig
from lerobot.async_inference.shared.metadata_recap import IDENTITY_METADATA_KEY, validate_policy_identity


def _checkpoint(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"test weights")
    (tmp_path / "config.json").write_text('{"policy": "smolvla"}')
    (tmp_path / "policy_preprocessor.json").write_text('{"mean": 0}')
    return tmp_path


def test_fingerprint_tracks_same_path_weight_and_processor_updates(tmp_path):
    root = _checkpoint(tmp_path)
    first = checkpoint_digest(root, checkpoint_files(root))
    (root / "model.safetensors").write_bytes(b"new weights")
    second = checkpoint_digest(root, checkpoint_files(root))
    (root / "policy_preprocessor.json").write_text('{"mean": 1}')
    third = checkpoint_digest(root, checkpoint_files(root))
    assert len({first, second, third}) == 3


def test_recap_server_loads_exact_resolved_checkpoint_and_returns_identity(tmp_path, monkeypatch):
    root = _checkpoint(tmp_path)
    server = PolicyServerRecap(
        PolicyServerRecapConfig(host="localhost", port=9999, recap_policy_version="iter-3")
    )
    loaded = []
    monkeypatch.setattr(
        PolicyServerDagger, "_load_or_reuse_policy", lambda _, spec: loaded.append(spec) or False
    )
    spec = RemotePolicyConfig("smolvla", str(root), {}, 8)
    assert server._load_or_reuse_policy(spec) is False
    identity = validate_policy_identity(server.recap_policy_identity)
    assert loaded[0].pretrained_name_or_path == str(root.resolve())
    assert identity["label"] == "iter-3"
    assert identity["requested_path"] == str(root)
    first_load = identity["server_load_id"]
    server._load_or_reuse_policy(spec)
    assert server.recap_policy_identity["server_load_id"] != first_load
    metadata = []
    context = SimpleNamespace(set_trailing_metadata=metadata.extend)
    monkeypatch.setattr(PolicyServerDagger, "SendPolicyInstructions", lambda *_: "response")
    assert server.SendPolicyInstructions(None, context) == "response"
    assert (
        validate_policy_identity(json.loads(dict(metadata)[IDENTITY_METADATA_KEY]))
        == server.recap_policy_identity
    )


def test_server_rejects_checkpoint_mutation_during_load(tmp_path, monkeypatch):
    root = _checkpoint(tmp_path)
    server = PolicyServerRecap(PolicyServerRecapConfig(host="localhost", port=9999))

    def mutate(_, spec):
        (Path(spec.pretrained_name_or_path) / "model.safetensors").write_bytes(b"changed while loading")
        return False

    monkeypatch.setattr(PolicyServerDagger, "_load_or_reuse_policy", mutate)
    with pytest.raises(RuntimeError, match="changed"):
        server._load_or_reuse_policy(RemotePolicyConfig("smolvla", str(root), {}, 8))
    assert server.recap_policy_identity is None


def test_recap_forbids_warm_weight_reuse_but_dagger_defaults_unchanged():
    with pytest.raises(ValueError, match="reuse_last_policy"):
        PolicyServerRecapConfig(host="localhost", port=9999, reuse_last_policy=True)


def test_hub_revision_resolves_once_and_loads_the_hashed_snapshot(tmp_path, monkeypatch):
    import lerobot.async_inference.server.policy_server_recap as server_module

    root = _checkpoint(tmp_path)
    requests = []
    loaded = []

    def snapshot(repo_id, **kwargs):
        requests.append((repo_id, kwargs))
        return str(root)

    monkeypatch.setattr(server_module, "snapshot_download", snapshot)
    monkeypatch.setattr(
        PolicyServerDagger, "_load_or_reuse_policy", lambda _, spec: loaded.append(spec) or False
    )
    server = PolicyServerRecap(
        PolicyServerRecapConfig(host="localhost", port=9999, recap_model_revision="v3")
    )
    server._load_or_reuse_policy(RemotePolicyConfig("smolvla", "test/hub-policy", {}, 8))
    assert requests[0][0] == "test/hub-policy"
    assert requests[0][1]["revision"] == "v3"
    assert len(requests) == 1
    assert loaded[0].pretrained_name_or_path == str(root.resolve())
    assert server.recap_policy_identity["hub_commit"] == root.name
    assert server.recap_policy_identity["requested_path"] == "test/hub-policy"
