# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers")
from transformers import utils as transformers_utils  # noqa: E402

from lerobot.policies.pi05 import modeling_pi05  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402


class _LoadOnlyPI05(modeling_pi05.PI05Policy):
    """Use the real checkpoint loader without allocating a multi-billion-parameter model."""

    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.model = torch.nn.Linear(2, 2, bias=False, dtype=dtype)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "shape", "keys"])
def test_bad_checkpoint_never_returns_a_random_or_partially_loaded_policy(tmp_path, failure):
    path = tmp_path / "model.safetensors"
    if failure == "corrupt":
        path.write_bytes(b"invalid safetensors file")
    elif failure == "shape":
        save_file({"model.weight": torch.ones(3, 3)}, path)
    elif failure == "keys":
        save_file({"model.unexpected": torch.ones(2, 2)}, path)
    with pytest.raises((RuntimeError, OSError, ValueError)):
        _LoadOnlyPI05.from_pretrained(tmp_path, config=PI05Config(device="cpu"), local_files_only=True)


@pytest.mark.parametrize("saved_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_checkpoint_resolution_uses_named_options_and_preserves_requested_dtype(
    tmp_path, monkeypatch, saved_dtype, dtype
):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.full((2, 2), 0.25, dtype=saved_dtype)}, path)
    received = {}

    def resolve(reference, filename, **kwargs):
        received.update(kwargs)
        return str(path)

    monkeypatch.setattr(transformers_utils, "cached_file", resolve)
    monkeypatch.setattr(modeling_pi05, "cached_file", resolve, raising=False)
    model = _LoadOnlyPI05.from_pretrained(
        tmp_path,
        config=PI05Config(device="cpu", dtype=dtype, precision_profile="checkpoint"),
        revision="pinned-revision",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        token="test-token",
        force_download=True,
        resume_download=False,
        proxies={"https": "test-proxy"},
    )
    assert received == {
        "revision": "pinned-revision",
        "cache_dir": tmp_path / "cache",
        "local_files_only": True,
        "token": "test-token",
        "force_download": True,
        "resume_download": False,
        "proxies": {"https": "test-proxy"},
    }
    assert model.model.weight.dtype == getattr(torch, dtype)
    torch.testing.assert_close(model.model.weight, torch.full((2, 2), 0.25, dtype=getattr(torch, dtype)))
