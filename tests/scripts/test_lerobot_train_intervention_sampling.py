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

from itertools import islice
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

from accelerate.data_loader import prepare_data_loader
from datasets import Dataset

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.sampler import EpisodeAwareSampler, InterventionBalancedSampler
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts import lerobot_train
from lerobot.utils.utils import cycle


class FlagDataset(torch.utils.data.Dataset):
    """An in-memory map dataset with a separate Arrow flag column for sampler setup."""

    def __init__(self, flags):
        self.flags = torch.tensor(flags, dtype=torch.bool)
        self.hf_dataset = Dataset.from_dict({"intervention": flags})
        self.meta = SimpleNamespace(
            episodes={"dataset_from_index": [0], "dataset_to_index": [len(flags)]},
            features={"intervention": {"dtype": "bool", "shape": (1,), "names": None}},
            has_language_columns=False,
        )
        self.episodes = None
        self.absolute_to_relative_idx = None

    def __len__(self):
        return len(self.flags)

    def __getitem__(self, index):
        return {"index": index, "intervention": self.flags[index]}


def make_cfg(**overrides):
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/merged"),
        policy=PI05Config(device="cpu", push_to_hub=False),
        num_workers=0,
        **overrides,
    )


def topology(world=1):
    return SimpleNamespace(device_type="cpu", dp_world_size=world)


def prepared_loader(cfg, dataset, world=1, rank=0, step=0):
    loader, _ = lerobot_train.make_dataloaders(cfg, dataset, None, step, topology(world))
    if cfg.num_workers:
        loader.timeout = 30
    epoch = loader.sampler.state_dict()["epoch"]
    prepared = prepare_data_loader(
        loader, num_processes=world, process_index=rank, device=torch.device("cpu"), rng_types=[]
    )
    # Mirrors the post-prepare initialization in train(), including nonzero resume epochs.
    prepared.set_epoch(epoch)
    return prepared


def batch_indices(loader, count):
    return [batch["index"].tolist() for batch in islice(cycle(loader), count)]


def assert_complete_pool_cycles(indices, flags):
    """Each exhausted group visits every eligible anchor before any are reused."""
    indices = np.asarray(indices)
    for intervention in (False, True):
        pool = np.flatnonzero(flags == intervention)
        selected = indices[flags[indices] == intervention]
        for start in range(0, len(selected), len(pool)):
            group_cycle = selected[start : start + len(pool)]
            assert len(np.unique(group_cycle)) == len(group_cycle)
            if len(group_cycle) == len(pool):
                np.testing.assert_array_equal(np.sort(group_cycle), pool)


@pytest.mark.parametrize("intervention_count", [1, 3, 17])
@pytest.mark.parametrize("batch,ratio", [(8, 0.25), (8, 0.5), (8, 0.75), (10, 0.3), (10, 0.7)])
def test_every_full_batch_has_exact_ratio_and_covers_both_pools(intervention_count, batch, ratio):
    flags = np.array([False] * 37 + [True] * intervention_count)
    sampler = InterventionBalancedSampler(EpisodeAwareSampler([0], [len(flags)]), flags, batch, ratio, seed=7)
    indices = np.array(list(sampler)).reshape(-1, batch)
    assert sampler.sampling_mode == "fixed"
    assert np.all(flags[indices].sum(axis=1) == batch * ratio)
    assert set(indices.reshape(-1)) == set(range(len(flags)))
    # Even a one-frame correction pool can supply its quota by repeating samples.
    assert len(indices) * batch == len(sampler)


@pytest.mark.parametrize("ratio", [0.5, 0.3])
def test_balancing_preserves_episode_selection_dropping_and_remapping(ratio):
    mapping = dict(zip([100, 101, 102, 103, 200, 201, 202, 203], range(8), strict=True))
    episode_sampler = EpisodeAwareSampler(
        [0, 100, 200],
        [4, 104, 204],
        episode_indices_to_use=[1, 2],
        drop_n_first_frames=1,
        drop_n_last_frames=1,
        absolute_to_relative_idx=mapping,
    )
    flags = np.array([False] * 4 + [True] * 4)
    sampler = InterventionBalancedSampler(episode_sampler, flags, 8, ratio)
    indices = np.array(list(sampler)).reshape(-1, 8)
    assert set(sampler.non_intervention_indices) == {1, 2}
    assert set(sampler.intervention_indices) == {5, 6}
    assert set(indices.reshape(-1)) <= {1, 2, 5, 6}
    if sampler.sampling_mode == "fixed":
        assert set(indices.reshape(-1)) == {1, 2, 5, 6}
        assert np.all(flags[indices].sum(axis=1) == 4)


@pytest.mark.parametrize("ratio", [0.5, 0.3])
def test_balanced_order_is_seeded_and_independent_of_global_rng(ratio):
    flags = np.array([False] * 37 + [True] * 5)

    def make(seed=7):
        return InterventionBalancedSampler(EpisodeAwareSampler([0], [42]), flags, 8, ratio, seed=seed)

    first, second = make(), make()
    epoch_zero = list(first)
    torch.rand(100)
    np.random.random(100)
    assert list(second) == epoch_zero
    epoch_one = list(first)
    assert epoch_one != epoch_zero
    assert list(second) == epoch_one
    assert list(make(seed=8)) != epoch_zero


@pytest.mark.parametrize("ratio", [0.25, 0.3, 0.5, 0.7])
@pytest.mark.parametrize("original_count,correction_count", [(37, 5), (5, 37), (17, 29)])
def test_group_cycles_continue_without_replacement_across_virtual_epochs(
    ratio, original_count, correction_count
):
    flags = np.array([False] * original_count + [True] * correction_count)
    sampler = InterventionBalancedSampler(EpisodeAwareSampler([0], [len(flags)]), flags, 8, ratio, seed=13)
    epochs = [list(sampler) for _ in range(8)]
    indices = np.concatenate(epochs)
    assert_complete_pool_cycles(indices, flags)
    for intervention in (False, True):
        selected = indices[flags[indices] == intervention]
        pool_size = np.count_nonzero(flags == intervention)
        assert len(selected) >= 2 * pool_size
        # Exhaustion reshuffles the group rather than repeating one fixed order.
        cycles = {
            tuple(selected[start : start + pool_size])
            for start in range(0, len(selected) - pool_size + 1, pool_size)
        }
        assert len(cycles) > 1
    if sampler.sampling_mode == "fixed":
        assert np.all(flags[indices.reshape(-1, 8)].sum(axis=1) == 8 * ratio)


@pytest.mark.parametrize("ratio", [0.5, 0.3, 0.7])
def test_resume_and_epoch_reposition_reconstruct_partially_consumed_group_cycles(ratio):
    flags = np.array([False] * 151 + [True] * 29)

    def make():
        return InterventionBalancedSampler(EpisodeAwareSampler([0], [len(flags)]), flags, 8, ratio, seed=13)

    sampler = make()
    reference = [list(sampler) for _ in range(10)]
    for start in (0, 8, len(sampler) - 8, len(sampler)):
        resumed = make()
        resumed.load_state_dict({"epoch": 7, "start_index": start})
        assert list(resumed) == reference[7][start:]
        assert list(resumed) == reference[8]
        resumed.set_epoch(2)
        assert list(resumed) == reference[2]


@pytest.mark.parametrize("ratio", [0.3, 0.7])
@pytest.mark.parametrize("large_intervention_pool", [False, True])
def test_probabilistic_mix_uses_group_probability_not_pool_size(ratio, large_intervention_pool):
    # A strongly imbalanced pool catches accidental uniform sampling over all frames,
    # or assigning each individual frame weight=ratio without normalizing by pool size.
    flags = np.array([large_intervention_pool] * 50_000 + [not large_intervention_pool] * 3)
    sampler = InterventionBalancedSampler(EpisodeAwareSampler([0], [len(flags)]), flags, 8, ratio, seed=7)
    indices = np.array(list(sampler)).reshape(-1, 8)
    counts = flags[indices].sum(axis=1)
    assert sampler.sampling_mode == "probabilistic"
    assert abs(flags[indices].mean() - ratio) < 0.012
    # Independent draws permit an entire batch from one pool, rather than silently
    # rounding the target to a fixed quota or alternating floor/ceil quotas.
    minority_counts = counts if ratio < 0.5 else 8 - counts
    assert np.any(minority_counts == 0)
    assert np.any(minority_counts > 3)
    small_pool = sampler.non_intervention_indices if large_intervention_pool else sampler.intervention_indices
    small_pool_counts = np.bincount(indices.reshape(-1), minlength=len(flags))[small_pool]
    assert np.max(np.abs(small_pool_counts / small_pool_counts.sum() - 1 / 3)) < 0.02


@pytest.mark.parametrize("ratio", [0.5, 0.3])
@pytest.mark.parametrize("world", [1, 2, 8])
def test_accelerate_sharding_and_resume_preserve_balanced_batches(ratio, world, tmp_path, monkeypatch):
    dataset = FlagDataset([False] * 37 + [True] * 5)
    cfg = make_cfg(intervention_sampling_ratio=ratio, batch_size=8)
    monkeypatch.setattr(
        lerobot_train, "load_training_metadata", lambda _: {"batch_size": 8, "dp_world_size": world}
    )
    for rank in range(world):
        loader = prepared_loader(cfg, dataset, world, rank)
        per_epoch = len(loader)
        reference = batch_indices(loader, per_epoch * 3)
        for batch in reference:
            assert len(batch) == 8
            if ratio == 0.5:
                assert dataset.flags[batch].sum().item() == 4
        for step in (1, per_epoch, per_epoch + 1, 2 * per_epoch + 1):
            resumed_cfg = make_cfg(intervention_sampling_ratio=ratio, batch_size=8, resume=True)
            resumed_cfg.checkpoint_path = tmp_path
            resumed = prepared_loader(resumed_cfg, dataset, world, rank, step)
            assert batch_indices(resumed, len(reference) - step) == reference[step:]


@pytest.mark.parametrize("ratio", [0.5, 0.3])
@pytest.mark.parametrize("world", [2, 3])
def test_sharded_batches_share_one_continuous_stream_per_group(ratio, world):
    dataset = FlagDataset([False] * 37 + [True] * 11)
    cfg = make_cfg(intervention_sampling_ratio=ratio, batch_size=8)
    reference, _ = lerobot_train.make_dataloaders(cfg, dataset, None, 0, topology(world))
    batches = batch_indices(reference, len(reference) * 4)
    rank_batches = [
        batch_indices(prepared_loader(cfg, dataset, world, rank), len(batches) // world)
        for rank in range(world)
    ]
    reconstructed = [batch for step in zip(*rank_batches, strict=True) for batch in step]
    assert reconstructed == batches
    assert_complete_pool_cycles(np.asarray(reconstructed).reshape(-1), dataset.flags.numpy())


@pytest.mark.parametrize("ratio", [0.5, 0.3])
def test_prefetched_worker_resume_preserves_group_cycles(ratio, tmp_path, monkeypatch):
    dataset = FlagDataset([False] * 37 + [True] * 11)
    cfg = make_cfg(intervention_sampling_ratio=ratio, batch_size=8)
    cfg.num_workers = 2
    monkeypatch.setattr(
        lerobot_train, "load_training_metadata", lambda _: {"batch_size": 8, "dp_world_size": 1}
    )
    loader = prepared_loader(cfg, dataset)
    per_epoch = len(loader)
    reference = batch_indices(loader, per_epoch * 3)
    cfg.resume = True
    cfg.checkpoint_path = tmp_path
    step = per_epoch + 1
    resumed = prepared_loader(cfg, dataset, step=step)
    assert batch_indices(resumed, len(reference) - step) == reference[step:]


@pytest.mark.parametrize("ratio", [0.5, 0.3])
@pytest.mark.parametrize("workers", [0, 2])
def test_training_loader_has_full_batches_with_workers_and_drops_tail_frames(ratio, workers):
    cfg = make_cfg(intervention_sampling_ratio=ratio, batch_size=8)
    cfg.num_workers = workers
    cfg.policy.drop_n_last_frames = 1
    dataset = FlagDataset([False] * 10 + [True] * 3)
    loader, eval_loader = lerobot_train.make_dataloaders(cfg, dataset, dataset, 0, topology())
    if workers:
        loader.timeout = 30
        eval_loader.timeout = 30
    for batch in loader:
        assert len(batch["intervention"]) == 8
        if ratio == 0.5:
            assert batch["intervention"].sum().item() == 4
        assert 12 not in batch["index"]
    # Held-out evaluation retains the dataset's original distribution and includes its tail.
    assert sum(batch["intervention"].sum().item() for batch in eval_loader) == 3


def test_default_loader_keeps_uniform_sampling():
    dataset = FlagDataset([False] * 10 + [True] * 3)
    loader, _ = lerobot_train.make_dataloaders(make_cfg(batch_size=8), dataset, None, 0, topology())
    batches = list(loader)
    assert isinstance(loader.sampler, EpisodeAwareSampler)
    assert sum(batch["intervention"].sum().item() for batch in batches) == 3
    assert sorted(torch.cat([batch["index"] for batch in batches]).tolist()) == list(range(13))


@pytest.mark.parametrize("ratio,batch", [(0.5, 8), (0.3, 8), (0.3, 10)])
def test_sampling_cli_round_trip_with_pretrained_stats(ratio, batch, tmp_path):
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=test/merged",
            "--policy.type=pi05",
            "--policy.device=cpu",
            "--policy.push_to_hub=false",
            f"--policy.pretrained_path={tmp_path}",
            "--use_pretrained_stats=true",
            f"--intervention_sampling_ratio={ratio}",
            f"--batch_size={batch}",
        ],
    )
    cfg.validate()
    assert cfg.use_pretrained_stats and cfg.intervention_sampling_ratio == ratio
    encoded = draccus.encode(cfg)
    assert draccus.decode(TrainPipelineConfig, encoded).intervention_sampling_ratio == ratio
    del encoded["intervention_sampling_ratio"]
    assert draccus.decode(TrainPipelineConfig, encoded).intervention_sampling_ratio is None


@pytest.mark.parametrize(
    "ratio,batch",
    [(0, 8), (1, 8), (-0.5, 8), (float("nan"), 8), (float("inf"), 8), (0.5, 0), (0.5, -1)],
)
def test_invalid_ratio_or_batch_is_rejected(ratio, batch):
    with pytest.raises(ValueError):
        make_cfg(intervention_sampling_ratio=ratio, batch_size=batch)._validate_intervention_sampling()
    with pytest.raises(ValueError):
        InterventionBalancedSampler(
            EpisodeAwareSampler([0], [10]), np.array([False] * 5 + [True] * 5), batch, ratio
        )


@pytest.mark.parametrize(
    "ratio,batch,mode",
    [
        (0.5, 7, "probabilistic"),
        (0.4, 8, "probabilistic"),
        (1e-12, 8, "probabilistic"),
        (1 - 1e-12, 8, "probabilistic"),
        (0.5, 1, "probabilistic"),
        (0.3, 8, "probabilistic"),
        (0.3, 10, "fixed"),
        (0.1, 30, "fixed"),
    ],
)
def test_sampling_mode_is_selected_automatically(ratio, batch, mode):
    make_cfg(intervention_sampling_ratio=ratio, batch_size=batch)._validate_intervention_sampling()
    flags = np.array([False] * 10 + [True] * 3)
    sampler = InterventionBalancedSampler(EpisodeAwareSampler([0], [13]), flags, batch, ratio)
    assert sampler.sampling_mode == mode
    assert len(list(sampler)) == len(sampler)
    if mode == "probabilistic":
        assert sampler.intervention_per_batch is None
        assert len(sampler) == ((len(flags) + batch - 1) // batch) * batch


def test_streaming_sampling_is_rejected():
    cfg = make_cfg(intervention_sampling_ratio=0.5)
    cfg.dataset.streaming = True
    with pytest.raises(ValueError, match="non-streaming"):
        cfg._validate_intervention_sampling()


@pytest.mark.parametrize("flags", [[False] * 10, [True] * 10])
def test_one_empty_group_is_rejected(flags):
    with pytest.raises(ValueError, match="both intervention=True and intervention=False"):
        lerobot_train.make_dataloaders(
            make_cfg(intervention_sampling_ratio=0.5), FlagDataset(flags), None, 0, topology()
        )


def test_missing_intervention_metadata_is_rejected():
    dataset = FlagDataset([False] * 10 + [True] * 3)
    dataset.meta.features.clear()
    with pytest.raises(ValueError, match="requires an intervention feature"):
        lerobot_train.make_dataloaders(
            make_cfg(intervention_sampling_ratio=0.5), dataset, None, 0, topology()
        )


@pytest.mark.parametrize("batch,world", [(4, 2), (8, 4)])
def test_changed_batch_or_world_on_resume_is_rejected(batch, world, tmp_path, monkeypatch):
    monkeypatch.setattr(
        lerobot_train, "load_training_metadata", lambda _: {"batch_size": 8, "dp_world_size": 2}
    )
    cfg = make_cfg(intervention_sampling_ratio=0.5, batch_size=batch, resume=True)
    cfg.checkpoint_path = tmp_path
    with pytest.raises(ValueError, match="same batch_size"):
        lerobot_train.make_dataloaders(cfg, FlagDataset([False] * 10 + [True] * 3), None, 1, topology(world))
