#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)

    def __len__(self) -> int:
        return self._num_frames


class InterventionBalancedSampler:
    """Use exact batch quotas when possible, otherwise mix by probability; resume deterministically.

    Eligible frame indices come from EpisodeAwareSampler, preserving episode selection,
    index remapping and frame dropping. Both modes consume an independently shuffled
    stream from each group, without replacement until that group is exhausted, then
    reshuffle it. Unfinished group cycles continue across virtual epoch boundaries.
    Probabilistic mode independently chooses a group for each sample; it makes as many
    draws as eligible frames, rounded up to full batches, without guaranteeing full
    pool coverage within a virtual epoch. Both modes round the number of batches to
    the data-parallel world size, so accelerate shards whole batches without tail padding.

    Group cycle positions are reconstructed from the seed and epoch, so resuming from
    an epoch/offset does not require saving prefetched iterator state. A cycle boundary
    may fall within a batch, in which case that batch can contain a repeated index.
    """

    def __init__(
        self,
        episode_sampler: EpisodeAwareSampler,
        intervention_flags: np.ndarray,
        batch_size: int,
        intervention_ratio: float = 0.5,
        num_processes: int = 1,
        seed: int = 0,
    ):
        """Configure the two frame pools and automatically select the sampling mode.

        Args:
            episode_sampler: Defines eligible dataset row indices.
            intervention_flags: Scalar boolean flags in dataset row order.
            batch_size: Number of samples per device, before gradient accumulation.
            intervention_ratio: Target fraction of intervention samples. Uses exact quotas
                when batch_size * ratio is integral, otherwise a per-sample probability.
            num_processes: Number of distinct data-parallel workers.
            seed: Seed for group selection and each group's deterministic shuffle cycles.

        Raises:
            ValueError: If the ratio or batch size is invalid, flags are invalid,
                or either eligible frame pool is empty.
        """
        if not 0 < intervention_ratio < 1 or batch_size < 1:
            raise ValueError("Intervention sampling requires batch_size >= 1 and 0 < intervention_ratio < 1.")
        if num_processes < 1:
            raise ValueError("num_processes must be positive.")
        flags = np.asarray(intervention_flags)
        if flags.dtype != np.bool_ or flags.ndim != 1:
            raise ValueError(
                "intervention_flags must be a one-dimensional boolean array without missing values."
            )
        indices = np.asarray(episode_sampler.indices, dtype=np.int64)
        if indices.min() < 0 or indices.max() >= len(flags):
            raise ValueError("intervention_flags do not cover the eligible dataset row indices.")
        self.intervention_indices = indices[flags[indices]]
        self.non_intervention_indices = indices[~flags[indices]]
        if not len(self.intervention_indices) or not len(self.non_intervention_indices):
            raise ValueError(
                "Balanced sampling requires both intervention=True and intervention=False frames "
                "after episode selection and frame dropping."
            )
        self.batch_size = batch_size
        self.intervention_ratio = intervention_ratio
        self.seed = seed
        count = batch_size * intervention_ratio
        exact = math.isclose(count, round(count), rel_tol=0, abs_tol=1e-8) and (0 < round(count) < batch_size)
        self.sampling_mode = "fixed" if exact else "probabilistic"
        if exact:
            self.intervention_per_batch = int(round(count))
            self.non_intervention_per_batch = batch_size - self.intervention_per_batch
            num_batches = max(
                math.ceil(len(self.intervention_indices) / self.intervention_per_batch),
                math.ceil(len(self.non_intervention_indices) / self.non_intervention_per_batch),
            )
        else:
            self.intervention_per_batch = None
            self.non_intervention_per_batch = None
            # Keep virtual epochs bounded even for probabilities close to 0 or 1.
            # A pool need not be exhausted before the next virtual epoch starts.
            num_batches = math.ceil(len(indices) / batch_size)
        self._num_batches = math.ceil(num_batches / num_processes) * num_processes
        self._epoch = 0
        self._start_index = 0
        self._intervention_prefix_counts = [0]

    def __len__(self) -> int:
        """Return the full virtual epoch length, including repeated samples."""
        return self._num_batches * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic epoch, preserving a pending resume offset."""
        self._epoch = epoch

    def state_dict(self) -> dict[str, int]:
        """Return the epoch and offset before the next iteration."""
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore a batch-aligned offset within an epoch."""
        epoch, start = state["epoch"], state["start_index"]
        if epoch < 0 or start < 0 or start > len(self) or start % self.batch_size:
            raise ValueError(
                "Balanced sampler state must have a nonnegative epoch and a batch-aligned offset."
            )
        self._epoch, self._start_index = epoch, start

    def __iter__(self) -> Iterator[int]:
        """Advance epoch state eagerly and iterate the requested epoch suffix."""
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _intervention_samples_before_epoch(self, epoch: int) -> int:
        """Recover group offsets without replaying every previous frame selection."""
        if self.intervention_per_batch is not None:
            return epoch * self._num_batches * self.intervention_per_batch

        while len(self._intervention_prefix_counts) <= epoch:
            previous_epoch = len(self._intervention_prefix_counts) - 1
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0, previous_epoch]))
            count = int(rng.binomial(len(self), self.intervention_ratio))
            self._intervention_prefix_counts.append(self._intervention_prefix_counts[-1] + count)
        return self._intervention_prefix_counts[epoch]

    def _shuffle_generator(self, *keys: int) -> torch.Generator:
        seed = int(np.random.SeedSequence([self.seed, *keys]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(seed)

    def _draw_pool(self, indices: np.ndarray, count: int, *, pool_id: int, start: int) -> np.ndarray:
        """Take a slice of a group's continuous stream of random permutations."""
        if len(indices) == 1:
            return np.full(count, indices[0], dtype=np.int64)
        samples = np.empty(count, dtype=np.int64)
        cycle, within_cycle = divmod(start, len(indices))
        offset = 0
        while offset < count:
            size = min(len(indices) - within_cycle, count - offset)
            generator = self._shuffle_generator(2, pool_id, cycle)
            order = torch.randperm(len(indices), generator=generator).numpy()
            samples[offset : offset + size] = indices[order[within_cycle : within_cycle + size]]
            offset += size
            cycle += 1
            within_cycle = 0
        return samples

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        intervention_start = self._intervention_samples_before_epoch(epoch)
        original_start = epoch * len(self) - intervention_start
        generator = self._shuffle_generator(1, epoch)
        if self.intervention_per_batch is None:
            # A binomial count placed at uniformly random positions has exactly the
            # same distribution as independent Bernoulli group choices. Sampling
            # the count separately makes resume cost O(epochs), not O(past samples).
            count = self._intervention_samples_before_epoch(epoch + 1) - intervention_start
            mask = torch.randperm(len(self), generator=generator).numpy() < count
        else:
            order = (
                torch.rand((self._num_batches, self.batch_size), generator=generator).argsort(dim=1).numpy()
            )
            mask = (order < self.intervention_per_batch).reshape(-1)
        # Randomize group slots before filling them, preserving each group's stream
        # order even when a shuffle cycle ends in the middle of a batch.
        indices = np.empty(len(self), dtype=np.int64)
        for pool_id, group_mask, pool, pool_start in (
            (0, ~mask, self.non_intervention_indices, original_start),
            (1, mask, self.intervention_indices, intervention_start),
        ):
            indices[group_mask] = self._draw_pool(
                pool, int(group_mask.sum()), pool_id=pool_id, start=pool_start
            )
        for index in indices[start:]:
            yield int(index)


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map a training loop step to a frame-sampler state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
