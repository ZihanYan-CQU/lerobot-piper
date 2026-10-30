"""Owned buffers and immutable request metadata; no wire format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Observation:
    """An owned sensor snapshot with acquisition and source timestamps."""

    sequence: int
    captured_at: float
    source_at: float
    values: dict[str, Any]


@dataclass(frozen=True)
class ActionChunk:
    # CUDA buffers are owned by the inference worker, including their allocation lifetime.
    """Owned model-space, absolute-space and CPU executable action buffers."""

    model: torch.Tensor
    absolute: torch.Tensor
    cpu: np.ndarray
    source_at: float

    def __len__(self) -> int:
        """Return the number of action rows represented by this object."""
        return len(self.cpu)


@dataclass(frozen=True)
class PlanSlice:
    """An immutable view of rows within an owned action chunk."""

    chunk: ActionChunk
    start: int
    stop: int

    def __len__(self) -> int:
        """Return the number of action rows represented by this object."""
        return self.stop - self.start


@dataclass(frozen=True)
class Prefix:
    """A stable snapshot of committed and queued actions at a known control step."""

    start_step: int
    pieces: tuple[PlanSlice, ...]

    def __len__(self) -> int:
        """Return the number of action rows represented by this object."""
        return sum(map(len, self.pieces))

    def tensor(self, name: str) -> torch.Tensor | None:
        """Called only by the GPU owner, outside the scheduling lock."""
        values = [getattr(piece.chunk, name)[piece.start : piece.stop] for piece in self.pieces]
        return torch.cat(values) if len(values) > 1 else values[0] if values else None

    def cpu(self) -> np.ndarray | None:
        """Read the executable prefix on CPU without transferring CUDA tensors."""
        values = [piece.chunk.cpu[piece.start : piece.stop] for piece in self.pieces]
        return np.concatenate(values) if len(values) > 1 else values[0] if values else None


@dataclass(frozen=True)
class Request:
    """One generation-scoped inference request, passed directly to its worker."""

    generation: int
    observation: Observation
    prefix: Prefix
    task: str
    started_at: float


@dataclass(frozen=True)
class Prediction:
    """A completed action chunk and the delay it was conditioned on."""

    chunk: ActionChunk
    conditioned_delay: int
    elapsed_s: float
