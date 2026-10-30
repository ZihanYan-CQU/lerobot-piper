"""Chunk replacement with an immutable in-flight interpolation endpoint.

All methods are called under the runtime lock. No method launches CUDA work.
"""

from dataclasses import dataclass

import numpy as np

from ..shared.types import ActionChunk, PlanSlice, Prefix


@dataclass(frozen=True)
class Command:
    """One CPU motor tick, committed only after successful hardware dispatch."""

    values: np.ndarray
    step: int
    final: bool
    source_at: float


class ActionSchedule:
    """Keep a mutable future chunk and one committed interpolation endpoint."""

    def __init__(self, multiplier: int):
        """Create an empty schedule at the requested interpolation multiplier."""
        if multiplier < 1:
            raise ValueError("Interpolation multiplier must be positive")
        self.multiplier = multiplier
        self.last_sent: np.ndarray | None = None
        self.clear()

    def clear(self, seed: np.ndarray | None = None) -> None:
        """Invalidate all motion, optionally seeding interpolation from measured pose."""
        self.completed_step = -1
        self._chunk: ActionChunk | None = None
        self._index = 0
        self._inflight: PlanSlice | None = None
        self._substep = 0
        self._origin: np.ndarray | None = None
        if seed is not None:
            self.last_sent = np.asarray(seed, dtype=np.float32).copy()

    @property
    def remaining(self) -> int:
        """Return the number of endpoints not yet fully dispatched."""
        return (0 if self._chunk is None else len(self._chunk) - self._index) + int(
            self._inflight is not None
        )

    @property
    def next_mutable_step(self) -> int:
        """Return the first step that a completed prediction may replace."""
        return self.completed_step + 1 + int(self._inflight is not None)

    def snapshot(self, limit: int) -> Prefix:
        """Retain stable buffer references beginning at the first uncompleted step."""
        pieces = []
        if self._inflight is not None and limit:
            pieces.append(self._inflight)
            limit -= 1
        if self._chunk is not None and limit:
            end = min(len(self._chunk), self._index + limit)
            if end > self._index:
                pieces.append(PlanSlice(self._chunk, self._index, end))
        return Prefix(self.completed_step + 1, tuple(pieces))

    def accept(self, prefix: Prefix, chunk: ActionChunk, *, trained: bool, delay: int) -> bool:
        """Trim elapsed rows and replace only the uncommitted future trajectory."""
        offset = self.next_mutable_step - prefix.start_step
        if offset < 0 or offset >= len(chunk):
            return False
        # Count the immutable interpolation endpoint as consumed. Guided RTC
        # retains the remote deployment's three-step slack, but cannot splice a
        # result beyond that conditioning budget without an overlap check.
        slack = 0 if trained else 3
        if len(prefix) and offset > delay + slack:
            return False
        self._chunk = chunk
        self._index = offset
        return True

    def command(self) -> Command | None:
        """Reserve the next endpoint and return its next interpolation tick."""
        if self._inflight is None:
            if self._chunk is None or self._index >= len(self._chunk):
                return None
            self._inflight = PlanSlice(self._chunk, self._index, self._index + 1)
            self._index += 1
            target = self._inflight.chunk.cpu[self._inflight.start]
            self._origin = target if self.last_sent is None else self.last_sent.copy()
            self._substep = 0
        row = self._inflight
        target = row.chunk.cpu[row.start]
        final = self._substep + 1 == self.multiplier
        fraction = (self._substep + 1) / self.multiplier
        values = target if final else self._origin + fraction * (target - self._origin)
        return Command(values, self.completed_step + 1, final, row.chunk.source_at)

    def acknowledge(self, command: Command, applied: np.ndarray) -> None:
        """Advance progress only after the corresponding motor write succeeds."""
        if self._inflight is None or command.step != self.completed_step + 1:
            raise RuntimeError("Command acknowledgement does not match the reserved action")
        self.last_sent = np.asarray(applied, dtype=np.float32).copy()
        self._substep += 1
        if command.final:
            self.completed_step = command.step
            self._inflight = None
            self._origin = None
