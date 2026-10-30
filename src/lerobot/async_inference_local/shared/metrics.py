"""Bounded, CPU-only timing samples. Call under the runtime condition lock."""

import math
from collections import deque


class Timings:
    """Bound timing history without imposing GPU synchronization on the control loop."""

    def __init__(self, capacity: int = 512):
        """Allocate empty timing series with a fixed per-series capacity."""
        self._samples: dict[str, deque[float]] = {}
        self.capacity = capacity
        self.rejected_chunks = 0
        self.accepted_chunks = 0

    def add(self, name: str, seconds: float) -> None:
        """Record one nonnegative CPU timing sample."""
        self._samples.setdefault(name, deque(maxlen=self.capacity)).append(max(0.0, seconds))

    def quantile(self, name: str, q: float) -> float:
        """Compute a nearest-rank quantile, returning zero before the first sample."""
        samples = sorted(self._samples.get(name, ()))
        return samples[max(0, math.ceil(q * len(samples)) - 1)] if samples else 0.0

    def report(self) -> dict:
        """Return bounded latency summaries in milliseconds."""
        return {
            "accepted_chunks": self.accepted_chunks,
            "rejected_chunks": self.rejected_chunks,
            "timings_ms": {
                name: {
                    "count": len(values),
                    "p50": self.quantile(name, 0.5) * 1000,
                    "p95": self.quantile(name, 0.95) * 1000,
                    "p99": self.quantile(name, 0.99) * 1000,
                    "max": max(values) * 1000,
                }
                for name, values in self._samples.items()
                if values
            },
        }
