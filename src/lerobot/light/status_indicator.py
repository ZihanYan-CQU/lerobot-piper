"""Hardware-independent status indication for robot control applications.

Applications publish semantic cues (for example ``AUTONOMOUS`` or ``FAULT``)
instead of talking to a particular USB light.  The default Orvyn adapter is
optional: when the device or its dependency is unavailable it degrades to a
no-op indicator, so lighting can never become a robot-control dependency.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class StatusCue(StrEnum):
    """Semantic states shared by teleoperation, recording, and DAgger."""

    TELEOPERATING = "teleoperating"
    RECORDING_PENDING = "recording_pending"
    RECORDING = "recording"
    AUTONOMOUS = "autonomous"
    PAUSED = "paused"
    CORRECTING = "correcting"
    RESETTING = "resetting"
    RESET_REQUIRED = "reset_required"
    WARNING = "warning"
    FAULT = "fault"
    COMPLETE = "complete"
    DISCONNECTING = "disconnecting"
    EXIT = "exit"


@dataclass(frozen=True)
class _IndicatorPattern:
    colors: tuple[tuple[int, int, int], ...]
    mode: str


_GREEN = ((0, 0, 0), (0, 0, 0), (0, 255, 0))
_ORANGE = ((0, 0, 0), (255, 127, 0), (0, 0, 0))
_RED = ((255, 0, 0), (0, 0, 0), (0, 0, 0))

_PATTERNS = {
    StatusCue.TELEOPERATING: _IndicatorPattern(_GREEN, "static"),
    StatusCue.RECORDING_PENDING: _IndicatorPattern(_GREEN, "blink"),
    StatusCue.RECORDING: _IndicatorPattern(_GREEN, "static"),
    StatusCue.AUTONOMOUS: _IndicatorPattern(_GREEN, "blink"),
    StatusCue.PAUSED: _IndicatorPattern(_ORANGE, "blink"),
    StatusCue.CORRECTING: _IndicatorPattern(_GREEN, "static"),
    StatusCue.RESETTING: _IndicatorPattern(_ORANGE, "blink"),
    StatusCue.RESET_REQUIRED: _IndicatorPattern(_ORANGE, "static"),
    StatusCue.WARNING: _IndicatorPattern(_ORANGE, "blink"),
    StatusCue.FAULT: _IndicatorPattern(_RED, "blink"),
    StatusCue.COMPLETE: _IndicatorPattern(_RED, "static"),
    StatusCue.DISCONNECTING: _IndicatorPattern(_RED, "blink"),
    StatusCue.EXIT: _IndicatorPattern(_RED, "static"),
}


@dataclass
class StatusIndicatorConfig:
    """Configuration for the optional application status indicator.

    ``auto`` currently selects Orvyn but tolerates missing hardware.  ``none``
    disables indication explicitly.  Set ``required`` only in installations
    where failing startup is preferable to running without a visible signal.
    """

    backend: str = "auto"
    required: bool = False
    timeout_ms: int = 2000
    brightness: int = 10
    blink_period_ms: int = 800
    base_brightness: int = 0

    def __post_init__(self) -> None:
        """Validate values before an indicator is constructed."""
        if self.backend not in ("auto", "orvyn", "none"):
            raise ValueError("status indicator backend must be 'auto', 'orvyn', or 'none'")
        if self.timeout_ms <= 0:
            raise ValueError("status indicator timeout_ms must be positive")
        if not 0 <= self.brightness <= 100:
            raise ValueError("status indicator brightness must be between 0 and 100")
        if self.blink_period_ms <= 0:
            raise ValueError("status indicator blink_period_ms must be positive")
        if not 0 <= self.base_brightness <= 30:
            raise ValueError("status indicator base_brightness must be between 0 and 30")


@runtime_checkable
class StatusIndicator(Protocol):
    """Small interface consumed by robot applications."""

    def start(self) -> None:
        """Open and configure the optional indicator backend."""
        ...

    def show(self, cue: StatusCue) -> None:
        """Display one semantic application cue."""
        ...

    def close(self) -> None:
        """Turn the indicator off and release its backend."""
        ...


class NullStatusIndicator:
    """No-op implementation used on hardware without a status light."""

    def start(self) -> None:
        """Accept startup without opening hardware."""
        pass

    def show(self, cue: StatusCue) -> None:
        """Ignore a semantic cue."""
        del cue

    def close(self) -> None:
        """Accept shutdown without releasing hardware."""
        pass


class OrvynStatusIndicator:
    """Thread-safe adapter from semantic cues to the Orvyn USB API."""

    def __init__(
        self,
        config: StatusIndicatorConfig,
        *,
        device: Any | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        """Store configuration; hardware is opened lazily by :meth:`start`."""
        self.config = config
        self._device = device
        self._owns_device = device is None
        self._log = log or logger
        self._lock = threading.RLock()
        self._started = False
        self._disabled = False

    def _disable_after_error(
        self,
        operation: str,
        exc: BaseException,
        *,
        raise_if_required: bool = False,
    ) -> None:
        self._disabled = True
        self._started = False
        if self._owns_device and self._device is not None:
            try:
                self._device.close()
            except Exception:
                self._log.debug("Could not close failed Orvyn status indicator", exc_info=True)
        self._device = None
        message = f"Orvyn status indicator {operation} failed; robot control will continue without it"
        if raise_if_required and self.config.required:
            raise RuntimeError(message) from exc
        self._log.warning("%s: %s", message, exc)

    def start(self) -> None:
        """Open and configure the Orvyn device, or disable it on failure."""
        with self._lock:
            if self._started or self._disabled:
                return
            try:
                if self._device is None:
                    # Lazy import keeps hidapi and Orvyn hardware optional on
                    # every robot that only needs the semantic interface.
                    from lerobot.light.light_api import OrvynDevice

                    self._device = OrvynDevice(timeout_ms=self.config.timeout_ms)
                    self._device.connect()
                self._device.set_led_config(base_brightness=self.config.base_brightness)
                self._device.set_led_mode(brightness=self.config.brightness)
                self._started = True
            except Exception as exc:
                self._disable_after_error("initialization", exc, raise_if_required=True)

    def show(self, cue: StatusCue) -> None:
        """Translate and display a semantic cue."""
        with self._lock:
            if not self._started and not self._disabled:
                self.start()
            if not self._started or self._device is None:
                return
            pattern = _PATTERNS[StatusCue(cue)]
            try:
                self._device.set_led_color(list(pattern.colors))
                self._device.set_led_mode(
                    mode=pattern.mode,
                    brightness=self.config.brightness,
                    period=self.config.blink_period_ms,
                )
            except Exception as exc:
                self._disable_after_error(f"cue {cue.value!r}", exc)

    def close(self) -> None:
        """Put the light to sleep and release owned hardware."""
        with self._lock:
            device = self._device
            try:
                if self._started and device is not None:
                    sleep_now = getattr(device, "sleep_now", None)
                    if sleep_now is not None:
                        sleep_now()
            except Exception as exc:
                self._log.warning("Could not put the Orvyn status indicator to sleep: %s", exc)
            finally:
                if self._owns_device and device is not None:
                    try:
                        device.close()
                    except Exception as exc:
                        self._log.warning("Could not close the Orvyn status indicator: %s", exc)
                self._device = None
                self._started = False
                self._disabled = True


def make_status_indicator(
    config: StatusIndicatorConfig,
    *,
    device: Any | None = None,
    log: logging.Logger | None = None,
) -> StatusIndicator:
    """Build the configured indicator without opening hardware yet."""
    if config.backend == "none":
        return NullStatusIndicator()
    return OrvynStatusIndicator(config, device=device, log=log)
