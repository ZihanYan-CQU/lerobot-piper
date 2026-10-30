import pytest

from lerobot.light import (
    NullStatusIndicator,
    OrvynStatusIndicator,
    StatusCue,
    StatusIndicatorConfig,
    make_status_indicator,
)


class _FakeOrvyn:
    def __init__(self, *, fail_config: bool = False) -> None:
        self.fail_config = fail_config
        self.configs: list[dict] = []
        self.colors: list[list[tuple[int, int, int]]] = []
        self.modes: list[dict] = []
        self.sleep_calls = 0

    def set_led_config(self, **kwargs) -> None:
        if self.fail_config:
            raise OSError("not connected")
        self.configs.append(kwargs)

    def set_led_color(self, colors) -> None:
        self.colors.append(list(colors))

    def set_led_mode(self, **kwargs) -> None:
        self.modes.append(kwargs)

    def sleep_now(self) -> None:
        self.sleep_calls += 1


def test_none_backend_is_a_noop() -> None:
    indicator = make_status_indicator(StatusIndicatorConfig(backend="none"))

    assert isinstance(indicator, NullStatusIndicator)
    indicator.start()
    indicator.show(StatusCue.FAULT)
    indicator.close()


def test_orvyn_adapter_maps_shared_semantic_cues() -> None:
    device = _FakeOrvyn()
    indicator = OrvynStatusIndicator(StatusIndicatorConfig(), device=device)

    indicator.show(StatusCue.AUTONOMOUS)
    assert device.configs == [{"base_brightness": 0}]
    assert device.colors[-1] == [(0, 0, 0), (0, 0, 0), (0, 255, 0)]
    assert device.modes[-1] == {"mode": "blink", "brightness": 10, "period": 800}

    indicator.show(StatusCue.CORRECTING)
    assert device.colors[-1] == [(0, 0, 0), (0, 0, 0), (0, 255, 0)]
    assert device.modes[-1]["mode"] == "static"

    indicator.show(StatusCue.RESET_REQUIRED)
    assert device.colors[-1] == [(0, 0, 0), (255, 127, 0), (0, 0, 0)]
    assert device.modes[-1]["mode"] == "static"

    indicator.show(StatusCue.EXIT)
    assert device.colors[-1] == [(255, 0, 0), (0, 0, 0), (0, 0, 0)]
    assert device.modes[-1]["mode"] == "static"


def test_optional_indicator_failure_degrades_to_noop() -> None:
    indicator = OrvynStatusIndicator(
        StatusIndicatorConfig(required=False),
        device=_FakeOrvyn(fail_config=True),
    )

    indicator.start()
    indicator.show(StatusCue.RECORDING)
    indicator.close()


def test_required_indicator_failure_is_reported() -> None:
    indicator = OrvynStatusIndicator(
        StatusIndicatorConfig(required=True),
        device=_FakeOrvyn(fail_config=True),
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        indicator.start()


def test_close_sleeps_the_indicator() -> None:
    sleeping_device = _FakeOrvyn()
    sleeping = OrvynStatusIndicator(StatusIndicatorConfig(), device=sleeping_device)
    sleeping.start()
    sleeping.close()
    assert sleeping_device.sleep_calls == 1
