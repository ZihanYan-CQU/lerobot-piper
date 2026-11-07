"""Qt plugin discovery must work when the active PyQt wheel is in a Chinese path."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt5")

from lerobot.yzh.wn_a100 import desktop


@pytest.mark.parametrize("bundled", [True, False], ids=["unicode-wheel", "system-qt"])
def test_qt_plugins_follow_loaded_binding_and_preserve_system_fallback(tmp_path, monkeypatch, bundled):
    """Use wheel plugins at Unicode paths without changing system Qt installs."""
    binding = tmp_path / "中文项目 路径" / "PyQt5"
    plugins = binding / "Qt5" / "plugins"
    platforms = plugins / "platforms"
    if bundled:
        platforms.mkdir(parents=True)
    monkeypatch.setattr(desktop.QtCore, "__file__", str(binding / "QtCore.so"))
    monkeypatch.setenv("QT_PLUGIN_PATH", "/existing/qt/plugins")
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/existing/qt/platforms")

    desktop._configure_qt_plugin_paths()

    expected_plugins = str(plugins) if bundled else "/existing/qt/plugins"
    expected_platforms = str(platforms) if bundled else "/existing/qt/platforms"
    assert os.environ["QT_PLUGIN_PATH"] == expected_plugins
    assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == expected_platforms
