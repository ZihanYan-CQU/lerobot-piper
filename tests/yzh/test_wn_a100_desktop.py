"""Regression checks for the reported desktop interaction failures."""

from __future__ import annotations

import os
import math
from dataclasses import replace
from datetime import datetime, timezone
from time import monotonic

import pytest

from lerobot.yzh.wn_a100 import remote
from PIL import ImageChops, ImageColor

# A private Qt display avoids moving the user's desktop while running regressions.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtCore import QAbstractAnimation, QEvent, QPoint, QPointF, QRect, QSize, Qt
from PyQt5.QtGui import QCursor, QFont, QMouseEvent, QTextDocument, QWheelEvent
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QToolTip

from lerobot.yzh.wn_a100 import monitor as core
from lerobot.yzh.wn_a100.desktop import (
    MonitorWindow,
    FLOATING_BALL_BASE_SIZE,
    FLOATING_BALL_MIN_SIZE,
    FLOATING_BALL_MAX_SIZE,
    FLOATING_BALL_SIZE,
    FLOATING_BALL_SIZE_STEP,
    TRAINING_RING_LAYOUT,
    TRAINING_TEXT_LAYOUT,
    BALL_MESSAGE_TEXT_LAYOUT,
    DISK_TEXT_LAYOUT,
    TRAINING_CAROUSEL_INTERVAL,
    TRAINING_TRANSITION_DURATION,
    BALL_AVAILABILITY_PAGE,
    BALL_DISK_PAGE,
    BALL_SUMMARY_PAGE,
    BALL_SSH_LOGIN_PAGE,
    PillowCanvas,
    _compact_gpu_indexes,
    NOTICE_ROW_Y,
    _ball_training_percent,
    _ball_size_value,
    _announcements_html,
    _request_badges,
)


@pytest.fixture
def window(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(core, "_load_config", lambda: {})
    monkeypatch.setattr(core, "_save_config", lambda _data: None)
    widget = MonitorWindow(
        core.build_parser().parse_args(["--demo", "--no-disk"]), start_pollers=False
    )
    widget.snapshot = core.demo_snapshot()
    widget.resize(970, 524)
    widget.move(
        250, 180
    )  # Leave room for positive drags without GNOME edge constraints.
    widget._render()
    widget.show()
    app.processEvents()
    QTest.qWait(200)  # Wait for the real window manager's initial ConfigureNotify.
    yield widget
    widget.close()
    app.processEvents()


@pytest.fixture
def multiple_requests() -> tuple[core.Announcement, ...]:
    # Identical body text deliberately exercises cache identity: each request
    # still has its own GPU group, deadline and available body width.
    text = "共享公告正文：模型训练用于实验对比，请留意申请安排。" * 5
    return (
        core.Announcement(
            "alice",
            "gpu_usage",
            text,
            (1, 3),
            expires_at=datetime(2026, 10, 7, 12, tzinfo=timezone.utc),
        ),
        core.Announcement(
            "bob",
            "gpu_usage",
            text,
            (2, 4, 5),
            expires_at=datetime(2026, 10, 8, 7, 25, tzinfo=timezone.utc),
        ),
        core.Announcement(
            "carol",
            "gpu_usage",
            text,
            tuple(range(8)),
            expires_at=datetime(2026, 10, 9, 16, 50, tzinfo=timezone.utc),
        ),
        core.Announcement("dave", "gpu_usage", text, (7,)),
    )


@pytest.fixture
def drawn_text(window, monkeypatch):
    items = []
    original = window.canvas.create_text

    def record(x, y, *, text, **style):
        items.append((x, y, text, style))
        return original(x, y, text=text, **style)

    monkeypatch.setattr(window.canvas, "create_text", record)
    return items


def assert_paired_notice_row(window, drawn, notice, *, offset=0.0) -> None:
    cards, deadline = _request_badges(notice)
    labels = [
        item for item in drawn if item[2] in (cards, deadline) and item[3].get("clip")
    ]
    assert [item[2] for item in labels] == [cards, deadline]
    card_x, card_y, _cards, card_style = labels[0]
    deadline_x, deadline_y, _deadline, deadline_style = labels[1]
    assert card_style["fill"] == core.COLORS["red"]
    assert deadline_style["fill"] == core.COLORS["amber"]
    assert card_y == pytest.approx(NOTICE_ROW_Y + offset)
    assert deadline_y == pytest.approx(card_y)
    assert "UTC" not in deadline and "时区" not in deadline
    assert (
        card_x + window.canvas.measure_text(cards, card_style["font"])[0] + 10
        < deadline_x
    )
    bodies = [
        item
        for item in drawn
        if item[3].get("clip")
        and item[3]["fill"] == core.COLORS["muted"]
        and item[2].startswith(notice.text[:4])
        and abs(item[1] - card_y) < 0.001
    ]
    assert len(bodies) == 1
    body_x, body_y, body, body_style = bodies[0]
    assert body_y == pytest.approx(deadline_y)
    assert (
        deadline_x
        + window.canvas.measure_text(deadline, deadline_style["font"])[0]
        + 10
        <= body_x
    )
    assert (
        body_x + window.canvas.measure_text(body, body_style["font"])[0]
        <= window.width() - 66
    )


def click_button(window: MonitorWindow, name: str) -> None:
    x1, y1, x2, y2 = window.button_boxes[name]
    QTest.mouseClick(
        window, Qt.LeftButton, pos=QPoint(round((x1 + x2) / 2), round((y1 + y2) / 2))
    )
    QApplication.processEvents()


def drag(window: MonitorWindow, start: QPoint, delta: QPoint) -> None:
    target_global = window.mapToGlobal(start) + delta
    QTest.mousePress(window, Qt.LeftButton, pos=start)
    event = QMouseEvent(
        QEvent.MouseMove,
        QPointF(start + delta),
        QPointF(target_global),
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(window, event)
    QTest.mouseRelease(window, Qt.LeftButton, pos=start)
    QApplication.processEvents()
    QTest.qWait(100)


def test_every_header_button_receives_clicks_and_minimize_restores(
    window, monkeypatch
) -> None:
    opened = []
    monkeypatch.setattr(window, "open_dashboard", lambda: opened.append(True))
    click_button(window, "open")
    assert opened == [True]
    click_button(window, "pin")
    assert not window.topmost
    click_button(window, "pin")
    assert window.topmost
    click_button(window, "corners")
    assert not window.rounded_frame
    click_button(window, "minimize")
    assert window.isMinimized()
    window.showNormal()
    QApplication.processEvents()
    assert window.isVisible() and not window.isMinimized()
    click_button(window, "corners")
    assert window.rounded_frame
    click_button(window, "close")
    assert window.closed


def test_round_corners_remove_the_background_and_hit_region(window) -> None:
    assert window.canvas.present().getpixel((0, 0))[3] == 0
    assert not window.mask().contains(QPoint(0, 0))
    click_button(window, "corners")
    assert window.canvas.present().getpixel((0, 0))[3] > 0
    assert window.mask().contains(QPoint(0, 0))


def test_repeated_drags_work_before_and_after_resize(window) -> None:
    initial = window.pos()
    for delta in (QPoint(40, 15), QPoint(-20, 25), QPoint(15, -10)):
        previous = window.pos()
        drag(window, QPoint(220, 28), delta)
        assert window.pos() == previous + delta
        assert window._press_global is None
    assert window.pos() != initial
    previous_size = window.size()
    drag(window, QPoint(window.width() - 3, window.height() - 3), QPoint(130, 70))
    assert window.width() == previous_size.width() + 130
    assert window.height() == previous_size.height() + 70
    previous = window.pos()
    drag(window, QPoint(220, 28), QPoint(30, 20))
    assert window.pos() == previous + QPoint(30, 20)
    window._render()
    assert len(window.card_boxes) == 8
    assert max(box[2] for box in window.card_boxes.values()) <= window.width()
    assert max(box[3] for box in window.card_boxes.values()) <= window.height() - 30


@pytest.fixture
def logged_out_ball(disk_window):
    window = disk_window
    publish_training(window)
    publish_disk(window, "2.3T")
    window.online = True
    window.ssh_connected = False
    window.ssh_error = "SSH 共享连接不存在"
    window.set_compact(True)
    return window


def show_login_page(window):
    window._stop_ball_training_transition()
    window._ball_training_index = BALL_SSH_LOGIN_PAGE
    window._render()
    return QPoint(window.ball_size // 2, window.ball_size // 2)


@pytest.mark.parametrize("own_count", [0, 1, 2])
@pytest.mark.parametrize("checking", [False, True])
def test_ssh_logged_out_ball_disables_private_pages_and_rings(
    logged_out_ball, drawn_arcs, drawn_text, own_count, checking
):
    window = logged_out_ball
    window.snapshot = snapshot_with_own_cards(window.snapshot, own_count)
    window.ssh_connected = None if checking else False
    drawn_arcs.clear()
    drawn_text.clear()
    window._sync_training_timer()
    window._render()
    public = BALL_SUMMARY_PAGE if own_count == 2 else BALL_AVAILABILITY_PAGE
    assert window._ball_page_set == (public, BALL_SSH_LOGIN_PAGE)
    assert not any(
        page in window._ball_page_set
        for page in (0, 1, BALL_DISK_PAGE)
    )
    assert drawn_arcs == []
    assert window._training_ring_at(training_ring_position(window, 0)) is None
    assert window.training_timer.isActive()
    assert "5.30%" not in window.toolTip() and "2.3T" not in window.toolTip()
    show_login_page(window)
    rows = window._ball_status_rows(BALL_SSH_LOGIN_PAGE)[0]
    assert rows[0] == ("SSH" if checking else "登录")


@pytest.mark.parametrize("own_count", [0, 1])
def test_ssh_login_success_refreshes_both_readers_and_restores_pages_without_restart(
    logged_out_ball, monkeypatch, own_count
):
    window = logged_out_ball
    window.snapshot = snapshot_with_own_cards(window.snapshot, own_count)
    calls = []
    monkeypatch.setattr(
        window.training_poller, "refresh", lambda: calls.append("training")
    )
    monkeypatch.setattr(window.disk_poller, "refresh", lambda: calls.append("disk"))
    show_login_page(window)
    window._set_training_hover(BALL_SSH_LOGIN_PAGE)
    window.ssh_events.put(remote.SessionEvent(True))
    window._drain_events()
    assert calls == ["training", "disk"]
    assert window._ball_page_set == (
        (0, BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
        if own_count
        else (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
    )
    assert window._hover_training_index is None
    assert BALL_SSH_LOGIN_PAGE not in window._ball_page_set
    assert window.training_timer.isActive()
    window.ssh_events.put(remote.SessionEvent(True))
    window._drain_events()
    assert calls == ["training", "disk"]  # No du restart on every health probe.


class FakeLoginTerminal:
    result = None

    def poll(self):
        return self.result


@pytest.mark.parametrize("size", [104, 128, 240])
def test_ssh_login_page_click_opens_one_local_terminal_without_expanding(
    logged_out_ball, monkeypatch, size
):
    window = logged_out_ball
    window.set_ball_size(size)
    center = show_login_page(window)
    opened, terminal = [], FakeLoginTerminal()
    monkeypatch.setattr(
        remote, "open_login_terminal", lambda config: opened.append(config) or terminal
    )
    QTest.mouseClick(window, Qt.LeftButton, pos=center)
    QTest.mouseClick(window, Qt.LeftButton, pos=center)
    assert opened == [window.ssh_config]
    assert window.compact_mode and not window.ssh_features_available
    assert window.ssh_login_pending
    assert window._ball_status_rows(BALL_SSH_LOGIN_PAGE)[0] == (
        "登录中",
        "请在终端",
        "完成认证",
    )
    assert BALL_DISK_PAGE not in window._ball_page_set


def test_ssh_login_page_drag_never_starts_authentication(logged_out_ball, monkeypatch):
    window = logged_out_ball
    window.move(200, 180)
    center = show_login_page(window)
    monkeypatch.setattr(
        remote, "open_login_terminal", lambda _: pytest.fail("drag must not log in")
    )
    initial = window.pos()
    drag(window, center, QPoint(15, 10))
    assert window.pos() == initial + QPoint(15, 10)
    assert window.compact_mode


def test_ssh_login_page_outer_edge_still_expands(logged_out_ball, monkeypatch):
    window = logged_out_ball
    show_login_page(window)
    monkeypatch.setattr(
        remote, "open_login_terminal", lambda _: pytest.fail("edge expands only")
    )
    QTest.mouseClick(window, Qt.LeftButton, pos=QPoint(window.ball_size // 2, 6))
    assert not window.compact_mode


def test_ssh_login_page_hover_pauses_carousel_and_keyboard_can_log_in(
    logged_out_ball, monkeypatch
):
    window = logged_out_ball
    center = show_login_page(window)
    move_ball_pointer(window, center)
    assert window._hover_training_index == BALL_SSH_LOGIN_PAGE
    assert not window.training_timer.isActive()
    calls = []
    monkeypatch.setattr(window, "login_ssh", lambda: calls.append(True))
    QTest.keyClick(window, Qt.Key_Return)
    assert calls == [True] and window.compact_mode
    window._set_training_hover(None)
    assert window.training_timer.isActive()


def test_ssh_login_failure_and_cancel_allow_retry_without_private_pages(
    logged_out_ball, monkeypatch
):
    window = logged_out_ball
    terminal = FakeLoginTerminal()
    monkeypatch.setattr(remote, "open_login_terminal", lambda _: terminal)
    show_login_page(window)
    window.login_ssh()
    terminal.result = 1
    window._drain_events()
    assert not window.ssh_login_pending
    assert "认证未完成" in window.ssh_login_error
    assert window._ball_status_rows(BALL_SSH_LOGIN_PAGE)[0][-1] == "点击重试"
    assert BALL_DISK_PAGE not in window._ball_page_set
    terminal.result = None
    window.login_ssh()
    assert window.ssh_login_pending


def test_ssh_terminal_exit_success_is_not_assumed_to_be_authenticated(
    logged_out_ball, monkeypatch
):
    window = logged_out_ball
    terminal = FakeLoginTerminal()
    terminal.result = 0
    monkeypatch.setattr(remote, "open_login_terminal", lambda _: terminal)
    window.login_ssh()
    window._drain_events()
    assert not window.ssh_connected and window.ssh_login_pending
    assert BALL_DISK_PAGE not in window._ball_page_set
    window._ssh_login_pending_until = monotonic() - 1
    window._drain_events()
    assert not window.ssh_login_pending
    assert "尚未检测到" in window.ssh_login_error


def test_ssh_no_terminal_error_stays_visible_and_retryable(
    logged_out_ball, monkeypatch
):
    window = logged_out_ball
    show_login_page(window)

    def missing(_config):
        raise RuntimeError("未找到本机终端 <not-html>")

    monkeypatch.setattr(remote, "open_login_terminal", missing)
    window.login_ssh()
    assert not window.ssh_login_pending and not window.ssh_connected
    assert "未找到本机终端 <not-html>" in window.toolTip()
    assert window._ball_status_rows(BALL_SSH_LOGIN_PAGE)[0][-1] == "点击重试"


def test_ssh_disconnect_during_private_transition_clears_old_frames_and_hover(
    disk_window, drawn_text, drawn_arcs
):
    window = disk_window
    window.online = True
    publish_training(window)
    window.set_compact(True)
    window._set_training_hover(0)
    window._ball_previous_page = BALL_DISK_PAGE
    window._ball_transition_progress = 0.25
    drawn_text.clear()
    drawn_arcs.clear()
    window.ssh_events.put(remote.SessionEvent(False, "connection lost"))
    window._drain_events()
    assert window._ball_previous_page is None
    assert window._hover_training_index is None
    assert drawn_arcs == []
    assert not any(
        item[2] in ("5.3%", "25.1%", "2.3T/3T")
        for item in ball_status_calls(drawn_text)
    )


def test_ssh_login_click_tracks_visible_fade_page_not_next_hidden_page(
    logged_out_ball, monkeypatch
):
    window = logged_out_ball
    window._ball_training_index = window._ball_page_set[0]
    window._ball_previous_page = BALL_SSH_LOGIN_PAGE
    window._ball_transition_progress = 0.2
    called = []
    monkeypatch.setattr(window, "login_ssh", lambda: called.append(True))
    center = QPoint(window.ball_size // 2, window.ball_size // 2)
    QTest.mouseClick(window, Qt.LeftButton, pos=center)
    assert called == [True]
    assert window.compact_mode


def test_ssh_finished_or_missing_logs_do_not_hide_login_or_mimic_auth_failure(
    logged_out_ball,
):
    window = logged_out_ball
    publish_training(window, completed_training_samples(window))
    assert BALL_SSH_LOGIN_PAGE in window._ball_page_set
    window.ssh_events.put(remote.SessionEvent(True))
    window._drain_events()
    publish_training(window, error="训练日志读取失败：File not found")
    assert window.ssh_connected
    assert BALL_SSH_LOGIN_PAGE not in window._ball_page_set


def test_ssh_login_page_is_not_added_when_all_private_readers_are_disabled(
    logged_out_ball,
):
    window = logged_out_ball
    window.training_enabled = False
    window.disk_enabled = False
    window._sync_training_timer()
    window._render()
    assert not window.ssh_login_required
    assert BALL_SSH_LOGIN_PAGE not in window._ball_page_set


def test_ssh_close_stops_health_probe_but_never_kills_login_terminal(
    logged_out_ball, monkeypatch
):
    window = logged_out_ball
    calls = []
    window.ssh_poller = remote.SessionPoller(window.ssh_events, window.ssh_config)
    monkeypatch.setattr(window.ssh_poller, "stop", lambda: calls.append("stop"))
    window.ssh_login_process = FakeLoginTerminal()  # No kill/terminate API.
    window.close()
    assert calls == ["stop"]


def test_narrow_layout_keeps_all_cards_and_repositions_buttons(window) -> None:
    window.resize(660, 524)
    QApplication.processEvents()
    QTest.qWait(150)
    window._render()
    assert window._columns() == 2
    assert len(window.card_boxes) == 8
    assert window.height() >= window._minimum_height(window.width())
    assert max(box[3] for box in window.card_boxes.values()) <= window.height() - 30
    assert window.button_boxes["close"][2] < window.width()


def test_floating_ball_click_restores_geometry_and_real_circular_hit_region(
    window,
) -> None:
    original = window.geometry()
    click_button(window, "compact")
    assert window.compact_mode
    assert window.size() == QSize(FLOATING_BALL_SIZE, FLOATING_BALL_SIZE)
    assert window.event_timer.isActive()
    assert not window.card_boxes and not window.button_boxes
    assert window.canvas.present().getpixel((0, 0))[3] == 0
    assert not window.mask().contains(QPoint(0, 0))
    assert window.mask().contains(QPoint(52, 52))
    QTest.mouseClick(window, Qt.LeftButton, pos=QPoint(52, 52))
    QApplication.processEvents()
    QTest.qWait(100)
    assert not window.compact_mode
    assert window.geometry() == original
    assert window.minimumWidth() == 650
    assert len(window.card_boxes) == 8


def test_floating_ball_repeated_drags_do_not_expand_and_position_is_remembered(
    window,
) -> None:
    window.set_compact(True)
    for delta in (QPoint(-50, 20), QPoint(35, 30), QPoint(-15, -20)):
        position = window.pos()
        drag(window, QPoint(52, 52), delta)
        assert window.compact_mode
        assert window.pos() == position + delta
        assert window.size() == QSize(FLOATING_BALL_SIZE, FLOATING_BALL_SIZE)
    position = window.pos()
    window.set_compact(False)
    window.set_compact(True)
    QApplication.processEvents()
    QTest.qWait(100)
    assert window.pos() == position
    # A little click jitter must not turn a click into a drag.
    drag(window, QPoint(52, 52), QPoint(2, 1))
    assert not window.compact_mode


def test_floating_ball_keeps_polling_and_pauses_only_hidden_notice_animation(
    window, multiple_requests
) -> None:
    publish_notices(window, *multiple_requests)
    window._next_notice()
    window.notice_animation.setCurrentTime(250)
    window.set_compact(True)
    assert window._previous_notice is None
    assert not window.notice_timer.isActive()
    fresh = core.demo_snapshot(tick=4)
    window.events.put(core.PollEvent(snapshot=fresh))
    window.events.put(core.PollEvent(announcements=multiple_requests[::-1]))
    window._drain_events()
    QApplication.processEvents()
    assert window.snapshot is fresh and window.online
    assert window.size() == QSize(FLOATING_BALL_SIZE, FLOATING_BALL_SIZE)
    assert not window.notice_timer.isActive()
    window._next_notice()
    assert window._previous_notice is None
    window.set_compact(False)
    QApplication.processEvents()
    assert window.notice_timer.isActive()
    assert window.snapshot is fresh and len(window.card_boxes) == 8


@pytest.fixture
def pid_tooltips(monkeypatch):
    shown, hidden = [], []
    monkeypatch.setattr(
        QToolTip,
        "showText",
        lambda position, text, widget: shown.append((QPoint(position), text, widget)),
    )
    monkeypatch.setattr(QToolTip, "hideText", lambda: hidden.append(True))
    return shown, hidden


def snapshot_with_pids(snapshot, by_gpu):
    return replace(
        snapshot,
        gpus=tuple(
            replace(
                gpu,
                user_usages=tuple(
                    (
                        replace(usage, pids=by_gpu.get(gpu.index, ()))
                        if usage.username.casefold() == "yzh"
                        else usage
                    )
                    for usage in gpu.user_usages
                ),
            )
            for gpu in snapshot.gpus
        ),
    )


def publish_processes(window, by_gpu):
    """Publish explicitly verified SSH metadata, independent of public PID rows."""
    samples = window.training_samples or tuple(
        remote.TrainingSample(log) for log in window.training_config.logs
    )
    samples = [
        replace(sample, processes=(), checked_at=monotonic()) for sample in samples
    ]
    samples[0] = replace(
        samples[0],
        processes=tuple(
            remote.TrainingProcess(pid, "lerobot-train", command, (gpu,))
            for gpu, commands in by_gpu.items()
            for pid, command in commands.items()
        ),
    )
    publish_training(window, tuple(samples))


def memory_bar_position(window, index, fraction=0.5):
    x1, y1, x2, y2 = window.memory_bar_boxes[index]
    return QPoint(round(x1 + fraction * (x2 - x1)), round((y1 + y2) / 2))


@pytest.mark.parametrize("width", [650, 970, 1250])
@pytest.mark.parametrize("fraction", [0.05, 0.5, 0.95])
def test_memory_bar_hover_lists_only_that_cards_own_pids_across_resizes(
    window, pid_tooltips, width, fraction
):
    shown, _hidden = pid_tooltips
    window.online = True
    window.snapshot = snapshot_with_pids(
        window.snapshot, {3: (3146357, 2233948), 6: (60606,)}
    )
    publish_processes(
        window,
        {
            3: {3146357: "train --job_name=first", 2233948: "train --job_name=second"},
            6: {60606: "train --job_name=other_card"},
        },
    )
    window.resize(width, max(window.height(), window._minimum_height(width)))
    QApplication.processEvents()
    window._render()
    assert set(window.memory_bar_boxes) == {3, 6}
    for index, expected, absent in (
        (3, ("2233948", "3146357"), "60606"),
        (6, ("60606",), "3146357"),
    ):
        position = memory_bar_position(window, index, fraction)
        assert window._memory_bar_at(position.x(), position.y()) == index
        move_ball_pointer(window, position)
        assert f"GPU {index} · yzh" in window.toolTip()
        assert all(f"PID：{pid}" in window.toolTip() for pid in expected)
        assert absent not in window.toolTip()
        assert window._memory_process_popup.editor.toPlainText() == window.toolTip()
        assert not shown  # The scrollable viewer suppresses the native tooltip.


def test_memory_pid_tooltip_excludes_other_users_and_keeps_request_warning(
    window, pid_tooltips
):
    window.online = True
    snapshot = snapshot_with_pids(window.snapshot, {3: (3146357,), 6: (60606,)})
    window.snapshot = replace(
        snapshot,
        gpus=tuple(
            (
                replace(
                    gpu,
                    user_usages=gpu.user_usages
                    + (core.UserGpuUsage("other", 1.0, 1, (888888,)),),
                )
                if gpu.index == 3
                else gpu
            )
            for gpu in snapshot.gpus
        ),
    )
    publish_notices(
        window, core.Announcement("alice", "gpu_usage", "申请3号卡进行训练", (3,))
    )
    publish_processes(
        window, {3: {3146357: "train --job_name=own"}, 6: {60606: "train"}}
    )
    move_ball_pointer(window, memory_bar_position(window, 3))
    assert "PID：3146357" in window.toolTip() and "888888" not in window.toolTip()
    assert "橙框" in window.toolTip() and "申请3号卡" in window.toolTip()


def test_memory_pid_tooltip_updates_without_mouse_movement_and_clears_on_process_exit(
    window, pid_tooltips
):
    shown, hidden = pid_tooltips
    window.online = True
    window.snapshot = snapshot_with_pids(window.snapshot, {3: (3146357,), 6: (60606,)})
    publish_processes(window, {3: {3146357: "train"}, 6: {60606: "train"}})
    window._render()
    move_ball_pointer(window, memory_bar_position(window, 3))
    updated = snapshot_with_pids(window.snapshot, {3: (2233948,), 6: (60606,)})
    window.events.put(core.PollEvent(snapshot=updated))
    window._drain_events()
    publish_processes(window, {3: {2233948: "train"}, 6: {60606: "train"}})
    assert "PID：2233948" in window._memory_process_popup.editor.toPlainText()
    assert "3146357" not in window.toolTip()
    departed = replace(
        updated,
        gpus=tuple(
            replace(gpu, user_usages=()) if gpu.index == 3 else gpu
            for gpu in updated.gpus
        ),
    )
    window.events.put(core.PollEvent(snapshot=departed))
    window._drain_events()
    assert 3 not in window.memory_bar_boxes
    assert window._memory_hover_position is None and hidden
    assert "2233948" not in window.toolTip()


def test_memory_pid_tooltip_missing_and_stale_ids_are_not_claimed_as_current(
    window, pid_tooltips
):
    window.online = True
    window.snapshot = snapshot_with_pids(window.snapshot, {3: (), 6: ()})
    window._render()
    position = memory_bar_position(window, 3)
    move_ball_pointer(window, position)
    assert "暂不显示 PID / 指令" in window.toolTip()
    assert "网页 PID 不可靠" in window.toolTip()
    window.events.put(
        core.PollEvent(snapshot=snapshot_with_pids(window.snapshot, {3: (3146357,)}))
    )
    window._drain_events()
    assert "PID：3146357" not in window.toolTip()
    publish_processes(window, {3: {3146357: "train"}})
    assert "PID：3146357" in window.toolTip()
    window.events.put(core.PollEvent(error="GPU API offline"))
    window._drain_events()
    assert "GPU 网页离线" in window.toolTip() and "SSH 核实" in window.toolTip()
    assert "PID：3146357" in window.toolTip()


def test_memory_hover_uses_ssh_pid_and_command_not_incorrect_public_records(window):
    window.online = True
    window.snapshot = snapshot_with_commands(
        window.snapshot, {3: {2987844: "wrong-public-command --job_name=web_wrong"}}
    )
    publish_processes(
        window, {3: {992732: "lerobot-train --job_name=SSH_actual --steps=20000"}}
    )
    hint = window._memory_pid_hint(3)
    assert "PID：992732" in hint and "--job_name=SSH_actual" in hint
    assert "进程名：lerobot-train" in hint and "进程指令（SSH）" in hint
    assert "2987844" not in hint and "web_wrong" not in hint


def test_relogin_rejects_old_process_event_queued_before_new_session(window):
    window.online = True
    publish_processes(window, {3: {992732: "lerobot-train --job_name=old_session"}})
    old_samples = window.training_samples
    window.ssh_events.put(remote.SessionEvent(False, "SSH disconnected"))
    window.ssh_events.put(remote.SessionEvent(True))
    window.training_events.put(remote.TrainingEvent(samples=old_samples))
    window._drain_events()
    assert not window._verified_gpu_processes(3)
    assert "992732" not in window._memory_pid_hint(3)
    publish_processes(window, {3: {992733: "lerobot-train --job_name=new_session"}})
    assert window._verified_gpu_processes(3)[0].pid == 992733


@pytest.mark.parametrize(
    "failure", ["ssh_error", "file_error", "exited", "expired", "logout", "relogin"]
)
def test_unverified_process_ids_disappear_instead_of_falling_back_to_web(
    command_popup, monkeypatch, failure
):
    window, popup, _cursor = command_popup
    assert "PID：3146357" in popup.editor.toPlainText()
    if failure == "ssh_error":
        publish_training(window, error="SSH disconnected")
    elif failure == "file_error":
        samples = tuple(
            remote.TrainingSample(sample.log, error="file missing")
            for sample in window.training_samples
        )
        publish_training(window, samples)
    elif failure == "exited":
        publish_processes(window, {})
    elif failure == "expired":
        checked_at = window.training_samples[0].checked_at
        monkeypatch.setattr(
            "lerobot.yzh.wn_a100.desktop.monotonic",
            lambda: checked_at + remote.PROCESS_STALE_AFTER,
        )
        window._render()
    else:
        window.ssh_events.put(remote.SessionEvent(False, "SSH disconnected"))
        window._drain_events()
        if failure == "relogin":
            window.ssh_events.put(remote.SessionEvent(True))
            window._drain_events()
    assert not popup.isVisible()
    assert "3146357" not in window.toolTip()
    assert "暂不显示 PID / 指令" in window.toolTip()
    assert not window._has_memory_commands(3)


@pytest.mark.parametrize("action", ["outside", "leave", "hide", "compact"])
def test_memory_pid_popup_hides_when_leaving_the_bar_or_full_panel(
    window, pid_tooltips, action
):
    _shown, hidden = pid_tooltips
    window.online = True
    window._render()
    move_ball_pointer(window, memory_bar_position(window, 3))
    assert window._memory_hover_position is not None
    if action == "outside":
        x1, y1, x2, _y2 = window.card_boxes[3]
        move_ball_pointer(window, QPoint(round((x1 + x2) / 2), round(y1 + 40)))
    elif action == "leave":
        QApplication.sendEvent(window, QEvent(QEvent.Leave))
    elif action == "hide":
        window.hide()
    else:
        window.set_compact(True)
    assert window._memory_hover_position is None and hidden
    assert "PID：" not in window.toolTip()
    if action == "compact":
        assert window.memory_bar_boxes == {}
        assert window._memory_bar_at(50, 50) is None
        move_ball_pointer(window, QPoint(50, 50))
        assert "PID" not in window.toolTip()


def test_pids_are_hover_metadata_only_not_extra_canvas_text_or_ball_information(
    window, drawn_text, pid_tooltips
):
    window.online = True
    window._render()
    original = window.canvas._image.copy()
    window.snapshot = snapshot_with_pids(window.snapshot, {3: (3146357,), 6: (60606,)})
    drawn_text.clear()
    window._render()
    assert (
        ImageChops.difference(
            original.convert("RGB"), window.canvas._image.convert("RGB")
        ).getbbox()
        is None
    )
    assert not any("PID" in item[2] or "3146357" in item[2] for item in drawn_text)
    window.set_compact(True)
    assert "PID" not in window.toolTip() and "3146357" not in window.toolTip()


def snapshot_with_commands(snapshot, by_gpu):
    return replace(
        snapshot,
        gpus=tuple(
            replace(
                gpu,
                user_usages=tuple(
                    (
                        replace(
                            usage,
                            pids=tuple(sorted(by_gpu[gpu.index])),
                            commands=tuple(
                                core.ProcessCommand(pid, command)
                                for pid, command in sorted(by_gpu[gpu.index].items())
                            ),
                        )
                        if usage.username.casefold() == "yzh" and gpu.index in by_gpu
                        else usage
                    )
                    for usage in gpu.user_usages
                ),
            )
            for gpu in snapshot.gpus
        ),
    )


@pytest.fixture
def command_popup(window, monkeypatch):
    command = (
        "lerobot-train --dataset.root='/home/yzh/具身智能/训练 数据/shape-matching' --policy.type=pi05 "
        + " ".join(f"--parameter_{index}={index}" for index in range(40))
    )
    window.online = True
    window.snapshot = snapshot_with_commands(
        window.snapshot,
        {
            3: {3146357: command},
            6: {2233948: "lerobot-train --steps=54321 --batch_size=16"},
        },
    )
    publish_processes(
        window,
        {
            3: {3146357: command},
            6: {2233948: "lerobot-train --steps=54321 --batch_size=16"},
        },
    )
    width = 650
    window.resize(width, window._minimum_height(width))
    QApplication.processEvents()
    window._render()
    position = memory_bar_position(window, 3)
    cursor = {"position": window.mapToGlobal(position)}
    monkeypatch.setattr(QCursor, "pos", lambda: cursor["position"])
    move_ball_pointer(window, position)
    QApplication.processEvents()
    # Model the pointer still resting on the bar, without using the real desktop.
    window._memory_pointer_on_bar = True
    window._memory_hide_timer.stop()
    popup = window._memory_process_popup
    assert popup is not None and popup.isVisible()
    return window, popup, cursor


def test_command_popup_is_scrollable_readonly_formatted_and_native_tip_is_suppressed(
    command_popup,
):
    window, popup, _cursor = command_popup
    text = popup.editor.toPlainText()
    assert popup.editor.isReadOnly() and not popup.editor.acceptRichText()
    assert "PID：3146357" in text
    assert "--dataset.root='/home/yzh/具身智能/训练 数据/shape-matching'" in text
    assert [line.strip() for line in text.splitlines() if "--parameter_" in line] == [
        f"--parameter_{index}={index}" for index in range(40)
    ]
    assert popup.editor.verticalScrollBar().maximum() > 0
    assert popup.editor.horizontalScrollBar().maximum() == 0
    area = popup.screen().availableGeometry()
    assert area.contains(popup.geometry())
    assert popup.height() < area.height()
    assert "CUDA_VISIBLE_DEVICES=" not in text and "nohup" not in text
    assert QApplication.sendEvent(window, QEvent(QEvent.ToolTip))


def test_two_pid_command_groups_never_mix_arguments_or_other_users(command_popup):
    window, popup, _cursor = command_popup
    snapshot = snapshot_with_commands(
        window.snapshot,
        {
            3: {
                111111: "lerobot-train --steps=111 --job_name=first",
                222222: "lerobot-train --steps=222 --job_name=second",
            }
        },
    )
    snapshot = replace(
        snapshot,
        gpus=tuple(
            (
                replace(
                    gpu,
                    user_usages=gpu.user_usages
                    + (
                        core.UserGpuUsage(
                            "other",
                            1,
                            1,
                            (99999,),
                            (core.ProcessCommand(99999, "other-user --private=value"),),
                        ),
                    ),
                )
                if gpu.index == 3
                else gpu
            )
            for gpu in snapshot.gpus
        ),
    )
    window.events.put(core.PollEvent(snapshot=snapshot))
    window._drain_events()
    publish_processes(
        window,
        {
            3: {
                111111: "lerobot-train --steps=111 --job_name=first",
                222222: "lerobot-train --steps=222 --job_name=second",
            },
        },
    )
    text = popup.editor.toPlainText()
    first, second = text.partition("PID：222222")[::2]
    assert (
        "PID：111111" in first
        and "--steps=111" in first
        and "--job_name=first" in first
    )
    assert "--steps=222" not in first
    assert "--steps=222" in second and "--job_name=second" in second
    assert "--steps=111" not in second
    assert "other-user" not in text and "99999" not in text


def test_popup_switches_to_the_corresponding_gpu_and_preserves_scroll_on_same_gpu_updates(
    command_popup,
):
    window, popup, cursor = command_popup
    scroll = popup.editor.verticalScrollBar()
    scroll.setValue(scroll.maximum() // 2)
    previous, geometry = scroll.value(), popup.geometry()
    window.events.put(
        core.PollEvent(
            snapshot=replace(
                window.snapshot, received_at=window.snapshot.received_at + 1
            )
        )
    )
    window._drain_events()
    assert scroll.value() == previous and popup.geometry() == geometry
    position = memory_bar_position(window, 6)
    cursor["position"] = window.mapToGlobal(position)
    move_ball_pointer(window, position)
    text = popup.editor.toPlainText()
    assert popup.current_gpu == 6 and "GPU 6" in popup.title.text()
    assert "PID：2233948" in text and "--steps=54321" in text
    assert "3146357" not in text and "--parameter_" not in text
    assert scroll.value() == 0


def test_popup_pointer_can_cross_gap_enter_and_scroll_without_early_dismissal(
    command_popup,
):
    window, popup, cursor = command_popup
    cursor["position"] = popup.geometry().center()
    QApplication.sendEvent(window, QEvent(QEvent.Leave))
    assert window._memory_hide_timer.isActive()
    QApplication.sendEvent(popup, QEvent(QEvent.Enter))
    assert not window._memory_hide_timer.isActive()
    QTest.qWait(280)
    assert popup.isVisible()
    popup.editor.verticalScrollBar().setValue(
        popup.editor.verticalScrollBar().maximum()
    )
    assert popup.editor.verticalScrollBar().value() > 0
    window._expire_memory_hover()
    assert popup.isVisible()
    cursor["position"] = QPoint(-9999, -9999)
    QApplication.sendEvent(popup, QEvent(QEvent.Leave))
    QTest.qWait(280)
    assert not popup.isVisible() and window._memory_hover_position is None


def test_popup_copy_button_copies_view_text_without_running_the_command(
    command_popup, monkeypatch
):
    _window, popup, _cursor = command_popup
    copied = []

    class Clipboard:
        def setText(self, text):
            copied.append(text)

    monkeypatch.setattr(QApplication, "clipboard", lambda: Clipboard())
    QTest.mouseClick(popup.copy_button, Qt.LeftButton)
    assert copied == [popup.editor.toPlainText()]
    assert "--parameter_39=39" in copied[0]


@pytest.mark.parametrize("action", ["escape", "compact", "hide", "close"])
def test_long_command_popup_closes_and_never_appears_in_ball(command_popup, action):
    window, popup, _cursor = command_popup
    if action == "escape":
        QTest.keyClick(popup, Qt.Key_Escape)
    elif action == "compact":
        window.set_compact(True)
        assert (
            "--parameter_" not in window.toolTip() and "3146357" not in window.toolTip()
        )
    elif action == "hide":
        window.hide()
    else:
        window.close()
    assert not popup.isVisible()
    assert window._memory_hover_position is None
    assert not window._memory_hide_timer.isActive()


def test_popup_updates_commands_without_pointer_movement_marks_outage_and_clears_on_exit(
    command_popup,
):
    window, popup, _cursor = command_popup
    snapshot = snapshot_with_commands(
        window.snapshot,
        {3: {3146357: "lerobot-train --steps=12345 --job_name=new_run"}},
    )
    window.events.put(core.PollEvent(snapshot=snapshot))
    window._drain_events()
    publish_processes(
        window, {3: {3146357: "lerobot-train --steps=12345 --job_name=new_run"}}
    )
    assert "--job_name=new_run" in popup.editor.toPlainText()
    assert "--parameter_" not in popup.editor.toPlainText()
    window.events.put(core.PollEvent(error="API offline"))
    window._drain_events()
    assert "GPU 网页离线" in popup.editor.toPlainText()
    assert "--job_name=new_run" in popup.editor.toPlainText()
    departed = replace(
        snapshot,
        gpus=tuple(
            replace(gpu, user_usages=()) if gpu.index == 3 else gpu
            for gpu in snapshot.gpus
        ),
    )
    window.events.put(core.PollEvent(snapshot=departed))
    window._drain_events()
    assert not popup.isVisible()
    assert window._memory_hover_position is None


@pytest.mark.parametrize(
    "area",
    [
        QRect(0, 0, 800, 800),
        QRect(0, 0, 1280, 720),
        QRect(-1920, 0, 1920, 1080),
        QRect(0, -1080, 1920, 1080),
    ],
)
def test_command_popup_is_bounded_on_each_screen_including_negative_coordinates(
    command_popup, monkeypatch, area
):
    _window, popup, _cursor = command_popup

    class Screen:
        def availableGeometry(self):
            return area

    monkeypatch.setattr(QApplication, "screenAt", lambda _position: Screen())
    for anchor in (area.topLeft(), area.bottomRight(), area.center()):
        popup.present(
            3, popup.editor.toPlainText(), anchor, reposition=True, stale=False
        )
        assert area.contains(popup.geometry())


def snapshot_with_own_cards(snapshot, count):
    # Several processes on a card still count as one occupied card.
    own_indexes = set((3, 6, 4)[:count])
    return replace(
        snapshot,
        gpus=tuple(
            replace(
                gpu,
                user_usages=tuple(
                    usage
                    for usage in gpu.user_usages
                    if usage.username.casefold() != core.HIGHLIGHT_USERNAME.casefold()
                )
                + (
                    (core.UserGpuUsage("YzH", 1.0, 5),)
                    if gpu.index in own_indexes
                    else ()
                ),
            )
            for gpu in snapshot.gpus
        ),
    )


def test_floating_ball_available_count_excludes_requested_idle_cards(
    window, drawn_text
) -> None:
    window.snapshot = snapshot_with_own_cards(window.snapshot, 1)
    window.snapshot = replace(
        window.snapshot,
        gpus=tuple(
            (
                replace(gpu, utilization=0, memory_percent=0)
                if gpu.index in (0, 1, 2)
                else replace(gpu, utilization=70, memory_percent=60)
            )
            for gpu in window.snapshot.gpus
        ),
    )
    window.online = True
    publish_notices(
        window, core.Announcement("request", "gpu_usage", "申请1号卡", (1,))
    )
    window.set_compact(True)
    assert window.available_gpu_indexes == {0, 2}
    rows = draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
    assert [item[2] for item in rows] == ["2张", "可用 GPU", "0、2"]
    assert "可用 GPU：0、2" in window.toolTip()
    window.events.put(core.PollEvent(notice_error="test outage"))
    drawn_text.clear()
    window._drain_events()
    assert [item[2] for item in ball_training_rows(drawn_text)] == [
        "3张",
        "空闲 GPU",
        "0–2",
    ]
    assert "可用 GPU：" not in window.toolTip()
    assert "申请信息尚未确认" in window.toolTip()


@pytest.mark.parametrize("owned_count", [0, 1, 2, 3])
@pytest.mark.parametrize("training_enabled", [True, False])
def test_ball_shows_availability_only_when_owning_fewer_than_two_cards(
    window, drawn_text, owned_count, training_enabled
) -> None:
    window.snapshot = snapshot_with_own_cards(window.snapshot, owned_count)
    window.training_enabled = training_enabled
    window.online = True
    publish_notices(window)
    publish_training(window)
    window.set_compact(True)
    for notices_known in (True, False):
        window.notice_error = "" if notices_known else "test outage"
        drawn_text.clear()
        window._render()
        texts = [item[2] for item in drawn_text]
        assert len(window.own_gpu_indexes) == owned_count
        assert window.show_ball_availability is (owned_count < 2)
        prefix = "可用" if notices_known else "空闲"
        if owned_count < 2:
            count = (
                len(window.available_gpu_indexes)
                if notices_known
                else window.snapshot.idle_gpu_count
            )
            if owned_count == 0:
                assert texts == [
                    f"{count}张",
                    f"{prefix} GPU",
                    _compact_gpu_indexes(window.available_gpu_indexes),
                ]
                assert not window.training_timer.isActive()
            elif training_enabled:
                assert not any(text.startswith(("可用", "空闲")) for text in texts)
                assert BALL_AVAILABILITY_PAGE in window._ball_page_set
                rows = draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
                assert rows[0][2] == f"{count}张"
                assert rows[1][2] == f"{prefix} GPU"
                draw_training_task(window, drawn_text, 0)
            else:
                assert f"{prefix}卡" in texts
                assert f"{count}/{len(window.snapshot.gpus)}" in texts
            assert ("可用 GPU：" in window.toolTip()) is notices_known
        else:
            assert not any(text.startswith(("可用", "空闲")) for text in texts)
            assert "可用 GPU：" not in window.toolTip()
            assert "空闲卡不代表可用卡" not in window.toolTip()
            if not training_enabled:
                assert f"{owned_count}张" in texts and "已用卡" in texts
        if training_enabled and owned_count:
            assert "5.3%" in texts and "25.1%" not in texts
        if not owned_count:
            assert not any("%" in text for text in texts)
        assert not any(text in ("申请", "冲突", "待确认") for text in texts)


@pytest.mark.parametrize("training_enabled", [True, False])
def test_ball_availability_updates_live_as_cards_are_acquired_and_released(
    window, drawn_text, training_enabled
) -> None:
    window.training_enabled = training_enabled
    publish_notices(window)
    window.set_compact(True)
    snapshot = window.snapshot
    for count in (0, 1, 2, 1, 0, 2):
        drawn_text.clear()
        window.events.put(
            core.PollEvent(snapshot=snapshot_with_own_cards(snapshot, count))
        )
        window._drain_events()
        assert window.online
        shown = any(item[2].startswith("可用") for item in drawn_text)
        if training_enabled and count == 1:
            assert BALL_AVAILABILITY_PAGE in window._ball_page_set
            rows = draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
            assert rows[1][2] == "可用 GPU"
        else:
            assert shown is (count < 2)
        assert ("可用 GPU：" in window.toolTip()) is (count < 2)
    for count in (0, 1, 2):
        window.events.put(
            core.PollEvent(snapshot=snapshot_with_own_cards(snapshot, count))
        )
        window._drain_events()
        drawn_text.clear()
        window.events.put(core.PollEvent(error="test offline"))
        window._drain_events()
        assert not window.show_ball_availability
        assert "离线" in [item[2] for item in drawn_text]
        assert not any(item[2].startswith(("可用", "空闲")) for item in drawn_text)


def ball_ring_color(window) -> tuple[int, ...]:
    return window.canvas._image.getpixel(
        (window.canvas._scaled(51.5), window.canvas._scaled(1.5))
    )


@pytest.fixture
def drawn_arcs(window, monkeypatch):
    arcs = []
    original = window.canvas.create_arc

    def record(*coordinates, **style):
        arcs.append({**style, "coordinates": coordinates})
        return original(*coordinates, **style)

    monkeypatch.setattr(window.canvas, "create_arc", record)
    return arcs


def publish_training(window, samples=None, *, error=""):
    window.training_events.put(
        remote.TrainingEvent(
            samples=(
                remote.demo_samples(window.training_config)
                if samples is None and not error
                else samples or ()
            ),
            error=error,
        )
    )
    window._drain_events()


def completed_training_samples(window):
    return tuple(
        replace(
            sample,
            progress=remote.TrainingProgress(
                sample.progress.total, sample.progress.total, eta_seconds=0
            ),
            file_age=86400,
        )
        for sample in remote.demo_samples(window.training_config)
    )


def ball_training_rows(drawn):
    return [item for item in drawn if item[3].get("clip") == (20, 25, 83, 80)]


def draw_training_task(window, drawn, index):
    window._stop_ball_training_transition()
    window._ball_training_index = index
    drawn.clear()
    window._render()
    layout = (
        DISK_TEXT_LAYOUT
        if index == BALL_DISK_PAGE
        else (
            BALL_MESSAGE_TEXT_LAYOUT
            if index >= BALL_AVAILABILITY_PAGE
            else TRAINING_TEXT_LAYOUT
        )
    )
    rows = [item for item in drawn if item[1] in layout]
    assert len(rows) == len(layout)
    assert [item[1] for item in rows] == list(layout)
    return rows


def finish_ball_transition(window):
    window.training_animation.stop()
    window._finish_ball_training_transition()


def training_ring_position(window, index, angle=0):
    x1, _y1, x2, _y2, width = TRAINING_RING_LAYOUT[index]
    radius = (x2 - x1 - width) / 2
    center = (FLOATING_BALL_BASE_SIZE - 1) / 2
    scale = window.ball_size / FLOATING_BALL_BASE_SIZE
    radians = math.radians(angle)
    return QPoint(
        round((center + radius * math.cos(radians)) * scale),
        round((center + radius * math.sin(radians)) * scale),
    )


def move_ball_pointer(window, position):
    QApplication.sendEvent(
        window,
        QMouseEvent(
            QEvent.MouseMove,
            QPointF(position),
            QPointF(window.mapToGlobal(position)),
            Qt.NoButton,
            Qt.NoButton,
            Qt.NoModifier,
        ),
    )


def snapshot_with_single_owned_gpu(snapshot, index):
    snapshot = snapshot_with_own_cards(snapshot, 0)
    return replace(
        snapshot,
        gpus=tuple(
            (
                replace(
                    gpu,
                    user_usages=gpu.user_usages + (core.UserGpuUsage("yzh", 1.0, 1),),
                )
                if gpu.index == index
                else gpu
            )
            for gpu in snapshot.gpus
        ),
    )


def ball_status_calls(drawn):
    return [item for item in drawn if item[3].get("clip") == (20, 25, 83, 80)]


@pytest.fixture
def disk_window(window):
    window.disk_enabled = True
    window.disk_config = remote.DiskConfig(window.training_config)
    window.disk_poller = remote.DiskPoller(
        window.disk_events, window.disk_config, demo=True
    )
    return window


def publish_disk(window, size=None, *, error="", scanning=False):
    sample = (
        remote.parse_disk_sample(
            {"user": "yzh", "home": "/home/yzh", "output": f"{size}\t/home/yzh\n"}
        )
        if size is not None
        else None
    )
    window.disk_events.put(
        remote.DiskEvent(sample=sample, error=error, scanning=scanning)
    )
    window._drain_events()


@pytest.mark.parametrize("own_count", [0, 1, 2])
def test_disk_page_is_added_to_idle_single_and_two_card_carousels(
    disk_window, own_count
):
    window = disk_window
    window.online = True
    window.snapshot = snapshot_with_own_cards(window.snapshot, own_count)
    publish_training(window)
    publish_disk(window, "2.3T")
    window.set_compact(True)
    expected = {
        0: (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE),
        1: (0, BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE),
        2: (0, 1, BALL_DISK_PAGE),
    }[own_count]
    assert window._ball_page_set == expected
    window._ball_training_index = window._ball_page_set[-2]
    window._next_ball_training()
    assert window.displayed_training_index == BALL_DISK_PAGE
    finish_ball_transition(window)
    window._next_ball_training()
    assert window.displayed_training_index == expected[0]


def test_idle_carousel_cycles_without_starting_extra_disk_scans(
    disk_window, drawn_text, drawn_arcs, monkeypatch
):
    window = disk_window
    window.online = True
    window.snapshot = snapshot_with_own_cards(window.snapshot, 0)
    publish_notices(window)
    publish_training(window)
    publish_disk(window, "2.3T")
    window.set_compact(True)
    scans = []
    monkeypatch.setattr(window.disk_poller, "refresh", lambda: scans.append(True))
    drawn_arcs.clear()
    for page in (BALL_DISK_PAGE, BALL_AVAILABILITY_PAGE) * 3:
        window._next_ball_training()
        assert window.displayed_training_index == page
        assert window._ball_previous_page is not None
        assert window.training_animation.state() == QAbstractAnimation.Running
        finish_ball_transition(window)
        rows = draw_training_task(window, drawn_text, page)
        assert [item[2] for item in rows] == list(window._ball_status_rows(page)[0])
        assert window.training_timer.isActive()
    assert not scans
    assert not drawn_arcs


@pytest.mark.parametrize("progress", [0.25, 0.75])
def test_idle_disk_fade_is_cancelled_as_soon_as_ssh_disconnects(
    disk_window, drawn_text, drawn_arcs, progress
):
    window = disk_window
    window.online = True
    window.snapshot = snapshot_with_own_cards(window.snapshot, 0)
    publish_notices(window)
    publish_training(window)
    publish_disk(window, "2.3T")
    window.set_compact(True)
    window._next_ball_training()
    assert window.displayed_training_index == BALL_DISK_PAGE
    window.training_animation.pause()
    window._animate_ball_training(progress)
    drawn_text.clear()
    drawn_arcs.clear()
    window.ssh_events.put(remote.SessionEvent(False, "SSH disconnected"))
    window._drain_events()
    assert window._ball_page_set == (BALL_AVAILABILITY_PAGE, BALL_SSH_LOGIN_PAGE)
    assert window.displayed_training_index == BALL_AVAILABILITY_PAGE
    assert window._ball_previous_page is None
    assert window.training_animation.state() == QAbstractAnimation.Stopped
    assert [item[2] for item in drawn_text] == list(
        window._ball_status_rows(BALL_AVAILABILITY_PAGE)[0]
    )
    assert not drawn_arcs
    assert "2.3T" not in window.toolTip()
    assert window.training_timer.isActive()


@pytest.mark.parametrize("size", [104, 128, 160, 240])
@pytest.mark.parametrize("used", ["2.3T", "3072G", "3.1T"])
def test_disk_usage_and_quota_are_readable_and_excess_is_red(
    disk_window, drawn_text, size, used
):
    window = disk_window
    window.online = True
    window.set_compact(True)
    window.set_ball_size(size)
    publish_disk(window, used)
    rows = draw_training_task(window, drawn_text, BALL_DISK_PAGE)
    assert [item[2] for item in rows] == [
        f"{used}/3T",
        f"{window.disk_sample.used_bytes / remote.QUOTA_BYTES:.1%}",
    ]
    over = used == "3.1T"
    assert all(
        item[3]["fill"] == (core.COLORS["red"] if over else core.COLORS["cyan"])
        for item in rows
    )
    assert len(ball_status_calls(drawn_text)) == 2
    assert ("超额" in window.toolTip()) is over
    for item, limit in zip(rows, (60, 50), strict=True):
        assert window.canvas.measure_text(item[2], item[3]["font"])[0] <= limit


def test_disk_alert_stays_on_all_pages_and_clears_after_successful_recovery(
    disk_window, drawn_text
):
    window = disk_window
    window.online = True
    publish_training(window)
    window.set_compact(True)
    publish_disk(window, "3.1T")
    for page in (0, 1, BALL_DISK_PAGE):
        draw_training_task(window, drawn_text, page)
        assert ball_ring_color(window) == ImageColor.getcolor(
            core.COLORS["red"], "RGBA"
        )
    assert "磁盘占用超过 3T" in window.toolTip()
    assert "占用冲突" not in window.toolTip()
    publish_disk(window, error="SSH 连接失效")
    rows = draw_training_task(window, drawn_text, BALL_DISK_PAGE)
    assert [item[2] for item in rows] == ["3.1T/3T", "103.3%"]
    assert "超额 · 上次" in window.toolTip()
    assert ball_ring_color(window) == ImageColor.getcolor(core.COLORS["red"], "RGBA")
    publish_disk(window, "2.3T")
    rows = draw_training_task(window, drawn_text, BALL_DISK_PAGE)
    assert [item[2] for item in rows] == ["2.3T/3T", "76.7%"]
    assert ball_ring_color(window) == ImageColor.getcolor(
        core.COLORS["yzh_border"], "RGBA"
    )
    assert "磁盘占用超过 3T" not in window.toolTip()


def test_unknown_failed_scanning_disk_is_not_invented_as_zero(disk_window, drawn_text):
    window = disk_window
    window.set_compact(True)
    for error, scanning, expected in (
        ("", False, "等待统计"),
        ("", True, "统计中"),
        ("permission", False, "读取失败"),
    ):
        publish_disk(window, error=error, scanning=scanning)
        rows = draw_training_task(window, drawn_text, BALL_DISK_PAGE)
        assert [item[2] for item in rows] == ["…/3T", "—%"]
        assert expected in window.toolTip()
        assert not window.disk_sample.over_quota


def test_disk_gpu_failures_do_not_invalidate_each_others_data(disk_window, drawn_text):
    window = disk_window
    window.online = True
    publish_training(window)
    publish_disk(window, "2.3T")
    window.set_compact(True)
    publish_disk(window, error="disk read error")
    assert window.online and window.training_samples[0].is_fresh(
        window.training_config.stale_after
    )
    rows = draw_training_task(window, drawn_text, BALL_DISK_PAGE)
    assert [item[2] for item in rows] == ["2.3T/3T", "76.7%"]
    assert "上次 · 更新失败" in window.toolTip()
    assert all(item[3]["fill"] == core.COLORS["subtle"] for item in rows)
    publish_disk(window, "2.3T")
    window.events.put(core.PollEvent(error="GPU API offline"))
    window._drain_events()
    rows = draw_training_task(window, drawn_text, BALL_DISK_PAGE)
    assert [item[2] for item in rows] == ["2.3T/3T", "76.7%"]
    assert all(item[3]["fill"] == core.COLORS["cyan"] for item in rows)


@pytest.mark.parametrize("own_count", [1, 2])
@pytest.mark.parametrize("online", [True, False])
def test_disk_page_has_only_two_lines_without_gpu_footer(
    disk_window, drawn_text, own_count, online
):
    window = disk_window
    window.online = online
    window.snapshot = snapshot_with_own_cards(window.snapshot, own_count)
    publish_training(window)
    publish_disk(window, "2.3T")
    window.set_compact(True)
    draw_training_task(window, drawn_text, BALL_DISK_PAGE)
    assert [item[2] for item in drawn_text] == ["2.3T/3T", "76.7%"]


@pytest.mark.parametrize(
    "previous, current",
    [
        (1, BALL_DISK_PAGE),
        (BALL_DISK_PAGE, 0),
        (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE),
        (BALL_DISK_PAGE, BALL_AVAILABILITY_PAGE),
    ],
)
@pytest.mark.parametrize("progress", [0.25, 0.75])
def test_disk_two_line_layout_transitions_as_one_group_without_old_task_labels(
    disk_window, drawn_text, previous, current, progress
):
    window = disk_window
    window.online = True
    if BALL_AVAILABILITY_PAGE in (previous, current):
        window.snapshot = snapshot_with_own_cards(window.snapshot, 0)
    publish_notices(window)
    publish_training(window)
    publish_disk(window, "2.3T")
    window.set_compact(True)
    window._ball_previous_page = previous
    window._ball_training_index = current
    drawn_text.clear()
    window._animate_ball_training(progress)
    rows = ball_status_calls(drawn_text)
    visible = previous if progress < 0.5 else current
    if visible == BALL_DISK_PAGE:
        texts, layout, color = (
            ["2.3T/3T", "76.7%"],
            DISK_TEXT_LAYOUT,
            core.COLORS["cyan"],
        )
    elif visible == BALL_AVAILABILITY_PAGE:
        texts, color = window._ball_status_rows(BALL_AVAILABILITY_PAGE)
        texts = list(texts)
        layout = BALL_MESSAGE_TEXT_LAYOUT
    else:
        texts = (
            ["25.1%", "GPU 6", "19:08:39"]
            if visible == 1
            else ["5.3%", "GPU 3", "24:41:33"]
        )
        layout, color = TRAINING_TEXT_LAYOUT, remote.TRAINING_COLORS[visible]
    assert [item[2] for item in rows] == texts
    offset = -2 if progress < 0.5 else 2
    assert [item[1] for item in rows] == [y + offset for y in layout]
    assert all(item[3]["fill"] == color and item[3]["opacity"] == 0.5 for item in rows)


def test_hover_interrupts_disk_animation_and_retains_disk_in_rotation(
    disk_window, drawn_text
):
    window = disk_window
    publish_training(window)
    publish_disk(window, "2.3T")
    window.set_compact(True)
    window._ball_training_index = 1
    window._next_ball_training()
    window.training_animation.pause()
    window._animate_ball_training(0.75)
    drawn_text.clear()
    move_ball_pointer(window, training_ring_position(window, 0))
    assert window.displayed_training_index == 0 and window._ball_previous_page is None
    assert not window.training_timer.isActive()
    assert [item[2] for item in ball_training_rows(drawn_text)] == [
        "5.3%",
        "GPU 3",
        "24:41:33",
    ]
    move_ball_pointer(window, QPoint(window.ball_size // 2, window.ball_size // 2))
    assert window.training_timer.isActive() and BALL_DISK_PAGE in window._ball_page_set


def test_disk_refresh_close_only_signal_owned_workers(disk_window, monkeypatch):
    window = disk_window
    calls = []
    monkeypatch.setattr(
        window.disk_poller, "refresh", lambda: calls.append("disk refresh")
    )
    monkeypatch.setattr(window.disk_poller, "stop", lambda: calls.append("disk stop"))
    monkeypatch.setattr(
        window.training_poller, "stop", lambda: calls.append("training stop")
    )
    window.force_refresh()
    window.close()
    assert calls == ["disk refresh", "training stop", "disk stop"]


@pytest.mark.parametrize("extra", [[], ["--no-training"], ["--server-id", "server1"]])
def test_disk_defaults_on_for_wn_a100_independently_of_training(window, extra):
    other = MonitorWindow(
        core.build_parser().parse_args(["--demo", *extra]), start_pollers=False
    )
    try:
        other.set_compact(True)
        enabled = "server1" not in extra
        assert other.disk_enabled is enabled
        assert (other.disk_poller is not None) is enabled
        assert (BALL_DISK_PAGE in other._ball_page_set) is enabled
        if "--no-training" in extra:
            assert other.training_poller is None
            assert other._ball_page_set == (BALL_SUMMARY_PAGE, BALL_DISK_PAGE)
            assert other.training_timer.isActive() is other.isVisible()
    finally:
        other.close()


@pytest.mark.parametrize("disk_enabled", [True, False])
@pytest.mark.parametrize(
    "log_state",
    ["current", "incomplete", "empty", "ssh_error", "no_config", "disabled"],
)
def test_no_own_gpu_processes_hide_old_jobs_but_preserve_enabled_disk_carousel(
    disk_window, drawn_text, drawn_arcs, disk_enabled, log_state
):
    window = disk_window
    window.disk_enabled = disk_enabled
    window.online = True
    snapshot = snapshot_with_own_cards(window.snapshot, 0)
    window.snapshot = replace(
        snapshot,
        gpus=tuple(
            replace(
                gpu,
                utilization=0 if gpu.index in (0, 2) else 70,
                memory_percent=0 if gpu.index in (0, 2) else 60,
            )
            for gpu in snapshot.gpus
        ),
    )
    publish_notices(window, core.Announcement("alice", "gpu_usage", "申请2号卡", (2,)))
    publish_disk(window, "2.3T")
    samples = remote.demo_samples(window.training_config)
    if log_state == "current":
        samples = (
            replace(samples[0], progress=None, gpu_indexes=()),
            replace(
                samples[1],
                progress=remote.TrainingProgress(20000, 20000, eta_seconds=0),
                gpu_indexes=(),
            ),
        )
    elif log_state == "empty":
        samples = tuple(replace(sample, progress=None) for sample in samples)
    publish_training(window, samples)
    if log_state == "ssh_error":
        publish_training(window, error="SSH disconnected")
    elif log_state == "no_config":
        window.training_config = None
        window.training_error = "训练配置不可用"
    elif log_state == "disabled":
        window.training_enabled = False
    window.set_compact(True)
    drawn_text.clear()
    drawn_arcs.clear()
    window._sync_training_timer()
    window._render()
    assert not window.own_gpu_indexes
    assert not window.all_training_steps_complete
    assert window.ball_hide_training
    assert window._ball_page_set == (
        (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
        if disk_enabled
        else (BALL_AVAILABILITY_PAGE,)
    )
    assert window.displayed_training_index == BALL_AVAILABILITY_PAGE
    assert window.training_timer.isActive() is disk_enabled
    assert drawn_arcs == []
    assert [item[2] for item in drawn_text] == ["1张", "可用 GPU", "0"]
    for index in (0, 1):
        assert window._training_ring_at(training_ring_position(window, index)) is None
    window._next_ball_training()
    finish_ball_transition(window)
    if disk_enabled:
        assert window.displayed_training_index == BALL_DISK_PAGE
        assert [
            item[2] for item in draw_training_task(window, drawn_text, BALL_DISK_PAGE)
        ] == ["2.3T/3T", "76.7%"]
        window._next_ball_training()
        finish_ball_transition(window)
    assert window.displayed_training_index == BALL_AVAILABILITY_PAGE


@pytest.mark.parametrize("state", ["hover0", "hover1", "fadeout", "fadein"])
def test_last_own_gpu_process_leaving_cancels_hover_and_old_animation_frames(
    disk_window, drawn_text, drawn_arcs, state
):
    window = disk_window
    window.online = True
    publish_training(window)
    publish_notices(window)
    window.set_compact(True)
    if state.startswith("hover"):
        move_ball_pointer(window, training_ring_position(window, int(state[-1])))
    else:
        window._ball_training_index = 1
        window._next_ball_training()
        window.training_animation.pause()
        window._animate_ball_training(0.25 if state == "fadeout" else 0.75)
    drawn_text.clear()
    drawn_arcs.clear()
    window.events.put(
        core.PollEvent(snapshot=snapshot_with_own_cards(window.snapshot, 0))
    )
    window._drain_events()
    assert window._hover_training_index is None
    assert window._ball_previous_page is None
    assert window.training_animation.state() == QAbstractAnimation.Stopped
    assert window._ball_transition_progress == 1
    assert window._ball_page_set == (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
    assert window.training_timer.isActive()
    assert drawn_arcs == []
    expected_page = (
        BALL_DISK_PAGE if state.startswith("fade") else BALL_AVAILABILITY_PAGE
    )
    assert window.displayed_training_index == expected_page
    assert [item[2] for item in drawn_text] == list(
        window._ball_status_rows(expected_page)[0]
    )
    assert all(item[3]["opacity"] == 1 for item in drawn_text)


def test_gpu_process_reappearing_restores_its_log_and_rotation_without_relaunch(
    disk_window, drawn_text, drawn_arcs
):
    window = disk_window
    window.online = True
    publish_notices(window)
    publish_training(window)
    window.set_compact(True)
    snapshot = window.snapshot
    for own_count in (0, 1, 2, 0):
        drawn_text.clear()
        drawn_arcs.clear()
        window.events.put(
            core.PollEvent(snapshot=snapshot_with_own_cards(snapshot, own_count))
        )
        window._drain_events()
        if own_count == 0:
            assert window._ball_page_set == (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
            assert window.training_timer.isActive()
            assert drawn_arcs == []
        else:
            assert not window.ball_hide_training
            expected = (
                (0, BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
                if own_count == 1
                else (0, 1, BALL_DISK_PAGE)
            )
            assert window._ball_page_set == expected
            assert window.training_timer.isActive()
            assert drawn_arcs


def test_idle_gpu_outage_keeps_disk_carousel_without_claiming_stale_free_cards(
    disk_window, drawn_text
):
    window = disk_window
    window.online = True
    publish_notices(window)
    publish_training(window)
    publish_disk(window, "2.3T")
    snapshot = snapshot_with_own_cards(window.snapshot, 0)
    window.events.put(core.PollEvent(snapshot=snapshot))
    window._drain_events()
    window.set_compact(True)
    drawn_text.clear()
    window.events.put(core.PollEvent(error="GPU API offline"))
    window._drain_events()
    assert window.ball_hide_training
    assert not window.show_ball_availability
    assert window._ball_page_set == (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
    assert window.training_timer.isActive()
    assert [item[2] for item in drawn_text] == ["…", "GPU 监控", "离线"]
    window._next_ball_training()
    finish_ball_transition(window)
    assert window.displayed_training_index == BALL_DISK_PAGE
    assert [
        item[2] for item in draw_training_task(window, drawn_text, BALL_DISK_PAGE)
    ] == ["2.3T/3T", "76.7%"]
    window._next_ball_training()
    finish_ball_transition(window)
    drawn_text.clear()
    window.events.put(core.PollEvent(snapshot=snapshot))
    window._drain_events()
    assert window.online and window.ball_hide_training
    assert drawn_text[1][2] == "可用 GPU"
    assert window.training_timer.isActive()


def test_unknown_gpu_snapshot_is_not_misinterpreted_as_no_user_processes(disk_window):
    window = disk_window
    window.online = False
    window.snapshot = None
    publish_training(window)
    window.set_compact(True)
    assert not window.ball_hide_training
    assert not window.show_ball_availability
    assert window._ball_page_set == (0, 1, BALL_DISK_PAGE)


def test_idle_disk_carousel_updates_usage_and_retains_quota_alert_on_both_pages(
    disk_window, drawn_text
):
    window = disk_window
    window.online = True
    window.snapshot = snapshot_with_own_cards(window.snapshot, 0)
    publish_notices(window)
    publish_training(window)
    window.set_compact(True)
    for size in ("2.3T", "3.1T"):
        draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
        drawn_text.clear()
        publish_disk(window, size)
        assert window._ball_page_set == (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
        assert window.training_timer.isActive()
        assert len(drawn_text) == 3 and drawn_text[1][2] == "可用 GPU"
        assert not any("/3T" in item[2] for item in drawn_text)
        rows = draw_training_task(window, drawn_text, BALL_DISK_PAGE)
        assert [item[2] for item in rows] == [
            f"{size}/3T",
            f"{window.disk_sample.used_bytes / remote.QUOTA_BYTES:.1%}",
        ]
        assert all(
            item[3]["fill"] == core.COLORS["red" if size == "3.1T" else "cyan"]
            for item in rows
        )
    assert ball_ring_color(window) == ImageColor.getcolor(core.COLORS["red"], "RGBA")
    draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
    assert ball_ring_color(window) == ImageColor.getcolor(core.COLORS["red"], "RGBA")
    assert "磁盘占用超过 3T" in window.toolTip()


@pytest.mark.parametrize("own_count", [0, 1, 2])
@pytest.mark.parametrize("notices_known", [True, False])
@pytest.mark.parametrize("disk_enabled", [True, False])
def test_completed_logs_hide_old_jobs_and_only_idle_cards_keep_disk_carousel(
    disk_window, drawn_text, drawn_arcs, own_count, notices_known, disk_enabled
):
    window = disk_window
    window.disk_enabled = disk_enabled
    window.online = True
    snapshot = snapshot_with_own_cards(window.snapshot, own_count)
    window.snapshot = replace(
        snapshot,
        gpus=tuple(
            replace(gpu, utilization=0, memory_percent=0) for gpu in snapshot.gpus
        ),
    )
    publish_notices(window, core.Announcement("alice", "gpu_usage", "申请0号卡", (0,)))
    if not notices_known:
        window.notice_error = "notice unavailable"
    publish_disk(window, "2.3T")
    window.set_compact(True)
    drawn_text.clear()
    drawn_arcs.clear()
    publish_training(window, completed_training_samples(window))
    assert window.all_training_steps_complete
    assert window.show_ball_availability
    disk_rotates = disk_enabled and own_count == 0
    assert window._ball_page_set == (
        (BALL_AVAILABILITY_PAGE, BALL_DISK_PAGE)
        if disk_rotates
        else (BALL_AVAILABILITY_PAGE,)
    )
    assert window.displayed_training_index == BALL_AVAILABILITY_PAGE
    assert window.training_timer.isActive() is disk_rotates
    assert window.training_animation.state() == QAbstractAnimation.Stopped
    assert drawn_arcs == []
    indexes = frozenset(range(8)) - window.own_gpu_indexes
    if notices_known:
        indexes -= {0}
    assert [item[2] for item in drawn_text] == [
        f"{len(indexes)}张",
        "可用 GPU" if notices_known else "空闲 GPU",
        _compact_gpu_indexes(indexes),
    ]
    assert not any("100%" in item[2] or "/3T" in item[2] for item in drawn_text)
    for index in (0, 1):
        position = training_ring_position(window, index)
        assert window._training_ring_at(position) is None
        move_ball_pointer(window, position)
        window._next_ball_training()
        finish_ball_transition(window)
        if disk_rotates:
            assert window.displayed_training_index == BALL_DISK_PAGE
            window._next_ball_training()
            finish_ball_transition(window)
        assert window.displayed_training_index == BALL_AVAILABILITY_PAGE


@pytest.mark.parametrize("hovered", [0, 1])
def test_finishing_while_hovering_a_ring_clears_the_old_task_override(
    disk_window, drawn_text, hovered
):
    window = disk_window
    window.online = True
    publish_notices(window)
    publish_training(window)
    window.set_compact(True)
    move_ball_pointer(window, training_ring_position(window, hovered))
    assert window._hover_training_index == hovered
    drawn_text.clear()
    publish_training(window, completed_training_samples(window))
    assert window._hover_training_index is None
    assert window.displayed_training_index == BALL_AVAILABILITY_PAGE
    assert len(ball_status_calls(drawn_text)) == 3
    assert ball_status_calls(drawn_text)[1][2] == "可用 GPU"


@pytest.mark.parametrize("progress", [0.25, 0.75])
def test_finishing_during_a_disk_transition_cancels_all_old_page_frames(
    disk_window, drawn_text, progress
):
    window = disk_window
    window.online = True
    publish_training(window)
    publish_disk(window, "3.1T")
    publish_notices(window)
    window.set_compact(True)
    window._ball_training_index = 1
    window._next_ball_training()
    window.training_animation.pause()
    window._animate_ball_training(progress)
    drawn_text.clear()
    publish_training(window, completed_training_samples(window))
    assert window._ball_previous_page is None
    assert window._ball_transition_progress == 1
    assert window._ball_page_set == (BALL_AVAILABILITY_PAGE,)
    rows = ball_status_calls(drawn_text)
    assert len(rows) == 3 and rows[1][2] == "可用 GPU"
    assert all(item[3]["opacity"] == 1 for item in rows)
    assert ball_ring_color(window) == ImageColor.getcolor(core.COLORS["red"], "RGBA")
    assert "磁盘占用超过 3T" in window.toolTip()


@pytest.mark.parametrize("log_index", [0, 1])
@pytest.mark.parametrize("reset", [False, True])
def test_new_or_reset_log_restores_training_rings_and_disk_rotation(
    disk_window, drawn_text, drawn_arcs, log_index, reset
):
    window = disk_window
    window.online = True
    window.snapshot = snapshot_with_own_cards(window.snapshot, 2)
    publish_notices(window)
    window.set_compact(True)
    completed = completed_training_samples(window)
    publish_training(window, completed)
    assert window._ball_page_set == (BALL_AVAILABILITY_PAGE,)
    updated = list(completed)
    updated[log_index] = replace(
        updated[log_index],
        progress=None if reset else remote.TrainingProgress(1, 20000, eta_seconds=60),
        file_age=0,
        checked_at=monotonic(),
    )
    drawn_text.clear()
    drawn_arcs.clear()
    publish_training(window, tuple(updated))
    assert not window.all_training_steps_complete
    assert not window.show_ball_availability
    expected = (1 - log_index,) if reset else (0, 1)
    assert window._ball_page_set == expected + (BALL_DISK_PAGE,)
    assert window.displayed_training_index == expected[0]
    assert window.training_timer.isActive()
    assert len(drawn_arcs) >= 2
    assert window._training_ring_at(training_ring_position(window, log_index)) == (
        None if reset else log_index
    )


@pytest.mark.parametrize("log_index", [0, 1])
@pytest.mark.parametrize(
    "state", ["incomplete", "rounded", "missing", "error", "unchecked"]
)
def test_only_two_successfully_read_final_steps_trigger_availability_mode(
    disk_window, log_index, state
):
    window = disk_window
    window.online = True
    window.set_compact(True)
    samples = list(completed_training_samples(window))
    changes = {
        "incomplete": {"progress": remote.TrainingProgress(100, 20000)},
        "rounded": {"progress": remote.TrainingProgress(19999, 20000)},
        "missing": {"progress": None},
        "error": {"error": "SSH disconnected"},
        "unchecked": {"checked_at": 0},
    }
    samples[log_index] = replace(samples[log_index], **changes[state])
    publish_training(window, tuple(samples))
    assert not window.all_training_steps_complete
    expected = (1 - log_index,) if state == "missing" else (0, 1)
    assert window._ball_page_set == expected + (BALL_DISK_PAGE,)


def test_completed_logs_do_not_claim_old_cards_available_during_gpu_outage(
    disk_window, drawn_text
):
    window = disk_window
    window.online = True
    publish_notices(window)
    window.set_compact(True)
    publish_training(window, completed_training_samples(window))
    snapshot = window.snapshot
    drawn_text.clear()
    window.events.put(core.PollEvent(error="GPU API offline"))
    window._drain_events()
    assert window.all_training_steps_complete
    assert not window.show_ball_availability
    assert [item[2] for item in drawn_text] == ["…", "GPU 监控", "离线"]
    assert window._ball_page_set == (BALL_AVAILABILITY_PAGE,)
    drawn_text.clear()
    window.events.put(core.PollEvent(snapshot=snapshot))
    window._drain_events()
    assert window.online and window.show_ball_availability
    assert ball_status_calls(drawn_text)[1][2] == "可用 GPU"


@pytest.mark.parametrize("task_index", [0, 1])
def test_single_card_rotates_its_actual_log_and_availability_not_another_log(
    window, drawn_text, drawn_arcs, task_index
):
    samples = tuple(
        replace(sample, gpu_indexes=(gpu,))
        for sample, gpu in zip(
            remote.demo_samples(window.training_config), (6, 0), strict=True
        )
    )
    window.snapshot = snapshot_with_single_owned_gpu(
        window.snapshot, samples[task_index].gpu_indexes[0]
    )
    window.online = True
    publish_notices(window)
    publish_training(window, samples)
    window.set_compact(True)
    assert window._ball_page_set == (task_index, BALL_AVAILABILITY_PAGE)
    assert window.displayed_training_index == task_index
    rows = draw_training_task(window, drawn_text, task_index)
    expected = [item[2] for item in rows]
    assert rows[1][2] == ("GPU 6", "GPU 0")[task_index]
    assert rows[0][2] == ("5.3%", "25.1%")[task_index]
    assert not any(item[2].startswith("可用") for item in drawn_text)
    for expected_page in (BALL_AVAILABILITY_PAGE, task_index, BALL_AVAILABILITY_PAGE):
        window._next_ball_training()
        assert window.displayed_training_index == expected_page
        finish_ball_transition(window)
        drawn_text.clear()
        drawn_arcs.clear()
        window._render()
        rows = ball_training_rows(drawn_text)
        assert len(rows) == 3
        if expected_page == BALL_AVAILABILITY_PAGE:
            assert rows[0][2] == f"{len(window.available_gpu_indexes)}张"
            assert rows[1][2] == "可用 GPU"
            assert rows[2][2] == _compact_gpu_indexes(window.available_gpu_indexes)
            assert all(item[3]["fill"] == core.COLORS["cyan"] for item in rows)
        else:
            assert [item[2] for item in rows] == expected
            assert all(
                item[3]["fill"] == remote.TRAINING_COLORS[task_index] for item in rows
            )
        arcs = [arc for arc in drawn_arcs if arc["fill"] in remote.TRAINING_COLORS]
        assert len(arcs) == 1
        assert arcs[0]["end"] == pytest.approx(
            -90 + 360 * (1060 if task_index == 0 else 5033) / 20000
        )


def test_single_card_waits_for_matching_log_instead_of_assigning_old_task(
    window, drawn_text
):
    window.snapshot = snapshot_with_single_owned_gpu(window.snapshot, 0)
    window.online = True
    publish_notices(window)
    publish_training(window)  # Existing logs belong to GPU 3 and GPU 6, not 0.
    window.set_compact(True)
    assert window._ball_page_set == (BALL_AVAILABILITY_PAGE,)
    assert not window.training_timer.isActive()
    rows = draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
    assert rows[1][2] == "可用 GPU"
    assert "5.3%" not in [item[2] for item in drawn_text]
    assert "25.1%" not in [item[2] for item in drawn_text]
    samples = (
        window.training_samples[0],
        replace(window.training_samples[1], gpu_indexes=(0,)),
    )
    drawn_text.clear()
    publish_training(window, samples)
    assert window._ball_page_set == (1, BALL_AVAILABILITY_PAGE)
    assert [item[2] for item in ball_training_rows(drawn_text)] == [
        "25.1%",
        "GPU 0",
        "19:08:39",
    ]


def test_single_card_available_information_excludes_own_card_even_if_low_load(
    window, drawn_text
):
    window.snapshot = snapshot_with_single_owned_gpu(window.snapshot, 0)
    window.snapshot = replace(
        window.snapshot,
        gpus=tuple(
            replace(gpu, utilization=0, memory_percent=0)
            for gpu in window.snapshot.gpus
        ),
    )
    window.online = True
    publish_notices(window)
    window.set_compact(True)
    assert 0 not in window.available_gpu_indexes
    rows = draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
    assert [item[2] for item in rows] == ["7张", "可用 GPU", "1–7"]


@pytest.mark.parametrize("size", [104, 128, 160, 240])
def test_available_card_numbers_keep_all_ids_and_fit_the_ball_at_every_size(
    window, drawn_text, size
):
    window.snapshot = snapshot_with_single_owned_gpu(window.snapshot, 7)
    window.snapshot = replace(
        window.snapshot,
        gpus=tuple(
            replace(gpu, utilization=0, memory_percent=0)
            for gpu in window.snapshot.gpus
        ),
    )
    window.online = True
    publish_notices(
        window, core.Announcement("request", "gpu_usage", "申请2号卡", (2, 5))
    )
    window.set_compact(True)
    window.set_ball_size(size)
    rows = draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
    assert [item[2] for item in rows] == ["5张", "可用 GPU", "0–1、3–4、6"]
    for item, limit in zip(rows, (56, 56, 50), strict=True):
        assert window.canvas.measure_text(item[2], item[3]["font"])[0] <= limit
    publish_notices(
        window, core.Announcement("request", "gpu_usage", "全部申请", tuple(range(8)))
    )
    rows = draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
    assert [item[2] for item in rows] == ["0张", "可用 GPU", "暂无"]


def test_compact_gpu_ranges_round_trip_every_mask_without_omitting_card_zero():
    for mask in range(256):
        indexes = frozenset(index for index in range(8) if mask & (1 << index))
        text = _compact_gpu_indexes(indexes)
        expanded = set()
        for group in text.split("、") if indexes else ():
            if "–" in group:
                first, last = map(int, group.split("–"))
                expanded.update(range(first, last + 1))
            else:
                expanded.add(int(group))
        assert expanded == indexes
    assert _compact_gpu_indexes(frozenset()) == "暂无"


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.49, 0.5, 0.51, 0.75, 1.0])
def test_smooth_transition_moves_and_fades_all_three_fields_as_one_group(
    window, drawn_text, progress
):
    window.set_compact(True)
    publish_training(window)
    window._next_ball_training()
    window.training_animation.pause()
    before = window.canvas._image.crop(
        (0, 0, window.canvas._scaled(104), window.canvas._scaled(20))
    )
    drawn_text.clear()
    window._animate_ball_training(progress)
    rows = ball_status_calls(drawn_text)
    assert len(rows) == 3
    expected_index = 0 if progress < 0.5 else 1
    assert [item[2] for item in rows] == (
        ["5.3%", "GPU 3", "24:41:33"]
        if expected_index == 0
        else ["25.1%", "GPU 6", "19:08:39"]
    )
    opacity = abs(1 - 2 * progress)
    offset = (-4 if progress < 0.5 else 4) * (1 - opacity)
    assert all(
        item[3]["fill"] == remote.TRAINING_COLORS[expected_index] for item in rows
    )
    assert [item[3]["opacity"] for item in rows] == pytest.approx([opacity] * 3)
    assert [item[1] for item in rows] == pytest.approx(
        [y + offset for y in TRAINING_TEXT_LAYOUT]
    )
    after = window.canvas._image.crop(
        (0, 0, window.canvas._scaled(104), window.canvas._scaled(20))
    )
    assert (
        ImageChops.difference(before.convert("RGB"), after.convert("RGB")).getbbox()
        is None
    )
    assert (
        window.canvas._image.getpixel(
            (window.canvas._scaled(51.5), window.canvas._scaled(51.5))
        )[3]
        == 255
    )


def test_transition_emits_intermediate_frames_then_finishes_without_a_hard_cut(window):
    window.set_compact(True)
    publish_training(window)
    assert window.training_animation.duration() == TRAINING_TRANSITION_DURATION == 480
    frames = QSignalSpy(window.training_animation.valueChanged)
    finished = QSignalSpy(window.training_animation.finished)
    window._next_ball_training()
    assert window._ball_previous_page == 0
    QTest.qWait(75)
    assert 0 < window._ball_transition_progress < 0.5
    assert finished.wait(1500)
    assert len(frames) >= 3
    assert window._ball_previous_page is None
    assert window._ball_transition_progress == 1.0
    assert window.displayed_training_index == 1


@pytest.mark.parametrize("index", [0, 1])
def test_hover_during_transition_immediately_shows_the_selected_task(
    window, drawn_text, index
):
    window.set_compact(True)
    publish_training(window)
    window._next_ball_training()
    window.training_animation.pause()
    window._animate_ball_training(0.25)
    drawn_text.clear()
    move_ball_pointer(window, training_ring_position(window, index))
    assert window.training_animation.state() == QAbstractAnimation.Stopped
    assert window._ball_previous_page is None
    assert window.displayed_training_index == index
    assert not window.training_timer.isActive()
    rows = ball_training_rows(drawn_text)
    assert len(rows) == 3
    assert rows[1][2] == ("GPU 3", "GPU 6")[index]
    assert all(item[3]["opacity"] == 1 for item in rows)
    move_ball_pointer(window, QPoint(window.ball_size // 2, window.ball_size // 2))
    assert window.training_timer.isActive()


@pytest.mark.parametrize("action", ["hide", "minimize", "expand", "drag", "close"])
def test_ball_transition_stops_when_hidden_minimized_expanded_dragged_or_closed(
    window, action
):
    window.set_compact(True)
    publish_training(window)
    window._next_ball_training()
    window.training_animation.pause()
    if action == "expand":
        window.set_compact(False)
    elif action == "drag":
        QTest.mousePress(
            window,
            Qt.LeftButton,
            pos=QPoint(window.ball_size // 2, window.ball_size // 2),
        )
    else:
        getattr(window, action)()
    assert window.training_animation.state() == QAbstractAnimation.Stopped
    assert window._ball_previous_page is None
    assert not window.training_timer.isActive()


@pytest.mark.parametrize("change", ["second_card", "offline", "new_log_mapping"])
def test_single_card_animation_never_keeps_stale_availability_or_wrong_task_on_changes(
    window, drawn_text, change
):
    window.online = True
    window.snapshot = snapshot_with_own_cards(window.snapshot, 1)
    publish_notices(window)
    publish_training(window)
    window.set_compact(True)
    window._next_ball_training()
    window.training_animation.pause()
    window._animate_ball_training(0.75)
    drawn_text.clear()
    if change == "second_card":
        window.events.put(core.PollEvent(snapshot=core.demo_snapshot()))
        window._drain_events()
        assert window._ball_page_set == (0, 1)
        assert not any(item[2].startswith(("可用", "空闲")) for item in drawn_text)
    elif change == "offline":
        window.events.put(core.PollEvent(error="GPU API offline"))
        window._drain_events()
        assert window._ball_page_set == (0, 1)
        assert "连接中断" in window.toolTip()
        assert not any(
            item[2] == "离线" for item in drawn_text
        )  # No crowded fourth line.
        assert not any(item[2].startswith(("可用", "空闲")) for item in drawn_text)
    else:
        samples = (
            replace(window.training_samples[0], gpu_indexes=(6,)),
            replace(window.training_samples[1], gpu_indexes=(3,)),
        )
        publish_training(window, samples)
        assert window._ball_page_set == (1, BALL_AVAILABILITY_PAGE)
        draw_training_task(window, drawn_text, 1)
        assert [item[2] for item in ball_training_rows(drawn_text)] == [
            "25.1%",
            "GPU 3",
            "19:08:39",
        ]
    assert window._ball_previous_page is None
    assert window.training_animation.state() == QAbstractAnimation.Stopped


@pytest.mark.parametrize("opacity", [0.0, 0.25, 0.5, 1.0])
def test_text_opacity_composites_over_background_instead_of_making_transparent_holes(
    opacity,
):
    canvas = PillowCanvas(100, 70)
    canvas.create_rectangle(0, 0, 100, 70, fill="#102030")
    before = canvas._image.copy()
    canvas.create_text(
        50,
        35,
        text="GPU 0",
        fill="#FFFFFF",
        font=("Noto Sans CJK SC", 14, "bold"),
        anchor="center",
        opacity=opacity,
    )
    assert canvas._image.getchannel("A").getextrema() == (255, 255)
    if opacity == 0:
        assert ImageChops.difference(before, canvas._image).getbbox() is None
    else:
        assert canvas._image.getchannel("R").getextrema()[1] == pytest.approx(
            16 + (255 - 16) * opacity, abs=1
        )


def test_ball_carousel_rotates_one_task_without_changing_either_progress_ring(
    window, drawn_text, drawn_arcs
):
    window.set_compact(True)
    publish_training(window)
    assert window.displayed_training_index == 0
    assert window.training_timer.interval() == TRAINING_CAROUSEL_INTERVAL == 4000
    assert window.training_timer.isActive()
    for expected_index in (1, 0, 1):
        drawn_text.clear()
        drawn_arcs.clear()
        window._next_ball_training()
        assert window.displayed_training_index == expected_index
        finish_ball_transition(window)
        drawn_text.clear()
        drawn_arcs.clear()
        window._render()
        rows = ball_training_rows(drawn_text)
        assert [item[2] for item in rows] == (
            ["5.3%", "GPU 3", "24:41:33"]
            if expected_index == 0
            else ["25.1%", "GPU 6", "19:08:39"]
        )
        assert all(
            item[3]["fill"] == remote.TRAINING_COLORS[expected_index] for item in rows
        )
        arcs = [arc for arc in drawn_arcs if arc["fill"] in remote.TRAINING_COLORS]
        assert [arc["end"] for arc in arcs] == pytest.approx(
            [-90 + 360 * 1060 / 20000, -90 + 360 * 5033 / 20000]
        )


def test_ball_carousel_timer_really_advances_the_central_task(window, drawn_text):
    window.set_compact(True)
    publish_training(window)
    QTest.mouseMove(window, QPoint(window.ball_size // 2, window.ball_size // 2))
    QApplication.processEvents()
    window.training_timer.stop()
    window.training_timer.setInterval(80)
    # Stop after one signal so slower rendering cannot cause a second flip.
    window.training_timer.timeout.connect(window.training_timer.stop)
    signals = QSignalSpy(window.training_timer.timeout)
    animation_finished = QSignalSpy(window.training_animation.finished)
    drawn_text.clear()
    window.training_timer.start()
    assert signals.wait(1500)
    assert len(signals) == 1
    assert window.displayed_training_index == 1
    assert animation_finished.wait(1500)
    assert [item[2] for item in ball_training_rows(drawn_text)[-3:]] == [
        "25.1%",
        "GPU 6",
        "19:08:39",
    ]


@pytest.mark.parametrize("size", [104, 128, 160, 240])
@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_hover_each_ring_selects_matching_three_line_status_and_pauses_carousel(
    window, drawn_text, size, angle
):
    window.set_compact(True)
    window.set_ball_size(size)
    publish_training(window)
    for index in (1, 0, 1):
        position = training_ring_position(window, index, angle)
        assert window._training_ring_at(position) == index
        drawn_text.clear()
        move_ball_pointer(window, position)
        assert window.displayed_training_index == index
        assert window._hover_training_index == index
        assert not window.training_timer.isActive()
        rows = ball_training_rows(drawn_text)
        assert len(rows) == 3
        assert rows[1][2] == ("GPU 3", "GPU 6")[index]
        assert rows[2][2] == ("24:41:33", "19:08:39")[index]
        assert all(item[3]["fill"] == remote.TRAINING_COLORS[index] for item in rows)
        window._next_ball_training()
        assert window.displayed_training_index == index
    center = QPoint(size // 2, size // 2)
    assert window._training_ring_at(center) is None
    move_ball_pointer(window, center)
    assert window._hover_training_index is None
    assert window.displayed_training_index == 1
    assert window.training_timer.isActive()
    window._next_ball_training()
    assert window.displayed_training_index == 0
    # The outline is a connection indicator, not a third training ring.
    assert window._training_ring_at(QPoint(size // 2, round(size / 104))) is None


def test_hovered_task_keeps_receiving_its_own_live_eta_and_gpu_updates(
    window, drawn_text, drawn_arcs
):
    window.set_compact(True)
    publish_training(window)
    move_ball_pointer(window, training_ring_position(window, 1))
    samples = tuple(
        replace(
            sample,
            gpu_indexes=(gpu,),
            progress=remote.TrainingProgress(step, 100, eta_seconds=eta),
        )
        for sample, gpu, step, eta in zip(
            window.training_samples, (6, 0), (30, 70), (60, 120), strict=True
        )
    )
    drawn_text.clear()
    drawn_arcs.clear()
    publish_training(window, samples)
    assert window.displayed_training_index == 1
    assert not window.training_timer.isActive()
    assert [item[2] for item in ball_training_rows(drawn_text)] == [
        "70.0%",
        "GPU 0",
        "00:02:00",
    ]
    arcs = [arc for arc in drawn_arcs if arc["fill"] in remote.TRAINING_COLORS]
    assert [arc["end"] for arc in arcs] == pytest.approx([18, 162])
    window.force_refresh()
    assert window._hover_training_index == 1
    assert not window.training_timer.isActive()
    QApplication.sendEvent(window, QEvent(QEvent.Leave))
    assert window._hover_training_index is None
    assert window.training_timer.isActive()


def test_carousel_pauses_while_dragging_hidden_minimized_and_expanded(window):
    window.set_compact(True)
    publish_training(window)
    center = QPoint(window.ball_size // 2, window.ball_size // 2)
    QTest.mousePress(window, Qt.LeftButton, pos=center)
    assert window._dragging and not window.training_timer.isActive()
    position = center + QPoint(40, 20)
    QApplication.sendEvent(
        window,
        QMouseEvent(
            QEvent.MouseMove,
            QPointF(position),
            QPointF(window.mapToGlobal(position)),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        ),
    )
    QTest.mouseRelease(window, Qt.LeftButton, pos=center)
    assert window.compact_mode and not window._dragging
    assert window.training_timer.isActive()
    QTest.qWait(120)
    window.hide()
    assert not window.training_timer.isActive()
    QTest.qWait(120)
    window.show()
    assert window.training_timer.isActive()
    QTest.qWait(120)
    window.minimize()
    assert not window.training_timer.isActive()
    # Let the real window manager acknowledge minimizing before restoring;
    # an immediate restore can be overtaken by its pending X11 state event.
    QTest.qWait(120)
    window.showNormal()
    QApplication.processEvents()
    QTest.qWait(120)
    assert not window.isMinimized()
    assert window.training_timer.isActive()
    window.set_compact(False)
    assert not window.training_timer.isActive()
    index = window.displayed_training_index
    window._next_ball_training()
    assert window.displayed_training_index == index
    window.set_compact(True)
    assert window.training_timer.isActive()
    window.close()
    assert not window.training_timer.isActive()


@pytest.mark.parametrize("no_config", [False, True])
def test_carousel_does_not_start_when_training_is_disabled_or_config_missing(
    window, no_config
):
    if no_config:
        window.training_config = None
    else:
        window.training_enabled = False
    window.set_compact(True)
    assert not window.training_timer.isActive()


def test_training_full_rings_and_central_percentages_are_independent_without_gpu_utilization(
    window, drawn_arcs, drawn_text
):
    window.online = True
    publish_notices(window, core.Announcement("a", "gpu_usage", "申请3号卡", (3,)))
    window.set_compact(True)
    samples = remote.demo_samples(window.training_config)
    samples = (
        replace(samples[0], progress=remote.TrainingProgress(1, 4, eta_seconds=60)),
        replace(samples[1], progress=remote.TrainingProgress(6, 10, eta_seconds=120)),
    )
    drawn_arcs.clear()
    drawn_text.clear()
    publish_training(window, samples)
    arcs = [arc for arc in drawn_arcs if arc["fill"] in remote.TRAINING_COLORS]
    assert len(arcs) == 2
    assert (arcs[0]["start"], arcs[0]["end"], arcs[0]["fill"]) == (
        -90,
        0,
        remote.TRAINING_COLORS[0],
    )
    assert arcs[1]["start"] == -90 and arcs[1]["end"] == pytest.approx(126)
    assert arcs[1]["fill"] == remote.TRAINING_COLORS[1]
    assert [arc["coordinates"] for arc in arcs] == [
        layout[:4] for layout in TRAINING_RING_LAYOUT
    ]
    assert [arc["width"] for arc in arcs] == [7, 6]
    for index, expected in enumerate(
        (("25.0%", "GPU 3", "00:01:00"), ("60.0%", "GPU 6", "00:02:00"))
    ):
        rows = draw_training_task(window, drawn_text, index)
        assert [item[2] for item in rows] == list(expected)
        assert all(item[3]["fill"] == remote.TRAINING_COLORS[index] for item in rows)
        assert [item for item in drawn_text if item[2].startswith("GPU ")] == [rows[1]]
    assert "训练" not in [item[2] for item in drawn_text]
    assert "GPU 均值" not in [item[2] for item in drawn_text]
    assert f"{window.snapshot.average_utilization:.0f}%" not in [
        item[2] for item in drawn_text
    ]
    assert "平均 GPU" not in window.toolTip()
    assert ball_ring_color(window) == ImageColor.getcolor(
        core.COLORS["yzh_border"], "RGBA"
    )
    text = window.toolTip()
    assert "training_2.log" in text and "training.log" in text
    assert (
        text.index("training_2.log")
        < text.index("25.00%")
        < text.index("00:01:00")
        < text.index("training.log")
        < text.index("60.00%")
        < text.index("00:02:00")
    )


def test_training_arc_unknown_zero_full_and_stale_states_are_not_fake_live_progress(
    window, drawn_arcs
):
    window.set_compact(True)
    drawn_arcs.clear()
    window._render()
    assert drawn_arcs == []  # Unknown jobs have neither tracks nor carousel pages.
    assert "等待日志" in window.toolTip() and "0.00%" not in window.toolTip()
    samples = remote.demo_samples(window.training_config)
    samples = (
        replace(samples[0], progress=remote.TrainingProgress(0, 20)),
        replace(samples[1], progress=remote.TrainingProgress(20, 20)),
    )
    drawn_arcs.clear()
    publish_training(window, samples)
    colored = [arc for arc in drawn_arcs if arc["fill"] in remote.TRAINING_COLORS]
    assert len(colored) == 1 and colored[0]["start"] == -90 and colored[0]["end"] == 270
    samples = (
        replace(samples[0], progress=remote.TrainingProgress(10, 20), file_age=181),
        samples[1],
    )
    drawn_arcs.clear()
    publish_training(window, samples)
    assert any(
        arc["fill"] == core.COLORS["subtle"] and arc["end"] == 90 for arc in drawn_arcs
    )
    assert "日志未更新" in window.toolTip()
    assert "上次日志估计，非当前倒计时" in window.toolTip()


@pytest.mark.parametrize(
    "etas, expected",
    [
        ((60, 120), ("00:01:00", "00:02:00")),
        ((88893, 68919), ("24:41:33", "19:08:39")),
        ((None, 0), ("待估算", "00:00:00")),
        ((360000, 3600000), ("100:00:00", "1000:00:00")),
    ],
)
def test_ball_percentage_eta_and_gpu_label_pairing_never_uses_gpu_sort_order(
    window, drawn_text, etas, expected
):
    window.set_compact(True)
    samples = remote.demo_samples(window.training_config)
    samples = tuple(
        replace(
            sample,
            gpu_indexes=(gpu,),
            progress=remote.TrainingProgress(step, 100, eta_seconds=eta),
        )
        for sample, gpu, step, eta in zip(samples, (6, 0), (25, 60), etas, strict=True)
    )
    publish_training(window, samples)
    for index in range(2):
        percentage, gpu, eta = draw_training_task(window, drawn_text, index)
        assert percentage[2] == ("25.0%", "60.0%")[index]
        assert gpu[2] == ("GPU 6", "GPU 0")[index]
        assert eta[2] == expected[index]
        assert percentage[1] < gpu[1] < eta[1]
        assert (
            percentage[3]["fill"]
            == gpu[3]["fill"]
            == eta[3]["fill"]
            == remote.TRAINING_COLORS[index]
        )
        assert window.canvas.measure_text(eta[2], eta[3]["font"])[0] <= 50
        assert len([item for item in drawn_text if item[2].startswith("GPU ")]) == 1
    assert "训练" not in [item[2] for item in drawn_text]


def test_ball_failed_or_stale_logs_gray_percentage_eta_and_gpu_together(
    window, drawn_text
):
    window.set_compact(True)
    samples = remote.demo_samples(window.training_config)
    samples = (replace(samples[0], file_age=181), samples[1])
    drawn_text.clear()
    publish_training(window, samples)
    first_rows = ball_training_rows(drawn_text)
    assert len(first_rows) == 3 and all(
        item[3]["fill"] == core.COLORS["subtle"] for item in first_rows
    )
    second_rows = draw_training_task(window, drawn_text, 1)
    assert all(item[3]["fill"] == remote.TRAINING_COLORS[1] for item in second_rows)
    publish_training(window, error="SSH disconnected")
    for index, expected in enumerate((("GPU 3", "24:41:33"), ("GPU 6", "19:08:39"))):
        rows = draw_training_task(window, drawn_text, index)
        assert all(item[3]["fill"] == core.COLORS["subtle"] for item in rows)
        assert [item[2] for item in rows[1:]] == list(expected)


@pytest.mark.parametrize("missing_index", [0, 1])
@pytest.mark.parametrize("state", ["waiting", "empty", "missing", "unparseable"])
def test_empty_training_pages_and_their_ring_hover_are_excluded(
    window, drawn_text, drawn_arcs, missing_index, state
):
    window.online = True
    samples = list(remote.demo_samples(window.training_config))
    samples[missing_index] = replace(
        samples[missing_index],
        progress=None,
        checked_at=0 if state == "waiting" else monotonic(),
        error="File not found" if state == "missing" else "",
        gpu_indexes=(
            ()
            if state in ("waiting", "missing")
            else samples[missing_index].gpu_indexes
        ),
    )
    window.set_compact(True)
    drawn_text.clear()
    drawn_arcs.clear()
    publish_training(window, tuple(samples))
    valid = 1 - missing_index
    assert window._ball_page_set == (valid,)
    assert not window.training_timer.isActive()
    assert window.displayed_training_index == valid
    assert (
        window._training_ring_at(training_ring_position(window, missing_index)) is None
    )
    assert window._training_ring_at(training_ring_position(window, valid)) == valid
    assert [item[2] for item in ball_status_calls(drawn_text)] == [
        ("5.3%", "25.1%")[valid],
        ("GPU 3", "GPU 6")[valid],
        ("24:41:33", "19:08:39")[valid],
    ]
    assert all(
        arc["coordinates"] == TRAINING_RING_LAYOUT[valid][:4] for arc in drawn_arcs
    )
    move_ball_pointer(window, training_ring_position(window, missing_index))
    assert window.displayed_training_index == valid


@pytest.mark.parametrize("missing_index", [0, 1])
def test_restored_missing_log_rejoins_both_rings_and_carousel(
    window, drawn_text, drawn_arcs, missing_index
):
    window.online = True
    samples = remote.demo_samples(window.training_config)
    missing = list(samples)
    missing[missing_index] = remote.TrainingSample(
        samples[missing_index].log,
        checked_at=monotonic(),
        error="[Errno 2] No such file or directory",
    )
    publish_training(window, tuple(missing))
    window.set_compact(True)
    assert window._ball_page_set == (1 - missing_index,)
    assert not window.training_timer.isActive()
    drawn_arcs.clear()
    publish_training(window, samples)
    assert window._ball_page_set == (0, 1)
    assert window.training_timer.isActive()
    assert {arc["fill"] for arc in drawn_arcs} >= set(remote.TRAINING_COLORS)
    for index in (0, 1):
        assert window._training_ring_at(training_ring_position(window, index)) == index
        rows = draw_training_task(window, drawn_text, index)
        assert [item[2] for item in rows] == [
            ("5.3%", "25.1%")[index],
            samples[index].gpu_label,
            remote.format_duration(samples[index].progress.eta_seconds),
        ]
    window._next_ball_training()
    finish_ball_transition(window)
    assert window.displayed_training_index == 0
    window._next_ball_training()
    finish_ball_transition(window)
    assert window.displayed_training_index == 1


@pytest.mark.parametrize("task", [0, 1])
def test_empty_log_replacing_visible_task_clears_old_animation_and_hover(
    window, drawn_text, task
):
    window.online = True
    publish_training(window)
    window.set_compact(True)
    window._next_ball_training()
    window.training_animation.pause()
    window._set_training_hover(task)
    samples = list(window.training_samples)
    samples[task] = replace(
        samples[task], progress=None, identity="new-empty-log", size=0
    )
    drawn_text.clear()
    publish_training(window, tuple(samples))
    assert window._ball_page_set == (1 - task,)
    assert window._hover_training_index is None
    assert window._ball_previous_page is None
    assert not window.training_timer.isActive()
    assert window.displayed_training_index == 1 - task
    assert "GPU ?" not in [item[2] for item in drawn_text]


def test_no_training_progress_uses_public_summary_then_restores_only_arriving_job(
    window, drawn_text
):
    window.online = True
    window.set_compact(True)
    assert window._ball_page_set == (BALL_SUMMARY_PAGE,)
    assert not window.training_timer.isActive()
    rows = draw_training_task(window, drawn_text, BALL_SUMMARY_PAGE)
    assert [item[2] for item in rows] == ["2张", "yzh 已用", "3、6"]
    samples = remote.demo_samples(window.training_config)
    publish_training(window, (replace(samples[0], progress=None), samples[1]))
    assert window._ball_page_set == (1,)
    assert window.displayed_training_index == 1


@pytest.mark.parametrize("size", [104, 128, 240])
@pytest.mark.parametrize(
    "state", ["disconnected", "pending", "checking", "config_error", "public_offline"]
)
def test_message_page_actual_glyphs_do_not_overlap_or_escape_ball(
    logged_out_ball, drawn_text, size, state
):
    window = logged_out_ball
    window.set_ball_size(size)
    if state == "pending":
        window.ssh_login_process = FakeLoginTerminal()
    elif state == "checking":
        window.ssh_connected = None
    elif state == "config_error":
        window.ssh_config = None
    elif state == "public_offline":
        window.online = False
        window.last_error = "GPU offline"
    page = BALL_SUMMARY_PAGE if state == "public_offline" else BALL_SSH_LOGIN_PAGE
    window._stop_ball_training_transition()
    window._ball_training_index = page
    drawn_text.clear()
    window._render()
    rows = ball_status_calls(drawn_text)
    assert [item[1] for item in rows] == list(BALL_MESSAGE_TEXT_LAYOUT)
    assert len(drawn_text) == 3  # In particular, no duplicated fourth offline row.
    bounds = [
        window.canvas._draw.textbbox(
            (window.canvas._scaled(x), window.canvas._scaled(y)),
            text,
            font=window.canvas._font(style["font"]),
            anchor="mm",
        )
        for x, y, text, style in rows
    ]
    clip = tuple(window.canvas._scaled(value) for value in (20, 25, 83, 80))
    assert all(
        left >= clip[0] and top >= clip[1] and right <= clip[2] and bottom <= clip[3]
        for left, top, right, bottom in bounds
    )
    gap = window.canvas._scaled(2)
    assert all(first[3] + gap <= second[1] for first, second in zip(bounds, bounds[1:]))


def test_ball_unknown_gpu_and_eta_do_not_invent_card_ids_or_completion_time(
    window, drawn_text
):
    window.set_compact(True)
    assert window._ball_page_set == (BALL_SUMMARY_PAGE,)
    assert not window.training_timer.isActive()
    assert not any(item[2] in ("GPU ?", "待估算") for item in drawn_text)
    # A real parsed progress value remains useful even if optional GPU / ETA
    # metadata is unavailable. Do not replace it with an invented card/time.
    publish_training(
        window,
        tuple(
            replace(
                sample,
                gpu_indexes=(),
                progress=replace(sample.progress, eta_seconds=None),
            )
            for sample in remote.demo_samples(window.training_config)
        ),
    )
    for index in range(2):
        rows = draw_training_task(window, drawn_text, index)
        assert [item[2] for item in rows] == [
            ("5.3%", "25.1%")[index],
            "GPU ?",
            "待估算",
        ]
        assert "00:00:00" not in [item[2] for item in drawn_text]


def test_training_ssh_outage_grays_both_values_then_recovers_and_gpu_channel_is_independent(
    window, drawn_arcs
):
    window.online = True
    window.set_compact(True)
    publish_training(window)
    steps = [sample.progress.step for sample in window.training_samples]
    drawn_arcs.clear()
    publish_training(window, error="SSH 共享连接不可用")
    assert [sample.progress.step for sample in window.training_samples] == steps
    assert not any(arc["fill"] in remote.TRAINING_COLORS for arc in drawn_arcs)
    assert sum(arc["fill"] == core.COLORS["subtle"] for arc in drawn_arcs) == 2
    assert window.online  # SSH outage must not turn the public GPU API offline.
    drawn_arcs.clear()
    publish_training(window)
    assert sum(arc["fill"] in remote.TRAINING_COLORS for arc in drawn_arcs) == 2
    drawn_arcs.clear()
    window.events.put(core.PollEvent(error="GPU API unavailable"))
    window._drain_events()
    assert not window.online
    assert sum(arc["fill"] in remote.TRAINING_COLORS for arc in drawn_arcs) == 2


@pytest.mark.parametrize("width", [650, 970, 1250])
def test_training_chips_fit_next_to_gpu_subtitle_and_keep_job_number_pairing(
    window, drawn_text, width
):
    publish_training(window)
    window.resize(width, window.height())
    QApplication.processEvents()
    QTest.qWait(100)
    drawn_text.clear()
    window._render()
    labels = [item for item in drawn_text if item[2].startswith(("训1 ", "训2 "))]
    assert [item[2] for item in labels] == ["训1 5.3%", "训2 25.2%"]
    assert [item[3]["fill"] for item in labels] == list(remote.TRAINING_COLORS)
    subtitle = next(item for item in drawn_text if item[0] == 22 and item[1] == 52)
    assert (
        subtitle[0]
        + window.canvas.measure_text(subtitle[2], subtitle[3]["font"])[0]
        + 10
        < labels[0][0]
    )
    assert (
        labels[-1][0]
        + window.canvas.measure_text(labels[-1][2], labels[-1][3]["font"])[0]
        < width - 18
    )


def test_training_refresh_and_close_stop_only_owned_pollers(window, monkeypatch):
    calls = []
    monkeypatch.setattr(
        window.training_poller, "refresh", lambda: calls.append("refresh")
    )
    monkeypatch.setattr(window.training_poller, "stop", lambda: calls.append("stop"))
    window.force_refresh()
    window.close()
    assert calls == ["refresh", "stop"]


def test_disabling_training_shows_card_counts_and_never_restores_gpu_utilization(
    window, drawn_arcs, drawn_text
):
    window.training_enabled = False
    window.online = True
    window.snapshot = snapshot_with_own_cards(window.snapshot, 1)
    window.set_compact(True)
    drawn_arcs.clear()
    drawn_text.clear()
    window._render()
    assert drawn_arcs == []
    assert f"{window.snapshot.idle_gpu_count}/{len(window.snapshot.gpus)}" in [
        item[2] for item in drawn_text
    ]
    assert not any("%" in item[2] for item in drawn_text)
    assert "平均 GPU" not in window.toolTip()
    assert "训练进度" not in window.toolTip()


def test_training_viewer_uses_cjk_font_plain_text_and_can_refresh(window, monkeypatch):
    from PyQt5.QtWidgets import QDialog, QTextEdit

    publish_training(window, error="<script>not active</script>")
    captured = []

    def inspect(dialog):
        viewer = dialog.findChild(QTextEdit)
        captured.append(
            (dialog.font().family(), viewer.toPlainText(), viewer.isReadOnly())
        )
        return 0

    monkeypatch.setattr(QDialog, "exec_", inspect)
    window.show_training()
    assert captured[0][0] == window.family
    assert captured[0][2]
    assert "<script>not active</script>" in captured[0][1]
    assert "ControlPersist=8h" in captured[0][1]


@pytest.mark.parametrize(
    ("step", "total", "label"),
    [
        (0, 20000, "0.0%"),
        (1, 20000, "<0.1%"),
        (1060, 20000, "5.3%"),
        (19999, 20000, "99.9%"),
        (20000, 20000, "100%"),
        (10**18 - 1, 10**18, "99.9%"),
    ],
)
def test_small_and_large_progress_labels_are_visible_without_false_completion(
    window, drawn_arcs, drawn_text, step, total, label
):
    samples = remote.demo_samples(window.training_config)
    samples = (
        replace(samples[0], progress=remote.TrainingProgress(step, total)),
        replace(samples[1], progress=None),
    )
    window.set_compact(True)
    drawn_arcs.clear()
    drawn_text.clear()
    publish_training(window, samples)
    assert _ball_training_percent(samples[0].progress) == label
    assert _ball_training_percent(None) == "…"
    values = ball_training_rows(drawn_text)
    assert [item[2] for item in values] == [label, "GPU 3", "待估算"]
    assert values[0][3]["font"][1] >= 12
    assert window.canvas.measure_text(label, values[0][3]["font"])[0] <= 56
    assert values[0][3]["fill"] == remote.TRAINING_COLORS[0]
    colored = [arc for arc in drawn_arcs if arc["fill"] in remote.TRAINING_COLORS]
    assert len(colored) == (1 if step else 0)
    if colored:
        assert colored[0]["end"] == pytest.approx(-90 + 360 * step / total)
    assert window._ball_page_set == (0,)
    assert window._training_ring_at(training_ring_position(window, 1)) is None
    assert not window.training_timer.isActive()


def test_round_caps_follow_real_endpoint_without_request_or_conflict_markers(
    window, monkeypatch, drawn_text
):
    import math

    ovals = []
    original = window.canvas.create_oval

    def record(*coordinates, **style):
        ovals.append((coordinates, style))
        return original(*coordinates, **style)

    monkeypatch.setattr(window.canvas, "create_oval", record)
    window.online = True
    publish_notices(
        window, core.Announcement("request", "gpu_usage", "申请3号卡", (3,))
    )
    window.set_compact(True)
    samples = remote.demo_samples(window.training_config)
    samples = (
        replace(samples[0], progress=remote.TrainingProgress(1, 20000)),
        replace(samples[1], progress=None),
    )
    ovals.clear()
    drawn_text.clear()
    publish_training(window, samples)
    lights = [
        coords for coords, style in ovals if style.get("fill") in ("#E8F4FF", "#F5ECFF")
    ]
    assert len(lights) == 1
    coords = lights[0]
    tip = ((coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2)
    angle = math.radians(-90 + 360 / 20000)
    outer = TRAINING_RING_LAYOUT[0]
    radius = (outer[2] - outer[0] - outer[4]) / 2
    assert tip == pytest.approx(
        (51.5 + radius * math.cos(angle), 51.5 + radius * math.sin(angle))
    )
    assert not any(item[2] in ("申请", "冲突", "待确认") for item in drawn_text)
    assert ball_ring_color(window) == ImageColor.getcolor(
        core.COLORS["yzh_border"], "RGBA"
    )
    assert not any(
        style.get("fill") in (core.COLORS["red"], core.COLORS["conflict_border"])
        for _coords, style in ovals
    )
    ovals.clear()
    publish_training(window, error="SSH disconnected")
    assert not any(
        style.get("fill") in ("#E8F4FF", "#F5ECFF") for _coords, style in ovals
    )


@pytest.mark.parametrize("extra", [["--no-training"], ["--server-id", "server1"]])
def test_disabled_or_other_server_does_not_create_a_training_poller(
    window, monkeypatch, extra
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Training config should not be loaded")

    monkeypatch.setattr(remote, "load_config", forbidden)
    other = MonitorWindow(
        core.build_parser().parse_args(["--demo", "--no-disk", *extra]),
        start_pollers=False,
    )
    try:
        assert not other.training_enabled
        assert other.training_poller is None
    finally:
        other.close()


def test_bad_training_configuration_does_not_break_gpu_ui_or_fake_a_percent(
    window, monkeypatch
):
    def fail(*args, **kwargs):
        raise remote.RemoteError("训练日志配置不可用")

    monkeypatch.setattr(remote, "load_config", fail)
    other = MonitorWindow(
        core.build_parser().parse_args(["--demo"]), start_pollers=False
    )
    try:
        other.set_compact(True)
        assert other.training_poller is None
        assert "训练日志配置不可用" in other.toolTip()
        assert "0.00%" not in other.toolTip()
    finally:
        other.close()


def test_floating_ball_ignores_conflicts_but_detail_tooltip_pairs_deadlines(
    window, multiple_requests
) -> None:
    window.online = True
    publish_notices(window)
    window.set_compact(True)
    assert ball_ring_color(window) == ImageColor.getcolor(
        core.COLORS["yzh_border"], "RGBA"
    )
    publish_notices(window, *multiple_requests)
    assert window.conflict_gpu_indexes == {3, 6}
    assert ball_ring_color(window) == ImageColor.getcolor(
        core.COLORS["yzh_border"], "RGBA"
    )
    tooltip = window.toolTip()
    assert "冲突" not in tooltip and "橙色提醒" not in tooltip
    assert "GPU 3、6" in tooltip and "yzh：GPU 3、6" in tooltip
    request_lines = [line for line in tooltip.splitlines() if line.startswith("申请：")]
    assert len(request_lines) == len(multiple_requests)
    for line, notice in zip(request_lines, multiple_requests, strict=True):
        cards, deadline = _request_badges(notice)
        assert line == f"申请：{cards} · {deadline}"
        assert "UTC" not in line
    publish_notices(window)
    assert not window.conflict_gpu_indexes
    assert ball_ring_color(window) == ImageColor.getcolor(
        core.COLORS["yzh_border"], "RGBA"
    )


def test_floating_ball_offline_does_not_present_stale_data_as_available(
    window, drawn_text
) -> None:
    window.online = True
    window.snapshot = snapshot_with_own_cards(window.snapshot, 1)
    publish_notices(window)
    publish_training(window)
    window.set_compact(True)
    drawn_text.clear()
    window.events.put(core.PollEvent(error="test outage"))
    window._drain_events()
    assert len(ball_status_calls(drawn_text)) == 3
    assert "离线" not in [
        item[2] for item in drawn_text
    ]  # Border / tooltip, not a fourth row.
    assert "5.3%" in [
        item[2] for item in drawn_text
    ]  # Independent SSH data remains live.
    assert not any(item[2].startswith("可用") for item in drawn_text)
    assert "连接中断，以下为上次数据" in window.toolTip()
    assert "可用 GPU：" not in window.toolTip()
    assert ball_ring_color(window) == ImageColor.getcolor(core.COLORS["red"], "RGBA")
    drawn_text.clear()
    window.events.put(
        core.PollEvent(snapshot=snapshot_with_own_cards(core.demo_snapshot(), 1))
    )
    window._drain_events()
    assert "离线" not in [item[2] for item in drawn_text]
    assert BALL_AVAILABILITY_PAGE in window._ball_page_set
    rows = draw_training_task(window, drawn_text, BALL_AVAILABILITY_PAGE)
    assert rows[1][2] == "可用 GPU"


def test_floating_ball_initial_loading_is_distinct_from_zero_utilization(
    window, drawn_text
) -> None:
    window.snapshot = None
    window.set_compact(True)
    assert "…" in [item[2] for item in drawn_text]
    assert "连接中" in [item[2] for item in drawn_text]
    assert "0%" not in [item[2] for item in drawn_text]


def test_floating_ball_topmost_minimize_and_keyboard_recovery(window) -> None:
    window.set_compact(True)
    geometry = window.geometry()
    window.set_topmost(False)
    QApplication.processEvents()
    QTest.qWait(100)
    assert window.compact_mode and window.geometry() == geometry
    window.minimize()
    assert window.isMinimized()
    window.showNormal()
    QApplication.processEvents()
    assert window.compact_mode and window.isVisible()
    QTest.keyClick(window, Qt.Key_M, Qt.ControlModifier)
    assert not window.compact_mode
    QTest.keyClick(window, Qt.Key_M, Qt.ControlModifier)
    assert window.compact_mode
    QTest.keyClick(window, Qt.Key_Return)
    assert not window.compact_mode


@pytest.mark.parametrize("size", [104, 128, 160, 192, 240])
def test_ball_size_scales_text_rings_and_hit_region_without_resizing_details(
    window, drawn_arcs, drawn_text, size
) -> None:
    original = window.geometry()
    window.online = True
    publish_notices(window)
    publish_training(window)
    window.set_compact(True)
    window.move(200, 160)
    center = window.geometry().center()
    drawn_arcs.clear()
    drawn_text.clear()
    window.set_ball_size(size)
    window._render()
    QApplication.processEvents()
    QTest.qWait(120)
    assert window.size() == QSize(size, size)
    assert (window.geometry().center() - center).manhattanLength() <= 2
    assert window._expanded_geometry == original
    assert window.canvas.view_scale == size / FLOATING_BALL_BASE_SIZE
    assert window.canvas._image.size == (size * core.RENDER_SCALE,) * 2
    assert window._frame.pixelColor(0, 0).alpha() == 0
    assert not window.mask().contains(QPoint(0, 0))
    assert window.mask().contains(QPoint(size // 2, size // 2))
    rings = [arc for arc in drawn_arcs if arc["fill"] in remote.TRAINING_COLORS]
    assert rings[0]["end"] == pytest.approx(-90 + 360 * 1060 / 20000)
    assert rings[1]["end"] == pytest.approx(-90 + 360 * 5033 / 20000)
    for index, expected in enumerate(("5.3%", "25.1%")):
        rows = draw_training_task(window, drawn_text, index)
        assert rows[0][2] == expected
        assert all(item[3]["fill"] == remote.TRAINING_COLORS[index] for item in rows)
    font = (window.family, 12, "bold")
    assert window.canvas._font(font).size == round(
        12 * core.FONT_PIXEL_RATIO * core.RENDER_SCALE * size / FLOATING_BALL_BASE_SIZE
    )
    assert window.canvas.measure_text("100%", font)[0] < 50
    window.set_compact(False)
    QApplication.processEvents()
    QTest.qWait(120)
    assert window.geometry() == original
    assert window.canvas.view_scale == 1.0
    assert window.ball_size == size


def send_ball_wheel(window, delta, modifiers=Qt.ControlModifier) -> None:
    position = QPoint(window.ball_size // 2, window.ball_size // 2)
    QApplication.sendEvent(
        window,
        QWheelEvent(
            QPointF(position),
            QPointF(window.mapToGlobal(position)),
            QPoint(),
            QPoint(0, delta),
            Qt.NoButton,
            modifiers,
            Qt.NoScrollPhase,
            False,
        ),
    )


def test_ball_ctrl_wheel_and_keys_resize_in_steps_with_safe_limits(window) -> None:
    window.set_compact(True)
    send_ball_wheel(window, 120, Qt.NoModifier)
    assert window.ball_size == FLOATING_BALL_SIZE
    for _ in range(2):
        send_ball_wheel(window, 40)
        assert window.ball_size == FLOATING_BALL_SIZE
    send_ball_wheel(window, 40)
    assert window.ball_size == FLOATING_BALL_SIZE + FLOATING_BALL_SIZE_STEP
    for _ in range(3):
        send_ball_wheel(window, -40)
    assert window.ball_size == FLOATING_BALL_SIZE
    QTest.keyClick(window, Qt.Key_Plus, Qt.ControlModifier)
    assert window.ball_size == FLOATING_BALL_SIZE + FLOATING_BALL_SIZE_STEP
    QTest.keyClick(window, Qt.Key_Minus, Qt.ControlModifier)
    assert window.ball_size == FLOATING_BALL_SIZE
    send_ball_wheel(window, 120 * 100)
    assert window.ball_size == FLOATING_BALL_MAX_SIZE
    send_ball_wheel(window, -120 * 100)
    assert window.ball_size == FLOATING_BALL_MIN_SIZE
    window.set_compact(False)
    original = window.geometry()
    send_ball_wheel(window, 120)
    assert window.ball_size == FLOATING_BALL_MIN_SIZE
    window.set_ball_size(160)
    assert window.geometry() == original


def test_ball_size_dialog_previews_and_cancel_restores_size_and_edge_position(
    window, monkeypatch
) -> None:
    window.set_compact(True)
    area = window.screen().availableGeometry()
    window.move(
        area.right() - window.ball_size + 1, area.bottom() - window.ball_size + 1
    )
    original = window.geometry()

    def cancel(dialog):
        assert dialog.font().family() == window.family
        assert dialog.intMinimum() == FLOATING_BALL_MIN_SIZE
        assert dialog.intMaximum() == FLOATING_BALL_MAX_SIZE
        assert dialog.intStep() == FLOATING_BALL_SIZE_STEP
        dialog.setIntValue(240)
        assert window.size() == QSize(240, 240)
        assert area.contains(window.geometry())
        return QDialog.Rejected

    monkeypatch.setattr(QInputDialog, "exec_", cancel)
    window.show_ball_size_dialog()
    QApplication.processEvents()
    QTest.qWait(150)
    assert window.geometry() == original
    assert window._ball_position == original.topLeft()

    def accept(dialog):
        dialog.setIntValue(160)
        return QDialog.Accepted

    monkeypatch.setattr(QInputDialog, "exec_", accept)
    window.show_ball_size_dialog()
    QApplication.processEvents()
    QTest.qWait(150)
    assert window.ball_size == 160
    assert window.size() == QSize(160, 160)
    assert area.contains(window.geometry())
    window.reset_position()
    QTest.qWait(150)
    assert window.ball_size == FLOATING_BALL_SIZE
    assert window.compact_mode and area.contains(window.geometry())


def test_ball_rapid_resize_still_drags_minimizes_and_restores_details(window) -> None:
    original = window.geometry()
    window.set_compact(True)
    for size in (104, 240, 160, 128, 200, 192):
        window.set_ball_size(size)
    QApplication.processEvents()
    QTest.qWait(200)
    position = window.pos()
    drag(window, QPoint(96, 96), QPoint(-25, 15))
    assert window.pos() == position + QPoint(-25, 15)
    assert window.size() == QSize(192, 192)
    window.minimize()
    window.showNormal()
    QApplication.processEvents()
    QTest.qWait(120)
    assert window.compact_mode and window.size() == QSize(192, 192)
    QTest.mouseClick(window, Qt.LeftButton, pos=QPoint(96, 96))
    QApplication.processEvents()
    QTest.qWait(150)
    assert not window.compact_mode and window.geometry() == original


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, 128),
        ("bad", 128),
        (float("inf"), 128),
        (-99, 104),
        (999, 240),
        ("160", 160),
    ],
)
def test_saved_ball_size_invalid_values_are_safe(value, expected) -> None:
    assert _ball_size_value(value) == expected


def test_ball_has_no_title_and_color_does_not_change_for_requests_or_conflicts(
    window, drawn_text
) -> None:
    window.online = True
    window.set_compact(True)

    def check():
        drawn_text.clear()
        window._render()
        assert not any(
            item[2] in ("训练", "申请", "冲突", "待确认") for item in drawn_text
        )
        assert ball_ring_color(window) == ImageColor.getcolor(
            core.COLORS["yzh_border"], "RGBA"
        )
        assert "冲突" not in window.toolTip() and "橙色提醒" not in window.toolTip()

    check()
    publish_notices(window, core.Announcement("a", "gpu_usage", "申请4号卡", (4,)))
    check()
    publish_notices(window, core.Announcement("a", "gpu_usage", "申请3号卡", (3,)))
    check()
    window.events.put(core.PollEvent(notice_error="test failure"))
    window._drain_events()
    check()
    publish_notices(window)
    check()


def test_floating_ball_config_saves_full_geometry_and_reopens_compact(
    window, monkeypatch
) -> None:
    original = window.geometry()
    saved = []
    monkeypatch.setattr(core, "_save_config", saved.append)
    window.set_ball_size(160)
    window.set_compact(True)
    drag(window, QPoint(52, 52), QPoint(-45, 25))
    position = window.pos()
    window.close()
    assert saved[0]["compact_mode"] is True
    assert saved[0]["ball_size"] == 160
    assert (saved[0]["width"], saved[0]["height"]) == (
        original.width(),
        original.height(),
    )
    assert (saved[0]["x"], saved[0]["y"]) == (original.x(), original.y())
    assert (saved[0]["ball_x"], saved[0]["ball_y"]) == (position.x(), position.y())
    monkeypatch.setattr(core, "_load_config", lambda: saved[0])
    reopened = MonitorWindow(
        core.build_parser().parse_args(["--demo"]), start_pollers=False
    )
    try:
        assert reopened.compact_mode
        assert reopened.pos() == position
        assert reopened.ball_size == 160
        assert reopened.size() == QSize(160, 160)
        # Startup clamps the detailed window to the current display, which may
        # be the offscreen test plugin's smaller 800-pixel virtual screen.
        area = QApplication.primaryScreen().availableGeometry()
        width = min(original.width(), area.width())
        height = max(original.height(), reopened._minimum_height(width))
        x = min(
            max(area.left(), original.x()), max(area.left(), area.right() - width + 1)
        )
        y = min(
            max(area.top(), original.y()), max(area.top(), area.bottom() - height + 1)
        )
        reopened.set_compact(False)
        assert reopened.geometry() == QRect(x, y, width, height)
    finally:
        reopened.close()


def test_floating_ball_stays_circular_in_square_or_windowed_mode(window) -> None:
    for windowed in (False, True):
        window.args.windowed = windowed
        window.rounded_frame = False
        original = window.geometry()
        window.set_compact(True)
        assert window.windowFlags() & Qt.FramelessWindowHint
        assert window.testAttribute(Qt.WA_TranslucentBackground)
        assert not window.mask().contains(QPoint(0, 0))
        assert window._edges_at(QPoint(1, 1)) == ""
        window.set_compact(False)
        QApplication.processEvents()
        QTest.qWait(100)
        assert bool(window.windowFlags() & Qt.FramelessWindowHint) is not windowed
        assert window.mask().contains(QPoint(0, 0))
        assert window.geometry() == original


def test_floating_ball_fast_repeated_mode_switches_reconcile_native_geometry(
    window,
) -> None:
    original = window.geometry()
    for _ in range(5):
        window.set_compact(True)
        window.set_compact(False)
    QApplication.processEvents()
    QTest.qWait(150)
    # X11 ConfigureNotify bursts occasionally settle around 200 ms even with
    # the correct geometry already restored. Check completion, with a deadline,
    # instead of treating a fixed compositor scheduling delay as a frozen UI.
    deadline = monotonic() + 1
    while window._changing_mode and monotonic() < deadline:
        QTest.qWait(20)
    assert not window.compact_mode and not window._changing_mode
    assert window.geometry() == original
    assert window.height() >= window._minimum_height(window.width())
    window.set_compact(True)
    position = window.pos()
    drag(window, QPoint(52, 52), QPoint(-25, 20))
    assert window.pos() == position + QPoint(-25, 20)
    assert window.size() == QSize(FLOATING_BALL_SIZE, FLOATING_BALL_SIZE)


def test_floating_ball_drag_is_clamped_to_screen_and_reset_keeps_mode(window) -> None:
    window.set_compact(True)
    area = window.screen().availableGeometry()
    drag(window, QPoint(52, 52), QPoint(10000, 10000))
    assert window.geometry().right() <= area.right()
    assert window.geometry().bottom() <= area.bottom()
    assert window.compact_mode
    window.reset_position()
    QApplication.processEvents()
    assert window.size() == QSize(FLOATING_BALL_SIZE, FLOATING_BALL_SIZE)
    assert area.contains(window.geometry())


def test_floating_ball_restores_maximized_state_and_normal_geometry(window) -> None:
    original = window.geometry()
    window.showMaximized()
    QApplication.processEvents()
    QTest.qWait(150)
    assert window.isMaximized()
    window.set_compact(True)
    QApplication.processEvents()
    QTest.qWait(100)
    assert not window.isMaximized() and window.compact_mode
    window.set_compact(False)
    QApplication.processEvents()
    QTest.qWait(150)
    assert window.isMaximized() and not window.compact_mode
    window.showNormal()
    QApplication.processEvents()
    QTest.qWait(150)
    assert window.geometry() == original


def publish_notices(window: MonitorWindow, *notices: core.Announcement) -> None:
    window.events.put(core.PollEvent(announcements=notices))
    window._drain_events()


def border_color(
    window: MonitorWindow, index: int, *, inset: float = 0
) -> tuple[int, ...]:
    x1, y1, x2, _y2 = window.card_boxes[index]
    return window.canvas._image.getpixel(
        (
            round((x1 + x2) / 2 * core.RENDER_SCALE),
            round((y1 + inset + 0.5) * core.RENDER_SCALE),
        )
    )


def test_rules_do_not_make_a_single_notice_scroll(window) -> None:
    request = core.Announcement("request", "gpu_usage", "alice申请3号卡", (3,))
    rules = core.Announcement(
        "rules", "operations", "默认服务器规则", is_server_rules=True
    )
    publish_notices(window, request, rules)
    assert window.banner_announcements == (request,)
    assert window.announcements == (request, rules)  # Rules remain in the full viewer.
    assert window._current_notice_text() == request.text
    assert not window.notice_timer.isActive()
    window._next_notice()
    assert window.notice_index == 0
    assert window._previous_notice is None
    publish_notices(window, rules)
    assert window.banner_announcements == ()
    assert "暂无" in window._current_notice_text()
    assert not window.notice_timer.isActive()


def test_multiple_notices_scroll_smoothly_in_the_banner_and_wrap_around(window) -> None:
    first = core.Announcement(
        "alice", "gpu_usage", "alice申请3号卡，用于模型训练。", (3,)
    )
    second = core.Announcement("bob", "gpu_usage", "bob申请4号卡，用于评测。", (4,))
    rules = core.Announcement(
        "rules", "operations", "默认服务器规则", is_server_rules=True
    )
    publish_notices(window, first, rules, second)
    assert window.notice_timer.isActive()
    assert window.notice_timer.interval() == 8000
    before = window.canvas.present().convert("RGB")
    # Exercise the actual timer without waiting eight seconds in a regression test.
    window.notice_timer.setInterval(30)
    QTest.qWait(130)
    window.notice_timer.stop()
    assert window._previous_notice == first
    assert window.notice_index == 1
    window.notice_animation.setCurrentTime(250)
    assert window._notice_scroll_progress == pytest.approx(0.5)
    during = window.canvas.present().convert("RGB")
    difference = ImageChops.difference(before, during).getbbox()
    assert difference is not None
    assert difference[1] >= 74 and difference[3] <= 136
    window.notice_animation.setCurrentTime(500)
    assert window._previous_notice is None
    assert window._current_notice_text() == second.text
    assert window.reserved_gpu_indexes == {3, 4}
    window._next_notice()
    window.notice_animation.setCurrentTime(500)
    assert window.notice_index == 0
    assert window._current_notice_text() == first.text


@pytest.mark.parametrize("width", [650, 970, 1250])
def test_multiple_request_labels_and_body_stay_paired_and_inline_through_scroll(
    window, multiple_requests, drawn_text, width
) -> None:
    rules = core.Announcement("rules", "operations", "默认规则", is_server_rules=True)
    publish_notices(window, multiple_requests[0], rules, *multiple_requests[1:])
    window.resize(width, window.height())
    QApplication.processEvents()
    QTest.qWait(100)
    assert window.banner_announcements == multiple_requests
    drawn_text.clear()
    window._render()
    assert_paired_notice_row(window, drawn_text, multiple_requests[0])

    for index, previous in enumerate(multiple_requests):
        following = multiple_requests[(index + 1) % len(multiple_requests)]
        window._next_notice()
        for elapsed in (0, 125, 250, 375):
            window.notice_animation.setCurrentTime(elapsed)
            drawn_text.clear()
            window._render()
            assert window._previous_notice == previous
            progress = window._notice_scroll_progress
            assert_paired_notice_row(
                window, drawn_text, previous, offset=-36 * progress
            )
            assert_paired_notice_row(
                window, drawn_text, following, offset=36 * (1 - progress)
            )
            labels = [
                item[2]
                for item in drawn_text
                if item[3].get("clip")
                and item[3]["fill"] in (core.COLORS["red"], core.COLORS["amber"])
            ]
            assert labels == [*_request_badges(previous), *_request_badges(following)]
            assert window.reserved_gpu_indexes == set(range(8))
        window.notice_animation.setCurrentTime(500)
        drawn_text.clear()
        window._render()
        assert window._previous_notice is None
        assert window.notice_index == (index + 1) % len(multiple_requests)
        assert_paired_notice_row(window, drawn_text, following)


def test_reordered_and_updated_requests_cancel_scroll_without_mixing_label_pairs(
    window, multiple_requests, drawn_text
) -> None:
    publish_notices(window, *multiple_requests)
    window._next_notice()
    window.notice_animation.setCurrentTime(250)
    updated = replace(
        multiple_requests[0],
        expires_at=datetime(2026, 10, 13, 4, 5, tzinfo=timezone.utc),
    )
    publish_notices(window, multiple_requests[2], updated)
    assert window._previous_notice is None
    assert window.notice_index == 0
    drawn_text.clear()
    window._render()
    assert_paired_notice_row(window, drawn_text, multiple_requests[2])
    window._next_notice()
    window.notice_animation.setCurrentTime(500)
    drawn_text.clear()
    window._render()
    assert_paired_notice_row(window, drawn_text, updated)
    assert _request_badges(multiple_requests[0])[1] not in [
        item[2] for item in drawn_text
    ]
    assert window.reserved_gpu_indexes == set(range(8))


def test_full_viewer_keeps_each_request_label_pair_and_body_in_the_same_paragraph(
    window, multiple_requests
) -> None:
    document = QTextDocument()
    document.setHtml(_announcements_html(multiple_requests, ""))
    paragraphs = []
    block = document.begin()
    while block.isValid():
        if block.text().startswith("GPU ") and "截止 " in block.text():
            paragraphs.append(block.text())
        block = block.next()
    assert len(paragraphs) == len(multiple_requests)
    for paragraph, notice in zip(paragraphs, multiple_requests, strict=True):
        cards, deadline = _request_badges(notice)
        assert (
            paragraph.index(cards)
            < paragraph.index(deadline)
            < paragraph.index(notice.text)
        )
        assert "UTC" not in paragraph


def test_all_reservations_remain_marked_and_conflicting_yzh_cards_turn_orange(
    window,
) -> None:
    publish_notices(
        window,
        core.Announcement("alice", "gpu_usage", "alice申请3、4号卡", (3, 4)),
        core.Announcement("bob", "gpu_usage", "bob申请6号卡", (6,)),
    )
    red = ImageColor.getcolor(core.COLORS["red"], "RGBA")
    orange = ImageColor.getcolor(core.COLORS["conflict_border"], "RGBA")
    assert window.reserved_gpu_indexes == {3, 4, 6}
    assert window.snapshot.gpus[3].has_user("yzh")
    for index in (3, 6):
        assert border_color(window, index) == orange
        assert border_color(window, index, inset=4) == red
    assert border_color(window, 4) == red
    assert border_color(window, 0) != red
    window._next_notice()
    window.notice_animation.setCurrentTime(500)
    assert border_color(window, 3) == orange
    assert border_color(window, 6) == orange
    assert border_color(window, 4) == red
    # The reservation is independent of whether GPU polling has returned yet.
    window.snapshot = None
    window._render()
    assert border_color(window, 4) == red


def test_conflict_color_tracks_current_usage_and_reverts_after_revocation(
    window,
) -> None:
    publish_notices(
        window, core.Announcement("request", "gpu_usage", "申请3号卡", (3,))
    )
    orange = ImageColor.getcolor(core.COLORS["conflict_border"], "RGBA")
    red = ImageColor.getcolor(core.COLORS["red"], "RGBA")
    green = ImageColor.getcolor(core.COLORS["yzh_border"], "RGBA")
    assert border_color(window, 3) == orange
    assert (
        border_color(window, 6) == green
    )  # yzh's unreserved card keeps its green frame.
    original = window.snapshot
    window.events.put(
        core.PollEvent(
            snapshot=replace(
                original,
                gpus=tuple(
                    (
                        replace(
                            gpu,
                            user_usages=(
                                core.UserGpuUsage("other", gpu.memory_used_gib, 1),
                            ),
                        )
                        if gpu.index == 3
                        else gpu
                    )
                    for gpu in original.gpus
                ),
            )
        )
    )
    window._drain_events()
    assert border_color(window, 3) == red  # Requested, but no longer used by yzh.
    window.events.put(core.PollEvent(snapshot=original))
    window._drain_events()
    assert border_color(window, 3) == orange
    publish_notices(window)
    assert border_color(window, 3) == green


def test_long_request_always_emphasizes_card_numbers_and_exact_deadline(
    window, monkeypatch
) -> None:
    notice = core.Announcement(
        "long-request",
        "gpu_usage",
        "非常长的训练原因，正文里的2027-01-01不是截止日期。" * 30,
        tuple(range(8)),
        expires_at=datetime(2026, 10, 7, 12, tzinfo=timezone.utc),
    )
    drawn = []
    original = window.canvas.create_text

    def record(x, y, *, text, **style):
        drawn.append((x, y, text, style))
        return original(x, y, text=text, **style)

    monkeypatch.setattr(window.canvas, "create_text", record)
    publish_notices(window, notice)
    window.resize(650, 524)
    QApplication.processEvents()
    QTest.qWait(100)
    drawn.clear()
    window._render()
    cards, deadline = _request_badges(notice)
    assert cards == "GPU 0、1、2、3、4、5、6、7"
    assert "2026-10-07" in deadline and "2027-01-01" not in deadline
    assert "UTC" not in deadline
    for label, color in ((cards, core.COLORS["red"]), (deadline, core.COLORS["amber"])):
        matches = [(x, y, style) for x, y, text, style in drawn if text == label]
        assert len(matches) == 1
        x, y, style = matches[0]
        assert style["fill"] == color
        assert "bold" in style["font"]
        assert (
            x + window.canvas.measure_text(label, style["font"])[0]
            <= window.width() - 66
        )
        assert y == NOTICE_ROW_Y
    assert_paired_notice_row(window, drawn, notice)


def test_notice_viewer_highlights_structured_data_and_escapes_external_html(
    window,
) -> None:
    notice = core.Announcement(
        "request",
        "gpu_usage",
        "alice <img src='https://example.test/x'> <script>bad()</script>",
        (3, 4),
        expires_at=datetime(2026, 10, 7, 12, tzinfo=timezone.utc),
    )
    html = _announcements_html((notice,), "")
    assert "<img" not in html and "<script" not in html
    document = QTextDocument()
    document.setHtml(html)
    assert notice.text in document.toPlainText()
    cards, deadline = _request_badges(notice)
    highlights = {}
    block = document.begin()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.text() in (cards, deadline):
                highlights[fragment.text()] = fragment.charFormat()
            iterator += 1
        block = block.next()
    for label, color in ((cards, core.COLORS["red"]), (deadline, core.COLORS["amber"])):
        assert highlights[label].fontWeight() >= QFont.Bold
        assert highlights[label].foreground().color().name() == color.lower()
    unknown = core.Announcement("unknown", "gpu_usage", "2026-10-09开始训练", (3,))
    assert _request_badges(unknown)[1] == "截止 未注明"


def test_new_announcements_cancel_old_scrolls_and_clear_revoked_reservations(
    window,
) -> None:
    publish_notices(
        window,
        core.Announcement("alice", "gpu_usage", "alice申请3号卡", (3,)),
        core.Announcement("bob", "gpu_usage", "bob申请4号卡", (4,)),
    )
    window._next_notice()
    assert window._previous_notice is not None
    window.events.put(core.PollEvent(notice_error="暂时断网"))
    window._drain_events()
    assert window.reserved_gpu_indexes == {3, 4}  # Keep last known notices on errors.
    publish_notices(window)
    assert window.reserved_gpu_indexes == set()
    assert window._previous_notice is None
    assert not window.notice_timer.isActive()
    assert not window.notice_animation.state()
    assert border_color(window, 3) == ImageColor.getcolor(
        core.COLORS["yzh_border"], "RGBA"
    )
    assert border_color(window, 4) == ImageColor.getcolor(
        core.COLORS["card_border"], "RGBA"
    )


def test_resize_during_notice_scroll_keeps_text_clipped_and_frame_size_correct(
    window,
) -> None:
    publish_notices(
        window,
        core.Announcement("first", "operations", "本服务器临时维护公告。" * 20),
        core.Announcement("second", "gpu_usage", "alice申请3号卡", (3,)),
    )
    window._next_notice()
    window.resize(660, 524)
    window._animate_notice(0.4)
    assert window._frame.width() == window.width()
    assert window._frame.height() == window.height()
    window.notice_animation.setCurrentTime(500)
    assert window._columns() == 2
    assert len(window.card_boxes) == 8
