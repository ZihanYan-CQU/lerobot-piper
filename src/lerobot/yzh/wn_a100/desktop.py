#!/usr/bin/env python3
"""Qt window shell for the WN-A100 monitor; imported only by the desktop CLI."""

from __future__ import annotations

import argparse
import math
import os
import queue
import signal
import sys
from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime
from html import escape
from pathlib import Path
from time import monotonic
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore
from PyQt5.QtCore import (
    QEvent,
    QEasingCurve,
    QPoint,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QUrl,
    QVariantAnimation,
)
from PyQt5.QtGui import (
    QDesktopServices,
    QCursor,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPainterPath,
    QRegion,
    QTextOption,
)
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from . import monitor as core
from . import remote

COLORS = core.COLORS
RENDER_SCALE = core.RENDER_SCALE
FONT_PIXEL_RATIO = core.FONT_PIXEL_RATIO
HISTORY_LENGTH = core.HISTORY_LENGTH
HIGHLIGHT_USERNAME = core.HIGHLIGHT_USERNAME
GpuMetric = core.GpuMetric
MINIMUM_WIDTH = 650
GRID_TOP = 148
CARD_HEIGHT = 148
CARD_GAP = 12
RESIZE_MARGIN = 7
NOTICE_ROW_Y = 114
FLOATING_BALL_SIZE = 128
FLOATING_BALL_BASE_SIZE = 104
FLOATING_BALL_MIN_SIZE = 104
FLOATING_BALL_MAX_SIZE = 240
FLOATING_BALL_SIZE_STEP = 8
TRAINING_RING_LAYOUT = ((3, 3, 100, 100, 7), (11, 11, 92, 92, 6))
TRAINING_TEXT_LAYOUT = (42, 57, 68)
BALL_MESSAGE_TEXT_LAYOUT = (34, 54, 70)
DISK_TEXT_LAYOUT = (45, 63)
TRAINING_CAROUSEL_INTERVAL = 4000
TRAINING_TRANSITION_DURATION = 480
BALL_AVAILABILITY_PAGE = 2
BALL_DISK_PAGE = 4
BALL_SUMMARY_PAGE = 5
BALL_SSH_LOGIN_PAGE = 6


def _compact_gpu_indexes(indexes: frozenset[int]) -> str:
    """Fit card numbers in the ball without losing any IDs."""
    groups: list[str] = []
    ordered = sorted(indexes)
    if not ordered:
        return "暂无"
    start = previous = ordered[0]
    groups.append(str(start))
    for index in ordered[1:]:
        if index != previous + 1:
            groups.append(str(index))
            start = index
        else:
            groups[-1] = f"{start}–{index}"
        previous = index
    return "、".join(groups)


def _ball_size_value(value: Any) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError, OverflowError):
        size = FLOATING_BALL_SIZE
    return min(FLOATING_BALL_MAX_SIZE, max(FLOATING_BALL_MIN_SIZE, size))


def _ball_training_percent(progress: remote.TrainingProgress | None) -> str:
    if progress is None:
        return "…"
    if progress.step == progress.total:
        return "100%"
    tenths = progress.step * 1000 // progress.total
    if progress.step > 0 and tenths == 0:
        return "<0.1%"
    # Keep an unfinished job below 100%, even near the rounding boundary.
    return f"{tenths // 10}.{tenths % 10}%"


def _request_badges(notice: core.Announcement) -> tuple[str, str]:
    indexes = "、".join(str(index) for index in notice.gpu_indexes) or "卡号待补充"
    return f"GPU {indexes}", f"截止 {core.format_notice_deadline(notice.expires_at)}"


def _announcements_html(notices: tuple[core.Announcement, ...], fallback: str) -> str:
    """Keep external notice text inert while emphasizing structured request data."""
    sections: list[str] = []
    for item in notices:
        kind = "GPU 使用申请" if item.kind == "gpu_usage" else "服务器公告"
        summary = ""
        if item.kind == "gpu_usage":
            cards, deadline = _request_badges(item)
            summary = (
                f'<b><span style="color:{COLORS["red"]}">{escape(cards)}</span>'
                f' &nbsp; <span style="color:{COLORS["amber"]}">{escape(deadline)}</span></b> &nbsp; '
            )
        text = escape(item.text).replace("\n", "<br>")
        sections.append(f"<h3>{kind}</h3><p>{summary}{text}</p>")
    return "<html><body>" + ("".join(sections) or escape(fallback)) + "</body></html>"


class PillowCanvas:
    """An RGBA drawing surface with 2x FreeType antialiasing."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.view_scale = 1.0
        self._font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}
        self._regular_face, self._bold_face = core._font_files()
        self.delete("all")

    def _scaled(self, value: float) -> int:
        return round(value * RENDER_SCALE * self.view_scale)

    def _font(self, specification: Any) -> ImageFont.FreeTypeFont:
        size = 10
        bold = False
        if isinstance(specification, tuple):
            if len(specification) >= 2:
                size = int(specification[1])
            bold = any("bold" in str(part).lower() for part in specification[2:])
        pixel_size = max(
            8, round(size * FONT_PIXEL_RATIO * RENDER_SCALE * self.view_scale)
        )
        key = (pixel_size, bold)
        if key not in self._font_cache:
            path, index = self._bold_face if bold else self._regular_face
            self._font_cache[key] = ImageFont.truetype(
                str(path), pixel_size, index=index
            )
        return self._font_cache[key]

    def delete(self, _tag: str) -> None:
        self._image = Image.new(
            "RGBA",
            (self.width * RENDER_SCALE, self.height * RENDER_SCALE),
            (0, 0, 0, 0),
        )
        self._draw = ImageDraw.Draw(self._image)

    def rounded_rectangle(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
        **kwargs: Any,
    ) -> int:
        fill = kwargs.get("fill") or None
        outline = kwargs.get("outline") or None
        width = max(1, self._scaled(float(kwargs.get("width", 1))))
        coordinates = tuple(self._scaled(value) for value in (x1, y1, x2, y2))
        style = dict(
            radius=self._scaled(radius),
            fill=fill,
            outline=outline,
            width=width,
        )
        clip = kwargs.get("clip")
        if clip is None:
            self._draw.rounded_rectangle(coordinates, **style)
        else:
            left, top, right, bottom = (self._scaled(value) for value in clip)
            layer = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
            translated = (
                coordinates[0] - left,
                coordinates[1] - top,
                coordinates[2] - left,
                coordinates[3] - top,
            )
            ImageDraw.Draw(layer).rounded_rectangle(translated, **style)
            self._image.alpha_composite(layer, dest=(left, top))
        return 0

    def create_text(self, x: float, y: float, *, text: str, **kwargs: Any) -> int:
        anchors = {"w": "lm", "e": "rm", "center": "mm", "n": "ma", "s": "md"}
        style = dict(
            fill=kwargs.get("fill", COLORS["text"]),
            font=self._font(kwargs.get("font")),
            anchor=anchors.get(kwargs.get("anchor", "w"), "lm"),
        )
        clip = kwargs.get("clip")
        opacity = max(0.0, min(1.0, float(kwargs.get("opacity", 1.0))))
        if not text or opacity == 0:
            return 0
        if clip is None and opacity == 1:
            self._draw.text((self._scaled(x), self._scaled(y)), text, **style)
        else:
            if clip is None:
                left, top, right, bottom = self._draw.textbbox(
                    (self._scaled(x), self._scaled(y)),
                    text,
                    font=style["font"],
                    anchor=style["anchor"],
                )
            else:
                left, top, right, bottom = (self._scaled(value) for value in clip)
            if right <= left or bottom <= top:
                return 0
            layer = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
            ImageDraw.Draw(layer).text(
                (self._scaled(x) - left, self._scaled(y) - top), text, **style
            )
            if opacity < 1:
                # Composite over the panel rather than replacing its alpha;
                # otherwise faded glyphs make holes through the desktop ball.
                layer.putalpha(
                    layer.getchannel("A").point(lambda alpha: round(alpha * opacity))
                )
            self._image.alpha_composite(layer, dest=(left, top))
        return 0

    def create_oval(
        self, x1: float, y1: float, x2: float, y2: float, **kwargs: Any
    ) -> int:
        self._draw.ellipse(
            tuple(self._scaled(value) for value in (x1, y1, x2, y2)),
            fill=kwargs.get("fill") or None,
            outline=kwargs.get("outline") or None,
            width=max(1, self._scaled(float(kwargs.get("width", 1)))),
        )
        return 0

    def create_arc(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        start: float,
        end: float,
        fill: str,
        width: float,
    ) -> int:
        self._draw.arc(
            tuple(self._scaled(value) for value in (x1, y1, x2, y2)),
            start=start,
            end=end,
            fill=fill,
            width=max(1, self._scaled(width)),
        )
        return 0

    def create_rectangle(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        **kwargs: Any,
    ) -> int:
        self._draw.rectangle(
            tuple(self._scaled(value) for value in (x1, y1, x2, y2)),
            fill=kwargs.get("fill") or None,
            outline=kwargs.get("outline") or None,
            width=max(1, self._scaled(float(kwargs.get("width", 1)))),
        )
        return 0

    def measure_text(self, text: str, font: Any) -> tuple[float, float]:
        left, top, right, bottom = self._draw.textbbox(
            (0, 0), text, font=self._font(font)
        )
        return (right - left) / (RENDER_SCALE * self.view_scale), (bottom - top) / (
            RENDER_SCALE * self.view_scale
        )

    def create_line(self, *coordinates: float, **kwargs: Any) -> int:
        if len(coordinates) == 1 and isinstance(coordinates[0], list | tuple):
            coordinates = tuple(coordinates[0])
        points = [
            (self._scaled(coordinates[index]), self._scaled(coordinates[index + 1]))
            for index in range(0, len(coordinates), 2)
        ]
        if len(points) >= 2:
            self._draw.line(
                points,
                fill=kwargs.get("fill", COLORS["text"]),
                width=max(1, self._scaled(float(kwargs.get("width", 1)))),
                joint="curve",
            )
        return 0

    def present(self) -> Image.Image:
        return self._image.resize((self.width, self.height), Image.Resampling.LANCZOS)


class ProcessCommandPopup(QFrame):
    """A non-modal, scrollable viewer. Display and copy only; never run commands."""

    def __init__(self, owner: "MonitorWindow") -> None:
        super().__init__(
            owner, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.owner = owner
        self.current_gpu: int | None = None
        self.setObjectName("processCommandPopup")
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TranslucentBackground)
        font = QFont(owner.family, 10)
        font.setStyleStrategy(QFont.PreferAntialias)
        self.setFont(font)
        self.setStyleSheet(
            "QFrame#processCommandPopup {background:#111A29; border:1px solid #22324A; border-radius:8px;}"
            "QLabel {color:#F5F7FB; border:0;}"
            "QTextEdit {background:#0B111C; color:#D7E3F4; border:1px solid #22324A; padding:8px;}"
            "QPushButton {background:#1B2940; color:#F5F7FB; border:1px solid #22324A; border-radius:4px; padding:4px 9px;}"
            "QPushButton:hover {background:#223955;}"
            "QScrollBar:vertical {background:#111A29; width:12px;}"
            "QScrollBar::handle:vertical {background:#435877; min-height:24px;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        header = QHBoxLayout()
        self.title = QLabel(self)
        title_font = QFont(font)
        title_font.setBold(True)
        self.title.setFont(title_font)
        header.addWidget(self.title, 1)
        self.copy_button = QPushButton("复制文本", self)
        self.copy_button.clicked.connect(self.copy_text)
        header.addWidget(self.copy_button)
        close = QPushButton("×", self)
        close.setFixedWidth(32)
        close.clicked.connect(owner._clear_memory_hover)
        header.addWidget(close)
        layout.addLayout(header)
        self.editor = QTextEdit(self)
        self.editor.setReadOnly(True)
        self.editor.setAcceptRichText(False)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QTextEdit.WidgetWidth)
        self.editor.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.editor)
        note = QLabel(
            "SSH 实际 PID / 指令 · 非完整启动脚本 · 按参数换行，仅供查看", self
        )
        note_font = QFont(font)
        note_font.setPointSize(8)
        note.setFont(note_font)
        note.setStyleSheet("color:#8E9AAF;")
        layout.addWidget(note)

    def paintEvent(self, event: Any) -> None:
        # A translucent tool window needs an explicit opaque panel behind its
        # child labels; otherwise native text can blend with the cards below.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(COLORS["panel_border"]))
        painter.setBrush(QColor(COLORS["card"]))
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 8, 8
        )

    def copy_text(self) -> None:
        QApplication.clipboard().setText(self.editor.toPlainText())

    def present(
        self, index: int, text: str, anchor: QPoint, *, reposition: bool, stale: bool
    ) -> None:
        changed_gpu = index != self.current_gpu
        scroll = self.editor.verticalScrollBar().value()
        if self.editor.toPlainText() != text:
            self.editor.setPlainText(text)
            if not changed_gpu:
                self.editor.verticalScrollBar().setValue(scroll)
        self.current_gpu = index
        self.title.setText(f"GPU {index} · {HIGHLIGHT_USERNAME} · PID 与进程指令")
        self.title.setStyleSheet("color:#FFB454;" if stale else "color:#F5F7FB;")
        if reposition or changed_gpu or not self.isVisible():
            screen = QApplication.screenAt(anchor) or self.owner.screen()
            area = screen.availableGeometry()
            width = max(120, min(760, area.width() - 24))
            limit = max(120, min(640, int(area.height() * 0.72), area.height() - 24))
            height = min(
                limit,
                max(
                    180,
                    80
                    + len(text.splitlines()) * self.editor.fontMetrics().lineSpacing(),
                ),
            )
            self.resize(width, height)
            x = anchor.x() + 14
            if x + width > area.right() - 8:
                x = anchor.x() - width - 14
            y = anchor.y() + 14
            if y + height > area.bottom() - 8:
                y = anchor.y() - height - 14
            x = max(area.left() + 8, min(x, area.right() - width - 8))
            y = max(area.top() + 8, min(y, area.bottom() - height - 8))
            self.move(x, y)
        self.show()

    def enterEvent(self, event: Any) -> None:
        self.owner._memory_hide_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self.owner._defer_memory_hover_clear()
        super().leaveEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key_Escape:
            self.owner._clear_memory_hover()
            event.accept()
        else:
            super().keyPressEvent(event)


class MonitorWindow(QWidget):
    """Resizable desktop monitor with native minimize and transparent corners."""

    def __init__(self, args: argparse.Namespace, *, start_pollers: bool = True) -> None:
        super().__init__()
        self.args = args
        self.snapshot: core.Snapshot | None = None
        self.announcements: tuple[core.Announcement, ...] = ()
        self.notice_loaded = False
        self.notice_error = ""
        self.notice_index = 0
        self._previous_notice: core.Announcement | None = None
        self._notice_scroll_progress = 1.0
        self._notice_line_cache: dict[tuple[str, float, bool], list[str]] = {}
        self.online = False
        self.status_message = "正在连接 WN-A100"
        self.last_error = ""
        self.histories: defaultdict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=core.HISTORY_LENGTH)
        )
        self.events: queue.Queue[core.PollEvent] = queue.Queue(maxsize=16)
        self.training_enabled = (
            not getattr(args, "no_training", False)
            and args.server_id == core.DEFAULT_SERVER_ID
        )
        self.disk_enabled = (
            not getattr(args, "no_disk", False)
            and args.server_id == core.DEFAULT_SERVER_ID
        )
        self.disk_events: queue.Queue[remote.DiskEvent] = queue.Queue(maxsize=4)
        self.disk_config: remote.DiskConfig | None = None
        self.disk_sample = remote.DiskSample()
        self.disk_poller: remote.DiskPoller | None = None
        self.ssh_config: remote.RemoteConfig | None = None
        self.ssh_events: queue.Queue[remote.SessionEvent] = queue.Queue(maxsize=4)
        self.ssh_poller: remote.SessionPoller | None = None
        self.ssh_connected: bool | None = True if args.demo else None
        self._process_session_started_at = 0.0
        self.ssh_error = ""
        self.ssh_login_error = ""
        self.ssh_login_process: Any = None
        self._ssh_login_pending_until = 0.0
        self._pressed_login_page = False
        self.training_events: queue.Queue[remote.TrainingEvent] = queue.Queue(maxsize=8)
        self.training_config: remote.RemoteConfig | None = None
        self.training_samples: tuple[remote.TrainingSample, ...] = ()
        self.training_error = ""
        self._ball_training_index = 0
        self._hover_training_index: int | None = None
        self._ball_page_set: tuple[int, ...] = ()
        self._ball_previous_page: int | None = None
        self._ball_transition_progress = 1.0
        self.training_poller: remote.TrainingPoller | None = None
        if self.training_enabled or self.disk_enabled:
            try:
                connection = remote.load_config(
                    getattr(args, "training_config", remote.DEFAULT_CONFIG)
                )
                self.ssh_config = connection
                if not args.demo:
                    self.ssh_poller = remote.SessionPoller(self.ssh_events, connection)
                if self.training_enabled:
                    self.training_config = connection
                    self.training_samples = tuple(
                        remote.TrainingSample(log) for log in connection.logs
                    )
                    self.training_poller = remote.TrainingPoller(
                        self.training_events, connection, demo=args.demo
                    )
                if self.disk_enabled:
                    self.disk_config = remote.DiskConfig(
                        connection,
                        getattr(args, "disk_interval", remote.DEFAULT_INTERVAL),
                        getattr(args, "disk_timeout", remote.DEFAULT_TIMEOUT),
                    )
                    self.disk_poller = remote.DiskPoller(
                        self.disk_events, self.disk_config, demo=args.demo
                    )
            except remote.RemoteError as exc:
                if self.training_enabled:
                    self.training_error = str(exc)
                if self.disk_enabled:
                    self.disk_sample = remote.DiskSample(error=str(exc))
        self.hovered_card: int | None = None
        self.hovered_button: str | None = None
        self.card_boxes: dict[int, tuple[float, float, float, float]] = {}
        self.memory_bar_boxes: dict[int, tuple[float, float, float, float]] = {}
        self._memory_hover_position: QPoint | None = None
        self._memory_hover_global_position: QPoint | None = None
        self._memory_pointer_on_bar = False
        self._memory_process_popup: ProcessCommandPopup | None = None
        self._memory_hide_timer = QTimer(self)
        self._memory_hide_timer.setSingleShot(True)
        self._memory_hide_timer.setInterval(250)
        self._memory_hide_timer.timeout.connect(self._expire_memory_hover)
        self.button_boxes: dict[str, tuple[float, float, float, float]] = {}
        self.button_hints: dict[str, str] = {}
        self.notice_box: tuple[float, float, float, float] = (0, 0, 0, 0)
        self._pressed_button: str | None = None
        self._press_global: QPoint | None = None
        self._press_geometry: QRect | None = None
        self._resize_edges = ""
        self._dragging = False
        self._pointer_moved = False
        self.compact_mode = False
        self._changing_mode = False
        self._expanded_geometry = QRect()
        self._mode_target_geometry = QRect()
        self._expanded_state = Qt.WindowNoState
        self._restore_maximized = False
        self._maximized_normal_geometry = QRect()
        self._last_mode_switch = 0.0
        self._ball_position: QPoint | None = None
        self._ball_wheel_delta = 0
        self._frame = QImage()
        self._mask_key: tuple[int, int, bool, bool] | None = None
        self._ready = False
        self.closed = False
        self.family = "Noto Sans CJK SC"

        saved = core._load_config()
        self.ball_size = _ball_size_value(saved.get("ball_size", FLOATING_BALL_SIZE))
        self.topmost = False if args.no_topmost else bool(saved.get("topmost", True))
        self.rounded_frame = bool(saved.get("rounded_frame", True))
        if "ball_x" in saved and "ball_y" in saved:
            self._ball_position = QPoint(int(saved["ball_x"]), int(saved["ball_y"]))
        flags = Qt.Window
        if not args.windowed:
            flags |= Qt.FramelessWindowHint
        if self.topmost:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, not args.windowed)
        self.setWindowTitle(f"{args.server_name} GPU Monitor")
        self.setWindowOpacity(args.opacity)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumWidth(MINIMUM_WIDTH)

        screen = QApplication.primaryScreen().availableGeometry()
        width = min(
            max(MINIMUM_WIDTH, int(saved.get("width", core.WINDOW_WIDTH))),
            screen.width(),
        )
        height = max(
            self._minimum_height(width), int(saved.get("height", core.WINDOW_HEIGHT))
        )
        x = int(saved.get("x", screen.right() - width - 28))
        y = int(saved.get("y", screen.y() + 48))
        x = min(max(screen.left(), x), max(screen.left(), screen.right() - width + 1))
        y = min(max(screen.top(), y), max(screen.top(), screen.bottom() - height + 1))
        self.setGeometry(x, y, width, height)
        self._expanded_geometry = QRect(self.geometry())
        self.canvas = PillowCanvas(width, height)
        self._ready = True

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(16)
        self.render_timer.timeout.connect(self._render)
        self.mode_timer = QTimer(self)
        self.mode_timer.setSingleShot(True)
        self.mode_timer.setInterval(50)
        self.mode_timer.timeout.connect(self._finish_mode_change)
        self.event_timer = QTimer(self)
        self.event_timer.setInterval(150)
        self.event_timer.timeout.connect(self._drain_events)
        self.event_timer.start()
        self.notice_timer = QTimer(self)
        self.notice_timer.setInterval(8000)
        self.notice_timer.timeout.connect(self._next_notice)
        self.training_timer = QTimer(self)
        self.training_timer.setInterval(TRAINING_CAROUSEL_INTERVAL)
        self.training_timer.timeout.connect(self._next_ball_training)
        self.training_animation = QVariantAnimation(self)
        self.training_animation.setDuration(TRAINING_TRANSITION_DURATION)
        self.training_animation.setStartValue(0.0)
        self.training_animation.setEndValue(1.0)
        self.training_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.training_animation.valueChanged.connect(self._animate_ball_training)
        self.training_animation.finished.connect(self._finish_ball_training_transition)
        self.notice_animation = QVariantAnimation(self)
        self.notice_animation.setDuration(500)
        self.notice_animation.setStartValue(0.0)
        self.notice_animation.setEndValue(1.0)
        self.notice_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.notice_animation.valueChanged.connect(self._animate_notice)
        self.notice_animation.finished.connect(self._finish_notice_scroll)

        poll_args = dict(
            api_base=args.api_base,
            server_id=args.server_id,
            server_name=args.server_name,
            timeout=args.timeout,
            demo=args.demo,
        )
        self.poller = core.Poller(self.events, interval=args.interval, **poll_args)
        self.notice_poller = core.AnnouncementPoller(
            self.events, interval=30, **poll_args
        )
        if start_pollers:
            self.poller.start()
            self.notice_poller.start()
            if self.ssh_poller:
                self.ssh_poller.start()
            if self.training_poller:
                self.training_poller.start()
            if self.disk_poller:
                self.disk_poller.start()
        self._render()
        if saved.get("compact_mode", False):
            self.set_compact(True)

    def _columns(self, width: int | None = None) -> int:
        return max(
            2, min(4, ((width or self.width()) - 36 + CARD_GAP) // (222 + CARD_GAP))
        )

    def _minimum_height(self, width: int) -> int:
        count = max(
            core.EXPECTED_GPU_COUNT, len(self.snapshot.gpus) if self.snapshot else 0
        )
        rows = math.ceil(count / self._columns(width))
        return GRID_TOP + rows * CARD_HEIGHT + (rows - 1) * CARD_GAP + 36

    @staticmethod
    def _inside(x: float, y: float, box: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _button_at(self, x: float, y: float) -> str | None:
        return next(
            (
                name
                for name, box in self.button_boxes.items()
                if self._inside(x, y, box)
            ),
            None,
        )

    def _card_at(self, x: float, y: float) -> int | None:
        return next(
            (
                index
                for index, box in self.card_boxes.items()
                if self._inside(x, y, box)
            ),
            None,
        )

    def _edges_at(self, position: QPoint) -> str:
        if self.compact_mode or self.args.windowed or self.isMaximized():
            return ""
        edges = ""
        if position.x() < RESIZE_MARGIN:
            edges += "l"
        elif position.x() >= self.width() - RESIZE_MARGIN:
            edges += "r"
        if position.y() < RESIZE_MARGIN:
            edges += "t"
        elif position.y() >= self.height() - RESIZE_MARGIN:
            edges += "b"
        return edges

    def _release_pointer(self) -> None:
        self._pressed_button = None
        self._pressed_login_page = False
        self._press_global = None
        self._press_geometry = None
        self._resize_edges = ""
        self._dragging = False
        self._pointer_moved = False

    def _record_geometry_intent(self, geometry: QRect) -> None:
        # Explicit UI/API moves are authoritative; native ConfigureNotify
        # events from an old mode must not replace that requested geometry.
        if self._ready:
            self._mode_target_geometry = QRect(geometry)
            if (
                not self.compact_mode
                and not self._changing_mode
                and not self.isMaximized()
            ):
                self._expanded_geometry = QRect(geometry)

    def setGeometry(self, *values: Any) -> None:
        geometry = QRect(values[0]) if len(values) == 1 else QRect(*values)
        self._record_geometry_intent(geometry)
        super().setGeometry(*values)

    def resize(self, *values: Any) -> None:
        geometry = QRect(self.geometry())
        geometry.setSize(QSize(values[0]) if len(values) == 1 else QSize(*values))
        self._record_geometry_intent(geometry)
        super().resize(*values)

    def move(self, *values: Any) -> None:
        geometry = QRect(self.geometry())
        geometry.moveTopLeft(QPoint(values[0]) if len(values) == 1 else QPoint(*values))
        self._record_geometry_intent(geometry)
        super().move(*values)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        self._release_pointer()
        self._press_global = event.globalPos()
        self._press_geometry = self.geometry()
        if self.compact_mode:
            self._pressed_login_page = self._ssh_login_at(event.pos())
            self._dragging = True
            self._hover_training_index = None
            self._sync_training_timer()
            self.setCursor(Qt.ClosedHandCursor)
            return
        self._pressed_button = self._button_at(event.x(), event.y())
        if self._pressed_button:
            return
        self._resize_edges = self._edges_at(event.pos())
        if not self._resize_edges and event.y() <= 68:
            self._dragging = True

    def mouseMoveEvent(self, event: Any) -> None:
        if self._press_global is not None and event.buttons() & Qt.LeftButton:
            delta = event.globalPos() - self._press_global
            geometry = self._press_geometry
            if self._dragging and geometry is not None:
                if self.compact_mode:
                    self._pointer_moved |= (
                        delta.manhattanLength() >= QApplication.startDragDistance()
                    )
                    if self._pointer_moved:
                        position = self._bounded_ball_position(
                            geometry.topLeft() + delta
                        )
                        self.move(position)
                        if self.mode_timer.isActive():
                            self._mode_target_geometry.moveTopLeft(position)
                    return
                self.move(geometry.topLeft() + delta)
                if self.mode_timer.isActive():
                    self._mode_target_geometry.moveTopLeft(geometry.topLeft() + delta)
                return
            if self._resize_edges and geometry is not None:
                width, height = geometry.width(), geometry.height()
                x, y = geometry.x(), geometry.y()
                if "l" in self._resize_edges:
                    width = max(MINIMUM_WIDTH, geometry.width() - delta.x())
                    x += geometry.width() - width
                elif "r" in self._resize_edges:
                    width = max(MINIMUM_WIDTH, geometry.width() + delta.x())
                minimum_height = self._minimum_height(width)
                if "t" in self._resize_edges:
                    height = max(minimum_height, geometry.height() - delta.y())
                    y += geometry.height() - height
                elif "b" in self._resize_edges:
                    height = max(minimum_height, geometry.height() + delta.y())
                else:
                    height = max(minimum_height, height)
                self.setGeometry(x, y, width, height)
                if self.mode_timer.isActive():
                    self._mode_target_geometry = QRect(x, y, width, height)
                return

        if self.compact_mode:
            login = self._ssh_login_at(event.pos())
            self.setCursor(Qt.PointingHandCursor if login else Qt.OpenHandCursor)
            self._set_training_hover(
                BALL_SSH_LOGIN_PAGE if login else self._training_ring_at(event.pos())
            )
            self.setToolTip(self._compact_tooltip())
            return

        next_button = self._button_at(event.x(), event.y())
        next_card = self._card_at(event.x(), event.y())
        edges = self._edges_at(event.pos())
        if edges in {"lt", "rb"}:
            cursor = Qt.SizeFDiagCursor
        elif edges in {"rt", "lb"}:
            cursor = Qt.SizeBDiagCursor
        elif edges in {"l", "r"}:
            cursor = Qt.SizeHorCursor
        elif edges in {"t", "b"}:
            cursor = Qt.SizeVerCursor
        elif next_button or self._inside(event.x(), event.y(), self.notice_box):
            cursor = Qt.PointingHandCursor
        elif event.y() <= 68:
            cursor = Qt.SizeAllCursor
        else:
            cursor = Qt.ArrowCursor
        self.setCursor(cursor)
        hint = self.button_hints.get(next_button, "")
        if self._inside(event.x(), event.y(), self.notice_box):
            hint = self._current_notice_text() + "\n点击查看全部公告"
        elif next_card is not None:
            hint = self._card_request_hint(next_card)
        if self._memory_bar_at(event.x(), event.y()) is not None:
            self._memory_pointer_on_bar = True
            self._memory_hide_timer.stop()
            self._memory_hover_position = QPoint(event.pos())
            self._memory_hover_global_position = QPoint(event.globalPos())
            self._refresh_memory_hover()
        else:
            self._defer_memory_hover_clear()
            self.setToolTip(hint)
        if next_button != self.hovered_button or next_card != self.hovered_card:
            self.hovered_button, self.hovered_card = next_button, next_card
            self._schedule_render()

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)
        if self.compact_mode:
            moved = self._pointer_moved
            login = self._pressed_login_page
            self._release_pointer()
            self.setCursor(Qt.OpenHandCursor)
            if not moved:
                if login and self._ball_center_at(event.pos()):
                    self.login_ssh()
                else:
                    self.set_compact(False)
            else:
                self._sync_training_timer()
            return
        button = self._pressed_button
        was_moving = self._dragging or bool(self._resize_edges)
        self._release_pointer()
        if button and button == self._button_at(event.x(), event.y()):
            self._activate_button(button)
        elif not was_moving and self._inside(event.x(), event.y(), self.notice_box):
            if event.x() > self.width() - 60:
                self._next_notice()
            else:
                self.show_announcements()

    def mouseDoubleClickEvent(self, event: Any) -> None:
        # A repeated header click must still start a new drag or button press.
        if self._card_at(event.x(), event.y()) is not None:
            self.open_dashboard()
        else:
            self.mousePressEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._defer_memory_hover_clear()
        self.hovered_button = None
        self.hovered_card = None
        self._set_training_hover(None)
        self._schedule_render()
        super().leaveEvent(event)

    def hideEvent(self, event: Any) -> None:
        self._clear_memory_hover()
        self._release_pointer()
        self._hover_training_index = None
        if hasattr(self, "training_timer"):
            self.training_timer.stop()
        self._stop_ball_training_transition()
        super().hideEvent(event)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._sync_training_timer()

    def resizeEvent(self, event: Any) -> None:
        if self._ready:
            if (
                not self.compact_mode
                and not self.isMaximized()
                and self._mode_target_geometry.width() >= MINIMUM_WIDTH
                and (
                    self.width() < MINIMUM_WIDTH
                    or 0 < event.oldSize().width() < MINIMUM_WIDTH
                    or (
                        self.size() != self._mode_target_geometry.size()
                        and (
                            not self.args.windowed
                            or (
                                monotonic() - self._last_mode_switch < 1.0
                                and not QApplication.mouseButtons() & Qt.LeftButton
                            )
                        )
                    )
                )
            ):
                # A delayed compact-size event is never a valid full-window
                # resize. Reconcile it even if it arrives after the debounce.
                self._changing_mode = True
            if self._changing_mode:
                self.mode_timer.start()
                if (
                    self._mode_target_geometry.isValid()
                    and (self.compact_mode or not self.isMaximized())
                    and self.size() != self._mode_target_geometry.size()
                ):
                    self.setGeometry(self._mode_target_geometry)
            elif not self.compact_mode:
                self.setMinimumHeight(self._minimum_height(self.width()))
                self._remember_expanded_geometry()
            self._schedule_render()
        super().resizeEvent(event)

    def moveEvent(self, event: Any) -> None:
        if self._ready and (
            not self._mode_target_geometry.isValid()
            or self.size() == self._mode_target_geometry.size()
        ):
            self._remember_expanded_geometry()
        super().moveEvent(event)

    def changeEvent(self, event: Any) -> None:
        if (
            self._ready
            and event.type() == QEvent.WindowStateChange
            and event.oldState() & Qt.WindowMaximized
            and not self.isMaximized()
            and not self.compact_mode
            and self._maximized_normal_geometry.isValid()
        ):
            self._mode_target_geometry = QRect(self._maximized_normal_geometry)
            self._expanded_geometry = QRect(self._maximized_normal_geometry)
            self._maximized_normal_geometry = QRect()
            self._changing_mode = True
            self.mode_timer.start()
        super().changeEvent(event)
        self._sync_training_timer()

    def _remember_expanded_geometry(self) -> None:
        if (
            not self.compact_mode
            and not self._changing_mode
            and not self.isMaximized()
            and not self.isMinimized()
            and self.width() >= MINIMUM_WIDTH
        ):
            self._expanded_geometry = QRect(self.geometry())
            self._mode_target_geometry = QRect(self.geometry())

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self._frame)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F5:
            self.force_refresh()
        elif event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            self.copy_status()
        elif event.key() == Qt.Key_M and event.modifiers() & Qt.ControlModifier:
            self.set_compact(not self.compact_mode)
        elif (
            self.compact_mode
            and event.modifiers() & Qt.ControlModifier
            and event.key() in (Qt.Key_Plus, Qt.Key_Equal, Qt.Key_Minus)
        ):
            step = (
                -FLOATING_BALL_SIZE_STEP
                if event.key() == Qt.Key_Minus
                else FLOATING_BALL_SIZE_STEP
            )
            self.set_ball_size(self.ball_size + step)
        elif self.compact_mode and event.key() in (Qt.Key_Return, Qt.Key_Space):
            if self._visible_ball_status_page() == BALL_SSH_LOGIN_PAGE:
                self.login_ssh()
            else:
                self.set_compact(False)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: Any) -> None:
        if not self.compact_mode or not event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return
        self._ball_wheel_delta += event.angleDelta().y()
        steps = int(self._ball_wheel_delta / 120)
        self._ball_wheel_delta -= steps * 120
        if steps:
            self.set_ball_size(self.ball_size + steps * FLOATING_BALL_SIZE_STEP)
        event.accept()

    def contextMenuEvent(self, event: Any) -> None:
        self._release_pointer()
        menu = QMenu(self)
        menu.setFont(QFont(self.family, 10))
        menu.setStyleSheet(
            "QMenu {background:#0B111C; color:#F5F7FB; border:1px solid #22324A; padding:6px;}"
            "QMenu::item {padding:8px 22px;}"
            "QMenu::item:selected {background:#1B2940; border-radius:5px;}"
            "QMenu::separator {height:1px; background:#22324A; margin:5px 8px;}"
        )
        menu.addAction(
            "展开详细窗口\tCtrl+M" if self.compact_mode else "收起为浮动球\tCtrl+M",
            lambda: self.set_compact(not self.compact_mode),
        )
        menu.addAction("最小化到任务栏", self.minimize)
        menu.addAction(
            f"调整悬浮球大小…（{self.ball_size} px）\tCtrl+滚轮",
            self.show_ball_size_dialog,
        )
        menu.addAction(
            "恢复默认悬浮球大小", lambda: self.set_ball_size(FLOATING_BALL_SIZE)
        )
        menu.addSeparator()
        menu.addAction("立即刷新\tF5", self.force_refresh)
        menu.addAction("打开完整监控页", self.open_dashboard)
        pin = menu.addAction("保持置顶")
        pin.setCheckable(True)
        pin.setChecked(self.topmost)
        pin.triggered.connect(self.set_topmost)
        corners = menu.addAction("圆角 / 直角外框", self.toggle_corners)
        corners.setEnabled(not self.compact_mode)
        menu.addAction("查看全部公告", self.show_announcements)
        menu.addAction("查看训练进度", self.show_training).setEnabled(
            self.training_enabled
        )
        menu.addSeparator()
        menu.addAction("复制当前状态\tCtrl+C", self.copy_status)
        menu.addAction("恢复默认位置和大小", self.reset_position)
        menu.addSeparator()
        menu.addAction("退出\tEsc", self.close)
        menu.exec_(event.globalPos())
        menu.deleteLater()

    def _activate_button(self, name: str) -> None:
        actions = {
            "open": self.open_dashboard,
            "pin": lambda: self.set_topmost(not self.topmost),
            "corners": self.toggle_corners,
            "compact": lambda: self.set_compact(True),
            "minimize": self.minimize,
            "close": self.close,
        }
        action = actions.get(name)
        if action:
            action()

    def minimize(self) -> None:
        self._release_pointer()
        self.showMinimized()

    def _bounded_ball_position(self, position: QPoint) -> QPoint:
        center = position + QPoint(self.ball_size // 2, self.ball_size // 2)
        screen = QApplication.screenAt(center) or self.screen()
        area = screen.availableGeometry()
        return QPoint(
            min(
                max(area.left(), position.x()),
                max(area.left(), area.right() - self.ball_size + 1),
            ),
            min(
                max(area.top(), position.y()),
                max(area.top(), area.bottom() - self.ball_size + 1),
            ),
        )

    def set_ball_size(self, size: int) -> None:
        """Scale only the ball, preserving its center and the full-window size."""
        size = _ball_size_value(size)
        if self.closed or size == self.ball_size:
            return
        self._release_pointer()
        self._hover_training_index = None
        if self.compact_mode:
            geometry = (
                self._mode_target_geometry
                if self.mode_timer.isActive()
                else self.normalGeometry() if self.isMinimized() else self.geometry()
            )
            center = geometry.center()
            self.ball_size = size
            position = self._bounded_ball_position(
                center - QPoint((size - 1) // 2, (size - 1) // 2)
            )
            self._changing_mode = True
            self._last_mode_switch = monotonic()
            self._mode_target_geometry = QRect(position.x(), position.y(), size, size)
            self.setMaximumSize(size, size)
            self.setGeometry(self._mode_target_geometry)
            self._ball_position = QPoint(position)
            self.mode_timer.start()
        else:
            self.ball_size = size
        self._mask_key = None
        self._sync_training_timer()
        self._render()

    def show_ball_size_dialog(self) -> None:
        original_size = self.ball_size
        original_position = (
            QPoint(
                self._mode_target_geometry.topLeft()
                if self.mode_timer.isActive()
                else self.pos()
            )
            if self.compact_mode
            else None
        )
        dialog = QInputDialog(self)
        dialog.setWindowTitle("调整悬浮球大小")
        dialog.setFont(QFont(self.family, 10))
        dialog.setStyleSheet(
            "QInputDialog {background:#0B111C; color:#F5F7FB;}"
            "QLabel {color:#F5F7FB;}"
            "QSpinBox {background:#111A29; color:#F5F7FB; border:1px solid #22324A; padding:6px;}"
            "QPushButton {background:#1B2940; color:#F5F7FB; border:0; padding:8px 18px;}"
        )
        dialog.setInputMode(QInputDialog.IntInput)
        dialog.setLabelText(
            f"直径（{FLOATING_BALL_MIN_SIZE}–{FLOATING_BALL_MAX_SIZE} 像素）\nCtrl＋滚轮或 Ctrl＋加减号也可调整；取消会恢复原尺寸。"
        )
        dialog.setIntRange(FLOATING_BALL_MIN_SIZE, FLOATING_BALL_MAX_SIZE)
        dialog.setIntStep(FLOATING_BALL_SIZE_STEP)
        dialog.setIntValue(self.ball_size)
        dialog.setOkButtonText("确定")
        dialog.setCancelButtonText("取消")
        dialog.intValueChanged.connect(self.set_ball_size)
        if dialog.exec_() != QDialog.Accepted:
            self.set_ball_size(original_size)
            if original_position is not None and self.compact_mode:
                # Previewing a larger ball near a screen edge can shift it.
                # Cancel restores both its size and its original position.
                self.move(original_position)
                self._ball_position = QPoint(original_position)
        dialog.deleteLater()

    def set_compact(self, enabled: bool) -> None:
        """Reuse the native window so polling and taskbar recovery remain intact."""
        if self.closed or bool(enabled) == self.compact_mode:
            return
        self._clear_memory_hover()
        pending_geometry = (
            QRect(self._mode_target_geometry) if self.mode_timer.isActive() else None
        )
        self._last_mode_switch = monotonic()
        self._changing_mode = True
        self.mode_timer.stop()
        was_visible = self.isVisible()
        self._release_pointer()
        self.hovered_card = self.hovered_button = None
        self._hover_training_index = None
        self.training_timer.stop()
        self._stop_ball_training_transition()
        self.notice_animation.stop()
        self.notice_timer.stop()
        self._previous_notice = None
        self._notice_scroll_progress = 1.0
        if enabled:
            geometry = (
                pending_geometry
                if pending_geometry is not None
                else (
                    self.normalGeometry()
                    if self.isMaximized() or self.isMinimized()
                    else self.geometry()
                )
            )
            if geometry.isValid():
                self._expanded_geometry = QRect(geometry)
            if pending_geometry is None or not self._restore_maximized:
                self._expanded_state = self.windowState() & Qt.WindowMaximized
            if self._expanded_state:
                self._maximized_normal_geometry = QRect(self._expanded_geometry)
            self._restore_maximized = False
            position = (
                self._ball_position
                if self._ball_position is not None
                else QPoint(
                    self._expanded_geometry.right() - self.ball_size + 1,
                    self._expanded_geometry.top(),
                )
            )
            self.compact_mode = True
            position = self._bounded_ball_position(position)
            self._mode_target_geometry = QRect(
                position.x(), position.y(), self.ball_size, self.ball_size
            )
            self.setWindowState(Qt.WindowNoState)
            self.setWindowFlag(Qt.FramelessWindowHint, True)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setMinimumSize(0, 0)
            self.setMaximumSize(self.ball_size, self.ball_size)
            self.setGeometry(position.x(), position.y(), self.ball_size, self.ball_size)
            self.setCursor(Qt.OpenHandCursor)
        else:
            self._ball_position = QPoint(
                pending_geometry.topLeft()
                if pending_geometry is not None
                else self.pos()
            )
            self.compact_mode = False
            self._mode_target_geometry = QRect(self._expanded_geometry)
            self.setWindowFlag(Qt.FramelessWindowHint, not self.args.windowed)
            self.setAttribute(Qt.WA_TranslucentBackground, not self.args.windowed)
            self.setMaximumSize(16777215, 16777215)
            self.setMinimumSize(
                MINIMUM_WIDTH, self._minimum_height(self._expanded_geometry.width())
            )
            self.setGeometry(self._expanded_geometry)
            self.setCursor(Qt.ArrowCursor)
            self.setToolTip("")
            if len(self.banner_announcements) > 1:
                self.notice_timer.start()
            self._restore_maximized = bool(self._expanded_state)
        self._mask_key = None
        if was_visible:
            self.show()
            # Reapply after mapping: switching a decorated X11 window to/from
            # frameless can otherwise restore the compact native geometry.
            if enabled:
                self.setGeometry(
                    position.x(), position.y(), self.ball_size, self.ball_size
                )
            else:
                self.setGeometry(self._expanded_geometry)
        self._mode_target_geometry = (
            QRect(position.x(), position.y(), self.ball_size, self.ball_size)
            if enabled
            else QRect(self._expanded_geometry)
        )
        # Old ConfigureNotify events can arrive after mapping the new X11 size.
        # Suspend responsive resize constraints through that short transition,
        # then reconcile the intended geometry once the native events settle.
        self.mode_timer.start()
        self._sync_training_timer()
        self._render()

    def _finish_mode_change(self) -> None:
        if self.closed:
            return
        if not self.compact_mode:
            self.setMinimumSize(
                MINIMUM_WIDTH, self._minimum_height(self._mode_target_geometry.width())
            )
        if not self.isMaximized() and self.geometry() != self._mode_target_geometry:
            self.setGeometry(self._mode_target_geometry)
            self.mode_timer.start()
            self._render()
            return
        if self._restore_maximized:
            self._restore_maximized = False
            self.setWindowState(self._expanded_state)
            self.mode_timer.start()
            return
        self._changing_mode = False
        self._mask_key = None
        self._render()

    def set_topmost(self, enabled: bool) -> None:
        geometry = self.geometry()
        state = self.windowState()
        self.topmost = bool(enabled)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, self.topmost)
        self.setGeometry(geometry)
        self.show()
        self.setWindowState(state)
        self._mask_key = None
        self._render()

    def toggle_corners(self) -> None:
        self.rounded_frame = not self.rounded_frame
        self._render()

    def reset_position(self) -> None:
        if self.compact_mode:
            self.set_ball_size(FLOATING_BALL_SIZE)
            screen = QApplication.primaryScreen().availableGeometry()
            self.move(
                self._bounded_ball_position(
                    QPoint(screen.right() - self.ball_size - 28, screen.y() + 48)
                )
            )
            self._render()
            return
        screen = QApplication.primaryScreen().availableGeometry()
        width = min(core.WINDOW_WIDTH, screen.width())
        height = max(core.WINDOW_HEIGHT, self._minimum_height(width))
        self.setGeometry(
            max(screen.left(), screen.right() - width - 28),
            screen.y() + 48,
            width,
            height,
        )
        self._render()

    def open_dashboard(self) -> None:
        url = f"{self.args.api_base.rstrip('/')}/index.html#{self.args.server_id}"
        if not QDesktopServices.openUrl(QUrl(url)):
            self.status_message = "网页打开失败，请检查默认浏览器"
            self._render()

    def copy_status(self) -> None:
        if self.snapshot:
            QApplication.clipboard().setText(core.format_snapshot(self.snapshot))
            self.status_message = "已复制当前状态"
            self._render()

    @property
    def ssh_login_required(self) -> bool:
        return bool(self.training_enabled or self.disk_enabled)

    @property
    def ssh_features_available(self) -> bool:
        return not self.ssh_login_required or self.ssh_connected is True

    @property
    def ssh_login_pending(self) -> bool:
        return bool(
            self.ssh_login_process is not None
            or monotonic() < self._ssh_login_pending_until
        )

    def _apply_ssh_session(self, session: remote.SessionEvent) -> None:
        was_connected = self.ssh_connected
        self.ssh_connected, self.ssh_error = session.connected, session.error
        if not session.connected or was_connected is not True:
            self._process_session_started_at = monotonic()
            # A fresh login must not revive PIDs cached before a disconnect.
            self.training_samples = tuple(
                replace(sample, processes=()) for sample in self.training_samples
            )
        if session.connected:
            self.ssh_login_error = ""
            self._ssh_login_pending_until = 0.0
            if was_connected is not True:
                # Do not wait for the disk reader's 15-minute schedule after
                # authentication, or reuse old log metadata as live progress.
                if self.training_poller:
                    self.training_poller.refresh()
                if self.disk_poller:
                    self.disk_poller.refresh()

    def _poll_ssh_login_terminal(self) -> bool:
        if self.ssh_login_process is not None:
            result = self.ssh_login_process.poll()
            if result is not None:
                self.ssh_login_process = None  # poll() reaps only our launcher.
                self._ssh_login_pending_until = (
                    monotonic() + 10 if result == 0 and not self.ssh_connected else 0
                )
                if result and not self.ssh_connected:
                    self.ssh_login_error = (
                        "登录终端已关闭或认证未完成，请点击登录页重试。"
                    )
                if self.ssh_poller:
                    self.ssh_poller.refresh()
                return True
        if (
            self._ssh_login_pending_until
            and monotonic() >= self._ssh_login_pending_until
        ):
            self._ssh_login_pending_until = 0.0
            if not self.ssh_connected:
                self.ssh_login_error = "尚未检测到共享连接，请点击登录页重试。"
            return True
        return False

    def login_ssh(self) -> None:
        if self.closed or self.ssh_features_available:
            return
        if self.ssh_login_pending:
            if self.ssh_poller:
                self.ssh_poller.refresh()
            return
        if not self.ssh_config:
            self.ssh_login_error = "SSH 配置不可用，请检查训练日志配置文件。"
        else:
            try:
                self.ssh_login_process = remote.open_login_terminal(self.ssh_config)
                self.ssh_login_error = ""
            except (OSError, RuntimeError) as exc:
                self.ssh_login_error = str(exc)
        # An opened terminal is not proof of authentication. Only an actual
        # successful probe restores the private carousel pages.
        if self.ssh_poller:
            self.ssh_poller.refresh()
        self._sync_training_timer()
        self._render()

    def _visible_ball_status_page(self) -> int:
        return (
            self._ball_previous_page
            if self._ball_previous_page is not None
            and self._ball_transition_progress < 0.5
            else self.displayed_training_index
        )

    def _ball_center_at(self, position: QPoint) -> bool:
        scale = self.ball_size / FLOATING_BALL_BASE_SIZE
        return self._inside(
            position.x() / scale, position.y() / scale, (20, 25, 83, 80)
        )

    def _ssh_login_at(self, position: QPoint) -> bool:
        return bool(
            self.compact_mode
            and not self.ssh_features_available
            and self._visible_ball_status_page() == BALL_SSH_LOGIN_PAGE
            and self._ball_center_at(position)
        )

    def force_refresh(self) -> None:
        self.poller.refresh()
        self.notice_poller.refresh()
        if self.ssh_poller:
            self.ssh_poller.refresh()
        if self.training_poller:
            self.training_poller.refresh()
        if self.disk_poller:
            self.disk_poller.refresh()
        self.status_message = "正在刷新"
        self._render()

    def _drain_events(self) -> None:
        changed = self._poll_ssh_login_terminal()
        while not self.closed:
            try:
                session = self.ssh_events.get_nowait()
            except queue.Empty:
                break
            self._apply_ssh_session(session)
            changed = True
        while not self.closed:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if event.snapshot:
                self.snapshot = event.snapshot
                self.online = True
                self.last_error = ""
                self.status_message = "实时监控中"
                for gpu in event.snapshot.gpus:
                    self.histories[gpu.index].append(gpu.utilization)
                if not self.compact_mode and not self._changing_mode:
                    self.setMinimumHeight(self._minimum_height(self.width()))
            elif event.error:
                self.online = False
                self.last_error = event.error
                self.status_message = "连接中断，正在自动重试"
            if event.announcements is not None:
                if event.announcements != self.announcements:
                    self.notice_animation.stop()
                    self._previous_notice = None
                    self._notice_scroll_progress = 1.0
                    self._notice_line_cache.clear()
                    self.notice_index = 0
                self.announcements = event.announcements
                self.notice_loaded = True
                self.notice_error = ""
                if len(self.banner_announcements) > 1 and not self.compact_mode:
                    if not self.notice_timer.isActive():
                        self.notice_timer.start()
                else:
                    self.notice_timer.stop()
            elif event.notice_error:
                self.notice_error = event.notice_error
        while not self.closed:
            try:
                event = self.training_events.get_nowait()
            except queue.Empty:
                break
            self.training_error = event.error
            self.training_samples = remote.apply_training_event(
                self.training_samples, event
            )
            changed = True
        while not self.closed:
            try:
                event = self.disk_events.get_nowait()
            except queue.Empty:
                break
            self.disk_sample = remote.apply_disk_event(self.disk_sample, event)
            changed = True
        if changed:
            self._sync_training_timer()
            self._render()

    @property
    def banner_announcements(self) -> tuple[core.Announcement, ...]:
        return tuple(item for item in self.announcements if not item.is_server_rules)

    @property
    def reserved_gpu_indexes(self) -> frozenset[int]:
        # Reservations must not disappear when the banner rotates to another notice.
        return frozenset(
            index for item in self.banner_announcements for index in item.gpu_indexes
        )

    @property
    def own_gpu_indexes(self) -> frozenset[int]:
        return frozenset(
            gpu.index
            for gpu in (self.snapshot.gpus if self.snapshot else ())
            if gpu.has_user(HIGHLIGHT_USERNAME)
        )

    @property
    def all_training_steps_complete(self) -> bool:
        """Both logs reached their final step, not merely a rounded 100% label.

        This changes the ball display, not the job's success status: final
        checkpoint saving may still be using a card, which is not available.
        """
        return bool(
            self.ssh_features_available
            and self.training_enabled
            and self.training_config
            and len(self.training_samples) == 2
            and all(
                sample.progress is not None
                and sample.progress.total > 0
                and sample.progress.step == sample.progress.total
                and sample.is_fresh(self.training_config.stale_after)
                for sample in self.training_samples
            )
        )

    @property
    def no_own_gpu_processes(self) -> bool:
        """The last GPU snapshot confirms no owned cards; unknown is not idle."""
        return bool(
            self.snapshot is not None
            and self.snapshot.gpus
            and not self.own_gpu_indexes
        )

    @property
    def ball_hide_training(self) -> bool:
        """Hide old jobs when idle or complete; disk-page eligibility is separate."""
        return self.no_own_gpu_processes or self.all_training_steps_complete

    @property
    def show_ball_availability(self) -> bool:
        """Show free cards when yzh has room or both training logs are complete."""
        return (
            self.online
            and self.snapshot is not None
            and (len(self.own_gpu_indexes) < 2 or self.all_training_steps_complete)
        )

    @property
    def conflict_gpu_indexes(self) -> frozenset[int]:
        return self.own_gpu_indexes & self.reserved_gpu_indexes

    @property
    def available_gpu_indexes(self) -> frozenset[int]:
        """An idle requested card is not available for new work."""
        return frozenset(
            gpu.index
            for gpu in (self.snapshot.gpus if self.snapshot else ())
            if gpu.utilization < 10
            and gpu.memory_percent < 10
            and not gpu.has_user(HIGHLIGHT_USERNAME)
            and gpu.index not in self.reserved_gpu_indexes
        )

    def _compact_tooltip(self) -> str:
        lines = [f"{self.args.server_name} · 浮动监控"]

        def indexes(values: frozenset[int]) -> str:
            return "、".join(str(index) for index in sorted(values)) or "无"

        if not self.online:
            lines.append(
                "连接中断，以下为上次数据"
                if self.snapshot
                else "连接中断，正在重试" if self.last_error else "正在连接服务器"
            )
        if self.snapshot:
            lines.append(f"平均显存 {self.snapshot.average_memory_percent:.0f}%")
            if (
                self.show_ball_availability
                and self.notice_loaded
                and not self.notice_error
            ):
                lines.append(f"可用 GPU：{indexes(self.available_gpu_indexes)}")
                lines.append("（利用率、显存均低于 10%，且未被申请）")
            own_memory = sum(
                usage.memory_gib
                for gpu in self.snapshot.gpus
                for usage in gpu.user_usages
                if usage.username.casefold() == HIGHLIGHT_USERNAME.casefold()
            )
            lines.append(
                f"{HIGHLIGHT_USERNAME}：GPU {indexes(self.own_gpu_indexes)} · "
                f"显存合计 {core._compact_memory(own_memory)}"
                if self.own_gpu_indexes
                else f"{HIGHLIGHT_USERNAME}：未使用 GPU"
            )
        if self.show_ball_availability and (
            not self.notice_loaded or self.notice_error
        ):
            lines.append("申请信息尚未确认；空闲卡不代表可用卡")
        for notice in self.banner_announcements:
            if notice.kind == "gpu_usage":
                cards, deadline = _request_badges(notice)
                lines.append(f"申请：{cards} · {deadline}")
        if self.snapshot:
            lines.append(
                f"GPU 更新于 {datetime.fromtimestamp(self.snapshot.received_at).strftime('%H:%M:%S')}"
            )
        if self.ssh_login_required and not self.ssh_features_available:
            lines.extend(("", "SSH 未登录；训练、磁盘轮播页和进度环已暂停。"))
            lines.append("登录页中央单击打开本机终端，密码只在终端输入。")
            if self.ssh_login_pending:
                lines.append("请在已打开的终端完成认证；关闭终端可取消后重试。")
            if self.ssh_login_error or self.ssh_error:
                lines.append(self.ssh_login_error or self.ssh_error)
        elif self.training_enabled:
            lines.extend(("", self._training_text(include_login=False)))
        if self.disk_enabled and self.ssh_features_available:
            lines.extend(("", remote.disk_text(self.disk_sample, self.disk_config)))
            if self.disk_sample.over_quota:
                lines.append("红色外框：磁盘占用超过 3T；数据时间见上方")
        lines.append(f"直径 {self.ball_size} px · Ctrl＋滚轮缩放 · 右键调整大小")
        lines.append(
            "登录页中央单击登录；外围单击展开 · 拖动移动 · 右键菜单"
            if self.ssh_login_required and not self.ssh_features_available
            else "单击展开 · 拖动移动 · 右键菜单"
        )
        return "\n".join(lines)

    def _training_text(self, *, include_login: bool = True) -> str:
        lines = [
            "训练进度：蓝色外圈 = 任务 1；紫色内圈 = 任务 2",
            "中央每 4 秒轮播一个任务：百分比、GPU 编号、日志预计剩余时间各一行，颜色对应圆环。",
            "仅占用一张卡时，对应任务与可用 GPU 数量 / 卡号轮播；切换时整组文字平滑淡出淡入。",
            "两份日志步数均已完成时，隐藏旧训练进度；有 yzh GPU 进程且日志开始新训练后恢复任务轮播。",
            "网页确认 yzh 没有 GPU 进程时，轮播可用 GPU 与已启用的磁盘用量；未登录时改为轮播登录页，不显示旧训练进度。",
            "鼠标停在任一进度环上立即显示对应任务并暂停轮播；移开后恢复。每个完整圆环各为 100%。",
            "GPU 编号自动关联日志与进程 CUDA_VISIBLE_DEVICES；GPU ? 表示无法可靠识别，不猜测文件名或用卡顺序。",
        ]
        if self.training_config:
            for sample in self.training_samples:
                lines.extend(("", remote.training_text(sample, self.training_config)))
            lines.append(
                "灰色弧段 = 上次进度 / 日志未更新；没有进度的日志不显示任务页和圆环"
            )
            if include_login:
                lines.extend(
                    (
                        "",
                        f"每 {self.training_config.interval:g} 秒只读日志末尾；{self.training_config.stale_after:g} 秒未更新时变灰。",
                        "连接失效时，在自己的终端重新登录（密码无需输入监控程序）：",
                        "也可收起为悬浮球，在未登录轮播页点击登录；训练 / 磁盘页在连接恢复前暂停。",
                        remote.login_command(self.training_config),
                    )
                )
        if self.training_error and not self.training_config:
            lines.append(self.training_error)
        return "\n".join(lines)

    def show_training(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.args.server_name} · yzh 训练进度")
        dialog.setFont(QFont(self.family, 11))
        dialog.setStyleSheet(
            "QDialog,QTextEdit {background:#0B111C; color:#F5F7FB;}"
            "QTextEdit {border:1px solid #22324A; padding:12px;}"
            "QPushButton {background:#1B2940; color:#F5F7FB; border:0; padding:8px 22px;}"
        )
        layout = QVBoxLayout(dialog)
        viewer = QTextEdit(dialog)
        viewer.setReadOnly(True)

        def refresh_viewer() -> None:
            # Plain text also prevents log paths/SSH errors becoming active HTML.
            text = self._training_text()
            if viewer.toPlainText() != text:
                viewer.setPlainText(text)

        refresh_viewer()
        timer = QTimer(dialog)
        timer.setInterval(1000)
        timer.timeout.connect(refresh_viewer)
        timer.start()
        layout.addWidget(viewer)
        close = QPushButton("关闭", dialog)
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, alignment=Qt.AlignRight)
        dialog.resize(780, 460)
        dialog.exec_()
        timer.stop()
        dialog.deleteLater()

    def _current_notice_text(self) -> str:
        notices = self.banner_announcements
        if notices:
            return notices[self.notice_index % len(notices)].text
        if self.notice_error:
            return "公告暂时无法获取，正在重试"
        return (
            "暂无 WN-A100 当前申请或临时公告"
            if self.notice_loaded
            else "正在读取当前公告…"
        )

    def _next_notice(self) -> None:
        if self.compact_mode:
            return
        notices = self.banner_announcements
        if len(notices) > 1 and self._previous_notice is None:
            self._previous_notice = notices[self.notice_index % len(notices)]
            self.notice_index = (self.notice_index + 1) % len(notices)
            self._notice_scroll_progress = 0.0
            self.notice_animation.start()
            self.notice_timer.start()
            self._render()

    def _animate_notice(self, progress: Any) -> None:
        self._notice_scroll_progress = float(progress)
        if self._ready and not self.closed and not self.compact_mode:
            # Repaint just the clipped banner rather than recomputing all GPU cards
            # on every animation frame.
            if (self.canvas.width, self.canvas.height) != (self.width(), self.height()):
                self._render()
            else:
                self._draw_notices(self.width())
                self._present_canvas()

    def _finish_notice_scroll(self) -> None:
        self._previous_notice = None
        self._animate_notice(1.0)

    def show_announcements(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.args.server_name} 当前公告")
        dialog.setFont(QFont(self.family, 11))
        dialog.setStyleSheet(
            "QDialog,QTextEdit {background:#0B111C; color:#F5F7FB;}"
            "QTextEdit {border:1px solid #22324A; padding:12px;}"
            "QPushButton {background:#1B2940; color:#F5F7FB; border:0; padding:8px 22px;}"
        )
        layout = QVBoxLayout(dialog)
        viewer = QTextEdit(dialog)
        viewer.setReadOnly(True)
        viewer.setHtml(
            _announcements_html(self.announcements, self._current_notice_text())
        )
        layout.addWidget(viewer)
        close = QPushButton("关闭", dialog)
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, alignment=Qt.AlignRight)
        dialog.resize(780, 460)
        dialog.exec_()
        dialog.deleteLater()

    def _schedule_render(self) -> None:
        if self._ready and not self.closed and not self.render_timer.isActive():
            self.render_timer.start()

    def _apply_mask(self) -> None:
        rounded = (
            self.rounded_frame and not self.args.windowed and not self.isMaximized()
        )
        key = (self.width(), self.height(), rounded, self.compact_mode)
        if key == self._mask_key:
            return
        self._mask_key = key
        if self.compact_mode:
            self.setMask(QRegion(self.rect(), QRegion.Ellipse))
        elif rounded:
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), 18, 18)
            self.setMask(QRegion(path.toFillPolygon().toPolygon()))
        else:
            # An explicit rectangular region also updates the X11 bounding/input
            # shape. Some compositors retain the previous shape after clearMask().
            self.setMask(QRegion(self.rect()))

    def closeEvent(self, event: Any) -> None:
        self._clear_memory_hover()
        self.closed = True
        self.poller.stop()
        self.notice_poller.stop()
        if self.training_poller:
            self.training_poller.stop()
        if self.disk_poller:
            self.disk_poller.stop()
        if self.ssh_poller:
            self.ssh_poller.stop()
        self.event_timer.stop()
        self.notice_timer.stop()
        self.training_timer.stop()
        self._stop_ball_training_transition()
        self.notice_animation.stop()
        self.render_timer.stop()
        self.mode_timer.stop()
        geometry = (
            self._expanded_geometry
            if self.compact_mode
            else self.normalGeometry() if self.isMinimized() else self.geometry()
        )
        ball_position = (
            (self.normalGeometry().topLeft() if self.isMinimized() else self.pos())
            if self.compact_mode
            else self._ball_position
        )
        ball_config = (
            {"ball_x": ball_position.x(), "ball_y": ball_position.y()}
            if ball_position is not None
            else {}
        )
        core._save_config(
            dict(
                x=geometry.x(),
                y=geometry.y(),
                width=geometry.width(),
                height=geometry.height(),
                topmost=self.topmost,
                rounded_frame=self.rounded_frame,
                compact_mode=self.compact_mode,
                ball_size=self.ball_size,
                **ball_config,
            )
        )
        event.accept()

    def _draw_text(self, x: float, y: float, text: str, **kwargs: Any) -> int:
        defaults = {"fill": COLORS["text"], "font": (self.family, 10), "anchor": "w"}
        defaults.update(kwargs)
        return self.canvas.create_text(x, y, text=text, **defaults)

    def _draw_button(self, name: str, x: float, glyph: str, hint: str) -> None:
        y = 17
        size = 30
        self.button_boxes[name] = (x, y, x + size, y + size)
        self.button_hints[name] = hint
        fill = (
            COLORS["button_hover"] if self.hovered_button == name else COLORS["panel"]
        )
        self.canvas.rounded_rectangle(
            x, y, x + size, y + size, 8, fill=fill, outline=""
        )
        color = COLORS["blue"] if name == "pin" and self.topmost else COLORS["muted"]
        if name == "compact":
            self.canvas.create_oval(
                x + 9, y + 9, x + 21, y + 21, fill=color, outline=""
            )
        elif name == "corners":
            self.canvas.rounded_rectangle(
                x + 8,
                y + 8,
                x + 22,
                y + 22,
                5 if self.rounded_frame else 0,
                fill="",
                outline=color,
                width=1.5,
            )
        else:
            self._draw_text(
                x + size / 2,
                y + size / 2,
                glyph,
                fill=color,
                font=(self.family, 13, "bold"),
                anchor="center",
            )

    def _draw_progress(
        self,
        x1: float,
        y1: float,
        x2: float,
        value: float,
        color: str,
    ) -> None:
        height = 7
        self.canvas.rounded_rectangle(
            x1, y1, x2, y1 + height, 4, fill=COLORS["track"], outline=""
        )
        fill_width = (x2 - x1) * core._clamp_percent(value) / 100
        if fill_width >= 2:
            self.canvas.rounded_rectangle(
                x1,
                y1,
                x1 + fill_width,
                y1 + height,
                min(4, fill_width / 2),
                fill=color,
                outline="",
            )

    def _card_request_hint(self, index: int) -> str:
        requests = [
            item.text for item in self.announcements if index in item.gpu_indexes
        ]
        if not requests:
            return ""
        metric = (
            next((gpu for gpu in self.snapshot.gpus if gpu.index == index), None)
            if self.snapshot
            else None
        )
        warning = (
            f"橙框：{HIGHLIGHT_USERNAME} 正在使用已被申请的卡"
            if metric and metric.has_user(HIGHLIGHT_USERNAME)
            else "红框：已被申请"
        )
        return warning + "\n" + "\n".join(requests)

    def _memory_bar_at(self, x: float, y: float) -> int | None:
        if self.compact_mode:
            return None
        return next(
            (
                index
                for index, box in self.memory_bar_boxes.items()
                if self._inside(x, y, box)
            ),
            None,
        )

    def _verified_gpu_processes(self, index: int) -> tuple[remote.TrainingProcess, ...]:
        """Never translate a public/host PID by command similarity or arithmetic."""
        if (
            not self.ssh_features_available
            or not self.training_config
            or self.training_config.target.split("@", 1)[0].casefold()
            != HIGHLIGHT_USERNAME.casefold()
        ):
            return ()
        now = monotonic()
        processes: dict[int, remote.TrainingProcess] = {}
        conflicts: set[int] = set()
        for sample in self.training_samples:
            if (
                not sample.checked_at
                or sample.checked_at < self._process_session_started_at
                or not 0 <= now - sample.checked_at < remote.PROCESS_STALE_AFTER
            ):
                continue
            for process in sample.processes:
                indexes = (
                    (sample.log.gpu_index,)
                    if sample.log.gpu_index is not None
                    else process.gpu_indexes
                )
                if index not in indexes:
                    continue
                if process.pid in processes and processes[process.pid] != process:
                    conflicts.add(process.pid)
                processes[process.pid] = process
        return tuple(
            processes[pid] for pid in sorted(processes) if pid not in conflicts
        )

    def _memory_pid_hint(self, index: int) -> str:
        metric = (
            next((gpu for gpu in self.snapshot.gpus if gpu.index == index), None)
            if self.snapshot
            else None
        )
        usages = (
            tuple(
                usage
                for usage in metric.user_usages
                if usage.username.casefold() == HIGHLIGHT_USERNAME.casefold()
            )
            if metric
            else ()
        )
        if not usages:
            return ""
        lines = [f"GPU {index} · {HIGHLIGHT_USERNAME} 进程"]
        processes = self._verified_gpu_processes(index)
        if not self.online:
            lines.append("GPU 网页离线；仅显示仍在有效期内的 SSH 核实信息")
        if processes:
            lines.append(
                "来源：SSH /proc；使用 SSH 终端 PID，核实结果有效期 30 秒，操作前请核实"
            )
        for process in processes:
            lines.extend(
                (
                    f"PID：{process.pid}",
                    f"进程名：{process.name or '未提供'}",
                    "进程指令（SSH）：",
                    core.format_process_command(process.command),
                    "",
                )
            )
        if not processes:
            lines.append(
                "SSH 未连接，暂不显示 PID / 指令；请先在悬浮球登录"
                if not self.ssh_features_available
                else "尚未匹配到当前日志的有效 SSH 进程，或核实信息已过期；暂不显示 PID / 指令"
            )
            lines.append("网页 PID 不可靠，不作为可操作的进程号显示")
        return "\n".join(lines)

    def _has_memory_commands(self, index: int) -> bool:
        return bool(self._verified_gpu_processes(index))

    def event(self, event: Any) -> bool:
        if (
            event.type() == QEvent.ToolTip
            and getattr(self, "_memory_process_popup", None) is not None
            and self._memory_process_popup.isVisible()
        ):
            return True  # Do not overlay an unscrollable native tooltip.
        return super().event(event)

    def _defer_memory_hover_clear(self) -> None:
        self._memory_pointer_on_bar = False
        if (
            self._memory_process_popup is not None
            and self._memory_process_popup.isVisible()
        ):
            self._memory_hide_timer.start()
        else:
            self._clear_memory_hover()

    def _expire_memory_hover(self) -> None:
        popup = self._memory_process_popup
        if self._memory_pointer_on_bar or (
            popup is not None
            and popup.isVisible()
            and popup.geometry().contains(QCursor.pos())
        ):
            return
        if QApplication.activePopupWidget() is not None:
            self._memory_hide_timer.start()
            return
        self._clear_memory_hover()

    def _clear_memory_hover(self) -> None:
        self._memory_hide_timer.stop()
        self._memory_pointer_on_bar = False
        if self._memory_process_popup is not None:
            self._memory_process_popup.hide()
        if self._memory_hover_position is not None:
            QToolTip.hideText()
            self.setToolTip("")
        self._memory_hover_position = self._memory_hover_global_position = None

    def _refresh_memory_hover(self) -> None:
        position, global_position = (
            self._memory_hover_position,
            self._memory_hover_global_position,
        )
        if position is None or global_position is None:
            return
        index = self._memory_bar_at(position.x(), position.y())
        if (
            self.closed
            or self.compact_mode
            or self._dragging
            or self._resize_edges
            or index is None
        ):
            self._clear_memory_hover()
            return
        hint = "\n\n".join(
            filter(None, (self._memory_pid_hint(index), self._card_request_hint(index)))
        )
        if not hint:
            self._clear_memory_hover()
            return
        self.setToolTip(hint)
        if self._has_memory_commands(index):
            QToolTip.hideText()
            if self._memory_process_popup is None:
                self._memory_process_popup = ProcessCommandPopup(self)
            self._memory_process_popup.present(
                index,
                hint,
                global_position,
                reposition=self._memory_process_popup.current_gpu != index,
                stale=not self.online,
            )
        else:
            if self._memory_process_popup is not None:
                self._memory_process_popup.hide()
            QToolTip.setFont(QFont(self.family, 10))
            QToolTip.showText(global_position, hint, self)

    def _draw_memory_progress(
        self,
        x1: float,
        y1: float,
        x2: float,
        metric: GpuMetric,
    ) -> None:
        """Draw the web dashboard's per-user memory segments and labels."""
        height = 15
        if metric.has_user(HIGHLIGHT_USERNAME):
            self.memory_bar_boxes[metric.index] = (x1, y1, x2, y1 + height)
        self.canvas.rounded_rectangle(
            x1,
            y1,
            x2,
            y1 + height,
            5,
            fill=COLORS["track"],
            outline="",
        )
        if metric.memory_total_gib <= 0:
            return

        track_width = x2 - x1
        cursor = x1
        label_font = (self.family, 7, "bold")
        for usage in metric.user_usages:
            if usage.memory_gib <= 0 or cursor >= x2:
                continue
            proportional_width = (
                track_width * usage.memory_gib / metric.memory_total_gib
            )
            segment_width = max(track_width * 0.004, proportional_width)
            segment_end = min(x2, cursor + segment_width)
            color = core.user_color(usage.username)
            if cursor == x1:
                self.canvas.rounded_rectangle(
                    cursor,
                    y1,
                    segment_end,
                    y1 + height,
                    min(5, segment_width / 2),
                    fill=color,
                    outline="",
                )
                if segment_width > 5:
                    self.canvas.create_rectangle(
                        cursor + 5,
                        y1,
                        segment_end,
                        y1 + height,
                        fill=color,
                        outline="",
                    )
            else:
                self.canvas.create_rectangle(
                    cursor,
                    y1,
                    segment_end,
                    y1 + height,
                    fill=color,
                    outline="",
                )

            label_width, _ = self.canvas.measure_text(usage.username, label_font)
            if label_width + 5 <= segment_end - cursor:
                self._draw_text(
                    (cursor + segment_end) / 2,
                    y1 + height / 2,
                    usage.username,
                    fill="#071018",
                    font=label_font,
                    anchor="center",
                )
            cursor = segment_end

    def _draw_process_summary(
        self,
        metric: GpuMetric,
        x1: float,
        x2: float,
        y: float,
    ) -> None:
        if not metric.process_count:
            self._draw_text(
                x2,
                y,
                "空闲",
                fill=COLORS["green"],
                font=(self.family, 8),
                anchor="e",
            )
            return

        prefix = f"{metric.process_count} 进程"
        usage_parts = [
            f"{usage.username} {core._compact_memory(usage.memory_gib)}"
            for usage in metric.user_usages
        ]
        full_text = " · ".join((prefix, *usage_parts))
        max_width = x2 - x1
        for size in (8, 7):
            font = (self.family, size)
            if self.canvas.measure_text(full_text, font)[0] <= max_width:
                self._draw_text(
                    x2,
                    y,
                    full_text,
                    fill=COLORS["muted"],
                    font=font,
                    anchor="e",
                )
                return

        # Keep every user visible by wrapping the compact summary onto two rows.
        font = (self.family, 7)
        first_line = prefix
        second_line = ""
        for part in usage_parts:
            separator = " · " if first_line else ""
            candidate = f"{first_line}{separator}{part}"
            if (
                not second_line
                and self.canvas.measure_text(candidate, font)[0] <= max_width
            ):
                first_line = candidate
            else:
                second_line = f"{second_line} · {part}" if second_line else part

        if second_line and self.canvas.measure_text(second_line, font)[0] <= max_width:
            self._draw_text(
                x2,
                y - 7,
                first_line,
                fill=COLORS["muted"],
                font=font,
                anchor="e",
            )
            self._draw_text(
                x2,
                y + 7,
                second_line,
                fill=COLORS["muted"],
                font=font,
                anchor="e",
            )
            return

        # Extremely crowded cards remain bounded; the segmented bar still shows
        # every user's relative share even when the textual list cannot fit.
        abbreviated = full_text
        font = (self.family, 6)
        while (
            abbreviated
            and self.canvas.measure_text(f"{abbreviated}…", font)[0] > max_width
        ):
            abbreviated = abbreviated[:-1]
        self._draw_text(
            x2,
            y,
            f"{abbreviated}…",
            fill=COLORS["muted"],
            font=font,
            anchor="e",
        )

    def _draw_sparkline(
        self, index: int, x1: float, y1: float, x2: float, y2: float, color: str
    ) -> None:
        values = list(self.histories[index])
        if len(values) < 2:
            self.canvas.create_line(
                x1, (y1 + y2) / 2, x2, (y1 + y2) / 2, fill=COLORS["track"]
            )
            return
        step = (x2 - x1) / max(1, HISTORY_LENGTH - 1)
        start_x = x2 - step * (len(values) - 1)
        points: list[float] = []
        for offset, value in enumerate(values):
            points.extend(
                (
                    start_x + offset * step,
                    y2 - core._clamp_percent(value) / 100 * (y2 - y1),
                )
            )
        self.canvas.create_line(
            *points, fill=color, width=1.5, smooth=True, splinesteps=12
        )

    def _draw_card(
        self,
        index: int,
        metric: GpuMetric | None,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        self.card_boxes[index] = (x, y, x + width, y + height)
        hovered = self.hovered_card == index
        fill = COLORS["card_hover"] if hovered else COLORS["card"]
        highlighted = metric is not None and metric.has_user(HIGHLIGHT_USERNAME)
        reserved = index in self.reserved_gpu_indexes
        conflict = reserved and highlighted
        self.canvas.rounded_rectangle(
            x,
            y,
            x + width,
            y + height,
            13,
            fill=fill,
            outline=(
                COLORS["conflict_border"]
                if conflict
                else (
                    COLORS["red"]
                    if reserved
                    else COLORS["yzh_border"] if highlighted else COLORS["card_border"]
                )
            ),
            width=2.5 if highlighted or reserved else 1,
        )
        if conflict:
            self.canvas.rounded_rectangle(
                x + 4,
                y + 4,
                x + width - 4,
                y + height - 4,
                9,
                outline=COLORS["red"],
                width=1.5,
            )

        if metric is None:
            self._draw_text(
                x + 15, y + 18, f"GPU {index}", font=(self.family, 11, "bold")
            )
            self._draw_text(
                x + width - 15,
                y + 18,
                "等待数据",
                fill=COLORS["subtle"],
                font=(self.family, 9),
                anchor="e",
            )
            self._draw_progress(x + 15, y + 57, x + width - 15, 0, COLORS["blue"])
            self._draw_progress(x + 15, y + 98, x + width - 15, 0, COLORS["green"])
            return

        util_color = core._level_color(metric.utilization)
        memory_color = core._level_color(metric.memory_percent, memory=True)
        self.canvas.create_oval(
            x + 15, y + 14, x + 22, y + 21, fill=util_color, outline=""
        )
        self._draw_text(
            x + 29, y + 18, f"GPU {metric.index}", font=(self.family, 11, "bold")
        )
        hardware: list[str] = []
        if metric.temperature_c is not None:
            hardware.append(f"{metric.temperature_c:.0f}°C")
        if metric.power_w is not None:
            hardware.append(f"{metric.power_w:.0f}W")
        self._draw_text(
            x + width - 15,
            y + 18,
            "  ·  ".join(hardware),
            fill=COLORS["muted"],
            font=(self.family, 9),
            anchor="e",
        )

        self._draw_text(
            x + 15, y + 43, "GPU 利用率", fill=COLORS["muted"], font=(self.family, 9)
        )
        self._draw_text(
            x + width - 15,
            y + 43,
            f"{metric.utilization:.0f}%",
            fill=util_color,
            font=(self.family, 11, "bold"),
            anchor="e",
        )
        self._draw_progress(
            x + 15, y + 56, x + width - 15, metric.utilization, util_color
        )

        self._draw_text(
            x + 15, y + 82, "显存占用", fill=COLORS["muted"], font=(self.family, 9)
        )
        self._draw_text(
            x + width - 15,
            y + 82,
            f"{metric.memory_used_gib:.1f} / {metric.memory_total_gib:.0f} GiB  {metric.memory_percent:.0f}%",
            fill=memory_color,
            font=(self.family, 9, "bold"),
            anchor="e",
        )
        self._draw_memory_progress(x + 15, y + 94, x + width - 15, metric)
        summary_x1 = x + 15
        if not metric.process_count:
            self._draw_sparkline(
                metric.index,
                x + 15,
                y + height - 32,
                x + 85,
                y + height - 13,
                util_color,
            )
            summary_x1 = x + 95
        self._draw_process_summary(metric, summary_x1, x + width - 15, y + height - 22)

    def _wrap_text(
        self, text: str, width: float, font: Any, *, max_lines: int = 2
    ) -> list[str]:
        lines: list[str] = []
        current = ""
        for character in text.replace("\n", "  "):
            candidate = current + character
            if current and self.canvas.measure_text(candidate, font)[0] > width:
                lines.append(current.strip())
                current = character
            else:
                current = candidate
        if current:
            lines.append(current.strip())
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]
            while last and self.canvas.measure_text(last + "…", font)[0] > width:
                last = last[:-1]
            lines[-1] = last + "…"
        return lines

    def _draw_notices(self, width: int) -> None:
        self.notice_box = (18, 74, width - 18, 136)
        self.canvas.rounded_rectangle(
            *self.notice_box,
            10,
            fill="#111C2B",
            outline=COLORS["card_border"],
            width=1,
        )
        notices = self.banner_announcements
        current: core.Announcement | None = None
        kind = ""
        if notices:
            current = notices[self.notice_index % len(notices)]
            kind = " · GPU 使用申请" if current.kind == "gpu_usage" else " · 服务器公告"
        title = "公告" + kind + (" · 上次数据" if self.notice_error else "")
        self._draw_text(
            31, 86, title, fill=COLORS["amber"], font=(self.family, 8, "bold")
        )
        count = (
            f"{self.notice_index + 1}/{len(notices)} · 自动滚动 · 查看全部"
            if len(notices) > 1
            else "查看全部" if self.announcements else ""
        )
        self._draw_text(
            width - 63,
            86,
            count,
            fill=COLORS["muted"],
            font=(self.family, 8),
            anchor="e",
        )
        offset = 36 * self._notice_scroll_progress if self._previous_notice else 0.0
        if self._previous_notice:
            self._draw_notice_text(self._previous_notice, width, -offset)
        self._draw_notice_text(
            current or self._current_notice_text(),
            width,
            36 - offset if self._previous_notice else 0,
        )
        if len(notices) > 1:
            self._draw_text(
                width - 40,
                110,
                "›",
                fill=COLORS["amber"],
                font=(self.family, 20),
                anchor="center",
            )

    def _draw_notice_text(
        self, notice: core.Announcement | str, width: int, offset: float
    ) -> None:
        text = notice.text if isinstance(notice, core.Announcement) else notice
        request = isinstance(notice, core.Announcement) and notice.kind == "gpu_usage"
        clip = (30, 98, width - 66, 132)
        body_x = 31.0
        body_width = float(width - 110)
        if request:
            cards, deadline = _request_badges(notice)
            # Labels and body are one scrolling row. Preserve the full labels
            # and leave room for the body even at the minimum window width.
            badge_font = (self.family, 10, "bold")
            for size in (10, 9, 8):
                badge_font = (self.family, size, "bold")
                total = sum(
                    self.canvas.measure_text(label, badge_font)[0] + 14
                    for label in (cards, deadline)
                )
                if total + 16 + 120 <= width - 97:
                    break
            x = 31.0
            for label, color, background in (
                (cards, COLORS["red"], "#301C29"),
                (deadline, COLORS["amber"], "#302519"),
            ):
                badge_width = self.canvas.measure_text(label, badge_font)[0] + 14
                self.canvas.rounded_rectangle(
                    x,
                    NOTICE_ROW_Y - 9 + offset,
                    x + badge_width,
                    NOTICE_ROW_Y + 10 + offset,
                    5,
                    fill=background,
                    outline=color,
                    width=0.5,
                    clip=clip,
                )
                self._draw_text(
                    x + 7,
                    NOTICE_ROW_Y + offset,
                    label,
                    fill=color,
                    font=badge_font,
                    clip=clip,
                )
                x += badge_width + 8
            body_x = x
            body_width = max(0.0, width - 66 - body_x)
        if body_width <= 0:
            return
        body_font = (self.family, 9)
        # Equal notice text does not imply equal available space: different GPU
        # groups/deadlines change the label widths and must not share a wrap.
        key = (text, body_width, request)
        if key not in self._notice_line_cache:
            if len(self._notice_line_cache) >= 64:
                self._notice_line_cache.clear()
            self._notice_line_cache[key] = self._wrap_text(
                text,
                body_width,
                body_font,
                max_lines=1 if request else 2,
            )
        for index, line in enumerate(self._notice_line_cache[key]):
            self._draw_text(
                body_x,
                (NOTICE_ROW_Y if request else 106 + 16 * index) + offset,
                line,
                font=body_font,
                fill=COLORS["muted"] if request else COLORS["text"],
                clip=clip,
            )

    def _render(self) -> None:
        if self.closed:
            return
        width, height = self.width(), self.height()
        if not self.compact_mode and width < MINIMUM_WIDTH:
            # A stale native compact ConfigureNotify must never produce
            # negative card widths while the requested geometry is restored.
            return
        self.canvas.width, self.canvas.height = width, height
        self.canvas.view_scale = (
            self.ball_size / FLOATING_BALL_BASE_SIZE if self.compact_mode else 1.0
        )
        self.canvas.delete("all")
        self.card_boxes.clear()
        self.memory_bar_boxes.clear()
        self.button_boxes.clear()
        self.button_hints.clear()
        self.notice_box = (0, 0, 0, 0)
        if self.compact_mode:
            self._clear_memory_hover()
            self._draw_floating_ball()
            self._present_canvas()
            return
        radius = (
            18
            if self.rounded_frame and not self.args.windowed and not self.isMaximized()
            else 0
        )
        self.canvas.rounded_rectangle(
            0,
            0,
            width - 1,
            height - 1,
            radius,
            fill=COLORS["panel"],
            outline=COLORS["panel_border"],
            width=1,
        )
        self._draw_text(22, 25, self.args.server_name, font=(self.family, 20, "bold"))
        if self.snapshot:
            subtitle = (
                f"{len(self.snapshot.gpus)} × NVIDIA A100 80GB   ·   "
                f"平均 GPU {self.snapshot.average_utilization:.0f}%   ·   "
                f"平均显存 {self.snapshot.average_memory_percent:.0f}%"
            )
        else:
            subtitle = (
                f"8 × NVIDIA A100 80GB   ·   每 {self.args.interval:g} 秒自动刷新"
            )
        # Keep the existing grid/banner geometry. Training chips share the
        # subtitle row, shortening GPU metadata only at the narrowest width.
        if self.training_enabled and self.training_config:
            font = (self.family, 8)
            labels = []
            for index, sample in enumerate(self.training_samples):
                value = f"{sample.progress.percent:.1f}%" if sample.progress else "…"
                labels.append(
                    (
                        f"训{index + 1} {value}",
                        (
                            remote.TRAINING_COLORS[index]
                            if sample.is_fresh(self.training_config.stale_after)
                            else COLORS["subtle"]
                        ),
                    )
                )
            chips_width = sum(
                self.canvas.measure_text(label, font)[0] + 14 for label, _ in labels
            )
            if (
                self.canvas.measure_text(subtitle, (self.family, 9))[0]
                + chips_width
                + 60
                > width
            ):
                subtitle = (
                    f"8 × A100 · GPU {self.snapshot.average_utilization:.0f}% · 显存 {self.snapshot.average_memory_percent:.0f}%"
                    if self.snapshot
                    else "8 × A100 · 等待 GPU 数据"
                )
            chip_x = width - 22 - chips_width
            for label, color in labels:
                self._draw_text(chip_x, 52, label, fill=color, font=font)
                chip_x += self.canvas.measure_text(label, font)[0] + 14
        self._draw_text(22, 52, subtitle, fill=COLORS["muted"], font=(self.family, 9))
        status_color = (
            COLORS["green"]
            if self.online
            else COLORS["red"] if self.last_error else COLORS["amber"]
        )
        status_font = (self.family, 9)
        pill_width = self.canvas.measure_text(self.status_message, status_font)[0] + 40
        pill_x = width - 241 - pill_width
        self.canvas.rounded_rectangle(
            pill_x,
            18,
            pill_x + pill_width,
            46,
            14,
            fill=COLORS["card"],
            outline=COLORS["card_border"],
        )
        self.canvas.create_oval(
            pill_x + 12, 29, pill_x + 19, 36, fill=status_color, outline=""
        )
        self._draw_text(
            pill_x + 27, 32, self.status_message, fill=COLORS["muted"], font=status_font
        )
        self._draw_button("compact", width - 226, "", "收起为浮动球 · Ctrl+M")
        self._draw_button(
            "corners", width - 190, "", "切为直角" if self.rounded_frame else "切为圆角"
        )
        self._draw_button("open", width - 154, "↗", "打开完整监控网页")
        self._draw_button(
            "pin",
            width - 118,
            "◆" if self.topmost else "◇",
            "取消置顶" if self.topmost else "保持置顶",
        )
        self._draw_button("minimize", width - 82, "–", "最小化 · 从任务栏恢复")
        self._draw_button("close", width - 46, "×", "退出")
        self._draw_notices(width)

        metrics = (
            {gpu.index: gpu for gpu in self.snapshot.gpus} if self.snapshot else {}
        )
        indexes = sorted(set(range(core.EXPECTED_GPU_COUNT)) | set(metrics))
        columns = self._columns()
        rows = math.ceil(len(indexes) / columns)
        card_width = (width - 36 - (columns - 1) * CARD_GAP) / columns
        card_height = max(
            CARD_HEIGHT, (height - GRID_TOP - 36 - (rows - 1) * CARD_GAP) / rows
        )
        for offset, index in enumerate(indexes):
            row, column = divmod(offset, columns)
            self._draw_card(
                index,
                metrics.get(index),
                18 + column * (card_width + CARD_GAP),
                GRID_TOP + row * (card_height + CARD_GAP),
                card_width,
                card_height,
            )
        if self.snapshot:
            parts = []
            if self.snapshot.cpu_percent is not None:
                parts.append(f"CPU {self.snapshot.cpu_percent:.1f}%")
            if self.snapshot.host_memory_percent is not None:
                parts.append(f"内存 {self.snapshot.host_memory_percent:.1f}%")
            parts.append(
                f"空闲 GPU {self.snapshot.idle_gpu_count}/{len(self.snapshot.gpus)}"
            )
            left_footer = "   ·   ".join(parts)
            updated = datetime.fromtimestamp(self.snapshot.received_at).strftime(
                "%H:%M:%S"
            )
            right_footer = f"更新于 {updated}"
        else:
            left_footer = "拖动顶部移动 · 拖动边缘缩放"
            right_footer = "正在获取 GPU 数据"
        if width >= 850:
            right_footer += "   ·   F5 刷新   ·   右键菜单"
        self._draw_text(
            22, height - 18, left_footer, fill=COLORS["subtle"], font=(self.family, 8)
        )
        self._draw_text(
            width - 29,
            height - 18,
            right_footer,
            fill=COLORS["subtle"],
            font=(self.family, 8),
            anchor="e",
        )
        if self.last_error:
            self.setToolTip(self.last_error)
        for offset in (0, 4, 8):
            self.canvas.create_line(
                width - 13 - offset,
                height - 10,
                width - 10,
                height - 13 - offset,
                fill=COLORS["subtle"],
                width=1,
            )
        self._present_canvas()
        self._refresh_memory_hover()

    def _draw_training_ring(
        self, index: int, sample: remote.TrainingSample | None
    ) -> None:
        x1, y1, x2, y2, width = TRAINING_RING_LAYOUT[index]
        self.canvas.create_arc(
            x1, y1, x2, y2, start=-90, end=270, fill=COLORS["track"], width=width
        )
        if not sample or not sample.progress or sample.progress.percent <= 0:
            return
        fresh = sample.is_fresh(self.training_config.stale_after)
        color = remote.TRAINING_COLORS[index] if fresh else COLORS["subtle"]
        percent = core._clamp_percent(sample.progress.percent)
        end = -90 + 3.6 * percent
        self.canvas.create_arc(
            x1, y1, x2, y2, start=-90, end=end, fill=color, width=width
        )
        if percent >= 100:
            return
        # ImageDraw arcs have square ends and draw their stroke inwards. Caps
        # are centered on the actual stroke, not on the bounding ellipse edge.
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        radius = (x2 - x1 - width) / 2
        for angle in (-90, end):
            radians = math.radians(angle)
            tip_x = center_x + radius * math.cos(radians)
            tip_y = center_y + radius * math.sin(radians)
            self.canvas.create_oval(
                tip_x - width / 2,
                tip_y - width / 2,
                tip_x + width / 2,
                tip_y + width / 2,
                fill=color,
            )
        # A tiny light at the real endpoint makes small percentages readable;
        # never draw one for unknown/zero progress or for disconnected data.
        if fresh:
            self.canvas.create_oval(
                tip_x - 1.3,
                tip_y - 1.3,
                tip_x + 1.3,
                tip_y + 1.3,
                fill="#E8F4FF" if index == 0 else "#F5ECFF",
            )

    @property
    def displayed_training_index(self) -> int:
        return (
            self._hover_training_index
            if self._hover_training_index is not None
            else self._ball_training_index
        )

    def _ball_status_pages(self) -> tuple[int, ...]:
        if self.ssh_login_required and not self.ssh_features_available:
            public_page = (
                BALL_AVAILABILITY_PAGE
                if self.ball_hide_training or self.show_ball_availability
                else BALL_SUMMARY_PAGE
            )
            return (public_page, BALL_SSH_LOGIN_PAGE)
        if self.ball_hide_training:
            pages = (BALL_AVAILABILITY_PAGE,)
            if self.no_own_gpu_processes and self.disk_enabled:
                pages += (BALL_DISK_PAGE,)
            return pages
        pages = self._ball_training_pages()
        if self.show_ball_availability:
            pages += (BALL_AVAILABILITY_PAGE,)
        if not pages:
            pages = (BALL_SUMMARY_PAGE,)
        return pages + (BALL_DISK_PAGE,) if self.disk_enabled else pages

    def _ball_training_pages(self) -> tuple[int, ...]:
        """No placeholder job pages, including when hovering an empty ring."""
        if (
            not self.training_enabled
            or not self.training_config
            or not self.ssh_features_available
            or self.ball_hide_training
        ):
            return ()
        pages = tuple(
            index
            for index, sample in enumerate(self.training_samples[:2])
            if sample.progress is not None
        )
        if self.show_ball_availability and len(self.own_gpu_indexes) == 1:
            # Never label another card's old log as the sole current task.
            pages = tuple(
                index
                for index in pages
                if self.own_gpu_indexes
                & frozenset(self.training_samples[index].effective_gpu_indexes)
            )
        return pages

    def _reconcile_ball_status_pages(self) -> tuple[int, ...]:
        pages = self._ball_status_pages()
        if pages != self._ball_page_set:
            first_task_arrived = not any(
                page in (0, 1) for page in self._ball_page_set
            ) and any(page in (0, 1) for page in pages)
            self._stop_ball_training_transition()
            self._ball_page_set = pages
            if (
                self._hover_training_index is not None
                and self._hover_training_index not in pages
            ):
                self._hover_training_index = None
            if first_task_arrived:
                self._ball_training_index = next(
                    page for page in pages if page in (0, 1)
                )
            elif self._ball_training_index not in pages:
                self._ball_training_index = pages[0]
        if (
            self._hover_training_index is None
            and self._ball_training_index not in pages
        ):
            self._ball_training_index = pages[0]
        return pages

    def _stop_ball_training_transition(self) -> None:
        if hasattr(self, "training_animation"):
            self.training_animation.stop()
        self._ball_previous_page = None
        self._ball_transition_progress = 1.0

    def _animate_ball_training(self, progress: Any) -> None:
        self._ball_transition_progress = float(progress)
        if self._ready and not self.closed and self.compact_mode:
            self._render()

    def _finish_ball_training_transition(self) -> None:
        self._ball_previous_page = None
        self._animate_ball_training(1.0)

    def _training_ring_at(self, position: QPoint) -> int | None:
        if (
            not self.compact_mode
            or not self.training_enabled
            or self.ball_hide_training
            or not self.ssh_features_available
        ):
            return None
        scale = self.ball_size / FLOATING_BALL_BASE_SIZE
        center = (FLOATING_BALL_BASE_SIZE - 1) / 2
        radius = math.hypot(
            position.x() / scale - center, position.y() / scale - center
        )
        matches = []
        visible = self._ball_training_pages()
        for index, (x1, _y1, x2, _y2, width) in enumerate(TRAINING_RING_LAYOUT):
            if index not in visible:
                continue
            stroke_radius = (x2 - x1 - width) / 2
            distance = abs(radius - stroke_radius)
            if distance <= width / 2 + 0.75:
                matches.append((distance, index))
        return min(matches)[1] if matches else None

    def _sync_training_timer(self) -> None:
        if not hasattr(self, "training_timer"):
            return
        self._reconcile_ball_status_pages()
        running = (
            not self.closed
            and self.compact_mode
            and len(self._ball_page_set) > 1
            and (
                (self.training_enabled and self.training_config is not None)
                or self.disk_enabled
                or BALL_SSH_LOGIN_PAGE in self._ball_page_set
            )
            and self.isVisible()
            and not self.isMinimized()
            and self._hover_training_index is None
            and not self._dragging
        )
        if running:
            if not self.training_timer.isActive():
                self.training_timer.start()
        else:
            self.training_timer.stop()
            self._stop_ball_training_transition()

    def _set_training_hover(self, index: int | None) -> None:
        changed = index != self._hover_training_index
        self._hover_training_index = index
        if index is not None:
            self._stop_ball_training_transition()
            self._ball_training_index = index
        self._sync_training_timer()
        if changed:
            self._render()

    def _next_ball_training(self) -> None:
        self._sync_training_timer()
        if not self.training_timer.isActive():
            return
        if self._ball_previous_page is not None:
            return
        pages = self._ball_page_set
        self._ball_previous_page = self._ball_training_index
        self._ball_training_index = pages[
            (pages.index(self._ball_training_index) + 1) % len(pages)
        ]
        self._ball_transition_progress = 0.0
        self.training_animation.start()
        self._render()

    def _ball_status_rows(self, page: int) -> tuple[tuple[str, ...], str]:
        if page == BALL_SSH_LOGIN_PAGE:
            if not self.ssh_config:
                return ("SSH", "配置错误", "检查配置"), COLORS["red"]
            if self.ssh_login_pending:
                return ("登录中", "请在终端", "完成认证"), COLORS["amber"]
            if self.ssh_connected is None:
                return ("SSH", "检查连接", "点击登录"), COLORS["amber"]
            return (
                (
                    "登录",
                    "SSH 未连接",
                    "点击重试" if self.ssh_login_error else "点击连接",
                ),
                COLORS["amber"],
            )
        if page == BALL_DISK_PAGE:
            rows, level = remote.disk_rows(
                self.disk_sample,
                (
                    self.disk_config.stale_after
                    if self.disk_config
                    else 2 * remote.DEFAULT_INTERVAL + remote.DEFAULT_TIMEOUT
                ),
            )
            return rows, (
                COLORS["red"]
                if level == "over"
                else COLORS["cyan"] if level == "normal" else COLORS["subtle"]
            )
        if page == BALL_SUMMARY_PAGE:
            if self.show_ball_availability:
                return self._ball_status_rows(BALL_AVAILABILITY_PAGE)
            if self.online and self.snapshot:
                return (
                    (
                        f"{len(self.own_gpu_indexes)}张",
                        "yzh 已用",
                        _compact_gpu_indexes(self.own_gpu_indexes),
                    ),
                    COLORS["yzh_border"],
                )
            return (
                (
                    "…",
                    "GPU 监控",
                    "离线" if self.last_error or self.snapshot else "连接中",
                ),
                COLORS["muted"],
            )
        if page == BALL_AVAILABILITY_PAGE:
            if not self.online or self.snapshot is None:
                return (
                    (
                        "…",
                        "GPU 监控",
                        "离线" if self.last_error or self.snapshot else "连接中",
                    ),
                    COLORS["muted"],
                )
            known = self.notice_loaded and not self.notice_error
            indexes = (
                self.available_gpu_indexes
                if known
                else frozenset(
                    gpu.index
                    for gpu in self.snapshot.gpus
                    if gpu.utilization < 10
                    and gpu.memory_percent < 10
                    and not gpu.has_user(HIGHLIGHT_USERNAME)
                )
            )
            return (
                (
                    f"{len(indexes)}张",
                    "可用 GPU" if known else "空闲 GPU",
                    _compact_gpu_indexes(indexes),
                ),
                COLORS["cyan"] if known else COLORS["muted"],
            )
        sample = (
            self.training_samples[page]
            if self.training_config and page < len(self.training_samples)
            else None
        )
        log = (
            sample.log
            if sample
            else self.training_config.logs[page] if self.training_config else None
        )
        fresh = sample is not None and sample.is_fresh(self.training_config.stale_after)
        return (
            (
                _ball_training_percent(sample.progress if sample else None),
                sample.gpu_label if sample else log.gpu_label if log else "GPU ?",
                remote.format_duration(
                    sample.progress.eta_seconds if sample and sample.progress else None
                ),
            ),
            remote.TRAINING_COLORS[page] if fresh else COLORS["subtle"],
        )

    def _draw_ball_status(self) -> int:
        page = self._visible_ball_status_page()
        opacity, offset = 1.0, 0.0
        if self._ball_previous_page is not None:
            progress = self._ball_transition_progress
            if progress < 0.5:
                page = self._ball_previous_page
                opacity = 1 - 2 * progress
                offset = -4 * (1 - opacity)
            else:
                opacity = 2 * progress - 1
                offset = 4 * (1 - opacity)
        texts, color = self._ball_status_rows(page)
        center = (FLOATING_BALL_BASE_SIZE - 1) / 2
        if page == BALL_DISK_PAGE:
            layout = DISK_TEXT_LAYOUT
            sizes, minimums, widths, bolds = (15, 11), (8, 8), (60, 50), (True, False)
        elif page in (
            BALL_SSH_LOGIN_PAGE,
            BALL_AVAILABILITY_PAGE,
            BALL_SUMMARY_PAGE,
        ):
            # CJK headings have taller glyphs than numerical percentages.
            # Separate rows and a smaller heading avoid ink overlap even at
            # the smallest ball size, without changing training's layout.
            layout = BALL_MESSAGE_TEXT_LAYOUT
            sizes, minimums, widths, bolds = (
                (14, 8, 8),
                (10, 5, 5),
                (56, 56, 50),
                (True, True, False),
            )
        else:
            layout = TRAINING_TEXT_LAYOUT
            sizes, minimums, widths, bolds = (
                (18, 8, 8),
                (12, 5, 5),
                (56, 56, 50),
                (True, True, False),
            )
        for y, text, base_size, minimum_size, max_width, bold in zip(
            layout,
            texts,
            sizes,
            minimums,
            widths,
            bolds,
            strict=True,
        ):
            size = base_size
            font = (self.family, size, "bold" if bold else "normal")
            while (
                size > minimum_size
                and self.canvas.measure_text(text, font)[0] > max_width
            ):
                size -= 1
                font = (self.family, size, "bold" if bold else "normal")
            self._draw_text(
                center,
                y + offset,
                text,
                font=font,
                fill=color,
                anchor="center",
                opacity=opacity,
                clip=(20, 25, 83, 80),
            )
        return page

    def _draw_floating_ball(self) -> None:
        center = (FLOATING_BALL_BASE_SIZE - 1) / 2
        connected = self.online and self.snapshot is not None
        requests_known = self.notice_loaded and not self.notice_error
        disk_alert = (
            self.disk_enabled
            and self.ssh_features_available
            and self.disk_sample.over_quota
        )
        if disk_alert:
            color = COLORS["red"]
        elif self.last_error or (self.snapshot and not connected):
            color = COLORS["red"]
        elif not connected:
            color = COLORS["amber"]
        elif self.own_gpu_indexes:
            color = COLORS["yzh_border"]
        else:
            color = COLORS["cyan"]
        self.canvas.create_oval(
            1,
            1,
            FLOATING_BALL_BASE_SIZE - 2,
            FLOATING_BALL_BASE_SIZE - 2,
            fill=COLORS["panel"],
            outline=color,
            width=2 if disk_alert else 1.5,
        )
        if self.training_enabled or self.disk_enabled or self.ball_hide_training:
            self._reconcile_ball_status_pages()
            samples = tuple(
                (
                    self.training_samples[index]
                    if self.training_config and index < len(self.training_samples)
                    else None
                )
                for index in range(2)
            )
            if (
                self.training_enabled
                and self.ssh_features_available
                and not self.ball_hide_training
            ):
                for index in self._ball_training_pages():
                    self._draw_training_ring(index, samples[index])
            self._draw_ball_status()
            # Never squeeze a fourth connection-status line below the ETA or
            # duplicate the offline message. The border/tooltip retain public
            # API status; the public status page already has its own wording.
        else:
            # Turning log monitoring off must not bring GPU utilization back.
            if self.show_ball_availability:
                value = f"{len(self.available_gpu_indexes) if requests_known else self.snapshot.idle_gpu_count}/{len(self.snapshot.gpus)}"
                bottom = "可用卡" if requests_known else "空闲卡"
            elif connected:
                value = f"{len(self.own_gpu_indexes)}张"
                bottom = "已用卡"
            else:
                value = "离线" if self.last_error or self.snapshot else "…"
                bottom = "数据过期" if self.snapshot else "连接中"
            self._draw_text(
                center,
                28,
                "A100",
                font=(self.family, 8, "bold"),
                fill=COLORS["muted"],
                anchor="center",
            )
            self._draw_text(
                center,
                50,
                value,
                font=(self.family, 21, "bold"),
                fill=COLORS["text"] if connected else color,
                anchor="center",
            )
            self._draw_text(
                center,
                75,
                bottom,
                font=(self.family, 8),
                fill=COLORS["muted"],
                anchor="center",
            )
        self.setToolTip(self._compact_tooltip())

    def _present_canvas(self) -> None:
        width, height = self.canvas.width, self.canvas.height
        final = self.canvas.present()
        raw = final.tobytes("raw", "RGBA")
        self._frame = QImage(
            raw, width, height, width * 4, QImage.Format_RGBA8888
        ).copy()
        self._apply_mask()
        self.update()


def _configure_qt_plugin_paths() -> None:
    # QLibraryInfo can lose non-ASCII characters in the wheel's install path.
    # Resolve plugins beside the loaded binding, keeping system Qt's discovery
    # unchanged when that installation does not include a bundled Qt5 tree.
    plugins = Path(QtCore.__file__).resolve().parent / "Qt5" / "plugins"
    platforms = plugins / "platforms"
    if platforms.is_dir():
        os.environ["QT_PLUGIN_PATH"] = str(plugins)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)


def run_desktop(args: argparse.Namespace) -> int:
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print(
            "当前没有图形桌面；请在桌面终端运行，或使用 --once 查看数据。",
            file=sys.stderr,
        )
        return 2
    _configure_qt_plugin_paths()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication([sys.argv[0]])
    app.setApplicationName("WN-A100 Monitor")
    app.setDesktopFileName("wn-a100-monitor")
    regular, bold = core._font_files()
    for path, _index in (regular, bold):
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    font = QFont("Noto Sans CJK SC", 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)
    app.setStyleSheet(
        "QToolTip {background:#111A29; color:#F5F7FB; border:1px solid #22324A; padding:7px;}"
    )
    window = MonitorWindow(args)
    signal.signal(signal.SIGINT, lambda _signum, _frame: window.close())
    window.show()
    return app.exec_()
