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

"""Qt presentation for the local merge helper; no dataset or ML imports at startup."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import (
    QCloseEvent,
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .merge_datasets_gui import (
        PROJECT_ROOT,
        REPO_ID_PATTERN,
        build_merge_command,
        default_repo_id,
        discover_datasets,
        is_dataset_dir,
        lerobot_home,
        natural_key,
        subprocess_environment,
    )
else:
    from merge_datasets_gui import (
        PROJECT_ROOT,
        REPO_ID_PATTERN,
        build_merge_command,
        default_repo_id,
        discover_datasets,
        is_dataset_dir,
        lerobot_home,
        natural_key,
        subprocess_environment,
    )

# Keep the same palette, spacing and controls as utils/github_upload_window.py.
COLORS = {
    "text": "#F4F7FC",
    "muted": "#C1CDDF",
    "subtle": "#A2B2CA",
    "blue": "#66ABFF",
    "green": "#6DE0AC",
    "amber": "#FFC779",
    "red": "#FF8297",
}
STYLE = """
QMainWindow { background: #080D16; }
QWidget { color: #F4F7FC; font-size: 14px; }
QLabel { background: transparent; }
QFrame[card="true"] { background: #0E1624; border: 1px solid #2B3C56; border-radius: 12px; }
QFrame[inset="true"] { background: #152134; border-radius: 8px; }
QLineEdit, QPlainTextEdit { background: #152134; color: #F4F7FC; border: 1px solid #2B3C56;
    border-radius: 7px; padding: 9px; selection-background-color: #316CAF; }
QLineEdit:focus { border-color: #66ABFF; }
QLineEdit { min-height: 22px; }
QLineEdit:disabled { color: #A2B2CA; }
QPushButton { background: #152134; border: 1px solid #344A69; border-radius: 7px;
    padding: 8px 13px; min-height: 19px; }
QPushButton:hover { background: #213653; border-color: #66ABFF; }
QPushButton:disabled { color: #8292A9; border-color: #26364C; }
QPushButton[primary="true"] { background: #66ABFF; color: #091320; border: none; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #8FC1FF; }
QPushButton[primary="true"]:disabled { background: #263B56; color: #A2B2CA; }
QPushButton[danger="true"] { color: #FF8297; }
QPushButton[danger="true"]:disabled { color: #8292A9; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QTreeWidget { background: #0E1624; alternate-background-color: #111D2E; color: #F4F7FC;
    border: none; outline: none; }
QTreeWidget::item { padding: 10px 3px; border-bottom: 1px solid #1D2B40; }
QTreeWidget::item:selected { background: #203F60; }
QTreeWidget::item:hover { background: #192E46; }
QTreeView::indicator { width: 17px; height: 17px; }
QHeaderView::section { background: #17263B; color: #C1CDDF; border: none; padding: 9px 5px; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: #152134; color: #C1CDDF; padding: 7px 16px; margin-right: 6px;
    margin-bottom: 6px; border-radius: 6px; }
QTabBar::tab:selected { background: #2B425F; color: #FFFFFF; }
QProgressBar { background: #20334D; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #66ABFF; border-radius: 3px; }
QSplitter::handle { background: #080D16; height: 10px; }
QScrollBar:vertical { width: 9px; background: #0E1624; }
QScrollBar::handle:vertical { background: #3A506F; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QToolTip { color: #F4F7FC; background: #20334D; border: 1px solid #66ABFF; padding: 6px; }
"""
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class DatasetEntry:
    """Read only the small info.json file for table and selection summaries."""

    path: Path
    episodes: int | None = None
    frames: int | None = None
    fps: float | None = None
    error: str = ""

    @classmethod
    def read(cls, path: Path) -> DatasetEntry:
        """Keep malformed metadata visible with an actionable error."""
        try:
            info = json.loads((path / "meta/info.json").read_text(encoding="utf-8"))
            return cls(path, int(info["total_episodes"]), int(info["total_frames"]), float(info["fps"]))
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            return cls(path, error=f"无法读取元信息：{path}\n{exc}")


class MergeWorker(QThread):
    """Stream the existing CLI in a new process group without blocking the window."""

    output = pyqtSignal(str)
    result = pyqtSignal(int, str)

    def __init__(self, command: list[str], parent: QWidget) -> None:
        """Capture an immutable command before the worker starts."""
        super().__init__(parent)
        self.command = command
        self.stop_requested = threading.Event()
        self.process: subprocess.Popen[str] | None = None

    def run(self) -> None:
        """Report success only once the CLI has exited successfully."""
        try:
            with subprocess.Popen(
                self.command,
                cwd=PROJECT_ROOT,
                env=subprocess_environment(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            ) as process:
                self.process = process
                if self.stop_requested.is_set():
                    self._signal(signal.SIGTERM)
                assert process.stdout is not None
                for line in process.stdout:
                    self.output.emit(line)
                code = process.wait()
            self.result.emit(code, "")
        except OSError as exc:
            self.result.emit(-1, str(exc))
        finally:
            self.process = None

    def _signal(self, value: int) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, value)

    def request_stop(self) -> None:
        """Handle cancellation even when the child has not been created yet."""
        self.stop_requested.set()
        self._signal(signal.SIGTERM)

    def force_stop(self) -> None:
        """Bound shutdown for children that ignore SIGTERM."""
        if self.stop_requested.is_set():
            self._signal(signal.SIGKILL)


def choose_font() -> QFont:
    """Prefer a scalable CJK font; Qt applies monitor DPI and antialiasing."""
    available = set(QFontDatabase().families())
    family = next(
        (
            name
            for name in ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei")
            if name in available
        ),
        "Sans Serif",
    )
    font = QFont(family)
    font.setPixelSize(14)
    font.setStyleStrategy(QFont.PreferAntialias)
    return font


def merge_icon() -> QIcon:
    """Draw a crisp branch/merge mark without image or font dependencies."""
    pixmap = QPixmap(52, 52)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#172B45"))
    painter.drawRoundedRect(0, 0, 52, 52, 12, 12)
    painter.setPen(QPen(QColor(COLORS["blue"]), 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawLine(16, 14, 16, 23)
    painter.drawLine(36, 14, 36, 23)
    painter.drawLine(16, 23, 26, 32)
    painter.drawLine(36, 23, 26, 32)
    painter.drawLine(26, 32, 26, 39)
    painter.setBrush(QColor("#65DDEB"))
    painter.setPen(Qt.NoPen)
    for x, y in ((16, 14), (36, 14), (26, 39)):
        painter.drawEllipse(x - 4, y - 4, 8, 8)
    painter.end()
    return QIcon(pixmap)


class MergeWindow(QMainWindow):
    """Keep source selection, output preview and progress in one desktop window."""

    def __init__(self, *, home: Path | None = None, source: Path | None = None, demo: bool = False) -> None:
        """Build the window and scan only lightweight local dataset metadata."""
        super().__init__()
        self.home = (home or lerobot_home()).expanduser().resolve()
        self.demo = demo
        self.busy = False
        self.repo_id_auto = True
        self.worker: MergeWorker | None = None
        self.output_dir: Path | None = None
        self._manual_paths: set[Path] = set()
        self._controls: list[QWidget] = []
        self._started_at = 0.0
        self.setWindowTitle("Dataset Merge · LeRobot 数据集合并")
        self.resize(1280, 940)
        self.setMinimumSize(1080, 800)
        self.setFont(choose_font())
        family = self.font().family().replace('"', '\\"')
        self.setStyleSheet(f'QWidget {{ font-family: "{family}"; }}\n' + STYLE)
        self.setWindowIcon(merge_icon())
        self._build(source or self.home)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_stop)
        self._populate([], set())
        if demo:
            self._load_demo()
        else:
            self.refresh()

    def _label(self, text: str, *, size: int = 14, color: str = "text", bold: bool = False) -> QLabel:
        label = QLabel(text)
        label.setTextFormat(Qt.PlainText)
        label.setStyleSheet(
            f"color: {COLORS[color]}; font-size: {size}px; font-weight: {600 if bold else 400};"
        )
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return label

    def _button(
        self, text: str, callback: Callable, *, primary: bool = False, lock: bool = True
    ) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("primary", primary)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda _checked=False: callback())
        if lock:
            self._controls.append(button)
        return button

    def _card(self, title: str, subtitle: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setProperty("card", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(self._label(title, size=17, bold=True))
        if subtitle:
            hint = self._label(subtitle, color="subtle")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        return card, layout

    def _build(self, source: Path) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        page = QVBoxLayout(central)
        page.setContentsMargins(24, 22, 24, 20)
        page.setSpacing(14)
        header = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(self.windowIcon().pixmap(52, 52))
        header.addWidget(logo)
        header.addSpacing(4)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(self._label("Dataset Merge", size=27, bold=True))
        titles.addWidget(self._label("LeRobot 数据集合并  /  将分批采集整理为一个数据集", color="muted"))
        header.addLayout(titles)
        header.addStretch()
        self.details_button = self._button("展开日志", self._toggle_details, lock=False)
        header.addWidget(self.details_button)
        header.addSpacing(12)
        self.state_badge = self._label("●  本地合并", color="green")
        header.addWidget(self.state_badge)
        page.addLayout(header)
        steps = self._label(
            "01  选择数据集       →       02  设置输出       →       03  合并与检查", color="muted"
        )
        steps.setStyleSheet(
            "background: #101C2C; border: 1px solid #2B3C56; border-radius: 8px;"
            "padding: 11px 16px; color: #C1CDDF;"
        )
        page.addWidget(steps)
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        body = QWidget()
        columns = QHBoxLayout(body)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(16)
        sources, layout = self._card("01  选择源数据", "勾选要合并的数据集；合并顺序与下方列表一致")
        self.source_scroll = QScrollArea()
        self.source_scroll.setWidgetResizable(True)
        self.source_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        source_row = QHBoxLayout()
        self.source_input = QLineEdit(str(source.expanduser().resolve()))
        self.source_input.setPlaceholderText("选择数据集所在的父目录")
        self.source_input.setToolTip("扫描当前目录及下一级命名空间目录")
        self.source_input.returnPressed.connect(self.refresh)
        self._controls.append(self.source_input)
        source_row.addWidget(self.source_input, 1)
        source_row.addWidget(self._button("浏览…", self.browse_source))
        self.refresh_button = self._button("刷新", self.refresh)
        self.refresh_button.setShortcut("Ctrl+R")
        source_row.addWidget(self.refresh_button)
        layout.addLayout(source_row)
        tools = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索数据集名称或路径…")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._filter)
        self._controls.append(self.search_input)
        tools.addWidget(self.search_input, 1)
        tools.addWidget(self._button("＋ 添加目录", self.add_dataset))
        layout.addLayout(tools)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["选择", "数据集", "Episodes", "帧数", "FPS"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.setMinimumHeight(140)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        for column, width in ((0, 66), (2, 85), (3, 88), (4, 60)):
            self.tree.header().setSectionResizeMode(column, QHeaderView.Fixed)
            self.tree.setColumnWidth(column, width)
        for column in (2, 3, 4):
            self.tree.headerItem().setTextAlignment(column, Qt.AlignRight | Qt.AlignVCenter)
        self.tree.itemChanged.connect(self._selection_changed)
        self._controls.append(self.tree)
        layout.addWidget(self.tree, 1)
        self.empty_label = self._label("尚未发现数据集\n选择源目录，或点击「添加目录」导入。", color="subtle")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.empty_label.setMinimumHeight(70)
        layout.addWidget(self.empty_label, 1)
        actions = QHBoxLayout()
        actions.setSpacing(7)
        actions.addWidget(self._button("全选", lambda: self._check_visible(True)))
        actions.addWidget(self._button("清空勾选", lambda: self._check_visible(False)))
        self.up_button = self._button("↑", lambda: self.move_selected(-1))
        self.down_button = self._button("↓", lambda: self.move_selected(1))
        self.up_button.setToolTip("选中行上移；按 Ctrl / Shift 可选择多行")
        self.down_button.setToolTip("选中行下移；按 Ctrl / Shift 可选择多行")
        actions.addWidget(self.up_button)
        actions.addWidget(self.down_button)
        actions.addStretch()
        self.dataset_count = self._label("0 个数据集", color="subtle")
        actions.addWidget(self.dataset_count)
        layout.addLayout(actions)
        metrics = QHBoxLayout()
        self.metrics: dict[str, QLabel] = {}
        for key, title in (("datasets", "已选数据集"), ("episodes", "Episodes 合计"), ("frames", "帧数合计")):
            panel = QFrame()
            panel.setProperty("inset", True)
            metric = QVBoxLayout(panel)
            metric.setContentsMargins(13, 10, 13, 10)
            self.metrics[key] = self._label("0", size=23, color="blue", bold=True)
            metric.addWidget(self.metrics[key])
            metric.addWidget(self._label(title, size=12, color="subtle"))
            metrics.addWidget(panel, 1)
        layout.addLayout(metrics)
        self.source_scroll.setWidget(sources)
        columns.addWidget(self.source_scroll, 3)
        self.output_scroll = QScrollArea()
        self.output_scroll.setWidgetResizable(True)
        self.output_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        target, layout = self._card("02  输出设置", "为合并后的数据集设置名称和保存位置")
        target.setMinimumWidth(332)
        layout.addWidget(self._label("输出 Repo ID", color="muted"))
        self.repo_input = QLineEdit()
        self.repo_input.setPlaceholderText("namespace/dataset")
        self.repo_input.textEdited.connect(self._repo_edited)
        self.repo_input.textChanged.connect(self._update_preview)
        self._controls.append(self.repo_input)
        layout.addWidget(self.repo_input)
        naming = QHBoxLayout()
        naming.addWidget(self._label("根据所选数据集自动命名", size=12, color="subtle"), 1)
        self.auto_button = self._button("恢复自动", self._reset_name)
        naming.addWidget(self.auto_button)
        layout.addLayout(naming)
        layout.addWidget(self._label("保存到", color="muted"))
        self.output_preview = QPlainTextEdit()
        self.output_preview.setReadOnly(True)
        self.output_preview.setFixedHeight(82)
        layout.addWidget(self.output_preview)
        self.output_hint = self._label("", size=12, color="subtle")
        self.output_hint.setWordWrap(True)
        layout.addWidget(self.output_hint)
        layout.addWidget(self._label("合并说明", color="muted", bold=True))
        for text in (
            "原始数据集保留，结果写入新目录",
            "按左侧列表顺序拼接 episodes",
            "缺失的 intervention 补为 False",
            "已有干预标记保留，仅在本地保存",
        ):
            label = self._label("·  " + text, size=13, color="muted")
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch()
        self.open_button = self._button("打开输出目录 ↗", self.open_output)
        self.open_button.setEnabled(False)
        layout.addWidget(self.open_button)
        self.output_scroll.setWidget(target)
        columns.addWidget(self.output_scroll, 2)
        splitter.addWidget(body)
        logs, layout = self._card("03  运行详情", "")
        tabs = QTabWidget()
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("准备就绪。选择至少两个数据集后，点击「开始合并」。")
        self.log_box.setMaximumBlockCount(4000)
        self.log_box.setMinimumHeight(80)
        self.command_box = QPlainTextEdit()
        self.command_box.setReadOnly(True)
        self.command_box.setPlaceholderText("选择数据集并填写有效的 Repo ID 后，这里会显示实际执行的命令。")
        tabs.addTab(self.log_box, "运行日志")
        tabs.addTab(self.command_box, "执行命令")
        layout.addWidget(tabs)
        logs.setMinimumHeight(200)
        splitter.addWidget(logs)
        self.detail_panel = logs
        self.detail_panel.hide()
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 180])
        page.addWidget(splitter, 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        page.addWidget(self.progress)
        footer = QHBoxLayout()
        statuses = QVBoxLayout()
        statuses.setSpacing(3)
        self.status_label = self._label("选择至少两个数据集", bold=True)
        self.detail_label = self._label("输出名称会随选择自动更新。", size=12, color="subtle")
        self.detail_label.setWordWrap(True)
        statuses.addWidget(self.status_label)
        statuses.addWidget(self.detail_label)
        footer.addLayout(statuses, 1)
        self.elapsed_label = self._label("", color="subtle")
        footer.addWidget(self.elapsed_label)
        self.stop_button = self._button("停止合并", self.stop_merge)
        self.stop_button.setProperty("danger", True)
        self.stop_button.setEnabled(False)
        footer.addWidget(self.stop_button)
        self.merge_button = self._button("开始合并  →", self.start_merge, primary=True)
        self.merge_button.setMinimumWidth(156)
        self.merge_button.setEnabled(False)
        footer.addWidget(self.merge_button)
        page.addLayout(footer)

    def _toggle_details(self) -> None:
        self._show_details(self.detail_panel.isHidden())

    def _show_details(self, visible: bool) -> None:
        self.detail_panel.setVisible(visible)
        self.details_button.setText("收起日志" if visible else "展开日志")

    def _items(self) -> list[QTreeWidgetItem]:
        return [self.tree.topLevelItem(index) for index in range(self.tree.topLevelItemCount())]

    def selected_paths(self) -> list[Path]:
        """Include checked rows hidden by search, always preserving displayed order."""
        return [item.data(1, Qt.UserRole).path for item in self._items() if item.checkState(0) == Qt.Checked]

    def _populate(self, entries: list[DatasetEntry], selected: set[Path]) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        for index, entry in enumerate(entries):
            path = entry.path
            label = path.relative_to(self.home).as_posix() if path.is_relative_to(self.home) else path.name
            item = QTreeWidgetItem(
                [
                    f"{index + 1:02}",
                    label,
                    "—" if entry.episodes is None else f"{entry.episodes:,}",
                    "—" if entry.frames is None else f"{entry.frames:,}",
                    "—" if entry.fps is None else f"{entry.fps:g}",
                ]
            )
            item.setData(1, Qt.UserRole, entry)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if path in selected else Qt.Unchecked)
            for column in range(5):
                item.setToolTip(column, entry.error or str(path))
            for column in (2, 3, 4):
                item.setTextAlignment(column, Qt.AlignRight | Qt.AlignVCenter)
            if entry.error:
                item.setForeground(1, QColor(COLORS["amber"]))
            self.tree.addTopLevelItem(item)
        self.tree.blockSignals(False)
        self._filter()
        self._selection_changed()

    def refresh(self) -> None:
        """Rescan without losing explicit additions, checks or the user's ordering."""
        if self.busy or self.demo:
            return
        try:
            if not self.source_input.text().strip():
                raise ValueError("请先选择源数据父目录。")
            paths = discover_datasets(Path(self.source_input.text().strip()))
        except (OSError, ValueError) as exc:
            self._set_status("读取数据集失败", str(exc), "amber")
            self._log(str(exc))
            return
        selected = set(self.selected_paths())
        available = set(paths) | self._manual_paths
        ordered = [
            item.data(1, Qt.UserRole).path
            for item in self._items()
            if item.data(1, Qt.UserRole).path in available
        ]
        ordered.extend(sorted(available - set(ordered), key=natural_key))
        self._populate([DatasetEntry.read(path) for path in ordered], selected)

    def browse_source(self) -> None:
        """Use Qt's native chooser so Chinese paths remain readable."""
        if self.busy or self.demo:
            return
        chosen = QFileDialog.getExistingDirectory(self, "选择源数据父目录", self.source_input.text())
        if chosen:
            self.source_input.setText(chosen)
            self.refresh()

    def add_dataset(self) -> None:
        """Add a dataset outside the scanned directory and select it once."""
        if self.busy or self.demo:
            return
        chosen = QFileDialog.getExistingDirectory(self, "选择 LeRobot 数据集目录", self.source_input.text())
        if not chosen:
            return
        path = Path(chosen).expanduser().resolve()
        if not is_dataset_dir(path):
            self._set_status("不是有效的数据集目录", f"找不到 meta/info.json 或 data/：{path}", "amber")
            return
        self._manual_paths.add(path)
        selected = set(self.selected_paths()) | {path}
        entries = [item.data(1, Qt.UserRole) for item in self._items()]
        if path not in {entry.path for entry in entries}:
            entries.append(DatasetEntry.read(path))
        self._populate(entries, selected)

    def _filter(self) -> None:
        query = self.search_input.text().casefold().strip()
        visible = 0
        for item in self._items():
            hidden = query not in str(item.data(1, Qt.UserRole).path).casefold()
            item.setHidden(hidden)
            visible += not hidden
        total = self.tree.topLevelItemCount()
        self.dataset_count.setText(f"显示 {visible} / {total} 个" if query else f"{total} 个数据集")
        self.empty_label.setText(
            "没有匹配的数据集\n换个关键词试试。"
            if total
            else "尚未发现数据集\n选择源目录，或点击「添加目录」导入。"
        )
        self.empty_label.setVisible(visible == 0)
        self.tree.setVisible(visible > 0)

    def _check_visible(self, checked: bool) -> None:
        self.tree.blockSignals(True)
        for item in self._items():
            if not checked or not item.isHidden():
                item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        self.tree.blockSignals(False)
        self._selection_changed()

    def move_selected(self, direction: int) -> None:
        """Move highlighted rows as a group while preserving their checkboxes."""
        if self.busy:
            return
        selected = {self.tree.indexOfTopLevelItem(item) for item in self.tree.selectedItems()}
        for index in sorted(selected, reverse=direction > 0):
            other = index + direction
            if 0 <= other < self.tree.topLevelItemCount() and other not in selected:
                item = self.tree.takeTopLevelItem(index)
                self.tree.insertTopLevelItem(other, item)
                item.setSelected(True)
                selected.remove(index)
                selected.add(other)
        self.tree.blockSignals(True)
        for index, item in enumerate(self._items()):
            item.setText(0, f"{index + 1:02}")
        self.tree.blockSignals(False)
        self._filter()
        self._selection_changed()

    def _selection_changed(self, *_args: Any) -> None:
        entries = [item.data(1, Qt.UserRole) for item in self._items() if item.checkState(0) == Qt.Checked]
        self.metrics["datasets"].setText(str(len(entries)))
        for key, attribute in (("episodes", "episodes"), ("frames", "frames")):
            values = [getattr(entry, attribute) for entry in entries]
            self.metrics[key].setText("—" if None in values else f"{sum(values):,}")
        if self.repo_id_auto:
            paths = (
                self.selected_paths()
                if len(entries) >= 2
                else [item.data(1, Qt.UserRole).path for item in self._items()]
            )
            self.repo_input.setText(default_repo_id(paths, Path(self.source_input.text()), self.home))
        self._update_preview()

    def _repo_edited(self, _text: str) -> None:
        self.repo_id_auto = False

    def _reset_name(self) -> None:
        self.repo_id_auto = True
        self._selection_changed()

    def _command(self) -> tuple[list[str], Path]:
        for item in self._items():
            entry = item.data(1, Qt.UserRole)
            if item.checkState(0) == Qt.Checked and entry.error:
                raise ValueError(entry.error)
        return build_merge_command(self.selected_paths(), self.repo_input.text(), self.home)

    def _update_preview(self, *_args: Any) -> None:
        repo_id = self.repo_input.text().strip()
        valid_id = bool(REPO_ID_PATTERN.fullmatch(repo_id))
        output = self.home / repo_id if valid_id else None
        self.output_preview.setPlainText(str(output) if output else "填写 namespace/dataset 后显示保存位置")
        self.output_hint.setText(
            "目录已存在，请更换输出名称。"
            if output and output.exists()
            else "LeRobot 根目录 + Repo ID · 自动创建新目录"
        )
        self.output_hint.setStyleSheet(
            f"color: {COLORS['amber' if output and output.exists() else 'subtle']}; font-size: 12px;"
        )
        if self.busy:
            return
        self.command_box.clear()
        self.merge_button.setEnabled(False)
        if self.demo:
            self._set_status("界面演示 · 当前为示例数据", "演示模式下不会读取或合并真实数据。", "amber")
            return
        try:
            command, _output = self._command()
        except (OSError, ValueError) as exc:
            self._set_status(
                "等待配置" if len(self.selected_paths()) < 2 else "请检查合并配置", str(exc), "muted"
            )
        else:
            self.command_box.setPlainText(shlex.join(command))
            self.merge_button.setEnabled(True)
            self._set_status(
                f"准备就绪 · 已选 {len(self.selected_paths())} 个数据集",
                "将按列表顺序合并；开始后检查数据完整性与特征兼容性。",
                "green",
            )

    def _set_status(self, title: str, detail: str, tone: str = "text") -> None:
        self.status_label.setText(title)
        self.status_label.setStyleSheet(f"color: {COLORS[tone]}; font-weight: 600;")
        self.detail_label.setText(detail)

    def _log(self, text: str) -> None:
        bar = self.log_box.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.log_box.appendPlainText(ANSI_ESCAPE.sub("", text).strip("\r\n"))
        if at_bottom:
            bar.setValue(bar.maximum())

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        for widget in self._controls:
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.merge_button.setEnabled(False)
        self.open_button.setEnabled(not busy and self.output_dir is not None and self.output_dir.is_dir())
        self.progress.setRange(0, 0 if busy else 1)
        self.progress.setValue(0)
        self.merge_button.setText("正在合并…" if busy else "开始合并  →")
        self.state_badge.setText("●  合并进行中" if busy else "●  本地合并")

    def start_merge(self) -> None:
        """Snapshot the current configuration, then freeze controls until the CLI exits."""
        if self.busy or self.demo or (self.worker is not None and self.worker.isRunning()):
            return
        try:
            command, output = self._command()
        except (OSError, ValueError) as exc:
            self._update_preview()
            self._set_status("无法开始合并", str(exc), "amber")
            return
        self.output_dir = output
        self._show_details(True)
        self._log("\n输入数据集（按顺序）：\n" + "\n".join(f"  {path}" for path in self.selected_paths()))
        self._log(f"输出目录：{output}\n命令：{shlex.join(command)}\n")
        self._set_busy(True)
        self._set_status("正在合并数据集…", "日志实时更新；完成后可直接打开输出目录。", "blue")
        self._started_at = time.monotonic()
        self._update_elapsed()
        self._elapsed_timer.start(1000)
        self.worker = MergeWorker(command, self)
        self.worker.output.connect(self._log)
        self.worker.result.connect(self._merge_result)
        self.worker.start()

    def stop_merge(self) -> None:
        """Stop the process group and retain partial output for the user to inspect."""
        if not self.busy or self.worker is None:
            return
        self.worker.request_stop()
        self.stop_button.setEnabled(False)
        self._set_status("正在停止…", "正在结束合并进程，请稍候。", "amber")
        self._kill_timer.start(4000)

    def _force_stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.force_stop()

    def _update_elapsed(self) -> None:
        seconds = int(time.monotonic() - self._started_at)
        self.elapsed_label.setText(f"用时 {seconds // 60:02}:{seconds % 60:02}")

    def _merge_result(self, code: int, error: str) -> None:
        self._elapsed_timer.stop()
        self._kill_timer.stop()
        self._update_elapsed()
        cancelled = self.worker is not None and self.worker.stop_requested.is_set()
        self._set_busy(False)
        self._update_preview()
        self._log(f"进程退出，退出码：{code}" + (f"\n{error}" if error else ""))
        if cancelled:
            self._set_status(
                "合并已停止", "输出目录可能包含未完成的数据，请检查后再使用或更换输出名称。", "amber"
            )
        elif code == 0:
            self.progress.setValue(1)
            self.state_badge.setText("●  合并完成")
            self._set_status("合并完成", f"已保存到 {self.output_dir}", "green")
        else:
            self._set_status(
                "无法启动合并" if error else f"合并失败 · 退出码 {code}",
                error or "查看运行日志了解原因；已有的输出目录可能尚未完成。",
                "red",
            )

    def open_output(self) -> None:
        """Open the completed or partial result in the system file manager."""
        if self.output_dir is not None and self.output_dir.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))

    def _load_demo(self) -> None:
        entries = [
            DatasetEntry(self.home / "block" / f"shape-matching_{start}-{start + 10}", 10, frames, 30)
            for start, frames in ((0, 8450), (10, 9120), (20, 7860), (30, 9310), (40, 8640))
        ]
        self._populate(entries, {entry.path for entry in entries[:3]})
        for widget in (self.source_input, self.refresh_button, self.open_button):
            widget.setEnabled(False)
        self.state_badge.setText("●  界面演示")
        self._log(
            "[示例] 选择源目录 → 勾选数据集 → 确认输出 → 开始合并\n可点击勾选、搜索和调整顺序，预览统计与自动命名。"
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt virtual method
        """Keep the worker alive until the requested stop has completed."""
        if self.busy or (self.worker is not None and self.worker.isRunning()):
            self._set_status("合并进程仍在运行", "如需关闭，请先点击「停止合并」，等待进程结束。", "amber")
            event.ignore()
            return
        event.accept()


def launch(*, source: Path | None = None, demo: bool = False) -> None:
    """Start Qt with plugin paths that also work in a Chinese-named project folder."""
    app = QApplication.instance()
    if app is None:
        plugins = Path(QtCore.__file__).resolve().parent / "Qt5/plugins"
        if (plugins / "platforms").is_dir():
            os.environ["QT_PLUGIN_PATH"] = str(plugins)
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins / "platforms")
            QtCore.QCoreApplication.setLibraryPaths([str(plugins)])
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(["lerobot-dataset-merge"])
    app.setStyle("Fusion")
    window = MergeWindow(source=source, demo=demo)
    window.show()
    app.exec_()
