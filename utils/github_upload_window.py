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

"""Qt desktop interface with native antialiased CJK text, independent of Tk builds."""

from __future__ import annotations

import queue
import threading
import webbrowser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent, QColor, QFont, QFontDatabase, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
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
    QShortcut,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .github_upload_config import (
        APPLICATION_ID,
        ConfigError,
        UploadPreferences,
        application_icon_path,
        default_config_path,
        load_preferences,
        parse_preferences,
        require_external_config,
        save_preferences,
        save_preferences_text,
    )
    from .github_upload_gui import (
        AUTH_HELP,
        GitRepo,
        Preview,
        UploadError,
        UploadProgress,
        UploadSettings,
        format_size,
        normalize_remote,
    )
else:
    from github_upload_config import (
        APPLICATION_ID,
        ConfigError,
        UploadPreferences,
        application_icon_path,
        default_config_path,
        load_preferences,
        parse_preferences,
        require_external_config,
        save_preferences,
        save_preferences_text,
    )
    from github_upload_gui import (
        AUTH_HELP,
        GitRepo,
        Preview,
        UploadError,
        UploadProgress,
        UploadSettings,
        format_size,
        normalize_remote,
    )

COLORS = {
    "window": "#080D16",
    "panel": "#0E1624",
    "card": "#152134",
    "border": "#2B3C56",
    "text": "#F4F7FC",
    "muted": "#C1CDDF",
    "subtle": "#A2B2CA",
    "blue": "#66ABFF",
    "cyan": "#65DDEB",
    "green": "#6DE0AC",
    "amber": "#FFC779",
    "red": "#FF8297",
}
STYLE = """
QMainWindow, QDialog { background: #080D16; }
QWidget { color: #F4F7FC; font-size: 14px; }
QLabel { background: transparent; }
QFrame[card="true"] { background: #0E1624; border: 1px solid #2B3C56; border-radius: 12px; }
QFrame[inset="true"] { background: #152134; border-radius: 9px; }
QLineEdit, QPlainTextEdit, QTextEdit { background: #152134; color: #F4F7FC;
    border: 1px solid #2B3C56; border-radius: 7px; padding: 8px; font-size: 15px;
    selection-background-color: #316CAF; }
QLineEdit:focus { border-color: #66ABFF; }
QLineEdit:disabled { color: #A2B2CA; }
QPushButton { background: #152134; border: 1px solid #344A69; border-radius: 8px;
    padding: 9px 15px; font-size: 14px; min-height: 18px; }
QPushButton:hover { background: #213653; border-color: #66ABFF; }
QPushButton:disabled { color: #A2B2CA; border-color: #26364C; }
QPushButton[primary="true"] { background: #66ABFF; color: #091320; border: none; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #8FC1FF; }
QPushButton[primary="true"]:disabled { background: #263B56; color: #A2B2CA; }
QCheckBox { spacing: 9px; font-size: 14px; background: transparent; }
QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #7891B2; border-radius: 4px; background: #152134; }
QCheckBox::indicator:checked { background: #66ABFF; border-color: #66ABFF; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { width: 9px; background: #0E1624; margin: 0; }
QScrollBar::handle:vertical { background: #3A506F; border-radius: 4px; min-height: 32px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: #152134; color: #C1CDDF; padding: 8px 18px; border-radius: 6px; margin: 0 6px 8px 0; }
QTabBar::tab:selected { background: #2B425F; color: #FFFFFF; }
QTreeWidget { background: #0E1624; alternate-background-color: #152134; color: #F4F7FC;
    border: none; outline: none; font-size: 14px; }
QTreeWidget::item { padding: 9px 4px; border-bottom: 1px solid #1D2B40; }
QTreeWidget::item:selected { background: #244668; }
QHeaderView::section { background: #17263B; color: #C1CDDF; border: none; padding: 8px 4px; }
QProgressBar { background: #20334D; border: none; border-radius: 4px; }
QProgressBar::chunk { background: #66ABFF; border-radius: 4px; }
QToolTip { color: #F4F7FC; background: #20334D; border: 1px solid #66ABFF; padding: 6px; }
"""


def choose_font() -> QFont:
    """Use a scalable CJK font and let Qt handle monitor DPI and antialiasing."""
    available = set(QFontDatabase().families())
    family = next(
        (
            name
            for name in ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei")
            if name in available
        ),
        "Sans Serif",
    )
    selected = QFont(family)
    selected.setPixelSize(15)
    selected.setStyleStrategy(QFont.PreferAntialias)
    return selected


class UploadWindow(QMainWindow):
    """Run Git in a worker while Qt renders all text, inputs and dialogs natively."""

    def __init__(
        self,
        directory: Path,
        *,
        demo: bool = False,
        preferences: UploadPreferences | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Build the upload form, preview and worker event polling in the GUI thread."""
        super().__init__()
        self.setWindowTitle("GitHub Upload · 代码上传")
        self.resize(1180, 920)
        self.setMinimumSize(1040, 780)
        self.setFont(choose_font())
        family = self.font().family().replace('"', '\\"')
        self.setStyleSheet(f'QWidget {{ font-family: "{family}"; }}\n' + STYLE)
        self.demo = demo
        self.preferences = UploadPreferences() if demo else (preferences or UploadPreferences())
        self.config_path = (config_path or default_config_path()).expanduser().resolve()
        self.busy = False
        self._closed = False
        self._preview: Preview | None = None
        self._modal: QDialog | None = None
        self._last_transfer: UploadProgress | None = None
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._controls: list[QWidget] = []
        self.inputs = {
            key: QLineEdit(str(directory) if key == "directory" else getattr(self.preferences, key))
            for key in ("directory", "remote", "branch", "message", "username", "name", "email", "token")
        }
        self.code_only = QCheckBox("仅上传当前代码")
        self.code_only.setChecked(self.preferences.code_only)
        self.create_ignore = QCheckBox("缺少 .gitignore 时创建基础忽略规则")
        self.create_ignore.setChecked(self.preferences.create_ignore)
        self._controls.extend([*self.inputs.values(), self.code_only, self.create_ignore])
        self.inputs["token"].setEchoMode(QLineEdit.Password)
        self._build()
        self._refresh_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self._refresh_shortcut.activated.connect(self._start_preview)
        self.inputs["directory"].textChanged.connect(self._directory_changed)
        self.code_only.toggled.connect(self._mode_changed)
        self._show_empty()
        if demo:
            self._load_demo()
        else:
            self._log("准备就绪。选择项目目录，点击「预览改动」开始。")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_events)
        self._timer.start(60)

    def _label(
        self, text: str, *, size: int = 14, color: str = "text", bold: bool = False, wrap: bool = False
    ) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color: {COLORS[color]}; font-size: {size}px; font-weight: {'600' if bold else '400'}; background: transparent;"
        )
        label.setWordWrap(wrap)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setTextFormat(Qt.PlainText)
        return label

    def _button(self, text: str, command: Callable, *, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("primary", primary)
        button.clicked.connect(lambda _checked=False: command())
        self._controls.append(button)
        return button

    def _card(self, title: str, subtitle: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setProperty("card", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(9)
        layout.addWidget(self._label(title, size=18, bold=True))
        layout.addWidget(self._label(subtitle, color="muted", wrap=True))
        return frame, layout

    def _field(self, layout: QVBoxLayout, key: str, title: str, placeholder: str = "") -> None:
        layout.addWidget(self._label(title, color="muted"))
        self.inputs[key].setPlaceholderText(placeholder)
        layout.addWidget(self.inputs[key])

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        page = QVBoxLayout(central)
        page.setContentsMargins(26, 22, 26, 20)
        page.setSpacing(16)
        header = QHBoxLayout()
        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(58, 58)
        logo.setPixmap(QIcon(str(application_icon_path())).pixmap(58, 58))
        header.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(self._label("GitHub Upload", size=28, bold=True))
        titles.addWidget(self._label("代码上传  /  从本地项目到 GitHub", color="muted"))
        header.addLayout(titles)
        header.addStretch()
        header.addWidget(self._button("参数文件", self._edit_config))
        header.addSpacing(12)
        header.addWidget(self.code_only)
        header.addSpacing(18)
        self.state_badge = self._label("●  准备就绪", color="green")
        header.addWidget(self.state_badge)
        page.addLayout(header)
        self.steps = self._label(
            "01  选择项目       →       02  检查改动       →       03  提交并上传", color="muted"
        )
        self.steps.setStyleSheet(
            "background: #101C2C; border: 1px solid #2B3C56; border-radius: 9px; padding: 10px 16px; color: #C1CDDF; font-size: 14px;"
        )
        page.addWidget(self.steps)
        body = QHBoxLayout()
        body.setSpacing(18)
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        form = QWidget()
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(0, 0, 10, 0)
        form_layout.setSpacing(14)
        project, fields = self._card("01  项目与目标", "选择项目根目录，连接你的 GitHub 仓库")
        fields.addWidget(self._label("本地项目目录", color="muted"))
        directory_row = QHBoxLayout()
        directory_row.addWidget(self.inputs["directory"], 1)
        directory_row.addWidget(self._button("浏览…", self._browse))
        fields.addLayout(directory_row)
        self._field(fields, "remote", "GitHub 仓库地址", "https://github.com/用户名/仓库.git")
        self._field(fields, "branch", "目标分支", "main")
        fields.addWidget(
            self._button("新建 GitHub 仓库 ↗", lambda: webbrowser.open("https://github.com/new"))
        )
        form_layout.addWidget(project)
        identity, fields = self._card("02  提交与身份", "描述本次更新，填写常用身份或使用已有 Git 凭据")
        self._field(fields, "message", "提交说明", "例如：Add recording scripts")
        pair = QHBoxLayout()
        for key, title in (("name", "提交者姓名"), ("email", "提交者邮箱")):
            column = QVBoxLayout()
            self._field(column, key, title)
            pair.addLayout(column)
        fields.addLayout(pair)
        self._field(fields, "username", "GitHub 用户名", "可选，例如 ZihanYan-CQU")
        fields.addWidget(self._label("访问 Token（可保存到本机参数文件）", color="muted"))
        secret = QHBoxLayout()
        secret.addWidget(self.inputs["token"], 1)
        self.token_toggle = self._button("显示", self._toggle_token)
        secret.addWidget(self.token_toggle)
        fields.addLayout(secret)
        fields.addWidget(self._label("Token 留空时使用已有 Git / SSH 凭据。", color="subtle", wrap=True))
        fields.addWidget(self._button("Token 使用说明 ↗", lambda: webbrowser.open(AUTH_HELP)))
        fields.addWidget(self.create_ignore)
        form_layout.addWidget(identity)
        config, fields = self._card("03  常用参数", "保存用户名、Token 和仓库，下次启动自动填入")
        path = QLineEdit(str(self.config_path))
        path.setReadOnly(True)
        path.setToolTip(str(self.config_path))
        fields.addWidget(path)
        actions = QHBoxLayout()
        actions.addWidget(self._button("编辑参数", self._edit_config))
        actions.addWidget(self._button("保存配置", self._save_config))
        actions.addWidget(self._button("重新读取", self._reload_config))
        fields.addLayout(actions)
        fields.addWidget(self._label("参数保存在项目外，仅当前系统用户可读写。", color="subtle", wrap=True))
        form_layout.addWidget(config)
        form_layout.addStretch()
        self.form_scroll.setWidget(form)
        body.addWidget(self.form_scroll, 1)
        review, layout = self._card("文件预览", "核对本次上传的文件和大小")
        self.file_badge = self._label("等待检查", color="muted")
        layout.addWidget(self.file_badge)
        self.branch_label = self._label("本地分支  —  /  当前提交  —", color="subtle", wrap=True)
        layout.addWidget(self.branch_label)
        counters = QHBoxLayout()
        self.counter_labels = {}
        self.counter_titles = {}
        for kind, title, color in (
            ("add", "新增", "green"),
            ("modify", "修改", "blue"),
            ("delete", "删除", "red"),
        ):
            box = QFrame()
            box.setProperty("inset", True)
            inside = QVBoxLayout(box)
            number = self._label("—", size=25, color=color)
            caption = self._label(title, color="muted")
            inside.addWidget(number)
            inside.addWidget(caption)
            self.counter_labels[kind] = number
            self.counter_titles[kind] = caption
            counters.addWidget(box, 1)
        layout.addLayout(counters)
        total = QHBoxLayout()
        self.size_title = self._label("文件总大小（未压缩）", color="muted")
        self.size_label = self._label("—", size=22, color="cyan")
        total.addWidget(self.size_title)
        total.addStretch()
        total.addWidget(self.size_label)
        layout.addLayout(total)
        self.tabs = QTabWidget()
        self.file_list = QTreeWidget()
        self.file_list.setHeaderLabels(["状态", "文件", "大小"])
        self.file_list.setRootIsDecorated(False)
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setUniformRowHeights(True)
        self.file_list.setSortingEnabled(False)
        self.file_list.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.file_list.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.file_list.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(4000)
        self.tabs.addTab(self.file_list, "文件改动")
        self.tabs.addTab(self.log_box, "操作日志")
        layout.addWidget(self.tabs, 1)
        self.review_note = self._label("", color="muted", wrap=True)
        layout.addWidget(self.review_note)
        body.addWidget(review, 1)
        page.addLayout(body, 1)
        footer = QHBoxLayout()
        footer.setSpacing(18)
        status = QVBoxLayout()
        self.status_label = self._label("从一个本地项目开始", size=16, bold=True)
        self.status_detail = self._label("预览文件，核对仓库地址后上传。", color="muted", wrap=True)
        status.addWidget(self.status_label)
        status.addWidget(self.status_detail)
        progress = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(9)
        self.progress_percent = self._label("", color="cyan")
        progress.addWidget(self.progress, 1)
        progress.addWidget(self.progress_percent)
        self.progress_container = QWidget()
        self.progress_container.setLayout(progress)
        progress.setContentsMargins(0, 0, 0, 0)
        self.progress_container.hide()
        status.addWidget(self.progress_container)
        footer.addLayout(status, 1)
        self.preview_button = self._button("预览改动", self._start_preview)
        self.upload_button = self._button(
            "提交并上传  →", lambda: self._start_preview(upload=True), primary=True
        )
        self.upload_button.setMinimumWidth(175)
        footer.addWidget(self.preview_button)
        footer.addWidget(self.upload_button)
        page.addLayout(footer)

    def _set_status(self, title: str, detail: str = "", *, color: str = "text") -> None:
        self.status_label.setText(title)
        self.status_label.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {COLORS[color]};")
        self.status_detail.setText(detail)

    def _show_empty(self) -> None:
        self.file_list.clear()
        self.size_label.setText("—")
        self.file_badge.setText("等待检查")
        self.branch_label.setText("本地分支  —  /  当前提交  —")
        for label in self.counter_labels.values():
            label.setText("—")
        self.review_note.setText(
            "仅上传当前代码，跳过旧历史和 LFS 资源。"
            if self.code_only.isChecked()
            else "上传完整提交历史；历史中的 LFS 数据需要完整可用。"
        )

    def _directory_changed(self) -> None:
        for key in ("remote", "branch"):
            self.inputs[key].clear()
        for key in ("name", "email"):
            self.inputs[key].setText(getattr(self.preferences, key))
        self._clear_token()
        self._preview = None
        self._show_empty()
        self._set_status("项目目录已更改", "点击预览，重新读取当前项目。")

    def _mode_changed(self) -> None:
        self._preview = None
        self._show_empty()
        self._set_status("上传模式已更改", "请重新预览本次要上传的文件。")

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择要上传的项目根目录", self.inputs["directory"].text()
        )
        if selected:
            self.inputs["directory"].setText(selected)

    def _toggle_token(self) -> None:
        visible = self.inputs["token"].echoMode() == QLineEdit.Password
        self.inputs["token"].setEchoMode(QLineEdit.Normal if visible else QLineEdit.Password)
        self.token_toggle.setText("隐藏" if visible else "显示")

    def _clear_token(self, *, restore: bool = True) -> None:
        self.inputs["token"].setText(self.preferences.token if restore else "")
        self.inputs["token"].setEchoMode(QLineEdit.Password)
        self.token_toggle.setText("显示")

    def _settings(self) -> UploadSettings:
        values = {key: entry.text().strip() for key, entry in self.inputs.items()}
        if not values["directory"]:
            raise UploadError("请选择本地项目目录。")
        values["directory"] = Path(values["directory"]).expanduser()
        values["branch"] = values["branch"] or "main"
        try:
            require_external_config(self.config_path, values["directory"])
        except ConfigError as exc:
            raise UploadError(str(exc)) from exc
        return UploadSettings(
            **values, code_only=self.code_only.isChecked(), create_ignore=self.create_ignore.isChecked()
        )

    def _save_config(self) -> None:
        if self.busy or self.demo:
            return
        try:
            settings = self._settings()
            values = {key: entry.text().strip() for key, entry in self.inputs.items()}
            values["directory"] = str(settings.directory.resolve())
            preferences = UploadPreferences(
                **values, code_only=settings.code_only, create_ignore=settings.create_ignore
            )
            save_preferences(self.config_path, preferences)
        except (UploadError, ConfigError) as exc:
            self._dialog("无法保存配置", str(exc), tone="amber")
            return
        self.preferences = preferences
        self._clear_token()
        self._log(f"常用参数已保存：{self.config_path}")
        self._set_status("常用参数已保存", "下次启动自动填入，也可以直接编辑参数文件。", color="green")

    def _reload_config(self) -> None:
        if self.busy or self.demo:
            return
        try:
            if not self.config_path.is_file():
                raise ConfigError("参数文件尚未创建，请先点击「保存配置」。")
            preferences = load_preferences(self.config_path)
            directory = Path(preferences.directory or self.inputs["directory"].text()).expanduser()
            if not directory.is_absolute():
                directory = self.config_path.parent / directory
            require_external_config(self.config_path, directory)
        except ConfigError as exc:
            self._dialog("无法读取配置", str(exc), tone="amber")
            return
        self.preferences = preferences
        self.inputs["directory"].setText(str(directory.resolve()))
        for key, entry in self.inputs.items():
            if key != "directory":
                entry.setText(getattr(preferences, key))
        self.code_only.setChecked(preferences.code_only)
        self.create_ignore.setChecked(preferences.create_ignore)
        self._clear_token()
        self._mode_changed()
        self._log(f"已重新读取参数：{self.config_path}")
        self._set_status("常用参数已重新读取", "点击预览，检查本次文件。")

    def _edit_config(self) -> None:
        if self.busy or self.demo:
            return
        if self._modal is not None:
            self._modal.raise_()
            return
        if not self.config_path.exists():
            self._save_config()
            if not self.config_path.is_file():
                return
        try:
            original = self.config_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            self._dialog(
                "无法打开参数文件", f"请检查文件权限与 UTF-8 编码：\n{self.config_path}", tone="amber"
            )
            return

        modal = QDialog(self)
        modal.setObjectName("configEditorDialog")
        modal.setWindowTitle("编辑参数文件")
        modal.setAttribute(Qt.WA_DeleteOnClose)
        modal.setModal(True)
        modal.resize(760, 650)
        modal.setMinimumSize(580, 440)
        layout = QVBoxLayout(modal)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        layout.addWidget(self._label("编辑参数文件", size=22, bold=True))
        layout.addWidget(self._label(str(self.config_path), color="muted", wrap=True))
        layout.addWidget(self._label("修改后点击「保存并应用」，窗口会立即更新。", color="muted", wrap=True))
        editor = QPlainTextEdit()
        editor.setObjectName("configTextEditor")
        editor.setPlainText(original)
        layout.addWidget(editor, 1)
        error = self._label("", color="red", wrap=True)
        error.setObjectName("configEditorError")
        error.hide()
        layout.addWidget(error)
        actions = QHBoxLayout()
        actions.addStretch()
        cancel = QPushButton("取消")
        cancel.clicked.connect(modal.reject)
        actions.addWidget(cancel)
        save = QPushButton("保存并应用")
        save.setProperty("primary", True)
        actions.addWidget(save)
        layout.addLayout(actions)

        def apply() -> None:
            text = editor.toPlainText()
            try:
                preferences = parse_preferences(text)
                directory = Path(preferences.directory or self.inputs["directory"].text()).expanduser()
                if not directory.is_absolute():
                    directory = self.config_path.parent / directory
                require_external_config(self.config_path, directory)
                if self.config_path.read_text(encoding="utf-8") != original:
                    raise ConfigError("参数文件已被其他程序修改，请关闭此编辑器后重新打开。")
                save_preferences_text(self.config_path, text)
            except ConfigError as exc:
                error.setText(str(exc))
                error.show()
                return
            except (OSError, UnicodeError):
                error.setText("无法读写参数文件，请检查文件是否存在及其权限。")
                error.show()
                return
            modal.accept()
            self._reload_config()
            self._set_status(
                "参数文件已保存并应用", "新的用户名、Token 与仓库等参数已填入窗口。", color="green"
            )

        save.clicked.connect(apply)
        self._modal = modal
        modal.finished.connect(self._dialog_closed)
        modal.open()

    def _render_preview(self, state: Preview) -> None:
        self._preview = state
        self.file_list.clear()
        lines = state.status.splitlines()
        self.file_badge.setText(f"{len(lines)} 个文件")
        self.branch_label.setText(f"本地分支  {state.branch}   /   当前提交  {state.head[:8] or '尚无'}")
        self.size_title.setText("文件总大小（未压缩）" if state.code_only else "当前改动大小（不含历史）")
        self.size_label.setText(format_size(state.total_bytes))
        counts = {"add": 0, "modify": 0, "delete": 0}
        for index, line in enumerate(lines):
            code = line[:2]
            kind = "delete" if "D" in code else "add" if code == "??" or "A" in code else "modify"
            counts[kind] += 1
            title = {"add": "新增", "modify": "修改", "delete": "删除"}[kind]
            color = {"add": "green", "modify": "blue", "delete": "red"}[kind]
            if state.code_only:
                title, color = "代码", "blue"
            filename = state.file_sizes[index][0] if index < len(state.file_sizes) else line[3:]
            size = format_size(state.file_sizes[index][1]) if index < len(state.file_sizes) else "—"
            item = QTreeWidgetItem([title, filename, "已删除" if kind == "delete" else size])
            item.setForeground(0, QColor(COLORS[color]))
            item.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
            item.setToolTip(1, filename)
            self.file_list.addTopLevelItem(item)
        titles = {"add": "新增", "modify": "修改", "delete": "删除"}
        if state.code_only:
            counts = {"add": len(lines), "modify": state.excluded_lfs, "delete": 0}
            titles = {"add": "文件", "modify": "跳过 LFS", "delete": "旧历史"}
        for key, value in counts.items():
            self.counter_labels[key].setText(str(value))
            self.counter_titles[key].setText(titles[key])
        self.review_note.setText(
            f"当前代码快照，跳过 {state.excluded_lfs} 个 LFS 资源。\n后续更新会接在个人仓库的最新版本之后。"
            if state.code_only
            else "包含完整提交历史及已有暂存内容。\n历史中的 LFS 对象也需要完整提供。"
        )

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        for control in self._controls:
            control.setEnabled(not busy and not self.demo)
        if busy:
            self._last_transfer = None
            self.progress.setRange(0, 0)
            self.progress_percent.setText("处理中")
            self.progress_container.show()
            self.state_badge.setText("●  正在处理")
        else:
            self.progress_container.hide()

    def _update_progress(self, state: UploadProgress) -> None:
        self.progress_container.show()
        if state.fraction is None:
            self.progress.setRange(0, 0)
            self.progress_percent.setText("处理中")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(round(state.fraction * 100))
            self.progress_percent.setText(f"{state.fraction:.0%}")
        details = [state.detail] if state.detail else []
        if state.transferred_bytes is not None:
            amount = format_size(state.transferred_bytes)
            if state.total_bytes is not None:
                details.append(f"{amount} / {format_size(state.total_bytes)}")
            else:
                details.append(f"已发送 {amount}")
                self._last_transfer = state
        if state.speed_bytes is not None:
            details.append(f"{format_size(state.speed_bytes)}/s")
        self._set_status(
            state.phase + (" · 当前阶段" if state.fraction is not None else "…"), "   ·   ".join(details)
        )

    def _background(self, job: Callable, event: str) -> None:
        self._set_busy(True)

        def worker() -> None:
            try:
                self._events.put((event, job()))
            except Exception as exc:
                self._events.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _start_preview(self, upload: bool = False) -> None:
        if self.busy or self.demo or self._modal is not None:
            return
        try:
            selected = self._settings()
        except UploadError as exc:
            self._dialog("请检查输入", str(exc), tone="amber")
            return
        self._set_status("正在读取项目改动…", "检查分支、文件与上传大小。")

        def job() -> tuple[Preview, bool]:
            repo = GitRepo(selected.directory, lambda text: self._events.put(("log", text)))
            repo.prepare(selected.branch, selected.create_ignore)
            return repo.preview(code_only=selected.code_only), upload

        self._background(job, "preview")

    def _preview_ready(self, state: Preview, upload: bool) -> None:
        for source, target in (
            ("branch", "branch"),
            ("origin", "remote"),
            ("name", "name"),
            ("email", "email"),
        ):
            value = getattr(state, source)
            if state.code_only and source == "branch":
                value = "main"
            if state.code_only and source == "origin":
                value = ""
            if source == "origin" and value:
                try:
                    value = normalize_remote(value)
                except UploadError:
                    value = ""
            if not self.inputs[target].text():
                self.inputs[target].setText(value)
        self._render_preview(state)
        self.tabs.setCurrentIndex(0)
        self.state_badge.setText("●  预览已更新")
        self._log(
            f"预览完成：{len(state.status.splitlines())} 个文件，{format_size(state.total_bytes)}（工作区文件大小）。"
        )
        self._set_status("预览已更新，可以准备上传", "核对仓库、分支和文件列表。")
        if upload:
            try:
                selected = self._settings()
                normalize_remote(selected.remote)
                if not selected.message:
                    raise UploadError("请填写提交说明。")
                if selected.token and not selected.remote.startswith("https://"):
                    raise UploadError("Token 仅用于 HTTPS 地址，使用 SSH 时请清空 Token。")
                if state.status and (not selected.name or not selected.email):
                    raise UploadError("请填写提交者姓名和邮箱，或先配置 Git 提交身份。")
            except UploadError as exc:
                self._dialog("请检查上传信息", str(exc), tone="amber")
                return
            self._confirm_upload(selected, state)

    def _confirm_upload(self, selected: UploadSettings, state: Preview) -> None:
        mode = (
            f"仅上传当前代码，跳过 {state.excluded_lfs} 个 LFS 资源，保留原项目本地历史。"
            if selected.code_only
            else "提交全部预览改动，并推送完整本地提交历史。"
        )
        summary = (
            f"本地项目\n{selected.directory}\n\nGitHub 目标\n{selected.remote}\n\n目标分支：{selected.branch}\n"
            f"{len(state.status.splitlines())} 个文件 · {format_size(state.total_bytes)}（{'未压缩' if state.code_only else '当前改动，不含历史'}）\n"
            f"提交说明：{selected.message}\n\n{mode}"
        )

        def confirm() -> None:
            self._set_status("正在提交并上传…", "远端确认后会显示上传结果。")
            self.tabs.setCurrentIndex(1)

            def job() -> str:
                repo = GitRepo(
                    selected.directory,
                    lambda text: self._events.put(("log", text)),
                    lambda progress: self._events.put(("progress", progress)),
                )
                return repo.upload(selected, expected=state)

            self._background(job, "success")

        self._dialog("确认上传到 GitHub", summary, confirm=confirm)

    def _dialog(
        self, title: str, message: str, *, confirm: Callable | None = None, tone: str = "blue"
    ) -> None:
        if self._modal is not None:
            self._modal.raise_()
            return
        modal = QDialog(self)
        modal.setWindowTitle(title)
        modal.setAttribute(Qt.WA_DeleteOnClose)
        modal.setModal(True)
        modal.resize(620, 510 if confirm else 420)
        modal.setMinimumSize(520, 360)
        layout = QVBoxLayout(modal)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        layout.addWidget(self._label(title, size=22, color=tone, bold=True))
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(message)
        layout.addWidget(text, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        if confirm:
            cancel = QPushButton("返回检查")
            cancel.clicked.connect(modal.reject)
            actions.addWidget(cancel)
        accept = QPushButton("确认提交并上传" if confirm else "知道了")
        accept.setProperty("primary", True)

        def activate() -> None:
            modal.accept()
            if confirm:
                confirm()

        accept.clicked.connect(activate)
        actions.addWidget(accept)
        layout.addLayout(actions)
        self._modal = modal
        modal.finished.connect(self._dialog_closed)
        modal.open()

    def _dialog_closed(self) -> None:
        self._modal = None

    def _log(self, text: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now():%H:%M:%S}   {text}\n")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _process_events(self) -> None:
        while not self._closed:
            try:
                event, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if event == "log":
                self._log(payload)
                continue
            if event == "progress":
                self._update_progress(payload)
                continue
            self._set_busy(False)
            if event == "preview":
                self._preview_ready(*payload)
            elif event == "error":
                self._clear_token()
                self._log(payload)
                self.tabs.setCurrentIndex(1)
                self.state_badge.setText("●  操作未完成")
                self._set_status("操作未完成，请查看原因", "日志已保留，修复后可重新预览。", color="red")
                self._dialog("操作未完成", payload, tone="red")
            elif event == "success":
                self._clear_token()
                self._log(f"上传成功，提交：{payload}")
                self.state_badge.setText("●  上传成功")
                self.progress.setRange(0, 100)
                self.progress.setValue(100)
                self.progress_percent.setText("100%")
                self.progress_container.show()
                sent = (
                    f" · 发送 {format_size(self._last_transfer.transferred_bytes)}"
                    if self._last_transfer is not None
                    else ""
                )
                self._set_status("代码已成功上传到 GitHub", f"提交 {payload[:12]}{sent}", color="green")
                self._dialog("上传成功", f"代码已推送到指定仓库和分支。\n\n提交：{payload}", tone="green")

    def _load_demo(self) -> None:
        self.inputs["directory"].setText("/home/yzh/projects/robot-learning")
        for key, value in {
            "remote": "https://github.com/yzh/robot-learning.git",
            "branch": "main",
            "message": "Add recording workflow",
            "username": "yzh",
            "name": "yzh",
            "email": "yzh@example.com",
        }.items():
            self.inputs[key].setText(value)
        files = (
            ("README.md", 4850),
            ("src/robot/config.py", 2150),
            ("scripts/train_policy.py", 12350),
            ("examples/record_episode.py", 6700),
            ("configs/so101.yaml", 900),
            ("utils/github_upload_gui.py", 25600),
        )
        self._render_preview(
            Preview(
                "main",
                "a73c9d1249b6",
                "\n".join("?? " + name for name, _size in files),
                "",
                "yzh",
                "yzh@example.com",
                code_only=True,
                excluded_lfs=3,
                file_sizes=files,
            )
        )
        self._set_busy(False)
        self.state_badge.setText("●  界面演示")
        self.upload_button.setText("演示模式 · 禁止上传")
        self._set_status("界面演示 · 当前内容为示例", "不会读取凭据、修改项目或连接 GitHub。", color="amber")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt virtual method
        """Keep an active Git operation alive and clear credentials when closing."""
        if self.busy:
            self._set_status(
                "Git 操作仍在进行，请稍候再关闭", "完成后可关闭窗口，避免中断上传。", color="amber"
            )
            event.ignore()
            return
        self._closed = True
        self._timer.stop()
        self._clear_token(restore=False)
        event.accept()


def launch(
    directory: Path,
    *,
    demo: bool = False,
    preferences: UploadPreferences | None = None,
    config_path: Path | None = None,
) -> None:
    """Start Qt in the project's uv environment; no Tk or Conda runtime is needed."""
    app = QApplication.instance()
    if app is None:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication([APPLICATION_ID])
    app.setApplicationName(APPLICATION_ID)
    app.setDesktopFileName(APPLICATION_ID)
    app.setWindowIcon(QIcon(str(application_icon_path())))
    app.setStyle("Fusion")
    app.setFont(choose_font())
    window = UploadWindow(directory, demo=demo, preferences=preferences, config_path=config_path)
    window.show()
    app.exec_()
