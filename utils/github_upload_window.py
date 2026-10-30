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

"""CustomTkinter desktop interface for the standalone GitHub upload helper."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, font
from typing import Any

import customtkinter as ctk

if __package__:
    from .github_upload_gui import AUTH_HELP, GitRepo, Preview, UploadError, UploadSettings, normalize_remote
else:
    from github_upload_gui import AUTH_HELP, GitRepo, Preview, UploadError, UploadSettings, normalize_remote

COLORS = {
    "window": "#070B12",
    "panel": "#0B111C",
    "card": "#111A29",
    "hover": "#192A42",
    "border": "#22324A",
    "track": "#202C3F",
    "text": "#F5F7FB",
    "muted": "#8E9AAF",
    "subtle": "#64758D",
    "blue": "#4C9BFF",
    "cyan": "#41D6E8",
    "green": "#48D597",
    "amber": "#FFB454",
    "red": "#FF6178",
}


class UploadWindow(ctk.CTk):
    """Present repository settings and run Git jobs without blocking the desktop."""

    def __init__(self, directory: Path, *, demo: bool = False) -> None:
        """Build the window; demo mode only displays a clearly marked sample."""
        super().__init__()
        self.title("GitHub Upload · 代码上传")
        self.geometry("1140x860")
        self.minsize(1010, 780)
        self.configure(fg_color=COLORS["window"])
        self.demo = demo
        self.busy = False
        self._closed = False
        self._poll_id: str | None = None
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._controls: list[Any] = []
        self._preview: Preview | None = None
        self._modal: ctk.CTkToplevel | None = None
        self._font_family = self._choose_font()
        self.values = {
            "directory": tk.StringVar(self, value=str(directory)),
            "remote": tk.StringVar(self),
            "branch": tk.StringVar(self),
            "message": tk.StringVar(self, value="Update local code"),
            "name": tk.StringVar(self),
            "email": tk.StringVar(self),
            "token": tk.StringVar(self),
        }
        self.create_ignore = tk.BooleanVar(self, value=True)
        self.code_only = tk.BooleanVar(self, value=True)
        self.show_token = tk.BooleanVar(self, value=False)
        self._build_header()
        self._build_steps()
        self._build_body()
        self._build_footer()
        self.values["directory"].trace_add("write", self._directory_changed)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Control-r>", lambda _event: self._start_preview())
        self._show_empty()
        if demo:
            self._load_demo()
        else:
            self._log("准备就绪。选择项目目录，点击「预览改动」开始。")
        self._poll_id = self.after(80, self._process_events)

    def _choose_font(self) -> str:
        available = set(font.families(self))
        for family in ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei"):
            if family in available:
                return family
        return "TkDefaultFont"

    def _font(self, size: int = 13, *, bold: bool = False) -> ctk.CTkFont:
        return ctk.CTkFont(family=self._font_family, size=size, weight="bold" if bold else "normal")

    def _label(self, parent: Any, text: str, *, size: int = 13, color: str = "text", **kwargs: Any) -> Any:
        return ctk.CTkLabel(
            parent,
            text=text,
            height=kwargs.pop("height", max(18, size + 7) * max(1, len(text.splitlines()))),
            text_color=COLORS[color],
            font=self._font(size),
            anchor="w",
            **kwargs,
        )

    def _button(
        self, parent: Any, text: str, command: Callable, *, primary: bool = False, **kwargs: Any
    ) -> Any:
        button = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=34,
            corner_radius=9,
            font=self._font(13, bold=primary),
            fg_color=COLORS["blue"] if primary else COLORS["card"],
            hover_color="#6AADFF" if primary else COLORS["hover"],
            text_color=COLORS["window"] if primary else COLORS["text"],
            text_color_disabled=COLORS["subtle"],
            border_width=0 if primary else 1,
            border_color=COLORS["border"],
            **kwargs,
        )
        self._controls.append(button)
        return button

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(20, 10))
        logo = ctk.CTkFrame(header, width=54, height=54, corner_radius=16, fg_color="#122840")
        logo.pack(side="left", padx=(0, 15))
        logo.pack_propagate(False)
        ctk.CTkLabel(
            logo, text="</>", font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["cyan"]
        ).pack(expand=True)
        titles = ctk.CTkFrame(header, fg_color="transparent")
        titles.pack(side="left")
        self._label(titles, "GitHub Upload", size=25).pack(anchor="w")
        self._label(titles, "代码上传   /   从本地项目到 GitHub", size=12, color="muted").pack(anchor="w")
        self.state_badge = ctk.CTkLabel(
            header,
            text="●  准备就绪",
            width=118,
            height=32,
            corner_radius=16,
            fg_color="#10271F",
            text_color=COLORS["green"],
            font=self._font(12),
        )
        self.state_badge.pack(side="right", padx=(15, 0))
        self._label(header, "LOCAL  →  GITHUB", size=11, color="subtle").pack(side="right")
        mode = ctk.CTkSwitch(
            header,
            text="仅上传当前代码",
            variable=self.code_only,
            command=self._mode_changed,
            progress_color=COLORS["blue"],
            text_color=COLORS["cyan"],
            font=self._font(12),
        )
        mode.pack(side="right", padx=(14, 25))
        self._controls.append(mode)

    def _build_steps(self) -> None:
        steps = ctk.CTkFrame(
            self, fg_color=COLORS["panel"], corner_radius=12, border_width=1, border_color=COLORS["border"]
        )
        steps.pack(fill="x", padx=28, pady=(0, 14))
        self.step_labels = []
        for index, title in enumerate(("选择项目", "检查改动", "提交并上传")):
            item = ctk.CTkFrame(steps, fg_color="transparent")
            item.pack(side="left", expand=True, pady=8)
            number = ctk.CTkLabel(
                item,
                text=f"0{index + 1}",
                width=29,
                height=28,
                corner_radius=8,
                fg_color="#13263C" if index == 0 else COLORS["card"],
                text_color=COLORS["cyan"] if index == 0 else COLORS["subtle"],
                font=self._font(11, bold=True),
            )
            number.pack(side="left", padx=(0, 10))
            label = self._label(item, title, size=12, color="text" if index == 0 else "muted")
            label.pack(side="left")
            self.step_labels.append((number, label))

    def _set_step(self, active: int) -> None:
        for index, (number, label) in enumerate(self.step_labels):
            number.configure(
                fg_color="#13263C" if index == active else COLORS["card"],
                text_color=COLORS["cyan"] if index == active else COLORS["subtle"],
            )
            label.configure(text_color=COLORS["text"] if index == active else COLORS["muted"])

    def _card(self, parent: Any, number: str, title: str, subtitle: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color=COLORS["panel"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        card.pack(fill="x", pady=(0, 12), padx=(0, 6))
        heading = ctk.CTkFrame(card, fg_color="transparent")
        heading.pack(fill="x", padx=18, pady=(12, 0))
        self._label(heading, number, size=11, color="cyan").pack(side="left", padx=(0, 10))
        self._label(heading, title, size=16).pack(side="left")
        self._label(card, subtitle, size=11, color="muted").pack(anchor="w", padx=18, pady=(1, 8))
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=18, pady=(0, 14))
        return inner

    def _entry(self, parent: Any, key: str, *, placeholder: str = "", secret: bool = False) -> Any:
        entry = ctk.CTkEntry(
            parent,
            textvariable=self.values[key],
            height=34,
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            fg_color=COLORS["card"],
            text_color=COLORS["text"],
            placeholder_text=placeholder,
            placeholder_text_color=COLORS["subtle"],
            font=self._font(12),
            show="•" if secret else "",
        )
        self._controls.append(entry)
        return entry

    def _field(self, parent: Any, key: str, title: str, *, placeholder: str = "") -> Any:
        self._label(parent, title, size=12, color="muted").pack(anchor="w", pady=(0, 4))
        entry = self._entry(parent, key, placeholder=placeholder)
        entry.pack(fill="x", pady=(0, 8))
        return entry

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=28)
        body.grid_columnconfigure(0, weight=1, uniform="body")
        body.grid_columnconfigure(1, weight=1, uniform="body")
        body.grid_rowconfigure(0, weight=1)
        form = ctk.CTkScrollableFrame(
            body,
            fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["subtle"],
            corner_radius=0,
        )
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 13))
        project = self._card(form, "01", "项目与目标", "选择项目根目录，连接你的 GitHub 仓库")
        self._label(project, "本地项目目录", size=12, color="muted").pack(anchor="w", pady=(0, 4))
        directory_row = ctk.CTkFrame(project, fg_color="transparent")
        directory_row.pack(fill="x", pady=(0, 8))
        self._entry(directory_row, "directory").pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._button(directory_row, "浏览…", self._browse, width=68).pack(side="right")
        self._field(project, "remote", "GitHub 仓库地址", placeholder="https://github.com/用户名/仓库.git")
        branch_row = ctk.CTkFrame(project, fg_color="transparent")
        branch_row.pack(fill="x")
        branch_field = ctk.CTkFrame(branch_row, fg_color="transparent")
        branch_field.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self._label(branch_field, "目标分支", size=12, color="muted").pack(anchor="w", pady=(0, 4))
        self._entry(branch_field, "branch", placeholder="预览后读取，例如 main").pack(fill="x")
        self._button(
            branch_row, "新建 GitHub 仓库 ↗", lambda: webbrowser.open("https://github.com/new"), width=150
        ).pack(side="right", anchor="s")
        self._label(project, "支持 HTTPS / SSH · 请先在 GitHub 创建目标仓库", size=10, color="subtle").pack(
            anchor="w", pady=(8, 0)
        )

        commit = self._card(form, "02", "提交与身份", "描述本次更新，使用已有 Git 凭据或临时 Token")
        self._field(commit, "message", "提交说明", placeholder="例如：Add robot recording scripts")
        identity = ctk.CTkFrame(commit, fg_color="transparent")
        identity.pack(fill="x")
        identity.grid_columnconfigure((0, 1), weight=1)
        for column, (key, label) in enumerate((("name", "提交者姓名"), ("email", "提交者邮箱"))):
            field = ctk.CTkFrame(identity, fg_color="transparent")
            field.grid(row=0, column=column, sticky="ew", padx=(0, 6) if column == 0 else (6, 0))
            self._field(field, key, label)
        auth_heading = ctk.CTkFrame(commit, fg_color="transparent")
        auth_heading.pack(fill="x", pady=(0, 4))
        self._label(auth_heading, "访问 Token", size=12, color="muted").pack(side="left")
        self._label(auth_heading, "可选 · 仅用于本次上传", size=10, color="subtle").pack(side="right")
        auth = ctk.CTkFrame(commit, fg_color="transparent")
        auth.pack(fill="x")
        self.token_entry = self._entry(auth, "token", secret=True)
        self.token_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.token_toggle = self._button(auth, "显示", self._toggle_token, width=58)
        self.token_toggle.pack(side="right")
        auth_help = ctk.CTkFrame(commit, fg_color="transparent")
        auth_help.pack(fill="x", pady=(6, 8))
        self._label(auth_help, "留空使用 Git / SSH 凭据", size=10, color="subtle").pack(side="left")
        help_button = ctk.CTkButton(
            auth_help,
            text="Token 使用说明 ↗",
            command=lambda: webbrowser.open(AUTH_HELP),
            width=114,
            height=22,
            fg_color="transparent",
            hover_color=COLORS["card"],
            text_color=COLORS["blue"],
            font=self._font(10),
        )
        help_button.pack(side="right")
        self._controls.append(help_button)
        checkbox = ctk.CTkCheckBox(
            commit,
            text="缺少 .gitignore 时创建基础忽略规则",
            variable=self.create_ignore,
            checkbox_width=17,
            checkbox_height=17,
            corner_radius=5,
            border_width=1,
            border_color=COLORS["border"],
            fg_color=COLORS["blue"],
            hover_color=COLORS["hover"],
            checkmark_color=COLORS["window"],
            text_color=COLORS["muted"],
            font=self._font(11),
        )
        checkbox.pack(anchor="w")
        self._controls.append(checkbox)

        review = ctk.CTkFrame(
            body,
            fg_color=COLORS["panel"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        review.grid(row=0, column=1, sticky="nsew", pady=(0, 2))
        review_header = ctk.CTkFrame(review, fg_color="transparent")
        review_header.pack(fill="x", padx=20, pady=(16, 5))
        self._label(review_header, "改动预览", size=17).pack(side="left")
        self.file_badge = self._label(review_header, "等待检查", size=11, color="muted")
        self.file_badge.pack(side="right")
        self.branch_label = self._label(review, "本地分支  —     /     当前提交  —", size=11, color="subtle")
        self.branch_label.pack(anchor="w", padx=20, pady=(0, 12))
        counters = ctk.CTkFrame(review, fg_color="transparent")
        counters.pack(fill="x", padx=18, pady=(0, 10))
        self.counter_labels: dict[str, Any] = {}
        self.counter_titles: dict[str, Any] = {}
        for index, (kind, title, color) in enumerate(
            (("add", "新增", "green"), ("modify", "修改", "blue"), ("delete", "删除", "red"))
        ):
            counters.grid_columnconfigure(index, weight=1, uniform="counts")
            block = ctk.CTkFrame(counters, corner_radius=10, fg_color=COLORS["card"])
            block.grid(row=0, column=index, sticky="ew", padx=(0, 8) if index < 2 else 0)
            row = ctk.CTkFrame(block, fg_color="transparent")
            row.pack(padx=13, pady=8)
            value = self._label(row, "—", size=24, color=color)
            value.pack(side="left", padx=(0, 10))
            self.counter_labels[kind] = value
            counter_title = self._label(row, title, size=11, color="muted")
            counter_title.pack(side="left")
            self.counter_titles[kind] = counter_title
        self.tabs = ctk.CTkTabview(
            review,
            fg_color=COLORS["panel"],
            corner_radius=10,
            border_width=0,
            segmented_button_fg_color=COLORS["card"],
            segmented_button_selected_color=COLORS["border"],
            segmented_button_selected_hover_color=COLORS["hover"],
            segmented_button_unselected_color=COLORS["card"],
            segmented_button_unselected_hover_color=COLORS["hover"],
            text_color=COLORS["text"],
            anchor="w",
        )
        self.tabs.pack(fill="both", expand=True, padx=13, pady=(0, 8))
        self.tabs.add("文件改动")
        self.tabs.add("操作日志")
        self.tabs._segmented_button.configure(font=self._font(12))
        self.file_list = ctk.CTkScrollableFrame(
            self.tabs.tab("文件改动"),
            fg_color=COLORS["panel"],
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["subtle"],
        )
        self.file_list.pack(fill="both", expand=True)
        self.log_box = ctk.CTkTextbox(
            self.tabs.tab("操作日志"),
            fg_color=COLORS["panel"],
            text_color=COLORS["muted"],
            font=self._font(12),
            wrap="word",
            border_width=0,
            scrollbar_button_color=COLORS["border"],
            state="disabled",
        )
        self.log_box.pack(fill="both", expand=True)
        note = ctk.CTkFrame(review, fg_color=COLORS["card"], corner_radius=9)
        note.pack(fill="x", padx=18, pady=(0, 18))
        self.review_note = self._label(
            note,
            "仅上传当前代码，跳过历史和 LFS 数据。\n本地开发历史保留在原项目中。",
            size=11,
            color="muted",
            justify="left",
        )
        self.review_note.pack(anchor="w", padx=12, pady=10)

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=28, pady=(14, 18))
        actions = ctk.CTkFrame(footer, fg_color="transparent")
        actions.pack(side="right", padx=(14, 0))
        self.preview_button = self._button(actions, "预览改动", self._start_preview, width=112)
        self.preview_button.pack(side="left", padx=(0, 10))
        self.upload_button = self._button(
            actions, "提交并上传  →", lambda: self._start_preview(upload=True), primary=True, width=174
        )
        self.upload_button.configure(height=44)
        self.upload_button.pack(side="left")
        status_frame = ctk.CTkFrame(footer, fg_color="transparent")
        status_frame.pack(side="left", fill="x", expand=True)
        self.status_label = self._label(status_frame, "从一个本地项目开始", size=12)
        self.status_label.pack(anchor="w")
        self.status_detail = self._label(
            status_frame, "首次预览会初始化 Git；上传前会再次确认目标与文件。", size=10, color="muted"
        )
        self.status_detail.pack(anchor="w", pady=(2, 0))
        self.progress = ctk.CTkProgressBar(
            status_frame,
            height=3,
            fg_color=COLORS["track"],
            progress_color=COLORS["blue"],
            mode="indeterminate",
        )

    def _set_status(self, title: str, detail: str = "", *, color: str = "text") -> None:
        self.status_label.configure(text=title, text_color=COLORS[color])
        self.status_detail.configure(text=detail)

    def _directory_changed(self, *_args: Any) -> None:
        for key in ("remote", "branch", "name", "email"):
            self.values[key].set("")
        self._clear_token()
        self._preview = None
        self._show_empty()
        self._set_step(0)
        self._set_status("项目目录已更改", "点击预览，重新读取当前项目的分支、仓库与提交者信息。")
        self.state_badge.configure(text="●  等待预览", text_color=COLORS["muted"], fg_color=COLORS["card"])

    def _mode_changed(self) -> None:
        self._preview = None
        self._show_empty()
        self.review_note.configure(
            text=(
                "仅上传当前代码，跳过历史和 LFS 数据。\n本地开发历史保留在原项目中。"
                if self.code_only.get()
                else "上传完整提交历史及其 LFS 数据。\n已有暂存内容也会一起提交。"
            )
        )
        self._set_status("上传模式已更改", "请重新预览本次要上传的文件。")

    def _browse(self) -> None:
        selected = filedialog.askdirectory(parent=self, title="选择要上传的项目根目录")
        if selected and selected != self.values["directory"].get():
            self.values["directory"].set(selected)

    def _toggle_token(self) -> None:
        self.show_token.set(not self.show_token.get())
        self.token_entry.configure(show="" if self.show_token.get() else "•")
        self.token_toggle.configure(text="隐藏" if self.show_token.get() else "显示")

    def _clear_token(self) -> None:
        self.values["token"].set("")
        self.show_token.set(False)
        self.token_entry.configure(show="•")
        self.token_toggle.configure(text="显示")

    def _show_empty(self) -> None:
        for child in self.file_list.winfo_children():
            child.destroy()
        for label in self.counter_labels.values():
            label.configure(text="—")
        self.file_badge.configure(text="等待检查")
        self.branch_label.configure(text="本地分支  —     /     当前提交  —")
        self._label(self.file_list, "{  }", size=38, color="subtle").pack(pady=(62, 12))
        self._label(self.file_list, "先看看本次有哪些变化", size=15).pack()
        self._label(
            self.file_list,
            "选择项目并点击「预览改动」\n新增、修改、删除的文件将在这里显示",
            size=11,
            color="muted",
            justify="center",
        ).pack(pady=(8, 40))

    def _render_preview(self, state: Preview) -> None:
        self._preview = state
        for child in self.file_list.winfo_children():
            child.destroy()
        lines = state.status.splitlines()
        counts = {"add": 0, "modify": 0, "delete": 0}
        self.file_badge.configure(text=f"{len(lines)} 个文件")
        self.review_note.configure(
            text=(
                f"当前代码快照 · 跳过 {state.excluded_lfs} 个 LFS 资源。\n后续上传将更新个人仓库中的代码版本。"
                if state.code_only
                else "上传完整提交历史，包括已有暂存内容。\n历史中的 LFS 对象也需要一并提供。"
            )
        )
        self.branch_label.configure(
            text=f"本地分支  {state.branch}     /     当前提交  {state.head[:8] or '尚无'}"
        )
        for index, line in enumerate(lines):
            code = line[:2]
            kind = "delete" if "D" in code else "add" if code == "??" or "A" in code else "modify"
            counts[kind] += 1
            color = {"add": "green", "modify": "blue", "delete": "red"}[kind]
            title = {"add": "新增", "modify": "修改", "delete": "删除"}[kind]
            if "R" in code:
                title = "重命名"
            elif "U" in code:
                title, color = "冲突", "amber"
            if state.code_only:
                title, color = "代码", "blue"
            if index >= 120:
                continue
            row = ctk.CTkFrame(self.file_list, fg_color=COLORS["card"], corner_radius=8)
            row.pack(fill="x", pady=(0, 6))
            marker = ctk.CTkLabel(
                row,
                text=title,
                width=48,
                height=18,
                text_color=COLORS[color],
                font=self._font(10),
                anchor="w",
            )
            marker.pack(side="left", padx=(11, 5), pady=9)
            self._label(row, line[3:], size=11, wraplength=330, justify="left").pack(
                side="left", fill="x", expand=True, padx=(0, 10), pady=9
            )
        titles = {"add": "新增", "modify": "修改", "delete": "删除"}
        if state.code_only:
            counts = {"add": len(lines), "modify": state.excluded_lfs, "delete": 0}
            titles = {"add": "文件", "modify": "跳过 LFS", "delete": "旧历史"}
        for key, value in counts.items():
            self.counter_labels[key].configure(text=str(value))
            self.counter_titles[key].configure(text=titles[key])
        if len(lines) > 120:
            self._label(
                self.file_list,
                f"已显示前 120 个，共 {len(lines)} 个文件将纳入本次上传。",
                size=11,
                color="muted",
                wraplength=360,
            ).pack(anchor="w", pady=10)
        if not lines:
            self._label(self.file_list, "✓", size=38, color="green").pack(pady=(58, 10))
            self._label(self.file_list, "工作区很整洁", size=15).pack()
            self._label(self.file_list, "没有待提交改动，可上传已有提交。", size=11, color="muted").pack(
                pady=(8, 0)
            )

    def _settings(self) -> UploadSettings:
        values = {key: variable.get().strip() for key, variable in self.values.items()}
        if not values["directory"]:
            raise UploadError("请选择本地项目目录。")
        values["directory"] = Path(values["directory"])
        values["branch"] = values["branch"] or "main"
        return UploadSettings(
            **values, create_ignore=self.create_ignore.get(), code_only=self.code_only.get()
        )

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        for control in self._controls:
            control.configure(state="disabled" if busy or self.demo else "normal")
        if busy:
            self.progress.pack(fill="x", pady=(7, 0))
            self.progress.start()
            self.state_badge.configure(text="●  正在处理", text_color=COLORS["cyan"], fg_color="#112736")
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _background(self, job: Callable, event: str) -> None:
        self._set_busy(True)

        def worker() -> None:
            try:
                self._events.put((event, job()))
            except Exception as exc:  # Ensure all worker failures restore usable controls.
                self._events.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _start_preview(self, upload: bool = False) -> None:
        if self.busy or self.demo or self._modal is not None:
            return
        try:
            selected = self._settings()
        except UploadError as exc:
            self._clear_token()
            self._dialog("请检查输入", str(exc), tone="amber")
            return
        self._set_step(1)
        self._set_status("正在读取项目改动…", "检查本地仓库、当前分支和待提交文件。")

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
                value = ""  # A code snapshot should target a personal repository supplied by the user.
            if source == "origin" and value:
                try:
                    value = normalize_remote(value)
                except UploadError:
                    value = ""
            if not self.values[target].get():
                self.values[target].set(value)
        self._render_preview(state)
        self.tabs.set("文件改动")
        mode = "当前代码快照" if state.code_only else "完整历史"
        self._log(f"预览完成：{mode}，共 {len(state.status.splitlines())} 个文件。")
        self.state_badge.configure(text="●  预览已更新", text_color=COLORS["cyan"], fg_color="#112736")
        self._set_status("预览已更新，可以准备上传", "检查仓库地址、目标分支与右侧文件列表。")
        if upload:
            try:
                selected = self._settings()
                normalize_remote(selected.remote)
                if not selected.message:
                    raise UploadError("请填写提交说明。")
                if not self.values["branch"].get().strip():
                    raise UploadError("请填写 GitHub 目标分支。")
                if selected.token and not selected.remote.startswith("https://"):
                    raise UploadError("Token 仅用于 HTTPS 地址。使用 SSH 地址时请清空 Token。")
                if state.status and (not selected.name or not selected.email):
                    raise UploadError("请填写提交者姓名和邮箱，或先配置 Git 提交身份。")
            except UploadError as exc:
                self._clear_token()
                self._dialog("请检查上传信息", str(exc), tone="amber")
                return
            self._confirm_upload(selected, state)

    def _confirm_upload(self, selected: UploadSettings, state: Preview) -> None:
        action = (
            f"仅上传当前代码，跳过 {state.excluded_lfs} 个 LFS 资源。\n"
            "首次生成独立版本，后续更新接在个人仓库版本之后。\n原项目的开发历史保留在本地。"
            if selected.code_only
            else "将提交预览中的全部改动（含已有暂存内容），\n并推送当前本地分支的提交历史。"
        )
        summary = (
            f"本地项目\n{selected.directory}\n\nGitHub 目标\n{selected.remote}\n\n"
            f"目标分支：{selected.branch}    ·    {len(state.status.splitlines())} 个文件改动\n"
            f"提交说明：{selected.message}\n\n" + action
        )

        def confirm() -> None:
            self._set_step(2)
            self._set_status("正在提交并上传…", "请保持窗口打开；Git 完成后会显示上传结果。")
            self.tabs.set("操作日志")

            def job() -> str:
                repo = GitRepo(selected.directory, lambda text: self._events.put(("log", text)))
                return repo.upload(selected, expected=state)

            self._background(job, "success")

        self._dialog("确认上传到 GitHub", summary, confirm=confirm, tone="blue")

    def _dialog(
        self, title: str, message: str, *, confirm: Callable | None = None, tone: str = "blue"
    ) -> None:
        if self._modal is not None:
            self._modal.lift()
            return
        modal = ctk.CTkToplevel(self)
        self._modal = modal
        modal.withdraw()
        modal.title(title)
        modal.configure(fg_color=COLORS["window"])
        width, height = (610, 520) if confirm else (570, 370)
        x = self.winfo_rootx() + max(0, (self.winfo_width() - width) // 2)
        y = self.winfo_rooty() + max(0, (self.winfo_height() - height) // 2)
        modal.geometry(f"{width}x{height}+{x}+{y}")
        modal.minsize(width, height)
        modal.transient(self)

        def dismiss() -> None:
            modal.grab_release()
            self._modal = None
            modal.destroy()

        def activate() -> None:
            if self._modal is modal:
                modal.lift()
                modal.focus_set()
                modal.grab_set()

        surface = ctk.CTkFrame(
            modal,
            fg_color=COLORS["window"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        surface.pack(fill="both", expand=True)
        heading = ctk.CTkFrame(surface, fg_color="transparent")
        heading.pack(fill="x", padx=26, pady=(18, 4))
        self._label(heading, "GITHUB UPLOAD", size=11, color=tone).pack(side="left")
        ctk.CTkButton(
            heading,
            text="×",
            command=dismiss,
            width=30,
            height=30,
            corner_radius=8,
            fg_color=COLORS["card"],
            hover_color=COLORS["hover"],
            text_color=COLORS["muted"],
            font=self._font(20),
        ).pack(side="right")
        self._label(surface, title, size=22).pack(anchor="w", padx=26, pady=(0, 18))
        text = ctk.CTkTextbox(
            surface,
            fg_color=COLORS["panel"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            font=self._font(13),
            wrap="word",
        )
        text.pack(fill="both", expand=True, padx=26, pady=(0, 20))
        text.insert("1.0", message)
        text.configure(state="disabled")
        buttons = ctk.CTkFrame(surface, fg_color="transparent")
        buttons.pack(fill="x", padx=26, pady=(0, 24))

        def accept() -> None:
            dismiss()
            if confirm:
                confirm()

        accept_button = ctk.CTkButton(
            buttons,
            text="确认提交并上传" if confirm else "知道了",
            command=accept,
            height=40,
            corner_radius=9,
            fg_color=COLORS["blue"],
            hover_color="#6AADFF",
            text_color=COLORS["window"],
            font=self._font(13, bold=True),
        )
        accept_button.pack(side="right")
        if confirm:
            ctk.CTkButton(
                buttons,
                text="返回检查",
                command=dismiss,
                height=40,
                width=110,
                corner_radius=9,
                fg_color=COLORS["card"],
                hover_color=COLORS["hover"],
                text_color=COLORS["text"],
                font=self._font(13),
            ).pack(side="right", padx=(0, 10))
        modal.protocol("WM_DELETE_WINDOW", dismiss)
        modal.bind("<Escape>", lambda _event: dismiss())
        modal.overrideredirect(True)
        modal.deiconify()
        self.after(100, activate)

    def _log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{datetime.now():%H:%M:%S}   {text}\n\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _process_events(self) -> None:
        if self._closed:
            return
        try:
            while True:
                event, payload = self._events.get_nowait()
                if event == "log":
                    self._log(payload)
                    continue
                self._set_busy(False)
                if event == "preview":
                    self._preview_ready(*payload)
                elif event == "error":
                    self._clear_token()
                    self._log(payload)
                    self.tabs.set("操作日志")
                    self.state_badge.configure(
                        text="●  操作未完成", text_color=COLORS["red"], fg_color="#311B28"
                    )
                    self._set_status(
                        "操作未完成，请查看原因", "日志已保留详细信息，修复后可重新预览。", color="red"
                    )
                    self._dialog("操作未完成", payload, tone="red")
                elif event == "success":
                    self._clear_token()
                    self._log(f"上传成功，提交：{payload}")
                    self.state_badge.configure(
                        text="●  上传成功", text_color=COLORS["green"], fg_color="#10271F"
                    )
                    self._set_status("代码已成功上传到 GitHub", f"提交 {payload[:12]}", color="green")
                    self._dialog(
                        "上传成功", f"代码已推送到指定的 GitHub 仓库和分支。\n\n提交：{payload}", tone="green"
                    )
        except queue.Empty:
            pass
        finally:
            if not self._closed:
                self._poll_id = self.after(80, self._process_events)

    def _load_demo(self) -> None:
        self.values["directory"].set("/home/yzh/projects/robot-learning")
        for key, value in {
            "remote": "https://github.com/yzh/robot-learning.git",
            "branch": "main",
            "message": "Add SO-101 recording workflow",
            "name": "yzh",
            "email": "yzh@example.com",
        }.items():
            self.values[key].set(value)
        self._render_preview(
            Preview(
                branch="main",
                head="a73c9d1249b6",
                status=(
                    "?? README.md\n?? src/robot/config.py\n?? scripts/train_policy.py\n"
                    "?? examples/record_episode.py\n?? configs/so101.yaml\n?? utils/github_upload_gui.py"
                ),
                origin=self.values["remote"].get(),
                name="yzh",
                email="yzh@example.com",
                code_only=True,
                excluded_lfs=3,
            )
        )
        self._set_step(1)
        self._set_busy(False)
        self.state_badge.configure(text="●  界面演示", text_color=COLORS["amber"], fg_color="#30261B")
        self.upload_button.configure(text="演示模式 · 禁止上传")
        self._set_status("界面演示 · 当前内容为示例", "不会读取、修改本地仓库或连接 GitHub。", color="amber")
        self._log("界面演示：文件与提交信息均为示例，没有执行 Git 操作。")

    def _close(self) -> None:
        if self.busy:
            self._set_status(
                "Git 操作仍在进行，请稍候再关闭",
                "完成后可关闭窗口，避免中断正在进行的提交或上传。",
                color="amber",
            )
            self.bell()
            return
        self._closed = True
        self._clear_token()
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.destroy()


def launch(directory: Path, *, demo: bool = False) -> None:
    """Open the styled desktop window with optional network-free sample content."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    try:
        window = UploadWindow(directory, demo=demo)
    except tk.TclError as exc:
        raise UploadError(f"无法打开图形窗口，请在有桌面显示的终端运行。\n{exc}") from exc
    window.mainloop()
