#!/usr/bin/env python

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
"""Review every camera of a local LeRobot episode in one desktop window.

Run ``uv run --no-sync python -m lerobot.yzh.lerobot_review_dataset`` and choose a dataset
folder in the CustomTkinter folder picker. Mark episodes for deletion, then
save a filtered copy. The original dataset stays in place.
"""

from __future__ import annotations

import contextlib
import re
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from tkinter import font as tkfont

try:
    import customtkinter as ctk
except ModuleNotFoundError:  # pragma: no cover - exercised only without the optional GUI dependency
    ctk = None

import av
from PIL import Image

from lerobot.datasets import LeRobotDataset, delete_episodes
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME

SPEEDS = ("0.25×", "0.5×", "1×", "1.5×", "2×", "3×", "4×")
PLAYBACK_MODES = ("播完暂停", "单集循环", "连续播放")
TICK_MS = 33
EPISODES_PER_PAGE = 10


@dataclass(frozen=True)
class LocalDataset:
    """Metadata and filesystem location for a selected local dataset."""

    repo_id: str
    root: Path
    meta: LeRobotDatasetMetadata


def is_dataset_root(path: Path) -> bool:
    """Check for the metadata and data folders required by this viewer."""
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def infer_repo_id(root: Path) -> str:
    """Derive an identifier for local metadata and the filtered copy."""
    root = root.expanduser().resolve()
    try:
        relative = root.relative_to(HF_LEROBOT_HOME.expanduser().resolve())
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) == 2:
        return relative.as_posix()
    return f"{root.parent.name}/{root.name}"


def load_local_dataset(root: Path) -> LocalDataset:
    """Read local dataset metadata without loading every frame."""
    root = root.expanduser().resolve()
    if not is_dataset_root(root):
        raise ValueError(f"请选择包含 meta/info.json 和 data/ 的 LeRobot 数据集目录：\n{root}")
    repo_id = infer_repo_id(root)
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    if meta.total_episodes < 1:
        raise ValueError("这个数据集没有 episode。")
    if not meta.video_keys:
        raise ValueError("这个数据集没有视频摄像头。")
    return LocalDataset(repo_id, root, meta)


def video_segment(dataset: LocalDataset, episode_index: int, camera_key: str) -> tuple[Path, float, float]:
    """Locate an episode's section of a camera MP4, including shared files."""
    meta = dataset.meta
    if not 0 <= episode_index < meta.total_episodes:
        raise ValueError(f"Episode 索引必须在 0 到 {meta.total_episodes - 1} 之间。")
    if camera_key not in meta.video_keys:
        raise ValueError(f"摄像头 {camera_key!r} 不存在。")
    episode = meta.episodes[episode_index]
    start = float(episode[f"videos/{camera_key}/from_timestamp"])
    end = float(episode[f"videos/{camera_key}/to_timestamp"])
    if start < 0 or end <= start:
        raise ValueError(f"Episode {episode_index} 的 {camera_key} 视频时间范围无效。")
    path = dataset.root / meta.get_video_file_path(episode_index, camera_key)
    if not path.is_file():
        raise FileNotFoundError(f"找不到视频文件：{path}")
    return path, start, end - start


class EpisodeVideoDecoder:
    """Seek and decode one camera against an episode-relative time line."""

    def __init__(self, path: Path, start_s: float, duration_s: float, fps: int):
        """Open an MP4 and position it at one episode's first frame."""
        self.start_s = start_s
        self.duration_s = duration_s
        self.fps = fps
        self.container = av.open(str(path))
        try:
            self.stream = self.container.streams.video[0]
        except IndexError:
            self.container.close()
            raise ValueError(f"视频文件没有图像流：{path}") from None
        self._frames = None
        self._next = None
        self._current = None
        self._cached_pts = None
        self._cached_image = None
        self._last_target = -1.0
        self.seek(0.0)

    def _absolute_time(self, relative_s: float) -> float:
        last_frame_s = max(0.0, self.duration_s - 1 / self.fps)
        return self.start_s + min(max(relative_s, 0.0), last_frame_s)

    @staticmethod
    def _frame_time(frame: av.VideoFrame) -> float:
        if frame.time is not None:
            return float(frame.time)
        return float(frame.pts * frame.time_base)

    def seek(self, relative_s: float) -> None:
        """Move to a time relative to the start of the selected episode."""
        target = self._absolute_time(relative_s)
        self.container.seek(int(target / float(self.stream.time_base)), stream=self.stream, backward=True)
        self._frames = iter(self.container.decode(self.stream))
        self._next = next(self._frames, None)
        self._current = None
        self._cached_pts = None
        self._cached_image = None
        self._last_target = target

    def frame_at(self, relative_s: float) -> Image.Image:
        """Return the frame aligned with an episode-relative time."""
        target = self._absolute_time(relative_s)
        if target < self._last_target - 1e-6:
            self.seek(relative_s)
        self._last_target = target
        limit = target + 0.5 / self.fps
        while self._next is not None and self._frame_time(self._next) <= limit:
            self._current = self._next
            self._next = next(self._frames, None)
        if self._current is None and self._next is not None:
            self._current = self._next
            self._next = next(self._frames, None)
        if self._current is None:
            raise RuntimeError("无法解码当前 episode 的视频帧。")
        if self._current.pts != self._cached_pts or self._cached_image is None:
            self._cached_image = self._current.to_image()
            self._cached_pts = self._current.pts
        return self._cached_image

    def close(self) -> None:
        """Release the video decoder and its file handle."""
        self.container.close()


def default_output_path(source: Path, parent: Path | None = None) -> Path:
    """Suggest an unused sibling directory for the filtered dataset."""
    parent = source.parent if parent is None else parent
    base = f"{source.name}_filtered"
    candidate = parent / base
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{base}_{suffix}"
        suffix += 1
    return candidate


def check_output_path(source: Path, output: Path) -> None:
    """Reject output paths that exist or overlap the source dataset."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请换一个目录：\n{output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("输出目录不能是源数据集目录，也不能包含源数据集或位于其内部。")


def output_repo_id(source_repo_id: str, output: Path) -> str:
    """Construct the local output identifier from its folder name."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output.name):
        raise ValueError("输出目录名称只能包含英文字母、数字、点、下划线和连字符。")
    namespace = source_repo_id.split("/", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", namespace):
        namespace = "local"
    return f"{namespace}/{output.name}"


def update_deletions(deletions: set[int], episode_index: int, total_episodes: int, delete: bool) -> set[int]:
    """Apply one review decision while preserving at least one episode."""
    if not 0 <= episode_index < total_episodes:
        raise ValueError("Episode 索引超出范围。")
    updated = deletions.copy()
    if delete:
        updated.add(episode_index)
        if len(updated) >= total_episodes:
            raise ValueError("数据集至少需要保留一个 episode。")
    else:
        updated.discard(episode_index)
    return updated


def save_filtered_dataset(
    dataset: LeRobotDataset, deletions: set[int], new_repo_id: str, target: Path
) -> Path:
    """Write a filtered copy in a temporary folder, then move it into place."""
    if not deletions:
        raise ValueError("还没有标记要删除的 episode。")
    if len(deletions) >= dataset.meta.total_episodes:
        raise ValueError("不能删除数据集中的全部 episode。")
    check_output_path(Path(dataset.root), target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lerobot-review-", dir=target.parent) as temporary:
        staged = Path(temporary) / "dataset"
        delete_episodes(dataset, sorted(deletions), output_dir=staged, repo_id=new_repo_id)
        check_output_path(Path(dataset.root), target)
        staged.rename(target)
    return target


UI_COLORS = {
    "page": ("#eef2f7", "#0b1220"),
    "surface": ("#ffffff", "#111827"),
    "surface_alt": ("#f8fafc", "#1f2937"),
    "border": ("#dbe3ef", "#334155"),
    "video": ("#e8eef7", "#0f172a"),
    "muted": ("#475569", "#cbd5e1"),
    "accent": ("#2563eb", "#3b82f6"),
    "accent_hover": ("#1d4ed8", "#2563eb"),
    "success": ("#15803d", "#22c55e"),
    "success_hover": ("#166534", "#16a34a"),
    "danger": ("#dc2626", "#ef4444"),
    "danger_hover": ("#b91c1c", "#dc2626"),
    "warning": ("#b45309", "#f59e0b"),
}


def _tk_xft_status() -> bool | None:
    """Report whether the active Python Tk library includes Xft support.

    Conda can install a ``noxft`` Tk build that silently renders every Tk
    font through the old X11 bitmap renderer.  In that build, changing the
    requested family cannot improve the glyphs.  Inspecting the shared
    library avoids showing a broken-looking panel and gives the user an
    actionable environment fix.  ``None`` means the library could not be
    identified, so the normal platform font handling is left untouched.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    candidates = [Path(sys.prefix) / "lib" / "libtk8.6.so"]
    candidates.extend(sorted((Path(sys.prefix) / "lib").glob("libtk*.so")))
    for library in dict.fromkeys(candidates):
        if not library.is_file():
            continue
        try:
            contents = library.read_bytes()
        except OSError:
            continue
        lowered = contents.lower()
        if b"no-xft" in lowered:
            return False
        if b"libxft.so" in lowered or b"xftfontopen" in lowered:
            return True
    return None


def _configure_ctk_rendering() -> None:
    """Prefer CustomTkinter's anti-aliased shape font for rounded controls."""
    if ctk is None:  # pragma: no cover - guarded by main
        return
    draw_engine = getattr(ctk, "DrawEngine", None)
    if draw_engine is None:  # pragma: no cover - defensive for older releases
        return

    font_manager = getattr(ctk, "FontManager", None)
    package_dir = Path(ctk.__file__).resolve().parent
    shapes_font = package_dir / "assets" / "fonts" / "CustomTkinter_shapes_font.otf"
    if font_manager is not None and shapes_font.is_file() and sys.platform.startswith("linux"):
        font_directories = (
            Path.home() / ".fonts",
            Path.home() / ".local" / "share" / "fonts",
        )
        for directory in font_directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                font_manager.linux_font_path = str(directory)
                if font_manager.load_font(str(shapes_font)):
                    draw_engine.preferred_drawing_method = "font_shapes"
                    return
            except OSError:
                continue

    # Circle shapes are slower than the font renderer, but their arcs are
    # visibly smoother than polygon corners when the optional font cannot be
    # installed (for example, in a locked-down home directory).
    draw_engine.preferred_drawing_method = "circle_shapes"


def _block_option_menu_mouse_events(option_menu: tk.Misc) -> None:
    """Prevent wheel and secondary-click events from changing menu values."""

    def swallow_event(_event: tk.Event) -> str:
        return "break"

    targets: list[tk.Misc] = [option_menu]
    dropdown = getattr(option_menu, "_dropdown_menu", None)
    if dropdown is not None:
        targets.append(dropdown)
    sequences = (
        "<MouseWheel>",
        "<Button-4>",
        "<Button-5>",
        "<ButtonRelease-4>",
        "<ButtonRelease-5>",
        "<Button-2>",
        "<ButtonRelease-2>",
        "<Button-3>",
        "<ButtonRelease-3>",
    )
    for target in targets:
        # Tk's native Menu puts the ``Menu`` class binding before the widget
        # binding. Move the widget tag first so returning ``break`` really
        # prevents the class handler from invoking an item.
        with contextlib.suppress(tk.TclError):
            tags = list(target.bindtags())
            widget_tag = str(target)
            if widget_tag in tags:
                tags.remove(widget_tag)
            target.bindtags((widget_tag, *tags))
        for sequence in sequences:
            with contextlib.suppress(tk.TclError):
                target.bind(sequence, swallow_event, add="+")


def choose_ui_font(root: tk.Misc) -> str:
    """Pick an installed font with complete Chinese glyph coverage.

    On some Linux Tk builds, ``font families`` does not report fonts that
    Fontconfig can already use.  We also measure a few glyphs because an
    unavailable family can silently resolve to Tk's old bitmap ``fixed``
    font even when the font file exists.
    """
    font_files = {
        "Noto Sans CJK SC": (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path.home() / ".local/share/fonts/NotoSansCJK-Regular.ttc",
        ),
        "Droid Sans Fallback": (Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),),
        "WenQuanYi Micro Hei": (Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),),
        "AR PL UMing CN": (Path("/usr/share/fonts/truetype/arphic/uming.ttc"),),
    }
    try:
        families = set(tkfont.families(root))
    except tk.TclError:
        families = set()
    candidates = (
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Droid Sans Fallback",
        "WenQuanYi Micro Hei",
        "AR PL UMing CN",
        "DejaVu Sans",
        "Liberation Sans",
    )
    usable: list[tuple[bool, bool, str]] = []
    for family in candidates:
        paths = font_files.get(family, ())
        if family not in families and not any(path.is_file() for path in paths):
            continue
        try:
            probe = tkfont.Font(root=root, family=family, size=16)
            narrow = probe.measure("iiii")
            wide = probe.measure("WWWW")
            cjk = probe.measure("中文")
            # ``tkinter.font.Font`` has no ``destroy`` method.  Temporary
            # named fonts are removed automatically when this object is
            # garbage-collected.
            del probe
        except tk.TclError:
            continue
        # Keep proportional fonts ahead of a fixed fallback.  The second
        # value records whether the font can draw Chinese glyphs.
        proportional = wide > narrow * 1.15
        has_cjk = cjk > 0
        usable.append((proportional, has_cjk, family))
        if proportional and has_cjk:
            return family
    for proportional, has_cjk, family in usable:
        if proportional and has_cjk:
            return family
    for proportional, _, family in usable:
        if proportional:
            return family
    try:
        default_family = str(tkfont.nametofont("TkDefaultFont", root=root).cget("family"))
        if default_family.casefold() in {"fixed", "monospace", "dejavu sans mono"}:
            return "DejaVu Sans"
        return default_family
    except tk.TclError:
        return "DejaVu Sans"


def choose_latin_font(root: tk.Misc) -> str:
    """Pick a proportional font for paths, numbers, and English labels."""
    try:
        families = set(tkfont.families(root))
    except tk.TclError:
        families = set()
    candidates = (
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans",
        "Noto Sans CJK SC",
        "AR PL UMing CN",
        "Droid Sans Fallback",
    )
    for family in candidates:
        if family not in families and family not in {"DejaVu Sans", "Liberation Sans"}:
            continue
        try:
            probe = tkfont.Font(root=root, family=family, size=16)
            narrow = probe.measure("iiii")
            wide = probe.measure("WWWW")
            del probe
        except tk.TclError:
            continue
        if wide > narrow * 1.15:
            return family
    return "DejaVu Sans"


class ReviewPanel:
    """Modern CustomTkinter review window with synchronized camera playback."""

    def __init__(self, root: tk.Misc):
        """Create the panel and start its playback timer."""
        if ctk is None:  # pragma: no cover - guarded by main, kept for direct construction
            raise RuntimeError("CustomTkinter is required for the review panel.")
        self.root = root
        self.font_family = choose_ui_font(root)
        self.font_latin_family = choose_latin_font(root)
        self.font_title = ctk.CTkFont(family=self.font_latin_family, size=28, weight="bold")
        self.font_section = ctk.CTkFont(family=self.font_family, size=19, weight="bold")
        self.font_body = ctk.CTkFont(family=self.font_family, size=16)
        self.font_small = ctk.CTkFont(family=self.font_family, size=14)
        self.font_button = ctk.CTkFont(family=self.font_family, size=15, weight="bold")
        self.font_latin = ctk.CTkFont(family=self.font_latin_family, size=16)
        self.root.configure(fg_color=UI_COLORS["page"])
        self.root.title("LeRobot Episode 审查")
        width = min(1560, max(960, root.winfo_screenwidth() - 80))
        height = min(980, max(700, root.winfo_screenheight() - 100))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(960, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.dataset: LocalDataset | None = None
        self.current_episode: int | None = None
        self.episode_page = 0
        self.episode_buttons: dict[int, object] = {}
        self.deletions: set[int] = set()
        self.saved_deletions: set[int] = set()
        self.dirty = False
        self.decoders: dict[str, EpisodeVideoDecoder] = {}
        self.image_labels: dict[str, object] = {}
        self.photos: dict[str, object] = {}
        self.position_s = 0.0
        self.duration_s = 0.0
        self.playing = False
        self.busy = False
        self.scrubbing = False
        self.resume_after_scrub = False
        self.last_tick = time.monotonic()
        self.resize_job: str | None = None

        self.dataset_var = tk.StringVar(value="未选择数据集 · 点击右上角“打开数据集”开始")
        self.episode_var = tk.StringVar()
        self.episode_range_var = tk.StringVar(value="/ —")
        self.episode_page_var = tk.StringVar(value="0 / 0 页")
        self.playback_mode_var = tk.StringVar(value="单集循环")
        self.speed_var = tk.StringVar(value="2×")
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        self.review_var = tk.StringVar(value="尚未开始审查")
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择包含 meta/info.json 的 LeRobot 数据集目录。")

        self._build_ui()
        self.root.after(100, self.open_dataset)
        self.root.after(TICK_MS, self._tick)

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 10))
        header.grid_columnconfigure(0, weight=1)

        title_block = ctk.CTkFrame(header, fg_color="transparent")
        title_block.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            title_block,
            text="LeRobot Episode Review",
            font=self.font_title,
            text_color=UI_COLORS["accent"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_block,
            text="逐集查看所有摄像头，标记需要删除的 Episode",
            font=self.font_small,
            text_color=UI_COLORS["muted"],
            anchor="w",
        ).pack(anchor="w", pady=(2, 3))
        ctk.CTkLabel(
            title_block,
            textvariable=self.dataset_var,
            font=self.font_latin,
            text_color=UI_COLORS["muted"],
            anchor="w",
        ).pack(anchor="w")

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.grid(row=0, column=1, sticky="e", padx=(18, 0))
        self.mode_var = tk.StringVar(value="深色")
        self.mode_menu = ctk.CTkOptionMenu(
            actions,
            values=["深色", "浅色"],
            variable=self.mode_var,
            command=self._appearance_changed,
            width=92,
            height=36,
            font=self.font_body,
            dropdown_font=self.font_body,
        )
        self.mode_menu.pack(side="left", padx=(0, 10))
        self.open_button = ctk.CTkButton(
            actions,
            text="打开数据集",
            command=self.open_dataset,
            width=140,
            height=38,
            corner_radius=10,
            font=self.font_button,
            fg_color=UI_COLORS["accent"],
            hover_color=UI_COLORS["accent_hover"],
        )
        self.open_button.pack(side="left")

        self.video_card = ctk.CTkFrame(
            self.root,
            fg_color=UI_COLORS["surface"],
            border_color=UI_COLORS["border"],
            border_width=1,
            corner_radius=16,
        )
        self.video_card.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 10))
        self.video_card.grid_rowconfigure(0, weight=1)
        self.video_card.grid_columnconfigure(0, weight=1)
        self.video_row = ctk.CTkFrame(self.video_card, fg_color="transparent")
        self.video_row.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.video_row.grid_rowconfigure(0, weight=1)
        self._show_video_placeholder()

        timeline_card = ctk.CTkFrame(self.root, fg_color="transparent")
        timeline_card.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 2))
        timeline_card.grid_columnconfigure(0, weight=1)
        self.timeline = ctk.CTkSlider(
            timeline_card,
            from_=0,
            to=1,
            height=18,
            corner_radius=9,
            button_length=18,
            fg_color=UI_COLORS["border"],
            progress_color=UI_COLORS["accent"],
            button_color=UI_COLORS["accent"],
            button_hover_color=UI_COLORS["accent_hover"],
            state="disabled",
            command=self._timeline_changed,
        )
        self.timeline.grid(row=0, column=0, sticky="ew", pady=(6, 0))
        self.timeline.bind("<ButtonPress-1>", self._begin_scrub)
        self.timeline.bind("<ButtonRelease-1>", self._end_scrub)
        ctk.CTkLabel(
            timeline_card,
            textvariable=self.time_var,
            width=120,
            font=self.font_latin,
            text_color=UI_COLORS["muted"],
            anchor="e",
        ).grid(row=0, column=1, padx=(14, 0))

        controls = ctk.CTkFrame(
            self.root,
            fg_color=UI_COLORS["surface"],
            border_color=UI_COLORS["border"],
            border_width=1,
            corner_radius=14,
        )
        controls.grid(row=3, column=0, sticky="ew", padx=24, pady=(2, 8))
        controls.grid_columnconfigure(0, weight=1)
        episode_group = ctk.CTkFrame(controls, fg_color="transparent")
        episode_group.grid(row=0, column=0, sticky="w", padx=14, pady=12)
        ctk.CTkLabel(episode_group, text="Episode", font=self.font_latin).pack(side="left", padx=(0, 10))
        self.episode_entry = ctk.CTkEntry(
            episode_group,
            textvariable=self.episode_var,
            width=70,
            height=34,
            corner_radius=9,
            state="disabled",
            font=self.font_latin,
            justify="center",
        )
        self.episode_entry.pack(side="left")
        self.episode_entry.bind("<Return>", self._jump_episode)
        ctk.CTkLabel(
            episode_group,
            textvariable=self.episode_range_var,
            font=self.font_latin,
            text_color=UI_COLORS["muted"],
        ).pack(side="left", padx=(6, 12))
        self.jump_button = ctk.CTkButton(
            episode_group,
            text="跳转",
            command=self._jump_episode,
            width=60,
            height=34,
            corner_radius=9,
            font=self.font_body,
            state="disabled",
        )
        self.jump_button.pack(side="left", padx=(0, 8))
        self.previous_button = ctk.CTkButton(
            episode_group,
            text="上一集",
            command=lambda: self._move_episode(-1),
            width=78,
            height=34,
            corner_radius=9,
            font=self.font_body,
            fg_color=UI_COLORS["surface_alt"],
            hover_color=UI_COLORS["border"],
            text_color=UI_COLORS["muted"],
        )
        self.previous_button.pack(side="left", padx=3)
        self.next_button = ctk.CTkButton(
            episode_group,
            text="下一集",
            command=lambda: self._move_episode(1),
            width=78,
            height=34,
            corner_radius=9,
            font=self.font_body,
            fg_color=UI_COLORS["surface_alt"],
            hover_color=UI_COLORS["border"],
            text_color=UI_COLORS["muted"],
        )
        self.next_button.pack(side="left", padx=3)

        playback_group = ctk.CTkFrame(controls, fg_color="transparent")
        playback_group.grid(row=0, column=1, sticky="e", padx=14, pady=12)
        self.play_button = ctk.CTkButton(
            playback_group,
            text="播放",
            command=self.toggle_play,
            width=100,
            height=36,
            corner_radius=10,
            font=self.font_button,
            fg_color=UI_COLORS["accent"],
            hover_color=UI_COLORS["accent_hover"],
        )
        self.play_button.pack(side="left")
        ctk.CTkLabel(playback_group, text="倍速", font=self.font_body, text_color=UI_COLORS["muted"]).pack(
            side="left", padx=(22, 8)
        )
        self.speed_menu = ctk.CTkOptionMenu(
            playback_group,
            values=list(SPEEDS),
            variable=self.speed_var,
            width=84,
            height=34,
            corner_radius=9,
            font=self.font_latin,
            dropdown_font=self.font_latin,
        )
        self.speed_menu.pack(side="left")
        _block_option_menu_mouse_events(self.speed_menu)

        episode_strip = ctk.CTkFrame(controls, fg_color="transparent")
        episode_strip.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
        episode_strip.grid_columnconfigure(0, weight=1)
        self.episode_buttons_frame = ctk.CTkFrame(episode_strip, fg_color="transparent")
        self.episode_buttons_frame.grid(row=0, column=0, sticky="ew")
        for column in range(EPISODES_PER_PAGE):
            self.episode_buttons_frame.grid_columnconfigure(column, weight=1, uniform="episode")
        page_group = ctk.CTkFrame(episode_strip, fg_color="transparent")
        page_group.grid(row=0, column=1, padx=(12, 0))
        self.previous_page_button = ctk.CTkButton(
            page_group,
            text="‹",
            command=lambda: self._move_episode_page(-1),
            width=32,
            height=30,
            font=self.font_latin,
            state="disabled",
        )
        self.previous_page_button.pack(side="left")
        ctk.CTkLabel(page_group, textvariable=self.episode_page_var, font=self.font_small, width=80).pack(
            side="left", padx=4
        )
        self.next_page_button = ctk.CTkButton(
            page_group,
            text="›",
            command=lambda: self._move_episode_page(1),
            width=32,
            height=30,
            font=self.font_latin,
            state="disabled",
        )
        self.next_page_button.pack(side="left")

        self.playback_modes = ctk.CTkSegmentedButton(
            controls,
            values=list(PLAYBACK_MODES),
            variable=self.playback_mode_var,
            command=self._playback_mode_changed,
            height=32,
            font=self.font_small,
        )
        self.playback_modes.grid(row=1, column=1, sticky="e", padx=14, pady=(0, 12))

        decisions = ctk.CTkFrame(self.root, fg_color="transparent")
        decisions.grid(row=4, column=0, sticky="ew", padx=24, pady=(0, 8))
        self.keep_button = ctk.CTkButton(
            decisions,
            text="保留当前 Episode",
            command=lambda: self.mark_current(False),
            width=154,
            height=38,
            corner_radius=10,
            font=self.font_button,
            fg_color=UI_COLORS["success"],
            hover_color=UI_COLORS["success_hover"],
            state="disabled",
        )
        self.keep_button.pack(side="left")
        self.delete_button = ctk.CTkButton(
            decisions,
            text="标记删除当前 Episode",
            command=lambda: self.mark_current(True),
            width=184,
            height=38,
            corner_radius=10,
            font=self.font_button,
            fg_color=UI_COLORS["danger"],
            hover_color=UI_COLORS["danger_hover"],
            state="disabled",
        )
        self.delete_button.pack(side="left", padx=(10, 0))
        self.save_button = ctk.CTkButton(
            decisions,
            text="生成过滤数据集",
            command=self.save,
            width=176,
            height=38,
            corner_radius=10,
            font=self.font_button,
            fg_color=UI_COLORS["accent"],
            hover_color=UI_COLORS["accent_hover"],
            state="disabled",
        )
        self.save_button.pack(side="right")

        self.deleted_card = ctk.CTkFrame(
            self.root,
            fg_color=UI_COLORS["surface"],
            border_color=UI_COLORS["border"],
            border_width=1,
            corner_radius=12,
        )
        self.deleted_card.grid(row=5, column=0, sticky="ew", padx=24, pady=(0, 8))
        self.deleted_card.grid_columnconfigure(0, weight=1)
        deleted_header = ctk.CTkFrame(self.deleted_card, fg_color="transparent")
        deleted_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        deleted_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            deleted_header,
            text="已标记删除的 Episode",
            font=self.font_button,
            text_color=UI_COLORS["muted"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.review_badge = ctk.CTkLabel(
            deleted_header,
            textvariable=self.review_var,
            width=210,
            height=28,
            corner_radius=8,
            fg_color=UI_COLORS["surface_alt"],
            text_color=UI_COLORS["muted"],
            font=self.font_small,
        )
        self.review_badge.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.deleted_list = ctk.CTkScrollableFrame(
            self.deleted_card,
            orientation="horizontal",
            height=52,
            corner_radius=8,
            fg_color=UI_COLORS["surface_alt"],
            scrollbar_fg_color=UI_COLORS["surface_alt"],
            scrollbar_button_color=UI_COLORS["border"],
            scrollbar_button_hover_color=UI_COLORS["muted"],
        )
        self.deleted_list.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        output_card = ctk.CTkFrame(
            self.root,
            fg_color=UI_COLORS["surface"],
            border_color=UI_COLORS["border"],
            border_width=1,
            corner_radius=12,
        )
        output_card.grid(row=6, column=0, sticky="ew", padx=24, pady=(0, 8))
        output_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(output_card, text="输出目录", font=self.font_button).grid(
            row=0, column=0, padx=(14, 10), pady=12
        )
        self.output_entry = ctk.CTkEntry(
            output_card,
            textvariable=self.output_var,
            height=36,
            corner_radius=9,
            font=self.font_latin,
            placeholder_text="过滤后的数据集将保存到这里",
        )
        self.output_entry.grid(row=0, column=1, sticky="ew", pady=12)
        ctk.CTkButton(
            output_card,
            text="选择位置",
            command=self.choose_output_parent,
            width=112,
            height=36,
            corner_radius=9,
            font=self.font_body,
            fg_color=UI_COLORS["surface_alt"],
            hover_color=UI_COLORS["border"],
            text_color=UI_COLORS["muted"],
        ).grid(row=0, column=2, padx=(10, 14), pady=12)

        status_card = ctk.CTkFrame(self.root, fg_color="transparent")
        status_card.grid(row=7, column=0, sticky="ew", padx=24, pady=(0, 14))
        status_card.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(
            status_card,
            textvariable=self.status_var,
            font=self.font_small,
            text_color=UI_COLORS["muted"],
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.busy_progress = ctk.CTkProgressBar(
            status_card,
            width=150,
            height=8,
            corner_radius=4,
            mode="indeterminate",
            progress_color=UI_COLORS["accent"],
        )
        self.busy_progress.grid(row=0, column=1, padx=(14, 0))
        self.busy_progress.grid_remove()
        self._update_action_state()

    def _show_video_placeholder(self) -> None:
        for child in self.video_row.winfo_children():
            child.destroy()
        self.image_labels.clear()
        self.photos.clear()
        placeholder = ctk.CTkLabel(
            self.video_row,
            text="打开数据集后，所有摄像头会在这里并排播放",
            font=self.font_section,
            text_color=UI_COLORS["muted"],
            fg_color=UI_COLORS["video"],
            corner_radius=12,
        )
        placeholder.grid(row=0, column=0, sticky="nsew")
        self.video_row.grid_columnconfigure(0, weight=1)

    def _appearance_changed(self, value: str) -> None:
        if ctk is not None:
            ctk.set_appearance_mode("dark" if value == "深色" else "light")

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.status_var.set(status)
        self.root.configure(cursor="watch" if busy else "")
        if busy:
            self.busy_progress.grid()
            self.busy_progress.start()
        else:
            self.busy_progress.stop()
            self.busy_progress.grid_remove()
        self._update_action_state()

    def _update_action_state(self) -> None:
        has_episode = self.dataset is not None and self.current_episode is not None
        episode_state = "normal" if self.dataset is not None and not self.busy else "disabled"
        self.episode_entry.configure(state=episode_state)
        self.jump_button.configure(state=episode_state)
        for button in self.episode_buttons.values():
            button.configure(state=episode_state)
        total = self.dataset.meta.total_episodes if self.dataset is not None else 0
        self.previous_page_button.configure(
            state="normal" if self.episode_page > 0 and not self.busy else "disabled"
        )
        self.next_page_button.configure(
            state="normal"
            if (self.episode_page + 1) * EPISODES_PER_PAGE < total and not self.busy
            else "disabled"
        )
        self.previous_button.configure(state="normal" if has_episode and not self.busy else "disabled")
        self.next_button.configure(state="normal" if has_episode and not self.busy else "disabled")
        playback_state = "normal" if self.decoders and not self.busy else "disabled"
        self.play_button.configure(state=playback_state)
        self.timeline.configure(state=playback_state)
        self.playback_modes.configure(state="disabled" if self.busy else "normal")
        review_state = "normal" if has_episode and not self.busy else "disabled"
        self.keep_button.configure(state=review_state)
        self.delete_button.configure(state=review_state)
        self.save_button.configure(state="normal" if self.deletions and not self.busy else "disabled")
        self.open_button.configure(state="disabled" if self.busy else "normal")

    def _choose_directory(self, initial: Path, title: str) -> Path | None:
        """Open a readable folder picker built with CustomTkinter."""
        if ctk is None:  # pragma: no cover - guarded by main
            return None
        current = initial.expanduser().resolve()
        if not current.is_dir():
            current = current.parent
        result: dict[str, Path | None] = {"path": None}
        dialog = ctk.CTkToplevel(self.root, fg_color=UI_COLORS["surface"])
        dialog.title(title)
        dialog.geometry("760x560")
        dialog.minsize(620, 420)
        dialog.transient(self.root)
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(dialog, text=title, font=self.font_section, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=24, pady=(22, 12)
        )
        path_row = ctk.CTkFrame(dialog, fg_color="transparent")
        path_row.grid(row=1, column=0, sticky="ew", padx=24)
        path_row.grid_columnconfigure(0, weight=1)
        path_var = tk.StringVar(value=str(current))
        path_entry = ctk.CTkEntry(path_row, textvariable=path_var, height=36, font=self.font_latin)
        path_entry.grid(row=0, column=0, sticky="ew")
        go_button = ctk.CTkButton(
            path_row,
            text="进入",
            width=76,
            height=36,
            corner_radius=9,
            font=self.font_body,
            fg_color=UI_COLORS["surface_alt"],
            hover_color=UI_COLORS["border"],
            text_color=UI_COLORS["muted"],
        )
        go_button.grid(row=0, column=1, padx=(8, 0))
        up_button = ctk.CTkButton(
            path_row,
            text="上一级",
            width=88,
            height=36,
            corner_radius=9,
            font=self.font_body,
            fg_color=UI_COLORS["surface_alt"],
            hover_color=UI_COLORS["border"],
            text_color=UI_COLORS["muted"],
        )
        up_button.grid(row=0, column=2, padx=(8, 0))

        list_card = ctk.CTkFrame(
            dialog,
            fg_color=UI_COLORS["surface_alt"],
            border_color=UI_COLORS["border"],
            border_width=1,
            corner_radius=12,
        )
        list_card.grid(row=2, column=0, sticky="nsew", padx=24, pady=14)
        list_card.grid_columnconfigure(0, weight=1)
        list_card.grid_rowconfigure(0, weight=1)
        directory_list = ctk.CTkScrollableFrame(list_card, fg_color="transparent")
        directory_list.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        directory_list.grid_columnconfigure(0, weight=1)
        status_var = tk.StringVar()
        ctk.CTkLabel(
            dialog,
            textvariable=status_var,
            font=self.font_small,
            text_color=UI_COLORS["muted"],
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 8))
        buttons = ctk.CTkFrame(dialog, fg_color="transparent")
        buttons.grid(row=4, column=0, sticky="e", padx=24, pady=(0, 20))

        def finish(path: Path | None) -> None:
            result["path"] = path
            with contextlib.suppress(tk.TclError):
                dialog.grab_release()
            dialog.destroy()

        ctk.CTkButton(
            buttons,
            text="取消",
            width=100,
            height=36,
            corner_radius=9,
            font=self.font_body,
            fg_color=UI_COLORS["surface_alt"],
            hover_color=UI_COLORS["border"],
            text_color=UI_COLORS["muted"],
            command=lambda: finish(None),
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            buttons,
            text="选择此目录",
            width=128,
            height=36,
            corner_radius=9,
            font=self.font_button,
            fg_color=UI_COLORS["accent"],
            hover_color=UI_COLORS["accent_hover"],
            command=lambda: finish(current),
        ).pack(side="left")

        def refresh(path: Path) -> None:
            nonlocal current
            path = path.expanduser()
            if not path.is_absolute():
                path = current / path
            try:
                path = path.resolve()
            except (OSError, RuntimeError):
                status_var.set("无法访问这个目录。")
                return
            if not path.is_dir():
                status_var.set("目录不存在，请输入有效的目录路径。")
                return
            current = path
            path_var.set(str(current))
            status_var.set(f"当前目录：{current}")
            for child in directory_list.winfo_children():
                child.destroy()
            try:
                directories = sorted(
                    (child for child in current.iterdir() if child.is_dir()),
                    key=lambda child: child.name.casefold(),
                )
            except OSError as exc:
                status_var.set(f"无法读取目录：{exc}")
                return
            if not directories:
                ctk.CTkLabel(
                    directory_list,
                    text="这个目录没有子目录。可以直接点击“选择此目录”。",
                    font=self.font_body,
                    text_color=UI_COLORS["muted"],
                    anchor="w",
                ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)
                return
            for row, directory in enumerate(directories):
                ctk.CTkButton(
                    directory_list,
                    text=f"目录  {directory.name}",
                    command=lambda selected=directory: refresh(selected),
                    height=36,
                    corner_radius=8,
                    font=self.font_body,
                    anchor="w",
                    fg_color="transparent",
                    hover_color=UI_COLORS["border"],
                    text_color=UI_COLORS["muted"],
                ).grid(row=row, column=0, sticky="ew", padx=2, pady=2)

        go_button.configure(command=lambda: refresh(Path(path_var.get())))
        up_button.configure(command=lambda: refresh(current.parent))
        path_entry.bind("<Return>", lambda event: refresh(Path(path_var.get())))
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        refresh(current)
        dialog.update_idletasks()
        parent_x = self.root.winfo_rootx()
        parent_y = self.root.winfo_rooty()
        parent_w = self.root.winfo_width()
        parent_h = self.root.winfo_height()
        dialog_w = dialog.winfo_width()
        dialog_h = dialog.winfo_height()
        dialog.geometry(
            f"{dialog_w}x{dialog_h}+{max(parent_x + (parent_w - dialog_w) // 2, 0)}+"
            f"{max(parent_y + (parent_h - dialog_h) // 2, 0)}"
        )
        dialog.grab_set()
        path_entry.focus_set()
        dialog.wait_window()
        return result["path"]

    def open_dataset(self) -> None:
        """Choose a dataset folder and load its metadata in the background."""
        if self.busy:
            return
        if self.dirty and not self._ask_yes_no(
            "打开其他数据集", "当前的删除标记尚未保存。确定要打开其他数据集吗？"
        ):
            return
        initial = self.dataset.root.parent if self.dataset else HF_LEROBOT_HOME
        if not initial.is_dir():
            initial = Path.home()
        selected = self._choose_directory(initial, "选择 LeRobot 数据集目录")
        if selected is None:
            return
        path = selected
        if not is_dataset_root(path):
            self._show_message("目录无效", f"请选择包含 meta/info.json 和 data/ 的目录：\n{path}", "error")
            return
        self._set_busy(True, f"正在读取 {path}…")
        threading.Thread(target=self._load_worker, args=(path,), daemon=True).start()

    def _load_worker(self, path: Path) -> None:
        try:
            dataset = load_local_dataset(path)
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._load_failed(error))
            return
        self.root.after(0, lambda: self._loaded(dataset))

    def _load_failed(self, error: str) -> None:
        self._set_busy(False, "数据集读取失败。")
        self._show_message("读取数据集失败", error, "error")

    def _loaded(self, dataset: LocalDataset) -> None:
        self._close_decoders()
        self.dataset = dataset
        self.current_episode = None
        self.episode_page = 0
        self.episode_var.set("")
        self.episode_range_var.set(f"/ {dataset.meta.total_episodes - 1}")
        self.deletions.clear()
        self.saved_deletions.clear()
        self.dirty = False
        self._refresh_deleted_episodes()
        self.dataset_var.set(
            f"{dataset.repo_id}  ·  {dataset.meta.total_episodes} 个 episode  ·  {dataset.root}"
        )
        self.output_var.set(str(default_output_path(dataset.root)))
        self._build_camera_panels()
        self._build_episode_buttons()
        self._set_busy(False, "点击编号或输入编号后回车切换 Episode；下方可切换循环或连续播放。")
        self.load_episode(0)

    def _build_camera_panels(self) -> None:
        for child in self.video_row.winfo_children():
            child.destroy()
        self.image_labels.clear()
        self.photos.clear()
        assert self.dataset is not None
        for column, key in enumerate(self.dataset.meta.video_keys):
            self.video_row.grid_columnconfigure(column, weight=1, uniform="camera")
            camera_card = ctk.CTkFrame(
                self.video_row,
                fg_color=UI_COLORS["surface_alt"],
                border_color=UI_COLORS["border"],
                border_width=1,
                corner_radius=12,
            )
            camera_card.grid(row=0, column=column, sticky="nsew", padx=5)
            camera_card.grid_columnconfigure(0, weight=1)
            camera_card.grid_rowconfigure(1, weight=1)
            camera_name = key.removeprefix("observation.images.").replace("_", " ").title()
            ctk.CTkLabel(
                camera_card,
                text=camera_name,
                font=self.font_latin,
                text_color=UI_COLORS["muted"],
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 7))
            label = ctk.CTkLabel(
                camera_card,
                text="正在加载视频…",
                font=self.font_body,
                text_color=UI_COLORS["muted"],
                fg_color=UI_COLORS["video"],
                corner_radius=10,
            )
            label.grid(row=1, column=0, sticky="nsew", padx=9, pady=(0, 9))
            label.bind("<Configure>", self._schedule_resize)
            self.image_labels[key] = label

    def _schedule_resize(self, event=None) -> None:
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(120, self._render_after_resize)

    def _render_after_resize(self) -> None:
        self.resize_job = None
        if not self.playing:
            self._render()

    def _build_episode_buttons(self) -> None:
        """Show one page of numbered buttons, highlighting the current and deleted episodes."""
        for child in self.episode_buttons_frame.winfo_children():
            child.destroy()
        self.episode_buttons.clear()
        if self.dataset is None:
            self.episode_page_var.set("0 / 0 页")
            return
        total = self.dataset.meta.total_episodes
        pages = (total + EPISODES_PER_PAGE - 1) // EPISODES_PER_PAGE
        self.episode_page = min(max(self.episode_page, 0), pages - 1)
        self.episode_page_var.set(f"{self.episode_page + 1} / {pages} 页")
        start = self.episode_page * EPISODES_PER_PAGE
        for column, index in enumerate(range(start, min(start + EPISODES_PER_PAGE, total))):
            selected = index == self.current_episode
            marked = index in self.deletions
            fg_color = UI_COLORS["accent"] if selected else UI_COLORS["surface_alt"]
            hover_color = UI_COLORS["accent_hover"] if selected else UI_COLORS["border"]
            text_color = "#ffffff" if selected else UI_COLORS["muted"]
            if marked:
                fg_color = UI_COLORS["danger"] if selected else ("#fee2e2", "#451a1a")
                hover_color = UI_COLORS["danger_hover"] if selected else ("#fecaca", "#7f1d1d")
                text_color = "#ffffff" if selected else ("#b91c1c", "#fca5a5")
            button = ctk.CTkButton(
                self.episode_buttons_frame,
                text=str(index),
                command=lambda episode=index: self.load_episode(episode),
                width=36,
                height=30,
                corner_radius=8,
                font=self.font_latin,
                fg_color=fg_color,
                hover_color=hover_color,
                text_color=text_color,
                border_width=2 if selected else 0,
                border_color=UI_COLORS["accent"],
                state="disabled" if self.busy else "normal",
            )
            button.grid(row=0, column=column, sticky="ew", padx=3)
            self.episode_buttons[index] = button

    def _move_episode_page(self, direction: int) -> None:
        """Browse episode numbers without interrupting the current video."""
        if self.busy or self.dataset is None:
            return
        self.episode_page += direction
        self._build_episode_buttons()
        self._update_action_state()

    def _close_decoders(self) -> None:
        self.playing = False
        for decoder in self.decoders.values():
            decoder.close()
        self.decoders.clear()
        self.photos.clear()

    def load_episode(self, index: int) -> None:
        """Open every camera at the selected episode and start playback."""
        if self.busy or self.dataset is None:
            return
        if not 0 <= index < self.dataset.meta.total_episodes:
            return
        self._close_decoders()
        self.scrubbing = False
        self.resume_after_scrub = False
        opened: dict[str, EpisodeVideoDecoder] = {}
        try:
            durations = []
            for key in self.dataset.meta.video_keys:
                path, start, duration = video_segment(self.dataset, index, key)
                opened[key] = EpisodeVideoDecoder(path, start, duration, self.dataset.meta.fps)
                durations.append(duration)
            self.duration_s = min(durations)
            self.decoders = opened
            self.current_episode = index
            self.position_s = 0.0
            self.timeline.configure(to=self.duration_s, state="normal")
            self.timeline.set(0.0)
            self.episode_var.set(str(index))
            self.episode_page = index // EPISODES_PER_PAGE
            self._build_episode_buttons()
            self._update_review_status()
            self._update_action_state()
            self.playing = True
            self._render()
            self.last_tick = time.monotonic()
            self.play_button.configure(text="暂停" if self.playing else "播放")
        except Exception as exc:
            self.playing = False
            for decoder in opened.values():
                decoder.close()
            self.decoders.clear()
            self._update_action_state()
            self.play_button.configure(text="播放")
            self.status_var.set(f"Episode {index} 加载失败。")
            self._show_message("视频加载失败", str(exc), "error")

    def _update_review_status(self) -> None:
        if self.current_episode is None:
            self.review_var.set("尚未开始审查")
            self.review_badge.configure(fg_color=UI_COLORS["surface_alt"], text_color=UI_COLORS["muted"])
            self._refresh_deleted_episodes()
            return
        marked = self.current_episode in self.deletions
        state = "待删除" if marked else "保留"
        self.review_var.set(f"当前：{state}  ·  已标记删除 {len(self.deletions)} Episodes")
        if marked:
            self.review_badge.configure(fg_color=("#fee2e2", "#451a1a"), text_color=("#b91c1c", "#fca5a5"))
        else:
            self.review_badge.configure(fg_color=("#dcfce7", "#153d2a"), text_color=("#166534", "#86efac"))
        self._refresh_deleted_episodes()

    def _refresh_deleted_episodes(self) -> None:
        """Refresh the horizontal list of episodes currently marked for deletion."""
        for child in self.deleted_list.winfo_children():
            child.destroy()
        if not self.deletions:
            ctk.CTkLabel(
                self.deleted_list,
                text="暂无标记删除的 Episode",
                font=self.font_small,
                text_color=UI_COLORS["muted"],
                anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=10, pady=8)
            return
        for column, episode_index in enumerate(sorted(self.deletions)):
            ctk.CTkButton(
                self.deleted_list,
                text=f"Episode {episode_index}",
                command=lambda index=episode_index: self.load_episode(index),
                width=112,
                height=34,
                corner_radius=8,
                font=self.font_latin,
                fg_color=("#fee2e2", "#451a1a"),
                hover_color=("#fecaca", "#7f1d1d"),
                text_color=("#b91c1c", "#fca5a5"),
            ).grid(row=0, column=column, padx=4, pady=7)

    def _jump_episode(self, event=None) -> str:
        """Load an explicitly submitted episode number, leaving invalid input editable."""
        if self.busy or self.dataset is None:
            return "break"
        try:
            index = int(self.episode_var.get().strip())
        except ValueError:
            index = -1
        if not 0 <= index < self.dataset.meta.total_episodes:
            self.status_var.set(f"请输入 0 到 {self.dataset.meta.total_episodes - 1} 之间的 Episode 编号。")
            self.episode_entry.focus_set()
            self.episode_entry.select_range(0, tk.END)
            return "break"
        self.load_episode(index)
        return "break"

    def _move_episode(self, direction: int) -> None:
        if self.dataset is None or self.current_episode is None:
            return
        index = (self.current_episode + direction) % self.dataset.meta.total_episodes
        self.load_episode(index)

    def _playback_mode_changed(self, mode: str) -> None:
        descriptions = {
            "播完暂停": "当前 Episode 播完后暂停。",
            "单集循环": "当前 Episode 播完后自动从头播放。",
            "连续播放": "播完自动播放下一集；最后一集播完后回到第 0 集。",
        }
        self.status_var.set(descriptions[mode])

    def _finish_episode(self) -> None:
        """Apply the selected playback mode after the final frame is rendered."""
        if not self.playing or self.busy or not self.decoders or self.dataset is None:
            return
        mode = self.playback_mode_var.get()
        if mode == "连续播放" and self.dataset.meta.total_episodes > 1:
            self._move_episode(1)
        elif mode in {"单集循环", "连续播放"}:
            self.seek_to(0.0)
            self.last_tick = time.monotonic()
        else:
            self.playing = False
            self.play_button.configure(text="播放")

    def toggle_play(self) -> None:
        """Pause playback or resume it from the current timeline position."""
        if not self.decoders or self.busy:
            return
        if self.position_s >= self.duration_s:
            self.seek_to(0.0)
        self.playing = not self.playing
        self.last_tick = time.monotonic()
        self.play_button.configure(text="暂停" if self.playing else "播放")

    def _begin_scrub(self, event=None) -> None:
        if not self.decoders or self.busy:
            return
        self.scrubbing = True
        self.resume_after_scrub = self.playing
        self.playing = False

    def _end_scrub(self, event=None) -> None:
        if not self.scrubbing:
            return
        self.scrubbing = False
        self.seek_to(float(self.timeline.get()))
        self.playing = self.resume_after_scrub and self.position_s < self.duration_s
        self.last_tick = time.monotonic()
        self.play_button.configure(text="暂停" if self.playing else "播放")

    def _timeline_changed(self, value: float) -> None:
        if not self.decoders:
            return
        self.position_s = min(max(float(value), 0.0), self.duration_s)
        self._render()

    def seek_to(self, position_s: float) -> None:
        """Show all cameras at one shared timeline position."""
        if not self.decoders:
            return
        self.position_s = min(max(position_s, 0.0), self.duration_s)
        self.timeline.set(self.position_s)
        self._render()

    def _render(self) -> None:
        if not self.decoders:
            return
        try:
            for key, decoder in self.decoders.items():
                image = decoder.frame_at(self.position_s)
                label = self.image_labels[key]
                width = label.winfo_width()
                height = label.winfo_height()
                if width < 50:
                    width = min(720, max(260, self.root.winfo_screenwidth() // len(self.decoders) - 50))
                if height < 50:
                    height = 460
                display = image.copy()
                display.thumbnail((max(1, width - 12), max(1, height - 12)), Image.Resampling.BILINEAR)
                photo = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
                label.configure(image=photo, text="")
                self.photos[key] = photo
        except Exception as exc:
            self.playing = False
            self.play_button.configure(text="播放")
            self.status_var.set("视频解码失败。")
            self._show_message("视频解码失败", str(exc), "error")
            return
        self.time_var.set(f"{self._format_time(self.position_s)} / {self._format_time(self.duration_s)}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes, remaining = divmod(int(seconds), 60)
        return f"{minutes:02d}:{remaining:02d}"

    def _tick(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_tick
        self.last_tick = now
        if self.playing and self.decoders and not self.scrubbing and not self.busy:
            speed = float(self.speed_var.get().removesuffix("×"))
            self.position_s = min(self.duration_s, self.position_s + elapsed * speed)
            self.timeline.set(self.position_s)
            self._render()
            if self.playing and self.position_s >= self.duration_s:
                self._finish_episode()
        self.root.after(TICK_MS, self._tick)

    def mark_current(self, delete: bool) -> None:
        """Mark the current episode for deletion or keep it."""
        if self.busy or self.dataset is None or self.current_episode is None:
            return
        try:
            self.deletions = update_deletions(
                self.deletions, self.current_episode, self.dataset.meta.total_episodes, delete
            )
        except ValueError as exc:
            self._show_message("无法全部删除", str(exc), "warning")
            return
        if delete:
            self.status_var.set(f"Episode {self.current_episode} 已标记删除；生成过滤数据集时应用。")
        else:
            self.status_var.set(f"Episode {self.current_episode} 已保留。")
        self.dirty = self.deletions != self.saved_deletions
        self._build_episode_buttons()
        self._update_review_status()
        self._update_action_state()

    def choose_output_parent(self) -> None:
        """Select the parent directory of the filtered dataset."""
        if self.dataset is None or self.busy:
            return
        selected = self._choose_directory(self.dataset.root.parent, "选择输出数据集的父目录")
        if selected is not None:
            self.output_var.set(str(default_output_path(self.dataset.root, selected)))

    def save(self) -> None:
        """Write a filtered copy of the source dataset in the background."""
        if self.busy or self.dataset is None:
            return
        if not self.deletions:
            self._show_message("没有删除标记", "请先点击“标记删除当前 Episode”。", "info")
            return
        raw_target = self.output_var.get().strip()
        if not raw_target:
            self._show_message("缺少输出目录", "请输入输出目录。", "error")
            return
        target = Path(raw_target).expanduser().resolve()
        try:
            check_output_path(self.dataset.root, target)
            new_repo_id = output_repo_id(self.dataset.repo_id, target)
        except (ValueError, FileExistsError) as exc:
            self._show_message("无法保存", str(exc), "error")
            return
        self.playing = False
        self.play_button.configure(text="播放")
        self._set_busy(True, f"正在生成过滤数据集：{target}")
        threading.Thread(
            target=self._save_worker,
            args=(self.dataset, self.deletions.copy(), new_repo_id, target),
            daemon=True,
        ).start()

    def _save_worker(
        self, dataset: LocalDataset, deletions: set[int], new_repo_id: str, target: Path
    ) -> None:
        try:
            loaded = LeRobotDataset(dataset.repo_id, root=dataset.root)
            result = save_filtered_dataset(loaded, deletions, new_repo_id, target)
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._save_failed(error))
            return
        self.root.after(0, lambda: self._saved(result))

    def _save_failed(self, error: str) -> None:
        self._set_busy(False, "生成过滤数据集失败；删除标记仍保留。")
        self._show_message("保存失败", error, "error")

    def _saved(self, target: Path) -> None:
        self.saved_deletions = self.deletions.copy()
        self.dirty = False
        self._set_busy(False, f"已保存：{target}")
        self._show_message("保存完成", f"过滤数据集已保存到：\n{target}", "success")

    def _show_message(self, title: str, message: str, kind: str = "info") -> None:
        self._dialog(title, message, kind=kind, confirm=False)

    def _ask_yes_no(self, title: str, message: str) -> bool:
        return self._dialog(title, message, kind="warning", confirm=True)

    def _dialog(self, title: str, message: str, kind: str, confirm: bool) -> bool:
        """Display a readable CustomTkinter modal instead of a native bitmap dialog."""
        if ctk is None:  # pragma: no cover - guarded by main
            return False
        dialog = ctk.CTkToplevel(self.root, fg_color=UI_COLORS["surface"])
        dialog.title(title)
        dialog.geometry("620x250" if confirm else "620x230")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(0, weight=1)
        result = {"value": False}
        colors = {
            "error": (("#fee2e2", "#451a1a"), ("#b91c1c", "#fca5a5"), "!"),
            "warning": (("#fef3c7", "#422006"), ("#b45309", "#fcd34d"), "!"),
            "success": (("#dcfce7", "#153d2a"), ("#166534", "#86efac"), "✓"),
            "info": (("#dbeafe", "#172554"), ("#1d4ed8", "#93c5fd"), "i"),
        }
        icon_bg, icon_fg, icon_text = colors.get(kind, colors["info"])
        body = ctk.CTkFrame(dialog, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew", padx=28, pady=(28, 12))
        body.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            body,
            text=icon_text,
            width=48,
            height=48,
            corner_radius=16,
            fg_color=icon_bg,
            text_color=icon_fg,
            font=self.font_title,
        ).grid(row=0, column=0, padx=(0, 18), sticky="n")
        ctk.CTkLabel(
            body,
            text=message,
            font=self.font_body,
            text_color=UI_COLORS["muted"],
            anchor="w",
            justify="left",
            wraplength=490,
        ).grid(row=0, column=1, sticky="ew", pady=4)
        buttons = ctk.CTkFrame(dialog, fg_color="transparent")
        buttons.grid(row=1, column=0, sticky="e", padx=28, pady=(0, 22))

        def finish(value: bool) -> None:
            result["value"] = value
            with contextlib.suppress(tk.TclError):
                dialog.grab_release()
            dialog.destroy()

        if confirm:
            ctk.CTkButton(
                buttons,
                text="取消",
                width=100,
                height=36,
                corner_radius=9,
                font=self.font_body,
                fg_color=UI_COLORS["surface_alt"],
                hover_color=UI_COLORS["border"],
                text_color=UI_COLORS["muted"],
                command=lambda: finish(False),
            ).pack(side="left", padx=(0, 10))
            ctk.CTkButton(
                buttons,
                text="确定",
                width=100,
                height=36,
                corner_radius=9,
                font=self.font_button,
                fg_color=UI_COLORS["accent"],
                hover_color=UI_COLORS["accent_hover"],
                command=lambda: finish(True),
            ).pack(side="left")
        else:
            ctk.CTkButton(
                buttons,
                text="知道了",
                width=110,
                height=36,
                corner_radius=9,
                font=self.font_button,
                fg_color=UI_COLORS["accent"],
                hover_color=UI_COLORS["accent_hover"],
                command=lambda: finish(False),
            ).pack(side="left")
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
        dialog.update_idletasks()
        parent_x = self.root.winfo_rootx()
        parent_y = self.root.winfo_rooty()
        parent_w = self.root.winfo_width()
        parent_h = self.root.winfo_height()
        dialog_w = dialog.winfo_width()
        dialog_h = dialog.winfo_height()
        dialog.geometry(
            f"{dialog_w}x{dialog_h}+{max(parent_x + (parent_w - dialog_w) // 2, 0)}+"
            f"{max(parent_y + (parent_h - dialog_h) // 2, 0)}"
        )
        dialog.grab_set()
        dialog.focus_force()
        dialog.wait_window()
        return bool(result["value"])

    def close(self) -> None:
        """Close the panel after handling pending review decisions."""
        if self.busy:
            self._show_message("请稍候", "正在读取或保存数据集，请等待完成。", "info")
            return
        if self.dirty and not self._ask_yes_no("退出审查", "当前有尚未保存的删除标记。确定退出吗？"):
            return
        self._close_decoders()
        self.root.destroy()


def main() -> None:
    """Start the CustomTkinter review panel."""
    if ctk is None:
        raise SystemExit(
            "此面板需要 CustomTkinter。请在项目根目录执行：\n"
            "  uv sync --locked --inexact --extra desktop --extra dataset\n"
            "同步时保留 dataset 及本机其他所需 extra，再通过 uv run --no-sync 运行对应命令。\n"
            f"当前解释器：{sys.executable}"
        )
    if _tk_xft_status() is False:
        raise SystemExit(
            "当前 Python 使用的是不支持 Xft/FreeType 的 Tk，文字会被渲染成位图。\n"
            "请通过 uv run --no-sync 使用项目 Python 环境；若仍出现此错误，需修复该解释器的 Tk。\n"
            "Python 桌面依赖统一由 uv sync --locked --inexact --extra desktop --extra dataset 管理，"
            "同步时保留 dataset 及本机其他所需 extra；desktop extra 不会替换解释器自带的 Tk。\n"
            f"当前解释器：{sys.executable}"
        )
    _configure_ctk_rendering()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    try:
        root = ctk.CTk()
    except tk.TclError as exc:
        raise SystemExit(f"无法启动图形面板：{exc}") from exc
    ReviewPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
