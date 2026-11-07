#!/usr/bin/env python3
# ruff: noqa: D101, D102, D103, D107

"""Linux desktop picker for merging local LeRobot datasets.

Run ``uv run --no-sync python src/lerobot/yzh/merge_datasets_gui.py`` with the
``dataset`` and ``desktop`` extras installed. The default Qt window shares the
GitHub upload tool's dark palette and antialiased Chinese text. The older GTK
and Tk interfaces remain available through ``LEROBOT_GUI_BACKEND=zenity|tk``.
The selected folders are passed to the existing ``lerobot-edit-dataset`` merge
operation via ``--operation.roots``; no dataset is uploaded to the Hub.
When mixing demonstrations and DAgger datasets, missing ``intervention`` flags
are filled with ``False`` in the output; existing flags are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, scrolledtext, ttk
except ImportError:
    # The Qt window and command-building helpers do not require a Tk installation.
    tk = None


def find_project_root() -> Path:
    """Find the repository root regardless of where this helper script lives."""
    script_path = Path(__file__).resolve()
    for candidate in (script_path, *script_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError("无法定位 LeRobot 项目根目录，请从项目目录中运行此程序。")


PROJECT_ROOT = find_project_root()
REPO_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
REPO_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def subprocess_environment(project_root: Path | None = None) -> dict[str, str]:
    """Keep user settings while excluding foreign Python and shared-library paths."""
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    if project_root is not None:
        environment["PYTHONPATH"] = str(project_root / "src")
    return environment


def lerobot_home() -> Path:
    """Find the same default local dataset root used by Hugging Face/LeRobot."""
    hf_home = Path(
        os.environ.get(
            "HF_HOME",
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface",
        )
    ).expanduser()
    return Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot")).expanduser().resolve()


def is_dataset_dir(path: Path) -> bool:
    """A local dataset root contains metadata and recorded data."""
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def natural_key(path: Path) -> list[tuple[int, int | str]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold()) for part in re.split(r"(\d+)", str(path))
    ]


def discover_datasets(parent: Path) -> list[Path]:
    """Show datasets in a folder or one namespace level below it."""
    parent = parent.expanduser().resolve()
    if not parent.is_dir():
        raise ValueError(f"源目录不存在：{parent}")
    if is_dataset_dir(parent):
        return [parent]

    found: set[Path] = set()
    for child in parent.iterdir():
        if not child.is_dir() or child.name == "hub":
            continue
        if is_dataset_dir(child):
            found.add(child.resolve())
            continue
        for grandchild in child.iterdir():
            if grandchild.is_dir() and is_dataset_dir(grandchild):
                found.add(grandchild.resolve())
    return sorted(found, key=natural_key)


def repo_id_for_path(path: Path, home: Path) -> str:
    """Infer an identifier; explicit roots keep this independent of storage location."""
    path = path.resolve()
    try:
        relative = path.relative_to(home.resolve())
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) == 2:
        return relative.as_posix()
    return f"{path.parent.name}/{path.name}"


def _repo_component(value: str) -> str | None:
    """Return a safe Repo ID component, or ``None`` when no name remains."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return component if REPO_COMPONENT_PATTERN.fullmatch(component) else None


def infer_namespace(paths: list[Path], source_parent: Path, home: Path) -> str:
    """Infer the Repo ID namespace from the source folder or selected datasets."""
    home = home.expanduser().resolve()
    parent = source_parent.expanduser().resolve()
    if is_dataset_dir(parent):
        parent = parent.parent

    # Selecting ``.../lerobot/block`` directly is the clearest namespace hint.
    try:
        relative_parent = parent.relative_to(home)
    except ValueError:
        relative_parent = None
    if relative_parent is not None and len(relative_parent.parts) == 1:
        namespace = _repo_component(relative_parent.parts[0])
        if namespace:
            return namespace

    # When the root ``.../lerobot`` is selected, use the common dataset parent
    # (``block`` for ``.../lerobot/block/dataset``).
    parent_names = {
        path.expanduser().resolve().parent.name for path in paths if path.expanduser().resolve().name
    }
    if len(parent_names) == 1:
        namespace = _repo_component(next(iter(parent_names)))
        if namespace:
            return namespace

    # A custom source folder outside HF_LEROBOT_HOME can still be a namespace
    # folder, so use its name as a final source-based hint.
    if parent != home:
        namespace = _repo_component(parent.name)
        if namespace:
            return namespace
    return "default"


def common_dataset_prefix(paths: list[Path]) -> str | None:
    """Find a meaningful common name prefix for the selected dataset folders."""
    names = [path.expanduser().resolve().name for path in paths if path.name]
    if len(names) < 2:
        return None
    prefix = os.path.commonprefix(names)
    # Dataset batches often differ only by a range suffix such as ``_0-10``;
    # remove that varying tail before adding the merge suffix.
    prefix = prefix.rstrip("._-")
    prefix = re.sub(r"[_-]?\d+$", "", prefix).rstrip("._-")
    prefix = _repo_component(prefix)
    return prefix if prefix and len(prefix) >= 2 else None


def default_repo_id(paths: list[Path], source_parent: Path, home: Path) -> str:
    """Build the initial ``namespace/dataset`` value shown by the GUI."""
    namespace = infer_namespace(paths, source_parent, home)
    prefix = common_dataset_prefix(paths)
    dataset_name = f"{prefix}_merged" if prefix else "merged_data"
    return f"{namespace}/{dataset_name}"


def build_merge_command(
    paths: list[Path], new_repo_id: str, home: Path, python: str = sys.executable
) -> tuple[list[str], Path]:
    """Validate the selection and build a shell-free merge invocation.

    Local LeRobot datasets are stored below ``HF_LEROBOT_HOME``.  The source
    picker may point at a namespace folder such as ``.../lerobot/block`` for
    discovery, but the output base is always the configured LeRobot home.  The
    final directory is therefore ``home / new_repo_id``.
    """
    new_repo_id = new_repo_id.strip()
    if not REPO_ID_PATTERN.fullmatch(new_repo_id):
        raise ValueError(
            "输出 Repo ID 必须是 namespace/dataset 格式，且只能含英文字母、数字、点、下划线和连字符。"
        )
    if len(paths) < 2:
        raise ValueError("请至少选择两个数据集。")

    sources = [path.expanduser().resolve() for path in paths]
    if len(sources) != len(set(sources)):
        raise ValueError("不能重复选择同一个数据集。")
    for source in sources:
        if not is_dataset_dir(source):
            raise ValueError(f"不是有效的本地 LeRobot 数据集目录：{source}")

    home = home.expanduser().resolve()
    if not home.is_dir():
        raise ValueError(f"LeRobot 根目录不存在：{home}")
    output_dir = (home / new_repo_id).resolve()
    if output_dir.exists():
        raise ValueError(f"输出目录已经存在，为避免覆盖已阻止合并：{output_dir}")
    for source in sources:
        if output_dir == source or output_dir.is_relative_to(source):
            raise ValueError(f"输出目录不能位于输入数据集内部：{source}")

    repo_ids = [repo_id_for_path(path, home) for path in sources]
    command = [
        python,
        "-u",
        "-m",
        "lerobot.scripts.lerobot_edit_dataset",
        "--new_repo_id",
        new_repo_id,
        "--new_root",
        str(output_dir),
        "--operation.type",
        "merge",
        "--operation.repo_ids",
        json.dumps(repo_ids, ensure_ascii=False),
        "--operation.roots",
        json.dumps([str(path) for path in sources], ensure_ascii=False),
        "--operation.fill_missing_intervention",
        "true",
        "--push_to_hub",
        "false",
    ]
    return command, output_dir


def configure_fonts(root: tk.Tk) -> tuple[tkfont.Font, tkfont.Font, tkfont.Font]:
    """Use a readable vector CJK font instead of Tk's small bitmap fallback."""
    current_scaling = float(root.tk.call("tk", "scaling"))
    if current_scaling < 1.35:
        root.tk.call("tk", "scaling", 1.35)
    families = set(tkfont.families(root))
    # Tk can omit locale-specific aliases from ``font.families`` even though
    # fontconfig can load the installed TTC file.  Check the file first so the
    # fallback never becomes the X11 ``Fixed`` bitmap font.
    noto_files = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path.home() / ".local/share/fonts/NotoSansCJK-Regular.ttc",
    )
    if any(path.is_file() for path in noto_files):
        family = "Noto Sans CJK SC"
        mono_family = "Noto Sans Mono CJK SC"
    elif "Droid Sans Fallback" in families:
        family = mono_family = "Droid Sans Fallback"
    else:
        family = next(
            (
                name
                for name in (
                    "Noto Sans CJK SC",
                    "Noto Sans CJK TC",
                    "Noto Sans CJK JP",
                    "WenQuanYi Micro Hei",
                    "DejaVu Sans",
                )
                if name in families
            ),
            "DejaVu Sans",
        )
        mono_family = family
    try:
        requested_size = int(os.environ.get("LEROBOT_GUI_FONT_SIZE", "16"))
    except ValueError:
        requested_size = 16
    size = max(13, min(requested_size, 24))
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
        tkfont.nametofont(name, root=root).configure(family=family, size=size)

    ui_font = tkfont.Font(root=root, family=family, size=size)
    strong_font = tkfont.Font(root=root, family=family, size=size, weight="bold")
    log_font = tkfont.Font(root=root, family=mono_family, size=max(size - 2, 12))
    root.option_add("*Font", ui_font)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    # Explicit named styles are needed on Linux because some themes ignore the
    # generic Tk font option for ttk widgets.
    style.configure("CJK.TLabel", font=ui_font)
    style.configure("CJKStrong.TLabel", font=strong_font)
    style.configure("CJK.TButton", font=strong_font, padding=(10, 7))
    style.configure("CJK.TEntry", font=ui_font, padding=(6, 5))
    return ui_font, strong_font, log_font


def zenity_call(args: list[str], *, input_text: str | None = None) -> tuple[int, str]:
    """Run a Zenity dialog and return its exit code and text output."""
    completed = subprocess.run(
        ["zenity", *args],
        env=subprocess_environment(),
        input=input_text,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return completed.returncode, completed.stdout.strip()


def zenity_directory(title: str, initialdir: Path) -> Path | None:
    """Select a directory with GTK/Zenity's antialiased file chooser."""
    code, output = zenity_call(
        [
            "--file-selection",
            "--directory",
            f"--title={title}",
            f"--filename={initialdir.expanduser().resolve()}{os.sep}",
        ]
    )
    return Path(output).expanduser().resolve() if code == 0 and output else None


def zenity_entry(title: str, prompt: str, default: str) -> str | None:
    code, output = zenity_call(["--entry", f"--title={title}", f"--text={prompt}", f"--entry-text={default}"])
    return output if code == 0 else None


def zenity_message(kind: str, title: str, text: str) -> None:
    zenity_call([f"--{kind}", f"--title={title}", f"--text={text}"])


def zenity_question(title: str, text: str) -> bool:
    code, _ = zenity_call(
        [
            "--question",
            f"--title={title}",
            f"--text={text}",
            "--ok-label=开始合并",
            "--cancel-label=取消",
        ]
    )
    return code == 0


def zenity_choose_datasets(paths: list[Path], home: Path) -> list[Path] | None:
    """Multi-select local datasets in a GTK checklist."""
    args = [
        "--list",
        "--checklist",
        "--multiple",
        "--separator=|",
        "--print-column=3",
        "--hide-column=3",
        "--title=选择要合并的数据集",
        "--text=勾选数据集；将按列表顺序合并：",
        "--width=1100",
        "--height=700",
        "--column=选择",
        "--column=数据集目录",
        "--column=完整路径",
    ]
    for path in paths:
        args.extend(
            [
                "FALSE",
                path.relative_to(home).as_posix() if path.is_relative_to(home) else str(path),
                str(path),
            ]
        )

    code, output = zenity_call(args)
    if code != 0:
        return None
    selected_tokens = [token for token in output.split("|") if token]
    by_token = {str(path): path for path in paths}
    by_token.update({path.relative_to(home).as_posix(): path for path in paths if path.is_relative_to(home)})
    return [by_token[token] for token in selected_tokens if token in by_token]


def run_merge_with_zenity(command: list[str], output_dir: Path, project_root: Path) -> int:
    """Run merge with a clear GTK progress window and a scrollable log."""
    env = subprocess_environment(project_root)
    logs: list[str] = []
    progress = None
    try:
        progress = subprocess.Popen(
            [
                "zenity",
                "--progress",
                "--pulsate",
                "--auto-close",
                "--title=正在合并数据集",
                "--text=正在准备合并，请稍候…",
                "--width=620",
            ],
            env=subprocess_environment(),
            stdin=subprocess.PIPE,
            text=True,
        )
        with subprocess.Popen(
            command,
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                logs.append(line)
                if progress.stdin is not None and progress.poll() is None:
                    with suppress(BrokenPipeError, OSError):
                        progress.stdin.write(f"# {line.strip()}\n")
                        progress.stdin.flush()
            return_code = process.wait()
    except OSError as exc:
        logs.append(f"启动失败：{exc}\n")
        return_code = 1
    finally:
        if progress is not None:
            if progress.stdin is not None:
                with suppress(BrokenPipeError, OSError):
                    progress.stdin.write("100\n")
                    progress.stdin.flush()
                    progress.stdin.close()
            progress.wait()

    result = "合并完成" if return_code == 0 else f"合并失败（退出码 {return_code}）"
    log_text = f"{result}\n输出目录：{output_dir}\n\n" + "".join(logs)
    zenity_call(
        [
            "--text-info",
            f"--title={result}",
            "--width=1100",
            "--height=700",
            "--font=Noto Sans CJK SC 14",
        ],
        input_text=log_text,
    )
    return return_code


def run_zenity_gui() -> None:
    """Use GTK/Zenity when available; it avoids the no-Xft Conda Tk build."""
    home = lerobot_home()
    source_parent = zenity_directory("选择源数据父目录", home)
    if source_parent is None:
        return
    try:
        paths = discover_datasets(source_parent)
    except (OSError, ValueError) as exc:
        zenity_message("error", "读取数据集失败", str(exc))
        return
    if not paths:
        zenity_message("error", "没有找到数据集", f"目录中没有找到有效的 LeRobot 数据集：\n{source_parent}")
        return

    selected = zenity_choose_datasets(paths, home)
    if selected is None:
        return
    suggestion_paths = selected if len(selected) >= 2 else paths
    repo_id = zenity_entry(
        "输出 Repo ID",
        "请输入 namespace/dataset 格式的输出 Repo ID（默认值可修改）：",
        default_repo_id(suggestion_paths, source_parent, home),
    )
    if repo_id is None:
        return

    try:
        command, output_dir = build_merge_command(selected, repo_id, home)
    except (OSError, ValueError) as exc:
        zenity_message("error", "无法开始合并", str(exc))
        return

    summary = "\n".join(str(path) for path in selected)
    if not zenity_question(
        "确认合并",
        f"将按以下顺序合并 {len(selected)} 个数据集：\n\n{summary}\n\n输出目录：{output_dir}",
    ):
        return
    run_merge_with_zenity(command, output_dir, PROJECT_ROOT)


class MergeDatasetGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.ui_font, self.strong_font, self.log_font = configure_fonts(root)
        self.home = lerobot_home()
        self.paths: list[Path] = []
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, str | int]] = queue.Queue()
        self.running = False
        self.stop_requested = threading.Event()
        self.close_after_stop = False

        root.title("LeRobot 本地数据集合并")
        root.geometry("1200x900")
        root.minsize(1050, 760)
        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(10, weight=1)

        self.source_parent = tk.StringVar(value=str(self.home))
        ttk.Label(frame, text="源数据父目录", style="CJKStrong.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.source_parent, style="CJK.TEntry").grid(
            row=0, column=1, sticky="ew", padx=10
        )
        ttk.Button(frame, text="浏览…", command=self.browse_source_parent, style="CJK.TButton").grid(
            row=0, column=2
        )
        ttk.Button(frame, text="刷新", command=self.refresh, style="CJK.TButton").grid(
            row=0, column=3, padx=(10, 0)
        )

        ttk.Label(frame, text="单击数据集可选中或取消；按列表顺序合并：", style="CJK.TLabel").grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(12, 4)
        )
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=2, column=0, columnspan=4, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.dataset_list = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            font=self.ui_font,
            background="#ffffff",
            foreground="#111111",
            selectbackground="#3478c7",
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightcolor="#3478c7",
            borderwidth=1,
            relief="solid",
            activestyle="none",
        )
        self.dataset_list.grid(row=0, column=0, sticky="nsew")
        self.dataset_list.bind("<<ListboxSelect>>", lambda _event: self.on_dataset_selection_changed())
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.dataset_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.dataset_list.configure(yscrollcommand=scrollbar.set)

        actions = ttk.Frame(frame)
        actions.grid(row=3, column=0, columnspan=4, sticky="ew", pady=8)
        ttk.Button(actions, text="全选", command=self.select_all, style="CJK.TButton").pack(side="left")
        ttk.Button(actions, text="清空", command=self.clear_selection, style="CJK.TButton").pack(
            side="left", padx=7
        )
        ttk.Button(actions, text="添加单个目录…", command=self.add_dataset, style="CJK.TButton").pack(
            side="left", padx=7
        )
        ttk.Button(actions, text="上移", command=lambda: self.move_selected(-1), style="CJK.TButton").pack(
            side="left"
        )
        ttk.Button(actions, text="下移", command=lambda: self.move_selected(1), style="CJK.TButton").pack(
            side="left", padx=7
        )
        self.selection_count = tk.StringVar(value="已选 0 组")
        ttk.Label(actions, textvariable=self.selection_count, style="CJKStrong.TLabel").pack(side="right")

        self.repo_id_auto = True
        self._setting_repo_id = False
        self.new_repo_id = tk.StringVar(value="")
        ttk.Label(frame, text="输出 Repo ID（默认自动生成，可修改）", style="CJKStrong.TLabel").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(frame, textvariable=self.new_repo_id, style="CJK.TEntry").grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(7, 0)
        )

        ttk.Label(frame, text="输出目录（LeRobot 根目录 + Repo ID）", style="CJKStrong.TLabel").grid(
            row=5, column=0, sticky="w", pady=(10, 0)
        )
        self.output_preview = tk.Label(frame, anchor="w", justify="left", wraplength=880, font=self.ui_font)
        self.output_preview.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        self.new_repo_id.trace_add("write", self.on_repo_id_changed)

        controls = ttk.Frame(frame)
        controls.grid(row=7, column=0, columnspan=4, sticky="ew", pady=10)
        self.merge_button = ttk.Button(
            controls, text="开始合并", command=self.start_merge, style="CJK.TButton"
        )
        self.merge_button.pack(side="left")
        self.stop_button = ttk.Button(
            controls, text="停止", command=self.stop_merge, state="disabled", style="CJK.TButton"
        )
        self.stop_button.pack(side="left", padx=8)
        self.status = tk.StringVar(value="就绪")
        ttk.Label(controls, textvariable=self.status, style="CJKStrong.TLabel").pack(side="left", padx=16)

        ttk.Label(frame, text="运行日志", style="CJKStrong.TLabel").grid(
            row=9, column=0, columnspan=4, sticky="w"
        )
        self.log = scrolledtext.ScrolledText(
            frame, height=13, state="disabled", wrap="word", font=self.log_font
        )
        self.log.grid(row=10, column=0, columnspan=4, sticky="nsew", pady=(4, 0))

        self.refresh()
        self.update_output_preview()
        self.root.after(100, self.drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def label_for(self, path: Path) -> str:
        try:
            return path.relative_to(self.home).as_posix()
        except ValueError:
            return str(path)

    def display_paths(self, selected: set[Path]) -> None:
        self.dataset_list.delete(0, tk.END)
        for index, path in enumerate(self.paths):
            self.dataset_list.insert(tk.END, self.label_for(path))
            if path in selected:
                self.dataset_list.selection_set(index)
        self.update_selection_count()
        self.update_default_repo_id()

    def selected_paths(self) -> list[Path]:
        return [self.paths[index] for index in self.dataset_list.curselection()]

    def update_selection_count(self) -> None:
        self.selection_count.set(f"已选 {len(self.dataset_list.curselection())} 组")

    def on_dataset_selection_changed(self) -> None:
        self.update_selection_count()
        self.update_default_repo_id()

    def on_repo_id_changed(self, *_args: object) -> None:
        if not self._setting_repo_id:
            self.repo_id_auto = False
        self.update_output_preview()

    def update_default_repo_id(self) -> None:
        """Refresh the suggestion until the user manually edits the Repo ID."""
        if not self.repo_id_auto:
            return
        selected = self.selected_paths()
        suggestion_paths = selected if len(selected) >= 2 else self.paths
        suggestion = default_repo_id(suggestion_paths, Path(self.source_parent.get()), self.home)
        if self.new_repo_id.get() != suggestion:
            self._setting_repo_id = True
            try:
                self.new_repo_id.set(suggestion)
            finally:
                self._setting_repo_id = False
        else:
            self.update_output_preview()

    def refresh(self) -> None:
        selected = set(self.selected_paths())
        try:
            self.paths = discover_datasets(Path(self.source_parent.get()))
        except (OSError, ValueError) as exc:
            messagebox.showerror("读取数据集失败", str(exc))
            return
        self.display_paths(selected)
        self.status.set(f"发现 {len(self.paths)} 个本地数据集")

    def browse_source_parent(self) -> None:
        chosen = filedialog.askdirectory(title="选择源数据父目录", initialdir=self.source_parent.get())
        if chosen:
            self.source_parent.set(chosen)
            self.refresh()

    def add_dataset(self) -> None:
        chosen = filedialog.askdirectory(
            title="选择一个 LeRobot 数据集目录", initialdir=self.source_parent.get()
        )
        if not chosen:
            return
        path = Path(chosen).resolve()
        if not is_dataset_dir(path):
            messagebox.showerror("无效目录", f"找不到 meta/info.json 或 data/：\n{path}")
            return
        selected = set(self.selected_paths()) | {path}
        if path not in self.paths:
            self.paths.append(path)
            self.paths.sort(key=natural_key)
        self.display_paths(selected)

    def select_all(self) -> None:
        self.dataset_list.selection_set(0, tk.END)
        self.update_selection_count()
        self.update_default_repo_id()

    def clear_selection(self) -> None:
        self.dataset_list.selection_clear(0, tk.END)
        self.update_selection_count()
        self.update_default_repo_id()

    def move_selected(self, direction: int) -> None:
        selected = set(self.dataset_list.curselection())
        indices = sorted(selected, reverse=direction > 0)
        for index in indices:
            other = index + direction
            if 0 <= other < len(self.paths) and other not in selected:
                self.paths[index], self.paths[other] = self.paths[other], self.paths[index]
                selected.remove(index)
                selected.add(other)
        self.display_paths({self.paths[index] for index in selected})

    def update_output_preview(self) -> None:
        repo_id = self.new_repo_id.get().strip()
        parent = self.home
        if REPO_ID_PATTERN.fullmatch(repo_id):
            output = parent / repo_id
            exists = output.exists()
            self.output_preview.configure(
                text=f"实际输出目录：{output}" + ("  （已存在，不能覆盖）" if exists else ""),
                fg="#b00020" if exists else "#333333",
            )
        else:
            self.output_preview.configure(text="请输入 namespace/dataset 格式的输出 Repo ID", fg="#b00020")

    def append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, text.replace("\r", "\n"))
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def start_merge(self) -> None:
        if self.running:
            return
        try:
            command, output = build_merge_command(self.selected_paths(), self.new_repo_id.get(), self.home)
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法开始合并", str(exc))
            return

        self.append_log("\n输入数据集（按此顺序合并）：\n")
        for path in self.selected_paths():
            self.append_log(f"  {path}\n")
        self.append_log(f"输出目录：{output}\n命令：{shlex.join(command)}\n\n")
        self.status.set("合并进行中…")
        self.running = True
        self.stop_requested.clear()
        self.merge_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self.run_merge, args=(command,), daemon=True).start()

    def run_merge(self, command: list[str]) -> None:
        env = subprocess_environment(PROJECT_ROOT)
        try:
            with subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors="replace",
                start_new_session=True,
            ) as process:
                self.process = process
                if self.stop_requested.is_set():
                    self.terminate_process(process)
                assert process.stdout is not None
                for line in process.stdout:
                    self.events.put(("log", line))
                return_code = process.wait()
            self.events.put(("done", return_code))
        except OSError as exc:
            self.events.put(("error", str(exc)))
        finally:
            self.process = None

    @staticmethod
    def terminate_process(process: subprocess.Popen[str]) -> None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)

    def stop_merge(self) -> None:
        if not self.running:
            return
        self.stop_requested.set()
        if self.process is not None and self.process.poll() is None:
            self.terminate_process(self.process)
        self.status.set("正在停止…")
        self.append_log("\n正在停止；输出目录可能包含未完成的数据，请检查后再使用。\n")
        self.stop_button.configure(state="disabled")

    def drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.append_log(str(payload))
                elif kind == "done":
                    self.running = False
                    if self.close_after_stop:
                        self.root.destroy()
                        return
                    self.merge_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status.set("合并完成" if payload == 0 else f"合并失败（退出码 {payload}）")
                    self.append_log(f"\n进程退出，退出码：{payload}\n")
                    self.update_output_preview()
                elif kind == "error":
                    self.running = False
                    if self.close_after_stop:
                        self.root.destroy()
                        return
                    self.merge_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status.set("无法启动合并")
                    self.append_log(f"\n启动失败：{payload}\n")
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)

    def close(self) -> None:
        if self.running and not messagebox.askyesno(
            "合并仍在运行", "关闭窗口会停止合并，可能留下不完整的输出目录。确定关闭吗？"
        ):
            return
        if self.running:
            self.close_after_stop = True
            self.stop_merge()
        else:
            self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot 本地数据集合并 · 深色桌面面板")
    parser.add_argument("--source", type=Path, help="启动时扫描的源数据父目录（Qt 界面）")
    parser.add_argument("--demo", action="store_true", help="预览示例界面，不读取或合并真实数据")
    parser.add_argument(
        "--backend",
        choices=("auto", "qt", "tk", "zenity"),
        default=os.environ.get("LEROBOT_GUI_BACKEND", "auto").strip().lower(),
        help="auto 默认使用 Qt，未安装 Qt 时使用兼容界面",
    )
    args = parser.parse_args()
    backend = args.backend
    if backend not in {"auto", "qt", "tk", "zenity"}:
        parser.error("LEROBOT_GUI_BACKEND 必须是 auto、qt、tk 或 zenity")
    if (args.source or args.demo) and backend in {"tk", "zenity"}:
        parser.error("--source 和 --demo 需要 Qt 界面")
    if backend in {"auto", "qt"}:
        try:
            if __package__:
                from .merge_datasets_window import launch
            else:
                from merge_datasets_window import launch
        except ImportError as exc:
            if not (exc.name or "").startswith("PyQt5"):
                raise
            if backend == "qt" or args.source or args.demo:
                parser.exit(
                    1,
                    "Qt 界面需要 desktop extra，请先执行 uv sync --locked --inexact --extra dataset --extra desktop\n",
                )
            print("未安装 PyQt5，正在使用兼容界面。", file=sys.stderr)
        else:
            launch(source=args.source, demo=args.demo)
            return
    if backend == "zenity" or (backend == "auto" and shutil.which("zenity")):
        run_zenity_gui()
        return
    if tk is None:
        parser.exit(1, "当前环境没有 Tk，请安装 desktop extra 后使用 Qt 界面。\n")
    root = tk.Tk()
    MergeDatasetGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
