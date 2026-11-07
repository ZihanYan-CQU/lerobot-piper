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

"""Local preferences and Linux desktop shortcut for the standalone upload tool."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APPLICATION_ID = "lerobot-github-upload"
APPLICATION_ICON = "github-mark-white.svg"


def application_icon_path() -> Path:
    """Locate the unchanged official GitHub mark bundled with the upload tool."""
    return Path(__file__).resolve().parent / "assets" / APPLICATION_ICON


class ConfigError(ValueError):
    """A configuration problem whose message never includes credential values."""


@dataclass(frozen=True)
class UploadPreferences:
    """Values restored on launch or explicitly saved by the user."""

    directory: str = ""
    remote: str = ""
    branch: str = "main"
    message: str = "Update local code"
    username: str = ""
    name: str = ""
    email: str = ""
    token: str = field(default="", repr=False)
    code_only: bool = True
    create_ignore: bool = True


def default_config_path() -> Path:
    """Keep credentials outside project trees and honor the desktop's XDG location."""
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config").expanduser()
    return base / "lerobot-github-upload" / "config.toml"


def _validated(values: dict) -> UploadPreferences:
    known = {item.name for item in fields(UploadPreferences)}
    if set(values) - known:
        raise ConfigError("参数文件含有不支持的字段，请对照示例中的参数名称。")
    for key, value in values.items():
        expected = bool if key in {"code_only", "create_ignore"} else str
        if not isinstance(value, expected):
            raise ConfigError(
                f"参数 {key} 的类型不正确，应为 {'true / false' if expected is bool else '字符串'}。"
            )
    for key in ("username", "token"):
        if any(character in values.get(key, "") for character in ("\r", "\n", "\0")):
            raise ConfigError(f"参数 {key} 不能包含换行符或空字符。")
    return UploadPreferences(**values)


def load_preferences(path: Path) -> UploadPreferences:
    """Read TOML without ever echoing malformed lines that may contain a token."""
    try:
        with path.expanduser().open("rb") as stream:
            return _validated(tomllib.load(stream))
    except FileNotFoundError:
        return UploadPreferences()
    except (tomllib.TOMLDecodeError, UnicodeError) as exc:
        raise ConfigError(f"参数文件格式有误，请检查引号和 true / false：\n{path}") from exc
    except OSError as exc:
        raise ConfigError(f"无法读取参数文件，请检查文件权限：\n{path}") from exc


def parse_preferences(text: str) -> UploadPreferences:
    """Validate editor contents without including secret-bearing lines in errors."""
    try:
        return _validated(tomllib.loads(text))
    except (tomllib.TOMLDecodeError, UnicodeError) as exc:
        raise ConfigError("参数格式有误，请检查引号、参数名称和 true / false。") from exc


def save_preferences_text(path: Path, text: str) -> None:
    """Save validated TOML from the editor, retaining comments and private permissions."""
    parse_preferences(text)
    try:
        _private_write(path.expanduser(), text if text.endswith("\n") else text + "\n")
    except OSError as exc:
        raise ConfigError(f"无法保存参数文件，请检查目录权限：\n{path}") from exc


def _private_write(path: Path, content: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".github-upload-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_preferences(path: Path, preferences: UploadPreferences) -> None:
    """Write an editable local file atomically, readable only by its owner."""
    values = asdict(_validated(asdict(preferences)))
    comments = {
        "directory": "本地项目根目录；留空则使用工具所在项目",
        "remote": "个人 GitHub 仓库地址，例如 https://github.com/用户名/仓库.git",
        "branch": "上传到 GitHub 的目标分支",
        "message": "默认提交说明，上传时仍可在窗口修改",
        "username": "GitHub 登录用户名；使用 HTTPS Token 时传给 Git",
        "name": "Git 提交者姓名；留空时读取项目的 Git 配置",
        "email": "Git 提交者邮箱；留空时读取项目的 Git 配置",
        "token": "填写自己的有效 Token；留空则使用已有 Git / SSH 凭据",
        "code_only": "true：仅上传当前代码，跳过旧历史与 LFS 资源",
        "create_ignore": "缺少 .gitignore 时创建基础忽略规则",
    }
    lines = [
        "# GitHub 上传参数（本机私有文件，请放在项目目录之外）",
        "# 修改后重新启动窗口，或点击「重新读取」。",
        "",
    ]
    for key, value in values.items():
        encoded = str(value).lower() if isinstance(value, bool) else json.dumps(value, ensure_ascii=False)
        lines.extend((f"# {comments[key]}", f"{key} = {encoded}", ""))
    try:
        _private_write(path.expanduser(), "\n".join(lines))
    except OSError as exc:
        raise ConfigError(f"无法保存参数文件，请检查目录权限：\n{path}") from exc


def require_external_config(path: Path, directory: Path) -> None:
    """Prevent an active credential file from being included in a project upload."""
    if path.expanduser().resolve().is_relative_to(directory.expanduser().resolve()):
        raise ConfigError(
            "请把参数文件放在待上传项目之外，例如 ~/.config/lerobot-github-upload/config.toml。"
        )


def desktop_directory() -> Path:
    """Find localized Linux desktops without evaluating a user-dirs shell file."""
    executable = shutil.which("xdg-user-dir")
    if executable:
        try:
            result = subprocess.run(
                [executable, "DESKTOP"], capture_output=True, text=True, timeout=5, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    return Path.home() / ("桌面" if (Path.home() / "桌面").is_dir() else "Desktop")


def desktop_entry(launcher: Path, config: Path) -> str:
    """Build a shell-free desktop command; credentials remain solely in the config file."""

    def quote(value: Path | str) -> str:
        text = str(value).replace("%", "%%")
        for character in ("\\", '"', "`", "$"):
            text = text.replace(character, "\\" + character)
        # Desktop string unescaping happens before Exec argument unquoting.
        return '"' + text.replace("\\", "\\\\") + '"'

    command = " ".join(
        ("/bin/bash", quote(launcher.resolve()), "--desktop", "--config", quote(config.resolve()))
    )
    icon = str(launcher.resolve().parent / "assets" / APPLICATION_ICON)
    icon = icon.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
    return (
        "[Desktop Entry]\nVersion=1.0\nType=Application\nName=GitHub 代码上传\n"
        "Comment=上传本地代码，自动读取本机参数\n"
        f"Exec={command}\nTryExec=/bin/bash\nIcon={icon}\nTerminal=false\n"
        "Categories=Development;RevisionControl;\nKeywords=GitHub;Git;上传;代码;\nStartupNotify=false\n"
        f"StartupWMClass={APPLICATION_ID}\n"
    )


def install_desktop_launcher(launcher: Path, config: Path, *, desktop: Path | None = None) -> Path:
    """Install a desktop icon and application-menu entry using absolute paths."""
    launcher = launcher.expanduser().resolve()
    if not launcher.is_file():
        raise ConfigError("找不到上传工具的启动脚本，请从项目中的 utils/github_upload_gui.py 运行安装命令。")
    content = desktop_entry(launcher, config.expanduser())
    desktop = desktop if desktop is not None else desktop_directory()
    shortcut = desktop / "GitHub代码上传.desktop"
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    try:
        _private_write(shortcut, content, 0o755)
        _private_write(data_home / "applications" / "lerobot-github-upload.desktop", content, 0o755)
    except OSError as exc:
        raise ConfigError("无法创建桌面启动图标，请检查桌面和应用菜单目录的权限。") from exc
    return shortcut
