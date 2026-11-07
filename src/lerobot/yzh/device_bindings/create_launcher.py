#!/usr/bin/env python3
"""Create an executable desktop shortcut for this copy of the device panel."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

LAUNCHER_NAME = "Piper设备配置.desktop"
MARKER = "X-LeRobot-DeviceBindings=true"


def desktop_entry(script: Path) -> str:
    """Quote the script as one Exec argument, including literal field-code characters."""
    argument = str(script.resolve()).replace("%", "%%")
    for character in ("\\", '"', "`", "$"):
        argument = argument.replace(character, "\\" + character)
    command = f'/usr/bin/python3 "{argument}"'
    # Desktop string escapes are decoded before the Exec argument escapes.
    command = command.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return (
        "[Desktop Entry]\nType=Application\nVersion=1.0\n"
        "Name=Piper 设备配置\nComment=配置和检查相机别名、机械臂 CAN 接口\n"
        f"Exec={command}\nTryExec=/usr/bin/python3\n"
        "Icon=camera-photo\nTerminal=false\nStartupNotify=true\n"
        f"Categories=Settings;HardwareSettings;\n{MARKER}\n"
    )


def desktop_directory() -> Path:
    """Use the user's configured desktop, including localized directory names."""
    executable = shutil.which("xdg-user-dir")
    if executable:
        result = subprocess.run([executable, "DESKTOP"], capture_output=True, text=True, check=True, timeout=5)
        directory = Path(result.stdout.strip())
        if directory.is_absolute():
            return directory
    return Path.home() / "Desktop"


def write_launcher(target: Path, script: Path) -> None:
    """Replace only our own launcher and mark it executable."""
    if target.is_symlink() or (target.exists() and MARKER not in target.read_text()):
        raise ValueError(f"{target} 不是本工具的快捷方式，拒绝覆盖。")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(desktop_entry(script), encoding="utf-8")
    target.chmod(0o755)


def main(argv: list[str] | None = None) -> int:
    """Generate a folder launcher or install one on the user's desktop."""
    parser = argparse.ArgumentParser(description="创建可双击的 Piper 设备配置快捷方式，无需 sudo。")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--desktop", action="store_true", help="创建到当前用户的桌面")
    destination.add_argument("--output", type=Path, help="指定快捷方式的保存位置")
    args = parser.parse_args(argv)
    try:
        root = Path(__file__).resolve().parent
        target = args.output or ((desktop_directory() if args.desktop else root) / LAUNCHER_NAME)
        write_launcher(target, root / "camera_aliases_gui.py")
        print(f"快捷方式已创建：{target}")
        if args.desktop:
            gio = "/usr/bin/gio" if Path("/usr/bin/gio").is_file() else shutil.which("gio")
            trusted = False
            if gio:
                try:
                    result = subprocess.run(
                        [gio, "set", str(target), "metadata::trusted", "true"],
                        capture_output=True, text=True, timeout=5,
                    )
                    trusted = result.returncode == 0
                except (OSError, subprocess.SubprocessError):
                    pass  # The file is usable after the desktop's "Allow Launching" action.
            if not trusted:
                print("如桌面提示未信任，请右键快捷方式并选择“允许启动”。")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"创建快捷方式失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
