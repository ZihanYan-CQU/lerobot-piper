#!/usr/bin/env python3
"""Share one desktop authorization over a window's private child-process pipes."""

from __future__ import annotations

import io
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import redirect_stderr, redirect_stdout, suppress
from pathlib import Path

if __package__:
    from . import camera_aliases as cameras
    from . import can_port_bindings as can
else:
    import camera_aliases as cameras
    import can_port_bindings as can

MAX_MESSAGE = 65536


def helper_command() -> list[str]:
    """Authorize the helper only; discovery, previews, and GTK stay unprivileged."""
    command = ["/usr/bin/python3", str(Path(__file__).resolve())]
    if os.geteuid() != 0:
        pkexec = shutil.which("pkexec")
        if pkexec is None:
            raise RuntimeError("找不到 pkexec，请安装系统的 policykit-1 软件包。")
        command.insert(0, pkexec)
    return command


class PrivilegedSession:
    """Lazily start one helper and serialize camera/CAN requests until window close."""

    def __init__(self, timeout: float = 300):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.errors = None
        self.closed = False

    def execute(self, request: dict) -> str:
        """Send one checked operation; never replay a request after a transport failure."""
        message = json.dumps(request, ensure_ascii=True) + "\n"
        if len(message) > MAX_MESSAGE:
            raise ValueError("配置请求过大，请减少一次提交的设备数量。")
        with self.lock:
            if self.closed:
                raise RuntimeError("设备配置窗口已关闭。")
            if self.process is not None and self.process.poll() is not None:
                self._stop()
            if self.process is not None and self.process.stdin.closed:
                raise RuntimeError("上次配置仍在结束中，请稍候刷新后重试。")
            if self.process is None:
                command = helper_command()
                self.errors = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
                try:
                    self.process = subprocess.Popen(
                        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=self.errors, text=True, encoding="utf-8", bufsize=1,
                    )
                except OSError:
                    self.errors.close()
                    self.errors = None
                    raise
            try:
                self.process.stdin.write(message)
                self.process.stdin.flush()
                if not select.select([self.process.stdout], [], [], self.timeout)[0]:
                    raise RuntimeError("配置执行超时，操作结果未确认；请刷新检查设备状态。")
                line = self.process.stdout.readline(MAX_MESSAGE + 1)
                if not line:
                    self.process.wait(timeout=2)
                    if self.process.returncode == 126:
                        raise RuntimeError("已取消系统授权，设备配置未修改。")
                    self.errors.seek(0)
                    detail = self.errors.read(4096).strip()
                    raise RuntimeError(detail or "系统授权进程已退出，请重试。")
                if len(line) > MAX_MESSAGE or not line.endswith("\n"):
                    raise ValueError("系统授权进程返回了无效响应。")
                reply = json.loads(line)
                if not isinstance(reply, dict) or type(reply.get("ok")) is not bool:
                    raise ValueError("系统授权进程返回了无效响应。")
                field = "output" if reply["ok"] else "error"
                if not isinstance(reply.get(field), str):
                    raise ValueError("系统授权进程返回了无效响应。")
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                self._stop()
                raise
            if not reply["ok"]:
                # A rejected selection does not end an otherwise healthy authorization.
                raise RuntimeError(reply["error"])
            return reply["output"]

    def _stop(self) -> None:
        """Close stdin so the helper exits on EOF, including after parent failure."""
        if self.process is None:
            return
        with suppress(OSError):
            self.process.stdin.close()
        # A helper still writing a failed/oversized reply must also see a closed pipe.
        self.process.stdout.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # Keep the handle: another save must not race an unfinished operation.
            return
        self.errors.close()
        self.process = self.errors = None

    def close(self) -> None:
        """End this window's authorization without changing any system-wide policy."""
        with self.lock:
            self.closed = True
            self._stop()
            if self.process is not None:
                process, errors = self.process, self.errors

                def reap():
                    process.wait()
                    process.stdout.close()
                    errors.close()

                threading.Thread(target=reap, daemon=True).start()
                self.process = self.errors = None


def string_list(request: dict, key: str) -> list[str]:
    """Require explicit mappings and identity snapshots rather than CLI defaults."""
    value = request.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"设备配置参数 {key} 无效或不完整，请刷新面板。")
    return value


def dispatch(request: dict) -> str:
    """Allow only camera bindings or selected arm roles, using the same CLI validators."""
    if not isinstance(request, dict):
        raise ValueError("设备配置请求格式无效。")
    expected = string_list(request, "expected")
    operation = request.get("operation")
    if operation == "camera":
        mappings = string_list(request, "mappings")
        # Parse first, and use --option=value so values cannot become CLI options.
        for value in mappings:
            cameras.parse_mapping(value)
        entry = cameras.main
        args = [f"--map={value}" for value in mappings]
        args.extend(f"--expect-camera={value}" for value in expected)
    elif operation == "can":
        master, follower, bitrate = (request.get(key) for key in ("master", "follower", "bitrate"))
        if not all(value is None or isinstance(value, str) for value in (master, follower)):
            raise ValueError("CAN 角色参数无效。")
        if not (master or follower) or type(bitrate) is not int:
            raise ValueError("请选择至少一个 CAN 端口并填写波特率。")
        entry = can.main
        args = [f"--bitrate={bitrate}"]
        for role, usb in (("master", master), ("follower", follower)):
            if usb:
                args.append(f"--{role}={usb}")
        args.extend(f"--expect-port={value}" for value in expected)
    else:
        raise ValueError("不支持的设备配置操作。")
    output, errors = io.StringIO(), io.StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        status = entry(args)
    if status:
        raise RuntimeError((errors.getvalue() + "\n" + output.getvalue()).strip())
    return output.getvalue()


def serve(input_stream, output_stream) -> None:
    """Serve a private pipe until EOF; there is no socket or general command interface."""
    if os.geteuid() != 0:
        raise PermissionError("设备配置辅助进程需要系统授权。")
    while line := input_stream.readline(MAX_MESSAGE + 1):
        if len(line) > MAX_MESSAGE or not line.endswith("\n"):
            raise ValueError("设备配置请求过大或不完整。")
        try:
            response = {"ok": True, "output": dispatch(json.loads(line))}
        except (OSError, ValueError, RuntimeError) as exc:
            response = {"ok": False, "error": str(exc)}
        output_stream.write(json.dumps(response, ensure_ascii=True) + "\n")
        output_stream.flush()


if __name__ == "__main__":
    try:
        serve(sys.stdin, sys.stdout)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
