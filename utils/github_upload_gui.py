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

"""Upload a local project to GitHub with a modern Chinese desktop interface.

Run ``bash utils/github_upload_gui.sh`` to locate or prepare a GUI environment.
Git operations use the standard library; the window uses the project's PyQt5 dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import queue
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

if __package__:
    from .github_upload_config import (
        ConfigError,
        UploadPreferences,
        default_config_path,
        install_desktop_launcher,
        load_preferences,
        require_external_config,
        save_preferences,
    )
else:
    from github_upload_config import (
        ConfigError,
        UploadPreferences,
        default_config_path,
        install_desktop_launcher,
        load_preferences,
        require_external_config,
        save_preferences,
    )

DEFAULT_IGNORE = """# Local environments and generated files
.venv/
.cache/
venv/
__pycache__/
*.py[cod]
.env
.env.*
!.env.example
*.pem
*.key
node_modules/
.DS_Store
outputs/
wandb/
"""
TOKEN_ENV = "LEROBOT_GITHUB_UPLOAD_TOKEN"
USERNAME_ENV = "LEROBOT_GITHUB_UPLOAD_USERNAME"
AUTH_HELP = "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens"


class UploadError(RuntimeError):
    """An actionable validation or Git error suitable for display in the GUI."""


@dataclass(frozen=True)
class UploadSettings:
    """Values copied from the GUI before starting a worker thread."""

    directory: Path
    remote: str
    branch: str
    message: str
    name: str = ""
    email: str = ""
    token: str = field(default="", repr=False)
    create_ignore: bool = True
    code_only: bool = False
    username: str = ""


@dataclass(frozen=True)
class Preview:
    """Repository state shown before committing the entire working tree."""

    branch: str
    head: str
    status: str
    origin: str
    name: str
    email: str
    signature: str = ""
    code_only: bool = False
    files: tuple[str, ...] = ()
    excluded_lfs: int = 0
    file_sizes: tuple[tuple[str, int], ...] = ()

    @property
    def total_bytes(self) -> int:
        """Size of selected working files before Git compression or delta encoding."""
        return sum(size for _name, size in self.file_sizes)


@dataclass(frozen=True)
class UploadProgress:
    """Progress for one real Git or file preparation stage, not an estimated overall percentage."""

    phase: str
    fraction: float | None = None
    detail: str = ""
    transferred_bytes: int | None = None
    total_bytes: int | None = None
    speed_bytes: float | None = None


def format_size(size: int | float) -> str:
    """Display byte counts consistently across file previews and live network progress."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "0 B"


def parse_git_progress(line: str) -> UploadProgress | None:
    """Read Git's C-locale progress, including compressed pack bytes and LFS output."""
    match = re.search(
        r"(Counting objects|Compressing objects|Writing objects|Uploading LFS objects):\s*(\d+)%\s*\((\d+)/(\d+)\)",
        line,
    )
    if not match:
        return None
    name, percent, done, total = match.groups()
    phase = {
        "Counting objects": "统计对象",
        "Compressing objects": "压缩打包",
        "Writing objects": "发送代码",
        "Uploading LFS objects": "发送 LFS 资源",
    }[name]
    sizes = re.search(
        r",\s*([\d.]+)\s*(bytes|[kKMGT]?i?B)(?:\s*\|\s*([\d.]+)\s*(bytes|[kKMGT]?i?B)/s)?", line
    )

    def as_bytes(number: str, unit: str) -> float:
        power = {"B": 0, "K": 1, "M": 2, "G": 3, "T": 4}[unit[0].upper()]
        return float(number) * (1024 if "i" in unit else 1000) ** power

    transferred = int(as_bytes(sizes[1], sizes[2])) if sizes else None
    speed = as_bytes(sizes[3], sizes[4]) if sizes and sizes[3] else None
    return UploadProgress(
        phase, min(int(percent), 100) / 100, f"{done} / {total} 个对象", transferred, speed_bytes=speed
    )


def normalize_remote(value: str) -> str:
    """Accept a GitHub HTTPS/SSH repository URL without embedded credentials."""
    value = value.strip()
    patterns = (
        r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?",
        r"git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?",
        r"ssh://git@github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value)
        if match and all(part not in {".", ".."} for part in match.groups()):
            return value.rstrip("/")
    raise UploadError("请输入 GitHub 仓库的 HTTPS 或 SSH 地址；地址中不能包含密码或 Token。")


def credential_helper(action: str) -> None:
    """Supply a temporary token only to Git's github.com HTTPS credential query."""
    if action != "get":
        return
    values = {}
    for line in sys.stdin:
        if not line.strip():
            break
        key, separator, value = line.rstrip("\n").partition("=")
        if separator:
            values[key] = value
    token = os.environ.get(TOKEN_ENV, "")
    username = os.environ.get(USERNAME_ENV, "") or "x-access-token"
    if any(character in username or character in token for character in ("\n", "\r", "\0")):
        return
    if values.get("protocol") == "https" and values.get("host") == "github.com" and token:
        sys.stdout.write(f"username={username}\npassword={token}\n\n")


class GitRepo:
    """Run Git without a shell and keep repository operations independent of Tk."""

    def __init__(
        self,
        directory: Path,
        log: Callable[[str], None] = lambda _text: None,
        progress: Callable[[UploadProgress], None] = lambda _progress: None,
    ):
        """Bind operations to one explicit local directory."""
        self.directory = directory.expanduser().resolve()
        self.log = log
        self.progress = progress
        self._last_progress: tuple[str, float] = ("", 0)
        self.git = shutil.which("git")
        if not self.git:
            raise UploadError("找不到 Git，请先安装 Git 并重新启动本工具。")
        if not self.directory.is_dir():
            raise UploadError("本地目录不存在，请选择要上传的项目文件夹。")

    def run(
        self,
        *args: str,
        check: bool = True,
        network: bool = False,
        token: str = "",
        username: str = "",
        input_text: str | None = None,
        stream_progress: bool = False,
    ) -> subprocess.CompletedProcess:
        """Execute a bounded, noninteractive Git command with optional ephemeral auth."""
        env = os.environ.copy()
        # A terminal's Git context must never redirect operations away from the selected directory.
        for key in tuple(env):
            if key.startswith("GIT_") or key in {TOKEN_ENV, USERNAME_ENV}:
                env.pop(key)
        env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", LC_ALL="C")
        command = [self.git, "-c", "core.quotepath=false", "-c", "color.ui=false"]
        if network:
            ssh = self.run("config", "--get", "core.sshCommand", check=False).stdout.strip() or "ssh"
            env["GIT_SSH_COMMAND"] = ssh + " -oBatchMode=yes"
            if token:
                if any(character in token or character in username for character in ("\n", "\r", "\0")):
                    raise UploadError("用户名和 Token 不能包含换行符或空字符。")
                helper = "!" + shlex.join(
                    [sys.executable, str(Path(__file__).resolve()), "--credential-helper"]
                )
                command += ["-c", "credential.helper=", "-c", f"credential.helper={helper}"]
                env[TOKEN_ENV] = token
                env[USERNAME_ENV] = username
        command.extend(args)
        try:
            if stream_progress:
                if input_text is not None:
                    raise ValueError("Streaming Git commands do not accept input_text")
                result = self._run_streaming(command, env, 900 if network else 180, token)
            else:
                result = subprocess.run(
                    command,
                    cwd=self.directory,
                    env=env,
                    stdin=subprocess.DEVNULL if input_text is None else None,
                    input=input_text,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=900 if network else 180,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise UploadError("Git 操作超时。请检查网络或 Git hooks；重试前先检查仓库状态。") from exc
        if token:
            result.stdout = result.stdout.replace(token, "[已隐藏 Token]")
            result.stderr = result.stderr.replace(token, "[已隐藏 Token]")
        if check and result.returncode:
            raise UploadError((result.stderr or result.stdout).strip() or "Git 命令执行失败。")
        return result

    def _report(self, state: UploadProgress) -> None:
        now = time.monotonic()
        previous_phase, previous_time = self._last_progress
        if state.phase != previous_phase or state.fraction in (None, 0, 1) or now - previous_time >= 0.06:
            self.progress(state)
            self._last_progress = (state.phase, now)

    def _run_streaming(
        self, command: list[str], env: dict[str, str], timeout: float, token: str = ""
    ) -> subprocess.CompletedProcess:
        """Drain both pipes concurrently and publish stderr progress before Git exits."""
        events: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
        output: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}

        def read_pipe(label: str, stream) -> None:
            try:
                while chunk := stream.read1(65536):
                    events.put((label, chunk))
            finally:
                events.put((label, None))

        with subprocess.Popen(
            command,
            cwd=self.directory,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        ) as process:
            readers = [
                threading.Thread(target=read_pipe, args=(label, stream), daemon=True)
                for label, stream in (("stdout", process.stdout), ("stderr", process.stderr))
            ]
            for reader in readers:
                reader.start()
            pending = b""
            ended = 0
            deadline = time.monotonic() + timeout

            def report_line(raw: bytes) -> None:
                line = raw.decode("utf-8", errors="replace")
                if token:
                    line = line.replace(token, "[已隐藏 Token]")
                state = parse_git_progress(line)
                if state:
                    self._report(state)
                    if state.phase == "发送代码" and state.fraction == 1:
                        self._report(
                            UploadProgress("等待远端确认", detail="代码已发送，正在等待服务器处理结果。")
                        )

            try:
                while ended < 2:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        label, chunk = events.get(timeout=min(0.1, remaining))
                    except queue.Empty:
                        continue
                    if chunk is None:
                        ended += 1
                        continue
                    output[label].extend(chunk)
                    if label == "stderr":
                        parts = re.split(b"[\r\n]", pending + chunk)
                        pending = parts.pop()
                        for part in parts:
                            report_line(part)
                if pending:
                    report_line(pending)
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                for reader in readers:
                    reader.join(timeout=1)
            return subprocess.CompletedProcess(
                command,
                process.returncode,
                output["stdout"].decode("utf-8", errors="replace"),
                output["stderr"].decode("utf-8", errors="replace"),
            )

    def validate_branch(self, branch: str) -> None:
        """Reject invalid destination refs before any repository mutation."""
        if (
            not branch
            or branch.startswith("-")
            or self.run("check-ref-format", f"refs/heads/{branch}", check=False).returncode
        ):
            raise UploadError("目标分支名称无效，例如可以填写 main 或 feature/my-code。")

    def prepare(self, branch: str = "main", create_ignore: bool = True) -> None:
        """Initialize a plain folder or validate the exact root of an existing worktree."""
        self.validate_branch(branch)
        root = self.run("rev-parse", "--show-toplevel", check=False)
        if root.returncode == 0:
            if Path(root.stdout.strip()).resolve() != self.directory:
                raise UploadError(f"所选目录是已有仓库的子目录，请选择仓库根目录：\n{root.stdout.strip()}")
        else:
            bare = self.run("rev-parse", "--is-bare-repository", check=False)
            if (self.directory / ".git").exists() or bare.returncode == 0:
                raise UploadError("该目录不是可用的 Git 工作区，请检查仓库状态。")
            self.log("初始化本地 Git 仓库……")
            self.run("init", "-b", branch)
        self._check_state()
        ignore = self.directory / ".gitignore"
        if create_ignore and not ignore.exists() and not ignore.is_symlink():
            with ignore.open("x", encoding="utf-8") as stream:
                stream.write(DEFAULT_IGNORE)
            self.log("已创建基础 .gitignore；请按项目需要补充数据、模型和私密文件的忽略规则。")

    def _check_state(self) -> None:
        if self.run("symbolic-ref", "--quiet", "--short", "HEAD", check=False).returncode:
            raise UploadError("当前处于 detached HEAD，请先在 Git 中切换或创建本地分支。")
        if self.run("ls-files", "--unmerged").stdout:
            raise UploadError("仓库有未解决的合并冲突，请先解决冲突。")
        for marker in (
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "rebase-merge",
            "rebase-apply",
            "sequencer",
            "BISECT_LOG",
        ):
            path = Path(self.run("rev-parse", "--git-path", marker).stdout.strip())
            if not path.is_absolute():
                path = self.directory / path
            if path.exists():
                raise UploadError("仓库正在合并、变基、拣选或二分查找，请先完成或退出该操作。")

    def preview(self, *, code_only: bool = False) -> Preview:
        """Read the branch, identity and complete staged/unstaged/untracked file list."""
        self._check_state()
        signature = hashlib.sha256()
        signature.update(self.run("ls-files", "--stage", "-z").stdout.encode())
        signature.update(self.run("diff", "--raw", "--no-ext-diff").stdout.encode())
        candidates = self.run(
            "ls-files", "--modified", "--deleted", "--others", "--exclude-standard", "-z"
        ).stdout
        candidates += self.run("diff", "--cached", "--name-only", "--no-renames", "-z").stdout
        for name in sorted(set(candidates.split("\0")) - {""}):
            try:
                info = (self.directory / name).lstat()
                state = (name, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            except FileNotFoundError:
                state = (name, None)
            signature.update(repr(state).encode())
        preview = Preview(
            branch=self.run("symbolic-ref", "--short", "HEAD").stdout.strip(),
            head=self.run("rev-parse", "--verify", "HEAD", check=False).stdout.strip(),
            status=self.run("status", "--short", "--untracked-files=all").stdout.rstrip(),
            origin=self.run("config", "--get", "remote.origin.url", check=False).stdout.strip(),
            name=self.run("config", "--get", "user.name", check=False).stdout.strip(),
            email=self.run("config", "--get", "user.email", check=False).stdout.strip(),
            signature=signature.hexdigest(),
        )
        if code_only:
            files, excluded = self._snapshot_files()
            preview = replace(
                preview,
                code_only=True,
                files=files,
                excluded_lfs=excluded,
                status="\n".join("?? " + repr(name) if "\n" in name else "?? " + name for name in files),
            )
            size_names = files
        else:
            # NUL separators keep sizes aligned with status rows even for renames and newline filenames.
            entries = self.run("status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.split("\0")
            names = []
            index = 0
            while index < len(entries) and entries[index]:
                entry = entries[index]
                names.append(entry[3:])
                index += 2 if "R" in entry[:2] or "C" in entry[:2] else 1
            size_names = tuple(names)
        sizes = []
        for filename in size_names:
            try:
                info = (self.directory / filename).lstat()
                size = info.st_size if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) else 0
            except FileNotFoundError:
                size = 0
            sizes.append((filename, size))
        preview = replace(preview, file_sizes=tuple(sizes))
        return preview

    def _snapshot_files(self) -> tuple[tuple[str, ...], int]:
        names = self.run("ls-files", "--cached", "--others", "--exclude-standard", "-z").stdout
        files = sorted(
            {
                name
                for name in names.split("\0")
                if name and ((self.directory / name).is_file() or (self.directory / name).is_symlink())
            }
        )
        if not files:
            return (), 0
        attributes = self.run(
            "check-attr", "-z", "--stdin", "filter", input_text="\0".join(files) + "\0"
        ).stdout.split("\0")
        excluded = {
            attributes[index] for index in range(0, len(attributes) - 2, 3) if attributes[index + 2] == "lfs"
        }
        return tuple(name for name in files if name not in excluded), len(excluded)

    def _copy_connection_config(self, destination: GitRepo) -> None:
        # Keep this repository's verified proxy settings in the isolated upload repository.
        # TemporaryDirectory is private and discarded after the operation; GUI tokens never enter config.
        result = self.run("config", "--local", "--get-regexp", r"^(http\.|url\.)", check=False)
        http_options = {"proxy", "version", "sslverify", "sslcainfo", "sslcapath", "proxysslcainfo"}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition(" ")
            if separator and (
                (key.startswith("http.") and key.rsplit(".", 1)[-1].lower() in http_options)
                or (key.startswith("url.") and key.lower().endswith(".insteadof"))
            ):
                destination.run("config", "--local", "--add", key, value)

    def _upload_snapshot(self, settings: UploadSettings, preview: Preview, name: str, email: str) -> str:
        if not preview.files:
            raise UploadError("没有可上传的代码文件；当前代码模式会跳过 Git LFS 资源。")
        self.log(
            f"仅上传当前代码：{len(preview.files)} 个文件，共 {format_size(preview.total_bytes)}（未压缩），"
            f"跳过 {preview.excluded_lfs} 个 LFS 资源。"
        )
        self.log("使用独立上传仓库，不携带 LeRobot 的旧提交历史。")
        remote = normalize_remote(settings.remote)
        with tempfile.TemporaryDirectory(prefix="github-code-upload-") as temporary:
            target = GitRepo(Path(temporary), self.log, self.progress)
            target.prepare(settings.branch, create_ignore=False)
            self._copy_connection_config(target)
            # A later upload builds on this personal repository's current branch, without force pushing.
            self._report(UploadProgress("连接目标仓库", detail="检查目标分支和访问权限。"))
            heads = target.run(
                "ls-remote",
                "--heads",
                remote,
                f"refs/heads/{settings.branch}",
                network=True,
                token=settings.token,
                username=settings.username,
            ).stdout.strip()
            if heads:
                self.log("读取个人仓库的最新版本，后续更新会接在这个版本之后。")
                self._report(UploadProgress("同步目标分支", detail="读取个人仓库的最新版本。"))
                target.run(
                    "fetch",
                    "--depth=1",
                    "--no-tags",
                    remote,
                    f"refs/heads/{settings.branch}",
                    network=True,
                    token=settings.token,
                    username=settings.username,
                )
                # Populate only the index and parent commit. No remote working files or LFS data are downloaded.
                target.run("reset", "--mixed", "FETCH_HEAD")
            copied_bytes = 0
            sizes = dict(preview.file_sizes)
            for index, filename in enumerate(preview.files, start=1):
                source = self.directory / filename
                copied = target.directory / filename
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, copied, follow_symlinks=False)
                copied_bytes += sizes[filename]
                self._report(
                    UploadProgress(
                        "准备文件",
                        index / len(preview.files),
                        f"{index} / {len(preview.files)} 个文件",
                        copied_bytes,
                        preview.total_bytes,
                    )
                )
            if self.preview(code_only=True) != preview:
                raise UploadError("生成快照时文件发生变化，请重新预览并上传。")
            self._report(UploadProgress("生成提交", detail="正在暂存代码快照并创建版本。"))
            target.run("add", "--all", "--", ".")
            # Preserve files already tracked in the source even when its ignore rules match them.
            target.run(
                "--literal-pathspecs",
                "add",
                "--force",
                "--pathspec-from-file=-",
                "--pathspec-file-nul",
                input_text="\0".join(preview.files) + "\0",
            )
            changed = target.run("diff", "--cached", "--quiet", check=False)
            if changed.returncode not in (0, 1):
                raise UploadError(changed.stderr.strip() or "无法检查代码快照。")
            if changed.returncode == 1:
                committed = target.run(
                    "-c",
                    f"user.name={name}",
                    "-c",
                    f"user.email={email}",
                    "commit",
                    "-m",
                    settings.message,
                )
                self.log(committed.stdout.strip())
            else:
                self.log("代码与个人仓库一致，无需创建重复提交。")
            head = target.run("rev-parse", "HEAD").stdout.strip()
            self.log(f"正在上传代码快照 {head[:12]} → {remote}，目标分支：{settings.branch}……")
            target._report(UploadProgress("开始上传", detail="连接远端并准备打包。"))
            pushed = target.run(
                "-c",
                "push.followTags=false",
                "push",
                "--porcelain",
                "--progress",
                "--recurse-submodules=no",
                remote,
                f"HEAD:refs/heads/{settings.branch}",
                network=True,
                token=settings.token,
                username=settings.username,
                stream_progress=True,
            )
            self.log(pushed.stdout.strip())
            return head

    def upload(self, settings: UploadSettings, expected: Preview | None = None) -> str:
        """Commit all current changes and push HEAD to one explicit GitHub branch."""
        remote = normalize_remote(settings.remote)
        self.validate_branch(settings.branch)
        if not settings.message.strip():
            raise UploadError("请填写提交说明。")
        if settings.token and not remote.startswith("https://"):
            raise UploadError("Token 适用于 HTTPS 地址；使用 SSH 地址时请清空 Token。")
        if any(
            character in settings.token or character in settings.username for character in ("\n", "\r", "\0")
        ):
            raise UploadError("用户名和 Token 不能包含换行符或空字符。")
        self._report(UploadProgress("检查项目", detail="核对预览文件和 Git 状态。"))
        self.prepare(settings.branch, settings.create_ignore)
        before = self.preview(code_only=settings.code_only)
        if expected is not None and before != expected:
            raise UploadError("预览后文件或分支已发生变化，请重新预览并确认上传。")
        name = settings.name.strip() or before.name
        email = settings.email.strip() or before.email
        if before.status and (not name or not email):
            raise UploadError("请填写提交者姓名和邮箱，或先配置 Git 的 user.name / user.email。")
        if settings.code_only:
            try:
                return self._upload_snapshot(settings, before, name, email)
            except UploadError as exc:
                raise UploadError(
                    f"当前代码尚未确认上传成功，原项目文件和历史已保留，可重试。\n\n{exc}"
                ) from exc
        self._report(UploadProgress("生成提交", detail="暂存改动并创建本地版本。"))
        self.log("暂存全部改动（包含新增、修改、删除及已有暂存内容）……")
        self.run("add", "--all", "--", ".")
        changed = self.run("diff", "--cached", "--quiet", check=False)
        if changed.returncode not in (0, 1):
            raise UploadError(changed.stderr.strip() or "无法检查暂存区。")
        if changed.returncode == 1:
            self.log("创建本地提交……")
            result = self.run(
                "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-m", settings.message
            )
            self.log(result.stdout.strip())
        elif not before.head:
            raise UploadError("目录中没有可提交的文件，请添加代码或调整 .gitignore。")
        else:
            self.log("没有新的文件改动，将推送现有提交。")
        head = self.run("rev-parse", "HEAD").stdout.strip()
        self.log(f"正在推送 {head[:12]} → {remote}，目标分支：{settings.branch}……")
        self._report(UploadProgress("开始上传", detail="连接远端并准备打包。"))
        try:
            result = self.run(
                "-c",
                "push.followTags=false",
                "push",
                "--porcelain",
                "--progress",
                "--recurse-submodules=no",
                remote,
                f"HEAD:refs/heads/{settings.branch}",
                network=True,
                token=settings.token,
                username=settings.username,
                stream_progress=True,
            )
        except UploadError as exc:
            raise UploadError(
                f"推送未确认成功；本地提交 {head[:12]} 已保留，修复后可重试。\n\n{exc}\n\n"
                "鉴权失败：检查 Token 权限或已配置的 SSH 密钥。\n"
                "仓库不存在：先在 GitHub 创建仓库。\n"
                "被拒绝（non-fast-forward / fetch first）：请先在 Git 中处理远端分歧。"
            ) from exc
        self.log(result.stdout.strip())
        return head


def launch(
    directory: Path,
    *,
    demo: bool = False,
    preferences: UploadPreferences | None = None,
    config_path: Path | None = None,
) -> None:
    """Load the optional desktop UI only when actually opening a window."""
    try:
        if __package__:
            from .github_upload_window import launch as open_window
        else:
            from github_upload_window import launch as open_window
    except ImportError as exc:
        raise UploadError(
            "缺少图形界面依赖，请运行 bash utils/github_upload_gui.sh 启动并自动准备环境。"
        ) from exc
    try:
        open_window(directory, demo=demo, preferences=preferences, config_path=config_path)
    except RuntimeError as exc:
        raise UploadError(str(exc)) from exc


def show_startup_error(message: str) -> None:
    """Make invalid local parameters visible when a desktop shortcut has no terminal."""
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
    except ImportError:
        return
    app = QApplication.instance() or QApplication(sys.argv[:1])
    QMessageBox.critical(None, "GitHub 上传工具无法启动", message)
    # Retain the application until the native message box has closed.
    _ = app


def main() -> int:
    """Parse the optional starting directory or serve Git's credential-helper protocol."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--credential-helper":
        credential_helper(sys.argv[2])
        return 0
    parser = argparse.ArgumentParser(description="GitHub 代码上传助手（Qt 深色桌面界面）")
    parser.add_argument("directory", nargs="?", type=Path, help="项目目录；省略时从参数文件读取")
    parser.add_argument("--config", type=Path, default=default_config_path(), help="本机 TOML 参数文件路径")
    parser.add_argument("--desktop", action="store_true", help=argparse.SUPPRESS)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--install-launcher", action="store_true", help="创建桌面双击启动图标和参数文件")
    action.add_argument("--demo", action="store_true", help="仅预览示例界面，不读写仓库或上传")
    args = parser.parse_args()
    try:
        config = args.config.expanduser().resolve()
        preferences = UploadPreferences() if args.demo else load_preferences(config)
        directory = args.directory or Path(preferences.directory or Path(__file__).resolve().parent.parent)
        directory = directory.expanduser()
        if not directory.is_absolute():
            directory = (Path.cwd() if args.directory else config.parent) / directory
        directory = directory.resolve()
        if not args.demo:
            require_external_config(config, directory)
        if args.install_launcher:
            if not config.exists():
                save_preferences(config, replace(preferences, directory=str(directory)))
            shortcut = install_desktop_launcher(Path(__file__).with_suffix(".sh"), config)
            print(f"已创建双击启动图标：{shortcut}\n参数文件：{config}")
            return 0
        launch(directory, demo=args.demo, preferences=preferences, config_path=config)
    except (UploadError, ConfigError) as exc:
        print(str(exc), file=sys.stderr)
        if args.desktop:
            show_startup_error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
