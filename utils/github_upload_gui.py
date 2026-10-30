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
Git operations use the standard library; the window uses CustomTkinter and Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

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
    if values.get("protocol") == "https" and values.get("host") == "github.com" and token:
        sys.stdout.write(f"username=x-access-token\npassword={token}\n\n")


class GitRepo:
    """Run Git without a shell and keep repository operations independent of Tk."""

    def __init__(self, directory: Path, log: Callable[[str], None] = lambda _text: None):
        """Bind operations to one explicit local directory."""
        self.directory = directory.expanduser().resolve()
        self.log = log
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
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Execute a bounded, noninteractive Git command with optional ephemeral auth."""
        env = os.environ.copy()
        # A terminal's Git context must never redirect operations away from the selected directory.
        for key in tuple(env):
            if key.startswith("GIT_") or key == TOKEN_ENV:
                env.pop(key)
        env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", LC_ALL="C")
        command = [self.git, "-c", "core.quotepath=false", "-c", "color.ui=false"]
        if network:
            ssh = self.run("config", "--get", "core.sshCommand", check=False).stdout.strip() or "ssh"
            env["GIT_SSH_COMMAND"] = ssh + " -oBatchMode=yes"
            if token:
                if "\n" in token or "\r" in token:
                    raise UploadError("Token 不能包含换行符。")
                helper = "!" + shlex.join(
                    [sys.executable, str(Path(__file__).resolve()), "--credential-helper"]
                )
                command += ["-c", "credential.helper=", "-c", f"credential.helper={helper}"]
                env[TOKEN_ENV] = token
        command.extend(args)
        try:
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
        self.log(f"仅上传当前代码：{len(preview.files)} 个文件，跳过 {preview.excluded_lfs} 个 LFS 资源。")
        self.log("使用独立上传仓库，不携带 LeRobot 的旧提交历史。")
        remote = normalize_remote(settings.remote)
        with tempfile.TemporaryDirectory(prefix="github-code-upload-") as temporary:
            target = GitRepo(Path(temporary), self.log)
            target.prepare(settings.branch, create_ignore=False)
            self._copy_connection_config(target)
            # A later upload builds on this personal repository's current branch, without force pushing.
            heads = target.run(
                "ls-remote",
                "--heads",
                remote,
                f"refs/heads/{settings.branch}",
                network=True,
                token=settings.token,
            ).stdout.strip()
            if heads:
                self.log("读取个人仓库的最新版本，后续更新会接在这个版本之后。")
                target.run(
                    "fetch",
                    "--depth=1",
                    "--no-tags",
                    remote,
                    f"refs/heads/{settings.branch}",
                    network=True,
                    token=settings.token,
                )
                # Populate only the index and parent commit. No remote working files or LFS data are downloaded.
                target.run("reset", "--mixed", "FETCH_HEAD")
            for filename in preview.files:
                source = self.directory / filename
                copied = target.directory / filename
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, copied, follow_symlinks=False)
            if self.preview(code_only=True) != preview:
                raise UploadError("生成快照时文件发生变化，请重新预览并上传。")
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
            pushed = target.run(
                "-c",
                "push.followTags=false",
                "push",
                "--porcelain",
                "--recurse-submodules=no",
                remote,
                f"HEAD:refs/heads/{settings.branch}",
                network=True,
                token=settings.token,
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
        if "\n" in settings.token or "\r" in settings.token:
            raise UploadError("Token 不能包含换行符。")
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
        try:
            result = self.run(
                "-c",
                "push.followTags=false",
                "push",
                "--porcelain",
                "--recurse-submodules=no",
                remote,
                f"HEAD:refs/heads/{settings.branch}",
                network=True,
                token=settings.token,
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


def launch(directory: Path, *, demo: bool = False) -> None:
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
        open_window(directory, demo=demo)
    except RuntimeError as exc:
        raise UploadError(str(exc)) from exc


def main() -> int:
    """Parse the optional starting directory or serve Git's credential-helper protocol."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--credential-helper":
        credential_helper(sys.argv[2])
        return 0
    parser = argparse.ArgumentParser(description="GitHub 代码上传助手（CustomTkinter 深色桌面界面）")
    parser.add_argument("directory", nargs="?", type=Path, default=Path.cwd(), help="打开时选中的项目目录")
    parser.add_argument("--demo", action="store_true", help="仅预览示例界面，不读写仓库或上传")
    args = parser.parse_args()
    try:
        launch(args.directory, demo=args.demo)
    except UploadError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
