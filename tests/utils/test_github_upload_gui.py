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

"""Exercise the upload helper against isolated local Git repositories, without GitHub access."""

import io
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.github_upload_gui import GitRepo, UploadError, UploadSettings, credential_helper, normalize_remote


@unittest.skipUnless(shutil.which("git"), "Git is required for repository integration tests")
class GitUploadTests(unittest.TestCase):
    remote_url = "https://github.com/test/project.git"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="github-upload-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.config_home = self.root / "config"
        self.config_home.mkdir()
        self.hooks = self.root / "empty-hooks"
        self.hooks.mkdir()
        isolated = patch.dict(
            os.environ,
            {
                "PATH": os.environ.get("PATH", os.defpath),
                "HOME": str(self.home),
                "XDG_CONFIG_HOME": str(self.config_home),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "LC_ALL": "C",
            },
            clear=True,
        )
        isolated.start()
        self.addCleanup(isolated.stop)
        self.directory = self.root / "project"
        self.directory.mkdir()
        self.remote = self.root / "remote.git"
        self.git(self.root, "init", "--bare", "--initial-branch=main", str(self.remote))
        self.logs = []
        self.repo = GitRepo(self.directory, self.logs.append)

    def git(self, directory, *args, check=True):
        return subprocess.run(
            ["git", "-c", "core.hooksPath=" + str(self.hooks), "-c", "commit.gpgsign=false", *args],
            cwd=directory,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=check,
            timeout=30,
        )

    def configure(self, directory):
        for key, value in (
            ("user.name", "Upload Test"),
            ("user.email", "upload-test@example.invalid"),
            ("core.hooksPath", str(self.hooks)),
            ("commit.gpgsign", "false"),
            ("tag.gpgsign", "false"),
            ("credential.helper", ""),
            ("protocol.file.allow", "always"),
            (f"url.{self.remote.as_uri()}.insteadOf", self.remote_url),
        ):
            self.git(directory, "config", "--local", key, value)

    def prepare(self, create_ignore=True):
        self.repo.prepare(create_ignore=create_ignore)
        self.configure(self.directory)

    def settings(self, **overrides):
        values = {
            "directory": self.directory,
            "remote": self.remote_url,
            "branch": "main",
            "message": "Upload test files",
        }
        values.update(overrides)
        return UploadSettings(**values)

    def write(self, name, content):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def remote_head(self, branch="main"):
        return self.git(self.remote, "rev-parse", "refs/heads/" + branch).stdout.strip()

    def start_history(self):
        self.prepare()
        self.write("source.txt", "original content\n")
        return self.repo.upload(self.settings())

    def assert_stale_preview_rejected(self, expected):
        index = self.git(self.directory, "ls-files", "--stage", "-z").stdout
        head = self.git(self.directory, "rev-parse", "HEAD").stdout
        branch = self.git(self.directory, "symbolic-ref", "HEAD").stdout
        remote_head = self.remote_head()
        working_diff = self.git(self.directory, "diff", "--no-ext-diff").stdout
        status = self.git(self.directory, "status", "--porcelain=v1", "-z").stdout

        with (
            patch.object(self.repo, "run", wraps=self.repo.run) as run,
            self.assertRaisesRegex(UploadError, "预览后.*发生变化"),
        ):
            self.repo.upload(self.settings(), expected=expected)

        for call in run.call_args_list:
            self.assertNotIn(call.args[0], ("add", "commit", "push"))
        self.assertEqual(self.git(self.directory, "ls-files", "--stage", "-z").stdout, index)
        self.assertEqual(self.git(self.directory, "rev-parse", "HEAD").stdout, head)
        self.assertEqual(self.git(self.directory, "symbolic-ref", "HEAD").stdout, branch)
        self.assertEqual(self.remote_head(), remote_head)
        self.assertEqual(self.git(self.directory, "diff", "--no-ext-diff").stdout, working_diff)
        self.assertEqual(self.git(self.directory, "status", "--porcelain=v1", "-z").stdout, status)

    def test_first_push_initializes_repository_and_respects_default_ignore(self):
        self.write("main.py", "print('hello')\n")
        self.write(".env", "PRIVATE_VALUE=not-for-upload\n")
        self.write(".env.example", "PRIVATE_VALUE=\n")
        self.write(".venv/secret.txt", "local environment\n")
        self.write("outputs/checkpoint.bin", "generated\n")
        self.prepare()

        preview = self.repo.preview()
        self.assertEqual(preview.branch, "main")
        self.assertEqual(preview.head, "")
        self.assertIn("main.py", preview.status)
        head = self.repo.upload(self.settings())

        self.assertEqual(head, self.remote_head())
        files = self.git(self.remote, "ls-tree", "-r", "--name-only", head).stdout.splitlines()
        self.assertCountEqual(files, [".gitignore", ".env.example", "main.py"])
        self.assertEqual(self.repo.preview().status, "")

    def test_update_delete_and_existing_partial_staging_are_all_committed(self):
        self.prepare()
        self.write("changed.txt", "original\n")
        removed = self.write("removed.txt", "delete me\n")
        self.repo.upload(self.settings())
        self.write("changed.txt", "staged version\n")
        self.git(self.directory, "add", "changed.txt")
        self.write("changed.txt", "working tree version\n")
        self.write("新增 file.txt", "new file\n")
        removed.unlink()
        preview = self.repo.preview()
        self.assertIn("MM changed.txt", preview.status)
        self.assertIn(" D removed.txt", preview.status)

        head = self.repo.upload(self.settings(message="Update, add, and remove"))

        self.assertEqual(self.remote_head(), head)
        self.assertEqual(self.git(self.remote, "show", "main:changed.txt").stdout, "working tree version\n")
        self.assertEqual(self.git(self.remote, "show", "main:新增 file.txt").stdout, "new file\n")
        self.assertNotEqual(
            self.git(self.remote, "cat-file", "-e", "main:removed.txt", check=False).returncode, 0
        )
        self.assertEqual(self.repo.preview().status, "")

    def test_repeat_upload_pushes_same_commit_without_empty_commit(self):
        self.prepare()
        self.write("source.txt", "source\n")
        first = self.repo.upload(self.settings())

        second = self.repo.upload(self.settings(message="Should not create another commit"))

        self.assertEqual(first, second)
        self.assertEqual(self.remote_head(), first)
        self.assertEqual(self.git(self.directory, "rev-list", "--count", "HEAD").stdout.strip(), "1")

    def test_identical_expected_preview_can_commit_and_push(self):
        initial = self.start_history()
        self.write("source.txt", "reviewed content\n")
        expected = self.repo.preview()
        self.assertEqual(expected, self.repo.preview())

        head = self.repo.upload(self.settings(), expected=expected)

        self.assertNotEqual(head, initial)
        self.assertEqual(head, self.remote_head())
        self.assertEqual(self.git(self.remote, "show", "main:source.txt").stdout, "reviewed content\n")

    def test_preview_rejects_changed_content_even_when_status_and_size_match(self):
        self.start_history()
        path = self.write("source.txt", "reviewed content\n")
        expected = self.repo.preview()
        previous_stat = path.stat()
        self.write("source.txt", "unseenxx content\n")
        os.utime(path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
        current = self.repo.preview()
        self.assertEqual(current.status, expected.status)
        self.assertEqual(path.stat().st_size, previous_stat.st_size)
        self.assertNotEqual(current.signature, expected.signature)

        self.assert_stale_preview_rejected(expected)

    def test_preview_rejects_new_file_before_staging(self):
        self.start_history()
        expected = self.repo.preview()
        self.write("unreviewed.txt", "not yet approved\n")

        self.assert_stale_preview_rejected(expected)

        self.assertEqual(self.git(self.directory, "ls-files", "--", "unreviewed.txt").stdout, "")

    def test_preview_rejects_head_change_with_unchanged_working_tree(self):
        self.start_history()
        expected = self.repo.preview()
        self.git(self.directory, "commit", "--allow-empty", "-m", "External commit")
        current = self.repo.preview()
        self.assertEqual(current.status, expected.status)
        self.assertNotEqual(current.head, expected.head)

        self.assert_stale_preview_rejected(expected)

    def test_preview_rejects_branch_change_at_same_commit(self):
        self.start_history()
        expected = self.repo.preview()
        self.git(self.directory, "checkout", "-b", "another-branch")
        current = self.repo.preview()
        self.assertEqual(current.head, expected.head)
        self.assertEqual(current.status, expected.status)
        self.assertNotEqual(current.branch, expected.branch)

        self.assert_stale_preview_rejected(expected)

    def test_preview_rejects_index_only_content_change_with_same_status(self):
        self.start_history()
        blob_path = self.root / "blob-source.txt"

        def stage_blob(content):
            blob_path.write_text(content)
            blob = self.git(self.directory, "hash-object", "-w", str(blob_path)).stdout.strip()
            self.git(self.directory, "update-index", "--cacheinfo", "100644," + blob + ",source.txt")

        stage_blob("first staged content\n")
        expected = self.repo.preview()
        previous_stat = (self.directory / "source.txt").stat()
        stage_blob("other staged content\n")
        current = self.repo.preview()
        self.assertEqual(current.status, expected.status)
        self.assertEqual((self.directory / "source.txt").stat(), previous_stat)
        self.assertNotEqual(current.signature, expected.signature)

        self.assert_stale_preview_rejected(expected)

    def test_linked_worktree_git_file_is_accepted_and_uploads_its_own_branch(self):
        initial = self.start_history()
        worktree = self.root / "linked-worktree"
        self.git(self.directory, "worktree", "add", "-b", "worktree-branch", str(worktree))
        self.assertTrue((worktree / ".git").is_file())
        worktree_repo = GitRepo(worktree)
        worktree_repo.prepare()
        (worktree / "source.txt").write_text("worktree content\n")
        expected = worktree_repo.preview()

        head = worktree_repo.upload(
            self.settings(directory=worktree, branch="worktree-branch"), expected=expected
        )

        self.assertEqual(head, self.remote_head("worktree-branch"))
        self.assertEqual(worktree_repo.preview().branch, "worktree-branch")
        self.assertEqual(self.repo.preview().head, initial)
        self.assertEqual(self.remote_head(), initial)
        self.assertEqual((self.directory / "source.txt").read_text(), "original content\n")

    def test_existing_ignore_and_origin_are_preserved_and_tags_are_not_pushed(self):
        ignore_content = "private-data/\n"
        self.write(".gitignore", ignore_content)
        self.write("private-data/local.txt", "private\n")
        self.write("source.txt", "source\n")
        self.prepare()
        original_origin = "https://github.com/upstream/original.git"
        self.git(self.directory, "remote", "add", "origin", original_origin)
        self.git(self.directory, "config", "push.followTags", "true")
        self.git(self.directory, "add", ".")
        self.git(self.directory, "commit", "-m", "Initial local commit")
        self.git(self.directory, "tag", "-a", "local-only", "-m", "Do not upload this tag")

        head = self.repo.upload(self.settings(branch="feature/upload"))

        self.assertEqual(head, self.remote_head("feature/upload"))
        self.assertEqual(self.repo.preview().branch, "main")
        self.assertEqual(self.repo.preview().origin, original_origin)
        self.assertEqual((self.directory / ".gitignore").read_text(), ignore_content)
        self.assertEqual(self.git(self.remote, "tag", "--list").stdout, "")

    def test_non_fast_forward_preserves_remote_and_local_commit_for_retry(self):
        self.prepare()
        self.write("source.txt", "initial\n")
        initial = self.repo.upload(self.settings())
        other = self.root / "other"
        self.git(self.root, "clone", self.remote.as_uri(), str(other))
        self.configure(other)
        (other / "remote-change.txt").write_text("remote change\n")
        self.git(other, "add", ".")
        self.git(other, "commit", "-m", "Someone else's change")
        self.git(other, "push", "origin", "main")
        remote_head = self.remote_head()
        self.write("source.txt", "local change\n")

        with self.assertRaisesRegex(UploadError, "本地提交.*已保留"):
            self.repo.upload(self.settings(message="Preserve local work"))

        local_head = self.repo.preview().head
        self.assertNotEqual(local_head, initial)
        self.assertEqual(self.git(self.directory, "rev-parse", "HEAD^").stdout.strip(), initial)
        self.assertEqual(self.remote_head(), remote_head)
        self.assertEqual(self.git(self.directory, "show", "HEAD:source.txt").stdout, "local change\n")
        with self.assertRaises(UploadError):
            self.repo.upload(self.settings(message="Retry"))
        self.assertEqual(self.repo.preview().head, local_head)
        self.assertEqual(self.remote_head(), remote_head)

    def test_subdirectory_is_rejected_without_creating_nested_repository_or_ignore(self):
        self.prepare(create_ignore=False)
        subdirectory = self.directory / "subdirectory"
        subdirectory.mkdir()

        with self.assertRaisesRegex(UploadError, "子目录"):
            GitRepo(subdirectory).prepare()

        self.assertFalse((subdirectory / ".git").exists())
        self.assertFalse((subdirectory / ".gitignore").exists())

    def test_invalid_branch_rejected_before_initializing_folder(self):
        for branch in ("-main", "name with spaces", "main..next", "main:other", "../outside", ""):
            with self.subTest(branch=branch), self.assertRaises(UploadError):
                self.repo.prepare(branch)
        self.assertFalse((self.directory / ".git").exists())
        self.assertFalse((self.directory / ".gitignore").exists())

    def test_detached_head_rejected_before_creating_ignore(self):
        self.prepare(create_ignore=False)
        self.write("source.txt", "source\n")
        self.git(self.directory, "add", ".")
        self.git(self.directory, "commit", "-m", "Initial")
        self.git(self.directory, "checkout", "--detach")

        with self.assertRaisesRegex(UploadError, "detached HEAD"):
            self.repo.prepare()

        self.assertFalse((self.directory / ".gitignore").exists())

    def test_merge_conflict_rejected_without_staging_or_creating_ignore(self):
        self.prepare(create_ignore=False)
        self.write("source.txt", "original\n")
        self.git(self.directory, "add", ".")
        self.git(self.directory, "commit", "-m", "Initial")
        self.git(self.directory, "checkout", "-b", "other")
        self.write("source.txt", "other branch\n")
        self.git(self.directory, "commit", "-am", "Other branch")
        self.git(self.directory, "checkout", "main")
        self.write("source.txt", "main branch\n")
        self.git(self.directory, "commit", "-am", "Main branch")
        self.assertNotEqual(self.git(self.directory, "merge", "other", check=False).returncode, 0)
        before = self.git(self.directory, "ls-files", "--unmerged").stdout

        with self.assertRaisesRegex(UploadError, "冲突"):
            self.repo.upload(self.settings())

        self.assertEqual(self.git(self.directory, "ls-files", "--unmerged").stdout, before)
        self.assertFalse((self.directory / ".gitignore").exists())

    def test_unusual_filenames_are_passed_as_files_not_options(self):
        self.prepare()
        names = ["--delete", "has spaces.txt", "中文.txt", "line\nbreak.txt"]
        for name in names:
            self.write(name, "content\n")

        self.repo.upload(self.settings())

        files = self.git(self.remote, "ls-tree", "-rz", "--name-only", "main").stdout.split("\0")
        self.assertCountEqual([name for name in files if name], [".gitignore", *names])

    def test_environment_git_directory_cannot_redirect_operations(self):
        self.prepare()
        self.write("source.txt", "selected directory\n")
        unrelated = self.root / "unrelated"
        unrelated.mkdir()
        self.git(unrelated, "init", "-b", "main")

        with patch.dict(os.environ, {"GIT_DIR": str(unrelated / ".git"), "GIT_WORK_TREE": str(unrelated)}):
            head = self.repo.upload(self.settings())

        self.assertEqual(head, self.remote_head())
        self.assertNotEqual(self.git(unrelated, "rev-parse", "--verify", "HEAD", check=False).returncode, 0)

    def test_code_snapshot_excludes_history_and_current_lfs_resources(self):
        self.prepare()
        self.write(".gitattributes", "*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
        self.write(
            "old.safetensors",
            "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 9000000\n",
        )
        self.git(self.directory, "add", ".")
        self.git(self.directory, "commit", "-m", "Upstream history with unavailable LFS data")
        (self.directory / "old.safetensors").unlink()
        self.write("current.safetensors", "large resource omitted from code backup\n")
        self.write("main.py", "print('code only')\n")
        self.write(".env", "PRIVATE=not-for-upload\n")
        expected = self.repo.preview(code_only=True)
        before_head = self.git(self.directory, "rev-parse", "HEAD").stdout
        before_index = self.git(self.directory, "ls-files", "--stage", "-z").stdout

        head = self.repo.upload(self.settings(code_only=True), expected=expected)

        self.assertEqual(head, self.remote_head())
        self.assertEqual(self.git(self.remote, "rev-list", "--count", "main").stdout.strip(), "1")
        files = self.git(self.remote, "ls-tree", "-r", "--name-only", head).stdout.splitlines()
        self.assertIn("main.py", files)
        self.assertNotIn("old.safetensors", files)
        self.assertNotIn("current.safetensors", files)
        self.assertNotIn(".env", files)
        self.assertEqual(expected.excluded_lfs, 1)
        self.assertEqual(self.git(self.directory, "rev-parse", "HEAD").stdout, before_head)
        self.assertEqual(self.git(self.directory, "ls-files", "--stage", "-z").stdout, before_index)

    def test_code_snapshot_updates_own_history_and_repeats_without_empty_commits(self):
        self.prepare()
        self.write("main.py", "version one\n")
        self.write("remove.py", "remove me\n")
        first = self.repo.upload(self.settings(code_only=True))
        repeated = self.repo.upload(self.settings(code_only=True))
        self.assertEqual(first, repeated)
        self.write("main.py", "version two\n")
        (self.directory / "remove.py").unlink()
        self.write("new.py", "new code\n")

        second = self.repo.upload(self.settings(code_only=True))

        self.assertEqual(self.remote_head(), second)
        self.assertEqual(self.git(self.remote, "rev-parse", "main^").stdout.strip(), first)
        self.assertEqual(self.git(self.remote, "rev-list", "--count", "main").stdout.strip(), "2")
        self.assertEqual(self.git(self.remote, "show", "main:main.py").stdout, "version two\n")
        self.assertNotEqual(
            self.git(self.remote, "cat-file", "-e", "main:remove.py", check=False).returncode, 0
        )

    def test_code_snapshot_keeps_partial_staging_and_copies_working_content(self):
        self.prepare()
        self.write("main.py", "original\n")
        self.git(self.directory, "add", ".")
        self.git(self.directory, "commit", "-m", "Local baseline")
        before_head = self.git(self.directory, "rev-parse", "HEAD").stdout
        self.write("main.py", "staged\n")
        self.git(self.directory, "add", "main.py")
        self.write("main.py", "current working version\n")
        before_index = self.git(self.directory, "ls-files", "--stage", "-z").stdout

        self.repo.upload(self.settings(code_only=True), expected=self.repo.preview(code_only=True))

        self.assertEqual(self.git(self.remote, "show", "main:main.py").stdout, "current working version\n")
        self.assertEqual(self.git(self.directory, "rev-parse", "HEAD").stdout, before_head)
        self.assertEqual(self.git(self.directory, "ls-files", "--stage", "-z").stdout, before_index)

    def test_code_snapshot_keeps_tracked_files_matching_ignore_rules_and_unusual_names(self):
        self.prepare()
        self.write(".gitignore", "tracked.cfg\n")
        self.write("tracked.cfg", "keep tracked source configuration\n")
        self.git(self.directory, "add", "--force", "tracked.cfg")
        for name in ("--delete", "中文 file.py", "line\nbreak.py", ":(glob)name.py"):
            self.write(name, "code\n")

        self.repo.upload(self.settings(code_only=True))

        actual = self.git(self.remote, "ls-tree", "-rz", "--name-only", "main").stdout.split("\0")
        for name in ("tracked.cfg", "--delete", "中文 file.py", "line\nbreak.py", ":(glob)name.py"):
            self.assertIn(name, actual)

    def test_code_snapshot_rejects_files_changed_after_preview(self):
        self.prepare()
        self.write("main.py", "version one\n")
        expected = self.repo.preview(code_only=True)
        self.write("main.py", "version two\n")

        with self.assertRaisesRegex(UploadError, "预览后.*发生变化"):
            self.repo.upload(self.settings(code_only=True), expected=expected)

        self.assertNotEqual(self.git(self.remote, "rev-parse", "--verify", "main", check=False).returncode, 0)

    def test_code_snapshot_preserves_executable_and_symbolic_link(self):
        self.prepare()
        script = self.write("run.sh", "#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        (self.directory / "linked.sh").symlink_to("run.sh")

        self.repo.upload(self.settings(code_only=True))

        tree = self.git(self.remote, "ls-tree", "main").stdout
        self.assertIn("100755 blob", tree)
        self.assertIn("120000 blob", tree)
        self.assertEqual(self.git(self.remote, "show", "main:linked.sh").stdout, "run.sh")

    def test_code_snapshot_refuses_concurrent_remote_update_without_force(self):
        self.prepare()
        self.write("main.py", "initial\n")
        self.repo.upload(self.settings(code_only=True))
        other = self.root / "other-snapshot"
        self.git(self.root, "clone", self.remote.as_uri(), str(other))
        self.configure(other)
        self.write("main.py", "my update\n")
        original_run = GitRepo.run
        advanced = []

        def advance_before_push(repo, *args, **kwargs):
            if kwargs.get("network") and "push" in args and not advanced:
                (other / "other.py").write_text("concurrent update\n")
                self.git(other, "add", ".")
                self.git(other, "commit", "-m", "Concurrent update")
                self.git(other, "push", "origin", "main")
                advanced.append(self.remote_head())
            return original_run(repo, *args, **kwargs)

        with patch.object(GitRepo, "run", advance_before_push), self.assertRaises(UploadError):
            self.repo.upload(self.settings(code_only=True))

        self.assertEqual(self.remote_head(), advanced[0])
        self.assertEqual(self.git(self.remote, "show", "main:main.py").stdout, "initial\n")
        self.assertEqual((self.directory / "main.py").read_text(), "my update\n")


class CredentialTests(unittest.TestCase):
    def test_remote_validation_accepts_github_and_rejects_embedded_credentials(self):
        for remote in (
            "https://github.com/user/project.git",
            "git@github.com:user/project.git",
            "ssh://git@github.com/user/project.git",
        ):
            with self.subTest(remote=remote):
                self.assertEqual(normalize_remote(remote + "/"), remote)
        for remote in (
            "https://username:secret@github.com/user/project.git",
            "https://secret@github.com/user/project.git",
            "https://github.com.evil.example/user/project.git",
            "https://github.com/user/project?token=secret",
            "http://github.com/user/project.git",
            "https://github.com/../project.git",
            "--upload-pack=unsafe",
        ):
            with self.subTest(remote=remote), self.assertRaises(UploadError):
                normalize_remote(remote)

    def test_credential_helper_only_answers_https_github_get_and_does_not_persist(self):
        token = "temporary-test-value"
        with (
            tempfile.TemporaryDirectory(prefix="github-credential-test-") as directory,
            patch.dict(
                os.environ,
                {"HOME": directory, "XDG_CONFIG_HOME": directory, "LEROBOT_GITHUB_UPLOAD_TOKEN": token},
                clear=True,
            ),
        ):
            for action, protocol, host, expected in (
                ("get", "https", "github.com", True),
                ("get", "https", "github.com.evil.example", False),
                ("get", "http", "github.com", False),
                ("store", "https", "github.com", False),
                ("erase", "https", "github.com", False),
            ):
                with self.subTest(action=action, protocol=protocol, host=host):
                    request = io.StringIO(f"protocol={protocol}\nhost={host}\n\n")
                    response = io.StringIO()
                    with patch("sys.stdin", request), patch("sys.stdout", response):
                        credential_helper(action)
                    expected_text = f"username=x-access-token\npassword={token}\n\n" if expected else ""
                    self.assertEqual(response.getvalue(), expected_text)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_token_is_not_in_settings_representation(self):
        settings = UploadSettings(
            Path("."), "https://github.com/user/project.git", "main", "test", token="secret"
        )
        self.assertNotIn("secret", repr(settings))


if __name__ == "__main__":
    unittest.main()
