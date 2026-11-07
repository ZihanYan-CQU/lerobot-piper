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

"""Verify local credential preferences, startup precedence and desktop commands."""

import contextlib
import io
import os
import shutil
import stat
import subprocess
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from utils import github_upload_gui
from utils.github_upload_config import (
    ConfigError,
    UploadPreferences,
    default_config_path,
    install_desktop_launcher,
    load_preferences,
    parse_preferences,
    require_external_config,
    save_preferences,
    save_preferences_text,
)


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="github-upload-config-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "private" / "config.toml"
        self.isolated = patch.dict(os.environ, {"XDG_DATA_HOME": str(self.root / "data")})
        self.isolated.start()
        self.addCleanup(self.isolated.stop)

    def test_round_trip_preserves_escaped_credentials_paths_and_booleans(self):
        values = UploadPreferences(
            directory=str(self.root / "项目 path"),
            remote="https://github.com/user/project.git",
            username="user-name",
            name='User "Name"',
            email="user@example.invalid",
            token='test-only-\\-"-value',
            code_only=False,
            create_ignore=False,
        )

        save_preferences(self.config, values)

        self.assertEqual(load_preferences(self.config), values)
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.config.parent.stat().st_mode), 0o700)
        self.assertNotIn(values.token, repr(values))
        save_preferences(self.config, replace(values, branch="feature/update"))
        self.assertEqual(load_preferences(self.config).branch, "feature/update")
        self.assertEqual(list(self.config.parent.iterdir()), [self.config])

    def test_missing_config_is_read_only_and_defaults_to_code_mode(self):
        self.assertTrue(load_preferences(self.config).code_only)
        self.assertFalse(self.config.parent.exists())
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root / "settings")}):
            self.assertEqual(default_config_path(), self.root / "settings/lerobot-github-upload/config.toml")

    def test_editor_saves_valid_toml_with_comments_and_refuses_invalid_text(self):
        content = '# Keep this personal note\nusername = "saved-user"\ntoken = "test-only-value"\n'
        save_preferences_text(self.config, content)
        self.assertEqual(self.config.read_text(), content)
        self.assertEqual(load_preferences(self.config).username, "saved-user")
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        with self.assertRaises(ConfigError) as raised:
            save_preferences_text(self.config, 'token = "invalid-secret-value')
        self.assertNotIn("invalid-secret-value", str(raised.exception))
        self.assertEqual(self.config.read_text(), content)
        self.assertFalse(parse_preferences("code_only = false").code_only)

    def test_malformed_or_wrongly_typed_config_never_echoes_secret_lines(self):
        self.config.parent.mkdir()
        token = "private-test-token-never-real"
        for content in (
            f'token = "{token}',
            f'token = ["{token}"]',
            f'unexpected = "{token}"',
            'code_only = "false"',
        ):
            self.config.write_text(content)
            with self.subTest(content=content), self.assertRaises(ConfigError) as raised:
                load_preferences(self.config)
            self.assertNotIn(token, str(raised.exception))
        with self.assertRaises(ConfigError):
            save_preferences(self.config, UploadPreferences(username="user\npassword=injected"))

    def test_active_config_inside_upload_tree_is_rejected_including_symlinks(self):
        project = self.root / "project"
        project.mkdir()
        with self.assertRaises(ConfigError):
            require_external_config(project / "secrets.toml", project)
        require_external_config(self.config, project)
        link = self.root / "alias"
        link.symlink_to(project, target_is_directory=True)
        with self.assertRaises(ConfigError):
            require_external_config(link / "secrets.toml", project)

    def test_cli_restores_config_and_explicit_directory_wins(self):
        project = self.root / "project"
        other = self.root / "other"
        values = UploadPreferences(directory=str(project), username="fixed-user", token="fixed-test-token")
        save_preferences(self.config, values)
        for directory, expected in (([], project), ([str(other)], other)):
            with (
                patch("sys.argv", ["upload", "--config", str(self.config), *directory]),
                patch.object(github_upload_gui, "launch") as launch,
            ):
                self.assertEqual(github_upload_gui.main(), 0)
            self.assertEqual(launch.call_args.args, (expected,))
            self.assertEqual(launch.call_args.kwargs["preferences"], values)

    def test_demo_never_reads_or_writes_even_an_invalid_config(self):
        self.config.parent.mkdir()
        self.config.write_text('token = "unfinished')
        with (
            patch("sys.argv", ["upload", "--demo", "--config", str(self.config)]),
            patch.object(github_upload_gui, "launch") as launch,
        ):
            self.assertEqual(github_upload_gui.main(), 0)
        self.assertTrue(launch.call_args.kwargs["demo"])
        self.assertEqual(launch.call_args.kwargs["preferences"].token, "")
        self.assertEqual(self.config.read_text(), 'token = "unfinished')

    def test_cli_install_preserves_existing_config_and_does_not_open_window(self):
        values = UploadPreferences(directory=str(self.root / "project"), token="preserve-test-token")
        save_preferences(self.config, values)
        original = self.config.read_bytes()
        with (
            patch("sys.argv", ["upload", "--install-launcher", "--config", str(self.config)]),
            patch.object(github_upload_gui, "install_desktop_launcher") as install,
            patch.object(github_upload_gui, "launch") as launch,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(github_upload_gui.main(), 0)
        install.assert_called_once()
        launch.assert_not_called()
        self.assertEqual(self.config.read_bytes(), original)
        self.assertNotIn(values.token, output.getvalue())

    def test_desktop_reports_invalid_config_without_echoing_token_or_opening_empty_window(self):
        self.config.parent.mkdir()
        self.config.write_text('token = "private-test-token')
        with (
            patch("sys.argv", ["upload", "--desktop", "--config", str(self.config)]),
            patch.object(github_upload_gui, "show_startup_error") as error,
            patch.object(github_upload_gui, "launch") as launch,
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(github_upload_gui.main(), 1)
        error.assert_called_once()
        launch.assert_not_called()
        self.assertNotIn("private-test-token", stderr.getvalue())
        self.assertNotIn("private-test-token", error.call_args.args[0])

    @unittest.skipUnless(shutil.which("desktop-file-validate"), "desktop validator unavailable")
    def test_desktop_files_validate_and_reference_config_without_credentials(self):
        launcher = self.root / "项目 code" / "start.sh"
        launcher.parent.mkdir()
        launcher.touch()
        values = UploadPreferences(token="secret-test-value")
        save_preferences(self.config, values)

        shortcut = install_desktop_launcher(launcher, self.config, desktop=self.root / "desktop")

        checked = subprocess.run(["desktop-file-validate", str(shortcut)], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
        self.assertTrue(os.access(shortcut, os.X_OK))
        self.assertNotIn(values.token, shortcut.read_text())
        self.assertIn("--config", shortcut.read_text())
        self.assertIn("StartupWMClass=lerobot-github-upload", shortcut.read_text())
        menu = self.root / "data/applications/lerobot-github-upload.desktop"
        self.assertEqual(menu.read_text(), shortcut.read_text())

    @unittest.skipUnless(Path("/usr/bin/gio").exists(), "GIO desktop launcher unavailable")
    def test_desktop_launch_passes_unusual_paths_literally_to_bash(self):
        folder = self.root / '中文 space % \\ " $ ` folder'
        folder.mkdir()
        launcher = folder / "start.sh"
        result_path = self.root / "arguments"
        # No GUI or network: the desktop entry runs only this local argument-recording script.
        launcher.write_text(f'printf \'%s\\n\' "$@" > "{result_path}"\n')
        config = folder / "config.toml"
        shortcut = install_desktop_launcher(launcher, config, desktop=self.root / "desktop")
        result = subprocess.run(
            ["/usr/bin/gio", "launch", str(shortcut)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        deadline = time.monotonic() + 3
        while not result_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(result_path.exists(), result.stderr)
        self.assertEqual(result_path.read_text().splitlines(), ["--desktop", "--config", str(config)])


if __name__ == "__main__":
    unittest.main()
