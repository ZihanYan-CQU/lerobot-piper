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

"""Exercise the Qt upload window offscreen using only local Git remotes."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAS_QT = importlib.util.find_spec("PyQt5") is not None
if HAS_QT:
    import PyQt5
    from PyQt5.QtCore import QCoreApplication
    from PyQt5.QtGui import QRawFont
    from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QPlainTextEdit, QPushButton

    from utils.github_upload_config import UploadPreferences, load_preferences, save_preferences
    from utils.github_upload_gui import GitRepo, UploadProgress
    from utils.github_upload_window import UploadWindow


@unittest.skipUnless(HAS_QT and shutil.which("git"), "PyQt5 and Git are required")
class UploadWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.platform = patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"})
        cls.platform.start()
        # Locate Qt plugins through the actual package path, including non-ASCII project paths.
        QCoreApplication.setLibraryPaths([str(Path(PyQt5.__file__).parent / "Qt5/plugins")])
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        cls.app.setQuitOnLastWindowClosed(False)

    @classmethod
    def tearDownClass(cls):
        cls.platform.stop()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="github-upload-qt-test-")
        self.root = Path(self.temporary.name)
        (self.root / "home").mkdir()
        self.environment = patch.dict(
            os.environ, {"HOME": str(self.root / "home"), "XDG_CONFIG_HOME": str(self.root / "home/.config")}
        )
        self.environment.start()
        self.project = self.root / "项目代码"
        self.project.mkdir()
        self.remote = self.root / "remote.git"
        subprocess.run(
            ["git", "init", "--bare", "--initial-branch=main", str(self.remote)],
            check=True,
            capture_output=True,
        )
        self.repo = GitRepo(self.project)
        self.repo.prepare()
        self.url = "https://github.com/qt-test/project.git"
        for key, value in (
            ("user.name", "Qt Test"),
            ("user.email", "qt@example.invalid"),
            ("commit.gpgsign", "false"),
            ("core.hooksPath", str(self.root / "no-hooks")),
            (f"url.{self.remote.as_uri()}.insteadOf", self.url),
        ):
            self.repo.run("config", key, value)
        (self.project / "main.py").write_text('print("clear Chinese text")\n')
        self.config = self.root / "private/config.toml"
        prefs = UploadPreferences(
            directory=str(self.project),
            remote=self.url,
            username="qt-test",
            name="Qt Test",
            email="qt@example.invalid",
            token="fixture-token-never-real",
        )
        save_preferences(self.config, prefs)
        self.window = UploadWindow(self.project, preferences=prefs, config_path=self.config)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        if self.window.busy:
            self.wait_for(lambda: not self.window.busy)
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.environment.stop()
        self.temporary.cleanup()

    def wait_for(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate(), self.window.log_box.toPlainText())

    def start_upload(self):
        self.window._start_preview(upload=True)
        self.wait_for(lambda: self.window._modal is not None)
        self.assertEqual(self.window._modal.windowTitle(), "确认上传到 GitHub")
        confirm = next(
            button
            for button in self.window._modal.findChildren(QPushButton)
            if button.text() == "确认提交并上传"
        )
        confirm.click()
        self.wait_for(lambda: not self.window.busy and self.window._modal is not None)

    def test_text_uses_scalable_chinese_font_and_readable_field_size(self):
        font = self.window.inputs["directory"].font()
        self.assertGreaterEqual(font.pixelSize(), 15)
        self.assertTrue(QRawFont.fromFont(font).supportsCharacter(ord("中")))
        self.assertGreaterEqual(self.window.file_list.font().pixelSize(), 14)
        self.assertEqual(self.window.inputs["token"].echoMode(), QLineEdit.Password)

    def test_preferences_save_reload_and_reuse_keeps_token_masked(self):
        self.window.inputs["token"].setText("saved-fixture-token")
        self.window._save_config()
        self.assertEqual(load_preferences(self.config).token, "saved-fixture-token")
        updated = replace(
            load_preferences(self.config),
            username="updated-user",
            token="reloaded-fixture-token",
            code_only=False,
        )
        save_preferences(self.config, updated)
        self.window._reload_config()
        self.assertEqual(self.window.inputs["username"].text(), "updated-user")
        self.assertFalse(self.window.code_only.isChecked())
        self.assertIn("完整", self.window.review_note.text())
        self.window.inputs["token"].clear()
        self.window._clear_token()
        self.assertEqual(self.window.inputs["token"].text(), updated.token)
        self.assertEqual(self.window.inputs["token"].echoMode(), QLineEdit.Password)
        self.assertNotIn(updated.token, self.window.log_box.toPlainText())

    def editor_controls(self):
        button = next(
            button for button in self.window.findChildren(QPushButton) if button.text() == "参数文件"
        )
        button.click()
        dialog = self.window._modal
        self.assertEqual(dialog.windowTitle(), "编辑参数文件")
        editor = dialog.findChild(QPlainTextEdit, "configTextEditor")
        save = next(button for button in dialog.findChildren(QPushButton) if button.text() == "保存并应用")
        return dialog, editor, save

    def test_top_panel_opens_file_and_save_applies_parameters_immediately(self):
        _dialog, editor, save = self.editor_controls()
        self.assertEqual(editor.toPlainText(), self.config.read_text())
        content = self.config.read_text().replace('username = "qt-test"', 'username = "new-user"')
        content = content.replace('token = "fixture-token-never-real"', 'token = "changed-fixture-token"')
        editor.setPlainText(content)
        save.click()
        self.assertIsNone(self.window._modal)
        self.assertEqual(load_preferences(self.config).username, "new-user")
        self.assertEqual(self.window.inputs["username"].text(), "new-user")
        self.assertEqual(self.window.inputs["token"].text(), "changed-fixture-token")
        self.assertEqual(self.window.inputs["token"].echoMode(), QLineEdit.Password)
        self.assertNotIn("changed-fixture-token", self.window.log_box.toPlainText())

    def test_editor_invalid_input_preserves_file_and_displays_safe_error(self):
        original = self.config.read_bytes()
        dialog, editor, save = self.editor_controls()
        editor.setPlainText('token = "invalid-fixture-secret')
        save.click()
        self.assertIs(self.window._modal, dialog)
        self.assertEqual(self.config.read_bytes(), original)
        error = dialog.findChild(QLabel, "configEditorError")
        self.assertFalse(error.isHidden())
        self.assertNotIn("invalid-fixture-secret", error.text())

    def test_editor_cancel_preserves_file_and_external_changes_are_not_overwritten(self):
        original = self.config.read_bytes()
        dialog, editor, _save = self.editor_controls()
        editor.setPlainText('username = "discard-this"')
        dialog.reject()
        self.assertEqual(self.config.read_bytes(), original)
        dialog, editor, save = self.editor_controls()
        external = original.decode().replace('username = "qt-test"', 'username = "external-user"')
        self.config.write_text(external)
        editor.setPlainText('username = "do-not-overwrite"')
        save.click()
        self.assertEqual(self.config.read_text(), external)
        self.assertIn("其他程序", dialog.findChild(QLabel, "configEditorError").text())

    def test_snapshot_gui_confirms_upload_and_preserves_source_history(self):
        (self.project / ".gitattributes").write_text("*.bin filter=lfs\n")
        (self.project / "weights.bin").write_text("excluded data")
        self.start_upload()
        self.assertEqual(self.window._modal.windowTitle(), "上传成功")
        self.assertEqual(self.window.progress.value(), 100)
        self.assertEqual(self.window._preview.excluded_lfs, 1)
        self.assertGreater(self.window._preview.total_bytes, 0)
        self.assertNotEqual(self.repo.run("rev-parse", "--verify", "HEAD", check=False).returncode, 0)
        files = subprocess.check_output(
            ["git", "--git-dir", str(self.remote), "ls-tree", "-r", "--name-only", "main"], text=True
        )
        self.assertIn("main.py", files)
        self.assertNotIn("weights.bin", files)
        self.assertNotIn("config.toml", files)
        self.assertEqual(self.window.inputs["token"].text(), self.window.preferences.token)
        self.assertNotIn(self.window.preferences.token, self.window.log_box.toPlainText())

    def test_full_history_mode_is_passed_to_backend(self):
        self.window.code_only.setChecked(False)
        self.start_upload()
        self.assertEqual(self.window._modal.windowTitle(), "上传成功")
        local = self.repo.run("rev-parse", "HEAD").stdout.strip()
        remote = subprocess.check_output(
            ["git", "--git-dir", str(self.remote), "rev-parse", "main"], text=True
        ).strip()
        self.assertEqual(local, remote)
        self.assertFalse(self.window._preview.code_only)

    def test_rejected_local_push_displays_error_and_restores_controls(self):
        hook = self.remote / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\necho 'fixture rejection' >&2\nexit 1\n")
        hook.chmod(0o755)
        self.start_upload()
        self.assertEqual(self.window._modal.windowTitle(), "操作未完成")
        self.assertIn("fixture rejection", self.window.log_box.toPlainText())
        self.assertTrue(self.window.upload_button.isEnabled())
        self.assertTrue(self.window.progress_container.isHidden())
        self.assertNotIn(self.window.preferences.token, self.window.log_box.toPlainText())

    def test_waiting_for_server_does_not_claim_success_and_busy_window_stays_open(self):
        self.window._set_busy(True)
        self.window._update_progress(UploadProgress("等待远端确认"))
        self.assertEqual(self.window.progress.maximum(), 0)
        self.assertIsNone(self.window._modal)
        self.assertFalse(self.window.close())
        self.assertFalse(self.window._closed)
        self.window._set_busy(False)


if __name__ == "__main__":
    unittest.main()
