"""Exercise selection and subprocess lifecycle without touching real datasets or hardware."""

from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

HAS_QT = importlib.util.find_spec("PyQt5") is not None
if HAS_QT:
    from PyQt5 import QtCore
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QFileDialog

    from lerobot.yzh import merge_datasets_gui, merge_datasets_window
    from lerobot.yzh.merge_datasets_window import DatasetEntry, MergeWindow


@unittest.skipUnless(HAS_QT, "the desktop extra is required")
class MergeWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.platform = patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"})
        cls.platform.start()
        plugins = Path(QtCore.__file__).resolve().parent / "Qt5/plugins"
        QtCore.QCoreApplication.setLibraryPaths([str(plugins)])
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)
        cls.app.setStyle("Fusion")

    @classmethod
    def tearDownClass(cls):
        cls.platform.stop()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lerobot-merge-ui-")
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name) / "数据集"
        self.paths = [
            self.make_dataset(f"block/batch_{start}-{start + 10}", 10, start + 100) for start in (20, 0, 10)
        ]
        self.window = MergeWindow(home=self.home)
        self.window.show()
        self.app.processEvents()
        self.addCleanup(self.close_window)

    def make_dataset(self, name, episodes, frames):
        path = self.home / name
        (path / "meta").mkdir(parents=True)
        (path / "data").mkdir()
        (path / "meta/info.json").write_text(
            json.dumps({"total_episodes": episodes, "total_frames": frames, "fps": 30}), encoding="utf-8"
        )
        return path

    def close_window(self):
        if self.window.worker is not None:
            self.window.worker.request_stop()
            self.window.worker.wait(5000)
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def wait_for(self, condition):
        deadline = time.monotonic() + 8
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(condition(), self.window.log_box.toPlainText())

    def select_all(self):
        self.window._check_visible(True)
        self.app.processEvents()

    def start_command(self, script, output):
        with patch.object(
            self.window, "_command", return_value=([sys.executable, "-u", "-c", script, str(output)], output)
        ):
            self.window.start_merge()

    def test_scan_selection_statistics_and_search_keep_the_merge_order(self):
        self.assertEqual(
            [item.data(1, Qt.UserRole).path.name for item in self.window._items()],
            ["batch_0-10", "batch_10-20", "batch_20-30"],
        )
        self.assertFalse(self.window.merge_button.isEnabled())
        self.select_all()
        self.assertEqual(self.window.metrics["datasets"].text(), "3")
        self.assertEqual(self.window.metrics["episodes"].text(), "30")
        self.assertEqual(self.window.metrics["frames"].text(), "330")
        self.assertEqual(self.window.repo_input.text(), "block/batch_merged")
        self.assertTrue(self.window.merge_button.isEnabled())
        selected = self.window.selected_paths()
        self.window.search_input.setText("batch_10-20")
        self.assertEqual(self.window.dataset_count.text(), "显示 1 / 3 个")
        self.assertEqual(self.window.selected_paths(), selected)
        command, _output = self.window._command()
        self.assertEqual(
            json.loads(command[command.index("--operation.roots") + 1]), list(map(str, selected))
        )

    def test_move_multiple_rows_and_refresh_preserve_checks_and_manual_name(self):
        self.select_all()
        items = self.window._items()
        items[1].setSelected(True)
        items[2].setSelected(True)
        self.window.move_selected(-1)
        expected = [self.home / "block" / name for name in ("batch_10-20", "batch_20-30", "batch_0-10")]
        self.assertEqual(self.window.selected_paths(), expected)
        self.window.repo_input.setFocus()
        self.window.repo_input.selectAll()
        QTest.keyClicks(self.window.repo_input, "custom/combined")
        self.assertFalse(self.window.repo_id_auto)
        self.window.refresh()
        self.assertEqual(self.window.selected_paths(), expected)
        self.assertEqual(self.window.repo_input.text(), "custom/combined")
        self.window._reset_name()
        self.assertEqual(self.window.repo_input.text(), "block/batch_merged")

    def test_external_dataset_is_not_duplicated_or_lost_on_refresh(self):
        external = self.make_dataset("other/external", 5, 50)
        self.window.source_input.setText(str(self.home / "block"))
        self.window.refresh()
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(external)):
            self.window.add_dataset()
            self.window.add_dataset()
        self.window.refresh()
        self.assertEqual(self.window.selected_paths(), [external])
        self.assertEqual(len(self.window._items()), 4)

    def test_invalid_metadata_and_existing_output_block_the_action(self):
        (self.paths[0] / "meta/info.json").write_text("not json", encoding="utf-8")
        self.window.refresh()
        self.select_all()
        self.assertFalse(self.window.merge_button.isEnabled())
        self.assertIn("无法读取元信息", self.window.detail_label.text())
        for item in self.window._items():
            if item.data(1, Qt.UserRole).error:
                item.setCheckState(0, Qt.Unchecked)
        self.assertTrue(self.window.merge_button.isEnabled())
        output = self.home / self.window.repo_input.text()
        output.mkdir()
        self.window._update_preview()
        self.assertFalse(self.window.merge_button.isEnabled())
        self.assertIn("已存在", self.window.output_hint.text())
        self.assertIn("已阻止合并", self.window.detail_label.text())

    def test_success_streams_logs_freezes_configuration_and_unlocks_the_result(self):
        self.select_all()
        output = self.home / self.window.repo_input.text()
        self.start_command(
            "import pathlib,sys,time; print('fixture ready', flush=True); time.sleep(.15); pathlib.Path(sys.argv[1]).mkdir(); print('fixture complete')",
            output,
        )
        self.assertTrue(self.window.busy)
        self.assertFalse(self.window.tree.isEnabled())
        self.assertFalse(self.window.repo_input.isEnabled())
        self.assertFalse(self.window.close())
        self.wait_for(lambda: not self.window.busy)
        self.window.worker.wait(1000)
        self.assertEqual(self.window.status_label.text(), "合并完成")
        self.assertIn("fixture complete", self.window.log_box.toPlainText())
        self.assertTrue(self.window.open_button.isEnabled())
        self.assertFalse(self.window.merge_button.isEnabled())
        self.assertEqual(self.window.progress.value(), 1)
        self.assertTrue(self.window.tree.isEnabled())

    def test_failure_and_cancellation_never_claim_success(self):
        self.select_all()
        output = self.home / "block/failed"
        self.start_command("import sys; print('fixture failure'); sys.exit(3)", output)
        self.wait_for(lambda: not self.window.busy)
        self.window.worker.wait(1000)
        self.assertIn("合并失败", self.window.status_label.text())
        self.assertIn("fixture failure", self.window.log_box.toPlainText())
        self.assertTrue(self.window.merge_button.isEnabled())
        self.start_command("import time; print('waiting for stop', flush=True); time.sleep(30)", output)
        self.wait_for(lambda: "waiting for stop" in self.window.log_box.toPlainText())
        self.window.stop_merge()
        self.wait_for(lambda: not self.window.busy)
        self.window.worker.wait(1000)
        self.assertEqual(self.window.status_label.text(), "合并已停止")
        self.assertEqual(self.window.progress.value(), 0)
        self.assertFalse(self.window.stop_button.isEnabled())
        self.assertTrue(self.window.tree.isEnabled())

    def test_demo_never_opens_a_directory_or_launches_a_merge(self):
        demo = MergeWindow(home=Path("/no-real-dataset"), demo=True)
        try:
            self.assertEqual(demo.metrics["datasets"].text(), "3")
            self.assertEqual(demo.metrics["episodes"].text(), "30")
            with patch.object(QFileDialog, "getExistingDirectory") as chooser:
                demo.browse_source()
                demo.add_dataset()
                demo.start_merge()
            chooser.assert_not_called()
            self.assertIsNone(demo.worker)
            self.assertFalse(demo.merge_button.isEnabled())
            self.assertIsNone(DatasetEntry.read(Path("/missing-dataset")).episodes)
        finally:
            demo.close()
            demo.deleteLater()


@unittest.skipUnless(HAS_QT, "the desktop extra is required")
class MergeLauncherTests(unittest.TestCase):
    def test_default_and_demo_launch_the_qt_window(self):
        for arguments, source, demo in (
            ([], None, False),
            (["--source", "/tmp/data", "--demo"], Path("/tmp/data"), True),
        ):
            with (
                patch.dict(os.environ, {"LEROBOT_GUI_BACKEND": "auto"}),
                patch.object(sys, "argv", ["merge-datasets", *arguments]),
                patch.object(merge_datasets_window, "launch") as launch,
            ):
                merge_datasets_gui.main()
            launch.assert_called_once_with(source=source, demo=demo)

    def test_old_zenity_backend_remains_explicitly_available(self):
        with (
            patch.object(sys, "argv", ["merge-datasets", "--backend", "zenity"]),
            patch.object(merge_datasets_gui, "run_zenity_gui") as launch,
        ):
            merge_datasets_gui.main()
        launch.assert_called_once_with()

    def test_missing_qt_falls_back_only_in_auto_mode(self):
        original_import = builtins.__import__

        def without_qt(name, *args, **kwargs):
            if name == "merge_datasets_window":
                raise ModuleNotFoundError("missing Qt", name="PyQt5")
            return original_import(name, *args, **kwargs)

        for backend in ("auto", "qt"):
            with (
                patch.object(sys, "argv", ["merge-datasets", "--backend", backend]),
                patch.object(builtins, "__import__", without_qt),
                patch.object(merge_datasets_gui.shutil, "which", return_value="/usr/bin/zenity"),
                patch.object(merge_datasets_gui, "run_zenity_gui") as fallback,
                redirect_stderr(io.StringIO()) as errors,
            ):
                if backend == "qt":
                    with self.assertRaises(SystemExit) as exit_context:
                        merge_datasets_gui.main()
                    self.assertEqual(exit_context.exception.code, 1)
                    self.assertIn("desktop", errors.getvalue())
                    fallback.assert_not_called()
                else:
                    merge_datasets_gui.main()
                    fallback.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
