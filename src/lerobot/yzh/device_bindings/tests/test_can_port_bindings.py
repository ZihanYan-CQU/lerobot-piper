"""Exercise CAN binding, role swaps, and recovery without touching network interfaces."""

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "can_port_bindings.py"
SPEC = importlib.util.spec_from_file_location("can_port_bindings", SCRIPT)
can = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = can
SPEC.loader.exec_module(can)


class CanBindingsTest(unittest.TestCase):
    def setUp(self):
        self.master = can.CanPort("can0", "1-1.2:1.0", 7, "master-serial", False, 500000)
        self.follower = can.CanPort("can_follower", "1-1.3:1.0", 8, "follower-serial", True, 1000000)
        self.ports = [self.master, self.follower]

    def test_default_bindings_and_invalid_selection(self):
        self.assertEqual(
            can.validate_bindings(self.ports, can.DEFAULT_MASTER, can.DEFAULT_FOLLOWER, 1000000),
            [(self.master, "can_master"), (self.follower, "can_follower")],
        )
        for master, follower, bitrate in (
            (self.master.usb, self.master.usb, 1000000),
            ("1-1.9:1.0", self.follower.usb, 1000000),
            (self.master.usb, self.follower.usb, 0),
            (self.master.usb, self.follower.usb, 2000000),
        ):
            with self.subTest(master=master, follower=follower, bitrate=bitrate), self.assertRaises(ValueError):
                can.validate_bindings(self.ports, master, follower, bitrate)

    def test_reconnected_adapter_is_rejected_even_at_the_same_usb_port(self):
        bindings = [(self.master, "can_master"), (self.follower, "can_follower")]
        tokens = [f"{port.usb}={can.identity_token(port)}" for port, _ in bindings]
        can.validate_expected(bindings, tokens)
        with self.assertRaisesRegex(ValueError, "设备已变化"):
            can.validate_expected([(replace(self.master, ifindex=12), "can_master"), bindings[1]], tokens)
        with self.assertRaisesRegex(ValueError, "不完整"):
            can.validate_expected(bindings, tokens[:1])

    def test_either_role_can_be_configured_without_the_other_adapter(self):
        self.assertEqual(
            can.validate_bindings([self.master], self.master.usb, None, 1000000),
            [(self.master, "can_master")],
        )
        self.assertEqual(
            can.validate_bindings([self.follower], None, self.follower.usb, 1000000),
            [(self.follower, "can_follower")],
        )
        with self.assertRaisesRegex(ValueError, "至少"):
            can.validate_bindings(self.ports, None, None, 1000000)

    def test_discovery_only_lists_usb_can_interfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            net = root / "net"
            net.mkdir()
            device = root / "usb" / "1-1.2" / "1-1.2:1.0"
            device.mkdir(parents=True)
            (device.parent / "serial").write_text("master-serial")
            for name, kind in (("can0", "280"), ("ether0", "1"), ("vcan0", "280")):
                path = net / name
                path.mkdir()
                (path / "type").write_text(kind)
                (path / "ifindex").write_text("7")
                (path / "flags").write_text("0x1")
                if name != "vcan0":
                    (path / "device").symlink_to(device)
            details = {"can0": {"linkinfo": {"info_data": {"bittiming": {"bitrate": 1000000}}}}}
            with patch.object(can, "SYS_NET", net), patch.object(can, "link_details", return_value=details):
                ports = can.discover_ports()
            self.assertEqual(len(ports), 1)
            self.assertEqual(ports[0].usb, "1-1.2:1.0")
            self.assertEqual(ports[0].bitrate, 1000000)
            self.assertTrue(ports[0].up)

    def exercise_activation(self, ports, *, roles=("can_master", "can_follower"), fail_second=False, false_success=False):
        state = {port.usb: port for port in ports}
        calls = []
        configurations = 0

        def command(*args):
            nonlocal configurations
            calls.append(args)
            self.assertEqual(args[:4], ("ip", "link", "set", "dev"))
            port = next(port for port in state.values() if port.name == args[4])
            action = args[5]
            if action == "name":
                self.assertFalse(any(other.name == args[6] and other.usb != port.usb for other in state.values()))
                port = replace(port, name=args[6])
            elif action == "type":
                configurations += 1
                if fail_second and configurations == 2:
                    raise RuntimeError("activation failed")
                port = replace(port, bitrate=int(args[-1]))
            else:
                self.assertIn(action, ("up", "down"))
                if not (false_success and action == "up"):
                    port = replace(port, up=action == "up")
            state[port.usb] = port
            return ""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for port in ports:
                (root / port.name).touch()
            with (
                patch.object(can, "SYS_NET", root),
                patch.object(can.os, "geteuid", return_value=0),
                patch.object(can.shutil, "which", return_value="/usr/bin/tool"),
                patch.object(can, "find_current", side_effect=lambda original: state[original.usb]),
                patch.object(can, "run", side_effect=command),
                redirect_stdout(io.StringIO()),
            ):
                bindings = [(port, role) for port, role in zip(ports, roles, strict=False) if role]
                if fail_second or false_success:
                    with self.assertRaisesRegex(RuntimeError, "CAN 绑定失败"):
                        can.apply_bindings(bindings, 1000000)
                else:
                    can.apply_bindings(bindings, 1000000)
        return state, calls

    def test_activation_configures_selected_port_and_skips_unchanged_port(self):
        state, calls = self.exercise_activation(self.ports)
        self.assertIn(("ip", "link", "set", "dev", "can_master", "type", "can", "bitrate", "1000000"), calls)
        self.assertFalse(any(call[4] == "can_follower" for call in calls))
        self.assertEqual(state[self.master.usb].name, "can_master")
        self.assertTrue(state[self.master.usb].up)

    def test_single_arm_activation_leaves_unselected_arm_unchanged(self):
        for roles, selected in ((("can_master", None), 0), ((None, "can_follower"), 1)):
            with self.subTest(roles=roles):
                ports = [self.master, replace(self.follower, name="can1", bitrate=500000)]
                state, calls = self.exercise_activation(ports, roles=roles)
                self.assertEqual(state[ports[1 - selected].usb], ports[1 - selected])
                self.assertEqual(state[ports[selected].usb].name, roles[selected])
                self.assertTrue(state[ports[selected].usb].up)

    def test_single_arm_rejects_a_name_occupied_by_an_unselected_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "can_master").touch()
            with (
                patch.object(can, "SYS_NET", root),
                patch.object(can.os, "geteuid", return_value=0),
                patch.object(can.shutil, "which", return_value="/usr/bin/ip"),
                patch.object(can, "run") as command,
                self.assertRaisesRegex(ValueError, "未选择"),
            ):
                can.apply_bindings([(self.master, "can_master")], 1000000)
            command.assert_not_called()

    def test_swapping_master_and_follower_frees_names_first(self):
        ports = [replace(self.master, name="can_follower", up=True), replace(self.follower, name="can_master")]
        state, _ = self.exercise_activation(ports)
        self.assertEqual(state[ports[0].usb].name, "can_master")
        self.assertEqual(state[ports[1].usb].name, "can_follower")

    def test_second_activation_failure_restores_original_names_rates_and_states(self):
        ports = [self.master, replace(self.follower, name="can1", bitrate=500000)]
        state, _ = self.exercise_activation(ports, fail_second=True)
        self.assertEqual(list(state.values()), ports)

    def test_zero_exit_without_correct_kernel_state_is_not_success(self):
        state, _ = self.exercise_activation(self.ports, false_success=True)
        self.assertEqual(list(state.values()), self.ports)

    def test_dry_run_never_activates_ports(self):
        with (
            patch.object(can, "discover_ports", return_value=self.ports),
            patch.object(can, "apply_bindings") as activate,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(can.main(["--dry-run"]), 0)
            activate.assert_not_called()
            self.assertIn("1-1.2:1.0 → can_master", output.getvalue())
            self.assertIn("1-1.3:1.0 → can_follower", output.getvalue())

    def test_explicit_single_role_does_not_add_the_other_default(self):
        for option, port, name in (("--master", self.master, "can_master"), ("--follower", self.follower, "can_follower")):
            with (
                self.subTest(option=option),
                patch.object(can, "discover_ports", return_value=[port]),
                patch.object(can, "apply_bindings") as activate,
            ):
                self.assertEqual(can.main([option, port.usb]), 0)
                activate.assert_called_once_with([(port, name)], 1000000)

    def test_legacy_single_interface_mode_resolves_the_only_adapter(self):
        with (
            patch.object(can, "discover_ports", return_value=[self.master]),
            patch.object(can, "apply_bindings") as activate,
        ):
            self.assertEqual(can.main(["--interface", "can0", "--bitrate", "500000"]), 0)
            activate.assert_called_once_with([(self.master, "can0")], 500000)


if __name__ == "__main__":
    unittest.main()
