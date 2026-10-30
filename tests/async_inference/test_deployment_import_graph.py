# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Keep canonical deployment entry points independent of historical versions."""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import lerobot.async_inference.client.robot_client_dagger as robot_client_dagger
import lerobot.async_inference.client.robot_client_deployment as robot_client_deployment
import lerobot.async_inference.server.policy_server_dagger as policy_server_dagger
import lerobot.async_inference.server.policy_server_deployment as policy_server_deployment

_HISTORICAL_CLIENT_MODULES = {
    "robot_client",
    "robot_client_rtc",
    "robot_client_rtc_interpolated",
    "robot_client_rtc_interpolated_filtered",
    "robot_client_rtc_interpolated_filtered_resettable",
    "robot_client_rtc_interpolated_filtered_resettable_dagger",
}
_HISTORICAL_SERVER_MODULES = {
    "policy_server",
    "policy_server_rtc",
    "policy_server_rtc_resettable",
    "policy_server_rtc_resettable_dagger",
}


def _relative_imports(module) -> set[str]:
    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text())
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module is not None
    }


def test_deployment_modules_import_no_historical_implementation():
    assert _relative_imports(robot_client_deployment).isdisjoint(_HISTORICAL_CLIENT_MODULES)
    assert _relative_imports(policy_server_deployment).isdisjoint(_HISTORICAL_SERVER_MODULES)


def test_dagger_modules_depend_only_on_canonical_deployment_layers():
    client_imports = _relative_imports(robot_client_dagger)
    server_imports = _relative_imports(policy_server_dagger)

    assert "robot_client_deployment" in client_imports
    assert client_imports.isdisjoint(_HISTORICAL_CLIENT_MODULES)
    assert "policy_server_deployment" in server_imports
    assert server_imports.isdisjoint(_HISTORICAL_SERVER_MODULES)


@pytest.mark.parametrize("role", ["server", "client"])
def test_current_entry_points_work_in_a_bundle_without_the_other_role(tmp_path, role):
    """Exercise the exact subset copied to each host, including pickle imports."""
    source = Path(policy_server_deployment.__file__).resolve().parent.parent
    bundle = tmp_path / "async_inference"
    bundle.mkdir()
    shutil.copy2(source / "__init__.py", bundle / "__init__.py")
    historical = _HISTORICAL_SERVER_MODULES | _HISTORICAL_CLIENT_MODULES
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", *(f"{name}.py" for name in historical))
    for directory in ("shared", role):
        shutil.copytree(source / directory, bundle / directory, ignore=ignored)

    script = """
import importlib
import pickle
import sys

import torch
import lerobot.async_inference as package

package.__path__ = [sys.argv[1]]
role = sys.argv[2]
prefix = 'policy_server' if role == 'server' else 'robot_client'
for variant in ('deployment', 'dagger', 'recap'):
    module = importlib.import_module(f'lerobot.async_inference.{role}.{prefix}_{variant}')
    assert module.__file__.startswith(sys.argv[1])

from lerobot.async_inference.shared.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.shared.image_transport import ImageResizeSpec, PackedPolicyImage
from lerobot.async_inference.shared.rtc_remote import (
    DEPLOYMENT_RTC_PROTOCOL_VERSION,
    RemoteRTCActionChunk,
    RemoteRTCObservation,
    RemoteRTCPolicySetup,
)

payload = (
    RemoteRTCPolicySetup(
        RemotePolicyConfig('pi05', 'checkpoint', {}, 1),
        30,
        protocol_version=DEPLOYMENT_RTC_PROTOCOL_VERSION,
    ),
    RemoteRTCObservation(TimedObservation(1.0, 0, {'joint.pos': 1.0}), 0, -1),
    RemoteRTCActionChunk(0, 0, 1.0, actions=[TimedAction(1.0, 0, torch.ones(3))]),
    PackedPolicyImage(ImageResizeSpec((3, 2, 2), (2, 2)), b'image'),
)
wire = pickle.dumps(payload)
restored = pickle.loads(wire)
assert restored[0] == payload[0]
assert restored[1] == payload[1]
torch.testing.assert_close(restored[2].actions[0].action, torch.ones(3))
assert restored[3] == payload[3]
other_role = 'client' if role == 'server' else 'server'
assert not any(name.startswith(f'lerobot.async_inference.{other_role}') for name in sys.modules)
assert b'lerobot.async_inference.server' not in wire
assert b'lerobot.async_inference.client' not in wire
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(bundle), role],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(source.parents[1])},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
