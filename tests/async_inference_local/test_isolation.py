import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import lerobot
import lerobot.async_inference_local as local


def test_no_static_import_of_remote_deployment_or_transport():
    root = Path(local.__file__).parent
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            for name in modules:
                assert not any(
                    name == banned or name.startswith(banned + ".")
                    for banned in (
                        "lerobot.async_inference",
                        "lerobot.transport",
                        "grpc",
                        "pickle",
                    )
                ), (path, name)


@pytest.mark.parametrize("entrypoint", ["deploy", "dagger"])
def test_entrypoint_and_pi05_model_import_with_remote_modules_blocked(entrypoint):
    # Fresh interpreter also catches indirect imports through reusable components.
    script = """
import importlib.abc
import sys
class BlockRemote(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + '.') for n in ('lerobot.async_inference', 'lerobot.transport')):
            raise AssertionError('Remote dependency: ' + fullname)
sys.meta_path.insert(0, BlockRemote())
sys.path.insert(0, sys.argv[1])
import lerobot
lerobot.__path__.insert(0, sys.argv[2])
from importlib import import_module
main = import_module('lerobot.async_inference_local.scripts.' + sys.argv[3]).main
from lerobot.policies.factory import get_policy_class
assert get_policy_class('pi05').__name__ == 'PI05Policy'
sys.argv = ['local', '--help']
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(lerobot.__file__).parents[1]),
            str(Path(local.__file__).parents[1]),
            entrypoint,
        ],
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "--inference.device" in result.stdout
    assert "--server_address" not in result.stdout
