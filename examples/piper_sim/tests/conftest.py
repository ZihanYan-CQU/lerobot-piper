import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


@pytest.fixture(scope="session")
def model_path():
    from lerobot_robot_piper_mujoco.config import default_model_path

    path = default_model_path()
    if not path.is_file():
        pytest.skip("Prepare the Menagerie Piper model with piper-sim prepare to run physics tests")
    return str(path.resolve())


@pytest.fixture
def robot(model_path):
    from lerobot_robot_piper_mujoco.config import PiperMujocoConfig
    from lerobot_robot_piper_mujoco.robot import PiperMujoco

    instance = PiperMujoco(
        PiperMujocoConfig(
            model_path=model_path,
            width=80,
            height=64,
            randomize=False,
        )
    )
    instance.connect()
    try:
        yield instance
    finally:
        instance.disconnect()
