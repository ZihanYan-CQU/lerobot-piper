"""Register configuration only; importing the plugin never connects CAN or opens a viewer."""

from .config import PiperMujocoConfig, PiperMujocoEnvConfig

__all__ = ["PiperMujoco", "PiperMujocoConfig", "PiperMujocoEnvConfig"]


def __getattr__(name: str):
    if name == "PiperMujoco":
        from .robot import PiperMujoco

        return PiperMujoco
    raise AttributeError(name)
