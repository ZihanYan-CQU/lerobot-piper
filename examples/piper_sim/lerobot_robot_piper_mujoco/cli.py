"""Simulation-only session orchestration, leaving the existing hardware CLIs untouched."""

import argparse
import importlib.metadata
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ACTION_KEYS, TASK, PiperMujocoConfig
from .controllers import (
    ExpertController,
    KeyboardController,
    LeaderController,
    PolicyController,
    align_virtual_follower,
)
from .robot import PiperMujoco


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _robot_config(args) -> PiperMujocoConfig:
    return PiperMujocoConfig(
        model_path=args.model_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_names=args.cameras.split(","),
        viewer=args.viewer,
        seed=args.seed,
        randomize=args.randomize,
        joint_signs=args.joint_signs,
        joint_offsets=args.joint_offsets,
        max_joint_step=args.max_joint_step,
    )


def _controller(args, robot):
    if getattr(args, "policy", None):
        return PolicyController(robot, args.policy, args.device)
    return {
        "expert": ExpertController,
        "keyboard": KeyboardController,
        "leader": LeaderController,
    }[args.controller](robot)


def _pace(started: float, fps: int, realtime: bool):
    if realtime:
        time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))


def _close_session(robot, controller):
    try:
        robot.disconnect()
    finally:
        if controller is not None:
            controller.close()


@contextmanager
def _recording_session(dataset):
    from lerobot.datasets import VideoEncodingManager

    with VideoEncodingManager(dataset):
        try:
            yield
        finally:
            # VideoEncodingManager finalizes on exit, so discard interrupted
            # frames while the writer is still open, not in outer cleanup.
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()


class VideoWriter:
    """Incremental preview video; no episode-sized RGB buffer."""

    def __init__(self, path: Path | None, fps: int):
        self.container = self.stream = None
        if path is not None:
            import av

            path.parent.mkdir(parents=True, exist_ok=True)
            self.container = av.open(str(path), mode="w")
            self.stream = self.container.add_stream("libx264", rate=fps)
            self.stream.pix_fmt = "yuv420p"

    def add(self, image: np.ndarray):
        if self.container is not None:
            import av

            self.stream.height, self.stream.width = image.shape[:2]
            for packet in self.stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                self.container.mux(packet)

    def close(self):
        if self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.container = None


def record(args):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    robot = PiperMujoco(_robot_config(args))
    root = args.root.resolve()
    manifest_path = root / "simulation.json"
    manifest = {"version": 1, "robot_config": asdict(robot.config), "episodes": []}
    dataset = controller = None
    completed = 0
    realtime = args.realtime if args.realtime is not None else args.controller != "expert"
    try:
        features = {
            **hw_to_dataset_features(robot.action_features, "action", use_video=args.video),
            **hw_to_dataset_features(robot.observation_features, "observation", use_video=args.video),
        }
        encoder = RGBEncoderConfig(vcodec="h264", preset="fast")
        if args.resume:
            # Refuse accidental mixing with a hardware dataset or a different simulation contract.
            manifest = json.loads(manifest_path.read_text())
            old_config = {k: v for k, v in manifest["robot_config"].items() if k not in {"seed", "viewer"}}
            new_config = {k: v for k, v in asdict(robot.config).items() if k not in {"seed", "viewer"}}
            if old_config != new_config:
                raise ValueError(
                    "Resume robot configuration differs from the dataset's simulation configuration"
                )
            dataset = LeRobotDataset.resume(
                args.repo_id, root=root, rgb_encoder=encoder, video_backend="pyav", image_writer_threads=4
            )
            if dataset.meta.fps != args.fps or dataset.meta.robot_type != robot.name:
                raise ValueError("Cannot resume a dataset with a different FPS or robot type")
            if len(manifest["episodes"]) != dataset.num_episodes:
                raise ValueError(
                    "Dataset and simulation manifest episode counts differ; inspect before resuming"
                )
        else:
            dataset = LeRobotDataset.create(
                args.repo_id,
                args.fps,
                features,
                root=root,
                robot_type=robot.name,
                use_videos=args.video,
                rgb_encoder=encoder,
                video_backend="pyav",
                image_writer_threads=4,
            )
        robot.connect()
        controller = _controller(args, robot)
        controller.connect()
        _write_json(manifest_path, manifest)
        stop = False
        max_attempts = args.attempts or max(args.episodes * 3, args.episodes)
        next_seed = max([args.seed, *[item["seed"] + 1 for item in manifest["episodes"]]])
        with _recording_session(dataset):
            for attempt in range(max_attempts):
                if completed >= args.episodes or stop:
                    break
                seed = next_seed + attempt
                robot.reset(seed)
                if isinstance(controller, LeaderController):
                    print(
                        "Aligning virtual follower to the real leader; virtual scene reset does not home the leader.",
                        flush=True,
                    )
                    align_virtual_follower(robot, controller)
                controller.reset()
                initial_state = robot.simulation.snapshot()
                discard = False
                frame_count = 0
                started_episode = time.monotonic()
                for _ in range(round(args.duration * args.fps)):
                    started = time.monotonic()
                    if not robot.simulation.viewer_running():
                        stop = True
                        break
                    keys = robot.simulation.pop_keys()
                    if isinstance(controller, KeyboardController):
                        controller.handle_keys(keys)
                    if 259 in keys:  # Backspace: discard and retry.
                        discard = True
                        break
                    if 257 in keys:  # Enter: save episode early.
                        break
                    observation = robot.get_observation()
                    applied = robot.send_action(controller.action(observation))
                    # Unlike the legacy loop, record the final clipped/limited target actually issued.
                    frame = {
                        **build_dataset_frame(dataset.features, observation, "observation"),
                        **build_dataset_frame(dataset.features, applied, "action"),
                        "task": TASK,
                    }
                    dataset.add_frame(frame)
                    frame_count += 1
                    _pace(started, args.fps, realtime)
                    if controller.done:
                        break
                success = bool(robot.simulation.success)
                if frame_count == 0 or discard or (args.success_only and not success):
                    dataset.clear_episode_buffer()
                    print(f"Discarded attempt {attempt}: frames={frame_count}, success={success}", flush=True)
                    continue
                index = dataset.num_episodes
                dataset.save_episode()
                manifest["episodes"].append(
                    {
                        "episode_index": index,
                        "seed": seed,
                        "source": controller.source,
                        "success": success,
                        "frames": frame_count,
                        "initial_state": initial_state,
                        "wall_seconds": time.monotonic() - started_episode,
                        "final_state": robot.simulation.snapshot(),
                    }
                )
                _write_json(manifest_path, manifest)
                completed += 1
                print(f"Saved episode {index}: frames={frame_count}, success={success}", flush=True)
        if completed < args.episodes and not stop:
            raise RuntimeError(
                f"Saved {completed}/{args.episodes} requested episodes in {max_attempts} attempts"
            )
    finally:
        try:
            if dataset is not None:
                dataset.finalize()
        finally:
            _close_session(robot, controller)
    print(json.dumps({"root": str(root), "saved_episodes": completed, "manifest": str(manifest_path)}))


def run_session(args):
    robot = PiperMujoco(_robot_config(args))
    controller = None
    results = []
    realtime = args.realtime if args.realtime is not None else args.controller != "expert" and not args.policy
    try:
        robot.connect()
        controller = _controller(args, robot)
        controller.connect()
        for episode in range(args.episodes):
            seed = args.seed + episode
            robot.reset(seed)
            if isinstance(controller, LeaderController):
                align_virtual_follower(robot, controller)
            controller.reset()
            video_path = args.video_dir / f"episode_{episode:04d}.mp4" if args.video_dir else None
            video = VideoWriter(video_path, args.fps)
            steps = 0
            started_episode = time.monotonic()
            try:
                for _ in range(round(args.duration * args.fps)):
                    started = time.monotonic()
                    if not robot.simulation.viewer_running():
                        break
                    keys = robot.simulation.pop_keys()
                    if isinstance(controller, KeyboardController):
                        controller.handle_keys(keys)
                    if 257 in keys:
                        break
                    observation = robot.get_observation()
                    video.add(observation[next(iter(robot.cameras))])
                    robot.send_action(controller.action(observation))
                    steps += 1
                    _pace(started, args.fps, realtime)
                    if controller.done or (args.policy and robot.simulation.success):
                        break
            finally:
                video.close()
            result = {
                "seed": seed,
                "steps": steps,
                "success": bool(robot.simulation.success),
                "sim_seconds": steps / args.fps,
                "wall_seconds": time.monotonic() - started_episode,
                "cube_position": robot.simulation.cube_position.tolist(),
                "goal_position": robot.simulation.goal_position.tolist(),
                "source": controller.source,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            if not robot.simulation.viewer_running():
                break
    finally:
        _close_session(robot, controller)
    report = {
        "episodes": results,
        "success_rate": float(np.mean([item["success"] for item in results])) if results else 0.0,
        "policy": args.policy,
        "robot_config": asdict(robot.config),
    }
    if args.report:
        _write_json(args.report, report)
    print(json.dumps({"success_rate": report["success_rate"], "episodes": len(results)}))


def replay(args):
    from lerobot.datasets import LeRobotDataset

    manifest = json.loads((args.root / "simulation.json").read_text())
    config = manifest["robot_config"]
    config["viewer"] = args.viewer
    if args.model_path:
        config["model_path"] = args.model_path
    robot = PiperMujoco(PiperMujocoConfig(**config))
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode], video_backend="pyav")
    initial = manifest["episodes"][args.episode]["initial_state"]
    video = VideoWriter(args.video, robot.config.fps)
    maximum_error = 0.0
    try:
        robot.connect()
        robot.simulation.restore(initial)
        action_names = dataset.features["action"]["names"]
        if tuple(action_names) != ACTION_KEYS:
            raise ValueError("Dataset action ordering differs from the Piper simulation contract")
        for item in dataset:
            started = time.monotonic()
            expected = np.asarray(item["observation.state"])
            maximum_error = max(maximum_error, float(np.max(np.abs(robot.simulation.state() - expected))))
            video.add(robot.simulation.image(next(iter(robot.cameras))))
            robot.send_action(dict(zip(action_names, map(float, item["action"]), strict=True)))
            _pace(started, robot.config.fps, args.realtime)
            if not robot.simulation.viewer_running():
                break
        result = {"success": bool(robot.simulation.success), "max_state_error": maximum_error}
        print(json.dumps(result))
        if maximum_error > args.tolerance:
            raise RuntimeError(f"Replay state error {maximum_error:.6f} exceeds tolerance {args.tolerance}")
    finally:
        try:
            video.close()
        finally:
            robot.disconnect()


def doctor(args):
    import mujoco

    import lerobot

    robot = PiperMujoco(_robot_config(args))
    try:
        robot.connect()
        before = robot.simulation.data.time
        obs = robot.get_observation()
        assert robot.simulation.data.time == before
        robot.send_action(dict(zip(ACTION_KEYS, map(float, robot.simulation.state()), strict=True)))
        report = {
            "lerobot_source": lerobot.__file__,
            "mujoco_version": mujoco.__version__,
            "plugin_version": importlib.metadata.version("lerobot_robot_piper_mujoco"),
            "action_keys": ACTION_KEYS,
            "cameras": {key: list(obs[key].shape) for key in robot.cameras},
            "control_tick_seconds": float(robot.simulation.data.time - before),
            "real_hardware_connected": False,
        }
        print(json.dumps(report, indent=2))
    finally:
        robot.disconnect()


def _add_robot_args(parser):
    parser.add_argument("--model-path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--cameras", default="top,wrist")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-signs", nargs=6, type=float, default=[1.0] * 6)
    parser.add_argument("--joint-offsets", nargs=6, type=float, default=[0.0] * 6)
    parser.add_argument("--max-joint-step", type=float, default=0.10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Copy or download the pinned Menagerie model")
    prepare.add_argument("--source", type=Path)
    prepare.add_argument("--destination", type=Path)
    check = sub.add_parser("doctor", help="Check physics and virtual cameras without CAN")
    _add_robot_args(check)
    check.set_defaults(func=doctor)
    for command in ("teleoperate", "record", "evaluate"):
        child = sub.add_parser(command)
        _add_robot_args(child)
        child.add_argument("--controller", choices=["expert", "leader", "keyboard"], default="expert")
        child.add_argument("--episodes", type=int, default=1)
        child.add_argument("--duration", type=float, default=20.0)
        child.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=None)
        if command == "record":
            child.add_argument("--root", type=Path, required=True)
            child.add_argument("--repo-id", default="local/piper_sim_pick_place")
            child.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
            child.add_argument("--resume", action="store_true")
            child.add_argument("--success-only", action="store_true")
            child.add_argument("--attempts", type=int)
            child.set_defaults(func=record)
        else:
            child.add_argument("--policy", help="A LeRobot checkpoint; overrides --controller")
            child.add_argument("--device", default="cpu")
            child.add_argument("--video-dir", type=Path)
            child.add_argument("--report", type=Path)
            child.set_defaults(func=run_session)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--root", type=Path, required=True)
    replay_parser.add_argument("--repo-id", default="local/piper_sim_pick_place")
    replay_parser.add_argument("--episode", type=int, default=0)
    replay_parser.add_argument("--viewer", action="store_true")
    replay_parser.add_argument("--model-path")
    replay_parser.add_argument("--realtime", action="store_true")
    replay_parser.add_argument("--video", type=Path)
    replay_parser.add_argument("--tolerance", type=float, default=0.002)
    replay_parser.set_defaults(func=replay)
    args = parser.parse_args()
    if args.command == "prepare":
        from .assets import prepare_assets

        print(prepare_assets(args.destination, args.source))
        return
    if (
        getattr(args, "episodes", 1) <= 0
        or not math.isfinite(getattr(args, "duration", 1))
        or getattr(args, "duration", 1) <= 0
        or (getattr(args, "attempts", None) is not None and args.attempts <= 0)
    ):
        parser.error("episodes, attempts, and duration must be finite and positive")
    needs_video = (args.command == "record" and args.video) or bool(getattr(args, "video_dir", None))
    if needs_video and (args.width % 2 or args.height % 2):
        parser.error("H.264 video requires even width and height")
    args.func(args)


if __name__ == "__main__":
    main()
