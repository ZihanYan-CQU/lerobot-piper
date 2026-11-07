"""Single-owner RLT service used by policy_server_deployment's opt-in RLT mode."""

import copy
import threading
from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue

from lerobot.policies.rlt_actor import RLTActorConfig, RLTActorPolicy
from lerobot.rl.rlt.features import FrozenRLTFeatures
from lerobot.rl.rlt.replay import FixedWindowReplayBuffer, replay_spec
from lerobot.rl.rlt.rtc import RLTOverlapPlanner, warmstart_rtc_actor
from lerobot.utils.constants import OBS_STATE

from ..shared.image_transport import (
    IMAGE_TRANSPORT_METADATA_KEY,
    image_transport_metadata,
    negotiate_image_transport,
    prepare_remote_observation,
)
from ..shared.rlt_remote import RLT_PROTOCOL_VERSION, RLTFailure, RLTRequest, RLTSetup


class RemoteRLTServer:
    def __init__(self, server, setup: RLTSetup, context):
        cfg = server.config.rlt
        if server.prefix != "policy_server_deployment":
            raise ValueError("RLT currently supports policy_server_deployment only")
        if not cfg.enabled:
            raise ValueError("Enable server --rlt.enabled=true before requesting RLT")
        if (
            setup.version != RLT_PROTOCOL_VERSION
            or setup.policy.policy_type != "pi05"
            or setup.fps != server.config.fps
            or setup.execution_horizon != cfg.execution_horizon
            or setup.prefix_steps != cfg.prefix_steps
        ):
            raise ValueError("RLT client/server protocol, fps and window settings must match")
        if (
            not isinstance(setup.session, str)
            or len(setup.session) != 32
            or any(ch not in "0123456789abcdef" for ch in setup.session)
        ):
            raise ValueError("Invalid RLT session identifier")
        actor = None

        def configure_features(source, identity):
            nonlocal actor
            if source.type != "pi05" or source.backend != "rlinf" or not source.use_rlt or source.use_peft:
                raise ValueError("RLT deployment requires a merged/full Stage 1 RLinf PI05 RLT checkpoint")
            expected_names = source.action_feature_names
            if not expected_names or [x.removesuffix(".pos") for x in expected_names] != [
                x.removesuffix(".pos") for x in setup.action_names
            ]:
                raise ValueError("Robot action order must match saved Stage 1 action_feature_names")
            actor_cfg = RLTActorConfig(
                feature_model_id=identity,
                z_dim=source.rlt_embed_dim,
                proprio_dim=source.input_features[OBS_STATE].shape[0],
                action_dim=source.output_features["action"].shape[0],
                execution_horizon=cfg.execution_horizon,
                reference_horizon=source.chunk_size,
                rtc_prefix_steps=cfg.prefix_steps,
                device=setup.policy.device,
                push_to_hub=False,
            )
            if cfg.actor_checkpoint:
                saved_cfg = RLTActorConfig.from_pretrained(cfg.actor_checkpoint)
                saved_cfg.device = setup.policy.device
                if saved_cfg.feature_model_id != identity:
                    raise ValueError("RLT actor belongs to different frozen Stage 1 weights/processors")
                loaded = RLTActorPolicy.from_pretrained(cfg.actor_checkpoint, config=saved_cfg, strict=True)
                actor_cfg = replace(
                    actor_cfg, hidden_dims=saved_cfg.hidden_dims, exploration_std=saved_cfg.exploration_std
                )
                if replay_spec(saved_cfg) == replay_spec(actor_cfg):
                    actor = loaded
                elif cfg.warmstart_stage2_actor:
                    actor = RLTActorPolicy(actor_cfg)
                    warmstart_rtc_actor(loaded, actor)
                else:
                    raise ValueError(
                        "RTC actor schema differs; explicitly enable warmstart_stage2_actor for policy-only transfer"
                    )
                actor.to(setup.policy.device).eval()
            return actor_cfg

        compile_model = server.config.pi05_inference.compile_model
        self.features = FrozenRLTFeatures.from_pretrained(
            setup.policy.pretrained_name_or_path,
            config_factory=configure_features,
            device=setup.policy.device,
            compile_model=bool(compile_model),
            compile_mode=server.config.pi05_inference.compile_mode or "default",
        )
        actor_cfg = self.features.config
        self.image_plan = {}
        if setup.client_image_resize:
            self.image_plan = negotiate_image_transport(
                "pi05",
                self.features.policy.config,
                setup.policy.lerobot_features,
                self.features.preprocessor,
                mode=setup.image_transport_mode,
            )
        if context is not None:
            context.send_initial_metadata(
                ((IMAGE_TRANSPORT_METADATA_KEY, image_transport_metadata(self.image_plan)),)
            )
        self.replay = FixedWindowReplayBuffer(actor_cfg, capacity=cfg.replay_capacity)
        self.planner = RLTOverlapPlanner(
            self.features,
            self.replay,
            actor=actor,
            session=setup.session,
            journal_dir=Path(cfg.replay_dir) / setup.session,
            policy_version=cfg.actor_version,
        )
        self.setup, self.server = setup, server
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.requests = Queue(maxsize=1)
        self.result = None
        self.busy = False
        self.latest_generation = -1
        self.failed = None
        self.thread = threading.Thread(target=self._run, name="RemoteRLTInference", daemon=True)
        self.thread.start()

    def submit(self, request):
        if not isinstance(request, RLTRequest) or request.session != self.setup.session:
            raise ValueError("RLT request belongs to a different session")
        with self.changed:
            if self.stop_event.is_set() or self.failed:
                raise RuntimeError("RLT worker stopped or failed")
            if request.generation < self.latest_generation:
                raise ValueError("Stale RLT generation")
            # Do not silently replace training observations/receipts as ordinary RTC does.
            if self.busy or self.result is not None:
                raise RuntimeError("RLT permits exactly one outstanding request; drain it before continuing")
            if not request.dry_run:
                self.latest_generation = request.generation
            self.busy = True
            self.requests.put_nowait(copy.deepcopy(request))

    def _run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    request = self.requests.get(timeout=0.1)
                except Empty:
                    continue
                observation = prepare_remote_observation(
                    request.observation,
                    self.setup.policy.lerobot_features,
                    self.features.policy.config.image_features,
                    self.image_plan,
                )
                request = replace(request, observation=observation)
                result = self.planner.process(
                    request,
                    admit=lambda request=request: (
                        not self.stop_event.is_set()
                        and (request.dry_run or request.generation == self.latest_generation)
                    ),
                    commit_guard=self.lock,
                )
                with self.changed:
                    if not self.stop_event.is_set():
                        self.result = result
                    self.busy = False
                    self.changed.notify_all()
        except Exception as exc:
            self.server._remember_fatal_cuda_error(exc)
            self.server.logger.exception("RLT inference/replay failed")
            with self.changed:
                self.failed = str(exc)
                self.result = RLTFailure(f"{type(exc).__name__}: {exc}")
                self.busy = False
                self.changed.notify_all()

    def get(self, context):
        with self.changed:
            while self.result is None and not self.stop_event.is_set():
                if context is not None and not context.is_active():
                    return None
                self.changed.wait(timeout=0.05)
            if context is not None and not context.is_active():
                return None
            result, self.result = self.result, None
            return result

    def close(self):
        with self.changed:
            self.stop_event.set()
            self.changed.notify_all()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("RLT worker did not stop; refusing a concurrent GPU owner")
