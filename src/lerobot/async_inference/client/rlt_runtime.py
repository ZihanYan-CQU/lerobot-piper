"""RLT overlap execution through the existing resettable deployment robot client."""

import copy
import importlib
import math
import pickle  # nosec
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import grpc
import torch

from lerobot.rl.rlt.rollout import RewardSignal
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from ..shared.image_transport import encode_observation_images, parse_image_transport_metadata
from ..shared.rlt_remote import (
    RLTFailure,
    RLTFinished,
    RLTPlan,
    RLTReceipt,
    RLTRequest,
    RLTSetup,
    RLTStepReceipt,
)


class ManualReward:
    """Sparse task feedback entered through /success, /failure or /truncate."""

    def __init__(self):
        self.lock = threading.Lock()
        self.signal = None

    def reset(self):
        with self.lock:
            self.signal = None

    def mark(self, outcome):
        with self.lock:
            self.signal = RewardSignal(
                1.0 if outcome == "success" else 0.0,
                terminated=outcome in {"success", "failure"},
                truncated=outcome == "truncate",
            )

    def __call__(self, observation, action):
        del observation, action
        with self.lock:
            return self.signal if self.signal is not None else RewardSignal(0.0)


def load_reward_provider(name):
    """Resolve before robot construction/connection so missing callbacks cannot actuate hardware."""
    if name == "manual":
        return ManualReward()
    if not isinstance(name, str) or ":" not in name:
        raise ValueError("RLT reward_provider must be 'manual' or 'module:function'")
    module, symbol = name.rsplit(":", 1)
    provider = getattr(importlib.import_module(module), symbol)
    if not callable(provider):
        raise TypeError("RLT reward provider is not callable")
    return provider


class RLTOverlapExecutor:
    """Transport-independent scheduler; one receipt row per model action, not motor tick."""

    def __init__(
        self,
        exchange,
        *,
        session,
        window=10,
        prefix=3,
        fps=30,
        margin=0.005,
        timeout=30,
        max_observation_age_s=0.2,
    ):
        from ..shared.rlt_config import validate_window

        validate_window(window, prefix)
        if not math.isfinite(fps) or fps <= 0 or not 0 <= margin < prefix / fps:
            raise ValueError("RLT prefix duration must exceed its positive timing margin")
        if not math.isfinite(max_observation_age_s) or max_observation_age_s <= 0:
            raise ValueError("RLT max_observation_age_s must be finite and positive")
        self.exchange, self.session = exchange, session
        self.window, self.prefix, self.fps = window, prefix, fps
        self.margin, self.timeout = margin, timeout
        self.max_observation_age_s = max_observation_age_s
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RLTTransport")
        self.spec = None

    def _exchange(self, request):
        result = self.exchange(request)
        arrived = time.monotonic()
        if isinstance(result, RLTFailure):
            raise RuntimeError(result.message)
        if not isinstance(result, (RLTPlan, RLTFinished)) or (
            result.session,
            result.generation,
            result.window,
        ) != (request.session, request.generation, request.window):
            raise ValueError("RLT response belongs to a different session/generation/window")
        if request.final:
            if not isinstance(result, RLTFinished):
                raise ValueError("RLT final receipt was not acknowledged")
        else:
            if not isinstance(result, RLTPlan) or result.start_step != request.start_step:
                raise ValueError("Invalid RLT plan start")
            spec = result.replay_spec
            if (
                result.execution_horizon != self.window
                or result.prefix_steps != self.prefix
                or spec.get("execution_horizon") != self.window
                or spec.get("rtc_prefix_steps") != self.prefix
                or not isinstance(spec.get("feature_model_id"), str)
                or not spec["feature_model_id"]
                or result.actions.shape != (spec.get("reference_horizon"), spec.get("action_dim"))
                or result.actions.device.type != "cpu"
                or not result.actions.is_floating_point()
                or result.actions.shape[0] < self.window + self.prefix
                or not torch.isfinite(result.actions).all()
                or type(result.policy_version) is not int
                or result.policy_version < 0
            ):
                raise ValueError("Invalid RLT plan shape, window, version or values")
            if self.spec is not None and self.spec != spec:
                raise ValueError("RLT server changed frozen features or replay contract")
            if request.committed_prefix is not None and not torch.equal(
                result.actions[: self.prefix], request.committed_prefix
            ):
                raise ValueError("RLT server changed an already committed prefix")
            self.spec = dict(spec)
        return result, arrived

    def run(self, initial_observation, step, *, generation, cancelled=None, observation_started_at=None):
        """Execute until a real boundary/cancellation; never extend a late plan's prefix."""
        cancelled = cancelled if cancelled is not None else threading.Event()
        request = RLTRequest(self.session, generation, 0, 0, initial_observation)
        submitted_at = time.monotonic()
        source_at = submitted_at if observation_started_at is None else observation_started_at
        if not math.isfinite(source_at) or source_at > submitted_at:
            raise ValueError("RLT source observation must have a finite, nonfuture monotonic timestamp")
        cold_deadline = source_at + self.max_observation_age_s
        if submitted_at >= cold_deadline:
            raise TimeoutError("RLT cold observation expired before inference")
        future = self.pool.submit(self._exchange, request)
        pending = None
        while True:
            if cancelled.is_set():
                raise InterruptedError("RLT execution generation cancelled")
            started = time.monotonic()
            receipts = []
            # The previous plan's reference tail is the only legal in-flight prefix.
            if pending is not None:
                for action in pending:
                    if cancelled.is_set():
                        raise InterruptedError("RLT execution generation cancelled")
                    if future.done():
                        # A completed failure invalidates unsent commands too.
                        future.result()
                    result = step(action.clone())
                    receipts.append(self._step_receipt(result, action.shape))
                    last_observation = result.observation
                    if result.terminated or result.truncated:
                        break
                ended = receipts[-1].terminated or receipts[-1].truncated
                if not ended and not future.done():
                    raise TimeoutError("RLT inference exceeded the committed RTC prefix; window aborted")
            # Cold/final waits are motion-free; cancellation must remain responsive.
            deadline = started + self.timeout
            if pending is None:
                deadline = min(deadline, cold_deadline)
            while not future.done():
                if cancelled.wait(0.01):
                    raise InterruptedError("RLT execution generation cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("RLT cold observation expired or inference timed out")
            plan, arrived = future.result()
            if cancelled.is_set():
                raise InterruptedError("RLT execution generation cancelled")
            if pending is None and time.monotonic() >= cold_deadline:
                raise TimeoutError("RLT cold observation expired during inference")
            if (
                pending is not None
                and not (receipts[-1].terminated or receipts[-1].truncated)
                and arrived > submitted_at + self.prefix / self.fps - self.margin
            ):
                raise TimeoutError("RLT result arrived outside the committed prefix timing budget")
            for action in plan.actions[len(receipts) : self.window]:
                if receipts and (receipts[-1].terminated or receipts[-1].truncated):
                    break
                if cancelled.is_set():
                    raise InterruptedError("RLT execution generation cancelled")
                result = step(action.clone())
                receipts.append(self._step_receipt(result, action.shape))
                last_observation = result.observation
            if cancelled.is_set():
                raise InterruptedError("RLT execution generation cancelled")
            n = len(receipts)
            ended = receipts[-1].terminated or receipts[-1].truncated
            receipt = RLTReceipt(plan.window, plan.start_step, plan.policy_version, tuple(receipts))
            pending = plan.actions[n : n + self.prefix].clone()
            request = RLTRequest(
                self.session,
                generation,
                plan.window + 1,
                plan.start_step + n,
                last_observation,
                pending,
                receipt,
                final=ended,
            )
            submitted_at = time.monotonic()
            future = self.pool.submit(self._exchange, request)
            if ended:
                while not future.done():
                    if cancelled.wait(0.01):
                        raise InterruptedError("RLT final receipt cancelled")
                    if time.monotonic() - submitted_at >= self.timeout:
                        raise TimeoutError("RLT final receipt timed out")
                response, _ = future.result()
                return response

    @staticmethod
    def _step_receipt(result, expected_shape):
        # Import lazily to keep the scheduler usable with ordinary task callback objects.
        from lerobot.rl.rlt.rollout import ExecutedStep

        if not isinstance(result, ExecutedStep):
            raise TypeError("RLT policy step must return ExecutedStep")
        if (
            not isinstance(result.executed_action, torch.Tensor)
            or result.executed_action.shape != expected_shape
            or not torch.isfinite(result.executed_action).all()
            or not math.isfinite(result.reward)
            or any(
                type(getattr(result, key)) is not bool for key in ("terminated", "truncated", "intervention")
            )
        ):
            raise ValueError("Invalid RLT actual command, reward or outcome")
        return RLTStepReceipt(
            result.executed_action.detach().float().cpu().clone(),
            float(result.reward),
            result.terminated,
            result.truncated,
            result.intervention,
        )

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class RemoteRLTClient:
    def __init__(self, client, reward):
        self.client, self.reward = client, reward
        self.session = uuid4().hex
        self.cancelled = threading.Event()
        cfg = client.config
        self.scheduler = RLTOverlapExecutor(
            self.exchange,
            session=self.session,
            window=cfg.rlt.execution_horizon,
            prefix=cfg.rlt.prefix_steps,
            fps=cfg.fps,
            margin=cfg.rlt.budget_margin_s,
            timeout=cfg.request_timeout_s,
            max_observation_age_s=cfg.observation_max_age_s,
        )
        # Receipt-bearing requests must never be automatically retried/replayed by gRPC.
        client.channel.close()
        client.channel = grpc.insecure_channel(
            cfg.server_address,
            [*grpc_channel_options(), ("grpc.enable_retries", 0)],
        )
        client.stub = services_pb2_grpc.AsyncInferenceStub(client.channel)

    def cancel(self):
        self.cancelled.set()

    def begin_rollout(self):
        """Initialize feedback/cancellation under the client's lifecycle lock, before RUNNING."""
        self.cancelled = threading.Event()
        if isinstance(self.reward, ManualReward):
            self.reward.reset()

    def mark_manual_outcome(self, outcome):
        """Serialize feedback admission with rollout start/reset/stop."""
        from .robot_client_deployment import RolloutState

        with self.client._lifecycle_lock:
            if (
                not isinstance(self.reward, ManualReward)
                or not self.client.running
                or self.client._rollout_state is not RolloutState.RUNNING
            ):
                return False
            self.reward.mark(outcome)
            return True

    def close(self):
        self.cancel()
        self.scheduler.close()

    def exchange(self, request):
        client = self.client
        if client._image_transport_plan:
            raw, _, _ = encode_observation_images(request.observation, client._image_transport_plan)
            request = replace(request, observation=raw)
        payload = pickle.dumps(request)  # nosec
        timeout = (
            client.config.startup_calibration_timeout_s
            if request.dry_run
            else client.config.request_timeout_s
        )
        client.stub.SendObservations(
            send_bytes_in_chunks(payload, services_pb2.Observation, silent=True),
            timeout=timeout,
        )
        response = client.stub.GetActions(services_pb2.Empty(), timeout=timeout)
        if not response.data:
            raise RuntimeError("RLT server returned an empty result")
        return pickle.loads(response.data)  # nosec

    def _capture(self, generation, *, after=None):
        """Return fresh sensors; an after bound applies to both state and camera timestamps."""
        client = self.client
        if after is not None and client.config.continuous_observation:
            deadline = time.monotonic() + client.config.observation_max_age_s
            with client._lifecycle_changed:
                while True:
                    if client._rollout_generation != generation or not client.running:
                        raise InterruptedError("RLT sensor wait cancelled by reset/stop")
                    sample = client._latest_observation
                    if (
                        sample is not None
                        and sample.generation == generation
                        and sample.started_at >= after
                        and (sample.oldest_camera_at is None or sample.oldest_camera_at >= after)
                        and sample.age(time.monotonic()) <= client.config.observation_max_age_s
                    ):
                        raw = copy.deepcopy(sample.observation)
                        raw["task"] = client._task
                        return raw, sample
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "RLT requires fresh post-command sensors from the observation collector"
                        )
                    client._lifecycle_changed.wait(timeout=min(remaining, 0.01))
        try:
            with client._observation_capture_lock:
                sample = client._capture_observation(generation, buffered_cameras=True)
        except Exception as error:
            with client._lifecycle_lock:
                if client._rollout_generation != generation or not client.running:
                    raise InterruptedError("RLT capture cancelled by reset/stop") from error
            raise
        with client._lifecycle_lock:
            if client._rollout_generation != generation or not client.running:
                raise InterruptedError("RLT captured observation invalidated by reset/stop")
        if sample.age(time.monotonic()) > client.config.observation_max_age_s:
            raise TimeoutError("RLT observation is stale")
        if after is not None and (
            sample.started_at < after
            or (sample.oldest_camera_at is not None and sample.oldest_camera_at < after)
        ):
            raise TimeoutError("RLT reward observation predates the completed motor command")
        raw = sample.observation
        raw["task"] = client.task
        return raw, sample

    def start(self):
        from .robot_client_deployment import RolloutState

        client, cfg = self.client, self.client.config
        with client._lifecycle_lock:
            if client._rollout_state is not RolloutState.READY or client._startup_in_progress.is_set():
                return False
            client._startup_in_progress.set()
            client._rollout_state = RolloutState.STARTING
        try:
            client.stub.Ready(services_pb2.Empty(), timeout=cfg.request_timeout_s)
            setup = RLTSetup(
                self.session,
                client.policy_config,
                tuple(client.robot.action_features),
                cfg.fps,
                cfg.rlt.execution_horizon,
                cfg.rlt.prefix_steps,
                client_image_resize=cfg.resize_images_on_client,
                image_transport_mode=cfg.image_transport_mode,
            )
            _, call = client.stub.SendPolicyInstructions.with_call(
                services_pb2.PolicySetup(data=pickle.dumps(setup)),
                timeout=cfg.startup_calibration_timeout_s,
            )
            client._image_transport_plan = parse_image_transport_metadata(
                call.initial_metadata(),
                expected_mode=cfg.image_transport_mode,
            )
            raw, _ = self._capture(client.rollout_generation)
            pending = None
            measured = []
            for index in range(2 + max(1, cfg.startup_calibration_runs)):
                if not client.running:
                    raise InterruptedError("RLT startup cancelled")
                request = RLTRequest(self.session, 0, index, 0, raw, pending, dry_run=True)
                started = time.monotonic()
                plan, _ = self.scheduler._exchange(request)
                if index >= 2:
                    measured.append(time.monotonic() - started)
                pending = plan.actions[
                    cfg.rlt.execution_horizon : cfg.rlt.execution_horizon + cfg.rlt.prefix_steps
                ].clone()
            budget = cfg.rlt.prefix_steps / cfg.fps - cfg.rlt.budget_margin_s
            if max(measured) >= budget:
                raise TimeoutError(
                    f"RLT measured RTT {max(measured):.3f}s exceeds prefix budget {budget:.3f}s; "
                    "increase prefix_steps below execution_horizon, reduce fps, or accelerate inference"
                )
            with client._lifecycle_lock:
                if client._rollout_state is not RolloutState.STARTING or not client.running:
                    raise InterruptedError("RLT startup cancelled")
                client._rollout_state = RolloutState.READY
            client.logger.info(
                "RLT ready; max dry RTT %.3fs, committed-prefix budget %.3fs; no dry-run replay",
                max(measured),
                budget,
            )
            return True
        except Exception as exc:
            client._fail_worker("RLT startup", exc)
            return False
        finally:
            client._startup_in_progress.clear()

    def run(self):
        from .robot_client_deployment import RolloutState

        client = self.client
        client._start_observation_collector()
        try:
            while client.running:
                with client._lifecycle_changed:
                    client._lifecycle_changed.wait_for(
                        lambda: not client.running or client._rollout_state is RolloutState.RUNNING,
                        timeout=0.1,
                    )
                    if not client.running:
                        break
                    if client._rollout_state is not RolloutState.RUNNING:
                        continue
                    generation = client._rollout_generation
                    cancelled = self.cancelled
                try:
                    raw, sample = self._capture(generation)
                    source_at = min(
                        sample.started_at,
                        sample.oldest_camera_at if sample.oldest_camera_at is not None else sample.started_at,
                    )
                    cold_deadline = source_at + client.config.observation_max_age_s
                    first_step = True
                    with client._lifecycle_lock:
                        if cancelled.is_set() or client._rollout_generation != generation:
                            raise InterruptedError("RLT initial observation invalidated by reset")
                        client.action_interpolator.reset()
                        initial = torch.tensor([raw[name] for name in client.robot.action_features])
                        if not torch.isfinite(initial).all():
                            raise ValueError("Non-finite initial robot pose")
                        client.action_interpolator.add(initial)
                        while client.action_interpolator.get() is not None:
                            pass

                    def execute(
                        action, generation=generation, cancelled=cancelled, cold_deadline=cold_deadline
                    ):
                        nonlocal first_step
                        deadline = cold_deadline if first_step else None
                        first_step = False
                        return self._step(action, generation, cancelled, source_deadline=deadline)

                    self.scheduler.run(
                        raw,
                        execute,
                        generation=generation,
                        cancelled=cancelled,
                        observation_started_at=source_at,
                    )
                except InterruptedError:
                    continue
                with client._lifecycle_changed:
                    if (
                        client._rollout_generation == generation
                        and client._rollout_state is RolloutState.RUNNING
                    ):
                        client._rollout_state = RolloutState.READY
                        client._lifecycle_changed.notify_all()
        except Exception as exc:
            client._fail_worker("RLT execution", exc)

    def _step(self, action, generation, cancelled, *, source_deadline=None):
        from lerobot.rl.rlt.rollout import ExecutedStep

        from .robot_client_deployment import RolloutState

        client = self.client
        started = time.monotonic()
        with client._lifecycle_lock:
            if cancelled.is_set() or client._rollout_generation != generation:
                raise InterruptedError("RLT interpolation invalidated by reset")
            client.action_interpolator.add(action)
        multiplier = client.action_interpolator.multiplier
        performed = None
        for index in range(multiplier):
            # Match deployment's motor -> lifecycle lock order. Reset can fence
            # a generation while this thread is waiting for motor ownership.
            with client._robot_command_lock, client._lifecycle_lock:
                if (
                    cancelled.is_set()
                    or not client.running
                    or client._rollout_generation != generation
                    or client._rollout_state is not RolloutState.RUNNING
                ):
                    raise InterruptedError("RLT command cancelled before motor dispatch")
                if time.monotonic() - started >= client.config.observation_max_age_s:
                    raise TimeoutError("RLT command expired while waiting for motor dispatch")
                if index == 0 and source_deadline is not None and time.monotonic() >= source_deadline:
                    raise TimeoutError("RLT cold observation expired before motor dispatch")
                target = client.action_interpolator.get()
                if target is None:
                    raise RuntimeError("Incomplete RLT interpolation interval")
                performed = client.robot.send_action(client._action_tensor_to_action_dict(target))
                sent_at = time.monotonic()
            deadline = started + (index + 1) / (client.config.fps * multiplier)
            if cancelled.wait(max(0.0, deadline - time.monotonic())):
                raise InterruptedError("RLT policy interval cancelled")
        if not isinstance(performed, dict):
            raise ValueError("Robot must report its actual sent action for RLT replay")
        actual = torch.tensor([performed[name] for name in client.robot.action_features], dtype=torch.float32)
        if not torch.isfinite(actual).all():
            raise ValueError("Robot reported non-finite RLT commands")
        raw, _ = self._capture(generation, after=sent_at)
        with client._lifecycle_lock:
            if cancelled.is_set() or client._rollout_generation != generation or not client.running:
                raise InterruptedError("RLT observation invalidated by reset")
        signal = self.reward(raw, actual.clone())
        if not isinstance(signal, RewardSignal):
            raise TypeError("RLT reward callback must return RewardSignal")
        with client._lifecycle_lock:
            if cancelled.is_set() or client._rollout_generation != generation or not client.running:
                raise InterruptedError("RLT task reward invalidated by reset")
            with client.latest_action_lock:
                client.latest_action += 1
        return ExecutedStep(
            raw, actual, signal.reward, signal.terminated, signal.truncated, signal.intervention
        )
