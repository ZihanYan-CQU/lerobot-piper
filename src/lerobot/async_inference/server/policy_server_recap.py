# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""DAgger-compatible RTC server with server-verified checkpoint identity.

The identity is returned in gRPC trailing metadata; the existing wire protocol
and DAgger entry points are unchanged. Hashing happens once per policy setup,
before rollout control starts. Keep checkpoint directories immutable while used.
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import draccus
from huggingface_hub import snapshot_download

from ..shared.helpers import RemotePolicyConfig, get_logger
from ..shared.metadata_recap import IDENTITY_METADATA_KEY
from .policy_server_dagger import PolicyServerDagger, PolicyServerDaggerConfig
from .policy_server_deployment import _serve_policy_server

_RUNTIME_SUFFIXES = {".safetensors", ".json", ".model", ".txt", ".tiktoken"}


def checkpoint_files(root: Path) -> dict[Path, tuple[int, int, int, int]]:
    """Stat the weights, configs, processors and tokenizer assets in this tree."""
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in _RUNTIME_SUFFIXES:
            stat = path.stat()
            files[path.relative_to(root)] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"RECAP checkpoint has no safetensors weights: {root}")
    return files


def checkpoint_digest(root: Path, files: dict[Path, tuple[int, int, int, int]]) -> str:
    """Hash names and file contents, independent of checkpoint location/mtime."""
    digest = hashlib.sha256()
    for relative_path in files:
        file_digest = hashlib.sha256()
        with (root / relative_path).open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                file_digest.update(block)
        digest.update(relative_path.as_posix().encode("utf-8") + b"\0" + file_digest.digest())
    return digest.hexdigest()


@dataclass
class PolicyServerRecapConfig(PolicyServerDaggerConfig):
    recap_policy_version: str | None = None  # Optional human-readable training/run label.
    recap_model_revision: str | None = None  # Hub branch/tag/commit, resolved once to a snapshot.

    def __post_init__(self):
        super().__post_init__()
        if self.reuse_last_policy:
            raise ValueError("RECAP requires reuse_last_policy=false to fingerprint freshly loaded weights")


class PolicyServerRecap(PolicyServerDagger):
    prefix = "policy_server_recap"
    logger = get_logger(prefix)

    def _load_or_reuse_policy(self, policy_specs: RemotePolicyConfig) -> bool:
        # Do not retain an earlier identity if loading this checkpoint fails.
        self.recap_policy_identity = None
        requested = str(policy_specs.pretrained_name_or_path)
        root = Path(requested).expanduser()
        revision = None
        if not root.is_dir():
            root = Path(
                snapshot_download(
                    requested,
                    revision=self.config.recap_model_revision,
                    allow_patterns=[f"*{suffix}" for suffix in sorted(_RUNTIME_SUFFIXES)],
                )
            )
            revision = root.name
        root = root.resolve()
        self.logger.info("Fingerprinting RECAP checkpoint %s before loading", root)
        files = checkpoint_files(root)
        digest = checkpoint_digest(root, files)
        reused = super()._load_or_reuse_policy(replace(policy_specs, pretrained_name_or_path=str(root)))
        if files != checkpoint_files(root):
            self._drop_cached_policy_runtime()
            raise RuntimeError("Checkpoint changed during RECAP setup; use an immutable checkpoint directory")
        self.recap_policy_identity = {
            "schema_version": 1,
            "model_version": f"sha256:{digest}",
            "identity_source": "server_checkpoint_files",
            "hash_scope": "relative paths + contents of safetensors/json/model/txt/tiktoken files",
            "policy_type": policy_specs.policy_type,
            "requested_path": requested,
            "resolved_path": str(root),
            "hub_commit": revision,
            "label": self.config.recap_policy_version,
            "server_load_id": str(uuid.uuid4()),
            "loaded_at_utc": datetime.now(UTC).isoformat(),
            "rtc_mode": self.config.rtc.mode,
            "fps": self.config.fps,
            "server_config": json.loads(json.dumps(asdict(self.config), default=str)),
        }
        return reused

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        response = super().SendPolicyInstructions(request, context)
        identity = getattr(self, "recap_policy_identity", None)
        if not self.running or identity is None:
            raise RuntimeError("RECAP policy setup did not establish a checkpoint identity")
        context.set_trailing_metadata(((IDENTITY_METADATA_KEY, json.dumps(identity).encode("utf-8")),))
        return response


PolicyServer = PolicyServerRecap


@draccus.wrap()
def serve(cfg: PolicyServerRecapConfig) -> None:
    _serve_policy_server(PolicyServerRecap(cfg))


if __name__ == "__main__":
    serve()
