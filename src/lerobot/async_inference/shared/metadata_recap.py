# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

"""Small shared RECAP metadata helpers, independent of policy/server imports."""

import json
import os
import re
import uuid
from pathlib import Path

IDENTITY_METADATA_KEY = "recap-policy-identity-bin"


def validate_policy_identity(identity: dict) -> dict:
    if (
        identity.get("schema_version") != 1
        or identity.get("identity_source") != "server_checkpoint_files"
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity.get("model_version", ""))
        or not identity.get("server_load_id")
        or not identity.get("policy_type")
    ):
        raise ValueError("Missing or invalid server-verified RECAP model identity")
    return identity


def atomic_write_json(path: Path, value: dict) -> None:
    """Publish one complete sidecar, retaining a previous version on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
