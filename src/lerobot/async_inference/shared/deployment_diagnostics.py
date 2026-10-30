# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Persistent fatal-signal diagnostics for standalone deployment servers."""

import faulthandler
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def configure_deployment_logging(*, debug: bool) -> None:
    """Choose the process log level before starting deployment workers.

    The shared logging initializer leaves the root level at NOTSET, which
    otherwise writes camera, transport and timer DEBUG records to disk even
    when the deployment-specific diagnostics flag is disabled.
    """
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)


@contextmanager
def server_fault_diagnostics(logger: logging.Logger, prefix: str, path: Path | None = None) -> Iterator[Path]:
    """Keep fatal traces and flushed server logs in the same file until shutdown.

    Python exception handlers cannot catch SIGSEGV. faulthandler writes directly
    to this open file descriptor, including the stacks of all Python threads.
    """
    path = (path or Path("logs") / f"{prefix}_fault_{os.getpid()}.log").expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    previously_enabled = faulthandler.is_enabled()
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)s pid=%(process)d "
                "thread=%(threadName)s/%(thread)d %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        try:
            faulthandler.enable(file=stream, all_threads=True)
            logger.info("Native crash diagnostics enabled: %s (Python %s)", path, sys.version.split()[0])
            yield path
        finally:
            # Never leave faulthandler pointing at a descriptor that may be
            # closed and reused by another file. Restore an enabled handler to
            # stderr; faulthandler exposes no getter for its previous target.
            faulthandler.disable()
            if previously_enabled:
                faulthandler.enable(file=sys.__stderr__, all_threads=True)
            logger.removeHandler(handler)
            handler.close()


def is_fatal_cuda_error(error: BaseException) -> bool:
    """Identify context-poisoning CUDA failures, including wrapped/cleanup errors."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current).lower()
        if any(
            marker in message
            for marker in (
                "cudaerrorillegaladdress",
                "cudaerrorassert",
                "cudaerrorlaunchfailure",
            )
        ) or (
            "cuda error:" in message
            and any(
                marker in message
                for marker in (
                    "illegal memory access",
                    "device-side assert",
                    "misaligned address",
                    "illegal instruction",
                    "unspecified launch failure",
                )
            )
        ):
            return True
        for previous in (current.__cause__, current.__context__):
            if previous is not None:
                pending.append(previous)
    return False
