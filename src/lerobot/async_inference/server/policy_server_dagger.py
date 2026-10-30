# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical DAgger policy-server entry point.

Human intervention and recording remain client-side.  This module extends
only ``policy_server_deployment`` and imports no historical server version.
Every rollout boundary is conveyed by the generation embedded in
``observation_id``.

The only DAgger-specific default is ``reuse_last_policy=False``.  Iterative
DAgger commonly overwrites a checkpoint at the same path between client
connections; disabling cross-connection model reuse avoids silently serving
old weights.  The model remains warm across any number of intervention
generations within one connected session.
"""

from dataclasses import dataclass, field

import draccus

from ..shared.helpers import get_logger
from .policy_server_deployment import (
    PolicyServerDeployment,
    PolicyServerDeploymentConfig,
    _serve_policy_server,
)


@dataclass
class PolicyServerDaggerConfig(PolicyServerDeploymentConfig):
    """Resettable RTC server with a checkpoint-fresh DAgger default."""

    reuse_last_policy: bool = field(
        default=False,
        metadata={
            "help": (
                "Reuse a matching policy across client reconnects. Disabled by default for "
                "iterative DAgger because the same checkpoint path may contain newer weights."
            )
        },
    )


class PolicyServerDagger(PolicyServerDeployment):
    prefix = "policy_server_dagger"
    logger = get_logger(prefix)


PolicyServer = PolicyServerDagger


@draccus.wrap()
def serve(cfg: PolicyServerDaggerConfig) -> None:
    _serve_policy_server(PolicyServerDagger(cfg))


if __name__ == "__main__":
    serve()
