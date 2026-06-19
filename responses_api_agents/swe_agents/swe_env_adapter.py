# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""OpenHands ``swe_agents`` adapter onto the decoupled ``swe_env`` infra (#1249).

This is the SELF_DRIVING migration path for the legacy OpenHands harness (plan §6):
instead of ``_build_apptainer_command`` + the two-container ``/trajectories_mount``
handshake, it provisions the agent's working container via ``swe_env.lifecycle``,
injects a sandbox-reachable model endpoint (egress, §6), lets the agent self-drive
inside that container, extracts the unified-diff patch, then scores it through the
**verifier** in its own fresh sandbox (§4a) — i.e. environment + verification are
fully decoupled from the agent loop.

Additive on purpose: the legacy ``SWEBenchWrapper.run()`` is left intact so the
existing (mocked) test suite stays green; flipping ``run()`` to call this — and
deleting the legacy in-worker eval after a dual-run reward-parity window — is the
final cutover step, gated on apptainer/OpenHands validation (SWE_ENV_DECOUPLE_STATUS.md).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from nemo_gym.sandbox import SandboxProvider
from resources_servers.swe_env.verify_task import verify_task
from responses_api_agents.swe_env import get_harness, model_endpoint, reward_from_report
from responses_api_agents.swe_env.harness import SweTask
from responses_api_agents.swe_env.lifecycle import CreateAdmission, SandboxRegistry, acquire_sandbox


def _provider_name(provider: Mapping[str, Any] | SandboxProvider) -> str:
    if isinstance(provider, Mapping):
        return next(iter(provider), "?")
    return getattr(provider, "name", "?")


async def run_self_driving(
    task: SweTask,
    *,
    provider: Mapping[str, Any] | SandboxProvider,
    agent_launch_command: str,
    model_server: Mapping[str, Any] | None = None,
    opensandbox_service_url: str | None = None,
    agent_timeout_s: int | float = 1800,
    registry: SandboxRegistry | None = None,
    admission: CreateAdmission | None = None,
) -> dict[str, Any]:
    """Run a SELF_DRIVING agent through swe_env, then score via the verifier.

    Returns the agent-owned result fields (the legacy ``run()`` would merge these
    into the frozen ``SWEBenchVerifyResponse`` it emits — §4a wire-ownership).
    """
    harness = get_harness(task.benchmark)
    spec = harness.build_spec(task)

    # Model-server egress: inject only a sandbox-reachable endpoint (never the global dict).
    if model_server is not None:
        endpoint = model_endpoint.resolve(
            _provider_name(provider), model_server, opensandbox_service_url=opensandbox_service_url
        )
        spec = dataclasses.replace(spec, env={**spec.env, **endpoint.to_sandbox_env()})

    # 1. Provision the agent's OWN working container and let it self-drive (one long exec).
    async with acquire_sandbox(
        provider, spec, registry=registry, admission=admission, instance_id=task.instance_id
    ) as env:
        await env.execute(agent_launch_command, cwd=task.repo_workdir, timeout_s=agent_timeout_s)
        diff = await env.execute(f"cd {task.repo_workdir} && git add -A && git diff --cached", cwd=task.repo_workdir)
        patch = diff.get("stdout", "") or ""

    # 2. Score the patch in the verifier's OWN fresh sandbox (decoupled verification).
    report = await verify_task(provider, dataclasses.replace(task, model_patch=patch))
    masked = report.error_kind is not None
    return {
        "instance_id": task.instance_id,
        "model_patch": patch,
        "resolved": report.resolved,
        "reward": reward_from_report(report),
        "patch_exists": bool(patch.strip()),
        "mask_sample": masked,
        "error_kind": report.error_kind,
    }
