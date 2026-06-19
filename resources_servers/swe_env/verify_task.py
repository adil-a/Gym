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

"""Server-private verification orchestrator (the sole verification entry point).

This module is imported ONLY by the verifier resources server. Agents never
import it — they POST a patch to ``/verify``. It runs the deterministic
fresh-sandbox sequence from the plan §4:

    acquire fresh sandbox -> reset_repo -> materialize -> run_eval -> grade -> teardown

Sandbox sourcing is fresh-only: every call starts and stops its own sandbox.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

# Importing these packages registers the swe_env providers + harnesses.
import responses_api_agents.swe_env.harnesses  # noqa: F401
import responses_api_agents.swe_env.providers  # noqa: F401
from nemo_gym.sandbox import SandboxProvider
from responses_api_agents.swe_env.environment import AsyncSweEnvironment
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import SweEvalReport, SweTask
from responses_api_agents.swe_env.registry import get_harness


class ProviderCapabilityError(RuntimeError):
    """Raised when a task's harness does not support the configured provider."""


def _provider_name(provider: Mapping[str, Any] | SandboxProvider) -> str:
    if isinstance(provider, Mapping):
        return next(iter(provider), "?")
    return getattr(provider, "name", "?")


async def verify_task(
    provider: Mapping[str, Any] | SandboxProvider,
    task: SweTask,
    *,
    run_golden: bool = False,
) -> SweEvalReport:
    """Grade ``task``'s patch in a fresh sandbox; return a report (reward-ready)."""
    harness = get_harness(task.benchmark)

    if run_golden:
        golden = task.metadata.get("golden_patch", "")
        task = dataclasses.replace(task, model_patch=golden)

    # Empty/falsy-patch fast path: no eval spin-up (ports app.py:1517-1524).
    if not (task.model_patch or "").strip():
        return SweEvalReport(instance_id=task.instance_id, patch_exists=False, resolved=False)

    provider_name = _provider_name(provider)
    if not harness.supports_provider(provider_name):
        raise ProviderCapabilityError(
            f"Harness {harness.name!r} does not support provider {provider_name!r} "
            f"(grade_strategy={harness.grade_strategy})"
        )

    spec = harness.build_spec(task)
    env: AsyncSweEnvironment | None = None
    try:
        env = await AsyncSweEnvironment.start(provider, spec)
        await harness.reset_repo(env, task)
        await harness.materialize(env, task)
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)
    except ProviderCapabilityError:
        raise
    except Exception as exc:  # infra failure -> mask via flag, never crash the server
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            error_kind="sandbox",
            tests_status={"exception": repr(exc)},
        )
    finally:
        if env is not None:
            try:
                await env.cleanup()
            except Exception:
                pass


def report_to_reward(report: SweEvalReport) -> float:
    """Convenience wrapper around :func:`reward_from_report`."""
    return reward_from_report(report)
