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

Imported ONLY by the verifier resources server; agents POST a patch to ``/verify``.
Runs the plan §4 fresh-only sequence via ``swe_env.lifecycle.acquire_sandbox``
(durable registry + create-admission + always-teardown), bounded by a per-call
eval timeout, with content-key idempotency so a retried/duplicated ``/verify``
(ServerClient retries are unbounded — plan §9) coalesces instead of spawning a
second fresh sandbox.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import tempfile
from collections.abc import Mapping
from typing import Any

# Importing these packages registers the swe_env providers + harnesses.
import responses_api_agents.swe_env.harnesses  # noqa: F401
import responses_api_agents.swe_env.providers  # noqa: F401
from nemo_gym.sandbox import SandboxProvider
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import SweEvalReport, SweTask
from responses_api_agents.swe_env.lifecycle import (
    CreateAdmission,
    SandboxRegistry,
    acquire_sandbox,
    content_key,
)
from responses_api_agents.swe_env.registry import get_harness


class ProviderCapabilityError(RuntimeError):
    """Raised when a task's harness does not support the configured provider."""


_DEFAULT_REGISTRY_ROOT = os.environ.get(
    "SWE_ENV_REGISTRY_ROOT", os.path.join(tempfile.gettempdir(), "swe_env_registry")
)
_DEFAULT_MAX_CREATES = int(os.environ.get("SWE_ENV_MAX_CONCURRENT_CREATES", "16"))

# Process-wide lifecycle state (verifier pinned to one worker — plan §9).
_registry = SandboxRegistry(_DEFAULT_REGISTRY_ROOT)
_admission = CreateAdmission(_DEFAULT_MAX_CREATES)
_idempotency: dict[str, asyncio.Future] = {}
_IDEMPOTENCY_CAP = 4096


def get_registry() -> SandboxRegistry:
    return _registry


def clear_idempotency_cache() -> None:
    _idempotency.clear()


def _provider_name(provider: Mapping[str, Any] | SandboxProvider) -> str:
    if isinstance(provider, Mapping):
        return next(iter(provider), "?")
    return getattr(provider, "name", "?")


async def verify_task(
    provider: Mapping[str, Any] | SandboxProvider,
    task: SweTask,
    *,
    run_golden: bool = False,
    registry: SandboxRegistry | None = None,
    admission: CreateAdmission | None = None,
    idempotent: bool = True,
    eval_timeout_s: float | None = None,
) -> SweEvalReport:
    """Grade ``task``'s patch in a fresh sandbox; return a (reward-ready) report."""
    harness = get_harness(task.benchmark)

    if run_golden:
        task = dataclasses.replace(task, model_patch=task.metadata.get("golden_patch", ""))

    # Empty/falsy-patch fast path: no eval spin-up (ports app.py:1517-1524).
    if not (task.model_patch or "").strip():
        return SweEvalReport(instance_id=task.instance_id, patch_exists=False, resolved=False)

    provider_name = _provider_name(provider)
    if not harness.supports_provider(provider_name):
        raise ProviderCapabilityError(
            f"Harness {harness.name!r} does not support provider {provider_name!r} "
            f"(grade_strategy={harness.grade_strategy})"
        )

    key = content_key(
        instance_id=task.instance_id, patch=task.model_patch, harness=task.benchmark, run_golden=run_golden
    )

    fut: asyncio.Future | None = None
    if idempotent:
        running = asyncio.get_running_loop()
        existing = _idempotency.get(key)
        # Only coalesce within the same loop (tests use a fresh loop per asyncio.run()).
        if existing is not None and existing.get_loop() is running:
            try:
                return await existing
            except Exception:
                pass  # prior attempt failed: fall through and retry
        if len(_idempotency) > _IDEMPOTENCY_CAP:
            _idempotency.clear()
        fut = running.create_future()
        _idempotency[key] = fut

    try:
        report = await _run_verify(provider, task, harness, key, registry, admission, eval_timeout_s)
        if fut is not None and not fut.done():
            fut.set_result(report)
        return report
    except Exception as exc:
        if fut is not None:
            if not fut.done():
                fut.set_exception(exc)
            _idempotency.pop(key, None)  # don't cache failures — allow a clean retry
        raise


async def _run_verify(
    provider: Mapping[str, Any] | SandboxProvider,
    task: SweTask,
    harness: Any,
    key: str,
    registry: SandboxRegistry | None,
    admission: CreateAdmission | None,
    eval_timeout_s: float | None,
) -> SweEvalReport:
    reg = registry if registry is not None else _registry
    adm = admission if admission is not None else _admission
    spec = harness.build_spec(task)
    timeout = eval_timeout_s if eval_timeout_s is not None else float(task.metadata.get("eval_timeout_s", 1800))
    try:
        async with acquire_sandbox(
            provider, spec, registry=reg, admission=adm, instance_id=task.instance_id, key=key
        ) as env:

            async def _sequence() -> SweEvalReport:
                await harness.reset_repo(env, task)
                await harness.materialize(env, task)
                artifacts = await harness.run_eval(env, task)
                return harness.grade(task, artifacts)

            return await asyncio.wait_for(_sequence(), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError):
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            error_kind="eval_timeout",
            tests_status={"timeout_s": timeout},
        )
    except Exception as exc:  # infra failure -> mask via flag, never crash the server
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            error_kind="sandbox",
            tests_status={"exception": repr(exc)},
        )


def report_to_reward(report: SweEvalReport) -> float:
    return reward_from_report(report)
