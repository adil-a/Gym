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

"""SWE environment verifier — the required, sole verification entry point (#1249).

A ``SimpleResourcesServer`` whose ``verify()`` extracts the agent's patch from
the response, builds a ``SweTask`` from the per-task metadata, grades it in its
**own fresh, stateless sandbox** via the server-private ``verify_task``
orchestrator, and returns the eval-side fields + reward.

It is NOT a host-agnostic exec-only server: for the apptainer provider it must
co-locate with ``.sif``/Lustre (and Docker for nested families). The reward is a
non-nullable float; masking is carried as ``reward=0.0`` + ``mask_sample``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.swe_env.verify_task import verify_task
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import SweTask


_FENCED_DIFF = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                return [str(v) for v in json.loads(stripped)]
            except json.JSONDecodeError:
                pass
        return [stripped] if stripped else []
    return [str(value)]


class SweEnvVerifierConfig(BaseResourcesServerConfig):
    """Verifier config. ``sandbox_provider`` is a single-key provider mapping."""

    sandbox_provider: dict[str, Any] = {"docker": {}}
    model_patch_field: str = "model_patch"


class SweEnvVerifyResponse(BaseVerifyResponse):
    resolved: bool = False
    patch_exists: bool = False
    patch_applied: bool = False
    eval_error: bool = False
    error_kind: str | None = None
    mask_sample: bool = False
    instance_id: str = ""


class SweEnvVerifier(SimpleResourcesServer):
    config: SweEnvVerifierConfig

    async def verify(self, body: BaseVerifyRequest) -> SweEnvVerifyResponse:
        task = build_task(body, self.config.model_patch_field)
        report = await verify_task(self.config.sandbox_provider, task)
        reward = reward_from_report(report)
        masked = report.error_kind is not None
        return SweEnvVerifyResponse(
            **body.model_dump(),
            reward=reward,
            resolved=report.resolved,
            patch_exists=report.patch_exists,
            patch_applied=report.patch_applied,
            eval_error=masked,
            error_kind=report.error_kind,
            mask_sample=masked,
            instance_id=report.instance_id,
        )


def build_task(body: BaseVerifyRequest, patch_field: str) -> SweTask:
    """Map a verify request (per-task metadata + agent response) onto a SweTask.

    Module-level (not a method) so it is unit-testable without instantiating the
    Pydantic server.
    """
    metadata: dict[str, Any] = dict(body.responses_create_params.metadata or {})
    patch = extract_patch(body.response, metadata, patch_field)
    return SweTask(
        instance_id=str(metadata.get("instance_id", "unknown")),
        image=metadata.get("image"),
        base_commit=metadata.get("base_commit"),
        repo_workdir=str(metadata.get("repo_workdir", "/testbed")),
        test_command=str(metadata.get("test_command", "")),
        test_framework=str(metadata.get("test_framework", "")),
        model_patch=patch,
        test_patch=str(metadata.get("test_patch", "")),
        fail_to_pass=_as_list(metadata.get("fail_to_pass")),
        pass_to_pass=_as_list(metadata.get("pass_to_pass")),
        benchmark=str(metadata.get("benchmark", "swe-bench-ext")),
        split=str(metadata.get("split", "test")),
        metadata=metadata,
    )


def extract_patch(response: NeMoGymResponse, metadata: dict[str, Any], patch_field: str) -> str:
    """Read the normalized patch field; fall back to a fenced diff in output text."""
    response_metadata = getattr(response, "metadata", None) or {}
    patch = response_metadata.get(patch_field)
    if patch:
        return str(patch)
    for item in getattr(response, "output", []) or []:
        text = _item_text(item)
        if text:
            match = _FENCED_DIFF.search(text)
            if match:
                return match.group(1)
    return ""


def _item_text(item: Any) -> str:
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return ""


if __name__ == "__main__":
    SweEnvVerifier.run_webserver()
