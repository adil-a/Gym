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

"""swe-bench-ext harness: flat, host-graded reference family.

Generalizes ``SweBenchExtDatasetProcessor`` (swe_agents/app.py:903-1061): reset
to base, apply the model patch (+ test patch), run the framework test command,
and grade host-side by parsing per-test pass/fail.

The full vendored ``swe_bench_ext`` parser (1606 lines) relocation is deferred;
this harness ships a focused pytest/unittest status parser sufficient for the
reference path. See SWE_ENV_DECOUPLE_STATUS.md.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.grading import compute_resolved
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask, SweTaskHarness


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


# Matches pytest "-rA" summary lines in either order:
#   "PASSED tests/test_x.py::test_a"  or  "tests/test_x.py::test_a PASSED"
_STATUS_LEADING = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.MULTILINE)
_STATUS_TRAILING = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR)\b", re.MULTILINE)


def parse_test_statuses(output: str) -> dict[str, str]:
    """Parse a {node_id: STATUS} map from pytest-style output (both orders)."""
    statuses: dict[str, str] = {}
    for match in _STATUS_LEADING.finditer(output):
        statuses[match.group(2)] = match.group(1)
    for match in _STATUS_TRAILING.finditer(output):
        statuses.setdefault(match.group(1), match.group(2))
    return statuses


class SweBenchExtHarness(SweTaskHarness):
    name = "swe-bench-ext"
    grade_strategy = "flat-host-grade"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        return SandboxSpec(
            image=task.image,
            workdir=task.repo_workdir,
            ttl_s=task.metadata.get("ttl_s", 1800),
            ready_timeout_s=task.metadata.get("ready_timeout_s", 600),
            env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_PAGER": "cat"},
            metadata={
                "instance_id": task.instance_id[:63],
                "benchmark": task.benchmark,
                "harness": self.name,
            },
            resources=SandboxResources.from_mapping(task.metadata.get("resources", {})),
            provider_options=task.metadata.get("provider_options", {}),
        )

    def supports_provider(self, provider_name: str) -> bool:
        return True  # flat, host-graded: works on any exec-capable provider

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        workdir = task.repo_workdir
        patch_applied = True
        # --recount tolerates wrong @@ hunk counts (common in model-generated diffs);
        # mirrors the legacy swe-bench-ext apply (swe_agents/app.py:989).
        apply_flags = "--recount --ignore-whitespace --ignore-space-change --whitespace=nowarn"
        if task.model_patch:
            applied = await env.execute(
                f"git apply -v {apply_flags} /root/patch.diff || git apply -v --3way {apply_flags} /root/patch.diff",
                cwd=workdir,
            )
            patch_applied = applied["returncode"] == 0
        if task.test_patch:
            await env.execute(
                f"git apply -v {apply_flags} /root/test_patch.diff "
                f"|| git apply -v --3way {apply_flags} /root/test_patch.diff",
                cwd=workdir,
            )
        test_command = task.test_command or "python -m pytest -rA -q"
        result = await env.execute(test_command, cwd=workdir, is_eval=True)
        return EvalArtifacts(
            test_output=result["output"],
            return_code=result["returncode"],
            patch_applied=patch_applied,
            raw={"error_type": result.get("error_type")},
        )

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        # Infra failure → mask via error_kind (never scored as "unresolved").
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )
        statuses = parse_test_statuses(artifacts.test_output)
        passed = [node for node, status in statuses.items() if status == "PASSED"]
        resolved = artifacts.patch_applied and compute_resolved(
            fail_to_pass=task.fail_to_pass,
            pass_to_pass=task.pass_to_pass,
            passed=passed,
        )
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status={"passed": passed, "all": statuses},
        )
