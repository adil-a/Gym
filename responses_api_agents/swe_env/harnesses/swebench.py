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

"""swe-bench / swe-bench-multilingual harness — host-side (flat) grading.

A single parametrized class serves both families. It runs the instance's official SWE-bench
eval script (``swebench.make_test_spec(...).eval_script``) inside the sandbox and grades the
produced log host-side with swebench's per-repo log parser, so it runs on any exec-capable
provider (docker / opensandbox).

NOTE: the apptainer-only nested ``run_local_evaluation`` path was removed when PR #1694 took
ownership of the apptainer provider. The swe_env-specific nested-apptainer grading (mounts/.sif
wiring + run_local_evaluation) is tracked for a follow-up PR (see APPTAINER_PR3_TRACKER.md).
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from typing import TYPE_CHECKING

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.grading import compute_resolved
from responses_api_agents.swe_env.harness import (
    EvalArtifacts,
    SweEvalReport,
    SweTask,
    SweTaskHarness,
    _ensure_trailing_newline,
)
from responses_api_agents.swe_env.harnesses import flat_eval


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


# Per-test status tokens swebench's repo parsers emit that count as a pass.
_SWEBENCH_PASS_STATUSES = frozenset({"PASSED", "XFAIL"})

# swe-bench families this harness serves.
_VALID_NAMES = frozenset({"swe-bench", "swe-bench-multilingual"})


class SweBenchHarness(SweTaskHarness):
    """SWE-bench (and multilingual) harness, host-side (flat) graded.

    Runs the instance's official eval script in the sandbox and parses the log host-side with
    swebench's per-repo parser. Construct one instance per family
    (``SweBenchHarness("swe-bench")`` / ``SweBenchHarness("swe-bench-multilingual")``).
    """

    grade_strategy = "flat-host-grade"

    def __init__(self, name: str = "swe-bench") -> None:
        """Initialize the harness for a given swe-bench family.

        Args:
            name: The swe-bench family to serve (``"swe-bench"`` or ``"swe-bench-multilingual"``).

        Raises:
            ValueError: If ``name`` is not a known swe-bench family.
        """
        if name not in _VALID_NAMES:
            raise ValueError(f"Unknown swe-bench family: {name!r} (expected one of {sorted(_VALID_NAMES)})")
        self.name = name

    # --- provisioning --------------------------------------------------------

    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for a task.

        Args:
            task: The task to provision a sandbox for.

        Returns:
            A ``SandboxSpec`` describing the image, workdir, environment, and any provider
            options carried on the task. Flat grading runs the eval script directly in the
            instance image, so no host harness/venv mounts are needed.
        """
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
            provider_options=dict(task.metadata.get("provider_options", {})),
        )

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Write the bare ``/root/patch.diff`` the eval script applies.

        Args:
            env: The environment used to write files into the sandbox.
            task: The task whose model patch is staged for the eval script (newline-normalized
                so the upstream ``git apply`` succeeds).
        """
        if task.model_patch:
            await env.write_text("/root/patch.diff", _ensure_trailing_newline(task.model_patch))

    def _flat_eval_script(self, task: SweTask) -> str:
        """Build the official SWE-bench eval script for host-side (flat) grading.

        Uses the ``swebench`` library's ``make_test_spec(...).eval_script`` (the per-repo recipe),
        prefixed with a step that applies the model patch from ``/root/patch.diff``. Returns an
        empty string if the instance dict is unavailable or the spec cannot be built, in which
        case the flat grader masks the sample as an eval error rather than scoring 0.

        Args:
            task: The task whose ``metadata['instance_dict']`` describes the SWE-bench instance.

        Returns:
            The eval-script text, or ``""`` when it cannot be constructed.
        """
        instance = task.metadata.get("instance_dict")
        if not instance:
            return ""
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec

            spec = make_test_spec(instance, namespace="swebench")
        except Exception:
            return ""
        apply_model = (
            "cd /testbed && "
            "(git apply -v /root/patch.diff || git apply -v --3way /root/patch.diff || "
            "echo 'NEMO_GYM_PATCH_APPLY_FAILED')\n"
        )
        return apply_model + spec.eval_script

    # --- server-private grading ----------------------------------------------

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Run the instance's eval script in-sandbox and collect its log.

        Args:
            env: The environment used to execute commands in the sandbox.
            task: The task to evaluate.

        Returns:
            An ``EvalArtifacts`` carrying the captured test output, return code, whether a patch
            existed, and the flat-eval markers.
        """
        if not task.metadata.get("eval_script"):
            task = dataclasses.replace(
                task, metadata={**task.metadata, "eval_script": self._flat_eval_script(task)}
            )
        return await flat_eval.flat_run_eval(env, task)

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Grade a task from its evaluation artifacts (host-side, flat).

        The SWE-bench family spans repos with different test runners (pytest, django's unittest
        runner, etc.). The generic flat parser is pytest-only and silently scores non-pytest
        repos (e.g. django) unresolved — even the gold patch. Grade with swebench's official
        per-repo log parser when importable; fall back to the generic parser only when swebench
        is absent.

        Args:
            task: The task being graded.
            artifacts: The evaluation artifacts produced by ``run_eval``.

        Returns:
            A ``SweEvalReport`` recording resolution, patch state, and any error kind.
        """
        report = self._swebench_flat_grade(task, artifacts)
        return report if report is not None else flat_eval.flat_grade(task, artifacts)

    def _swebench_flat_grade(self, task: SweTask, artifacts: EvalArtifacts) -> "SweEvalReport | None":
        """Grade a flat eval log with swebench's official per-repo log parser.

        The generic :func:`flat_eval.flat_grade` parser only recognises pytest-style
        ``PASSED <node_id>`` lines, so repos with other test runners (e.g. django's unittest
        runner) parse as zero passing tests and grade unresolved — even for the gold patch.
        This path uses ``swebench.harness.grading.get_logs_eval`` (the same per-repo parser the
        nested harness uses), keeping docker flat grading faithful to the official result.

        Args:
            task: The task being graded (supplies the instance dict + fail/pass test ids).
            artifacts: The artifacts produced by :func:`flat_eval.flat_run_eval`.

        Returns:
            A ``SweEvalReport`` with the official verdict, or ``None`` when swebench is
            unavailable / the spec cannot be built (caller falls back to the generic parser).
        """
        # Mirror flat_grade's infra masks so a sandbox/timeout/eval_error never scores 0.
        error_type = artifacts.raw.get("error_type")
        if error_type in {"sandbox", "timeout", "eval_error"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=error_type,
            )
        instance = task.metadata.get("instance_dict")
        if not instance:
            return None
        try:
            from swebench.harness.grading import get_logs_eval
            from swebench.harness.test_spec.test_spec import make_test_spec
        except Exception:
            return None  # swebench absent -> caller falls back to the generic parser
        log_fp = None
        try:
            spec = make_test_spec(instance, namespace="swebench")
            with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
                handle.write(artifacts.test_output or "")
                log_fp = handle.name
            status_map, markers_found = get_logs_eval(spec, log_fp)
        except Exception:
            return None
        finally:
            if log_fp is not None and os.path.exists(log_fp):
                os.unlink(log_fp)
        passed = [node for node, status in status_map.items() if status in _SWEBENCH_PASS_STATUSES]
        resolved = bool(markers_found) and compute_resolved(
            fail_to_pass=task.fail_to_pass,
            pass_to_pass=task.pass_to_pass,
            passed=passed,
        )
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=bool(markers_found),
            patch_exists=bool(task.model_patch),
            tests_status={"passed": passed, "all": status_map},
        )
