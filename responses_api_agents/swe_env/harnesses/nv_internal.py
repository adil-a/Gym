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

"""nv-internal-1 harness: flat, host-graded NVIDIA-internal family.

Ports ``NVInternalDatasetProcessor`` + ``check_tests_passed`` (swe_agents/app.py
:539-686, :670). Unlike swe-bench-ext, this family does not run any in-container
grading harness: it ships a per-instance ``run_script.sh`` + ``parsing_script.py``
that emit a structured ``output.json`` test report. The recipe is the classic
3-hop:

    1. ``bash run_script.sh <test_files> > stdout.log 2> stderr.log``  (keep streams separate)
    2. ``python parsing_script.py stdout.log stderr.log output.json``  (parse to JSON report)
    3. read ``output.json`` back host-side

Grading is then a pure host-side parse of that report's ``{tests: [{name, status}]}``
shape, identical to the legacy ``check_tests_passed`` rule. Because the family is
flat and host-graded, it runs on any exec-capable provider (e.g. docker); it does
not require apptainer. The mounted scripts/patch of the legacy apptainer command
(swe_agents/app.py:1864-1873) become ``materialize`` uploads here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.grading import compute_resolved
from responses_api_agents.swe_env.harness import (
    EvalArtifacts,
    SweEvalReport,
    SweTask,
    SweTaskHarness,
    _ensure_trailing_newline,
)


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


def parse_passed_tests(report: dict[str, Any]) -> list[str]:
    """Extract PASSED test names from a parsing_script ``output.json`` report.

    The report shape is ``{"tests": [{"name": ..., "status": "PASSED"|...}, ...]}``
    (mirrors ``check_tests_passed`` swe_agents/app.py:679).
    """
    return [
        test["name"]
        for test in report.get("tests", [])
        if isinstance(test, dict) and test.get("status") == "PASSED" and "name" in test
    ]


class NVInternalHarness(SweTaskHarness):
    name = "nv-internal-1"
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

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Upload run_script.sh + parsing_script.py + the model patch.

        Mirrors the legacy apptainer mounts of these three files
        (swe_agents/app.py:1864-1873). The scripts live in ``task.metadata``
        (read off the dataset ``instance_dict["run_script.sh"]`` /
        ``["parsing_script.py"]`` at app.py:581-582).
        """
        if task.model_patch:
            await env.write_text("/root/patch.diff", _ensure_trailing_newline(task.model_patch))
        run_script = task.metadata.get("run_script", "")
        parsing_script = task.metadata.get("parsing_script", "")
        if run_script:
            await env.write_text("/root/run_script.sh", _ensure_trailing_newline(run_script))
        if parsing_script:
            await env.write_text("/root/parsing_script.py", _ensure_trailing_newline(parsing_script))

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Reset the checkout to ``base_commit`` (own reset; app.py:601-602).

        The legacy processor works in ``/app`` and does ``git reset --hard`` +
        ``git checkout`` of the base commit (not ``git clean``), so we override
        the default reset to match.
        """
        if task.base_commit:
            await env.execute(
                f"git reset --hard {task.base_commit} && git checkout {task.base_commit}",
                cwd=task.repo_workdir,
            )

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        workdir = task.repo_workdir
        # Apply the model patch with rejection to tolerate conflicts (app.py:605):
        # `--reject` writes .rej files instead of failing; `|| true` keeps going.
        patch_applied = True
        if task.model_patch:
            applied = await env.execute(
                "git apply --ignore-space-change --ignore-whitespace --reject -v /root/patch.diff",
                cwd=workdir,
            )
            patch_applied = applied["returncode"] == 0

        # Optional per-instance repo setup hook (app.py:570-572, :608).
        repo_cmd = task.metadata.get("before_repo_set_cmd", "").strip()
        if repo_cmd:
            repo_cmd = repo_cmd.split("\n")[-1]
            setup = await env.execute(repo_cmd, cwd=workdir, is_eval=True)
            if setup.get("error_type") in {"sandbox", "timeout"}:
                return EvalArtifacts(
                    test_output=setup["output"],
                    return_code=setup["returncode"],
                    patch_applied=patch_applied,
                    raw={"error_type": setup.get("error_type")},
                )

        # Hop 1: run the per-instance script, keeping stdout/stderr separate
        # (app.py:611). The selected test files are passed positionally.
        test_files = _format_test_files(task.metadata.get("selected_test_files_to_run", []))
        run = await env.execute(
            f"bash /root/run_script.sh {test_files} > /root/stdout.log 2> /root/stderr.log || true",
            cwd=workdir,
            is_eval=True,
        )
        if run.get("error_type") in {"sandbox", "timeout"}:
            return EvalArtifacts(
                test_output=run["output"],
                return_code=run["returncode"],
                patch_applied=patch_applied,
                raw={"error_type": run.get("error_type")},
            )

        # Hop 2: parse the logs into a JSON report (app.py:614).
        parse = await env.execute(
            "python /root/parsing_script.py /root/stdout.log /root/stderr.log /root/output.json",
            cwd=workdir,
            is_eval=True,
        )
        if parse.get("error_type") in {"sandbox", "timeout"}:
            return EvalArtifacts(
                test_output=parse["output"],
                return_code=parse["returncode"],
                patch_applied=patch_applied,
                raw={"error_type": parse.get("error_type")},
            )

        # Hop 3: read the report back host-side (instead of mounting it out).
        report = await env.execute("cat /root/output.json", cwd=workdir, is_eval=True)
        return EvalArtifacts(
            test_output=report["output"],
            return_code=report["returncode"],
            patch_applied=patch_applied,
            raw={"error_type": report.get("error_type")},
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
        try:
            report = json.loads(artifacts.test_output) if artifacts.test_output.strip() else {}
        except (ValueError, TypeError):
            report = {}
        passed = parse_passed_tests(report)
        # check_tests_passed (app.py:670): empty report or no required tests → unresolved.
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
            tests_status={"passed": passed, "report": report},
        )


def _format_test_files(test_files: Any) -> str:
    """Build the comma-joined test-files argument (app.py:575-579).

    Accepts a list, or a string that is either a comma-joined value or a
    ``repr``-style list (the legacy ``selected_test_files_to_run`` is stored as
    a stringified list and ``eval``-ed at app.py:577).
    """
    if isinstance(test_files, (list, tuple)):
        return ",".join(str(item) for item in test_files)
    if isinstance(test_files, str):
        stripped = test_files.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return ",".join(str(item) for item in parsed)
            except ValueError:
                pass
        return stripped
    return ""
