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

"""r2e-gym harness: nested, in-container-graded family.

Ports ``R2EGymDatasetProcessor`` (swe_agents/app.py:466-536). Unlike the flat
``swe-bench-ext`` family, r2e-gym does NOT grade host-side: the per-instance
``report.json`` is produced by the *vendored* r2e-gym evaluation harness
(``src/r2egym/agenthub/run/run_local_evaluation.py``) running inside the
container. ``grade()`` therefore only parses that report's already-computed
``resolved`` verdict rather than reconstructing it from per-test status.

Two r2e-gym-specific wrinkles are preserved from the legacy processor:

* **Test hiding during the agent phase** (app.py:1912-1922). ``/r2e_tests``
  holds the held-out evaluation tests, and ``run_tests.sh`` launches them, so
  both are removed from the agent's checkout (root, ``/root``, ``/testbed``).
  During *grading* (the verifier) these are present, because the nested harness
  re-materializes them — ``hide_eval_tests_commands`` is exposed for the agent
  adapter to run after ``materialize`` and is intentionally NOT invoked by
  ``run_eval``.
* **r2egym_setup mount** (app.py:1875-1880). The prebuilt R2E-Gym venv has
  hardcoded absolute paths in its uv wrappers, so the setup dir is bind-mounted
  at both ``/r2egym_setup`` and its original absolute path. These mounts are
  surfaced via ``provider_options["mounts"]`` for the apptainer provider.

This family requires apptainer + a real ``.sif`` container and cannot run on
this workstation (exec-only / docker). ``supports_provider`` fails fast on any
non-apptainer provider. Real-instance validation is therefore deferred to an
apptainer cluster; the unit tests here exercise spec construction, the provider
gate, the test-hiding command shape, and report parsing with a FakeSandbox.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask, SweTaskHarness


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


# Location the nested r2e-gym harness writes its per-instance report to inside
# the container. ``run_eval`` redirects ``run_local_evaluation.py`` here and
# then reads it back host-side for parsing.
_REPORT_PATH = "/root/r2egym_report.json"


class R2EGymHarness(SweTaskHarness):
    name = "r2e-gym"
    grade_strategy = "nested-harness"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        setup_dir = task.metadata.get("r2egym_setup_dir", "/r2egym_setup")
        # The prebuilt uv venv has hardcoded absolute paths, so the setup dir is
        # bind-mounted at both ``/r2egym_setup`` and its original absolute path
        # (app.py:1875-1880). The apptainer provider consumes ``mounts``.
        mounts = [
            {"src": setup_dir, "dst": "/r2egym_setup"},
            {"src": setup_dir, "dst": setup_dir},
        ]
        provider_options = dict(task.metadata.get("provider_options", {}))
        provider_options.setdefault("mounts", mounts)
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
            provider_options=provider_options,
        )

    def supports_provider(self, provider_name: str) -> bool:
        # Nested family: the vendored harness only runs under apptainer with a
        # real .sif. Fail fast on exec-only providers (docker/local).
        return provider_name == "apptainer"

    def hide_eval_tests_commands(self) -> list[str]:
        """Shell commands that strip the held-out eval tests from the agent's checkout.

        Ports app.py:1912-1922. ``/r2e_tests`` holds the evaluation tests the
        agent must not see; ``run_tests.sh`` launches them. We only delete
        ``run_tests.sh`` when it references ``r2e_tests`` (substring guard) to
        avoid clobbering an unrelated file with that name. The agent adapter
        runs these after ``materialize``; the verifier does NOT (the nested
        harness needs the tests back for grading).
        """
        commands: list[str] = []
        for root_dir in ["", "/root", "/testbed"]:
            commands.append(
                f"rm -rf {root_dir}/r2e_tests && "
                f"if grep -qs r2e_tests {root_dir}/run_tests.sh; then rm -rf {root_dir}/run_tests.sh; fi"
            )
        return commands

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        # The nested r2e-gym harness reads the model patch from the predictions
        # file, applies it, runs the held-out tests, and writes ``report.json``.
        # We build the in-container command (mirrors app.py:504-522) and redirect
        # its report to ``_REPORT_PATH``, then read it back host-side for grading.
        setup_dir = task.metadata.get("r2egym_setup_dir", "/r2egym_setup")
        predictions_path = task.metadata.get("predictions_path", "/root/predictions.jsonl")
        dataset_path = task.metadata.get("dataset_path", "/root/dataset/data.jsonl")
        timeout = task.metadata.get("tests_timeout", 1800)
        output_dir = task.metadata.get("eval_output_dir", "/root/eval-outputs")
        eval_cmd = (
            "cd /r2egym_setup/R2E-Gym && "
            f'export UV_INSTALL_DIR="{setup_dir}/uv" && '
            f'export UV_PYTHON_INSTALL_DIR="{setup_dir}/python" && '
            f'export PATH="{setup_dir}/uv/bin:$PATH" && '
            f"env -u VIRTUAL_ENV {setup_dir}/R2E-Gym/venv/bin/python "
            "src/r2egym/agenthub/run/run_local_evaluation.py "
            f"--predictions_path {predictions_path} "
            f"--instance_id {task.instance_id} "
            f"--timeout {timeout} "
            f"--dataset {dataset_path} "
            f"--output_dir {output_dir} && "
            # Surface the per-instance report at a stable, well-known path.
            f"cp {output_dir}/report.json {_REPORT_PATH}"
        )
        result = await env.execute(eval_cmd, cwd=task.repo_workdir, is_eval=True, timeout_s=timeout + 120)
        report_text = ""
        if result["returncode"] == 0:
            report = await env.execute(f"cat {_REPORT_PATH}", cwd=task.repo_workdir, is_eval=True)
            if report["returncode"] == 0:
                report_text = report["output"]
        return EvalArtifacts(
            test_output=report_text or result["output"],
            return_code=result["returncode"],
            # The nested harness applies the patch itself; absent a host apply
            # step we treat a clean eval as "applied" and let grade() mask
            # infra failures via error_kind.
            patch_applied=result["returncode"] == 0,
            raw={"error_type": result.get("error_type"), "report_json": report_text},
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
        report_text = artifacts.raw.get("report_json") or artifacts.test_output
        try:
            report = json.loads(report_text)
        except (json.JSONDecodeError, TypeError):
            # The nested harness never produced a parseable report → eval error.
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind="eval_error",
            )
        # report.json is keyed by instance_id (app.py:385/528 standard SWE-bench
        # shape); fall back to the sole entry if the key was rewritten.
        entry = report.get(task.instance_id)
        if entry is None and len(report) == 1:
            entry = next(iter(report.values()))
        if not isinstance(entry, dict):
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind="eval_error",
            )
        # The nested harness has already computed ``resolved``; trust it.
        resolved = bool(entry.get("resolved", False))
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status=entry.get("tests_status", {}),
        )
