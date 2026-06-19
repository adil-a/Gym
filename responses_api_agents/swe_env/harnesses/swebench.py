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

"""swe-bench / swe-bench-multilingual harness: nested, in-container grading.

Ports ``SweBenchDatasetProcessor`` and ``SweBenchMultilingualDatasetProcessor``
(swe_agents/app.py:326-463) into a single parametrized class. Both families run
the upstream SWE-bench ``run_local_evaluation`` harness *inside* the sandbox
(the pre-built venv is bind-mounted from the host setup dir — see the mount
switch at app.py:1850-1862), then read the harness's ``report.json`` to decide
``resolved`` (app.py:2107-2114: ``report[instance_id]["resolved"]``).

Because the nested harness shells out to its own Docker/Apptainer runtime to
spin up the per-instance image, these families are gated to the ``apptainer``
provider via ``supports_provider`` (fail-fast on exec-only providers). They
cannot run on this box (no apptainer + no real ``.sif``), so ``run_eval`` only
*builds and issues* the in-container eval command and ``grade`` parses the
emitted ``report.json``. Real-instance validation is deferred to an apptainer
cluster; the unit tests cover ``build_spec`` / ``supports_provider`` /
``materialize`` / ``grade`` against a scripted ``FakeSandbox``.
"""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask, SweTaskHarness
from responses_api_agents.swe_env.harnesses import flat_eval


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


# Where the nested harness reads predictions / dataset and writes its report.
# These mirror the legacy mounts (app.py:1828 dataset, :369-377 run_local_evaluation).
_DATASET_PATH = "/root/dataset/data.jsonl"
_PREDICTIONS_PATH = "/root/predictions.jsonl"
_REPORT_PATH = "/root/report.json"

# Per-family in-container setup dir + the python entrypoint used to invoke the
# upstream ``run_local_evaluation`` module. Keyed by harness/dataset name.
#   * swe-bench: HeyyyyyyG/SWE-bench fork mounted at /swebench_setup (app.py:361-369, 1853)
#   * swe-bench-multilingual: Kipok/SWE-bench fork mounted at
#     /swebench_multilingual_setup (app.py:431-439, 1856-1862)
_FAMILY_CONFIG: dict[str, dict[str, str]] = {
    "swe-bench": {
        "setup_dir": "/swebench_setup",
        "harness_subdir": "SWE-bench",
    },
    "swe-bench-multilingual": {
        "setup_dir": "/swebench_multilingual_setup",
        "harness_subdir": "SWE-bench_Multilingual",
    },
}


class SweBenchHarness(SweTaskHarness):
    """Nested SWE-bench (+ multilingual) harness.

    A single class serves both registry keys; construct one instance per family
    (``SweBenchHarness("swe-bench")`` / ``SweBenchHarness("swe-bench-multilingual")``)
    or let ``grade`` fall back to ``task.benchmark`` for family-specific config.
    """

    grade_strategy = "nested-harness"

    def __init__(self, name: str = "swe-bench", *, flat_eval: bool = False) -> None:
        if name not in _FAMILY_CONFIG:
            raise ValueError(f"Unknown swe-bench family: {name!r} (expected one of {sorted(_FAMILY_CONFIG)})")
        self.name = name
        # Opt-in flat (host-graded) mode — see harnesses/flat_eval.py. When True
        # the harness runs the instance's eval script directly in the sandbox
        # and parses the log host-side, lifting the apptainer-only gate so it can
        # run on docker/opensandbox. Default False keeps the nested behavior.
        self.flat_eval = flat_eval
        if flat_eval:
            self.grade_strategy = "flat-host-grade"

    # --- provisioning --------------------------------------------------------

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
                # Bind-mount the host-built SWE-bench harness venv at both its
                # canonical path and the in-container alias (uv hardcodes
                # absolute paths). Mirrors app.py:1850-1862.
                "mounts": self._family_mounts(task),
            },
            resources=SandboxResources.from_mapping(task.metadata.get("resources", {})),
            provider_options=task.metadata.get("provider_options", {}),
        )

    def supports_provider(self, provider_name: str) -> bool:
        # Flat mode is host-graded (no nested container), so it runs on any
        # exec-capable provider. Only a flat-capable harness instance lifts the
        # apptainer-only restriction (see harnesses/flat_eval.py gating notes).
        if self.flat_eval:
            return True
        # Nested family: the upstream harness manages its own container runtime.
        # Reject exec-only providers (docker/fake) and require apptainer.
        return provider_name == "apptainer"

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        # The nested harness consumes a predictions JSONL keyed by instance_id
        # rather than a bare patch.diff (app.py:370 --predictions_path).
        prediction = {
            "instance_id": task.instance_id,
            "model_name_or_path": task.metadata.get("model_name_or_path", "nemo-gym"),
            "model_patch": task.model_patch or "",
        }
        await env.write_text(_PREDICTIONS_PATH, json.dumps(prediction) + "\n")

    # --- server-private grading ----------------------------------------------

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        # Opt-in flat mode: run the instance's eval script in-sandbox and grade
        # the log host-side (docker/opensandbox-capable). Default path below is
        # the nested run_local_evaluation harness (apptainer-only).
        if flat_eval.flat_eval_enabled(self.flat_eval, task):
            return await flat_eval.flat_run_eval(env, task)

        host_setup_dir = task.metadata.get("setup_dir") or self._family_config(task)["setup_dir"]
        harness_subdir = self._family_config(task)["harness_subdir"]
        venv_python = f"{host_setup_dir}/{harness_subdir}/venv/bin/python"
        timeout = int(task.metadata.get("tests_timeout", 1800))
        run_id = task.metadata.get("run_id", task.instance_id)
        split = task.split or "test"

        # Build the in-container eval command: run the upstream harness against
        # the materialized predictions and redirect its report.json to a known
        # path. Mirrors app.py:357-377 (HeyyyyyyG / Kipok forks of SWE-bench).
        eval_cmd = (
            f"cd {host_setup_dir}/{harness_subdir} && "
            f"env -u VIRTUAL_ENV {shlex.quote(venv_python)} -m swebench.harness.run_local_evaluation "
            f"--predictions_path {shlex.quote(_PREDICTIONS_PATH)} "
            f"--instance_ids {shlex.quote(task.instance_id)} "
            f"--timeout {timeout} "
            f"--dataset_name {shlex.quote(_DATASET_PATH)} "
            f"--split {shlex.quote(split)} "
            f"--run_id {shlex.quote(str(run_id))}"
        )
        # The upstream harness writes logs/run_evaluation/<run_id>/<model>/<instance>/report.json;
        # locate it and copy to a stable path so grade() can read a single file.
        collect_cmd = (
            f"REPORT=$(find logs/run_evaluation/{shlex.quote(str(run_id))} -name report.json | head -n1); "
            f'if [ -n "$REPORT" ]; then cp "$REPORT" {shlex.quote(_REPORT_PATH)}; fi'
        )
        result = await env.execute(f"{eval_cmd} && {collect_cmd}", cwd=task.repo_workdir, is_eval=True)

        # Read the emitted report.json back out of the sandbox for host-side grading.
        report_text = ""
        if result.get("error_type") not in {"sandbox", "timeout"}:
            cat = await env.execute(f"cat {shlex.quote(_REPORT_PATH)}", cwd=task.repo_workdir)
            if cat["returncode"] == 0:
                report_text = cat["output"]

        return EvalArtifacts(
            test_output=result["output"],
            return_code=result["returncode"],
            patch_applied=bool(task.model_patch),
            raw={"error_type": result.get("error_type"), "report_json": report_text},
        )

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        # Flat mode: host-side parse of the eval-script log. Detected from either
        # the harness flag/task opt-in OR the artifacts produced by flat_run_eval
        # (so a flat run_eval is always graded flat, even on a shared instance).
        if flat_eval.flat_eval_enabled(self.flat_eval, task) or artifacts.raw.get("flat"):
            return flat_eval.flat_grade(task, artifacts)

        # Infra failure -> mask via error_kind (never scored as "unresolved").
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )

        report_text = artifacts.raw.get("report_json") or ""
        resolved = False
        try:
            report = json.loads(report_text)
            # Upstream harness keys report.json by instance_id (app.py:2107-2111).
            entry = report.get(task.instance_id, {}) if isinstance(report, dict) else {}
            resolved = bool(entry.get("resolved", False))
        except (json.JSONDecodeError, TypeError, AttributeError):
            # Missing / malformed report -> eval failure; mask rather than score 0.
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind="eval_error",
            )

        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status={"report": report_text},
        )

    # --- helpers -------------------------------------------------------------

    def _family_config(self, task: SweTask) -> dict[str, str]:
        # Prefer the instance's own name; fall back to task.benchmark so a single
        # shared instance can still serve either family.
        name = self.name if self.name in _FAMILY_CONFIG else task.benchmark
        return _FAMILY_CONFIG.get(name, _FAMILY_CONFIG["swe-bench"])

    def _family_mounts(self, task: SweTask) -> list[dict[str, str]]:
        cfg = self._family_config(task)
        host_setup_dir = task.metadata.get("host_setup_dir")
        mounts: list[dict[str, str]] = [
            # Dataset mounted at the fixed in-container path the harness reads.
            {"src": task.metadata.get("dataset_path", _DATASET_PATH), "dst": _DATASET_PATH},
        ]
        if host_setup_dir:
            # Bind the host setup dir at both the alias and its canonical path
            # (uv venvs hardcode absolute paths). See app.py:1853-1862.
            mounts.append({"src": host_setup_dir, "dst": cfg["setup_dir"]})
            mounts.append({"src": host_setup_dir, "dst": host_setup_dir})
        return mounts
