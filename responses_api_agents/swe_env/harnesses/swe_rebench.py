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

"""swe-rebench harness: flat, host-graded family with a vendored log parser.

Ports ``SWERebenchDatasetProcessor`` + ``_load_rebench_log_parsers`` +
``_normalize_test_name`` (swe_agents/app.py:689-900). Like swe-bench-ext this is
a flat host-graded family: reset to base, apply the test patch + model patch,
run the install/test commands, then parse the test log **host-side**.

Two things make swe-rebench different from swe-bench-ext:

* **JAVA env** — the legacy apptainer launcher injects
  ``_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false`` for SWE-rebench tasks
  (swe_agents/app.py:1936-1937). We surface it via ``build_spec.env`` so it is
  set for the whole sandbox session.
* **Dynamic log parser** — swe-rebench has no single uniform pytest summary; the
  correct per-test PASSED/FAILED status comes from a repo-specific parser keyed
  by ``log_parser`` and shipped in the cloned ``SWE-rebench-V2`` repo
  (``lib/agent/log_parsers.py`` or ``agent/log_parsers.py``). We import it
  dynamically (mirrors ``_load_rebench_log_parsers``), guarded by try/except.

The cloned ``SWE-rebench-V2`` directory must be provisioned out-of-band (see
``responses_api_agents/swe_agents/setup_scripts/swe_rebench.sh``). When it is
absent or the named parser cannot be resolved, ``grade`` masks the sample via
``error_kind`` rather than scoring a misleading ``unresolved``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask, SweTaskHarness


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


# JAVA flag the legacy launcher injects for every SWE-rebench task
# (swe_agents/app.py:1936-1937).
_JAVA_OPTIONS = "-Djava.net.preferIPv6Addresses=false"

# Patch-apply flags shared by the model + test patch; mirrors the non-fatal
# ``git apply --reject`` style of the legacy run command (app.py:798-801).
_APPLY_FLAGS = "--reject --recount --ignore-space-change --whitespace=nowarn"

# Timing/duration suffixes some test runners append to node names; stripped so
# the parser output lines up with the (already-normalized) expected node ids.
# Ports ``SWERebenchDatasetProcessor._normalize_test_name`` (app.py:736-744).
_REBENCH_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    """Strip trailing timing annotations from a test node name.

    Ports ``SWERebenchDatasetProcessor._normalize_test_name`` (app.py:736-744).
    """
    for pattern in _REBENCH_TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _load_rebench_log_parsers(rebench_repo_dir: Path):
    """Dynamically import the cloned SWE-rebench-V2 ``log_parsers`` module.

    Ports ``_load_rebench_log_parsers`` (app.py:689-710): prefers
    ``lib/agent/log_parsers.py`` then falls back to ``agent/log_parsers.py``,
    temporarily prepending the repo (and its ``lib`` dir) to ``sys.path`` so the
    module's intra-repo imports resolve. Raises ``FileNotFoundError`` if the
    cloned directory has not been provisioned.
    """
    lp_path = rebench_repo_dir / "lib" / "agent" / "log_parsers.py"
    if not lp_path.exists():
        lp_path = rebench_repo_dir / "agent" / "log_parsers.py"
    if not lp_path.exists():
        raise FileNotFoundError(
            f"SWE-rebench-V2 log_parsers not found under {rebench_repo_dir}; "
            "provision the clone via setup_scripts/swe_rebench.sh"
        )

    extra_paths = [str(rebench_repo_dir), str(rebench_repo_dir / "lib")]
    added: list[str] = []
    for p in extra_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    try:
        spec = importlib.util.spec_from_file_location("_rebench_log_parsers", str(lp_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def _resolve_parser(log_parsers, log_parser_name: str) -> Callable[[str], dict[str, str]] | None:
    """Resolve a parser callable from the loaded module (ports app.py:859)."""
    name_to_parser = getattr(log_parsers, "NAME_TO_PARSER", {}) or {}
    return name_to_parser.get(log_parser_name) or getattr(log_parsers, log_parser_name, None)


def _as_list(value: Any) -> list[str]:
    """Coerce a test-command/install/list field to a list of strings.

    The legacy processor accepts these as either a JSON-encoded string, a bare
    string, or a list (app.py:749-755, 766-769).
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return [value]
            return _as_list(parsed)
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


class SweRebenchHarness(SweTaskHarness):
    name = "swe-rebench"
    grade_strategy = "flat-host-grade"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        # _JAVA_OPTIONS mirrors the legacy ``--env`` injection (app.py:1936-1937).
        env = {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_PAGER": "cat",
            "_JAVA_OPTIONS": _JAVA_OPTIONS,
        }
        env.update(task.metadata.get("env", {}))
        return SandboxSpec(
            image=task.image,
            workdir=task.repo_workdir,
            ttl_s=task.metadata.get("ttl_s", 1800),
            ready_timeout_s=task.metadata.get("ready_timeout_s", 600),
            env=env,
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
        install_config = task.metadata.get("install_config", {}) or {}
        install_cmds = _as_list(install_config.get("install"))
        test_cmds = _as_list(install_config.get("test_cmd")) or ([task.test_command] if task.test_command else [])

        # Apply the test patch first, then the model patch. Both are non-fatal
        # (``|| true``) just like the legacy run script (app.py:798-801): a
        # failed apply still runs the tests, and grading flags non-application.
        patch_applied = True
        if task.test_patch:
            await env.execute(f"git apply {_APPLY_FLAGS} /root/test_patch.diff || true", cwd=workdir)
        if task.model_patch:
            applied = await env.execute(
                f"git apply {_APPLY_FLAGS} /root/patch.diff",
                cwd=workdir,
            )
            patch_applied = applied["returncode"] == 0

        # Install commands are non-fatal (app.py:803-806); failures there should
        # not abort the test run.
        for cmd in install_cmds:
            await env.execute(cmd, cwd=workdir)

        test_block = "\n".join(test_cmds) if test_cmds else "python -m pytest -rA -q"
        result = await env.execute(test_block, cwd=workdir, is_eval=True)
        return EvalArtifacts(
            test_output=result["output"],
            return_code=result["returncode"],
            patch_applied=patch_applied,
            raw={"error_type": result.get("error_type")},
        )

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        # Infra failure -> mask via error_kind (never scored as "unresolved").
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )

        install_config = task.metadata.get("install_config", {}) or {}
        log_parser_name = install_config.get("log_parser", "")
        # The cloned SWE-rebench-V2 dir is provisioned out-of-band; its absence,
        # an unknown parser name, or a parser crash all mask the sample via
        # ``error_kind`` (mirrors the legacy "Unknown log parser" / "No test
        # output" guard rails at app.py:844-870) rather than mis-scoring it.
        rebench_repo_dir = task.metadata.get("rebench_repo_dir")
        if not rebench_repo_dir:
            return self._masked(task, artifacts, "eval_error")
        try:
            log_parsers = _load_rebench_log_parsers(Path(rebench_repo_dir))
            parser = _resolve_parser(log_parsers, log_parser_name)
            if parser is None:
                return self._masked(task, artifacts, "eval_error")
            results = parser(artifacts.test_output)
        except Exception:
            return self._masked(task, artifacts, "eval_error")

        results = {_normalize_test_name(k): v for k, v in (results or {}).items()}
        passed_set = {k for k, v in results.items() if v == "PASSED"}
        fail_to_pass_set = {_normalize_test_name(n) for n in task.fail_to_pass}
        pass_to_pass_set = {_normalize_test_name(n) for n in task.pass_to_pass}

        # Resolution rule mirrors postprocess_after_run (app.py:888): every
        # FAIL_TO_PASS and PASS_TO_PASS test must be in the passed set.
        required = fail_to_pass_set | pass_to_pass_set
        resolved = (
            artifacts.patch_applied
            and bool(required)
            and fail_to_pass_set <= passed_set
            and pass_to_pass_set <= passed_set
        )
        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status={"passed": sorted(passed_set), "all": results},
        )

    @staticmethod
    def _masked(task: SweTask, artifacts: EvalArtifacts, kind: str) -> SweEvalReport:
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            patch_applied=artifacts.patch_applied,
            error_kind=kind,
        )
