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

"""Unit tests for the opt-in flat (host-graded) eval mode of the *nested* families.

Two layers:

* **Parser unit tests on recorded fixture logs** (``fixtures/flat_eval/*.txt``,
  ``.txt`` so the repo's ``*.log`` gitignore rule does not drop them):
  these RUN in CI. They cover the SWE-bench eval-script log parser
  (``parse_eval_log``) on a success log, a failure log, the bad-code logs
  (patch-apply-failed / timeout), a no-markers log, and the
  output-outside-markers fallback. The fixtures mirror the real
  ``>>>>> Start/End Test Output`` shape the upstream eval script emits (verified
  against ``swebench.harness.log_parsers.python.parse_log_pytest`` while
  authoring them).

* **Flat run_eval + grade via FakeSandbox** (also CI): drives the flat path of
  both nested harnesses (``swe-bench``, ``r2e-gym``) end-to-end with a scripted
  provider that returns a fixture log, asserting ``resolved`` is computed from
  ``FAIL_TO_PASS`` / ``PASS_TO_PASS``.

* **Golden-patch equivalence scaffold** (``test_flat_vs_nested_equivalence_on_gold``):
  SKIPPED unless ``SWE_ENV_RUN_REAL_CONTAINERS=1``. Proving flat ``resolved`` ==
  nested ``resolved`` on gold patches needs apptainer + Docker + published
  per-instance SWE-bench ``.sif`` images, which are NOT available here. The
  scaffold documents the comparison so it can be run on a real cluster.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask
from responses_api_agents.swe_env.harnesses import flat_eval
from responses_api_agents.swe_env.harnesses.r2egym import R2EGymHarness
from responses_api_agents.swe_env.harnesses.swebench import SweBenchHarness


_FIXTURES = Path(__file__).parent / "fixtures" / "flat_eval"


def _fixture(name: str) -> str:
    # Fixtures are stored as ``.txt`` (the repo gitignores ``*.log``); callers
    # may pass either the ``.log`` stem name or the real ``.txt`` name.
    path = _FIXTURES / name
    if not path.exists() and path.suffix == ".log":
        path = path.with_suffix(".txt")
    return path.read_text()


# ---- parser: recorded fixture logs (CI) -------------------------------------


def test_parse_success_log_all_pass():
    status_map, applied = flat_eval.parse_eval_log(_fixture("resolved_success.log"))
    assert applied is True
    assert status_map == {
        "tests/test_ext_autodoc.py::test_format_signature": "PASSED",
        "tests/test_ext_autodoc.py::test_autodoc_inherited": "PASSED",
        "tests/test_ext_autodoc.py::test_autodoc_exclude_members": "PASSED",
        "tests/test_ext_autodoc.py::test_optional_feature": "SKIPPED",
    }
    assert sorted(flat_eval.passed_tests(status_map)) == [
        "tests/test_ext_autodoc.py::test_autodoc_exclude_members",
        "tests/test_ext_autodoc.py::test_autodoc_inherited",
        "tests/test_ext_autodoc.py::test_format_signature",
    ]


def test_parse_failure_log_strips_failed_reason():
    status_map, applied = flat_eval.parse_eval_log(_fixture("unresolved_failure.log"))
    assert applied is True
    # The "FAILED <id> - <reason>" line keeps only the node id (upstream behavior).
    assert status_map["tests/test_ext_autodoc.py::test_format_signature"] == "FAILED"
    assert "tests/test_ext_autodoc.py::test_autodoc_inherited" in flat_eval.passed_tests(status_map)


def test_parse_apply_patch_failed_is_untrusted():
    # A bad code (patch-apply-failed) -> empty map + patch_applied False.
    status_map, applied = flat_eval.parse_eval_log(_fixture("apply_patch_failed.log"))
    assert status_map == {}
    assert applied is False


def test_parse_timeout_is_untrusted():
    status_map, applied = flat_eval.parse_eval_log(_fixture("tests_timeout.log"))
    assert status_map == {}
    assert applied is False


def test_parse_no_markers_is_untrusted():
    status_map, applied = flat_eval.parse_eval_log(_fixture("no_markers.log"))
    assert status_map == {}
    assert applied is False


def test_parse_fallback_outside_markers():
    # Markers present but empty between them; per-test lines appear after the
    # End marker. The whole-log fallback recovers them.
    status_map, applied = flat_eval.parse_eval_log(_fixture("fallback_outside_markers.log"))
    assert applied is True
    assert len(flat_eval.passed_tests(status_map)) == 3


def test_parse_xfail_counts_as_pass():
    log = "\n".join(
        [
            flat_eval.APPLY_PATCH_PASS,
            flat_eval.START_TEST_OUTPUT,
            "XFAIL tests/test_x.py::test_known_bug",
            "PASSED tests/test_x.py::test_ok",
            flat_eval.END_TEST_OUTPUT,
        ]
    )
    status_map, applied = flat_eval.parse_eval_log(log)
    assert applied is True
    assert set(flat_eval.passed_tests(status_map)) == {
        "tests/test_x.py::test_known_bug",
        "tests/test_x.py::test_ok",
    }


# ---- flat_grade over parsed fixtures (CI) -----------------------------------


def _task(benchmark: str = "swe-bench", **overrides) -> SweTask:
    base = dict(
        instance_id="repo__inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["tests/test_ext_autodoc.py::test_format_signature"],
        pass_to_pass=["tests/test_ext_autodoc.py::test_autodoc_inherited"],
        benchmark=benchmark,
    )
    base.update(overrides)
    return SweTask(**base)


def _flat_artifacts(log: str) -> EvalArtifacts:
    return EvalArtifacts(test_output=log, return_code=0, patch_applied=True, raw={"error_type": None, "flat": True})


def test_flat_grade_resolved_on_success():
    report = flat_eval.flat_grade(_task(), _flat_artifacts(_fixture("resolved_success.log")))
    assert report.resolved is True
    assert report.patch_applied is True
    assert report.patch_exists is True
    assert reward_from_report(report) == 1.0


def test_flat_grade_unresolved_on_failure():
    report = flat_eval.flat_grade(_task(), _flat_artifacts(_fixture("unresolved_failure.log")))
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_flat_grade_unresolved_on_apply_failed():
    # A failed patch apply is a legitimate unresolved (not an infra mask).
    report = flat_eval.flat_grade(_task(), _flat_artifacts(_fixture("apply_patch_failed.log")))
    assert report.resolved is False
    assert report.patch_applied is False
    assert report.error_kind is None
    assert reward_from_report(report) == 0.0


def test_flat_grade_masks_infra_error():
    artifacts = EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout", "flat": True})
    report = flat_eval.flat_grade(_task(), artifacts)
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


def test_flat_grade_masks_missing_eval_script():
    artifacts = EvalArtifacts(test_output="", return_code=1, raw={"error_type": "eval_error", "flat": True})
    report = flat_eval.flat_grade(_task(), artifacts)
    assert report.error_kind == "eval_error"
    assert reward_from_report(report) == 0.0


# ---- gating (CI) ------------------------------------------------------------


def test_flat_eval_enabled_harness_flag():
    assert flat_eval.flat_eval_enabled(True, _task()) is True


def test_flat_eval_enabled_task_metadata():
    assert flat_eval.flat_eval_enabled(False, _task(metadata={"flat_eval": True})) is True


def test_flat_eval_disabled_by_default():
    assert flat_eval.flat_eval_enabled(False, _task()) is False


def test_swebench_supports_provider_gating():
    # Default (nested): apptainer only.
    nested = SweBenchHarness("swe-bench")
    assert nested.supports_provider("apptainer") is True
    assert nested.supports_provider("docker") is False
    assert nested.supports_provider("opensandbox") is False
    # Flat-capable instance: any exec provider.
    flat = SweBenchHarness("swe-bench", flat_eval=True)
    assert flat.supports_provider("docker") is True
    assert flat.supports_provider("opensandbox") is True
    assert flat.grade_strategy == "flat-host-grade"


def test_r2egym_supports_provider_gating():
    nested = R2EGymHarness()
    assert nested.supports_provider("apptainer") is True
    assert nested.supports_provider("docker") is False
    flat = R2EGymHarness(flat_eval=True)
    assert flat.supports_provider("docker") is True
    assert flat.supports_provider("opensandbox") is True
    assert flat.grade_strategy == "flat-host-grade"


# ---- flat run_eval end-to-end via FakeSandbox (CI) --------------------------


class _FakeFlatProvider:
    """Scripted provider: ``bash eval.sh ...`` streams a fixture log; ``cat`` echoes it."""

    name = "fake-flat-eval"

    def __init__(self, *, log_text="", run_rc=0, error_type=None, stream_empty=False, **_):
        self._log_text = log_text
        self._run_rc = run_rc
        self._error_type = error_type
        self._stream_empty = stream_empty
        self.commands: list[str] = []
        self.uploaded: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
        if command.startswith("cat "):
            return SandboxExecResult(stdout=self._log_text, stderr="", return_code=0)
        # The eval script run.
        stdout = "" if self._stream_empty else self._log_text
        return SandboxExecResult(stdout=stdout, stderr="", return_code=self._run_rc, error_type=self._error_type)

    async def upload_file(self, handle, local_path, remote_path):
        try:
            with open(local_path, encoding="utf-8") as fh:
                self.uploaded[remote_path] = fh.read()
        except OSError:
            self.uploaded[remote_path] = ""
        return None

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        return None

    async def aclose(self):
        return None


register_provider("fake-flat-eval", _FakeFlatProvider, override=True)


def _drive_flat(harness, task, *, log_text, run_rc=0, error_type=None, stream_empty=False):
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    async def _go():
        provider = {
            "fake-flat-eval": {
                "log_text": log_text,
                "run_rc": run_rc,
                "error_type": error_type,
                "stream_empty": stream_empty,
            }
        }
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        try:
            await harness.materialize(env, task)
            artifacts = await harness.run_eval(env, task)
            return harness.grade(task, artifacts), artifacts, env.sandbox._provider
        finally:
            await env.cleanup()

    return asyncio.run(_go())


def test_swebench_flat_run_eval_resolved():
    harness = SweBenchHarness("swe-bench", flat_eval=True)
    task = _task(metadata={"eval_script": "echo running", "flat_eval": True})
    report, artifacts, provider = _drive_flat(harness, task, log_text=_fixture("resolved_success.log"))
    assert artifacts.raw["flat"] is True
    assert report.resolved is True
    assert reward_from_report(report) == 1.0
    # The eval script was uploaded into the sandbox.
    assert provider.uploaded.get(flat_eval.EVAL_SCRIPT_PATH, "").startswith("echo running")


def test_swebench_flat_run_eval_unresolved():
    harness = SweBenchHarness("swe-bench", flat_eval=True)
    task = _task(metadata={"eval_script": "echo running"})
    report, _artifacts, _ = _drive_flat(harness, task, log_text=_fixture("unresolved_failure.log"))
    assert report.resolved is False


def test_swebench_flat_run_eval_stream_empty_uses_log_file():
    # When the streamed output is empty, run_eval reads back the tee'd log file.
    harness = SweBenchHarness("swe-bench", flat_eval=True)
    task = _task(metadata={"eval_script": "echo running"})
    report, _artifacts, provider = _drive_flat(
        harness, task, log_text=_fixture("resolved_success.log"), stream_empty=True
    )
    assert any(cmd.startswith("cat ") for cmd in provider.commands)
    assert report.resolved is True


def test_swebench_flat_run_eval_masks_sandbox_error():
    harness = SweBenchHarness("swe-bench", flat_eval=True)
    task = _task(metadata={"eval_script": "echo running"})
    report, artifacts, _ = _drive_flat(harness, task, log_text="", run_rc=1, error_type="sandbox")
    assert artifacts.raw["error_type"] == "sandbox"
    assert report.error_kind == "sandbox"


def test_swebench_flat_run_eval_missing_script_masks_eval_error():
    harness = SweBenchHarness("swe-bench", flat_eval=True)
    task = _task(metadata={})  # no eval_script
    report, artifacts, _ = _drive_flat(harness, task, log_text="")
    assert artifacts.raw["error_type"] == "eval_error"
    assert report.error_kind == "eval_error"


def test_r2egym_flat_run_eval_resolved_via_task_metadata():
    # Per-task opt-in on a flat-capable instance.
    harness = R2EGymHarness(flat_eval=True)
    task = _task(benchmark="r2e-gym", instance_id="r2e__pkg-1", metadata={"eval_script": "echo run"})
    report, artifacts, _ = _drive_flat(harness, task, log_text=_fixture("resolved_success.log"))
    assert artifacts.raw["flat"] is True
    assert report.resolved is True


# ---- infra-gated golden-patch equivalence scaffold --------------------------


@pytest.mark.skipif(
    os.environ.get("SWE_ENV_RUN_REAL_CONTAINERS") != "1",
    reason=(
        "Real flat-vs-nested equivalence needs apptainer + Docker + published per-instance "
        "SWE-bench .sif images, which are not available in CI/this workstation. "
        "Set SWE_ENV_RUN_REAL_CONTAINERS=1 on a cluster that has them."
    ),
)
def test_flat_vs_nested_equivalence_on_gold():  # pragma: no cover - infra-gated
    """Scaffold: flat resolved == nested resolved on gold patches.

    On a real cluster this would, for a small set of instances with their gold
    ``model_patch``:

      1. Run the NESTED path (apptainer): ``SweBenchHarness("swe-bench")`` with
         ``run_local_evaluation`` -> nested ``resolved``.
      2. Run the FLAT path (docker/apptainer):
         ``SweBenchHarness("swe-bench", flat_eval=True)`` with the upstream
         ``make_test_spec(instance).eval_script`` -> flat ``resolved``.
      3. Assert ``flat_report.resolved == nested_report.resolved`` for every
         instance (gold patches must resolve under BOTH graders).

    The dataset, .sif images, and both runtimes are provisioned out-of-band; see
    ``harnesses/flat_eval.py`` for why this is infra-gated.
    """
    raise AssertionError("equivalence harness must be implemented against a real cluster")
