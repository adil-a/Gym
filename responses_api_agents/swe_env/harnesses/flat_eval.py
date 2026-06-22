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

"""Flat (host-graded) eval-script mode shared by the *nested* SWE families.

Background
----------
The nested families (``swe-bench`` / ``swe-bench-multilingual`` in
``swebench.py``, and ``r2e-gym`` in ``r2egym.py``) normally grade by running the
upstream ``run_local_evaluation`` harness *inside* the sandbox. That harness
shells out to its own Docker/Apptainer runtime to spin up the per-instance image
(nested containerization), so those families gate ``supports_provider`` to
``apptainer`` only.

This module adds an **opt-in flat mode** that mirrors the flat families
(``swe_bench_ext.py`` / ``nv_internal.py`` / ``swe_rebench.py``): instead of
invoking the nested grader, we run the instance's *eval script* directly in the
sandbox and parse the produced log **host-side**, computing ``resolved`` from
``FAIL_TO_PASS`` / ``PASS_TO_PASS`` via :func:`compute_resolved`. Because there
is no nested container, this runs on any exec-capable provider (docker /
opensandbox), not just apptainer.

The eval script is the upstream SWE-bench eval script
(``swebench.harness.test_spec.make_test_spec(instance).eval_script``). It resets
the repo, applies the gold/model + test patch, runs the repo's test command, and
**wraps the test output between two sentinel markers**::

    >>>>> Start Test Output
    ... per-test "PASSED <id>" / "FAILED <id>" lines ...
    >>>>> End Test Output

plus patch-apply / reset / timeout status codes (``>>>>> Applied Patch`` etc.).
See ``swebench/harness/constants/__init__.py`` (TestStatus + the ``>>>>>``
codes) and ``swebench/harness/grading.py::get_logs_eval`` for the host-side
parse this module re-implements *without* importing ``swebench`` — grading must
run in the verifier/CI where the heavy ``swebench`` package (and its Docker
deps) may be absent.

Gating (do NOT regress)
-----------------------
* The nested (apptainer) path remains the **default**. Flat mode is opt-in via a
  harness-level flag (``flat_eval=True`` on the harness constructor) and/or a
  per-task ``SweTask.metadata["flat_eval"]`` key. Existing behavior is unchanged
  until flat mode is explicitly selected.
* ``supports_provider`` only lifts the apptainer-only restriction when the
  harness instance was constructed in flat mode. A per-task flag alone does NOT
  lift it (the provider is chosen at provisioning time from the harness
  capability, before any task metadata is consulted there).

Equivalence is infra-gated
--------------------------
Proving flat ``resolved`` == nested ``resolved`` on gold patches requires
apptainer + Docker + the published per-instance SWE-bench ``.sif`` images, which
are NOT available in this environment. The equivalence test
(``test_flat_eval.py::test_flat_vs_nested_equivalence_on_gold``) is therefore
env-gated/skipped (``SWE_ENV_RUN_REAL_CONTAINERS``); the *parser* unit tests on
recorded fixture logs DO run in CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from responses_api_agents.swe_env.grading import compute_resolved
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


# --- upstream SWE-bench eval-log sentinels (verbatim from
#     swebench/harness/constants/__init__.py so we never import swebench at
#     grade time). -----------------------------------------------------------
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

# Codes that mean the harness/patch/test setup failed before tests could be
# trusted; their presence forces an empty status map + patch_applied=False
# (mirrors swebench/harness/grading.py::get_logs_eval "bad_codes").
_BAD_CODES = (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT)

# Per-test status tokens a pytest-style test runner emits at the start of a line
# ("PASSED tests/test_x.py::test_a"). Verbatim from TestStatus in
# swebench/harness/constants. XFAIL counts as a pass (matches
# swebench/harness/grading.py::test_passed).
_PASS_TOKENS = ("PASSED", "XFAIL")
_FAIL_TOKENS = ("FAILED", "ERROR")
_STATUS_TOKENS = _PASS_TOKENS + _FAIL_TOKENS + ("SKIPPED",)

# Where the flat path writes the eval script + its captured log inside the
# sandbox. Distinct from the nested predictions/report paths.
EVAL_SCRIPT_PATH = "/root/eval.sh"
EVAL_LOG_PATH = "/root/eval_output.log"


def parse_eval_log(log: str) -> tuple[dict[str, str], bool]:
    """Parse a SWE-bench eval-script log host-side.

    Re-implements ``swebench/harness/grading.py::get_logs_eval`` for the common
    pytest-style runner *without* importing ``swebench``:

    1. If any "bad code" (patch-apply / reset / tests-error / timeout) is
       present, the run is untrustworthy -> return ``({}, False)``.
    2. If the ``Start``/``End`` test-output markers are missing, the test patch
       never applied -> return ``({}, False)``.
    3. Otherwise extract the slice between the markers and parse per-test
       ``"<STATUS> <node_id>"`` lines into a ``{node_id: STATUS}`` map. As a
       fallback (output sometimes escapes the markers, e.g. to stderr) we also
       scan the *whole* log when the slice yields nothing — mirroring upstream's
       second ``log_parser(content, ...)`` pass.

    Returns ``(status_map, patch_applied)``. ``patch_applied`` is ``True`` only
    when the markers were found and no bad code fired.
    """
    if any(code in log for code in _BAD_CODES):
        return {}, False
    if START_TEST_OUTPUT not in log or END_TEST_OUTPUT not in log:
        return {}, False

    between = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    status_map = _parse_pytest_status_lines(between)
    if not status_map:
        # Fallback: some runners emit per-test lines outside the markers.
        status_map = _parse_pytest_status_lines(log)
    return status_map, True


def _parse_pytest_status_lines(text: str) -> dict[str, str]:
    """Parse ``"<STATUS> <node_id>"`` pytest-style lines into a status map.

    Ports ``swebench/harness/log_parsers/python.py::parse_log_pytest``: a status
    line *starts* with one of the TestStatus tokens, then the node id is the
    second whitespace field. FAILED lines may read ``"FAILED <id> - <reason>"``;
    we strip the trailing reason exactly as upstream does (``" - "`` -> `" "``).
    """
    status_map: dict[str, str] = {}
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        token = next((t for t in _STATUS_TOKENS if line.startswith(t)), None)
        if token is None:
            continue
        if token == "FAILED":
            line = line.replace(" - ", " ")
        fields = line.split()
        if len(fields) <= 1:
            continue
        node_id = fields[1]
        # Don't let a later SKIPPED/duplicate clobber a recorded PASS/FAIL for
        # the same node; first decisive status wins.
        status_map.setdefault(node_id, fields[0])
    return status_map


def passed_tests(status_map: dict[str, str]) -> list[str]:
    """Node ids whose status counts as a pass (PASSED or XFAIL)."""
    return [node for node, status in status_map.items() if status in _PASS_TOKENS]


async def flat_run_eval(env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
    """Run the instance's eval script in the sandbox and capture its log.

    The eval script must be supplied on the task (built host-side, see
    :func:`flat_eval_enabled`'s docstring) via ``task.metadata["eval_script"]``.
    We write it into the sandbox, run it, and tee its combined output to
    :data:`EVAL_LOG_PATH`; the captured stdout/stderr already contain the
    ``>>>>>`` markers, so we grade off ``test_output`` directly. The log file is
    also read back as a robustness fallback when the streamed output is empty.
    """
    eval_script = task.metadata.get("eval_script", "")
    if not eval_script:
        # No script to run -> mask as an eval error rather than scoring 0.
        return EvalArtifacts(
            test_output="",
            return_code=1,
            patch_applied=False,
            raw={"error_type": "eval_error", "flat": True},
        )

    await env.write_text(EVAL_SCRIPT_PATH, eval_script if eval_script.endswith("\n") else eval_script + "\n")
    # The script is self-contained (it resets + applies patches + runs tests);
    # `|| true` keeps the captured log even on a non-zero test exit so grade()
    # can parse per-test status. Combined output is also tee'd to a log file.
    result = await env.execute(
        f"bash {EVAL_SCRIPT_PATH} 2>&1 | tee {EVAL_LOG_PATH}; exit ${{PIPESTATUS[0]}}",
        cwd=task.repo_workdir,
        is_eval=True,
        timeout_s=task.metadata.get("tests_timeout"),
    )
    log_text = result["output"]
    if not log_text.strip() and result.get("error_type") not in {"sandbox", "timeout"}:
        # Streamed output was empty; fall back to the tee'd log file.
        cat = await env.execute(f"cat {EVAL_LOG_PATH}", cwd=task.repo_workdir)
        if cat["returncode"] == 0:
            log_text = cat["output"]

    return EvalArtifacts(
        test_output=log_text,
        return_code=result["returncode"],
        patch_applied=bool(task.model_patch),
        raw={"error_type": result.get("error_type"), "flat": True},
    )


def flat_grade(task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
    """Host-side grade of a flat eval-script log (mirrors the flat families).

    Infra failures (sandbox/timeout) are masked via ``error_kind``. A log with a
    bad code or missing markers grades as unresolved with ``patch_applied`` set
    from the parse (this matches the flat families: a failed setup is a
    legitimate unresolved, not an infra mask).
    """
    if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            patch_applied=artifacts.patch_applied,
            error_kind=artifacts.raw["error_type"],
        )
    # A missing eval script is an eval error (masked), not a 0 score.
    if artifacts.raw.get("error_type") == "eval_error":
        return SweEvalReport(
            instance_id=task.instance_id,
            patch_exists=bool(task.model_patch),
            patch_applied=artifacts.patch_applied,
            error_kind="eval_error",
        )

    status_map, log_patch_applied = parse_eval_log(artifacts.test_output)
    passed = passed_tests(status_map)
    resolved = log_patch_applied and compute_resolved(
        fail_to_pass=task.fail_to_pass,
        pass_to_pass=task.pass_to_pass,
        passed=passed,
    )
    return SweEvalReport(
        instance_id=task.instance_id,
        resolved=resolved,
        patch_applied=log_patch_applied,
        patch_exists=bool(task.model_patch),
        tests_status={"passed": passed, "all": status_map},
    )


def flat_eval_enabled(harness_flag: bool, task: SweTask) -> bool:
    """Whether the flat mode should be used for this task.

    Flat mode is selected when the harness was constructed in flat mode
    (``harness_flag``) OR the task opts in via ``metadata["flat_eval"]``. The
    harness flag is what lifts the ``supports_provider`` apptainer-only gate (see
    module docstring); the per-task key only affects ``run_eval`` / ``grade``
    dispatch on an already-flat-capable harness.
    """
    return bool(harness_flag) or bool(task.metadata.get("flat_eval", False))
