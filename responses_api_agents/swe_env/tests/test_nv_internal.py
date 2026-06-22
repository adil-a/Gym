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

"""Unit tests for the nv-internal-1 harness, driven by a FakeSandbox provider.

nv-internal-1 is flat + host-graded, so it runs on any exec-capable provider.
The scripted provider returns the parsing_script ``output.json`` report on the
``cat /root/output.json`` hop; grading is a pure host-side parse.
"""

from __future__ import annotations

import asyncio
import json

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.environment import AsyncSweEnvironment
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask
from responses_api_agents.swe_env.harnesses.nv_internal import (
    NVInternalHarness,
    _format_test_files,
    parse_passed_tests,
)


class _FakeProvider:
    """Scripted provider: ``cat /root/output.json`` returns a canned report."""

    name = "fake-nv"

    def __init__(self, *, report="", apply_rc=0, **_):
        self._report = report
        self._apply_rc = apply_rc

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "cat /root/output.json" in command:
            return SandboxExecResult(stdout=self._report, stderr="", return_code=0)
        if "git apply" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=self._apply_rc)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, *a, **k):
        return None

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        return None

    async def aclose(self):
        return None


register_provider("fake-nv", _FakeProvider, override=True)


def _task(**overrides) -> SweTask:
    base = dict(
        instance_id="nv-inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/app",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["pkg/test_x.py::a"],
        pass_to_pass=["pkg/test_x.py::b"],
        benchmark="nv-internal-1",
        metadata={
            "run_script": "echo run\n",
            "parsing_script": "import sys\n",
            "selected_test_files_to_run": ["pkg/test_x.py"],
        },
    )
    base.update(overrides)
    return SweTask(**base)


def _report(*passed, failed=()):
    tests = [{"name": name, "status": "PASSED"} for name in passed]
    tests += [{"name": name, "status": "FAILED"} for name in failed]
    return json.dumps({"tests": tests})


async def _run(provider_cfg, task) -> SweEvalReport:
    harness = NVInternalHarness()
    env = await AsyncSweEnvironment.start({"fake-nv": provider_cfg}, harness.build_spec(task))
    try:
        await harness.reset_repo(env, task)
        await harness.materialize(env, task)
        artifacts = await harness.run_eval(env, task)
    finally:
        await env.cleanup()
    return harness.grade(task, artifacts)


# ---- pure helpers -----------------------------------------------------------


def test_parse_passed_tests():
    report = {"tests": [{"name": "a", "status": "PASSED"}, {"name": "b", "status": "FAILED"}]}
    assert parse_passed_tests(report) == ["a"]
    assert parse_passed_tests({}) == []
    # Malformed entries are ignored, not crashed on.
    assert parse_passed_tests({"tests": ["junk", {"status": "PASSED"}]}) == []


def test_format_test_files():
    assert _format_test_files(["a", "b"]) == "a,b"
    assert _format_test_files('["a", "b"]') == "a,b"
    assert _format_test_files("a,b") == "a,b"
    assert _format_test_files(None) == ""


def test_build_spec():
    harness = NVInternalHarness()
    assert harness.name == "nv-internal-1"
    assert harness.grade_strategy == "flat-host-grade"
    spec = harness.build_spec(_task())
    assert spec.image == "img:tag"
    assert spec.workdir == "/app"
    assert spec.metadata["instance_id"] == "nv-inst-1"


def test_supports_any_provider():
    assert NVInternalHarness().supports_provider("docker") is True
    assert NVInternalHarness().supports_provider("apptainer") is True


def test_grade_masks_on_infra_error():
    harness = NVInternalHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout"}))
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_sandbox_error():
    harness = NVInternalHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "sandbox"}))
    assert report.error_kind == "sandbox"
    assert reward_from_report(report) == 0.0


def test_grade_empty_report_is_unresolved():
    harness = NVInternalHarness()
    # check_tests_passed: empty report → False (app.py:676-677).
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=0, patch_applied=True))
    assert report.resolved is False


def test_grade_malformed_report_is_unresolved():
    harness = NVInternalHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="not json", return_code=0, patch_applied=True))
    assert report.resolved is False


# ---- full reset -> materialize -> run_eval -> grade -------------------------


def test_resolved():
    report = _report("pkg/test_x.py::a", "pkg/test_x.py::b")
    result = asyncio.run(_run({"report": report}, _task()))
    assert result.patch_applied is True
    assert result.resolved is True
    assert reward_from_report(result) == 1.0


def test_unresolved_failing_required_test():
    # f2p test failed → unresolved.
    report = _report("pkg/test_x.py::b", failed=["pkg/test_x.py::a"])
    result = asyncio.run(_run({"report": report}, _task()))
    assert result.resolved is False
    assert reward_from_report(result) == 0.0


def test_unresolved_missing_required_test():
    # Only one required test present in the report → unresolved.
    report = _report("pkg/test_x.py::a")
    result = asyncio.run(_run({"report": report}, _task()))
    assert result.resolved is False


def test_patch_not_applied_is_unresolved():
    # Patch rejected (apply_rc != 0): even with all tests passing, unresolved.
    report = _report("pkg/test_x.py::a", "pkg/test_x.py::b")
    result = asyncio.run(_run({"report": report, "apply_rc": 1}, _task()))
    assert result.patch_applied is False
    assert result.resolved is False
