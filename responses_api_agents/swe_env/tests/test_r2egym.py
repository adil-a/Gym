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

"""Unit tests for the r2e-gym nested harness, driven by a FakeSandbox provider.

r2e-gym is a nested-harness family: it cannot run on this box (no apptainer / no
real .sif), so these tests cover spec construction, the apptainer-only provider
gate, the agent-phase test-hiding command shape, and report parsing fed a
scripted ``report.json``. Real-instance validation is deferred to an apptainer
cluster.
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
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask
from responses_api_agents.swe_env.harnesses.r2egym import R2EGymHarness


class _FakeProvider:
    """Scripted provider: the eval command returns a canned rc; ``cat`` returns the report."""

    name = "fake-r2egym"

    def __init__(self, *, report_text="", eval_rc=0, **_):
        self._report_text = report_text
        self._eval_rc = eval_rc

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command.startswith("cat "):
            return SandboxExecResult(stdout=self._report_text, stderr="", return_code=0)
        if "run_local_evaluation.py" in command:
            return SandboxExecResult(stdout="eval done", stderr="", return_code=self._eval_rc)
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


register_provider("fake-r2egym", _FakeProvider, override=True)


def _task(**overrides) -> SweTask:
    base = dict(
        instance_id="r2e__pkg-42",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="r2e-gym",
    )
    base.update(overrides)
    return SweTask(**base)


def _report(instance_id: str, resolved: bool) -> str:
    return json.dumps(
        {
            instance_id: {
                "resolved": resolved,
                "tests_status": {"FAIL_TO_PASS": {"success": ["t::a"], "failure": []}},
            }
        }
    )


# ---- spec + provider gate ---------------------------------------------------


def test_harness_identity():
    harness = R2EGymHarness()
    assert harness.name == "r2e-gym"
    assert harness.grade_strategy == "nested-harness"


def test_build_spec_mounts_setup_dir():
    harness = R2EGymHarness()
    spec = harness.build_spec(_task(metadata={"r2egym_setup_dir": "/abs/setup"}))
    assert spec.image == "img:tag"
    assert spec.workdir == "/testbed"
    assert spec.metadata["instance_id"] == "r2e__pkg-42"
    assert spec.metadata["harness"] == "r2e-gym"
    mounts = spec.provider_options["mounts"]
    # Bind-mounted at both /r2egym_setup and its original absolute path.
    assert {"src": "/abs/setup", "dst": "/r2egym_setup"} in mounts
    assert {"src": "/abs/setup", "dst": "/abs/setup"} in mounts


def test_build_spec_truncates_long_instance_id():
    harness = R2EGymHarness()
    spec = harness.build_spec(_task(instance_id="x" * 100))
    assert len(spec.metadata["instance_id"]) == 63


def test_supports_provider_apptainer_only():
    harness = R2EGymHarness()
    assert harness.supports_provider("apptainer") is True
    assert harness.supports_provider("docker") is False
    assert harness.supports_provider("fake-r2egym") is False


def test_hide_eval_tests_commands_shape():
    harness = R2EGymHarness()
    commands = harness.hide_eval_tests_commands()
    # One command per checkout root (root, /root, /testbed).
    assert len(commands) == 3
    joined = " ".join(commands)
    assert "rm -rf /r2e_tests" in joined
    assert "rm -rf /root/r2e_tests" in joined
    assert "rm -rf /testbed/r2e_tests" in joined
    # Substring guard before deleting run_tests.sh.
    assert "grep -qs r2e_tests" in commands[0]


# ---- grade() over the nested report.json ------------------------------------


def test_grade_resolved_from_report():
    harness = R2EGymHarness()
    report = _report("r2e__pkg-42", resolved=True)
    out = harness.grade(_task(), EvalArtifacts(test_output=report, return_code=0, raw={"report_json": report}))
    assert out.resolved is True
    assert out.patch_exists is True
    assert reward_from_report(out) == 1.0


def test_grade_unresolved_from_report():
    harness = R2EGymHarness()
    report = _report("r2e__pkg-42", resolved=False)
    out = harness.grade(_task(), EvalArtifacts(test_output=report, return_code=0, raw={"report_json": report}))
    assert out.resolved is False
    assert reward_from_report(out) == 0.0


def test_grade_single_entry_fallback_on_key_mismatch():
    harness = R2EGymHarness()
    # Report keyed by a different id than the task; sole entry is used.
    report = _report("some-other-id", resolved=True)
    out = harness.grade(_task(), EvalArtifacts(test_output=report, return_code=0, raw={"report_json": report}))
    assert out.resolved is True


def test_grade_masks_on_infra_error():
    harness = R2EGymHarness()
    out = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout"}))
    assert out.error_kind == "timeout"
    assert reward_from_report(out) == 0.0


def test_grade_unparseable_report_is_eval_error():
    harness = R2EGymHarness()
    out = harness.grade(_task(), EvalArtifacts(test_output="not json", return_code=0, raw={"report_json": "not json"}))
    assert out.error_kind == "eval_error"
    assert reward_from_report(out) == 0.0


# ---- run_eval over the FakeSandbox ------------------------------------------


def _run_eval(report_text: str, eval_rc: int = 0) -> EvalArtifacts:
    async def _go() -> EvalArtifacts:
        harness = R2EGymHarness()
        task = _task()
        provider = {"fake-r2egym": {"report_text": report_text, "eval_rc": eval_rc}}
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        try:
            return await harness.run_eval(env, task)
        finally:
            await env.cleanup()

    return asyncio.run(_go())


def test_run_eval_then_grade_resolved():
    report = _report("r2e__pkg-42", resolved=True)
    artifacts = _run_eval(report)
    assert artifacts.return_code == 0
    assert artifacts.patch_applied is True
    out = R2EGymHarness().grade(_task(), artifacts)
    assert out.resolved is True


def test_run_eval_eval_failure_marks_not_applied():
    artifacts = _run_eval("", eval_rc=1)
    assert artifacts.return_code == 1
    assert artifacts.patch_applied is False
