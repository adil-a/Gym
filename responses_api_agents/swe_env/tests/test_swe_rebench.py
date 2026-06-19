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

"""Unit tests for the swe-rebench harness (FakeSandbox provider).

The nested SWE-rebench-V2 log parser is provisioned out-of-band on a cluster.
Here we stand up a tiny fake ``agent/log_parsers.py`` in a tmp dir so the real
``_load_rebench_log_parsers`` import + ``NAME_TO_PARSER`` resolution path is
exercised end to end, then drive resolved / unresolved / masked grade paths.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask
from responses_api_agents.swe_env.harnesses.swe_rebench import (
    SweRebenchHarness,
    _normalize_test_name,
)


class _FakeProvider:
    """Scripted provider: test command returns a canned transcript."""

    name = "fake-rebench"

    def __init__(self, *, test_output="", test_rc=0, apply_rc=0, **_):
        self._test_output = test_output
        self._test_rc = test_rc
        self._apply_rc = apply_rc

    async def create(self, spec):
        raw = {"workdir": spec.workdir, "env": spec.env}
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw=raw)

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "git apply" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=self._apply_rc)
        if "pytest" in command or "test" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=self._test_rc)
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


register_provider("fake-rebench", _FakeProvider, override=True)


# A standalone log_parsers module the harness will import dynamically. The
# parser splits "<node> <STATUS>" lines into {node: STATUS}; mirrors the shape
# of the real SWE-rebench-V2 parsers (a NAME_TO_PARSER registry + callables).
_FAKE_LOG_PARSERS = textwrap.dedent(
    """
    def parse_simple(log):
        results = {}
        for line in log.splitlines():
            line = line.strip()
            if not line:
                continue
            node, _, status = line.rpartition(" ")
            if node and status:
                results[node] = status
        return results

    NAME_TO_PARSER = {"simple": parse_simple}
    """
)


def _write_fake_parsers(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "SWE-rebench-V2"
    (repo_dir / "agent").mkdir(parents=True)
    (repo_dir / "agent" / "log_parsers.py").write_text(_FAKE_LOG_PARSERS)
    return repo_dir


def _task(**overrides) -> SweTask:
    base = dict(
        instance_id="rebench-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        test_command="python -m pytest -rA -q",
        model_patch="diff --git a/x b/x\n",
        test_patch="diff --git a/t b/t\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="swe-rebench",
    )
    base.update(overrides)
    return SweTask(**base)


# ---- pure helpers -----------------------------------------------------------


def test_normalize_test_name_strips_timing():
    assert _normalize_test_name("t::a [ 12 ms ]") == "t::a"
    assert _normalize_test_name("t::a [0.3s]") == "t::a"
    assert _normalize_test_name("t::a in 1.2 sec") == "t::a"
    assert _normalize_test_name("t::a (5 ms)") == "t::a"
    assert _normalize_test_name("  t::a  ") == "t::a"
    # No timing suffix -> unchanged.
    assert _normalize_test_name("pkg::mod::test_x") == "pkg::mod::test_x"


def test_build_spec_sets_java_env():
    harness = SweRebenchHarness()
    spec = harness.build_spec(_task())
    assert spec.env["_JAVA_OPTIONS"] == "-Djava.net.preferIPv6Addresses=false"
    assert spec.metadata["harness"] == "swe-rebench"
    assert spec.image == "img:tag"


# ---- grade paths (real dynamic-import of the fake parser) --------------------


def test_grade_resolved(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}},
    )
    # Both required tests pass; timing suffix on one exercises normalization.
    artifacts = EvalArtifacts(test_output="t::a [ 12 ms ] PASSED\nt::b PASSED\n", patch_applied=True)
    report = harness.grade(task, artifacts)
    assert report.resolved is True
    assert report.error_kind is None
    assert set(report.tests_status["passed"]) == {"t::a", "t::b"}


def test_grade_unresolved_missing_pass_to_pass(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}},
    )
    artifacts = EvalArtifacts(test_output="t::a PASSED\nt::b FAILED\n", patch_applied=True)
    report = harness.grade(task, artifacts)
    assert report.resolved is False
    assert report.error_kind is None


def test_grade_unresolved_when_patch_not_applied(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}},
    )
    artifacts = EvalArtifacts(test_output="t::a PASSED\nt::b PASSED\n", patch_applied=False)
    report = harness.grade(task, artifacts)
    assert report.resolved is False


def test_grade_masks_missing_clone():
    harness = SweRebenchHarness()
    # No rebench_repo_dir in metadata -> the clone is not provisioned.
    report = harness.grade(_task(), EvalArtifacts(test_output="t::a PASSED\n", patch_applied=True))
    assert report.error_kind == "eval_error"
    assert report.resolved is False


def test_grade_masks_unknown_parser(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "does_not_exist"}},
    )
    report = harness.grade(task, EvalArtifacts(test_output="t::a PASSED\n", patch_applied=True))
    assert report.error_kind == "eval_error"


def test_grade_masks_on_infra_error():
    harness = SweRebenchHarness()
    report = harness.grade(_task(), EvalArtifacts(test_output="", return_code=1, raw={"error_type": "timeout"}))
    assert report.error_kind == "timeout"


# ---- run_eval (FakeSandbox) -------------------------------------------------


def test_run_eval_then_grade_resolved(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(
        metadata={
            "rebench_repo_dir": str(repo_dir),
            "install_config": {"log_parser": "simple", "test_cmd": "python -m pytest -rA -q"},
        },
    )
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    provider = {"fake-rebench": {"test_output": "t::a PASSED\nt::b PASSED\n", "test_rc": 0}}

    async def _run():
        spec = harness.build_spec(task)
        env = await AsyncSweEnvironment.start(provider, spec)
        try:
            await harness.reset_repo(env, task)
            await harness.materialize(env, task)
            artifacts = await harness.run_eval(env, task)
        finally:
            await env.cleanup()
        return artifacts

    artifacts = asyncio.run(_run())
    assert artifacts.patch_applied is True
    report = harness.grade(task, artifacts)
    assert report.resolved is True


def test_run_eval_patch_not_applied(tmp_path):
    repo_dir = _write_fake_parsers(tmp_path)
    harness = SweRebenchHarness()
    task = _task(metadata={"rebench_repo_dir": str(repo_dir), "install_config": {"log_parser": "simple"}})
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    # apply_rc=1 -> model patch fails to apply -> patch_applied False -> unresolved.
    provider = {"fake-rebench": {"test_output": "t::a PASSED\nt::b PASSED\n", "apply_rc": 1}}

    async def _run():
        spec = harness.build_spec(task)
        env = await AsyncSweEnvironment.start(provider, spec)
        try:
            await harness.run_eval(env, task)
            return await harness.run_eval(env, task)
        finally:
            await env.cleanup()

    artifacts = asyncio.run(_run())
    assert artifacts.patch_applied is False
    assert harness.grade(task, artifacts).resolved is False
