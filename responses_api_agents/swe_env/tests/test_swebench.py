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

"""Unit tests for the nested swe-bench / swe-bench-multilingual harness.

The nested families run the upstream ``run_local_evaluation`` harness inside an
apptainer sandbox; they cannot execute on this box (no apptainer + no real
``.sif``). These tests therefore validate provisioning (``build_spec`` /
``supports_provider`` / ``materialize``) and host-side ``grade`` parsing of a
sample ``report.json`` against a scripted ``FakeSandbox``. Real-instance
evaluation is deferred to an apptainer cluster.
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
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import EvalArtifacts, SweTask
from responses_api_agents.swe_env.harnesses.swebench import (
    _PREDICTIONS_PATH,
    _REPORT_PATH,
    SweBenchHarness,
)


class _FakeProvider:
    """Scripted provider: ``run_local_evaluation`` is a no-op, ``cat`` returns a report.

    Records uploaded text so ``materialize`` can be asserted.
    """

    name = "fake-swebench"

    def __init__(self, *, report_text="", report_rc=0, eval_rc=0, **_):
        self._report_text = report_text
        self._report_rc = report_rc
        self._eval_rc = eval_rc
        self.uploaded: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command.startswith("cat "):
            return SandboxExecResult(stdout=self._report_text, stderr="", return_code=self._report_rc)
        # run_local_evaluation + collect step.
        return SandboxExecResult(stdout="ran nested harness", stderr="", return_code=self._eval_rc)

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


register_provider("fake-swebench", _FakeProvider, override=True)


def _task(**overrides) -> SweTask:
    base = dict(
        instance_id="repo__inst-1",
        image="img:tag",
        base_commit="abc123",
        repo_workdir="/testbed",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="swe-bench",
        split="test",
    )
    base.update(overrides)
    return SweTask(**base)


def _sample_report(instance_id: str, resolved: bool) -> str:
    return json.dumps(
        {
            instance_id: {
                "resolved": resolved,
                "patch_is_None": False,
                "patch_successfully_applied": True,
                "tests_status": {"FAIL_TO_PASS": {"success": ["t::a"], "failure": []}},
            }
        }
    )


# ---- provisioning -----------------------------------------------------------


def test_grade_strategy_is_nested():
    assert SweBenchHarness("swe-bench").grade_strategy == "nested-harness"
    assert SweBenchHarness("swe-bench-multilingual").grade_strategy == "nested-harness"


def test_unknown_family_rejected():
    try:
        SweBenchHarness("not-a-family")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown family")


def test_build_spec_image_and_mounts():
    harness = SweBenchHarness("swe-bench")
    task = _task(metadata={"host_setup_dir": "/host/swe_swebench_setup"})
    spec = harness.build_spec(task)
    assert spec.image == "img:tag"
    assert spec.workdir == "/testbed"
    assert spec.metadata["instance_id"] == "repo__inst-1"
    assert spec.metadata["harness"] == "swe-bench"
    mounts = spec.metadata["mounts"]
    dsts = {m["dst"] for m in mounts}
    assert "/root/dataset/data.jsonl" in dsts
    # Host setup dir bind-mounted at both the alias and its canonical path.
    assert "/swebench_setup" in dsts
    assert "/host/swe_swebench_setup" in dsts


def test_build_spec_multilingual_mount_alias():
    harness = SweBenchHarness("swe-bench-multilingual")
    task = _task(benchmark="swe-bench-multilingual", metadata={"host_setup_dir": "/host/ml"})
    spec = harness.build_spec(task)
    dsts = {m["dst"] for m in spec.metadata["mounts"]}
    assert "/swebench_multilingual_setup" in dsts


def test_supports_provider_fail_fast_on_docker():
    harness = SweBenchHarness("swe-bench")
    assert harness.supports_provider("apptainer") is True
    assert harness.supports_provider("docker") is False
    assert harness.supports_provider("fake-swebench") is False


def test_materialize_writes_predictions_jsonl():
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    provider = {"fake-swebench": {}}

    async def run():
        harness = SweBenchHarness("swe-bench")
        task = _task()
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        # Reach into the underlying provider instance to inspect uploads.
        await harness.materialize(env, task)
        return env.sandbox._provider

    sandbox_provider = asyncio.run(run())
    assert _PREDICTIONS_PATH in sandbox_provider.uploaded
    prediction = json.loads(sandbox_provider.uploaded[_PREDICTIONS_PATH])
    assert prediction["instance_id"] == "repo__inst-1"
    assert prediction["model_patch"] == "diff --git a/x b/x\n"


# ---- grade (sample report.json) ---------------------------------------------


def test_grade_resolved_from_report():
    harness = SweBenchHarness("swe-bench")
    task = _task()
    artifacts = EvalArtifacts(
        test_output="ran",
        return_code=0,
        patch_applied=True,
        raw={"error_type": None, "report_json": _sample_report(task.instance_id, True)},
    )
    report = harness.grade(task, artifacts)
    assert report.resolved is True
    assert report.patch_applied is True
    assert report.patch_exists is True
    assert reward_from_report(report) == 1.0


def test_grade_unresolved_from_report():
    harness = SweBenchHarness("swe-bench")
    task = _task()
    artifacts = EvalArtifacts(raw={"error_type": None, "report_json": _sample_report(task.instance_id, False)})
    report = harness.grade(task, artifacts)
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_infra_error():
    harness = SweBenchHarness("swe-bench")
    report = harness.grade(_task(), EvalArtifacts(raw={"error_type": "timeout"}))
    assert report.error_kind == "timeout"
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_missing_report():
    harness = SweBenchHarness("swe-bench")
    report = harness.grade(_task(), EvalArtifacts(raw={"error_type": None, "report_json": ""}))
    assert report.error_kind == "eval_error"
    assert reward_from_report(report) == 0.0


# ---- run_eval (FakeSandbox: nested command issued, report read back) --------


def test_run_eval_reads_report_and_grades():
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment

    task = _task()
    report_text = _sample_report(task.instance_id, True)
    provider = {"fake-swebench": {"report_text": report_text}}

    async def run():
        harness = SweBenchHarness("swe-bench")
        env = await AsyncSweEnvironment.start(provider, harness.build_spec(task))
        await harness.materialize(env, task)
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts), artifacts

    report, artifacts = asyncio.run(run())
    assert artifacts.raw["report_json"] == report_text
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_run_eval_report_path_constant_is_stable():
    # The collect step copies the nested harness report to this fixed path.
    assert _REPORT_PATH == "/root/report.json"
