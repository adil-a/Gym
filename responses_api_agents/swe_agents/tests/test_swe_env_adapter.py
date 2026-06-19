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

"""SELF_DRIVING swe_env adapter for swe_agents: provision -> self-drive -> extract
patch -> score via the verifier, all through the decoupled swe_env infra."""

from __future__ import annotations

import asyncio

import responses_api_agents.swe_env.harnesses  # noqa: F401  (register harnesses)
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxStatus, register_provider
from resources_servers.swe_env.verify_task import clear_idempotency_cache
from responses_api_agents.swe_agents.swe_env_adapter import run_self_driving
from responses_api_agents.swe_env.harness import SweTask


_GOLD = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"


class _FakeProvider:
    name = "fake-adapter"

    def __init__(self, *, diff_output=_GOLD, test_output="PASSED test_calc.py::test_add\n", **_):
        self._diff = diff_output
        self._test_output = test_output

    async def create(self, spec):
        return SandboxHandle(sandbox_id="h", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if "git diff" in command:
            return SandboxExecResult(stdout=self._diff, stderr="", return_code=0)
        if "pytest" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=0)
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


register_provider("fake-adapter", _FakeProvider, override=True)


def _task() -> SweTask:
    return SweTask(
        instance_id="adapter-1",
        image="img:tag",
        base_commit="HEAD",
        repo_workdir="/testbed",
        test_command="python -m pytest -rA -q",
        fail_to_pass=["test_calc.py::test_add"],
        benchmark="swe-bench-ext",
    )


def test_self_driving_agent_patch_is_verified_resolved():
    clear_idempotency_cache()
    out = asyncio.run(
        run_self_driving(
            _task(),
            provider={"fake-adapter": {}},
            agent_launch_command="bash /openhands_setup/run_infer.sh",
            model_server={"model": "qwen"},
        )
    )
    assert out["model_patch"].startswith("--- a/calc.py")
    assert out["resolved"] is True
    assert out["reward"] == 1.0
    assert out["patch_exists"] is True
    assert out["mask_sample"] is False


def test_self_driving_no_patch_is_unresolved():
    clear_idempotency_cache()
    out = asyncio.run(
        run_self_driving(
            _task(),
            provider={"fake-adapter": {"diff_output": ""}},
            agent_launch_command="bash /openhands_setup/run_infer.sh",
        )
    )
    assert out["patch_exists"] is False
    assert out["resolved"] is False
    assert out["reward"] == 0.0
