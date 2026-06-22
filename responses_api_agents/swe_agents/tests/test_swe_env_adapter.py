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
from responses_api_agents.swe_agents.swe_env_adapter import (
    build_openhands_launch_command,
    openhands_config_toml,
    provision_and_extract_patch,
    run_self_driving,
)
from responses_api_agents.swe_env.harness import SweTask


_GOLD = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"

# Records the env of every spec a fake sandbox was created with (egress-injection assertions).
_CREATED_ENVS: list[dict] = []
# Records files staged into a fake sandbox (target paths) for stage_files assertions.
_UPLOADED_PATHS: list[str] = []


class _FakeProvider:
    name = "fake-adapter"

    def __init__(
        self,
        *,
        diff_output=_GOLD,
        test_output="PASSED test_calc.py::test_add\n",
        output_jsonl_patch=None,
        **_,
    ):
        self._diff = diff_output
        self._test_output = test_output
        # When set, the agent emits its patch via an OpenHands-style output.jsonl (not git diff).
        self._output_jsonl_patch = output_jsonl_patch

    async def create(self, spec):
        _CREATED_ENVS.append(dict(spec.env or {}))
        return SandboxHandle(sandbox_id="h", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if self._output_jsonl_patch is not None:
            if "find" in command and "output.jsonl" in command:
                return SandboxExecResult(stdout="/root/eval/x/output.jsonl\n", stderr="", return_code=0)
            if command.startswith("cat "):
                import json

                row = {"instance_id": "adapter-1", "test_result": {"git_patch": self._output_jsonl_patch}}
                return SandboxExecResult(stdout=json.dumps(row) + "\n", stderr="", return_code=0)
        if "git diff" in command:
            return SandboxExecResult(stdout=self._diff, stderr="", return_code=0)
        if "pytest" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, handle, source_path, target_path):
        _UPLOADED_PATHS.append(target_path)
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


def test_self_driving_extra_env_is_injected_into_sandbox():
    """OpenHands-style egress: NEMO_GYM_* vars must reach the agent sandbox verbatim."""
    clear_idempotency_cache()
    _CREATED_ENVS.clear()
    oh_env = {
        "NEMO_GYM_CONFIG_DICT": '{"head_server": {"host": "127.0.0.1", "port": 9099}}',
        "NEMO_GYM_MODEL_SERVER_NAME": "vllm_model",
        "NEMO_GYM_METRICS_FPATH": "/root/metrics.json",
    }
    asyncio.run(
        run_self_driving(
            _task(),
            provider={"fake-adapter": {}},
            agent_launch_command="bash run_infer.sh",
            extra_env=oh_env,
        )
    )
    # The agent sandbox (first created) carries the injected egress env.
    assert _CREATED_ENVS, "no sandbox created"
    agent_env = _CREATED_ENVS[0]
    for key, value in oh_env.items():
        assert agent_env.get(key) == value


def test_self_driving_patch_from_output_jsonl_is_verified():
    """OpenHands emits its patch via output.jsonl[test_result][git_patch], not git diff."""
    clear_idempotency_cache()
    out = asyncio.run(
        run_self_driving(
            _task(),
            provider={"fake-adapter": {"output_jsonl_patch": _GOLD}},
            agent_launch_command="bash run_infer.sh",
            patch_output_glob="/root/eval",
        )
    )
    assert out["model_patch"].startswith("--- a/calc.py")
    assert out["patch_exists"] is True
    assert out["resolved"] is True
    assert out["reward"] == 1.0


def test_self_driving_output_jsonl_missing_yields_empty_patch():
    clear_idempotency_cache()
    out = asyncio.run(
        run_self_driving(
            _task(),
            # output_jsonl_patch set but find returns a path; cat returns empty row patch
            provider={"fake-adapter": {"output_jsonl_patch": ""}},
            agent_launch_command="bash run_infer.sh",
            patch_output_glob="/root/eval",
        )
    )
    assert out["patch_exists"] is False
    assert out["resolved"] is False
    assert out["reward"] == 0.0


def test_provision_and_extract_patch_stages_files_and_returns_patch_without_verifying():
    """Agent-side primitive the worker uses: stage files, self-drive, return patch (NO grading)."""
    _UPLOADED_PATHS.clear()
    patch = asyncio.run(
        provision_and_extract_patch(
            _task(),
            provider={"fake-adapter": {"output_jsonl_patch": _GOLD}},
            agent_launch_command="bash run_infer.sh",
            extra_env={"NEMO_GYM_MODEL_SERVER_NAME": "vllm_model"},
            stage_files={"/root/config.toml": "[llm.model]\n", "/root/dataset/data.jsonl": "{}\n"},
            patch_output_glob="/root/eval",
        )
    )
    # Returns the patch (a plain str), runs no verification.
    assert isinstance(patch, str) and patch.startswith("--- a/calc.py")
    # Both staged files were written into the sandbox before launch.
    assert "/root/config.toml" in _UPLOADED_PATHS
    assert "/root/dataset/data.jsonl" in _UPLOADED_PATHS


def test_openhands_config_toml_uses_nonnative_fc():
    toml = openhands_config_toml("Qwen/Qwen2.5-Coder-3B-Instruct", temperature=0.0, top_p=1.0)
    assert "[llm.model]" in toml
    assert 'model = "Qwen/Qwen2.5-Coder-3B-Instruct"' in toml
    # non-native FC is the robust choice for small open models (validated)
    assert "native_tool_calling = false" in toml
    assert "log_completions_folder" in toml


def test_build_openhands_launch_command_has_runtime_local_egress_and_dataset():
    cmd = build_openhands_launch_command(
        setup_dir="/gym/responses_api_agents/swe_agents/swe_openhands_setup",
        instance_id="psf__requests-2317",
        dataset_name="SWE-Gym",
        split="test",
        ng_config_dict_quoted="'<<cfg>>'",
        model_server_name="vllm_model",
        agent_cls="CodeActAgent",
        max_iter=30,
    )
    # RUNTIME=local self-drive + the OpenHands runner
    assert "export RUNTIME=local" in cmd
    assert "run_infer.sh" in cmd
    # egress routes OpenHands' NemoGymClient back to the real model server
    assert "export NEMO_GYM_CONFIG_DICT='<<cfg>>'" in cmd
    assert "export NEMO_GYM_MODEL_SERVER_NAME=vllm_model" in cmd
    assert "NEMO_GYM_METRICS_FPATH" in cmd
    # dataset name selects OpenHands' workspace; instance + output dir wired
    assert "SWE-Gym test /root/eval_results psf__requests-2317" in cmd
    # git dubious-ownership guard for the host-owned bind mount under a root container
    assert "safe.directory '*'" in cmd
