# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from asyncio import Semaphore
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.benchflow_agent.app import (
    BenchFlowAgent,
    BenchFlowAgentConfig,
    BenchFlowRunRequest,
)
from responses_api_agents.benchflow_agent.utils import BenchFlowAgentUtils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GLOBAL_CONFIG = {
    "policy_model_name": "test_model",
    "test_model_server": {"responses_api_models": {"vllm_model": {"host": "policy-host", "port": 9000}}},
}


def _make_server(**config_overrides) -> BenchFlowAgent:
    """Create a BenchFlow agent server with test defaults (validation bypassed)."""
    defaults: Dict[str, Any] = dict(
        name="benchflow_agent",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        concurrency=1,
        model_server={"type": "responses_api_models", "name": "test_model_server"},
        tasks_dir="/tmp/test_tasks",
        jobs_dir="/tmp/test_jobs",
        images_dir="/images/build-123",
        task_config_overrides={"environment": {"memory_mb": 131072}, "agent": {"timeout_sec": 1000000000.0}},
    )
    defaults.update(config_overrides)
    config = BenchFlowAgentConfig(**defaults)
    return BenchFlowAgent.model_construct(
        config=config,
        server_client=MagicMock(),
        sem=Semaphore(config.concurrency),
    )


def _make_run_request(instance_id="skillsbench::3d-scan-calc", **kwargs) -> BenchFlowRunRequest:
    params: Dict[str, Any] = dict(temperature=1.0, top_p=1.0, input=[])
    params.update(kwargs)
    return BenchFlowRunRequest(
        instance_id=instance_id,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(**params),
    )


def _fake_result(**overrides) -> SimpleNamespace:
    """A stand-in for benchflow's RolloutResult (a plain object, not pydantic)."""
    defaults: Dict[str, Any] = dict(
        task_name="3d-scan-calc",
        rewards={"reward": 1.0},
        trajectory=[
            {"type": "agent_thought", "text": "thinking"},
            {"type": "agent_message", "text": "Working on it."},
            {
                "type": "tool_call",
                "tool_call_id": "tc_1",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "content": "file.txt",
            },
            {"type": "user_message", "text": "ignored"},
        ],
        error=None,
        error_category=None,
        verifier_error=None,
        n_tool_calls=1,
        n_input_tokens=100,
        n_output_tokens=50,
        n_cache_read_tokens=0,
        total_tokens=150,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_benchflow_module(evaluation_cls, captured) -> MagicMock:
    """Build a fake ``benchflow.evaluation`` module that records the args it receives."""

    class _FakeEvaluationConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class _FakeRetryConfig:
        def __init__(self, **kwargs):
            captured["retry_kwargs"] = kwargs

    module = MagicMock()
    module.Evaluation = evaluation_cls
    module.EvaluationConfig = _FakeEvaluationConfig
    module.RetryConfig = _FakeRetryConfig
    return module


def _fake_evaluation_cls(captured, fake_result, read_task=None):
    """A fake ``Evaluation`` that captures construction args and fires ``on_result``.

    When ``read_task`` is set, it reads ``<tasks_dir>/<read_task>/task.md`` at construction
    time (after the agent's copy+edit) so tests can assert on the edited file before the
    temp dir is cleaned up.
    """

    class _FakeEvaluation:
        def __init__(self, tasks_dir, jobs_dir, config, job_name, on_result):
            captured["tasks_dir"] = tasks_dir
            captured["jobs_dir"] = jobs_dir
            captured["job_name"] = job_name
            if read_task is not None:
                captured["edited_task_md"] = (Path(tasks_dir) / read_task / "task.md").read_text()
            self._on_result = on_result

        async def run(self):
            self._on_result("ignored", fake_result)

    return _FakeEvaluation


# ===========================================================================
#  run() — response building
# ===========================================================================


class TestRun:
    async def test_run_success(self):
        server = _make_server()
        with (
            patch("responses_api_agents.benchflow_agent.app.get_global_config_dict", return_value=_GLOBAL_CONFIG),
            patch.object(BenchFlowAgent, "_run_benchflow_task", new=AsyncMock(return_value=_fake_result())),
        ):
            response = await server.run(_make_run_request())

        assert response.reward == 1.0
        # agent_thought -> message, agent_message -> message, tool_call -> fc + fco; user_message skipped.
        types = [o.model_dump()["type"] for o in response.response.output]
        assert types == ["message", "message", "function_call", "function_call_output"]
        assert response.response.output[1].model_dump()["content"][0]["text"] == "Working on it."
        assert response.response.usage.total_tokens == 150
        assert response.response.model == "test_model"
        assert response.instance_id == "skillsbench::3d-scan-calc"
        assert response.responses_create_params.temperature == 1.0

    async def test_run_reward_zero_empty_trajectory(self):
        server = _make_server()
        result = _fake_result(rewards={"reward": 0.0}, trajectory=[])
        with (
            patch("responses_api_agents.benchflow_agent.app.get_global_config_dict", return_value=_GLOBAL_CONFIG),
            patch.object(BenchFlowAgent, "_run_benchflow_task", new=AsyncMock(return_value=result)),
        ):
            response = await server.run(_make_run_request())

        assert response.reward == 0.0
        assert len(response.response.output) == 0

    async def test_run_none_result(self):
        server = _make_server()
        with (
            patch("responses_api_agents.benchflow_agent.app.get_global_config_dict", return_value=_GLOBAL_CONFIG),
            patch.object(BenchFlowAgent, "_run_benchflow_task", new=AsyncMock(return_value=None)),
        ):
            response = await server.run(_make_run_request())

        assert response.reward == 0.0
        assert response.response.output == []

    async def test_run_failed_execution(self):
        server = _make_server()
        with (
            patch("responses_api_agents.benchflow_agent.app.get_global_config_dict", return_value=_GLOBAL_CONFIG),
            patch.object(BenchFlowAgent, "_run_benchflow_task", new=AsyncMock(side_effect=RuntimeError("boom"))),
        ):
            response = await server.run(_make_run_request(instance_id="skillsbench::fail", temperature=0.3))

        assert response.reward == 0.0
        assert response.response.output == []
        assert response.responses_create_params.temperature == 0.3
        data = response.model_dump()
        assert "boom" in data["metadata"]["error"]
        assert data["metadata"]["task_name"] == "fail"

    async def test_responses_not_implemented(self):
        server = _make_server()
        with pytest.raises(NotImplementedError):
            await server.responses(NeMoGymResponseCreateParamsNonStreaming(input=[]))

    def test_model_post_init_sets_semaphore(self):
        server = _make_server(concurrency=4)
        server.model_post_init(None)
        assert server.sem is not None


# ===========================================================================
#  _run_benchflow_task — temp-copy + edit, then run Evaluation (benchflow faked)
# ===========================================================================


class TestRunBenchflowTask:
    async def test_overrides_copy_edit_and_run(self, tmp_path):
        # Source task: a task.md with frontmatter we expect to be preserved + merged.
        task_md = "---\nenvironment:\n  allow_internet: false\n---\n\n## prompt\n\nDo the thing.\n"
        (tmp_path / "mytask").mkdir()
        (tmp_path / "mytask" / "task.md").write_text(task_md, encoding="utf-8")

        server = _make_server(
            tasks_dir=str(tmp_path),
            images_dir="/imgs",
            agent="openhands",
            max_retries=3,
            task_config_overrides={"environment": {"memory_mb": 131072}, "agent": {"timeout_sec": 9.0}},
        )
        captured: Dict[str, Any] = {}
        fake_result = _fake_result()
        fake_module = _fake_benchflow_module(_fake_evaluation_cls(captured, fake_result, read_task="mytask"), captured)
        gc = {**_GLOBAL_CONFIG, "policy_api_key": "EMPTY"}

        with (
            patch.dict("sys.modules", {"benchflow": MagicMock(), "benchflow.evaluation": fake_module}),
            patch("responses_api_agents.benchflow_agent.app.get_global_config_dict", return_value=gc),
        ):
            result = await server._run_benchflow_task("mytask", "test_model")

        assert result is fake_result
        # Evaluation ran against a temp copy, not the source tasks_dir, and the copy is gone.
        assert captured["tasks_dir"] != str(tmp_path)
        assert not Path(captured["tasks_dir"]).exists()
        # The overrides are applied to the copied task.md, NOT passed to EvaluationConfig.
        assert "task_config_overrides" not in captured["config_kwargs"]
        assert captured["config_kwargs"]["model"] == "hosted_vllm/test_model"
        assert captured["config_kwargs"]["environment"] == "singularity"
        assert captured["config_kwargs"]["agent"] == "openhands"
        assert captured["config_kwargs"]["concurrency"] == 1
        assert captured["config_kwargs"]["include_tasks"] == {"mytask"}
        assert captured["config_kwargs"]["agent_env"]["BENCHFLOW_PROVIDER_BASE_URL"] == "http://policy-host:9000/v1"
        assert captured["retry_kwargs"]["max_retries"] == 3
        assert captured["job_name"].startswith("mytask_")
        # The edited frontmatter: overrides merged, original keys + markdown body preserved.
        fm_text, body = BenchFlowAgentUtils._split_frontmatter(captured["edited_task_md"])
        fm = yaml.safe_load(fm_text)
        assert fm["environment"]["docker_image"] == "/imgs/mytask.sif"
        assert fm["environment"]["memory_mb"] == 131072
        assert fm["environment"]["allow_internet"] is False
        assert fm["agent"]["timeout_sec"] == 9.0
        assert "Do the thing." in body

    async def test_no_overrides_uses_source_tasks_dir(self):
        server = _make_server(images_dir=None, task_config_overrides=None)
        captured: Dict[str, Any] = {}
        fake_result = _fake_result()
        fake_module = _fake_benchflow_module(_fake_evaluation_cls(captured, fake_result), captured)
        gc = {**_GLOBAL_CONFIG, "policy_api_key": "EMPTY"}

        with (
            patch.dict("sys.modules", {"benchflow": MagicMock(), "benchflow.evaluation": fake_module}),
            patch("responses_api_agents.benchflow_agent.app.get_global_config_dict", return_value=gc),
        ):
            result = await server._run_benchflow_task("mytask", "test_model")

        assert result is fake_result
        # No overrides => no temp copy; Evaluation points straight at the source dir.
        assert captured["tasks_dir"] == server.config.tasks_dir
        assert "task_config_overrides" not in captured["config_kwargs"]


# ===========================================================================
#  Helper methods
# ===========================================================================


class TestHelpers:
    def test_parse_task_name(self):
        assert BenchFlowAgent._parse_task_name("skillsbench::3d-scan-calc") == "3d-scan-calc"
        assert BenchFlowAgent._parse_task_name("3d-scan-calc") == "3d-scan-calc"
        assert BenchFlowAgent._parse_task_name("  spaced-task  ") == "spaced-task"

    @pytest.mark.parametrize("instance_id", ["", "skillsbench::", "::"])
    def test_parse_task_name_rejects_empty(self, instance_id):
        with pytest.raises(ValueError, match="instance_id must contain a task name"):
            BenchFlowAgent._parse_task_name(instance_id)

    def test_build_task_config_overrides_injects_sif(self):
        server = _make_server(images_dir="/imgs/", task_config_overrides={"agent": {"timeout_sec": 5.0}})
        overrides = server._build_task_config_overrides("mytask")
        assert overrides["environment"]["docker_image"] == "/imgs/mytask.sif"
        assert overrides["agent"]["timeout_sec"] == 5.0

    def test_build_task_config_overrides_no_images_dir(self):
        server = _make_server(images_dir=None, task_config_overrides={"environment": {"memory_mb": 42}})
        overrides = server._build_task_config_overrides("mytask")
        assert overrides == {"environment": {"memory_mb": 42}}

    def test_build_task_config_overrides_empty(self):
        server = _make_server(images_dir=None, task_config_overrides=None)
        assert server._build_task_config_overrides("mytask") == {}

    def test_build_task_config_overrides_images_dir_wins(self):
        server = _make_server(
            images_dir="/imgs", task_config_overrides={"environment": {"docker_image": "ubuntu:22.04"}}
        )
        overrides = server._build_task_config_overrides("mytask")
        assert overrides["environment"]["docker_image"] == "/imgs/mytask.sif"

    def test_build_task_config_overrides_does_not_mutate_config(self):
        server = _make_server(images_dir="/imgs", task_config_overrides={"environment": {"memory_mb": 1}})
        server._build_task_config_overrides("t1")
        assert "docker_image" not in server.config.task_config_overrides["environment"]

    def test_build_agent_env(self):
        server = _make_server(agent_env={"DEBUG": "1"})
        gc = {**_GLOBAL_CONFIG, "policy_api_key": "KEY"}
        with patch("responses_api_agents.benchflow_agent.app.get_global_config_dict", return_value=gc):
            env = server._build_agent_env()
        assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "http://policy-host:9000/v1"
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "KEY"
        assert env["DEBUG"] == "1"

    def test_build_agent_env_default_api_key(self):
        server = _make_server()
        with patch("responses_api_agents.benchflow_agent.app.get_global_config_dict", return_value=_GLOBAL_CONFIG):
            env = server._build_agent_env()
        assert env["BENCHFLOW_PROVIDER_API_KEY"] == "EMPTY"

    def test_resolve_model_base_url(self):
        server = _make_server()
        assert server._resolve_model_base_url(_GLOBAL_CONFIG) == "http://policy-host:9000/v1"

    @pytest.mark.parametrize(
        "value, expected",
        [("3d-scan-calc", "3d-scan-calc"), ("a/b:c d", "a_b_c_d"), ("...", "task")],
    )
    def test_sanitize_path_component(self, value, expected):
        assert BenchFlowAgent._sanitize_path_component(value) == expected


# ===========================================================================
#  Utils — task.md editing
# ===========================================================================


class TestApplyTaskConfigOverrides:
    def test_merges_and_preserves_body(self, tmp_path):
        (tmp_path / "task.md").write_text(
            "---\nagent:\n  timeout_sec: 1\nenvironment:\n  allow_internet: false\n---\n\n# Title\n\nBody text.\n",
            encoding="utf-8",
        )
        BenchFlowAgentUtils.apply_task_config_overrides(
            tmp_path, {"environment": {"memory_mb": 99}, "agent": {"timeout_sec": 2}}
        )
        fm_text, body = BenchFlowAgentUtils._split_frontmatter((tmp_path / "task.md").read_text())
        data = yaml.safe_load(fm_text)
        assert data["agent"]["timeout_sec"] == 2  # overridden
        assert data["environment"]["memory_mb"] == 99  # added
        assert data["environment"]["allow_internet"] is False  # preserved
        assert "Body text." in body  # body preserved

    def test_creates_frontmatter_when_absent(self, tmp_path):
        (tmp_path / "task.md").write_text("# Just a body\n\nNo frontmatter here.\n", encoding="utf-8")
        BenchFlowAgentUtils.apply_task_config_overrides(tmp_path, {"environment": {"memory_mb": 7}})
        fm_text, body = BenchFlowAgentUtils._split_frontmatter((tmp_path / "task.md").read_text())
        assert yaml.safe_load(fm_text)["environment"]["memory_mb"] == 7
        assert "No frontmatter here." in body

    def test_empty_overrides_is_noop(self, tmp_path):
        original = "---\nagent:\n  timeout_sec: 1\n---\n\nBody.\n"
        (tmp_path / "task.md").write_text(original, encoding="utf-8")
        BenchFlowAgentUtils.apply_task_config_overrides(tmp_path, {})
        assert (tmp_path / "task.md").read_text() == original

    def test_missing_task_md_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="task.md"):
            BenchFlowAgentUtils.apply_task_config_overrides(tmp_path, {"agent": {"timeout_sec": 1}})


class TestSplitFrontmatter:
    def test_with_frontmatter(self):
        fm, body = BenchFlowAgentUtils._split_frontmatter("---\na: 1\n---\nbody\n")
        assert fm == "a: 1\n"
        assert body == "body\n"

    def test_no_frontmatter(self):
        fm, body = BenchFlowAgentUtils._split_frontmatter("just body\n")
        assert fm == ""
        assert body == "just body\n"

    def test_unclosed_fence(self):
        text = "---\na: 1\nno close\n"
        fm, body = BenchFlowAgentUtils._split_frontmatter(text)
        assert fm == ""
        assert body == text

    def test_empty(self):
        assert BenchFlowAgentUtils._split_frontmatter("") == ("", "")


class TestDeepMerge:
    def test_nested_merge_and_replace(self):
        base = {"a": {"x": 1, "y": 2}, "b": 1, "lst": [1, 2]}
        out = BenchFlowAgentUtils._deep_merge(base, {"a": {"y": 20, "z": 3}, "lst": [9]})
        assert out == {"a": {"x": 1, "y": 20, "z": 3}, "b": 1, "lst": [9]}
        assert base["a"] == {"x": 1, "y": 2}  # original not mutated


# ===========================================================================
#  Utils — response building
# ===========================================================================


class TestExtractReward:
    @pytest.mark.parametrize(
        "rewards, expected",
        [
            ({"reward": 1.0}, 1.0),
            ({"reward": 0.0}, 0.0),
            (None, 0.0),
            ({}, 0.0),
            ({"accuracy": 0.75}, 0.75),
            ("not-a-dict", 0.0),
        ],
    )
    def test_extract_reward(self, rewards, expected):
        assert BenchFlowAgentUtils.extract_reward(rewards) == expected


class TestTrajectoryToOutput:
    def test_converts_events(self):
        trajectory = [
            {"type": "agent_message", "text": "hello"},
            {"type": "agent_thought", "text": "hmm"},
            {
                "type": "tool_call",
                "tool_call_id": "tc1",
                "kind": "execute",
                "title": "ls",
                "status": "completed",
                "content": "out",
            },
            {"type": "user_message", "text": "skip"},
            {"type": "unknown", "text": "skip"},
            "not-a-dict",
        ]
        items = BenchFlowAgentUtils.trajectory_to_output(trajectory)
        assert [i["type"] for i in items] == ["message", "message", "function_call", "function_call_output"]
        assert items[0]["content"][0]["text"] == "hello"
        assert items[1]["content"][0]["text"] == "<think>hmm</think>"
        assert items[2]["call_id"] == "tc1"
        assert items[2]["name"] == "execute"
        assert items[3]["call_id"] == "tc1"
        assert items[3]["output"] == "out"

    def test_tool_call_without_content_has_no_output(self):
        trajectory = [{"type": "tool_call", "tool_call_id": "tc1", "kind": "read", "content": None}]
        items = BenchFlowAgentUtils.trajectory_to_output(trajectory)
        assert [i["type"] for i in items] == ["function_call"]

    def test_tool_call_content_list_is_serialized(self):
        trajectory = [{"type": "tool_call", "tool_call_id": "tc1", "kind": "read", "content": [{"a": 1}]}]
        items = BenchFlowAgentUtils.trajectory_to_output(trajectory)
        assert items[1]["type"] == "function_call_output"
        assert json.loads(items[1]["output"]) == [{"a": 1}]

    def test_tool_call_falls_back_to_title_then_tool(self):
        trajectory = [
            {"type": "tool_call", "title": "only-title", "content": None},
            {"type": "tool_call", "content": None},
        ]
        items = BenchFlowAgentUtils.trajectory_to_output(trajectory)
        assert items[0]["name"] == "only-title"
        assert items[1]["name"] == "tool"

    @pytest.mark.parametrize("trajectory", [None, [], "not-a-list"])
    def test_empty(self, trajectory):
        assert BenchFlowAgentUtils.trajectory_to_output(trajectory) == []


class TestExtractUsage:
    def test_with_tokens(self):
        result = SimpleNamespace(n_input_tokens=100, n_output_tokens=50, n_cache_read_tokens=5, total_tokens=155)
        usage = BenchFlowAgentUtils.extract_usage(result)
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 155
        assert usage["input_tokens_details"]["cached_tokens"] == 5

    def test_with_none_tokens(self):
        result = SimpleNamespace(n_input_tokens=None, n_output_tokens=None, n_cache_read_tokens=None, total_tokens=None)
        usage = BenchFlowAgentUtils.extract_usage(result)
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0


class TestGetDefaultResponseObject:
    def test_shape(self):
        obj = BenchFlowAgentUtils.get_default_response_object()
        assert obj["object"] == "response"
        assert obj["status"] == "completed"
        assert obj["id"].startswith("resp_")
        assert obj["usage"]["total_tokens"] == 0
