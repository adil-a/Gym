# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import responses_api_agents.swe_agents.app as swe_app
from nemo_gym.config_types import ModelServerRef, OmegaConf
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.swe_agents.app import (
    AgentPromptOverride,
    BaseDatasetHarnessProcessor,
    ExecuteContainerCommandArgs,
    NVInternalDatasetProcessor,
    R2EGymDatasetProcessor,
    RunOpenHandsAgent,
    SweBenchDatasetProcessor,
    SWEBenchMetrics,
    SWEBenchVerifyResponse,
    SWEBenchWrapper,
    SWEBenchWrapperConfig,
    SWEBenchWrapperInstanceConfig,
    SWEBenchWrapperServerConfig,
    SWERebenchDatasetProcessor,
    runner_ray_remote,
    update_metrics,
)


SWE_AGENTS_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _cleanup_swebench_results():
    """Remove swebench_results_* dirs that model_post_init creates in the source tree."""
    yield
    for d in SWE_AGENTS_DIR.glob("swebench_results_*"):
        shutil.rmtree(d, ignore_errors=True)


########################################
# Helpers
########################################


def _minimal_server_config() -> SWEBenchWrapperConfig:
    return SWEBenchWrapperConfig(
        host="localhost",
        port=9003,
        name="test_swe_agent",
        entrypoint="responses_api_agents/swe_agents",
        container_formatter=["docker://custom/{instance_id}"],
        swebench_tests_timeout=900,
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        concurrency=1,
    )


def _create_wrapper(monkeypatch) -> SWEBenchWrapper:
    """Create a SWEBenchWrapper with all setup calls mocked."""
    monkeypatch.setattr(swe_app, "get_global_config_dict", MagicMock(return_value=OmegaConf.create({})))
    monkeypatch.setattr(BaseDatasetHarnessProcessor, "_run_setup_command", MagicMock(return_value=None))

    config = _minimal_server_config()
    wrapper = SWEBenchWrapper(config=config, server_client=MagicMock(spec=ServerClient))
    return wrapper


def _make_instance_config(tmpdir: str, **overrides) -> SWEBenchWrapperInstanceConfig:
    """Build a minimal SWEBenchWrapperInstanceConfig for testing."""
    persistent_dir = Path(tmpdir) / "persistent"
    persistent_dir.mkdir(parents=True, exist_ok=True)
    base_mounted_dir = Path("/trajectories_mount")

    defaults = dict(
        host="localhost",
        port=9003,
        name="test_swe_agent",
        entrypoint="responses_api_agents/swe_agents",
        agent_framework="swe_agent",
        container_formatter=["docker://custom/{instance_id}"],
        swebench_tests_timeout=900,
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        concurrency=1,
        ng_global_config_dict_str="'{}'",
        model_server_name="test_model",
        openhands_setup_dir=Path(tmpdir) / "openhands",
        swebench_setup_dir=Path(tmpdir) / "swebench",
        swebench_multilingual_setup_dir=Path(tmpdir) / "swebench_multilingual",
        r2e_gym_setup_dir=Path(tmpdir) / "r2e",
        swe_rebench_setup_dir=Path(tmpdir) / "rebench",
        run_session_id="test_session",
        base_results_dir=Path(tmpdir) / "results",
        metrics_fpath=persistent_dir / "metrics.json",
        problem_info={
            "problem_statement": "Fix bug",
            "instance_id": "django__django-12345",
            "base_commit": "abc123",
            "dataset_name": "SWE-bench",
            "split": "test",
            "instance_dict": "{}",
            "container_formatter": ["docker://custom/{instance_id}"],
        },
        body=NeMoGymResponseCreateParamsNonStreaming(
            model="test-model",
            input=[],
            metadata={
                "problem_statement": "Fix bug",
                "instance_id": "django__django-12345",
                "base_commit": "abc123",
                "dataset_name": "SWE-bench",
                "split": "test",
                "instance_dict": "{}",
            },
        ),
        persistent_dir=persistent_dir,
        ray_queue_timestamp=time.time(),
        inference_params={"temperature": 1.0, "top_p": 1.0},
        agent_run_id="test_run_123",
        instance_dataset_path=persistent_dir / "data.jsonl",
        trajectories_root=persistent_dir / "trajectories" / "django__django-12345",
        prediction_path=persistent_dir / "output.jsonl",
        output_for_eval_mounted_path=base_mounted_dir / "output_for_eval.jsonl",
        output_for_eval_path=persistent_dir / "output_for_eval.jsonl",
        model_patch_path=persistent_dir / "patch.diff",
        container="/path/to/container.sif",
        eval_dir_in_openhands="evaluation/oh/test_run_123",
        openhands_config_file_path="/tmp/config_test.toml",
        agent_script_path=persistent_dir / "agent_script.sh",
        final_eval_apptainer_spinup_timestamp_fpath=persistent_dir / "final_eval_ts",
        final_eval_apptainer_spinup_timestamp_mounted_fpath=base_mounted_dir / "final_eval_ts",
        generation_apptainer_spinup_timestamp_fpath=persistent_dir / "gen_ts",
        generation_apptainer_spinup_timestamp_mounted_fpath=base_mounted_dir / "gen_ts",
        base_mounted_dir=base_mounted_dir,
        profiling_dir=persistent_dir / "profiling",
        profiling_mounted_dir=base_mounted_dir / "profiling",
    )
    defaults.update(overrides)
    return SWEBenchWrapperInstanceConfig(**defaults)


########################################
# Config model tests
########################################


class TestAgentPromptOverride:
    def test_defaults(self) -> None:
        override = AgentPromptOverride()
        assert override.user_prompt_template is None
        assert override.system_prompt_template is None
        assert override.agent_cls == "CodeActAgent"
        assert override.diversify_tool_names is False
        assert override.camel_case_tool_names is False

    def test_custom_values(self) -> None:
        override = AgentPromptOverride(
            user_prompt_template="/path/user.j2",
            system_prompt_template="/path/system.j2",
            agent_cls="CodexAgent",
            diversify_tool_names=True,
            camel_case_tool_names=True,
        )
        assert override.agent_cls == "CodexAgent"
        assert override.diversify_tool_names is True
        assert override.camel_case_tool_names is True

    def test_all_agent_cls_values(self) -> None:
        for cls in ["CodeActAgent", "OpenCodeAgent", "CodexAgent", "Terminus2Agent"]:
            override = AgentPromptOverride(agent_cls=cls)
            assert override.agent_cls == cls


class TestSWEBenchWrapperConfig:
    def test_default_values(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            model_server=ModelServerRef(type="responses_api_models", name="test"),
        )
        assert config.agent_config is None
        assert config.agent_tools_file is None
        assert config.agent_max_turns == 100
        assert config.swebench_tests_timeout == 30 * 60
        assert config.swebench_agent_timeout == 45 * 60
        assert config.apptainer_memory_limit_mb == 32 * 1024
        assert config.command_exec_timeout == 5 * 60
        assert config.concurrency == 256
        assert config.dataset_path is None
        assert config.agent_prompt_overrides is None
        assert config.agent_prompt_override_random is False
        assert config.openhands_should_log is False
        assert config.debug is False
        assert config.agent_framework_repo is None
        assert config.agent_framework_commit == "HEAD"

    def test_custom_values(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            agent_config="custom/config",
            agent_tools_file="tools.json",
            agent_max_turns=50,
            container_formatter=["docker://custom/{instance_id}"],
            swebench_tests_timeout=900,
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        )
        assert config.agent_config == "custom/config"
        assert config.agent_tools_file == "tools.json"
        assert config.agent_max_turns == 50
        assert config.container_formatter == ["docker://custom/{instance_id}"]
        assert config.swebench_tests_timeout == 900

    def test_multiple_container_formatters(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            container_formatter=[
                "docker://first/{instance_id}",
                "docker://second/{instance_id}",
            ],
            model_server=ModelServerRef(type="responses_api_models", name="test"),
        )
        assert len(config.container_formatter) == 2

    def test_string_container_formatter(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            container_formatter="docker://single/{instance_id}",
            model_server=ModelServerRef(type="responses_api_models", name="test"),
        )
        assert config.container_formatter == "docker://single/{instance_id}"

    def test_with_agent_prompt_overrides(self) -> None:
        config = SWEBenchWrapperConfig(
            host="localhost",
            port=9003,
            name="test_agent",
            entrypoint="responses_api_agents/swe_agents",
            model_server=ModelServerRef(type="responses_api_models", name="test"),
            agent_prompt_overrides=[
                AgentPromptOverride(agent_cls="CodeActAgent"),
                AgentPromptOverride(agent_cls="CodexAgent"),
            ],
        )
        assert len(config.agent_prompt_overrides) == 2


class TestSWEBenchWrapperServerConfig:
    def test_creation(self) -> None:
        config = SWEBenchWrapperServerConfig(
            ng_global_config_dict_str="'{}'",
            model_server_name="test_model",
            openhands_setup_dir=Path("/tmp/openhands"),
            swebench_setup_dir=Path("/tmp/swebench"),
            r2e_gym_setup_dir=Path("/tmp/r2e"),
            swe_rebench_setup_dir=Path("/tmp/rebench"),
            swebench_multilingual_setup_dir=Path("/tmp/swebench_ml"),
            run_session_id="test123",
            base_results_dir=Path("/tmp/results"),
        )
        assert config.model_server_name == "test_model"
        assert config.run_session_id == "test123"


class TestExecuteContainerCommandArgs:
    def test_creation(self) -> None:
        args = ExecuteContainerCommandArgs(
            command="echo hello",
            expected_file_pattern="/tmp/output.json",
            mode="agent",
            timeout=300,
        )
        assert args.command == "echo hello"
        assert args.mode == "agent"
        assert args.timeout == 300

    def test_eval_mode(self) -> None:
        args = ExecuteContainerCommandArgs(
            command="run_eval",
            expected_file_pattern="/tmp/report.json",
            mode="eval",
            timeout=600,
        )
        assert args.mode == "eval"


class TestSWEBenchWrapperInstanceConfig:
    def test_instance_id_property(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir)
            assert config.instance_id == "django__django-12345"

    def test_resolved_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_instance_config(tmpdir)
            assert config.resolved_user_prompt_template is None
            assert config.resolved_system_prompt_template is None
            assert config.resolved_agent_cls == "CodeActAgent"
            assert config.resolved_diversify_tool_names is False
            assert config.resolved_camel_case_tool_names is False


class TestSWEBenchMetrics:
    def test_defaults(self) -> None:
        metrics = SWEBenchMetrics()
        assert metrics.resolved is None
        assert metrics.patch_exists is None
        assert metrics.ray_queue_time is None
        assert metrics.openhands_run_time is None
        assert metrics.final_eval_time is None

    def test_with_values(self) -> None:
        metrics = SWEBenchMetrics(resolved=True, patch_exists=True, ray_queue_time=1.5)
        assert metrics.resolved is True
        assert metrics.ray_queue_time == 1.5


class TestSWEBenchVerifyResponse:
    def test_fields_exist(self) -> None:
        fields = SWEBenchVerifyResponse.model_fields
        assert "resolved" in fields
        assert "patch_exists" in fields
        assert "instance_config" in fields


########################################
# update_metrics tests
########################################


class TestUpdateMetrics:
    def test_basic_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "metrics.json"
            fpath.write_text(json.dumps({"a": 1, "b": 2}))

            update_metrics(fpath, {"b": 3, "c": 4})

            result = json.loads(fpath.read_text())
            assert result == {"a": 1, "b": 3, "c": 4}

    def test_none_values_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "metrics.json"
            fpath.write_text(json.dumps({"a": 1, "b": None}))

            update_metrics(fpath, {"c": None, "d": 5})

            result = json.loads(fpath.read_text())
            assert result == {"a": 1, "d": 5}
            assert "b" not in result
            assert "c" not in result


########################################
# BaseDatasetHarnessProcessor tests
########################################


class TestBaseDatasetHarnessProcessor:
    def test_parent_dir(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        assert processor.parent_dir == Path(swe_app.__file__).parent

    def test_setup_returns_none(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        assert processor.setup() is None

    def test_postprocess_after_run_returns_none(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        assert processor.postprocess_after_run(Path("/tmp/report.json")) is None

    def test_run_setup_command_success(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        processor._run_setup_command("true")

    def test_run_setup_command_failure(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        with pytest.raises(AssertionError, match="Command failed"):
            processor._run_setup_command("false")

    def test_setup_directory_lock(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir) / "target"
            setup_dir.mkdir()
            lock_path = setup_dir.parent / f".{setup_dir.name}.lockdir"
            with processor._setup_directory_lock(setup_dir, "test"):
                assert lock_path.exists()
            assert not lock_path.exists()

    def test_setup_directory_lock_stale_lock(self) -> None:
        config = _minimal_server_config()
        processor = BaseDatasetHarnessProcessor(config=config)
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dir = Path(tmpdir) / "target"
            setup_dir.mkdir()
            lock_path = setup_dir.parent / f".{setup_dir.name}.lockdir"
            lock_path.mkdir()
            # Make it appear stale by backdating mtime
            import os

            old_time = time.time() - 7200  # 2 hours ago
            os.utime(lock_path, (old_time, old_time))

            with processor._setup_directory_lock(setup_dir, "test"):
                pass  # should break the stale lock


########################################
# NVInternalDatasetProcessor tests
########################################


class TestNVInternalDatasetProcessor:
    def _make_processor(self, tmpdir, instance_dict_override=None) -> NVInternalDatasetProcessor:
        instance_dict = {
            "base_dockerfile": "ENV FOO=bar",
            "instance_dockerfile": "ENV BAZ=qux",
            "before_repo_set_cmd": "cd /app\npip install .",
            "selected_test_files_to_run": '["test_a.py", "test_b.py"]',
            "run_script.sh": "#!/bin/bash\npytest $1",
            "parsing_script.py": "import sys\nprint('done')",
            "base_commit": "abc123",
        }
        if instance_dict_override:
            instance_dict.update(instance_dict_override)

        config = _make_instance_config(
            tmpdir,
            problem_info={
                "problem_statement": "Fix bug",
                "instance_id": "nv__test-123",
                "base_commit": "abc123",
                "dataset_name": "nv-internal-1",
                "split": "test",
                "instance_dict": json.dumps(instance_dict),
                "container_formatter": ["docker://custom/{instance_id}"],
            },
        )
        return NVInternalDatasetProcessor(config=config)

    def test_check_tests_passed_all_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.check_tests_passed(
                {"tests": [{"name": "test_a", "status": "PASSED"}, {"name": "test_b", "status": "PASSED"}]},
                {"test_a"},
                {"test_b"},
            )
            assert result is True

    def test_check_tests_passed_some_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.check_tests_passed(
                {"tests": [{"name": "test_a", "status": "PASSED"}, {"name": "test_b", "status": "FAILED"}]},
                {"test_a"},
                {"test_b"},
            )
            assert result is False

    def test_check_tests_passed_empty_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            assert processor.check_tests_passed({}, set(), set()) is False
            assert processor.check_tests_passed(None, set(), set()) is False

    def test_check_tests_passed_no_passed_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.check_tests_passed(
                {"tests": [{"name": "test_a", "status": "FAILED"}]},
                {"test_a"},
                set(),
            )
            assert result is False

    def test_check_tests_passed_empty_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(tmpdir)
            result = processor.check_tests_passed(
                {"tests": [{"name": "test_a", "status": "PASSED"}]},
                set(),
                set(),
            )
            assert result is False

    def test_postprocess_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(
                tmpdir,
                {
                    "fail_to_pass": '["test_a"]',
                    "pass_to_pass": '["test_b"]',
                },
            )
            report_file = Path(tmpdir) / "report.json"
            report_file.write_text(
                json.dumps(
                    {
                        "tests": [
                            {"name": "test_a", "status": "PASSED"},
                            {"name": "test_b", "status": "PASSED"},
                        ]
                    }
                )
            )
            processor.postprocess_after_run(report_file)
            result = json.loads(report_file.read_text())
            assert processor.config.instance_id in result
            assert result[processor.config.instance_id]["resolved"] is True

    def test_postprocess_after_run_list_f2p(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._make_processor(
                tmpdir,
                {
                    "fail_to_pass_select": ["test_a"],
                    "pass_to_pass_select": ["test_b"],
                },
            )
            report_file = Path(tmpdir) / "report.json"
            report_file.write_text(
                json.dumps(
                    {
                        "tests": [
                            {"name": "test_a", "status": "PASSED"},
                            {"name": "test_b", "status": "FAILED"},
                        ]
                    }
                )
            )
            processor.postprocess_after_run(report_file)
            result = json.loads(report_file.read_text())
            assert result[processor.config.instance_id]["resolved"] is False


########################################
# SWERebenchDatasetProcessor tests
########################################


class TestSWERebenchDatasetProcessor:
    def test_normalize_test_name_timing_bracket(self) -> None:
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo [1.5ms]") == "test_foo"

    def test_normalize_test_name_timing_in(self) -> None:
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo in 200 msec") == "test_foo"

    def test_normalize_test_name_timing_paren(self) -> None:
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo (1.5s)") == "test_foo"

    def test_normalize_test_name_no_change(self) -> None:
        assert SWERebenchDatasetProcessor._normalize_test_name("test_bar") == "test_bar"

    def test_normalize_test_name_multiple_patterns(self) -> None:
        # Only the matching pattern should be removed
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo [2s]") == "test_foo"
        assert SWERebenchDatasetProcessor._normalize_test_name("test_foo [200ms]") == "test_foo"

    def test_postprocess_no_test_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_dict = {"install_config": {"log_parser": "pytest_parser"}}
            config = _make_instance_config(
                tmpdir,
                problem_info={
                    "problem_statement": "Fix",
                    "instance_id": "owner__repo-1",
                    "base_commit": "abc",
                    "dataset_name": "SWE-rebench",
                    "split": "test",
                    "instance_dict": json.dumps(instance_dict),
                    "container_formatter": ["/containers/{instance_id}.sif"],
                },
            )
            processor = SWERebenchDatasetProcessor(config=config)
            report_file = Path(tmpdir) / "report.json"
            report_file.write_text("{}")
            # test_output.log does not exist
            processor.postprocess_after_run(report_file)
            result = json.loads(report_file.read_text())
            assert result["owner__repo-1"]["resolved"] is False
            assert "No test output" in result["owner__repo-1"]["error"]


########################################
# SweBenchDatasetProcessor tests
########################################


class TestSweBenchDatasetProcessor:
    def test_setup_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _minimal_server_config()

            with patch.object(
                BaseDatasetHarnessProcessor,
                "parent_dir",
                new_callable=lambda: property(lambda self: Path(tmpdir)),
            ):
                setup_dir = Path(tmpdir) / "swe_swebench_setup"
                setup_dir.mkdir()
                swebench_dir = setup_dir / "SWE-bench"
                swebench_dir.mkdir()
                (setup_dir / "uv").mkdir()
                (setup_dir / "python").mkdir()

                processor = SweBenchDatasetProcessor(config=config)
                result = processor.setup()
                assert result == setup_dir


########################################
# runner_ray_remote tests
########################################


class TestRunnerRayRemote:
    def test_is_ray_remote(self) -> None:
        assert hasattr(runner_ray_remote, "remote")


########################################
# RunOpenHandsAgent tests
########################################


class TestRunOpenHandsAgent:
    def _make_agent(self, tmpdir, **overrides) -> RunOpenHandsAgent:
        config = _make_instance_config(tmpdir, **overrides)
        return RunOpenHandsAgent(config=config)

    def test_openhands_dir_copy_from_host_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            # Create required dirs
            eval_dir = Path(agent.config.openhands_setup_dir) / "OpenHands" / agent.config.eval_dir_in_openhands
            eval_dir.mkdir(parents=True, exist_ok=True)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            result = agent._openhands_dir_copy_from_host(output_file_path=None)
            assert result is None

    def test_openhands_dir_copy_from_host_with_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            eval_dir = Path(agent.config.openhands_setup_dir) / "OpenHands" / agent.config.eval_dir_in_openhands
            eval_dir.mkdir(parents=True, exist_ok=True)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            # Create an output file in the eval dir
            output_file = eval_dir / "output.jsonl"
            output_file.write_text('{"test": true}\n')

            agent.config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
            result = agent._openhands_dir_copy_from_host(output_file_path=str(output_file))
            assert result == str(agent.config.prediction_path)
            assert agent.config.prediction_path.exists()

    def test_openhands_dir_copy_from_host_relative_output_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            eval_dir = Path(agent.config.openhands_setup_dir) / "OpenHands" / agent.config.eval_dir_in_openhands
            eval_dir.mkdir(parents=True, exist_ok=True)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            # Create output.jsonl in a subdirectory matching the glob pattern
            sub_dir = eval_dir / "a" / "b" / "c"
            sub_dir.mkdir(parents=True)
            (sub_dir / "output.jsonl").write_text('{"data": 1}\n')

            agent.config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
            result = agent._openhands_dir_copy_from_host(output_file_path="nonexistent.jsonl")
            assert result is not None

    def test_openhands_dir_copy_no_output_file_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            eval_dir = Path(agent.config.openhands_setup_dir) / "OpenHands" / agent.config.eval_dir_in_openhands
            eval_dir.mkdir(parents=True, exist_ok=True)
            traj_root = agent.config.trajectories_root
            traj_root.mkdir(parents=True, exist_ok=True)

            agent.config.prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with pytest.raises(FileNotFoundError, match="No output.jsonl found"):
                agent._openhands_dir_copy_from_host(output_file_path="nonexistent.jsonl")


########################################
# SWEBenchWrapper tests
########################################


class TestSWEBenchWrapper:
    def test_model_post_init(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        assert wrapper._sem is not None
        assert wrapper._vllm_converter is not None
        assert wrapper._swe_bench_wrapper_server_config is not None
        assert wrapper._swe_bench_wrapper_server_config.run_session_id is not None

    def test_resolve_absolute_path_none(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        assert wrapper._resolve_absolute_path(None) is None
        assert wrapper._resolve_absolute_path("") is None

    def test_resolve_absolute_path_absolute(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        result = wrapper._resolve_absolute_path("/absolute/path/file.txt")
        assert result == "/absolute/path/file.txt"

    def test_resolve_absolute_path_relative(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        result = wrapper._resolve_absolute_path("relative/path/file.txt")
        assert result.endswith("relative/path/file.txt")
        assert Path(result).is_absolute()


class TestSWEBenchWrapperGetOpenhandsTrajectory:
    def test_with_completions(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_id = "django__django-12345"
            completions_dir = Path(tmpdir) / instance_id / "llm_completions" / instance_id
            completions_dir.mkdir(parents=True)

            completion_data = {
                "messages": [
                    {"content": [{"type": "text", "text": "system prompt"}], "role": "system"},
                    {"content": [{"type": "text", "text": "Fix the bug"}], "role": "user"},
                ],
                "provider_specific_fields": {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                },
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": "I'll fix it",
                                "role": "assistant",
                            }
                        }
                    ]
                },
                "kwargs": {"tools": [{"type": "function", "function": {"name": "execute_bash"}}]},
            }
            (completions_dir / "001_completion.json").write_text(json.dumps(completion_data))

            messages, tools = wrapper.get_openhands_trajectory_from_completions(Path(tmpdir), instance_id)
            assert len(messages) == 3  # system, user, assistant
            assert messages[2]["role"] == "assistant"
            assert messages[2]["prompt_token_ids"] == [1, 2]
            assert len(tools) == 1

    def test_no_completions_dir(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            messages, tools = wrapper.get_openhands_trajectory_from_completions(Path(tmpdir), "nonexistent")
            assert messages == []
            assert tools == []

    def test_no_completion_files(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_id = "test-instance"
            completions_dir = Path(tmpdir) / instance_id / "llm_completions" / instance_id
            completions_dir.mkdir(parents=True)

            messages, tools = wrapper.get_openhands_trajectory_from_completions(Path(tmpdir), instance_id)
            assert messages == []
            assert tools == []

    def test_assistant_with_no_content_or_tool_calls(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_id = "test-instance"
            completions_dir = Path(tmpdir) / instance_id / "llm_completions" / instance_id
            completions_dir.mkdir(parents=True)

            completion_data = {
                "messages": [{"role": "user", "content": "hello"}],
                "response": {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "role": "assistant",
                            }
                        }
                    ]
                },
                "kwargs": {},
            }
            (completions_dir / "001_completion.json").write_text(json.dumps(completion_data))

            messages, tools = wrapper.get_openhands_trajectory_from_completions(Path(tmpdir), instance_id)
            assert len(messages) == 1  # only user, assistant not appended


class TestSWEBenchWrapperSetupParams:
    def _setup_oh_dirs(self, wrapper):
        oh_dir = wrapper._swe_bench_wrapper_server_config.openhands_setup_dir / "OpenHands"
        for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
            (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
        miniforge = wrapper._swe_bench_wrapper_server_config.openhands_setup_dir / "miniforge3"
        miniforge.mkdir(parents=True, exist_ok=True)

    def test_basic_setup_params(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix bug",
                    "instance_id": "django__django-12345",
                    "base_commit": "abc123",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "django/django"}),
                },
            )

            params, processor = wrapper._setup_params(body)
            assert isinstance(params, SWEBenchWrapperInstanceConfig)
            assert isinstance(processor, SweBenchDatasetProcessor)
            assert params.instance_id == "django__django-12345"
            # #1249 A6: the legacy two-container commands are no longer built by _setup_params; the
            # decoupled verifier path owns launch + eval, so these fields stay None.
            assert params.eval_command is None
            assert params.agent_command is None
            assert params.eval_via_verifier is True
            assert params.metrics_fpath.exists()

    def test_setup_params_nv_internal(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "nv__test-1.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix",
                    "instance_id": "nv__test-1",
                    "base_commit": "abc",
                    "dataset_name": "nv-internal-1",
                    "split": "test",
                    "instance_dict": json.dumps(
                        {
                            "base_dockerfile": "",
                            "instance_dockerfile": "",
                            "before_repo_set_cmd": "",
                            "selected_test_files_to_run": "[]",
                            "run_script.sh": "#!/bin/bash",
                            "parsing_script.py": "print('ok')",
                            "base_commit": "abc",
                        }
                    ),
                },
            )

            params, processor = wrapper._setup_params(body)
            assert isinstance(processor, NVInternalDatasetProcessor)

    def test_setup_params_r2e_gym(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "repo_final_1.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix",
                    "instance_id": "org__Repo-1",
                    "base_commit": "abc",
                    "dataset_name": "R2E-Gym/R2E-Gym-Subset",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "org/Repo"}),
                },
            )

            params, processor = wrapper._setup_params(body)
            assert isinstance(processor, R2EGymDatasetProcessor)

    def test_setup_params_with_prompt_overrides(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        wrapper.config.agent_prompt_overrides = [
            AgentPromptOverride(agent_cls="CodexAgent"),
            AgentPromptOverride(agent_cls="OpenCodeAgent"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix",
                    "instance_id": "django__django-12345",
                    "base_commit": "abc",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "django/django"}),
                },
            )

            params, _ = wrapper._setup_params(body)
            # deterministic selection based on instance_id
            assert params.resolved_agent_cls in ["CodexAgent", "OpenCodeAgent"]


class TestSWEBenchWrapperResponses:
    def _setup_oh_dirs(self, wrapper):
        oh_dir = wrapper._swe_bench_wrapper_server_config.openhands_setup_dir / "OpenHands"
        for subdir in [".eval_sessions", "logs", "evaluation/oh"]:
            (oh_dir / subdir).mkdir(parents=True, exist_ok=True)
        miniforge = wrapper._swe_bench_wrapper_server_config.openhands_setup_dir / "miniforge3"
        miniforge.mkdir(parents=True, exist_ok=True)

    @pytest.mark.asyncio
    async def test_responses_success(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix bug",
                    "instance_id": "django__django-12345",
                    "base_commit": "abc123",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "django/django"}),
                },
            )

            mock_response = NeMoGymResponse(
                id="swebench-django__django-12345",
                created_at=123,
                model="test-model",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                metadata={
                    "input": "[]",
                    "metrics": json.dumps({"resolved": True}),
                    "instance_config": "{}",
                },
            )

            with patch.object(wrapper, "_inner_responses", new_callable=AsyncMock, return_value=mock_response):
                result = await wrapper.responses(body)
                assert result.id == "swebench-django__django-12345"

    @pytest.mark.asyncio
    async def test_responses_exception_writes_traceback(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        with tempfile.TemporaryDirectory() as tmpdir:
            container_file = Path(tmpdir) / "django__django-12345.sif"
            container_file.touch()

            wrapper.config.container_formatter = [str(Path(tmpdir) / "{instance_id}.sif")]
            self._setup_oh_dirs(wrapper)

            body = NeMoGymResponseCreateParamsNonStreaming(
                model="test-model",
                input=[],
                temperature=1.0,
                top_p=1.0,
                metadata={
                    "problem_statement": "Fix bug",
                    "instance_id": "django__django-12345",
                    "base_commit": "abc",
                    "dataset_name": "SWE-bench",
                    "split": "test",
                    "instance_dict": json.dumps({"repo": "django/django"}),
                },
            )

            with patch.object(
                wrapper,
                "_inner_responses",
                new_callable=AsyncMock,
                side_effect=RuntimeError("test error"),
            ):
                with pytest.raises(RuntimeError, match="test error"):
                    await wrapper.responses(body)


class TestSWEBenchWrapperRun:
    @pytest.mark.asyncio
    async def test_run_resolved(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)

        mock_response = NeMoGymResponse(
            id="swebench-test",
            created_at=123,
            model="test-model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            metadata={
                "input": "[]",
                "metrics": json.dumps({"resolved": True, "patch_exists": True}),
                "instance_config": _make_instance_config(tempfile.mkdtemp()).model_dump_json(),
            },
        )

        with patch.object(SWEBenchWrapper, "responses", new_callable=AsyncMock, return_value=mock_response):
            from nemo_gym.base_resources_server import BaseRunRequest

            body = BaseRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    model="test-model",
                    input=[],
                    metadata={
                        "problem_statement": "Fix",
                        "instance_id": "test-1",
                        "base_commit": "abc",
                        "dataset_name": "SWE-bench",
                        "split": "test",
                        "instance_dict": "{}",
                    },
                )
            )

            result = await wrapper.run(body)
            assert isinstance(result, SWEBenchVerifyResponse)
            assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_run_not_resolved(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)

        mock_response = NeMoGymResponse(
            id="swebench-test",
            created_at=123,
            model="test-model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            metadata={
                "input": "[]",
                "metrics": json.dumps({"resolved": False, "patch_exists": True}),
                "instance_config": _make_instance_config(tempfile.mkdtemp()).model_dump_json(),
            },
        )

        with patch.object(SWEBenchWrapper, "responses", new_callable=AsyncMock, return_value=mock_response):
            from nemo_gym.base_resources_server import BaseRunRequest

            body = BaseRunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    model="test-model",
                    input=[],
                    metadata={
                        "problem_statement": "Fix",
                        "instance_id": "test-1",
                        "base_commit": "abc",
                        "dataset_name": "SWE-bench",
                        "split": "test",
                        "instance_dict": "{}",
                    },
                )
            )

            result = await wrapper.run(body)
            assert isinstance(result, SWEBenchVerifyResponse)
            assert result.reward == 0.0


########################################
# _load_rebench_log_parsers tests
########################################


class TestLoadRebenchLogParsers:
    def test_loads_from_agent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rebench_dir = Path(tmpdir)
            agent_dir = rebench_dir / "agent"
            agent_dir.mkdir()
            (agent_dir / "log_parsers.py").write_text("NAME_TO_PARSER = {'test': lambda x: {}}\n")

            from responses_api_agents.swe_agents.app import _load_rebench_log_parsers

            mod = _load_rebench_log_parsers(rebench_dir)
            assert hasattr(mod, "NAME_TO_PARSER")
            assert "test" in mod.NAME_TO_PARSER

    def test_loads_from_lib_agent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rebench_dir = Path(tmpdir)
            lib_agent_dir = rebench_dir / "lib" / "agent"
            lib_agent_dir.mkdir(parents=True)
            (lib_agent_dir / "log_parsers.py").write_text("NAME_TO_PARSER = {'lib_test': lambda x: {}}\n")

            from responses_api_agents.swe_agents.app import _load_rebench_log_parsers

            mod = _load_rebench_log_parsers(rebench_dir)
            assert "lib_test" in mod.NAME_TO_PARSER


class TestDecoupledCutover:
    """#1249 run() cutover (eval_via_verifier): mask re-join parity + verifier POST contract."""

    def test_should_mask_sample_all_combinations(self) -> None:
        # resolved + clean agent finish -> NOT masked
        assert swe_app._should_mask_sample(True, None, False, False) is False
        # resolved but the agent hit max-turns / context window -> accidental reward, masked
        assert swe_app._should_mask_sample(True, "max_iteration", False, False) is True
        assert swe_app._should_mask_sample(True, "context_window", False, False) is True
        # resolved + a different agent error (stuck_in_loop) -> NOT masked on that arm
        assert swe_app._should_mask_sample(True, "stuck_in_loop", False, False) is False
        # eval timed out -> masked regardless of resolved
        assert swe_app._should_mask_sample(False, None, True, False) is True
        # agent timed out (wall-clock) -> masked regardless
        assert swe_app._should_mask_sample(False, None, False, True) is True
        # unresolved, clean -> NOT masked
        assert swe_app._should_mask_sample(False, None, False, False) is False

    @pytest.mark.asyncio
    async def test_verify_patch_via_server_builds_request_and_parses_subset(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        monkeypatch.setattr(swe_app, "raise_for_status", AsyncMock(return_value=None))
        monkeypatch.setattr(
            swe_app,
            "get_response_json",
            AsyncMock(return_value={"resolved": True, "error_kind": None, "patch_exists": True, "reward": 1.0}),
        )
        wrapper.server_client.post = AsyncMock(return_value=MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            instance_dict = {
                "base_commit": "abc123",
                "test_patch": "TP",
                "FAIL_TO_PASS": ["test_x.py::test_a"],
                "PASS_TO_PASS": ["test_x.py::test_b"],
            }
            params = _make_instance_config(
                tmpdir,
                eval_via_verifier=True,
                verifier_server_name="swe_verifier",
                container_formatter="docker://swebench/sweb.eval.x86_64.{instance_id}",
                problem_info={
                    "instance_id": "psf__requests-2317",
                    "base_commit": "abc123",
                    "dataset_name": "swe-bench-ext",
                    "split": "test",
                    "instance_dict": json.dumps(instance_dict),
                },
            )
            # the decoupled worker persists the patch into metrics before run() POSTs to the verifier
            params.metrics_fpath.write_text(json.dumps({"model_patch": "<<DIFF>>"}))

            subset = await wrapper._verify_patch_via_server(params)

        assert subset["resolved"] is True
        call = wrapper.server_client.post.call_args
        assert call.kwargs["server_name"] == "swe_verifier"
        assert call.kwargs["url_path"] == "/verify"
        req = call.kwargs["json"]
        assert req["response"]["metadata"]["model_patch"] == "<<DIFF>>"
        md = req["responses_create_params"]["metadata"]
        assert md["instance_id"] == "psf__requests-2317"
        # image resolved from the docker formatter (id munged); test_command carries the F2P+P2P ids
        assert md["image"] == "swebench/sweb.eval.x86_64.psf_1776_requests-2317"
        assert "test_x.py::test_a" in md["test_command"] and "test_x.py::test_b" in md["test_command"]
        assert md["benchmark"] == "swe-bench-ext"

    @pytest.mark.asyncio
    async def test_verify_patch_via_server_infra_error_is_masked_not_raised(self, monkeypatch) -> None:
        wrapper = _create_wrapper(monkeypatch)
        wrapper.server_client.post = AsyncMock(side_effect=RuntimeError("connreset"))
        with tempfile.TemporaryDirectory() as tmpdir:
            params = _make_instance_config(
                tmpdir,
                eval_via_verifier=True,
                verifier_server_name="swe_verifier",
                problem_info={
                    "instance_id": "psf__requests-2317",
                    "base_commit": "abc",
                    "dataset_name": "swe-bench-ext",
                    "split": "test",
                    "instance_dict": json.dumps({"FAIL_TO_PASS": [], "PASS_TO_PASS": []}),
                },
            )
            params.metrics_fpath.write_text(json.dumps({"model_patch": "<<DIFF>>"}))
            subset = await wrapper._verify_patch_via_server(params)
        # never raises; returns a masked subset so the agent still emits a present row (§4a)
        assert subset["resolved"] is False
        assert subset["error_kind"] == "sandbox"
        assert subset["patch_exists"] is True
