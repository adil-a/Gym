# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Agent harness server for SWE-bench-style software-engineering evaluation tasks.

This module wires up a Responses API agent that runs OpenHands inside a per-task
sandbox to produce a code patch, then scores the patch by POSTing it to a separate
verifier server. It defines the agent configuration, per-instance configuration,
dataset/harness processors for the supported benchmark formats, the Ray worker that
drives a single task, and the FastAPI server that exposes the agent endpoints.
"""

import asyncio
import glob
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import sys
import time
import uuid
from asyncio import Semaphore
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from subprocess import Popen
from subprocess import run as subprocess_run
from traceback import format_exc
from typing import Any, Dict, Literal, Optional, Tuple, Union

import ray
from openai.types.responses.function_tool import FunctionTool
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import OmegaConf, get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_models.vllm_model.app import VLLMConverter, split_responses_input_output_items


########################################
# START Configuration
########################################


class AgentPromptOverride(BaseModel):
    """Prompt and tool-naming overrides applied to a single agent run."""

    user_prompt_template: Optional[str] = Field(
        default=None,
        description="Path to the user prompt template file",
    )
    system_prompt_template: Optional[str] = Field(
        default=None,
        description="Path to the system prompt template file",
    )
    agent_cls: Literal["CodeActAgent", "OpenCodeAgent", "CodexAgent", "Terminus2Agent"] = Field(
        default="CodeActAgent",
        description="Class to use for the agent",
    )
    diversify_tool_names: Optional[bool] = Field(
        default=False,
        description="If True, randomly select from tool names each run. If False, use the tool names in the order they are defined.",
    )
    camel_case_tool_names: Optional[bool] = Field(
        default=False,
        description="If True, convert tool names to camel case. If False, use the tool names as is.",
    )


class SWEBenchWrapperConfig(BaseResponsesAPIAgentConfig):
    """Server-level configuration for the SWE-bench agent harness."""

    model_server: ModelServerRef

    # Agent framework configuration
    agent_config: Optional[str] = Field(default=None, description="Path to agent configuration file")
    agent_tools_file: Optional[str] = Field(
        default=None, description="Path to JSON file containing tool definitions in OpenAI format (for SWE-agent)"
    )
    agent_max_turns: int = Field(default=100, description="Maximum iterations for the agent")
    agent_framework_repo: Optional[str] = Field(
        default=None,
        description="URL of the SWE-agent/OpenHands repo to pass to git clone. If None, will use the official repo",
    )

    agent_framework_commit: str = Field(
        default="HEAD", description="Which commit to use when cloning the SWE-agent/OpenHands repo"
    )
    # Container configuration
    container_formatter: str | list[str] = Field(
        default="docker://swebench/sweb.eval.x86_64.{instance_id}", description="Container path template"
    )
    swebench_tests_timeout: int = Field(default=30 * 60, description="Timeout for running tests (seconds)")

    swebench_agent_timeout: int = Field(default=45 * 60, description="Timeout for running the agent (seconds)")

    apptainer_memory_limit_mb: int = Field(
        default=32 * 1024, description="Memory limit for the apptainer container (MB)"
    )

    command_exec_timeout: int = Field(default=5 * 60, description="Timeout for executing the command (seconds)")

    # Concurrency control
    concurrency: int = Field(default=256, description="Maximum number of concurrent SWE-bench runs")

    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to the dataset for SWE-bench evaluation",
    )

    verify_golden_patch: bool = Field(
        default=False,
        description=(
            "If True, skip the agent run and use the sample's golden patch "
            "(instance_dict['patch']) as the model patch. The patch is graded via the "
            "decoupled verifier (the same /verify POST the agent path uses), so this "
            "verifies that the dataset sample actually resolves when its golden patch is "
            "applied. Currently supported for dataset_name == 'swe-bench-ext'."
        ),
    )

    agent_prompt_overrides: Optional[list[AgentPromptOverride]] = Field(
        default=None,
        description="List of (user_prompt_template, system_prompt_template, agent_cls) overrides. "
        "If multiple are provided, one is selected per instance_id (deterministic or random based on "
        "agent_prompt_override_random).",
    )
    agent_prompt_override_random: bool = Field(
        default=False,
        description="If True, randomly select from agent_prompt_overrides each run. "
        "If False (default), selection is deterministic per instance_id.",
    )

    openhands_should_log: bool = False
    debug: bool = False

    # Retained (default True) for config compatibility; there is a single eval path and no branch to gate.
    eval_via_verifier: bool = Field(
        default=True,
        description=(
            "Run OpenHands in a single working sandbox via the decoupled swe_env infra "
            "(acquire_sandbox + self-drive + output.jsonl patch extraction) and score the patch by "
            "POSTing to the swe_env verifier (verifier_server_name). This is the only supported eval "
            "path."
        ),
    )
    verifier_server_name: Optional[str] = Field(
        default=None,
        description="Name of the resources_servers/swe_env verifier to POST /verify to when eval_via_verifier=True.",
    )
    sandbox_provider: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Single-key swe_env sandbox provider mapping for the decoupled path "
        "(e.g. {'docker': {...}} or {'apptainer': {...}}). Defaults to apptainer when eval_via_verifier=True.",
    )


class SWEBenchWrapperServerConfig(BaseModel):
    """Per-run server state computed once at startup (session id, setup dirs, results dir)."""

    ng_global_config_dict_str: str
    model_server_name: str
    openhands_setup_dir: Path
    swebench_setup_dir: Path
    r2e_gym_setup_dir: Path
    swe_rebench_setup_dir: Path
    swebench_multilingual_setup_dir: Path
    run_session_id: str
    base_results_dir: Path


class ExecuteContainerCommandArgs(BaseModel):
    """Arguments describing a command to run inside a container and the file it produces."""

    command: str
    expected_file_pattern: str
    mode: Union[Literal["agent"], Literal["eval"]]
    timeout: int


class SWEBenchWrapperInstanceConfig(SWEBenchWrapperServerConfig, SWEBenchWrapperConfig):
    """Fully resolved configuration for a single task instance.

    Combines the server-level config and per-run server state with the per-instance
    problem info, paths, resolved prompt overrides, and timing inputs used by the worker.
    """

    metrics_fpath: Path
    problem_info: Dict[str, Any]
    body: NeMoGymResponseCreateParamsNonStreaming
    persistent_dir: Path
    ray_queue_timestamp: float
    inference_params: Dict[str, Any]
    agent_run_id: str
    instance_dataset_path: Path
    trajectories_root: Path
    prediction_path: Path
    output_for_eval_mounted_path: Path
    output_for_eval_path: Path
    model_patch_path: Path
    # Not populated; the image is resolved via _resolve_image_name. Kept Optional/None for config compatibility.
    container: Optional[str] = None
    eval_dir_in_openhands: str
    openhands_config_file_path: str
    agent_script_path: Path
    final_eval_apptainer_spinup_timestamp_fpath: Path
    final_eval_apptainer_spinup_timestamp_mounted_fpath: Path
    generation_apptainer_spinup_timestamp_fpath: Path
    generation_apptainer_spinup_timestamp_mounted_fpath: Path
    base_mounted_dir: Path
    profiling_dir: Path
    profiling_mounted_dir: Path

    # Resolved prompt override fields (selected from agent_prompt_overrides based on instance_id)
    resolved_user_prompt_template: Optional[str] = None
    resolved_system_prompt_template: Optional[str] = None
    resolved_agent_cls: str = "CodeActAgent"
    resolved_diversify_tool_names: Optional[bool] = False
    resolved_camel_case_tool_names: Optional[bool] = False

    # Optional fields that are not populated. Kept Optional/None to avoid config churn for callers that set them.
    eval_command: Optional[ExecuteContainerCommandArgs] = None
    eval_apptainer_command_str: Optional[str] = None
    agent_command: Optional[ExecuteContainerCommandArgs] = None
    agent_apptainer_command_str: Optional[str] = None
    agent_script: Optional[str] = None

    # GRPO related fields
    mask_sample: bool = False

    @property
    def instance_id(self) -> str:
        return self.problem_info["instance_id"]


class SWEBenchMetrics(BaseModel):
    """Per-task outcome and timing metrics persisted and reported for a run."""

    resolved: Optional[bool] = None
    patch_exists: Optional[bool] = None
    model_patch: Optional[str] = None

    # Failure-mode signals used to decide mask_sample downstream.
    # agent_error_kind is one of: "max_iteration", "context_window",
    # "stuck_in_loop", "other", or None if the agent finished cleanly.
    agent_error_kind: Optional[str] = None
    agent_timed_out: Optional[bool] = None
    eval_timed_out: Optional[bool] = None

    # Profiling time metrics to report
    ray_queue_time: Optional[float] = None
    openhands_run_time: Optional[float] = None
    generation_apptainer_spinup_time: Optional[float] = None
    create_runtime_time: Optional[float] = None
    connect_to_runtime_time: Optional[float] = None
    initialize_runtime_time: Optional[float] = None
    total_command_exec_time: Optional[float] = None
    total_model_call_time: Optional[float] = None
    final_eval_apptainer_spinup_time: Optional[float] = None
    final_eval_time: Optional[float] = None


class SWEBenchVerifyResponse(SWEBenchMetrics, BaseVerifyResponse):
    """Verify response combining the reward/response payload with task metrics and config."""

    instance_config: SWEBenchWrapperInstanceConfig


########################################
# START Dataset and harness handling
########################################


class BaseDatasetHarnessProcessor(BaseModel):
    """Base class for dataset- and harness-specific setup and post-run processing."""

    config: SWEBenchWrapperConfig | SWEBenchWrapperInstanceConfig

    ########################################
    # START Setup logic
    ########################################

    @property
    def parent_dir(self) -> Path:
        return Path(__file__).parent

    def _run_setup_command(self, command: str) -> None:
        """Run a setup shell command and assert it exits successfully.

        Args:
            command: The shell command to execute.
        """
        process = Popen(command, shell=True)
        return_code = process.wait()
        assert return_code == 0, f"Command failed: {command}"

    @contextmanager
    def _setup_directory_lock(self, setup_dir: Path, label: str):
        """Acquire a cross-node directory lock for the duration of the context.

        Uses an atomic ``mkdir`` as the lock primitive so it works on shared filesystems
        where advisory file locks are unreliable. Polls until the lock is acquired, breaking
        a stale lock that is older than the threshold, and removes the lock on exit.

        Args:
            setup_dir: The directory whose setup is being guarded; the lock lives beside it.
            label: Human-readable name used in log messages.

        Yields:
            None: Control once the lock is held.
        """
        lock_dir = setup_dir.parent
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f".{setup_dir.name}.lockdir"

        print(f"Acquiring {label} setup lock at {lock_path}", flush=True)
        max_wait = 3600
        poll_interval = 5
        waited = 0
        while True:
            try:
                lock_path.mkdir(exist_ok=False)
                break
            except FileExistsError:
                stale_threshold = 3600
                try:
                    lock_age = time.time() - lock_path.stat().st_mtime
                    if lock_age > stale_threshold:
                        print(f"  Lock appears stale ({lock_age:.0f}s old), breaking it", flush=True)
                        shutil.rmtree(lock_path, ignore_errors=True)
                        continue
                except OSError:
                    pass
                if waited >= max_wait:
                    raise TimeoutError(f"Timed out waiting for {label} setup lock after {max_wait}s")
                if waited % 30 == 0:
                    print(
                        f"  Waiting for {label} setup lock (held by another process, {waited}s elapsed)...", flush=True
                    )
                time.sleep(poll_interval)
                waited += poll_interval
        try:
            yield
        finally:
            shutil.rmtree(lock_path, ignore_errors=True)

    # Setup method is sync since there's been no need to concurrently set up.
    def setup(self) -> Path:
        """Set up the dataset or harness for this processor.

        Returns:
            Path: The setup directory. The base implementation does nothing.
        """
        pass

    def postprocess_after_run(self, report_file: Path) -> None:
        """Post-process the run output, typically producing or rewriting the report file.

        Args:
            report_file: Path to the report file to read and/or write. The base
                implementation does nothing.
        """
        pass


class SweBenchDatasetProcessor(BaseDatasetHarnessProcessor):
    """Dataset processor for SWE-bench tasks."""

    def setup(self) -> Path:
        """Clone and install the SWE-bench harness under a locked setup directory.

        Returns:
            Path: The setup directory containing the prepared SWE-bench environment.
        """
        swebench_repo = "https://github.com/HeyyyyyyG/SWE-bench.git"
        swebench_commit = "HEAD"

        setup_dir = self.parent_dir / "swe_swebench_setup"
        setup_dir.mkdir(parents=True, exist_ok=True)

        with self._setup_directory_lock(setup_dir, "SWE-bench"):
            swebench_dir = setup_dir / "SWE-bench"
            uv_dir = setup_dir / "uv"
            python_dir = setup_dir / "python"

            if swebench_dir.exists():
                print(f"SWE-bench already set up at {setup_dir}")
                return setup_dir

            print(f"Setting up SWE-bench environment at {setup_dir}...", flush=True)
            script_fpath = self.parent_dir / "setup_scripts/swebench.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
SWEBENCH_DIR={swebench_dir} \\
SWEBENCH_REPO={swebench_repo} \\
SWEBENCH_COMMIT={swebench_commit} \\
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir


class SweBenchMultilingualDatasetProcessor(BaseDatasetHarnessProcessor):
    """Dataset processor for SWE-bench Multilingual tasks."""

    def setup(self) -> Path:
        """Clone and install the SWE-bench Multilingual harness under a locked setup directory.

        Returns:
            Path: The setup directory containing the prepared environment.
        """
        swebench_repo = "https://github.com/Kipok/SWE-bench.git"
        swebench_commit = "HEAD"

        setup_dir = self.parent_dir / "swe_swebench_multilingual_setup"
        setup_dir.mkdir(parents=True, exist_ok=True)

        with self._setup_directory_lock(setup_dir, "SWE-bench_Multilingual"):
            swebench_multilingual_dir = setup_dir / "SWE-bench_Multilingual"
            uv_dir = setup_dir / "uv"
            python_dir = setup_dir / "python"

            if swebench_multilingual_dir.exists():
                print(f"SWE-bench_Multilingual already set up at {setup_dir}")
                return setup_dir

            print(f"Setting up SWE-bench_Multilingual environment at {setup_dir}...", flush=True)
            script_fpath = self.parent_dir / "setup_scripts/swebench_multilingual.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
SWEBENCH_DIR={swebench_multilingual_dir} \\
SWEBENCH_REPO={swebench_repo} \\
SWEBENCH_COMMIT={swebench_commit} \\
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir


class R2EGymDatasetProcessor(BaseDatasetHarnessProcessor):
    """Dataset processor for R2E-Gym tasks."""

    def setup(self) -> Path:
        """Clone and install the R2E-Gym harness under a locked setup directory.

        Verifies an existing install by importing ``r2egym`` and rebuilds if missing.

        Returns:
            Path: The setup directory containing the prepared R2E-Gym environment.
        """
        eval_harness_repo = "https://github.com/sdevare-nv/nv-R2E-Gym.git"
        eval_harness_commit = "local-eval"

        setup_dir = self.parent_dir / "swe_r2e_gym_setup"

        with self._setup_directory_lock(setup_dir, "R2E-Gym"):
            r2e_gym_dir = setup_dir / "R2E-Gym"
            uv_dir = setup_dir / "uv"
            python_dir = setup_dir / "python"

            # Check if setup is complete by verifying venv and installed module
            venv_dir = r2e_gym_dir / "venv"
            python_bin = venv_dir / "bin" / "python"
            if r2e_gym_dir.exists() and venv_dir.exists() and python_bin.exists():
                result = subprocess_run([str(python_bin), "-c", "import r2egym"])
                if result.returncode == 0:
                    print(f"R2E-Gym already set up at {setup_dir}", flush=True)
                    return setup_dir

                print("R2E-Gym directory exists but module not properly installed, rebuilding...", flush=True)

            print(f"Setting up R2E-Gym environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = self.parent_dir / "setup_scripts/r2e_gym.sh"
            command = f"""SETUP_DIR={setup_dir} \\
UV_DIR={uv_dir} \\
PYTHON_DIR={python_dir} \\
R2E_GYM_DIR={r2e_gym_dir} \\
EVAL_HARNESS_REPO={eval_harness_repo} \\
EVAL_HARNESS_COMMIT={eval_harness_commit} \\
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir


class NVInternalDatasetProcessor(BaseDatasetHarnessProcessor):
    """Dataset processor for the nv-internal dataset format."""

    def postprocess_after_run(self, report_file: Path) -> None:
        """Grade the test results and overwrite the report file with a resolution summary.

        Reads the fail-to-pass and pass-to-pass test sets from the instance, checks whether
        they all passed, and writes a report keyed by instance id.

        Args:
            report_file: Path to the report file containing raw test results; rewritten in place.
        """
        instance_dict = json.loads(self.config.problem_info["instance_dict"])

        fail_to_pass_str = instance_dict.get("fail_to_pass_select", instance_dict.get("fail_to_pass", "[]"))
        pass_to_pass_str = instance_dict.get("pass_to_pass_select", instance_dict.get("pass_to_pass", "[]"))

        if isinstance(fail_to_pass_str, str):
            f2p = set(json.loads(fail_to_pass_str))
        else:
            f2p = set(fail_to_pass_str)

        if isinstance(pass_to_pass_str, str):
            p2p = set(json.loads(pass_to_pass_str))
        else:
            p2p = set(pass_to_pass_str)

        with open(report_file, "r+") as f:
            test_results = json.loads(f.read())
            is_resolved = self.check_tests_passed(
                test_results,
                f2p,
                p2p,
            )
            report_dict = dict(
                resolved=is_resolved,
                patch_exists=True,
                patch_successfully_applied=is_resolved,
                metadata={
                    "test_results": test_results,
                    "f2p": list(f2p),
                    "p2p": list(p2p),
                },
            )
            f.seek(0)
            f.write(json.dumps({self.config.instance_id: report_dict}, indent=4))

    def check_tests_passed(
        self,
        test_results: dict[str, Any],
        f2p: set[str],
        p2p: set[str],
    ) -> bool:
        """Check whether every required test passed.

        Args:
            test_results: Parsed test results, a mapping with a ``tests`` list of
                ``{"name", "status"}`` entries.
            f2p: Set of fail-to-pass test names that must pass.
            p2p: Set of pass-to-pass test names that must pass.

        Returns:
            bool: True if all required tests passed, otherwise False.
        """
        if not test_results:
            return False

        passed_tests = {test["name"] for test in test_results.get("tests", []) if test.get("status") == "PASSED"}
        required_tests = f2p.union(p2p)

        # Check if all required tests passed
        if len(passed_tests) == 0 or len(required_tests) == 0:
            return False

        return required_tests <= passed_tests


def _load_rebench_log_parsers(rebench_repo_dir: Path):
    """Dynamically import the SWE-rebench log-parsers module from a checked-out repo.

    Temporarily prepends the repo directories to ``sys.path`` so the module's own imports
    resolve, loads it from its file location, and restores ``sys.path`` afterward.

    Args:
        rebench_repo_dir: Path to the checked-out SWE-rebench repository.

    Returns:
        module: The loaded log-parsers module.
    """
    lp_path = rebench_repo_dir / "lib" / "agent" / "log_parsers.py"
    if not lp_path.exists():
        lp_path = rebench_repo_dir / "agent" / "log_parsers.py"

    extra_paths = [str(rebench_repo_dir), str(rebench_repo_dir / "lib")]
    added: list[str] = []
    for p in extra_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    try:
        spec = importlib.util.spec_from_file_location("_rebench_log_parsers", str(lp_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


class SWERebenchDatasetProcessor(BaseDatasetHarnessProcessor):
    """Dataset processor for SWE-rebench tasks."""

    def setup(self) -> Path:
        """Clone the SWE-rebench repository under a locked setup directory.

        Returns:
            Path: The setup directory containing the prepared SWE-rebench environment.
        """
        setup_dir = self.parent_dir / "swe_rebench_setup"

        with self._setup_directory_lock(setup_dir, "SWE-rebench"):
            rebench_dir = setup_dir / "SWE-rebench-V2"

            if rebench_dir.exists() and (rebench_dir / "agent" / "log_parsers.py").exists():
                print(f"SWE-rebench-V2 already set up at {setup_dir}", flush=True)
                return setup_dir

            print(f"Setting up SWE-rebench-V2 environment at {setup_dir}...", flush=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = self.parent_dir / "setup_scripts/swe_rebench.sh"
            command = f"""SETUP_DIR={setup_dir} \
REBENCH_DIR={rebench_dir} \
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir

    @staticmethod
    def _normalize_test_name(name: str) -> str:
        """Strip trailing timing annotations from a test name and trim whitespace.

        Args:
            name: The raw test name as parsed from the test output.

        Returns:
            str: The normalized test name.
        """
        _REBENCH_TIMING_NORMALIZE_RES = [
            re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
            re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
            re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
        ]
        for pattern in _REBENCH_TIMING_NORMALIZE_RES:
            name = pattern.sub("", name)
        return name.strip()

    def postprocess_after_run(self, report_file: Path) -> None:
        """Parse SWE-rebench test output on the host and write a resolution report.

        Loads the instance-specific log parser, normalizes the parsed test names, compares
        them against the expected fail-to-pass and pass-to-pass sets, and writes a report
        keyed by instance id. Parsing on the host avoids requiring Python inside the container.

        Args:
            report_file: Path to the report file to write; the sibling ``test_output.log``
                in the same directory supplies the raw test output.
        """
        report_path = Path(report_file)
        test_output_path = report_path.parent / "test_output.log"

        instance_id = self.config.instance_id
        instance_dict = json.loads(self.config.problem_info["instance_dict"])
        install_config = instance_dict.get("install_config", {})
        log_parser_name = install_config.get("log_parser", "")

        if not test_output_path.exists():
            report = {
                instance_id: {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": False,
                    "error": "No test output produced inside container",
                }
            }
            report_path.write_text(json.dumps(report, indent=2))
            return

        setup_dir = self.parent_dir / "swe_rebench_setup"
        log_parsers = _load_rebench_log_parsers(setup_dir / "SWE-rebench-V2")

        parser = log_parsers.NAME_TO_PARSER.get(log_parser_name) or getattr(log_parsers, log_parser_name, None)
        if parser is None:
            report = {
                instance_id: {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "error": f"Unknown log parser: {log_parser_name}",
                }
            }
            report_path.write_text(json.dumps(report, indent=2))
            return

        test_output = test_output_path.read_text(errors="replace")
        results = parser(test_output)
        results = {self._normalize_test_name(k): v for k, v in results.items()}
        passed = sorted(k for k, v in results.items() if v == "PASSED")

        eval_meta_dir = self.config.persistent_dir / "eval_meta"
        expected_passed = json.loads((eval_meta_dir / "expected_passed.json").read_text())
        norm_f2p = json.loads((eval_meta_dir / "fail_to_pass.json").read_text())
        norm_p2p = json.loads((eval_meta_dir / "pass_to_pass.json").read_text())

        passed_set = set(passed)
        fail_to_pass_set = set(norm_f2p)
        pass_to_pass_set = set(norm_p2p)

        from_fail_to_pass = sorted(passed_set & fail_to_pass_set)
        failed_from_pass_to_pass = sorted(pass_to_pass_set - passed_set)
        resolved = (fail_to_pass_set <= passed_set) and (pass_to_pass_set <= passed_set)

        report = {
            instance_id: {
                "resolved": resolved,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "from_fail_to_pass": from_fail_to_pass,
                "failed_from_pass_to_pass": failed_from_pass_to_pass,
                "passed_match": passed == expected_passed,
            }
        }
        report_path.write_text(json.dumps(report, indent=2))


class SweBenchExtDatasetProcessor(BaseDatasetHarnessProcessor):
    """Dataset processor for SWE-Bench-Ext format tasks."""

    def postprocess_after_run(self, report_file: Path) -> None:
        """Parse SWE-Bench-Ext test output on the host and write a resolution report.

        Reads the test framework and the fail-to-pass and pass-to-pass sets from the
        instance's eval metadata, parses the raw test output, and writes a report keyed by
        instance id. Parsing on the host avoids requiring Python inside the container.

        Args:
            report_file: Path to the report file to write; the sibling ``test_output.log``
                in the same directory supplies the raw test output.
        """
        from responses_api_agents.swe_env.parsing import parse_and_check_tests

        report_path = Path(report_file)
        test_output_path = report_path.parent / "test_output.log"
        instance_id = self.config.instance_id

        if not test_output_path.exists():
            report = {
                instance_id: {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": False,
                    "error": "No test output produced inside container",
                }
            }
            report_path.write_text(json.dumps(report, indent=2))
            return

        eval_meta_dir = self.config.persistent_dir / "eval_meta"
        fail_to_pass = json.loads((eval_meta_dir / "fail_to_pass.json").read_text())
        pass_to_pass = json.loads((eval_meta_dir / "pass_to_pass.json").read_text())
        test_framework = (eval_meta_dir / "test_framework.txt").read_text().strip()

        test_output = test_output_path.read_text(errors="replace")

        result = parse_and_check_tests(
            test_output=test_output,
            test_framework=test_framework,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            instance_id=instance_id,
        )

        report = {instance_id: result}
        report_path.write_text(json.dumps(report, indent=2))


class OpenHandsHarnessProcessor(BaseDatasetHarnessProcessor):
    """Harness processor that installs the OpenHands agent framework."""

    def setup(self) -> Path:
        """Clone and install the OpenHands agent framework under a locked setup directory.

        Returns:
            Path: The setup directory containing the prepared OpenHands install.
        """
        setup_dir = self.parent_dir / "swe_openhands_setup"

        with self._setup_directory_lock(setup_dir, "OpenHands"):
            openhands_dir = setup_dir / "OpenHands"
            miniforge_dir = setup_dir / "miniforge3"

            if openhands_dir.exists() and Path(openhands_dir / ".venv" / "bin" / "python").exists():
                print(f"OpenHands already set up at {setup_dir}", flush=True)
                return setup_dir

            print(f"Setting up OpenHands environment at {setup_dir}...", flush=True)
            rmtree(setup_dir, ignore_errors=True)
            setup_dir.mkdir(parents=True, exist_ok=True)

            script_fpath = self.parent_dir / "setup_scripts/openhands.sh"
            command = f"""SETUP_DIR={setup_dir} \\
MINIFORGE_DIR={miniforge_dir} \\
OPENHANDS_DIR={openhands_dir} \\
AGENT_FRAMEWORK_REPO={self.config.agent_framework_repo} \\
AGENT_FRAMEWORK_COMMIT={self.config.agent_framework_commit} \\
    {script_fpath}"""
            self._run_setup_command(command)

            return setup_dir


########################################
# START Ray worker logic
########################################


def _classify_agent_error(err: Optional[str]) -> Optional[str]:
    """Classify an agent error message into a coarse failure-mode category.

    Args:
        err: The agent error message, or None/empty if the agent finished cleanly.

    Returns:
        Optional[str]: One of ``"max_iteration"``, ``"context_window"``,
        ``"stuck_in_loop"``, or ``"other"``; None when there is no error.
    """
    if not err:
        return None
    s = str(err)
    if "maximum iteration" in s:
        return "max_iteration"
    if "ContextWindow" in s or "context window" in s.lower():
        return "context_window"
    if "stuck in a loop" in s.lower():
        return "stuck_in_loop"
    return "other"


def _resolve_image_name(container_formatter: "str | list[str]", instance_id: str) -> str:
    """Resolve a sandbox image name from a container formatter template.

    Substitutes the instance id into the template (replacing ``__`` with ``_1776_`` and
    lowercasing) and strips a leading ``docker://`` scheme. For the default docker SWE-bench
    formatter this yields a Docker Hub image name; apptainer/.sif resolution is owned by the provider.

    Args:
        container_formatter: A template string, or a list whose first element is used,
            optionally containing the ``{instance_id}`` placeholder.
        instance_id: The task instance id to substitute into the template.

    Returns:
        str: The resolved image name with any ``docker://`` prefix removed.
    """
    fmt = container_formatter[0] if isinstance(container_formatter, list) else container_formatter
    if "{instance_id}" in fmt:
        fmt = fmt.format(instance_id=instance_id.replace("__", "_1776_").lower())
    return fmt[len("docker://") :] if fmt.startswith("docker://") else fmt


def _should_mask_sample(
    resolved: bool,
    agent_error_kind: Optional[str],
    eval_timed_out: bool,
    agent_timed_out: bool,
) -> bool:
    """Decide whether to mask this sample from the GRPO gradient.

    A sample is masked when: the patch passed eval but the agent did not actually submit
    (max-turns or context window), so the reward is accidental; the final eval timed out;
    or the agent itself timed out on wall-clock.

    Args:
        resolved: Whether the task was scored as resolved.
        agent_error_kind: The classified agent failure mode, or None if the agent finished cleanly.
        eval_timed_out: Whether evaluation timed out.
        agent_timed_out: Whether the agent timed out.

    Returns:
        bool: True if the sample should be masked.
    """
    return bool(
        (resolved and agent_error_kind in ("max_iteration", "context_window")) or eval_timed_out or agent_timed_out
    )


@ray.remote(
    scheduling_strategy="SPREAD",
    runtime_env={
        "py_executable": sys.executable,
    },
    num_cpus=0.1,
)
def runner_ray_remote(params_dict: dict[str, Any]) -> Optional[Path]:
    """Ray remote entrypoint that runs a single task instance in a worker.

    Validates the instance config from the serialized dict and drives the OpenHands agent
    for that task.

    Args:
        params_dict: Serialized ``SWEBenchWrapperInstanceConfig`` for the task.

    Returns:
        Optional[Path]: The report file path if one is produced, otherwise None.
    """
    # Ray may not pick up the proper model fields unless the models are rebuilt here.
    SWEBenchWrapperInstanceConfig.model_rebuild(force=True)
    RunOpenHandsAgent.model_rebuild(force=True)

    params = SWEBenchWrapperInstanceConfig.model_validate(params_dict)
    run_oh = RunOpenHandsAgent(config=params)
    report_file = asyncio.run(run_oh.process_single_datapoint())

    return report_file


def update_metrics(metrics_fpath: Path, update_dict: Dict[str, Any]) -> None:
    """Merge non-null metric values into the JSON metrics file on disk.

    Reads the existing metrics, drops null entries from both the existing and update
    dicts, merges them (update values winning), and writes the result back.

    Args:
        metrics_fpath: Path to the JSON metrics file to read and rewrite.
        update_dict: Metric values to merge in; null values are ignored.
    """
    with metrics_fpath.open() as f:
        existing_dict = json.loads(f.read())

    existing_dict = {k: v for k, v in existing_dict.items() if v is not None}
    update_dict = {k: v for k, v in update_dict.items() if v is not None}

    with metrics_fpath.open("w") as f:
        json.dump(existing_dict | update_dict, f)


# _TOOL_PARAM_BOOL_FIELDS_DEFAULT_FALSE = ("defer_loading",)


# def _dump_tool_as_tool_param(tool: BaseModel) -> Dict[str, Any]:
#     """Dump a response Tool pydantic model to a ToolParam-compatible dict."""
#     data = tool.model_dump()
#     for key in _TOOL_PARAM_BOOL_FIELDS_DEFAULT_FALSE:
#         if data.get(key) is None:
#             data[key] = False
#     return data


class RunOpenHandsAgent(BaseModel):
    """Drives a single OpenHands agent run for one task instance and collects its output."""

    config: SWEBenchWrapperInstanceConfig

    def _openhands_dir_copy_from_host(self, output_file_path: Optional[str]) -> Optional[str]:
        """Copy an OpenHands run's output and latest LLM completion off the eval directory.

        Locates the run's ``output.jsonl`` (falling back to the most recent one under the eval
        directory) and copies it to the configured prediction path, copies the latest LLM
        completion into the trajectories tree, then removes the eval directory and config file.

        Args:
            output_file_path: Path to the run output file, relative to the eval directory or
                absolute; if falsy, no output is copied.

        Returns:
            Optional[str]: The destination path of the copied output, or None if none was copied.
        """
        data_point = self.config.problem_info
        eval_dir_in_openhands = self.config.eval_dir_in_openhands
        config_file_path = self.config.openhands_config_file_path

        eval_dir_on_host = Path(self.config.openhands_setup_dir) / "OpenHands" / eval_dir_in_openhands
        trajectories_root = self.config.trajectories_root
        llm_completions_dir = trajectories_root / "llm_completions" / data_point["instance_id"]
        trajectories_root.mkdir(parents=True, exist_ok=True)
        llm_completions_dir.mkdir(parents=True, exist_ok=True)

        dest_output: Optional[str] = None
        if output_file_path:
            source_output = Path(output_file_path)
            if not source_output.is_absolute():
                source_output = eval_dir_on_host / source_output
            if not source_output.exists():
                output_candidates = sorted(eval_dir_on_host.glob("*/*/*/output.jsonl"), key=os.path.getmtime)
                if not output_candidates:
                    raise FileNotFoundError(
                        f"No output.jsonl found under {eval_dir_on_host} for {data_point['instance_id']}."
                    )
                source_output = output_candidates[-1]

            dest_output_path = self.config.prediction_path
            shutil.copy2(source_output, dest_output_path)
            dest_output = str(dest_output_path)

        completion_candidates = glob.glob(str(eval_dir_on_host / "*/*/*/llm_completions/*/*.json"))
        if completion_candidates:
            latest_completion = max(completion_candidates, key=os.path.getmtime)
            shutil.copy2(
                latest_completion,
                llm_completions_dir / Path(latest_completion).name,
            )

        shutil.rmtree(eval_dir_on_host, ignore_errors=True)
        try:
            Path(config_file_path).unlink()
        except OSError:
            pass

        return dest_output

    async def process_single_datapoint(self) -> Optional[Path]:
        """Run the agent (or substitute the golden patch) for this task instance.

        The agent runs in a single working sandbox, self-drives, and persists its patch and
        agent metrics; the eval and reward happen later in ``run()`` via the verifier POST. When
        ``verify_golden_patch`` is set, the sample's gold patch is substituted instead of running
        the agent.

        Returns:
            Optional[Path]: Always None; the patch is persisted to the metrics file rather than
            returned as a report file.
        """
        if self.config.verify_golden_patch:
            return await self._run_golden_patch_verification()

        return await self._run_decoupled_agent()

    async def _run_decoupled_agent(self) -> Optional[Path]:
        """Provision one working sandbox, self-drive OpenHands, and persist the extracted patch.

        Builds the task and launch command, stages the agent config and dataset files, provisions
        the sandbox via the swe_env infra, runs OpenHands locally inside it, and records the
        extracted patch and agent-side metrics (timing, error classification, timeout). The
        eval and reward happen in ``run()`` via a POST to the verifier.

        Returns:
            Optional[Path]: Always None; the patch is persisted to the metrics file.
        """
        from responses_api_agents.swe_agents.swe_env_adapter import (
            build_openhands_launch_command,
            openhands_config_toml,
            provision_and_collect,
        )
        from responses_api_agents.swe_env.harness import SweTask

        def _as_list(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return [v]
            return v or []

        data_point = self.config.problem_info
        instance_dict = json.loads(data_point["instance_dict"])
        setup_dir = str(self.config.openhands_setup_dir)
        gym_root = str(Path(setup_dir).resolve().parents[2])

        metrics = SWEBenchMetrics(ray_queue_time=time.time() - self.config.ray_queue_timestamp)
        metrics.openhands_run_time = -time.time()

        # Provider: explicit config, else docker with the Gym repo bind-mounted at its host path
        # (resolves OpenHands' venv abs-symlinks + the nemo_gym editable install) + host network.
        provider = self.config.sandbox_provider or {
            "docker": {"network": "host", "run_args": ["-v", f"{gym_root}:{gym_root}:ro"]}
        }
        task = SweTask(
            instance_id=self.config.instance_id,
            image=_resolve_image_name(self.config.container_formatter, self.config.instance_id),
            base_commit=instance_dict.get("base_commit", "") or "",
            repo_workdir="/testbed",
            test_command="",
            model_patch="",
            test_patch=instance_dict.get("test_patch", "") or "",
            fail_to_pass=_as_list(instance_dict.get("FAIL_TO_PASS")),
            pass_to_pass=_as_list(instance_dict.get("PASS_TO_PASS")),
            benchmark=data_point["dataset_name"],
            split=data_point.get("split", "test"),
            metadata={"ttl_s": self.config.swebench_agent_timeout + 600, "ready_timeout_s": 900},
        )
        launch = build_openhands_launch_command(
            setup_dir=setup_dir,
            instance_id=self.config.instance_id,
            dataset_name=data_point["dataset_name"],
            split=data_point.get("split", "test"),
            ng_config_dict_quoted=self.config.ng_global_config_dict_str,
            model_server_name=self.config.model_server_name,
            agent_cls=self.config.resolved_agent_cls,
            max_iter=self.config.agent_max_turns,
            command_exec_timeout=self.config.command_exec_timeout,
            tmux_memory_limit_mb=self.config.apptainer_memory_limit_mb,
        )
        stage_files = {
            "/root/config.toml": openhands_config_toml(
                self.config.body.model,
                temperature=self.config.inference_params.get("temperature", 0.0),
                top_p=self.config.inference_params.get("top_p", 1.0),
            ),
            "/root/dataset/data.jsonl": json.dumps(instance_dict),
        }

        try:
            result = await provision_and_collect(
                task,
                provider=provider,
                agent_launch_command=launch,
                stage_files=stage_files,
                patch_output_glob="/root/eval_results",
                agent_timeout_s=self.config.swebench_agent_timeout,
            )
            patch = result.get("patch") or None
            if patch and not patch.endswith("\n"):
                patch += "\n"
            metrics.openhands_run_time += time.time()
            metrics.model_patch = patch
            metrics.patch_exists = bool(patch)
            metrics.agent_error_kind = _classify_agent_error(result.get("agent_error"))
            # The environment does not raise on agent timeout; it returns error_type="timeout"
            # instead, so the except below never fires for a timed-out agent. Recover the timeout
            # signal from the provision result's error_type, with a wall-clock fallback, so the
            # sample is masked correctly.
            metrics.agent_timed_out = result.get("error_type") == "timeout" or (
                metrics.openhands_run_time is not None
                and metrics.openhands_run_time >= self.config.swebench_agent_timeout
            )
        except Exception as e:  # noqa: BLE001
            print(f"Decoupled agent run failed for {self.config.instance_id}: {e}", flush=True)
            metrics.openhands_run_time += time.time()
            metrics.patch_exists = False
            metrics.agent_timed_out = (
                metrics.openhands_run_time is not None
                and metrics.openhands_run_time >= self.config.swebench_agent_timeout
            )
        update_metrics(self.config.metrics_fpath, metrics.model_dump())
        return None

    async def _run_golden_patch_verification(self) -> Optional[Path]:
        """Skip the agent run and persist the sample's golden patch for verification.

        Writes the sample's gold patch (``instance_dict['patch']``) as the worker's
        ``model_patch`` in the metrics file, exactly where ``_run_decoupled_agent`` would leave an
        agent patch, so it is later graded by the verifier POST. Currently supported only for the
        ``swe-bench-ext`` dataset.

        Returns:
            Optional[Path]: Always None; the gold patch is persisted to the metrics file.
        """
        instance_id = self.config.instance_id
        dataset_name = self.config.problem_info.get("dataset_name")
        # TODO(sugam): add support for other datasets
        if dataset_name != "swe-bench-ext":
            raise NotImplementedError(
                f"verify_golden_patch is only supported for dataset_name=='swe-bench-ext' (got {dataset_name!r})."
            )

        instance_dict = json.loads(self.config.problem_info["instance_dict"])
        golden_patch = instance_dict.get("patch") or ""
        if not golden_patch.strip():
            raise ValueError(f"No golden patch found in instance_dict['patch'] for {instance_id}.")
        if not golden_patch.endswith("\n"):
            golden_patch += "\n"

        metrics = SWEBenchMetrics(ray_queue_time=time.time() - self.config.ray_queue_timestamp)
        metrics.model_patch = golden_patch
        metrics.patch_exists = True
        # No agent ran, so there is no agent error to classify (mask re-join stays clean).
        metrics.agent_error_kind = None
        update_metrics(self.config.metrics_fpath, metrics.model_dump())

        return None


########################################
# START Server logic
########################################


class SWEBenchWrapper(SimpleResponsesAPIAgent):
    """Responses API agent server that runs OpenHands on SWE-bench-style tasks and scores patches."""

    config: SWEBenchWrapperConfig

    _sem: Optional[Semaphore] = None
    _vllm_converter: Optional[VLLMConverter] = None
    _swe_bench_wrapper_server_config: Optional[SWEBenchWrapperServerConfig] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ########################################
    # START Init
    ########################################

    def model_post_init(self, context: Any) -> None:
        """Initialize per-run server state, run dataset/harness setup, and build helpers.

        Generates a run session id and results directory, runs the OpenHands and dataset
        processor setups, and creates the concurrency semaphore and vLLM converter.

        Args:
            context: The Pydantic post-init context passed through to the superclass.
        """
        run_session_id = f"{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        workspace_root = Path(__file__).parent
        self._swe_bench_wrapper_server_config = SWEBenchWrapperServerConfig(
            run_session_id=run_session_id,
            base_results_dir=workspace_root / f"swebench_results_{run_session_id}",
            ng_global_config_dict_str=shlex.quote(OmegaConf.to_yaml(get_global_config_dict())),
            model_server_name=self.config.model_server.name,
            openhands_setup_dir=OpenHandsHarnessProcessor(config=self.config).setup(),
            swebench_setup_dir=SweBenchDatasetProcessor(config=self.config).setup(),
            swebench_multilingual_setup_dir=SweBenchMultilingualDatasetProcessor(config=self.config).setup(),
            r2e_gym_setup_dir=R2EGymDatasetProcessor(config=self.config).setup(),
            swe_rebench_setup_dir=SWERebenchDatasetProcessor(config=self.config).setup(),
        )

        self._sem = Semaphore(self.config.concurrency)
        self._vllm_converter = VLLMConverter(return_token_id_information=True)

        return super().model_post_init(context)

    ########################################
    # START Results processing logic
    ########################################

    def get_openhands_trajectory_from_completions(self, trajectories_dir: Path, instance_id: str) -> tuple:
        """Read the trajectory and tools from the LLM completion files dumped by OpenHands.

        Loads the most recent completion file for the instance, appends the final assistant
        message (with any token-id and log-prob fields) to its message list, and returns the
        messages together with the tool definitions.

        Args:
            trajectories_dir: Directory containing the per-instance trajectory output.
            instance_id: The task instance id whose completions are read.

        Returns:
            tuple: A ``(messages, tools)`` pair; both are empty lists when no completion
            files are found.
        """
        messages, tools = [], []

        completions_dir = trajectories_dir / instance_id / "llm_completions" / instance_id
        if not completions_dir.exists():
            print(f"No llm_completions directory found: {completions_dir}", flush=True)
            return messages, tools

        completion_files = sorted(completions_dir.glob("*.json"))
        if not completion_files:
            print(f"No completion files found in: {completions_dir}", flush=True)
            return messages, tools

        last_file = completion_files[-1]

        with open(last_file, "r") as f:
            data = json.load(f)

        messages = data["messages"]
        provider_specific_fields = data.get("provider_specific_fields", {})
        final_assistant_message = data["response"]["choices"][0]["message"]

        for key in ["prompt_token_ids", "generation_token_ids", "generation_log_probs"]:
            if key in provider_specific_fields:
                final_assistant_message[key] = provider_specific_fields[key]

        if final_assistant_message.get("content") or final_assistant_message.get("tool_calls"):
            messages.append(final_assistant_message)

        tools = data.get("kwargs", {}).get("tools", [])

        return messages, tools

    ########################################
    # START Main methods
    ########################################

    def _resolve_absolute_path(self, path: Optional[str]) -> Optional[str]:
        """Resolve a possibly relative path against the package parent directory.

        Args:
            path: A relative or absolute path string, or None.

        Returns:
            Optional[str]: The absolute path string, or None if no path was given.
        """
        if not path:
            return None
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(PARENT_DIR / p)

    def _setup_params(
        self, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> Tuple[SWEBenchWrapperInstanceConfig, BaseDatasetHarnessProcessor]:
        """Build the per-instance config and select the dataset processor for a request.

        Creates the persistent working directory, writes the instance dataset file, maps
        inference parameters from the Responses request, resolves any prompt overrides, and
        chooses the dataset processor based on the dataset name.

        Args:
            body: The Responses API create-params request carrying the task metadata.

        Returns:
            Tuple[SWEBenchWrapperInstanceConfig, BaseDatasetHarnessProcessor]: The resolved
            per-instance config and the matching dataset processor.
        """
        problem_info = body.metadata | {"container_formatter": self.config.container_formatter}
        instance_id = problem_info.get("instance_id", "unknown")

        # Create persistent directory for I/O and logs in local workspace
        instance_dir = f"{instance_id}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        persistent_dir = self._swe_bench_wrapper_server_config.base_results_dir / instance_dir
        persistent_dir.mkdir(parents=True, exist_ok=True)

        agent_run_id = f"{instance_id}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        # To avoid making HF dataset API calls, we write the instance dictionary to a file and mount it in the container.
        instance_dataset_dir = persistent_dir / "instance_datasets"
        instance_dataset_dir.mkdir(parents=True, exist_ok=True)
        instance_dataset_path = instance_dataset_dir / f"{agent_run_id}.jsonl"
        instance_dict = json.loads(problem_info["instance_dict"])
        if "repo" in instance_dict and "repo_name" not in instance_dict:
            instance_dict["repo_name"] = instance_dict["repo"]
        with open(instance_dataset_path, "w") as f:
            f.write(json.dumps(instance_dict) + "\n")

        trajectories_root = persistent_dir / "trajectories" / instance_id
        output_for_eval_mounted_path = (
            Path("/trajectories_mount") / "trajectories" / instance_id / "output_for_eval.jsonl"
        )
        output_for_eval_path = trajectories_root / "output_for_eval.jsonl"
        prediction_path = trajectories_root / "output.jsonl"

        # Map from Responses to OpenHands
        inference_params = {}
        for param, key in [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_output_tokens", "tokens_to_generate"),
        ]:
            value = getattr(body, param, None)
            if value is not None:
                inference_params[key] = value

        eval_dir_in_openhands = f"evaluation/oh/{agent_run_id}"
        openhands_config_file_path = f"/tmp/config_{agent_run_id}.toml"

        agent_script_name = f"agent_script_{agent_run_id}.sh"
        agent_script_path = persistent_dir / agent_script_name

        # persistent_dir is mounted here in each container
        base_mounted_dir = Path("/trajectories_mount")

        params: SWEBenchWrapperInstanceConfig = SWEBenchWrapperInstanceConfig(
            **self.config.model_dump(),
            **self._swe_bench_wrapper_server_config.model_dump(),
            problem_info=problem_info,
            body=body,
            persistent_dir=persistent_dir,
            metrics_fpath=persistent_dir / "nemo_gym_metrics.json",
            base_mounted_dir=base_mounted_dir,
            profiling_dir=persistent_dir / "profiling",
            profiling_mounted_dir=base_mounted_dir / "profiling",
            ray_queue_timestamp=time.time(),
            inference_params=inference_params,
            agent_run_id=agent_run_id,
            instance_dataset_path=instance_dataset_path,
            trajectories_root=trajectories_root,
            output_for_eval_mounted_path=output_for_eval_mounted_path,
            output_for_eval_path=output_for_eval_path,
            prediction_path=prediction_path,
            model_patch_path=persistent_dir / "patch.diff",
            eval_dir_in_openhands=eval_dir_in_openhands,
            openhands_config_file_path=openhands_config_file_path,
            agent_script_path=agent_script_path,
            final_eval_apptainer_spinup_timestamp_fpath=persistent_dir / "final_eval_apptainer_spinup_timestamp",
            final_eval_apptainer_spinup_timestamp_mounted_fpath=base_mounted_dir
            / "final_eval_apptainer_spinup_timestamp",
            generation_apptainer_spinup_timestamp_fpath=persistent_dir / "generation_apptainer_spinup_timestamp",
            generation_apptainer_spinup_timestamp_mounted_fpath=base_mounted_dir
            / "generation_apptainer_spinup_timestamp",
        )

        params.metrics_fpath.write_text("{}")

        if params.agent_prompt_overrides:
            overrides = params.agent_prompt_overrides
            if params.agent_prompt_override_random:
                selected = random.choice(overrides)
            else:
                rng = random.Random(instance_id)
                selected = rng.choice(overrides)

            params.resolved_user_prompt_template = self._resolve_absolute_path(selected.user_prompt_template)
            params.resolved_system_prompt_template = self._resolve_absolute_path(selected.system_prompt_template)
            params.resolved_agent_cls = selected.agent_cls
            params.resolved_diversify_tool_names = selected.diversify_tool_names

        if params.problem_info["dataset_name"] == "nv-internal-1":
            dataset_processor = NVInternalDatasetProcessor(config=params)
        elif params.problem_info["dataset_name"] == "swe-bench-ext":
            dataset_processor = SweBenchExtDatasetProcessor(config=params)
        elif "SWE-rebench" in params.problem_info["dataset_name"]:
            dataset_processor = SWERebenchDatasetProcessor(config=params)
        elif "R2E-Gym" in params.problem_info["dataset_name"]:
            dataset_processor = R2EGymDatasetProcessor(config=params)
        elif "SWE-bench_Multilingual" in params.problem_info["dataset_name"]:
            dataset_processor = SweBenchMultilingualDatasetProcessor(config=params)
        else:
            dataset_processor = SweBenchDatasetProcessor(config=params)

        # The agent launch and patch egress are owned by swe_env_adapter; eval is the verifier POST.
        return params, dataset_processor

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        """Handle a Responses request: run the agent for the task and return the response.

        Sets up the per-instance config, persists it, and delegates to the inner handler. On
        failure the traceback is written to the persistent directory before the exception is
        re-raised.

        Args:
            body: The Responses API create-params request carrying the task metadata.

        Returns:
            NeMoGymResponse: The response containing the agent trajectory and task metrics.
        """
        params, dataset_processor = self._setup_params(body)

        with (params.persistent_dir / "params.json").open("w") as f:
            f.write(params.model_dump_json(indent=4))

        try:
            return await self._inner_responses(params, dataset_processor)
        except Exception as e:
            traceback_file = params.persistent_dir / "traceback.err"
            with traceback_file.open("w") as f:
                f.write(format_exc())

            print(f"Hit an exception in {self.config.name}! See {traceback_file} for more details", file=sys.stderr)

            raise e

    async def _verify_patch_via_server(self, params: SWEBenchWrapperInstanceConfig) -> Dict[str, Any]:
        """POST the worker's patch to the swe_env verifier and return its eval subset.

        Builds a verify request carrying the per-task metadata the verifier reads plus the patch
        in ``response.metadata.model_patch``, forwarding the instance's own per-framework test
        command and framework when present and falling back to a conda+pytest default otherwise.
        The call is bounded by a timeout. On any transport failure it returns a masked subset
        (``resolved=False``, ``error_kind='sandbox'``) rather than raising, so the agent always
        emits a present (masked) row instead of dropping the rollout.

        Args:
            params: The resolved per-instance config holding the patch and task metadata.

        Returns:
            Dict[str, Any]: The verifier's eval subset, or a masked subset on failure.
        """
        persisted = SWEBenchMetrics.model_validate_json(params.metrics_fpath.read_text())
        patch = persisted.model_patch or ""
        instance_dict = json.loads(params.problem_info["instance_dict"])

        def _as_list(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return [v]
            return v or []

        f2p = _as_list(instance_dict.get("FAIL_TO_PASS"))
        p2p = _as_list(instance_dict.get("PASS_TO_PASS"))
        # Forward the instance's own per-framework eval command and framework when present; only
        # fall back to the conda+pytest default when the row ships no test_command. This supports
        # both SWE-bench-Verified (no per-row command) and multi-framework swe-bench-ext rows
        # (cargo/go/npm/... that carry their own command and framework).
        test_framework = instance_dict.get("test_framework", "") or ""
        test_command = instance_dict.get("test_command", "") or ""
        if not test_command:
            nodeids = " ".join("'" + n + "'" for n in f2p + p2p)
            test_command = (
                "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "
                f"python -m pytest -rA {nodeids}"
            )
        task_metadata = {
            "instance_id": params.instance_id,
            "image": _resolve_image_name(params.container_formatter, params.instance_id),
            "base_commit": instance_dict.get("base_commit", "") or "",
            "repo_workdir": "/testbed",
            "test_command": test_command,
            "test_framework": test_framework,
            "test_patch": instance_dict.get("test_patch", "") or "",
            "fail_to_pass": f2p,
            "pass_to_pass": p2p,
            "benchmark": params.problem_info["dataset_name"],
            "split": params.problem_info.get("split", "test"),
        }
        verify_request = {
            "responses_create_params": params.body.model_dump() | {"metadata": task_metadata},
            "response": {
                "id": f"swebench-{params.instance_id}",
                "created_at": int(time.time()),
                "model": params.body.model,
                "object": "response",
                "output": [],
                "metadata": {"model_patch": patch},
            },
        }

        async def _do_verify() -> Dict[str, Any]:
            """POST the verify request and return the parsed JSON response.

            Returns:
                Dict[str, Any]: The verifier's JSON response body.
            """
            verify_response = await self.server_client.post(
                server_name=params.verifier_server_name,
                url_path="/verify",
                json=verify_request,
            )
            await raise_for_status(verify_response)
            return await get_response_json(verify_response)

        # Bound the whole call (including the client's disconnect-retry loop) so a hung or retried
        # verify cannot pin a rollout slot indefinitely.
        verify_timeout_s = float(getattr(params, "swebench_tests_timeout", None) or 900) + 900
        try:
            return await asyncio.wait_for(_do_verify(), timeout=verify_timeout_s)
        except Exception as e:  # noqa: BLE001  (incl. asyncio.TimeoutError -> masked, never pins a slot)
            print(f"Verifier POST failed for {params.instance_id}: {e}", flush=True)
            return {"resolved": False, "error_kind": "sandbox", "patch_exists": bool(patch)}

    async def _inner_responses(
        self, params: SWEBenchWrapperInstanceConfig, dataset_processor: BaseDatasetHarnessProcessor
    ) -> NeMoGymResponse:
        """Run the agent worker, score the patch, decide masking, and build the response.

        Dispatches the task to the Ray worker, grades the resulting patch either via the verifier
        POST or by post-processing the worker's report, decides whether to mask the sample from the
        GRPO gradient, reconstructs the trajectory and tool definitions, updates the metrics file,
        and assembles the response.

        Args:
            params: The resolved per-instance config for the task.
            dataset_processor: The dataset processor used to post-process worker output.

        Returns:
            NeMoGymResponse: The response containing output items, tools, and metrics metadata.
        """
        maybe_report_file = await runner_ray_remote.remote(params.model_dump())
        metrics_to_update = dict()

        if params.eval_via_verifier:
            # The worker persisted the patch (no in-worker eval); grade it by POSTing to the
            # verifier. The resolved/eval signals feed the same metrics and mask logic below.
            eval_subset = await self._verify_patch_via_server(params)
            metrics_to_update["resolved"] = bool(eval_subset.get("resolved"))
            metrics_to_update["eval_timed_out"] = eval_subset.get("error_kind") == "eval_timeout"
            if eval_subset.get("patch_exists") is not None:
                metrics_to_update["patch_exists"] = bool(eval_subset.get("patch_exists"))
        elif maybe_report_file:
            dataset_processor.postprocess_after_run(maybe_report_file)

            report = json.loads(Path(maybe_report_file).read_text())
            assert params.instance_id in report, (
                f"Report is malformatted. Expected instance ID key: {params.instance_id}. Report: {report}"
            )
            resolved = report[params.instance_id]["resolved"]
            metrics_to_update["resolved"] = resolved
        else:
            metrics_to_update["resolved"] = False

        # Decide whether to mask this sample from the GRPO gradient.
        persisted_metrics = SWEBenchMetrics.model_validate_json(params.metrics_fpath.read_text())
        resolved_now = metrics_to_update.get("resolved", False)
        agent_error_kind = persisted_metrics.agent_error_kind
        # eval_timed_out may come from a persisted metric or from the verifier POST (in
        # metrics_to_update and not yet persisted); prefer the latter when present.
        eval_timed_out = bool(metrics_to_update.get("eval_timed_out", persisted_metrics.eval_timed_out))
        agent_timed_out = bool(persisted_metrics.agent_timed_out)
        if _should_mask_sample(resolved_now, agent_error_kind, eval_timed_out, agent_timed_out):
            params.mask_sample = True

        trajectories_dir = params.persistent_dir / "trajectories"
        chat_completions_trajectory, chat_completions_tools = self.get_openhands_trajectory_from_completions(
            trajectories_dir, params.instance_id
        )

        tools = [
            FunctionTool.model_validate(tool["function"] | {"type": "function"}) for tool in chat_completions_tools
        ]
        responses_items = self._vllm_converter.chat_completions_messages_to_responses_items(
            chat_completions_trajectory
        )
        input_items, output_items = split_responses_input_output_items(responses_items)

        update_metrics(params.metrics_fpath, metrics_to_update)

        return NeMoGymResponse(
            id=f"swebench-{params.instance_id}",
            created_at=int(time.time()),
            model=params.body.model,
            object="response",
            output=output_items,
            parallel_tool_calls=params.body.parallel_tool_calls,
            tool_choice=params.body.tool_choice,
            tools=tools,
            metadata={
                "input": json.dumps([i.model_dump() for i in input_items]),
                "metrics": params.metrics_fpath.read_text(),
                "instance_config": params.model_dump_json(),
            },
        )

    async def run(self, body: BaseRunRequest) -> SWEBenchVerifyResponse:
        """Run one task end to end under the concurrency limit and return its reward and metrics.

        Acquires the concurrency semaphore, runs the agent via ``responses``, extracts the
        trajectory metadata and metrics, and assembles a verify response whose reward is 1.0 when
        the task resolved and 0.0 otherwise.

        Args:
            body: The run request carrying the Responses create-params for the task.

        Returns:
            SWEBenchVerifyResponse: The reward, response, metrics, and resolved instance config.
        """
        async with self._sem:
            body.responses_create_params.parallel_tool_calls = True
            body.responses_create_params.tool_choice = "auto"

            response = await self.responses(body.responses_create_params)

            metadata, response.metadata = response.metadata, None
            responses_create_params = body.responses_create_params.model_dump() | {
                "input": json.loads(metadata["input"]),
                "tools": [t.model_dump() for t in response.tools] if response.tools else [],
            }
            metrics = SWEBenchMetrics.model_validate_json(metadata["metrics"])

            return SWEBenchVerifyResponse(
                responses_create_params=responses_create_params,
                response=response,
                reward=1.0 if metrics.resolved else 0.0,
                **metrics.model_dump(),
                instance_config=SWEBenchWrapperInstanceConfig.model_validate_json(
                    metadata["instance_config"]
                ).model_dump(),
            )


if __name__ == "__main__":
    SWEBenchWrapper.run_webserver()
