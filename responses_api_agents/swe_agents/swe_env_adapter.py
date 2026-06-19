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

"""OpenHands ``swe_agents`` adapter onto the decoupled ``swe_env`` infra (#1249).

This is the SELF_DRIVING migration path for the legacy OpenHands harness (plan §6):
instead of ``_build_apptainer_command`` + the two-container ``/trajectories_mount``
handshake, it provisions the agent's working container via ``swe_env.lifecycle``,
injects a sandbox-reachable model endpoint (egress, §6), lets the agent self-drive
inside that container, extracts the unified-diff patch, then scores it through the
**verifier** in its own fresh sandbox (§4a) — i.e. environment + verification are
fully decoupled from the agent loop.

Additive on purpose: the legacy ``SWEBenchWrapper.run()`` is left intact so the
existing (mocked) test suite stays green; flipping ``run()`` to call this — and
deleting the legacy in-worker eval after a dual-run reward-parity window — is the
final cutover step, gated on apptainer/OpenHands validation (SWE_ENV_DECOUPLE_STATUS.md).
"""

from __future__ import annotations

import dataclasses
import json
import shlex
from collections.abc import Mapping
from typing import Any

from nemo_gym.sandbox import SandboxProvider
from resources_servers.swe_env.verify_task import verify_task
from responses_api_agents.swe_env import get_harness, model_endpoint, reward_from_report
from responses_api_agents.swe_env.harness import SweTask
from responses_api_agents.swe_env.lifecycle import CreateAdmission, SandboxRegistry, acquire_sandbox


def _provider_name(provider: Mapping[str, Any] | SandboxProvider) -> str:
    if isinstance(provider, Mapping):
        return next(iter(provider), "?")
    return getattr(provider, "name", "?")


async def _extract_patch_from_output_jsonl(env, output_glob: str) -> str:
    """Read the agent's unified diff from the newest OpenHands ``output.jsonl``.

    OpenHands (``RUNTIME=local``) writes its result row to
    ``{eval_output_dir}/.../output.jsonl`` with the patch at
    ``row["test_result"]["git_patch"]`` — NOT to the working tree, so a plain
    ``git diff`` would miss it. Validated against a real rollout (psf__requests-2317).
    """
    found = await env.execute(f"find {shlex.quote(output_glob)} -name output.jsonl 2>/dev/null | head -1")
    path = (found.get("stdout", "") or "").strip()
    if not path:
        return ""
    catted = await env.execute(f"cat {shlex.quote(path)}")
    raw = (catted.get("stdout", "") or "").strip()
    if not raw:
        return ""
    row = json.loads(raw.splitlines()[-1])
    return (row.get("test_result") or {}).get("git_patch", "") or ""


# --- OpenHands SELF_DRIVING launch builders (validated against psf__requests-2317) ---------------
# These mirror the legacy get_run_command env (app.py:1162-1245) but target a SINGLE swe_env
# sandbox (no apptainer two-container handshake): the Gym repo is bind-mounted at its host path
# (so OpenHands' venv abs-symlinks + the nemo_gym editable install resolve), OpenHands self-drives
# RUNTIME=local on the family's workdir, and the patch is read from output.jsonl.

_OH_OUTPUT_DIR = "/root/eval_results"
_OH_CONFIG_FILE = "/root/config.toml"
_OH_DATA_JSONL = "/root/dataset/data.jsonl"
_OH_METRICS_FPATH = "/root/nemo_gym_metrics.json"


def openhands_config_toml(model: str, *, temperature: float = 0.0, top_p: float = 1.0) -> str:
    """OpenHands ``[llm.model]`` config. ``native_tool_calling=false`` is more robust for small
    open models that don't emit a strict tool-call format (validated with Qwen2.5-Coder-3B)."""
    return (
        "[llm.model]\n"
        f'model = "{model}"\n'
        'api_key = "EMPTY"\n'
        'custom_llm_provider = "openai"\n'
        "native_tool_calling = false\n"
        f"temperature = {float(temperature)}\n"
        f"top_p = {float(top_p)}\n"
        "log_completions = true\n"
        'log_completions_folder = "/root/completions"\n'
    )


def build_openhands_launch_command(
    *,
    setup_dir: str,
    instance_id: str,
    dataset_name: str,
    split: str,
    ng_config_dict_quoted: str,
    model_server_name: str,
    agent_cls: str = "CodeActAgent",
    max_iter: int = 100,
    command_exec_timeout: int = 300,
    tmux_memory_limit_mb: int = 8192,
) -> str:
    """Build the in-sandbox bash that runs OpenHands ``run_infer.sh`` (RUNTIME=local).

    ``ng_config_dict_quoted`` is the already-shlex-quoted NeMo Gym global config dict
    (``config.ng_global_config_dict_str``) — egress routes OpenHands' ``NemoGymClient`` back to
    the real model server. ``dataset_name`` selects OpenHands' workspace via its DATASET_TYPE
    (e.g. official SWE-bench images -> ``SWE-Gym`` -> ``/testbed``).
    """
    oh = f"{setup_dir}/OpenHands"
    return (
        "set -e && "
        f"export PATH={setup_dir}/miniforge3/bin:$PATH && "
        "git config --global --add safe.directory '*' && "
        f"mkdir -p /root/completions /root/dataset {_OH_OUTPUT_DIR} && "
        "uid=$(id -ru 2>/dev/null || id -u) && export TMUX_TMPDIR=/tmp && "
        "export TMUX=/tmp/tmux-$uid/default && mkdir -p /tmp/tmux-$uid && chmod 700 /tmp/tmux-$uid && "
        "tmux -S /tmp/tmux-$uid/default start-server || true && "
        f"cd {oh} && export RUNTIME=local && "
        "export LOG_LEVEL=CRITICAL && export LOG_TO_FILE=False && export DEBUG=False && "
        f"export NEMO_GYM_METRICS_FPATH={_OH_METRICS_FPATH} && echo '{{}}' > $NEMO_GYM_METRICS_FPATH && "
        f"export NEMO_GYM_CONFIG_DICT={ng_config_dict_quoted} && "
        f"export NEMO_GYM_MODEL_SERVER_NAME={model_server_name} && "
        f"export VIRTUAL_ENV={oh}/.venv && export PATH=$PATH:{oh}/.venv/bin && "
        "export POETRY_VIRTUALENVS_IN_PROJECT=true && export POETRY_VIRTUALENVS_CREATE=false && "
        f"export POETRY_VIRTUALENVS_PATH={oh} && "
        f"export TMUX_MEMORY_LIMIT={tmux_memory_limit_mb} && export COMMAND_EXEC_TIMEOUT={command_exec_timeout} && "
        "export PYTHONDONTWRITEBYTECODE=1 && "
        "./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
        f"llm.model '' {agent_cls} 0 {max_iter} 1 {dataset_name} {split} {_OH_OUTPUT_DIR} "
        f"{instance_id} {_OH_DATA_JSONL} {_OH_CONFIG_FILE}"
    )


async def provision_and_extract_patch(
    task: SweTask,
    *,
    provider: Mapping[str, Any] | SandboxProvider,
    agent_launch_command: str,
    model_server: Mapping[str, Any] | None = None,
    opensandbox_service_url: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    stage_files: Mapping[str, str] | None = None,
    patch_output_glob: str | None = None,
    agent_timeout_s: int | float = 1800,
    registry: SandboxRegistry | None = None,
    admission: CreateAdmission | None = None,
) -> str:
    """Agent-side ONLY: provision a working sandbox, self-drive, return the unified-diff patch.

    No verification happens here — grading is the verifier's job (over HTTP, §4a), so this is
    the function the agent worker uses for the decoupled cutover. The patch crosses back to
    ``run()``, which POSTs it to the verifier.

    Two egress styles (validated end-to-end against a docker-provider OpenHands rollout):

    * ``model_server`` -> a sandbox-reachable OpenAI ``base_url`` (``model_endpoint.resolve``),
      for agents that call the model via a standard OpenAI/litellm client (e.g. mini-swe-agent).
    * ``extra_env`` -> injected verbatim, for agents hard-wired to NeMo Gym's ``ServerClient``.
      The in-tree OpenHands fork's ``CodeActAgent`` unconditionally routes through
      ``NemoGymClient`` (no litellm fallback), so it needs ``NEMO_GYM_CONFIG_DICT`` +
      ``NEMO_GYM_MODEL_SERVER_NAME`` + ``NEMO_GYM_METRICS_FPATH`` — NOT ``OPENAI_BASE_URL``.

    ``stage_files`` writes ``{remote_path: content}`` into the live sandbox before launch
    (e.g. OpenHands ``config.toml`` + the instance ``data.jsonl``). Patch source: the OpenHands
    ``output.jsonl`` when ``patch_output_glob`` is given, else ``git diff --cached`` on ``repo_workdir``.
    """
    harness = get_harness(task.benchmark)
    spec = harness.build_spec(task)

    # Model-server egress: inject only a sandbox-reachable endpoint (never the global dict).
    if model_server is not None:
        endpoint = model_endpoint.resolve(
            _provider_name(provider), model_server, opensandbox_service_url=opensandbox_service_url
        )
        spec = dataclasses.replace(spec, env={**spec.env, **endpoint.to_sandbox_env()})
    # NeMo-Gym-client egress / any extra in-sandbox env (e.g. OpenHands NEMO_GYM_* vars).
    if extra_env:
        spec = dataclasses.replace(spec, env={**spec.env, **dict(extra_env)})

    # Provision the agent's OWN working container, stage files, and let it self-drive.
    async with acquire_sandbox(
        provider, spec, registry=registry, admission=admission, instance_id=task.instance_id
    ) as env:
        for remote_path, content in (stage_files or {}).items():
            await env.write_text(remote_path, content)
        await env.execute(agent_launch_command, cwd=task.repo_workdir, timeout_s=agent_timeout_s)
        if patch_output_glob:
            return await _extract_patch_from_output_jsonl(env, patch_output_glob)
        diff = await env.execute(f"cd {task.repo_workdir} && git add -A && git diff --cached", cwd=task.repo_workdir)
        return diff.get("stdout", "") or ""


async def run_self_driving(
    task: SweTask,
    *,
    provider: Mapping[str, Any] | SandboxProvider,
    agent_launch_command: str,
    model_server: Mapping[str, Any] | None = None,
    opensandbox_service_url: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    stage_files: Mapping[str, str] | None = None,
    patch_output_glob: str | None = None,
    agent_timeout_s: int | float = 1800,
    registry: SandboxRegistry | None = None,
    admission: CreateAdmission | None = None,
) -> dict[str, Any]:
    """provision_and_extract_patch + in-process ``verify_task`` (standalone/test convenience).

    The production cutover keeps verification over HTTP (the agent worker calls
    ``provision_and_extract_patch`` and ``run()`` POSTs to the verifier); this bundled form
    exists for standalone reproduction and tests where co-launching the verifier is overkill.
    """
    patch = await provision_and_extract_patch(
        task,
        provider=provider,
        agent_launch_command=agent_launch_command,
        model_server=model_server,
        opensandbox_service_url=opensandbox_service_url,
        extra_env=extra_env,
        stage_files=stage_files,
        patch_output_glob=patch_output_glob,
        agent_timeout_s=agent_timeout_s,
        registry=registry,
        admission=admission,
    )
    # Score the patch in the verifier's OWN fresh sandbox (decoupled verification).
    report = await verify_task(provider, dataclasses.replace(task, model_patch=patch))
    masked = report.error_kind is not None
    return {
        "instance_id": task.instance_id,
        "model_patch": patch,
        "resolved": report.resolved,
        "reward": reward_from_report(report),
        "patch_exists": bool(patch.strip()),
        "mask_sample": masked,
        "error_kind": report.error_kind,
    }
