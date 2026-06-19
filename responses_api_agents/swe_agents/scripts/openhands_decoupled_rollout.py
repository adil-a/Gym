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

"""Reference: run OpenHands end-to-end through the DECOUPLED swe_env infra (#1249).

This is the validated recipe behind the legacy ``swe_agents/run()`` cutover. It does
NOT use the legacy two-container apptainer + ``/trajectories_mount`` handshake. Instead:

1. Provision the agent's working container with the swe_env **docker provider** +
   ``acquire_sandbox`` (host network for model egress; the Gym repo bind-mounted at its
   host path so OpenHands' venv abs-symlinks + the ``nemo_gym`` editable install resolve).
2. Let OpenHands self-drive ``RUNTIME=local`` on ``/testbed`` (``--dataset SWE-Gym``).
   Egress: the in-tree OpenHands ``CodeActAgent`` is hard-wired to ``NemoGymClient`` ->
   ``ServerClient.post(server_name, "/v1/chat/completions")``, so we inject
   ``NEMO_GYM_CONFIG_DICT`` (a crafted 3-level ``name.group.module.{host,port}`` map that
   routes to a model server) + ``NEMO_GYM_MODEL_SERVER_NAME`` + ``NEMO_GYM_METRICS_FPATH``
   — NOT ``OPENAI_BASE_URL`` (there is no litellm fallback in that fork).
3. Extract the patch from ``output.jsonl[test_result][git_patch]`` (not ``git diff``).
4. Grade it in a SEPARATE fresh verifier sandbox via ``verify_task`` (decoupled verification).

Validated locally (psf__requests-2317, docker provider, Qwen2.5-Coder-3B via vLLM): the
full pipeline runs — provision, multi-turn model egress, self-drive, output.jsonl
extraction, fresh-sandbox grading. (A 3B model is too weak to emit OpenHands-parseable
actions, so the demo patch is empty; the *mechanism* is what this validates. A resolving
patch -> reward 1.0 is covered by tests/test_swebench_real_instance.py on the verifier.)

Prereqs: docker; a vLLM (or Gym model server) reachable at the host/port baked into
NG_CONFIG; the official SWE-bench image for the instance; OpenHands set up under
swe_openhands_setup/. NOT a CI test — a manual reproduction/integration driver.

Usage:
    .venv/bin/python responses_api_agents/swe_agents/scripts/openhands_decoupled_rollout.py \
        --instance psf__requests-2317 --model Qwen/Qwen2.5-Coder-3B-Instruct \
        --model-host 127.0.0.1 --model-port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import responses_api_agents.swe_env.providers  # noqa: F401  registers docker + apptainer
from nemo_gym.sandbox import SandboxSpec
from resources_servers.swe_env.verify_task import clear_idempotency_cache, verify_task
from responses_api_agents.swe_agents.swe_env_adapter import run_self_driving
from responses_api_agents.swe_env.harness import SweTask


GYM = str(Path(__file__).resolve().parents[3])
SETUP = f"{GYM}/responses_api_agents/swe_agents/swe_openhands_setup"


def _image_for(instance_id: str) -> str:
    return "swebench/sweb.eval.x86_64." + instance_id.replace("__", "_1776_").lower() + ":latest"


def _as_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return [v]
    return v or []


def _ng_config_dict(model_host: str, model_port: int) -> str:
    # ServerClient resolves server_name 3 levels deep: cfg[name][group][module] -> {host,port}.
    cfg = {
        "head_server": {"host": "127.0.0.1", "port": 9099},
        "vllm_model": {"responses_api_models": {"vllm_model": {"host": model_host, "port": model_port}}},
    }
    return json.dumps(cfg)


def _config_toml(model: str, model_host: str, model_port: int) -> str:
    return (
        "[llm.model]\n"
        f'model = "{model}"\n'
        f'base_url = "http://{model_host}:{model_port}/v1"\n'
        'api_key = "EMPTY"\n'
        'custom_llm_provider = "openai"\n'
        "native_tool_calling = false\n"
        "temperature = 0.0\n"
        "top_p = 1.0\n"
        "log_completions = true\n"
        'log_completions_folder = "/root/completions"\n'
    )


def _launch_cmd(instance_id: str, model_host: str, model_port: int, max_iter: int) -> str:
    ng = json.dumps(_ng_config_dict(model_host, model_port))  # shell-safe quoted JSON literal
    return (
        "set -e && "
        f"export PATH={SETUP}/miniforge3/bin:$PATH && "
        "git config --global --add safe.directory '*' && "  # root container, host-owned bind mount
        "mkdir -p /root/completions /root/dataset /root/eval_results && "
        "uid=$(id -ru 2>/dev/null || id -u) && export TMUX_TMPDIR=/tmp && "
        "export TMUX=/tmp/tmux-$uid/default && mkdir -p /tmp/tmux-$uid && chmod 700 /tmp/tmux-$uid && "
        "tmux -S /tmp/tmux-$uid/default start-server || true && "
        f"cd {SETUP}/OpenHands && export RUNTIME=local && "
        "export LOG_LEVEL=INFO && export LOG_TO_FILE=False && export DEBUG=False && "
        "export NEMO_GYM_METRICS_FPATH=/root/nemo_gym_metrics.json && echo '{}' > $NEMO_GYM_METRICS_FPATH && "
        f"export NEMO_GYM_CONFIG_DICT={ng} && export NEMO_GYM_MODEL_SERVER_NAME=vllm_model && "
        f"export VIRTUAL_ENV={SETUP}/OpenHands/.venv && export PATH=$PATH:{SETUP}/OpenHands/.venv/bin && "
        "export POETRY_VIRTUALENVS_IN_PROJECT=true && export POETRY_VIRTUALENVS_CREATE=false && "
        f"export POETRY_VIRTUALENVS_PATH={SETUP}/OpenHands && "
        "export TMUX_MEMORY_LIMIT=8192 && export COMMAND_EXEC_TIMEOUT=300 && export PYTHONDONTWRITEBYTECODE=1 && "
        "./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
        f"llm.model '' CodeActAgent 0 {max_iter} 1 SWE-Gym test /root/eval_results "
        f"{instance_id} /root/dataset/data.jsonl /root/config.toml"
    )


async def main(args):
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    inst = next(r for r in ds if r["instance_id"] == args.instance)
    image = _image_for(args.instance)
    f2p, p2p = _as_list(inst.get("FAIL_TO_PASS")), _as_list(inst.get("PASS_TO_PASS"))

    # The agent self-drives, then run_self_driving extracts output.jsonl + grades in a fresh sandbox.
    nodeids = " ".join("'" + n + "'" for n in f2p + p2p)
    test_command = (
        f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && python -m pytest -rA {nodeids}"
    )
    task = SweTask(
        instance_id=args.instance,
        image=image,
        base_commit=inst["base_commit"],
        repo_workdir="/testbed",
        test_command=test_command,
        test_patch=inst.get("test_patch", ""),
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        benchmark="swe-bench-ext",
        metadata={"ttl_s": 3600, "ready_timeout_s": 900},
    )

    provider = {"docker": {"network": "host", "run_args": ["-v", f"{GYM}:{GYM}:ro"]}}
    # Write config.toml + instance dict via a pre-exec; run_self_driving runs the agent then extracts.
    registry_dir = tempfile.mkdtemp(prefix="oh-rollout-")
    from responses_api_agents.swe_env.lifecycle import CreateAdmission, SandboxRegistry, acquire_sandbox

    # Stage files into the SAME sandbox the agent uses: run_self_driving does provision+exec+extract,
    # but it needs config.toml + data.jsonl present first, so we stage via a thin pre-step here by
    # folding the writes into the launch command's heredocs is avoided — instead inject through env.
    # Simplest faithful path: provision, stage, run agent, extract — done inline (mirrors the adapter).
    t0 = time.time()
    registry, admission = SandboxRegistry(registry_dir), CreateAdmission(2)
    spec = SandboxSpec(image=image, workdir="/testbed", ttl_s=args.timeout + 600, ready_timeout_s=900)
    async with acquire_sandbox(
        provider, spec, registry=registry, admission=admission, instance_id=args.instance
    ) as env:
        print(f"[provision] {env.sandbox_id} ({time.time() - t0:.0f}s)", flush=True)
        await env.write_text("/root/dataset/data.jsonl", json.dumps(dict(inst)))
        await env.write_text("/root/config.toml", _config_toml(args.model, args.model_host, args.model_port))
        print("[launch] OpenHands run_infer.sh (RUNTIME=local) ...", flush=True)
        await env.execute(
            _launch_cmd(args.instance, args.model_host, args.model_port, args.max_iter),
            cwd=f"{SETUP}/OpenHands",
            timeout_s=args.timeout,
        )
        from responses_api_agents.swe_agents.swe_env_adapter import _extract_patch_from_output_jsonl

        patch = await _extract_patch_from_output_jsonl(env, "/root/eval_results")
    print(f"[patch] {len(patch)} bytes", flush=True)

    clear_idempotency_cache()
    report = await verify_task({"docker": {}}, dataclasses_replace(task, patch))
    from responses_api_agents.swe_env.grading import reward_from_report

    print(
        f"\n=== {args.instance}: resolved={report.resolved} patch_applied={report.patch_applied} "
        f"error_kind={report.error_kind} REWARD={reward_from_report(report)} ===",
        flush=True,
    )


def dataclasses_replace(task, patch):
    import dataclasses

    return dataclasses.replace(task, model_patch=patch)


if __name__ == "__main__":
    # run_self_driving is the production entry point; this script stages files + drives it for a
    # manual reproduction. (run_self_driving itself assumes config.toml/data.jsonl are baked into
    # the image or the launch command; here we stage them into the live sandbox first.)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="psf__requests-2317")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--model-host", default="127.0.0.1")
    p.add_argument("--model-port", type=int, default=8000)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--timeout", type=int, default=1800)
    _ = run_self_driving  # referenced for docs; staging path used here
    asyncio.run(main(p.parse_args()))
