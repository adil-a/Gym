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

"""Per-instance OpenClaw config templates. Rendered host-side and written into
<persistent_dir>/openclaw_home/ before launching the agent container."""


def build_openclaw_json(
    *,
    workspace_path: str,
    model_name: str,
    upstream_base_url: str,
    upstream_api_key: str,
    gateway_auth_token: str,
    agent_env_bin: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    model_idle_timeout_seconds: int | None = None,
) -> dict:
    model_key = f"vllm/{model_name}"
    # Security-wrapper dir stays first (denylisted git/curl/... must resolve to the wrapper);
    # the dataset's runtime-env bin rides next so the agent's `python` is the repo interpreter.
    # See openclaw/dataset_env.py for why pathPrepend is the only lever that reaches the agent.
    exec_path_prepend = ["/openclaw_setup/bin"] + ([agent_env_bin] if agent_env_bin else [])
    # OpenClaw's openai-responses transport forwards exactly temperature (verbatim) and maxTokens
    # (-> max_output_tokens) from agents.defaults.params; an unset param is omitted. It never wires
    # top_p, so top_p is injected by the stream shim instead (see stream_shim.StreamShim).
    agent_params: dict = {}
    if temperature is not None:
        agent_params["temperature"] = temperature
    if max_output_tokens is not None:
        agent_params["maxTokens"] = max_output_tokens
    agents_defaults: dict = {
        "workspace": workspace_path,
        "models": {model_key: {"alias": "policy"}},
        "model": {"primary": model_key},
        "skipBootstrap": True,
        "heartbeat": {"every": "0m"},
        "startupContext": {"enabled": False},
    }
    if agent_params:
        agents_defaults["params"] = agent_params
    # The shim forwards NON-streamed (buffers the full turn), so OpenClaw's llm-idle watchdog
    # bounds the MAX single-turn generation time, not inter-token gaps. A slow turn can trip a
    # same-turn retry → duplicate proxy entries (deduped in trajectory_reconstruction).
    # `timeoutSeconds` raises the watchdog; omitted when None (provider default).
    vllm_provider: dict = {
        "baseUrl": upstream_base_url,
        "apiKey": upstream_api_key,
        "api": "openai-responses",
        "models": [
            {
                "id": model_name,
                "name": "policy",
                "api": "openai-responses",
                "input": ["text"],
                # HARDCODED False — do NOT enable for vLLM. `model.reasoning` only changes
                # the OUTBOUND request shape: (1) sends the system prompt as role:"developer"
                # → vLLM 400 "Unexpected message role"; (2) injects reasoning-effort/encrypted
                # params vllm_model can't translate. compat.supportsDeveloperRole:false does
                # NOT reliably suppress it. We lose nothing: inbound thinking still round-trips
                # on-policy via vllm_model's enable_thinking → <think> → structured reasoning
                # item, which OpenClaw replays verbatim with reasoning:false (wet-test verified).
                "reasoning": False,
            }
        ],
    }
    if model_idle_timeout_seconds is not None:
        vllm_provider["timeoutSeconds"] = model_idle_timeout_seconds
    return {
        "gateway": {
            "auth": {"mode": "token", "token": gateway_auth_token},
            "mode": "local",
            "port": 18789,
            "bind": "loopback",
            "controlUi": {"enabled": False},
            "tailscale": {"mode": "off"},
        },
        "discovery": {"mdns": {"mode": "off"}, "wideArea": {"enabled": False}},
        "update": {"auto": {"enabled": False}, "checkOnStart": False},
        "models": {
            "mode": "replace",
            "providers": {
                "vllm": vllm_provider,
            },
        },
        "agents": {
            "defaults": agents_defaults,
        },
        "tools": {
            # apply_patch is OpenAI/Codex-only in openclaw and is gated off for the
            # vllm provider; advertising it only produces a misleading warning.
            "allow": ["read", "write", "edit", "exec", "process"],
            "fs": {"workspaceOnly": True},
            "exec": {
                "ask": "off",
                "security": "full",
                "pathPrepend": exec_path_prepend,
                "applyPatch": {"workspaceOnly": False},
            },
            "loopDetection": {"enabled": False},
        },
        "skills": {
            "limits": {"maxSkillsInPrompt": 0, "maxSkillsPromptChars": 0},
            "allowBundled": ["__nonexistent_skill__"],
            "entries": {
                "healthcheck": {"enabled": False},
                "node-connect": {"enabled": False},
                "skill-creator": {"enabled": False},
                "taskflow": {"enabled": False},
                "taskflow-inbox-triage": {"enabled": False},
                "weather": {"enabled": False},
            },
        },
        "plugins": {
            "allow": ["vllm"],
            "bundledDiscovery": "allowlist",
            "slots": {"memory": "none"},
        },
    }


PI_SETTINGS: dict = {
    "compaction": {"enabled": False},
    "retry": {"enabled": False},
    "branchSummary": {"skipPrompt": True},
}


EXEC_APPROVALS: dict = {
    "version": 1,
    "defaults": {"security": "full", "ask": "off"},
    "agents": {"main": {"security": "full", "ask": "off"}},
}
