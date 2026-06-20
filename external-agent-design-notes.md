# External Agent Support in NeMo-Gym — Design Notes

**Status:** Draft for team review
**Issue:** [#1305](https://github.com/NVIDIA-NeMo/Gym/issues/1305)
**Related PR:** [#1306](https://github.com/NVIDIA-NeMo/Gym/pull/1306) (draft docs page; will merge alongside this code change)

## TL;DR

Let customers point `ng_collect_rollouts` at their own agent URL (Claude Code, LangChain app, custom service) instead of forcing them to wrap their agent as a Gym agent server. The external service replaces the **agent server slot** in Gym's stack — it implements an OpenAI Responses API endpoint, and `ng_collect_rollouts` orchestrates everything around it (seed_session, verify, aggregate_metrics).

V1 covers **evaluation** only. RL training (where Gym would manage the model server so the trainer can hot-swap checkpoints) is deferred.

## Motivation

Customers post-training agents (not foundation models) typically build their own agent harnesses. Today they have two options to evaluate against NeMo-Gym environments:

1. **Rewrite their agent** as a Gym `SimpleResponsesAPIAgent` subclass. Heavy lift, ties them to Gym's class hierarchy.
2. **Bypass `ng_collect_rollouts` entirely** and write their own evaluation loop calling Gym's resources server endpoints directly (the path PR #1306 documents). Works, but loses `+num_repeats`, `+num_samples_in_parallel`, resume-from-cache, W&B logging.

Neither lets a customer plug in *fast*. The goal is `+agent_url=http://...` and you get all the existing rollout-collection ergonomics for free.

Per Chris Wing, both eval and RL-training use cases matter; we're targeting eval first since the RL case needs more design (model server checkpoint hot-swap, logprob/token propagation, etc.).

## Architecture

### Today: internal agent flow

`ng_collect_rollouts` makes **two** outbound calls per rollout (`/run` and, after the batch, `/aggregate_metrics`), both to the Gym-managed agent server. The agent server is the thing that talks to the resources server and model server.

```
ng_collect_rollouts        resources server          SimpleAgent (Gym)         model server
        │                          │                       │                       │
        │  1. POST /run            │                       │                       │
        │     (body = full task row including verifier_metadata)                   │
        ├──────────────────────────────────────────────────►                       │
        │                          │                       │                       │
        │                          │  1a. POST /seed_session                       │
        │                          │◄──────────────────────┤                       │
        │                          ├──────────────────────►│  (cookies)            │
        │                          │                       │                       │
        │                          │                       │  1b-i. POST /v1/responses
        │                          │                       │  (Gym-internal model call)
        │                          │                       ├──────────────────────►│
        │                          │                       │◄──────────────────────┤
        │                          │                       │                       │
        │                          │  1b-ii. POST /<tool>  │                       │
        │                          │  (if model emitted a tool call)               │
        │                          │◄──────────────────────┤                       │
        │                          ├──────────────────────►│                       │
        │                          │     (loop 1b-i / 1b-ii until model emits text)│
        │                          │                       │                       │
        │                          │  1c. POST /verify     │                       │
        │                          │◄──────────────────────┤                       │
        │                          ├──────────────────────►│ (BaseVerifyResponse)  │
        │                          │                       │                       │
        │◄─────────────────────────────────────────────────┤ (BaseVerifyResponse w/ reward)
        │                          │                       │                       │
        │     (write to output JSONL; repeat 1 per task)                           │
        │                          │                       │                       │
        │  2. POST /aggregate_metrics                                              │
        ├──────────────────────────────────────────────────►                       │
        │                          │  2a. POST /aggregate_metrics (proxied)        │
        │                          │◄──────────────────────┤                       │
        │                          ├──────────────────────►│                       │
        │◄─────────────────────────────────────────────────┤ (batch metrics)       │
```

### New: external agent URL flow

The Gym agent server is replaced by a URL the user supplies. `ng_collect_rollouts` now makes **four** outbound calls per rollout (`/seed_session`, `/v1/responses`, `/verify`, plus `/aggregate_metrics` after the batch). The external agent only handles the model+tool loop (2a/2b).

```
ng_collect_rollouts        resources server          external agent URL        model server
        │                          │                       │                  (user's, configured
        │                          │                       │                   into the agent)
        │                          │                       │                       │
        │  1. POST /seed_session   │                       │                       │
        ├─────────────────────────►│                       │                       │
        │◄─────────────────────────┤  (sets session cookie)│                       │
        │                          │                       │                       │
        │  2. POST /v1/responses (body = responses_create_params,                  │
        │     cookies from step 1,                                                 │
        │     header X-NeMo-Gym-Resources-Server: <url>)                           │
        ├─────────────────────────────────────────────────►│                       │
        │                          │                       │                       │
        │                          │                       │  2a. POST /v1/responses
        │                          │                       │     (user-configured URL,
        │                          │                       │      agent's own model call)
        │                          │                       ├──────────────────────►│
        │                          │                       │◄──────────────────────┤
        │                          │                       │                       │
        │                          │  2b. POST /<tool_name>│                       │
        │                          │  (forwarded cookies; if model emitted a tool call)
        │                          │◄──────────────────────┤                       │
        │                          ├──────────────────────►│                       │
        │                          │     (loop 2a / 2b until model emits text)     │
        │                          │                       │                       │
        │◄─────────────────────────────────────────────────┤ (NeMoGymResponse)     │
        │                          │                       │                       │
        │  3. POST /verify (body = responses_create_params + response + verifier_metadata,
        │     cookies from step 1) │                       │                       │
        ├─────────────────────────►│                       │                       │
        │◄─────────────────────────┤ (BaseVerifyResponse with reward)              │
        │                          │                       │                       │
        │     (write to output JSONL; repeat 1–3 per task)                         │
        │                          │                       │                       │
        │  4. POST /aggregate_metrics (after all rollouts complete)                │
        ├─────────────────────────►│                       │                       │
        │◄─────────────────────────┤ (batch metrics)       │                       │
```

### What moved between the two flows

| Responsibility                  | Today                          | New                                              |
|---------------------------------|--------------------------------|--------------------------------------------------|
| Call `/seed_session`            | `SimpleAgent.run()` (1a)       | `ng_collect_rollouts` (1)                        |
| Run model + tool loop           | `SimpleAgent.responses()`      | External agent (2a/2b)                           |
| Call `/verify`                  | `SimpleAgent.run()` (1c)       | `ng_collect_rollouts` (3)                        |
| Call `/aggregate_metrics`       | Proxied via `SimpleAgent` (2a) | `ng_collect_rollouts` directly to resources (4)  |
| Cookie threading                | Inside `SimpleAgent`           | `ng_collect_rollouts` → header on step 2, agent's HTTP client auto-forwards on 2b, `ng_collect_rollouts` reuses on step 3 |
| Resources server URL knowledge  | Read from agent's YAML config  | Sent to external agent via `X-NeMo-Gym-Resources-Server` header on step 2 |

## Design decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **`ng_collect_rollouts` calls `/seed_session` and `/verify`** (not the external agent) | Keeps the external agent Gym-unaware. If it had to call `/verify` itself, every agent author would need to know Gym's schema and URLs. |
| 2 | **`ng_collect_rollouts` calls `/aggregate_metrics` directly on the resources server** | Today this is proxied via the agent, but the proxy adds no value. Cuts the indirection. |
| 3 | **External agent contract = OpenAI Responses API (`POST /v1/responses`)** | Public standard. Many customers already have services that speak it (or have parts of their stack that do). A vLLM server with an OpenAI route technically satisfies the contract for stateless evals. |
| 4 | **Resources server URL passed as HTTP header** (`X-NeMo-Gym-Resources-Server`) | Lets the request body stay OpenAI-pure. Agent reads the header only if it makes tool calls; ignored otherwise. |
| 5 | **CLI: `+agent_url=...`, mutually exclusive with `+agent_name`** | `agent_name` continues to work for Gym-internal agents. Validated at config-load time. |
| 6 | **`agent_ref` in output JSONL = `{"url": "<url>"}`** for url-based rollouts | Clean distinction from name-based rollouts. Downstream code (e.g. `ng_reward_profile`) handles both keys. |

## What this lets users do

End-to-end, with the new flag:

```bash
# 1. Start the environment (resources server, optionally model server). No agent.
ng_run "+config_paths=[resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml]"

# 2. Start your own agent on some URL. It only needs one HTTP route:
#    POST /v1/responses → returns OpenAI-shaped NeMoGymResponse
#    (Many existing services already speak this.)

# 3. Run rollouts against your agent.
ng_collect_rollouts \
    +agent_url=http://localhost:9000 \
    +input_jsonl_fpath=path/to/data.jsonl \
    +output_jsonl_fpath=/tmp/rollouts.jsonl \
    +num_repeats=4 \
    +num_samples_in_parallel=10
```

Output JSONL, rollout metrics, resume-from-cache, W&B uploads — all work the same as for internal agents.

## Out of scope for V1

- **RL training integration.** The model server URL would need to be Gym-controlled (so the trainer can hot-swap checkpoints), token IDs and logprobs would need to flow back through the external agent into training, etc. Treated as a V2 follow-up.
- **Tool discovery.** External agents learn which tools exist by reading the resources server's code today. A `GET /tools` catalog endpoint could be added later; not needed for V1.
- **Auto-starting a resources server from `ng_collect_rollouts`.** User runs `ng_run` separately, same as today.
- **Auth / TLS for external agents.** Assumes trusted network. Cross-network deployments are the customer's problem (mTLS, firewall, etc.).

## Open questions

- **Multi-turn / Claude Code-style examples.** PR #1306's docs example is single-turn MCQA. We should add a multi-turn example showing a subprocess-driven agent (closer to Claude Code's shape) — but probably as a follow-up doc PR after the code lands.
- **Backwards-compat warning for `agent_name` config.** Do we want to surface a deprecation hint nudging users toward `agent_url` when they're using the simplest agents (e.g. `simple_agent`)? Probably not — `agent_name` is still the right answer for Gym-internal agents.
- **Per-row `agent_url` in JSONL.** The current plan only supports a CLI-level `+agent_url`. Should rows be able to specify `agent_ref: {"url": "..."}` directly (mixing internal and external agents in one run)? Trivial to support — just a question of whether anyone needs it.

## Implementation summary

Code changes are localized:

- `nemo_gym/rollout_collection.py` — add `agent_url` config field, mutual-exclusivity validator, external-agent code path (seed_session → external POST → verify), aggregate-metrics direct to resources server.
- `nemo_gym/global_config.py` — small helper to look up the resources server's host/port from the global config.
- `nemo_gym/server_utils.py` — minor: lets `ServerClient` POST to an arbitrary URL (or use `nemo_gym.server_utils.request()` directly).
- Tests in `tests/unit_tests/test_rollout_collection.py`.
- Docs: `docs/get-started/rollout-collection.md`, `docs/reference/cli-commands.md`, and the PR #1306 docs page (add a section on the `+agent_url` path).

No changes needed to:
- Any resources server (`resources_servers/*`)
- Any existing agent server (`responses_api_agents/*`)
- The base classes (`base_resources_server.py`, `base_responses_api_agent.py`)
