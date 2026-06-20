# Verifiers (PrimeIntellect-ai) — external-agent integration review

Scope: evaluating whether the WAY `PrimeIntellect-ai/verifiers` integrates external agents is
the right route for NeMo-Gym issue #1396. Clone: `/tmp/agent-research/PrimeIntellect-ai__verifiers`.

## (a) How this library does external agents

Verifiers has **two distinct mechanisms**, and the distinction matters for #1396:

1. **Interception-proxy pattern (`CliAgentEnv` + `InterceptionServer`)** — the verified "external
   agent" flow. An *unmodified* CLI agent binary (Claude Code, OpenCode, etc.) runs inside a Prime
   sandbox. Verifiers starts an aiohttp `InterceptionServer`, opens a `prime_tunnel.Tunnel`, and sets
   the agent's `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` to `{tunnel}/rollout/{id}/v1` plus an
   `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` shared secret (`cli_agent_env.py:324` `build_env_vars`,
   `:230` `setup_state`). Each LLM request the agent emits is *intercepted* and becomes one
   `MultiTurnEnv.rollout()` step: the server queues the request, the env runs the real model via the
   base-class path, then delivers the synthesized completion back (`interception_utils.py:241`
   `_handle_request`, `:585` `deliver_response`, `:598` `synthesize_stream`). The agent never knows
   it is being driven. Reward is computed **in-process** by `Rubric.score_rollout()`
   (`rubrics/rubric.py:317`) — there is **no HTTP `/verify` endpoint** anywhere in the framework.
   So verifiers' notion of "external agent" = *agent code is external/unmodified, but the
   orchestration + scoring loop stays fully inside verifiers*. This is the **inverse** of #1396's
   ask (external orchestrator hitting Gym's `/verify`).

2. **NeMo-Gym adapter (`packages/harnesses/harnesses/nemo_gym.py` +
   `packages/tasksets/tasksets/nemo_gym.py`)** — verifiers as the *outer* harness driving NeMo-Gym.
   `NeMoGymHarness` boots an entire NeMo-Gym stack **in-process** (imports `nemo_gym.cli.RunHelper`,
   reuses Gym's `RolloutCollectionHelper`, monkeypatches Gym's CLI to skip the policy-model process),
   stands up a `NeMoGymModelProxy` registered as Gym's `policy_model` server, and routes each Gym
   model call back into the matching verifiers rollout via a per-rollout `routing_model` name
   (`nemo_gym.py:256` `PersistentNeMoGymRunner`, `:473` `NeMoGymModelProxy`, `:676`
   `build_nemo_gym_global_config`). It then posts a Gym row through Gym's own
   `RolloutCollectionHelper.run_examples`, reads back `reward`/`response`, and maps them onto verifiers
   `state` (`nemo_gym.py:883` `apply_nemo_gym_result`). This proves verifiers CAN consume Gym, but it
   does so by **deeply adopting Gym's internals** (Hydra/OmegaConf config, global aiohttp client,
   Ray env toggles, CLI monkeypatching) — exactly the heavy-ceremony coupling #1396 wants to escape,
   just pointed the other direction.

Bottom line: verifiers' external-agent design keeps the loop and scorer *inside the framework* and
exposes it via an LLM-API interception proxy. It is a strong reference architecture for "bring your
own unmodified agentic client" (R13) but does not provide the "external orchestrator calls Gym
`/verify` over a clean versioned HTTP contract" shape that issue #1396 centers on.

## (b) Requirement coverage (R1–R15)

| Req | Support | Evidence (clone paths) |
| --- | --- | --- |
| R1 agent code in external repo | yes | Agent is an arbitrary unmodified binary in a sandbox; `run_command` is just a shell string. `cli_agent_env.py:92` (`run_command`), `:345` `start_agent`. |
| R2 hit `/verify` without wrapping the whole loop | no | No `/verify` HTTP endpoint exists; reward is in-process `Rubric.score_rollout` (`rubrics/rubric.py:317`). External integration still wraps the entire loop inside a verifiers `Env`. |
| R3 custom output formats (beyond Messages/Chat) | partial | Interception serializes 4 protocols (chat/completions/responses/anthropic messages) via `protocol` switch (`interception_utils.py:264-272`, `serialize_intercept_response:749`), but the canonical internal type is fixed `Response`/`Messages` (`verifiers/types.py`); "custom" beyond these provider shapes is not a first-class contract. |
| R4 fast-path validation / conformance test | partial | `prime eval run`/`vf-eval` (`verifiers/scripts/eval.py`, `pyproject.toml:214`) is the canonical smoke path, and `tests/test_v1_endpoint_protocols.py` validates protocol shaping — but there is no standalone "does my external integration conform" tool short of running a full eval. |
| R5 clean HTTP contract, no internal-lib adoption | partial | The interception proxy IS a clean HTTP surface (`/rollout/{id}/v1/...`, `/health`, bearer secret; `interception_utils.py:85-130`). But to add a new external integration you adopt verifiers' `Env`/`Taskset`/`Harness` Python classes; and the Gym adapter (`harnesses/nemo_gym.py`) adopts Gym internals wholesale. No versioned `/run` request/response contract. |
| R6 external access to model server w/ URL discovery + auth | yes (pattern; see caveat) | Agent reaches the served model via injected `*_BASE_URL` + secret (`cli_agent_env.py:324`), tunnel URL auto-discovered/regenerated (`get_tunnel_url:183`), HMAC-checked auth (`interception_utils.py:234` `_authorized`). The NeMoGymModelProxy exposes a wildcard `/v1/{tail:.*}` (catches `/v1/responses`, `/v1/chat/completions`), `/v1/models` discovery, and optional auth (`nemo_gym.py:494-509`, `:572`). **Caveat:** in both flows the agent reaches an *interception/routing proxy that fronts a model*, not Gym's own `/v1/responses` server directly — the proxy is the URL-discovered, authenticated surface. The mechanism is the right shape for R6, but it is the proxy (not Gym's model server) that is exposed. |
| R7 output feeds profiling + TRAINING (token_ids/logprobs) | yes | `parse_tokens` reads `prompt_token_ids`/`token_ids`/`logprobs` off the served model and attaches `ResponseTokens` (`clients/openai_completions_client.py:148-178`, `clients/openai_chat_completions_client.py:472`); falls back to `None` for eval-only. A purpose-built `NeMoRLChatCompletionsClient` (`clients/nemorl_chat_completions_client.py:19-93`) specifically lifts NeMo Gym's `prompt_token_ids`/`generation_token_ids`/`generation_log_probs` into the shape `parse_tokens` expects and re-attaches them to outgoing prompts. `GenerateOutputs` (with `prompt_ids`/`completion_ids`, `types.py:1208`) feeds vf-eval and the prime-rl / `RLTrainer` training stack (`packages/verifiers-rl`). |
| R8 simple config (no Hydra/OmegaConf) | partial | Verifiers' own config is Pydantic + TOML (`docs/byo-harness.md` TOML section) — simple. BUT the Gym adapter pulls in OmegaConf and builds Gym Hydra config (`harnesses/nemo_gym.py:121` `infer_nemo_gym_agent_from_config`, `:676` `build_nemo_gym_global_config`). For a NeMo-Gym-side adoption the Hydra dependency remains. |
| R9 stateful session mapping OR opt-out | yes | Per-rollout isolation via `rollout_id` keyed queues/intercepts and per-rollout sandbox (`interception_utils.py:191` `register_rollout`, `cli_agent_env.py:236`). NeMoGymModelProxy routes concurrent rollouts by per-rollout model name (`tests/test_v1_nemo_gym_harness.py:29`). No cookie coupling. |
| R10 cross-infra timeout/retry/auth | yes | Prime Tunnel exposes the proxy across networks with liveness checks + auto-recreate (`cli_agent_env.py:183-228`); sandbox client has retry/backoff/jitter knobs (`cli_agent_env.py:109-119`); long OpenAI/HTTPX timeouts injected (`build_env_vars:330`); SSE keepalive guards intermediaries (`interception_utils.py:491-516`). |
| R11 4k–65k concurrent /run + backpressure | partial | `max_concurrent` semaphore in eval (`envs/environment.py:827,928`; `eval_utils.py:1065`), `sandbox_creations_per_minute` rate limit + connection caps (`cli_agent_env.py:116-118`). But each rollout = one sandbox + (optionally) one tunnel; no evidence of validated 65k-concurrent operation, and tunnel/sandbox-per-rollout is heavier than a pure HTTP `/run`. |
| R12 incremental adoption ramp | partial | Spectrum exists conceptually: base endpoint loop → `fn=` program → `command=` CLI agent → full interception harness (`docs/byo-harness.md` ProgramConfig table). But every rung lives inside a verifiers `Env`; there is no "config-only, zero-code" rung equivalent to #1396's lightest tier. |
| R13 bring-your-own agentic client (Claude Code/Cursor) | yes | Core design goal: any unmodified CLI agent via `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` redirection (`cli_agent_env.py:316-339`, README `:22`). Concrete harnesses: OpenCode, Pi, mini-swe-agent, Terminus (`packages/harnesses`). |
| R14 expose MCP tools + Agent Skills | partial | MCP tools: `MCPEnv` connects stdio MCP servers and exposes tools to the model (`envs/experimental/mcp_env.py:140`); interception server even has a `/vf/tools` proxy (`interception_utils.py:107-114`). "Agent Skills" in the repo are *authoring* skills for env developers (`skills/*/SKILL.md`), not runtime per-product Skills servers like Siemens Fuse. No multi-product MCP+Skills routing layer. |
| R15 access-controlled data lake / RAG, multi-tenant | no | No data-lake / access-controlled RAG / multi-tenant construct. Tasksets can carry tools/data but there is no secured shared retrieval layer or tenant model (`docs/byo-harness.md` Toolsets section is per-env, in-process). |

## (c) Pros of adopting this approach in NeMo-Gym

- **Best-in-class "bring-your-own unmodified agent" pattern (R13).** The interception-proxy trick —
  redirect `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL`, treat each LLM call as one step — lets Claude
  Code / Cursor / OpenCode run *verbatim* with zero agent-side changes. This is precisely the
  Siemens "agentic client → platform" shape (slide 2), and Gym has no equivalent today.
- **Free RL token capture (R7).** `parse_tokens` harvests `prompt_token_ids`/`token_ids`/`logprobs`
  straight off the served vLLM response (`openai_completions_client.py:148`), so an external agent
  yields training-grade data with no agent cooperation — directly satisfies Gym's training-pipeline
  requirement. Stronger still: verifiers already ships a **NeMo-Gym-specific** client
  (`clients/nemorl_chat_completions_client.py`) that lifts Gym's exact
  `prompt_token_ids`/`generation_token_ids`/`generation_log_probs` fields into the verifiers token
  shape and re-attaches them to follow-up prompts — i.e., the token-capture interop with Gym's
  vllm_model server is not hypothetical, it is implemented.
- **Mature cross-infra plumbing (R6/R10).** Auto-discovered tunnel URL with liveness/recreate, HMAC
  auth, retry/backoff, and SSE keepalives are exactly the cross-cluster/cloud/laptop concerns #1396
  raises, already battle-tested.
- **Existence proof of Gym interop.** `harnesses/nemo_gym.py` + `tasksets/nemo_gym.py` already drive
  Gym `/run` and read back `reward`/`response`, with a per-rollout model-routing proxy whose
  concurrency is unit-tested (`tests/test_v1_nemo_gym_harness.py`). The request/response mapping
  (`apply_nemo_gym_result`, `nemo_gym_row_from_task`) is a ready blueprint for the inverse contract.
- **Protocol fan-out is solved (R3-ish).** `serialize_intercept_response` already emits chat,
  completions, responses, and Anthropic-messages shapes from one internal `Response`.

## (d) Cons of adopting this approach in NeMo-Gym

- **Architectural inversion vs. #1396's core ask (R2/R5).** #1396 wants an *external orchestrator*
  to hit Gym's `/verify` over a clean versioned HTTP contract without wrapping Gym's loop. Verifiers
  does the opposite: the loop and the **in-process `Rubric` scorer** stay inside verifiers; there is
  **no `/verify` HTTP endpoint** (`rubrics/rubric.py:317`). Adopting verifiers' model means rebuilding
  Gym's loop around verifiers, not exposing Gym's verifier to outsiders.
- **Heavy per-rollout footprint (R11).** Each rollout provisions a Prime sandbox and (without a fixed
  `interception_url`) a tunnel. That is materially heavier than Gym's current `POST /run` and is not
  demonstrated at 65k concurrency. Backpressure exists (`max_concurrent`, creations-per-minute) but
  the unit of work is a VM/container, not a request.
- **Deep-coupling adapter (R8/R5).** The one place verifiers talks to Gym today
  (`harnesses/nemo_gym.py`) imports Gym's CLI/global-config/aiohttp internals, monkeypatches
  `run_command`/`setup_env_command`, toggles Ray env vars, and pulls in OmegaConf — the exact
  internal-plumbing ceremony #1396 calls out as the problem.
- **Prime-platform dependencies.** `prime_sandboxes`, `prime_tunnel`, and the Prime eval/upload flow
  (`AGENTS.md`) are first-class. The interception pattern is reusable in principle, but the shipped
  implementation is wired to Prime infrastructure.
- **No data-lake / multi-tenant / Skills-routing layer (R14/R15).** The Siemens "Fuse Agent" pieces
  — secured AI Data Lake, multi-tenant access control, per-product MCP+Skills auto-discovery — are
  absent. MCP tools exist; the platform layer does not.
- **Reward is binary-agnostic but not a contract.** Gym's `BaseVerifyResponse(reward)` is a typed
  HTTP response; verifiers' reward is a float assembled from weighted in-process funcs, harder to
  consume from an arbitrary external stack without running verifiers.

## (e) Engineering-disruption rating: HIGH

Rationale: Adopting verifiers' external-agent *approach* (the interception proxy) is not an additive
feature on top of today's `SimpleAgent /run → /verify` flow — it inverts ownership of the rollout
loop. Gym today owns the loop (agent server loops `/v1/responses` ↔ `/<tool>` then `/verify`).
Verifiers' model puts the loop + scorer inside the framework and exposes only an LLM-API proxy to
the agent. To "adopt the approach" Gym would need: (1) a new model-server variant that acts as an
interception proxy (forward to `vllm_model`, capture token_ids/logprobs, turn each intercepted call
into a step) — the verified flow doc already scopes this as net-new work; (2) a queue-based agent
server that blocks on intercepted requests; (3) sandbox + tunnel lifecycle management; and (4) a way
to still reach `/verify` for reward. The existing `harnesses/nemo_gym.py` shows the integration is
achievable, but only by importing and monkeypatching Gym internals — i.e., it does not reduce
disruption, it relocates it. None of this is config-only; all of it touches Gym's server-type
hierarchy and lifecycle.

A *lighter* extraction is possible — borrow only the proxy/redirect idea (`build_env_vars` +
`InterceptionServer`) to support unmodified agentic clients (R13), while keeping Gym's `/run`/`/verify`
contract — but that is selective inspiration, not "adopt verifiers' approach," and still requires a
new proxy model server and sandbox/tunnel management.

## (f) Fit verdict

Verifiers is an **excellent reference for R13/R6/R7/R10** (unmodified bring-your-own agentic clients,
authenticated model-endpoint access with URL discovery, free RL token capture, cross-infra tunnel
plumbing) and proves Gym interop is feasible. But its external-agent philosophy is the **inverse** of
issue #1396: it keeps the orchestration loop and the (in-process, non-HTTP) scorer inside the
framework and exposes an LLM-API interception proxy, rather than exposing Gym's `/verify` behind a
clean versioned HTTP `/run` contract for an external orchestrator (R2/R5 = no), and it brings no
data-lake/multi-tenant/Skills-routing layer (R15 = no). It is therefore **not the right wholesale
route** for #1396, but it is the **strongest single source to mine** for the "bring-your-own agent +
model-endpoint interception + token capture" sub-capabilities, especially the `CliAgentEnv` /
`InterceptionServer` redirect pattern and the existing `harnesses/nemo_gym.py` request/response
mapping.

## Fact-check

Independent re-grep / re-read of the clone (`/tmp/agent-research/PrimeIntellect-ai__verifiers`,
HEAD `ffcf6ce`). Verdict: the review is **accurate and well-cited**. Every load-bearing mechanism
claim and file/symbol/line citation was confirmed in the source. Edits below are clarifications and
one evidence strengthening; no coverage rating flipped direction.

### Confirmed claims (all 16 load-bearing claims verified)

1. **CliAgentEnv interception mechanism** — confirmed. `CliAgentEnv(SandboxMixin, vf.MultiTurnEnv)`
   at `cli_agent_env.py:84`; `__init__` 92-158; `setup_state` at :230; `build_env_vars` at :324;
   `start_agent` at :345. Each intercepted LLM request becomes one rollout step
   (`get_prompt_messages`:533 → `get_model_response`:545 → `deliver_response`/`synthesize_stream` in
   the `finally`). Accurate.
2. **build_env_vars sets `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` + injects `OPENAI_API_KEY`/
   `ANTHROPIC_API_KEY` = shared secret** — confirmed verbatim at `cli_agent_env.py:324-339`
   (`ANTHROPIC_BASE_URL` is the same URL with `/v1` stripped; both keys set to
   `interception_server.secret`).
3. **InterceptionServer aiohttp routes + HMAC auth** — confirmed. Routes
   `/rollout/{id}/v1/{chat/completions,completions,responses,messages}` at
   `interception_utils.py:91-106`, `/vf/tools` (POST :107-110, GET list :111-114), `/health` :127.
   `_authorized` HMAC (`Authorization: Bearer <secret>` or `x-api-key`) at :234-239.
4. **No HTTP `/verify`; reward is in-process `Rubric.score_rollout` (weighted sum)** — confirmed.
   `score_rollout` at `rubric.py:317`, reward = `sum(reward*weight)` (:334-349). Repo-wide
   `grep -rn '/verify'` finds **only** a `verify.sh` reference inside one environment's test
   (`environments/opencode_harbor/.../test_outputs.py:12`) — no server route. The reviewer's prose
   "grep for /verify finds only a deep experimental evaluator" is loosely worded: the literal
   `/verify` grep hits a test's `verify.sh`; the "deep experimental evaluator" the reviewer means is
   the QUEST `obj_task_eval/evaluator.py` `verify()`/`batch_verify()` (an in-process eval helper, not
   an HTTP route). Substance is correct: **no `/verify` endpoint exists.**
5. **parse_tokens reads `prompt_token_ids`/`token_ids`/`logprobs.token_logprobs` → `ResponseTokens`
   (prompt+completion ids+masks+logprobs), None when unavailable** — confirmed at
   `openai_completions_client.py:148-178` (and a sibling at `openai_chat_completions_client.py:472`).
6. **NeMo-Gym adapter files** — confirmed. `packages/harnesses/harnesses/nemo_gym.py`
   (`NeMoGymHarness`:165, `PersistentNeMoGymRunner`:256, `NeMoGymModelProxy`:473) and
   `packages/tasksets/tasksets/nemo_gym.py` (`NeMoGymTaskset`:51).
7. **Harness boots Gym in-process via `nemo_gym.cli.RunHelper` + `RolloutCollectionHelper`,
   monkeypatches `run_command`/`setup_env_command` to skip the policy_model process** — confirmed.
   Imports at :314-326; `skip_nemo_gym_policy_model_process` at :742 patches both
   `run_command` (:753) and `setup_env_command` (:746), returning a `NoopPolicyModelProcess` /
   `"true"` for the `policy_model` prefix.
8. **NeMoGymModelProxy registers as Gym's `policy_model` and routes by per-rollout model name
   `verifiers-nemo-gym-proxy-<id>`; concurrency-by-model unit-tested** — confirmed.
   `nemo_gym_proxy_model_name`:713 builds `f"{PROXY_MODEL_NAME}-{rollout_id}"` (PROXY_MODEL_NAME at
   :26); routing in `_endpoint_for_request`:615-627; test
   `tests/test_v1_nemo_gym_harness.py:29` `test_nemo_gym_proxy_routes_concurrent_rollouts_by_model`.
   (Note: this test lives at the **repo root** `tests/`, not under `packages/harnesses/tests/` — the
   review's `tests/test_v1_nemo_gym_harness.py:29` path is correct as repo-root-relative.)
9. **Adapter imports OmegaConf, builds Gym Hydra-style config, toggles `RAY_ENABLE_UV_RUN_RUNTIME_ENV`**
   — confirmed. `infer_nemo_gym_agent_from_config` imports OmegaConf (:121-142);
   `build_nemo_gym_global_config` (:676-691) sets `policy_base_url`/`policy_api_key`/
   `policy_model_name` + the `policy_model` server config; `disable_ray_uv_run_runtime_env` (:787-808)
   sets the env var and the ray-constants attribute to `0`/`False` and restores them.
10. **apply_nemo_gym_result: reward must be numeric, `response.output` → completion messages, other
    fields → metrics** — confirmed at :883-912 (raises `TypeError` for bool / non-numeric reward;
    `messages_from_nemo_gym_response` parses `response.output`).
11. **serialize_intercept_response emits 4 protocol shapes from one internal `Response`** — confirmed
    at `interception_utils.py:749-975` (`openai_chat_completions` inline, plus
    `serialize_anthropic_message_response`, `serialize_openai_responses_response`,
    `serialize_openai_completion_response`).
12. **MCPEnv connects stdio MCP servers + exposes tools; `skills/` are author-time dev SKILL.md, not
    runtime per-product Skills servers** — confirmed. `MCPEnv(vf.ToolEnv)` at `mcp_env.py:149` uses
    `StdioServerParameters`/`stdio_client` (:7-8, :49-55). `skills/` holds 8 developer skills
    (`create-environments`, `evaluate-environments`, `train-with-environments`, …), each a `SKILL.md`
    — none is a runtime Skills server. (Minor: the review cites `mcp_env.py:140`; line 140 is inside
    the `MCPTool` wrapper's `Tool(...)` build and the `MCPEnv` class header is at :149 — off by ~9
    lines, substance correct.)
13. **Concurrency bounded by `max_concurrent` semaphore; sandbox creation rate-limited; each rollout
    provisions a sandbox + optionally a tunnel** — confirmed. `max_concurrent` param at
    `environment.py:827`, `sem = await maybe_semaphore(max_concurrent)` at :928; `eval_utils.py:1065`
    resolves total concurrency; `sandbox_creations_per_minute`/connection caps at
    `cli_agent_env.py:116-118`. Tunnel is **optional**: only created when `interception_url is None`
    (`cli_agent_env.py:243-249`) — the review's "(without a fixed `interception_url`) a tunnel"
    correctly reflects this.
14. **get_tunnel_url manages a `prime_tunnel.Tunnel` with periodic liveness checks + auto-recreate**
    — confirmed at :183-228 (`TUNNEL_CHECK_INTERVAL=60s`, `check_registered`, recreate on dead/expired).
15. **byo-harness.md ProgramConfig spectrum (base loop → `fn=` Python → `command=` CLI) and
    `prime eval run`/`vf-eval` as canonical validation; all inside a verifiers Env** — confirmed.
    Table at `docs/byo-harness.md:393-398`; `vf-eval` script at `pyproject.toml:214`; `AGENTS.md:10`
    treats `prime eval run` as canonical.
16. **Prime-platform dependencies are first-class** — confirmed: `prime-tunnel>=0.1.8`,
    `prime-sandboxes>=0.2.25` are hard deps in `pyproject.toml:40-41`; prime-rl / `packages/verifiers-rl`
    `RLTrainer` is the training consumer.

### Refuted / corrected claims

- **None refuted.** No mechanism error, no stale path, no wrong symbol, no fabricated capability was
  found. The two imprecisions are cosmetic: (a) the `/verify` grep wording in §(a) (clarified above —
  the literal grep hits a test `verify.sh`, the "evaluator" is QUEST's in-process `verify()`); (b)
  `mcp_env.py:140` vs the `MCPEnv` class at :149. Neither changes any conclusion.

### Corrections applied to the report

- **R7 evidence strengthened (still `yes`).** The original cited only the generic `parse_tokens`. Added
  the purpose-built `NeMoRLChatCompletionsClient` (`clients/nemorl_chat_completions_client.py:19-93`),
  which lifts Gym's exact `prompt_token_ids`/`generation_token_ids`/`generation_log_probs` into the
  verifiers token shape and re-attaches them to follow-up prompts. This makes R7 interop with Gym's
  vllm_model server *implemented*, not hypothetical — the (c) pro was updated to say so.
- **R6 caveat added (still `yes (pattern)`).** Clarified that in both flows the agent reaches an
  interception/routing proxy that *fronts* a model, not Gym's own `/v1/responses` server directly.
  The mechanism is the right shape for R6 (URL discovery + auth on a model endpoint), but the exposed
  surface is a proxy, not Gym's model server — so R6's literal phrasing in #1396 ("reach **Gym's**
  model servers") is satisfied by the *pattern*, not by direct exposure. Also corrected the proxy
  citation to the real handler/route (`nemo_gym.py:494-509` wildcard `/v1/{tail:.*}` + `/v1/models`).

### Coverage-rating changes

- **No rating flipped.** All R1–R15 ratings (R1 yes, R2 no, R3 partial, R4 partial, R5 partial,
  R6 yes, R7 yes, R8 partial, R9 yes, R10 yes, R11 partial, R12 partial, R13 yes, R14 partial,
  R15 no) are supported by the clone and stand.
- R6 wording tightened to `yes (pattern; see caveat)` — the support level is unchanged; the label now
  flags that the exposed endpoint is a proxy.
- R7 kept `yes`, evidence expanded (NeMoRL client) — if anything this is now the **best-supported**
  "yes" in the matrix.
- R11 (`partial`) and R15 (`no`) were spot-checked for over/under-claiming and are correctly harsh:
  no 65k-concurrent demonstration exists and the unit of work is a sandbox/VM; there is genuinely no
  data-lake / multi-tenant / access-controlled-RAG construct (Siemens R15) anywhere in the tree.

### Adjusted fit verdict

**Unchanged in direction, confirmed with higher confidence.** Verifiers is **not the right wholesale
route** for #1396 — it keeps the orchestration loop and the in-process `Rubric` scorer inside the
framework and exposes an LLM-API interception proxy, the architectural **inverse** of #1396's
"external orchestrator calls Gym `/verify` over a clean versioned HTTP `/run` contract" (R2 = no,
R5 = partial). It brings no data-lake/multi-tenant/Skills-routing layer (R15 = no) and its agent-side
"Skills" are author-time dev docs, not runtime per-product Skills servers (R14 = partial). However it
is the **strongest single source to mine**: the `CliAgentEnv` + `InterceptionServer`
`OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` redirect (R13), the tunnel/HMAC/SSE cross-infra plumbing
(R6/R10), and — newly emphasized — an **already-shipped, implemented NeMo-Gym token-capture client
and bidirectional harness** (`nemorl_chat_completions_client.py`, `harnesses/nemo_gym.py`,
`tasksets/nemo_gym.py`) that prove Gym↔verifiers interop end-to-end, at the cost of deeply adopting
Gym internals (OmegaConf/Hydra config, CLI monkeypatching, Ray toggles, global aiohttp client).
Engineering disruption to *adopt verifiers' approach wholesale* remains **HIGH** (it relocates rather
than removes the ceremony); a **selective extraction** of the redirect/interception + token-capture
ideas, kept behind Gym's existing `/run`/`/verify` contract, is the recommended path.
