# Harbor — external-agent integration review (for NeMo-Gym issue #1396)

_Clone reviewed: `/tmp/agent-research/harbor-framework__harbor` · verified flow: `/opt/Gym/diagrams/harbor-flow.md`_

## (a) How Harbor does "external agents"

Harbor (from the Terminal-Bench team) is **not** an HTTP-microservice framework like NeMo-Gym. It is an
in-process Python library + CLI (`harbor run`) that, for each task, **spins up a container
environment** (`BaseEnvironment`: Docker/Daytona/Modal/E2B/GKE/Singularity/…), **installs an agent into
that container**, runs it, then runs a **`tests/test.sh` script inside the container that writes a numeric
reward to `/logs/verifier/reward.txt`** (`src/harbor/verifier/verifier.py`). "External agent" means one of
two things, neither of which is an out-of-tree HTTP client:

1. **Installed agents** (`src/harbor/agents/installed/*.py`, e.g. `claude_code.py`): a black-box CLI
   (Claude Code, Codex, Cursor, Gemini, OpenHands, even `nemo_agent.py`) is `npm/pip`-installed into the
   container and shelled out to via `environment.exec(...)`. The agent reaches **its own model** via env
   vars like `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL` — Harbor does not host or proxy the policy model.
2. **Custom Python agents**: you subclass `BaseAgent`/`BaseInstalledAgent` and pass
   `--agent-import-path module.path:ClassName` (`src/harbor/agents/factory.py:create_agent_from_import_path`,
   `examples/agents/marker_agent.py`). The class can live outside the Harbor tree but **must import Harbor's
   library types** (`BaseAgent`, `BaseEnvironment`, `AgentContext`) and is loaded **in-process** by the CLI.

There is **no `/run` HTTP endpoint and no `/verify` HTTP endpoint**. The FastAPI servers in the repo are
(i) an internal container command-exec server for Singularity (`environments/singularity/server.py`, routes
`/exec`,`/health`,`/shutdown`), (ii) a local results **viewer** (`viewer/server.py`, many `/api/...` routes),
and (iii) an ephemeral OAuth callback receiver (`auth/callback_server.py`, route `/auth/callback`). None of
these is a `/run`/`/verify` contract an external repo could call. Concurrency is an in-process
`asyncio.Semaphore(n_concurrent_trials)` in `src/harbor/trial/queue.py` (`TrialQueue`), driven from
`src/harbor/job.py`. RL is supported by a documented `HarborRolloutInterface` pattern (SkyRL/Tinker) that
calls `Job.run()` in-process and reads `token_ids`/`mask_ids` out of `agent_result.metadata`
(`docs/content/docs/training-workflows/rl.mdx`, `src/harbor/models/agent/rollout_detail.py`).

The net effect: Harbor's "external agent" story is the **mirror image** of issue #1396. NeMo-Gym wants
external agent *clients* to call **into** Gym's verification/model HTTP services. Harbor wants to pull external
agent *CLIs/classes* **into** its own managed container+process tree and grade them with an in-container shell
script. It is a strong **container-eval/RL-rollout harness**, but it is the opposite of the "decoupled HTTP
`/verify` you can hit from your own repo" pattern the issue is asking for.

## (b) Requirement coverage (R1–R15)

| Req | Support | Evidence (clone paths) |
|-----|---------|------------------------|
| **R1** agent in external repo | **partial** | `--agent-import-path mod:Cls` loads an out-of-tree class (`src/harbor/agents/factory.py:108-130`, `examples/agents/marker_agent.py`), but it must subclass Harbor's `BaseAgent` and is imported **in-process** — not a separately deployed service. The richer "installed/adapter" path lives **inside** the Harbor tree (`adapters/*`, `agents/installed/*`). |
| **R2** hit `/verify` without wrapping the loop | **no** | There is no HTTP `/verify`. Verification is a container-side `tests/test.sh` → `/logs/verifier/reward.txt` executed by `Verifier.verify()` (`src/harbor/verifier/verifier.py:132-220`); you cannot call it standalone over HTTP — it is invoked only inside a `Trial` that also owns the agent run. |
| **R3** custom output formats | **partial** | Agents populate a free-form `AgentContext` (`models/agent/context.py`) and an optional ATIF trajectory (`models/trajectories/*`, RFC `rfcs/0001-trajectory-format`); reward is whatever `test.sh` emits. But the contract is "files in a container," not Messages/Chat/custom-over-HTTP. |
| **R4** fast-path validation / conformance | **partial** | Oracle-agent verification (`solution/solve.sh` must hit 100% reward, `docs/.../adapters.mdx` Step 3) plus `cli/quality_checker` / `cli/debug_checker` / `scripts/validate_adapter.py` give a conformance harness — but it validates **tasks/adapters**, not an external HTTP integration, and requires building containers. |
| **R5** clean HTTP contract, no internal-lib adoption | **no** | The contract is a **Python ABC** (`BaseAgent.setup/run`, `agents/base.py:91-132`) you must subclass + import, plus a container filesystem convention. No versioned HTTP request/response; integrating requires adopting Harbor's library types. |
| **R6** external access to model server w/ discovery+auth | **no** | Harbor does **not** host a policy-model server. Agents reach their own model via `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL` env (`agents/installed/claude_code.py:1271`, `nemo_agent.py` `_LLM_PROVIDERS`). For RL it points agents at an external vLLM URL (`rl.mdx` `kwargs.base_url`) — there is no Gym-style `/v1/responses` discovery+auth handshake to expose. |
| **R7** output feeds profiling + TRAINING (token_ids/logprobs) | **yes** | `RolloutDetail` carries `prompt_token_ids`/`completion_token_ids`/`logprobs` (`models/agent/rollout_detail.py`); `AgentContext.rollout_details` + `agent_result.metadata["token_ids"]/["mask_ids"]` feed `HarborRolloutInterface.run()` → SkyRL `Rollout` (`docs/.../rl.mdx`). Two strategies documented: vLLM interception or agent-returned tokens. Metrics/pass@k aggregation exist (`job.py`, `utils/pass_at_k.py`). |
| **R8** simple config (no Hydra/OmegaConf) | **yes** | Config is CLI flags + plain TOML (`task.toml`, `models/task/config.py`) / optional YAML job config (`run_<adapter>.yaml`). No Hydra/OmegaConf. `harbor run -d <dataset@ver> -a <agent> -m <model>` is a one-liner. |
| **R9** stateful session mapping OR opt-out | **partial / n/a** | No cookie-session concept (no HTTP loop). Statefulness is the container itself + per-task session dirs (`claude_code.py:_get_session_dir`); multi-step tasks carry state across steps (`trial/multi_step.py`). There is nothing to "opt out of" because there is no Gym-style session middleware. |
| **R10** cross-infra timeout/retry/auth | **yes** | Pluggable environments for laptops→clouds→K8s (`environments/{docker,daytona,modal,e2b,gke,…}.py`); retry/backoff in `trial/queue.py` (`RetryConfig`) and `auth/retry.py`; provider auth in `src/harbor/auth/*`; per-phase network policy/allowlist (`models/task/config.py:NetworkPolicy`). |
| **R11** 4k–65k concurrent `/run` + backpressure | **partial** | Concurrency is an in-process `asyncio.Semaphore(n_concurrent_trials)` (`trial/queue.py:39`, `job.py:108`) bounded by container-provisioning, not an HTTP service absorbing 4k–65k concurrent requests with capacity signaling. Cloud backends scale trials, but the "thousands of concurrent **`/run` HTTP calls** with backpressure" model does not apply — there is no `/run` server. |
| **R12** incremental adoption ramp | **partial** | Spectrum exists (built-in agent → custom `--agent-import-path` → full adapter w/ parity), but the floor is "build a container + write `test.sh` + subclass an ABC," which is a steeper zero-code floor than #1396's "config-only endpoint URL." |
| **R13** bring-your-own agentic client (Claude Code/Cursor) | **yes** | First-class: `agents/installed/claude_code.py`, `cursor_cli.py`, `codex.py`, `gemini_cli.py`, `openhands.py`, etc., selectable via `--agent`. This is Harbor's core strength and directly matches the Siemens "agentic client" box. |
| **R14** expose MCP tools + Agent Skills | **yes** | `MCPServerConfig` (stdio/sse/streamable-http) in `task.toml` (`models/task/config.py:499`), registered into the agent (`claude_code.py:_build_register_mcp_servers_command`); Skills dir copied into the agent (`skills.py`, `claude_code.py:_build_register_skills_command`, `EnvironmentConfig.skills_dir`); MCP-sidecar tutorial `docs/.../tutorials/mcp-server-task.mdx`. Strong match to Siemens "Tools via MCP" + "Agent Skills." |
| **R15** access-controlled data lake / RAG, multi-tenant | **no** | No data-lake/RAG/multi-tenant abstraction. Closest primitives: per-phase network allowlist (`NetworkPolicy`), MCP sidecar services, and registry namespacing `org/name` (`adapters.mdx` Step 8). No access-controlled vectorized-data injection or tenant isolation as the slide describes. |

## (c) PROS of adopting Harbor's approach in NeMo-Gym

- **Bring-your-own agentic client is solved and battle-tested (R13).** 25+ real CLI agents (Claude Code,
  Cursor, Codex, Gemini, OpenHands, plus NVIDIA's own `nemo_agent.py`) are already integrated as black-box
  CLIs the user picks with `--agent`. This is exactly the Siemens "agentic client" box and is far ahead of
  Gym's current single-turn `SimpleAgent`.
- **MCP + Agent Skills are first-class (R14).** Declarative `MCPServerConfig` (stdio/sse/streamable-http) and
  a skills directory are wired straight into the agent's config inside the container, plus an MCP-sidecar
  tutorial. This is a near-direct implementation of the Siemens "Tools via MCP / Agent Skills" columns.
- **RL token plumbing already exists (R7).** `RolloutDetail` (prompt/completion token_ids + logprobs),
  `AgentContext.rollout_details`, and the documented `HarborRolloutInterface` (SkyRL/Tinker) match Gym's
  training-pipeline requirement, including the two token-collection strategies (vLLM interception vs.
  agent-returned tokens) that Gym would also need for black-box agents.
- **Simple config (R8) and broad cross-infra execution (R10).** TOML/CLI instead of Hydra; pluggable
  Docker/Daytona/Modal/E2B/GKE/Singularity backends with retry, network allowlists, and provider auth —
  useful prior art for Gym's "laptops → clusters → clouds" requirement.
- **Mature container-eval + parity discipline (R4-adjacent).** Oracle-verification, quality/debug checkers,
  and a rigorous parity protocol are a strong template for validating new environments.

## (d) CONS of adopting Harbor's approach in NeMo-Gym

- **No HTTP `/verify` and no HTTP `/run` — the central ask of #1396 is absent (R2, R5).** Verification is an
  in-container `test.sh` → `reward.txt`, invoked only inside a `Trial` that also owns the agent run. There is
  no way for an external repo to "just POST `/verify`." Adopting Harbor's model would mean abandoning Gym's
  decoupled resources-server `/verify` contract, not extending it.
- **The integration contract is a Python ABC + container filesystem, not a clean HTTP boundary (R1, R5).**
  Even the "external" `--agent-import-path` path requires subclassing `BaseAgent` and importing Harbor's
  library types in-process — the very "adopt our internal abstractions" coupling the issue wants to avoid.
- **It inverts the data/model flow (R6).** Harbor does not host the policy model; agents call their *own*
  model endpoint. Gym's requirement that external harnesses **reach Gym's `/v1/responses` with discovery +
  auth** has no analog — Harbor would push you to run a separate model server the agent points at, losing the
  "consistent inference for eval and production" guarantee.
- **Concurrency model is process-local, not service backpressure (R11).** An `asyncio.Semaphore` over
  container-backed trials does not map onto "4k–65k concurrent `/run` requests with capacity signaling." The
  scaling bottleneck is container provisioning, which is heavier than Gym's HTTP rollout fan-out.
- **High zero-code floor (R12).** The cheapest real integration still requires a Dockerfile, an
  `instruction.md`, a `test.sh`, and (for custom agents) an ABC subclass — heavier than #1396's "config-only,
  endpoint-URL" entry rung.
- **No data-lake/RAG/multi-tenant story (R15).**

## (e) Engineering-disruption rating: **HIGH**

Adopting Harbor's external-agent *approach* would require NeMo-Gym to replace its core abstraction, not
extend it. Gym today is HTTP microservices: `ng_collect_rollouts → POST /run (SimpleAgent) → POST
/v1/responses (model server) ↔ POST /<tool> (resources server) → POST /verify → BaseVerifyResponse`
(see `/opt/Gym/nemo gym.png`). Harbor's model is in-process orchestration of **containerized** agents graded by
an **in-container shell script**, with the policy model living **outside** the framework. To make Gym work
"the Harbor way" you would add a container-environment layer, a CLI-shelling agent runner, an in-container
`verify()` that runs `test.sh`, and you would lose the standalone HTTP `/verify` and `/v1/responses` services
that issue #1396 is explicitly trying to open up to external callers. That is a foundational rewrite of the
verification and inference planes, hence HIGH.

The disruption is **lower if scoped to harvesting components rather than the architecture**: Harbor's
agent-installer catalog (R13), `MCPServerConfig`/skills wiring (R14), and `RolloutDetail`/`HarborRolloutInterface`
token plumbing (R7) are reusable ideas/patterns that map onto a new Gym agent server. But that is "borrow
Harbor's agent ideas," not "adopt Harbor's external-agent integration route."

## (f) FIT VERDICT

**Harbor is the wrong route for the #1396 external-agent epic, but the right reference for the Siemens
client/MCP/skills/RL pieces.** Its strengths (bring-your-own CLI agents R13, MCP+Skills R14, RL token export
R7, simple config R8, cross-infra execution R10) are real and directly relevant to the Siemens "Fuse Agent"
pattern. However, the epic's load-bearing requirements — a decoupled HTTP `/verify` you can hit from an
external repo without wrapping the agent loop (R2/R5), external access to Gym's `/v1/responses` model server
with discovery+auth (R6), and HTTP-service-level concurrency/backpressure for 4k–65k `/run` (R11) — are
absent or inverted in Harbor, which uses no HTTP `/run`/`/verify` contract and does not host the policy model.
Recommendation: **do not adopt Harbor's integration architecture; mine it for the agent-catalog, MCP/Skills,
and RL-token patterns** while keeping Gym's HTTP microservice contract as the integration boundary the issue
calls for.

## Fact-check

Independent re-grep/re-read of the clone (`/tmp/agent-research/harbor-framework__harbor`) against issue #1396
and the Siemens slides. Net result: the report is **substantively accurate**. Every mechanism claim and every
R# rating holds up; only three minor accuracy fixes were needed, all applied above. No rating changed.

### Corrections applied to the report
- **FastAPI server enumeration was incomplete (section a).** The report listed only the Singularity exec server
  and the viewer. A full route grep (`grep -rn "@app.(get|post|...)" src/`) shows a third FastAPI app —
  `src/harbor/auth/callback_server.py` route `/auth/callback` (ephemeral OAuth receiver) — and the Singularity
  server also exposes `/shutdown` (not just `/exec`,`/health`). Fixed the prose. This does **not** change the
  conclusion: there is still **no `/run` and no `/verify` route anywhere** in the repo (the grep returned zero
  hits for `"/run"`/`"/verify"`), so R2/R5 are unaffected.
- **Table R6 rating string normalized** from "n/a / no" to **no**, matching the structured coverage call and the
  evidence (Harbor hosts no policy model; agents reach their own model via `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL`).
- **Table R15 rating string normalized** from "no / partial" to **no**. The "closest primitives" (NetworkPolicy
  allowlist, MCP sidecars, `org/name` registry namespacing at `docs/.../adapters.mdx:512`) are not a data-lake /
  RAG / multi-tenant abstraction, so the honest rating is a clean **no**.

### Confirmed claims (verified in the clone)
- **No HTTP `/run` or `/verify`.** Full route grep over `src/` yields only `/exec`,`/health`,`/shutdown`
  (singularity), `/api/...` (viewer), `/auth/callback` (auth). Zero `/run`/`/verify`.
- **Verifier mechanism.** `Verifier.verify()` (`src/harbor/verifier/verifier.py:132-220`) runs the in-container
  test script via `environment.exec`, then parses `reward.json` first (`_parse_reward_json`, line 76/209) and
  falls back to `reward.txt` (`_parse_reward_text`, line 61/211) into `VerifierResult` (line 220). Docs confirm
  `tests/test.sh` writes a numeric reward to `/logs/verifier/reward.txt` (`adapters.mdx:163`).
- **Verification is Trial-internal.** `Trial` (`trial/trial.py:61`) `run()` (line 286) calls `agent.run()` (377)
  and `verifier.verify()` (449/516); not a standalone HTTP service. Confirmed.
- **External-agent integration = subclass + import-path.** `--agent-import-path mod:Cls`
  (`cli/trials.py:130-133`) → `AgentFactory.create_agent_from_import_path` (`agents/factory.py:108-130`),
  loaded in-process; `examples/agents/marker_agent.py` subclasses `BaseAgent`. Confirmed.
- **`BaseAgent` ABC signature.** static `name()` (base.py:75), `version()` (80), async `setup(environment)`
  (91), async `run(instruction, environment, context)` (107). Confirmed exactly.
- **Installed agents shell out and reach their own model.** `claude_code.py` run() env block uses
  `ANTHROPIC_BASE_URL` (line 1271, within ~1268-1342); OPENAI_BASE_URL is used elsewhere (codex/qwen/opencode/
  swe_agent and `nemo_agent.py` `_LLM_PROVIDERS` line 103/111). Harbor hosts no policy server.
- **RL token plumbing.** `RolloutDetail` is a `TypedDict` with `prompt_token_ids`/`completion_token_ids`/
  `logprobs` (`models/agent/rollout_detail.py`); `AgentContext.rollout_details`+`metadata`
  (`models/agent/context.py:21,29`); `HarborRolloutInterface.run()` reads `token_ids`/`mask_ids` from
  `trial_result.agent_result.metadata` → SkyRL `Rollout` (`rl.mdx:80-94`). Two documented strategies — vLLM
  interception vs agent-returned metadata (`rl.mdx:108-110`). Token fields are used in real agent code
  (`terminus_2.py`, `openhands_sdk*.py`), not just docs. `utils/pass_at_k.py` exists and feeds `job.py`.
- **RL points agents at external vLLM.** `kwargs.base_url` + `model_name="hosted_vllm/<model>"` (`rl.mdx:55-57`).
- **Concurrency = in-process semaphore.** `asyncio.Semaphore(n_concurrent)` (`trial/queue.py:39`), built in
  `Job` with `n_concurrent=self.config.n_concurrent_trials` (`job.py:111-112`; report said ~108, close enough).
- **Config is TOML + CLI (+ optional YAML/JSON), no Hydra/OmegaConf.** `task.toml` (`models/task/config.py`,
  `task.py`), optional `--config-path` YAML/JSON validated into Pydantic `TrialConfig`
  (`cli/trials.py:42-47,432-440`). The lone `hydra`/`omegaconf` grep hit (`leaderboard/dynamic_validation.py`)
  is a **false positive** — it matched the words "Rehydrate"/"harbor_version", not any import. No Hydra/OmegaConf.
- **MCP.** `MCPServerConfig` with `MCPTransport = Literal["stdio","sse","streamable-http"]`
  (`models/task/config.py:496-518`), registered via `ClaudeCode._build_register_mcp_servers_command` which
  writes `$CLAUDE_CONFIG_DIR/.claude.json` (line 1185-1211). Confirmed.
- **Agent Skills.** `resolve_skills`/`compute_skill_digest` (`skills.py:14,29`), `EnvironmentConfig.skills_dir`
  (`config.py:347`), `ClaudeCode._build_register_skills_command` (line 1156). Confirmed.
- **25+ CLI agents.** `AgentFactory._AGENTS` lists **28** classes (`factory.py:44-73`), incl. ClaudeCode,
  CursorCli, Codex, GeminiCli, OpenHands, MiniSweAgent, NemoAgent. Confirmed.
- **Pluggable environments + retry/auth/network.** `environments/` has docker, daytona, modal, e2b, gke,
  singularity, apple_container, runloop, novita, tensorlake; `RetryConfig` (`trial/queue.py`), `NetworkMode`/
  `NetworkPolicy` (`config.py:20,28`), `auth/`. Confirmed.
- **Adapter onboarding floor.** Dockerfile + instruction.md + solution/solve.sh + tests/test.sh, oracle must
  hit 100% reward before parity (`adapters.mdx:34-40,249,284`). Confirmed.
- **No data-lake/RAG/multi-tenant.** A repo-wide grep for data-lake / vector-store / RAG / multi-tenant returned
  no source hits (only a `docs/bun.lock` substring). Confirmed.

### Refuted claims
- **None.** No load-bearing claim was refuted. The only inaccuracy was the incomplete FastAPI-server list in
  claim #1 (it omitted `auth/callback_server.py` and the `/shutdown` route); corrected as an accuracy fix, not
  a refutation — the operative assertion (no `/run`/`/verify` HTTP endpoint) is true.

### Coverage-rating changes
- **None.** All fifteen R# ratings are well-calibrated and unchanged. Two table cells had hedged dual-strings
  ("n/a / no" for R6, "no / partial" for R15) that were normalized to their single correct value (**no**, **no**)
  to match the evidence and the structured coverage calls; the effective verdict for those rows did not move.
  - One watch-item kept as-is: **R7 "yes"** is about token-level export capability, which genuinely exists, but
    the data feeds **Harbor's own** SkyRL/Tinker training path, not Gym's aggregation+training pipeline. The
    report already flags this in section (c)/(f), so "yes" on the capability is defensible and retained.

### Adjusted fit verdict
**Unchanged. Harbor is the wrong architectural route for the #1396 external-agent epic (no HTTP `/run`/`/verify`,
no hosted policy model, process-local concurrency), but the right reference for the Siemens client/MCP/skills/RL
pieces (R13/R14/R7/R8/R10).** Engineering-disruption rating **HIGH** stands. The fact-check strengthens, rather
than alters, the original conclusion: the report's mechanism analysis and R-coverage are accurate after three
cosmetic corrections.
