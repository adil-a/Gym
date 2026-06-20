# External-agent integration (#1396): recommendation

_Synthesis of the three fact-checked reviews in this folder (`harbor-review.md`, `verifiers-review.md`, `openenv-review.md`). Decision criteria: PRIMARY = coverage of customer-requested functionality (R1–R15); SECONDARY = engineering disruption to today's `/run`→`/verify` flow._

## Bottom line

**None of the three is the right _wholesale_ architecture for #1396 — and that's the finding, not a cop-out.** Issue #1396 is fundamentally about **inverting control**: let an *external* orchestrator call Gym's existing HTTP services (`/run`, a standalone `/verify`, `/v1/responses`) over a clean, versioned, low-ceremony contract. All three libraries do the opposite — they **own the orchestration loop / environment server** and pull the agent into *their* world. So adopting any of them wholesale is rated **HIGH disruption** and would mean replacing Gym's verification/inference planes, not opening them up.

**The right route is to build #1396's HTTP contract natively in Gym and mine each library for specific, proven patterns.** Pattern extraction is LOW–MEDIUM disruption; wholesale adoption is HIGH.

## Functionality scoreboard (final, fact-checked R1–R15)

| Req | Harbor | verifiers | OpenEnv |
|---|---|---|---|
| R1 agent in external repo | partial | **yes** | **yes** |
| R2 hit `/verify` w/o wrapping loop | no | no | partial |
| R3 custom output formats | partial | partial | partial |
| R4 fast-path conformance test | partial | partial | **yes** |
| R5 clean HTTP contract, no internal-lib | no | partial | partial |
| R6 reach model server w/ discovery+auth | no | **yes** | partial |
| R7 training token_ids/logprobs | **yes** | **yes** | **yes** |
| R8 simple config (no Hydra) | **yes** | partial | **yes** |
| R9 stateful session / opt-out | partial | **yes** | **yes** |
| R10 cross-infra timeout/retry/auth | **yes** | **yes** | partial |
| R11 4k–65k concurrent `/run` + backpressure | partial | partial | **no** |
| R12 incremental adoption ramp | partial | partial | **yes** |
| R13 bring-your-own agentic client | **yes** | **yes** | **yes** |
| R14 MCP tools + Agent Skills | **yes** | partial | partial |
| R15 access-controlled data lake / multi-tenant | no | no | no |
| **Tally (yes / partial / no)** | **5 / 6 / 4** | **6 / 7 / 2** | **7 / 6 / 2** |

On raw coverage (primary criterion): **OpenEnv (7 yes) ≈ verifiers (6 yes) > Harbor (5 yes, and 4 hard "no"s on the core HTTP asks R2/R5/R6).** But the "no"s that hurt most — R2 (HTTP `/verify`), R5 (clean contract), R11 (concurrency), R15 (multi-tenant data lake) — are **Gym-side gaps none of the three fills**, which is exactly why this is a build-not-buy decision.

## What to mine from each (this is the actionable part)

**verifiers — the highest-leverage source for BYO-agent + RL training (R6/R7/R13).**
- The interception-proxy trick: redirect the agent's `OPENAI_BASE_URL`/`ANTHROPIC_BASE_URL` at a proxy, treat each LLM call as one rollout step, capture `token_ids`/`logprobs` off the served model for free.
- **It already ships working NeMo-Gym interop**: `clients/nemorl_chat_completions_client.py` (lifts Gym's `prompt_token_ids`/`generation_token_ids`/`generation_log_probs`), `harnesses/nemo_gym.py`, `tasksets/nemo_gym.py`, with concurrency unit-tested. This is real code to start from, not a concept.

**OpenEnv — the best match to the Siemens MCP-tools surface + #1396 ergonomics (R4/R12/R13, MCP half of R14).**
- `SessionMCPBridge` → HTTP `/mcp` → `codex mcp add --url`: a small recipe to re-expose a Gym resources server as MCP tools so Claude Code/Cursor/Codex drive it without importing Gym. Directly mirrors the Siemens "agentic client → MCP Tools + Skills → product servers" slide.
- `openenv skills add` skill-install plumbing (where to drop `SKILL.md` per client), the `_resolve_env_reward` reward-provenance guard, `transparent_proxy` logprob capture from a black-box CLI, and `openenv validate --url` (a conformance-test template for #1396's R4).

**Harbor — the broadest BYO-agent catalog + most mature MCP+Skills config (R13/R14/R7).**
- 28 ready CLI-agent adapters (Claude Code, Cursor, Codex, Gemini, OpenHands, plus NVIDIA's `nemo_agent.py`), declarative `MCPServerConfig` (stdio/sse/streamable-http) + skills-dir wiring, and the `RolloutDetail` token schema. Best if you want many agents out of the box.

## Agent contract & key design choices

The recommendation hinges on **where the agent boundary falls** — what the external agent sees vs. what stays inside Gym:

```
 FACING THE AGENT (the contract)      |  INTERNAL TO GYM (hidden)
  - task prompt (NO answer key)       |   - verifier logic + verifier_metadata
  - model endpoint /v1/responses      |   - policy weights / vLLM / trainer loop
  - tools (HTTP /<tool> or MCP)       |   - token_ids / logprobs / mask capture
  - session handle (or opt-out)       |   - session state, cookies, Ray, Hydra
  - /verify: submit output -> reward   |   - tool implementation + sandbox
  - capacity / backpressure signal     |   - metric aggregation -> training
```

- **A thin Gym-side broker drives.** It owns `/run`, **seeds the session**, and runs `/verify`; the external agent only *generates*. The agent receives `{prompt, model URL + token, session handle, tools}` and returns its output. (Today's internal `SimpleAgent` does seed + loop + verify together — externally we split it.)
- **`verifier_metadata` must move to Gym's side of `/verify` (must-do, not optional).** Today it rides along in the `/run` body to a *trusted* internal agent; an external/untrusted agent must never see the answer key or it can game/leak the reward. Gym holds it; the agent submits an output and receives only a scalar reward.
- **Control spectrum (R12):** lightest rung = the agent (in its own repo) calls Gym's `/v1/responses` + tools + `/verify` directly (Gym is the server); deepest rung = the agent exposes `/run` that Gym calls. The broker model above is the middle ground and the safe default for untrusted agents.
- **MCP ≠ Skills, and Skills are mostly *not Gym's job*.** MCP is the runtime tool/data *connection* Gym exposes; Agent Skills are packaged procedural *know-how the agent loads*. The Siemens ask needs both — but the **Skills publishing/auto-discovery framework (second half of R14) belongs to the customer's agent platform** (the Fuse Agent), *not* Gym, so it is **not** a Gym build. Gym's only skills touchpoint is **optional**: let an *environment author* bundle a `SKILL.md` for that env's own tools and have the contract deliver it (Harbor-style `skills_dir` copy-in). Gym's real R14 obligation is the **MCP tool exposure** — see step 3.

## Recommended path

1. **Build the #1396 contract natively in Gym (the real epic):** a versioned HTTP `/run`, a **standalone `/verify`** callable from an external repo without wrapping the agent loop, **model-server discovery + auth** for `/v1/responses`, cookie-session opt-out, and HTTP-level backpressure for 4k–65k concurrent `/run`. None of the three gives you this — it's the core build.
2. **Model access — two tiers; don't over-build.**
   - **2a. Cooperative / default (what #1396 actually asks for):** expose the model server with discovery + auth (R6) and accept training-compatible output — `token_ids`/`logprobs`/masks reported by the harness (R7). **No interception needed.** This is the issue's stated route and covers most external integrations.
   - **2b. Black-box fallback (only when the agent's loop is opaque and won't report tokens):** point its model `base_url` at an **interception proxy** that captures `token_ids`/`logprobs` at the call boundary and rebuilds masked, on-policy trajectories. Start from verifiers' existing `harnesses/nemo_gym.py` / `nemorl_chat_completions_client.py`. *Limits:* you can only train tokens **your** policy generated (you can't train weights you don't own); opaque loops can break on-policy attribution; the agent must be redirectable to your endpoint, else it's eval-only.
3. **Expose tools over MCP via a Streamable-HTTP `/mcp` gateway** on the facing side (the broker) that translates MCP `tools/list`/`tools/call` to the resources server's existing `POST /<tool>` — and **keep the plain `POST /<tool>` contract too**, for non-MCP harnesses (same tools, two front doors; satisfies R5 + R14 without coupling them). Build it **async (FastAPI + Gym's aiohttp client), not a stdlib `ThreadingHTTPServer`** like OpenEnv's bridge (whose threaded server is its concurrency weakness) so it meets R11; carry auth and map the MCP session to Gym's session handle. Mine OpenEnv's `SessionMCPBridge` *logic*, its `validate` conformance check (R4), and its reward-provenance guard; mine Harbor's `skills_dir` copy-in for the **optional** env-skill delivery (the Skills *framework* is the customer's job — see above).
4. **Optionally borrow Harbor's agent catalog** if breadth of supported CLI agents matters.
5. **R15 (access-controlled data lake / multi-tenant RAG) is greenfield** — no candidate addresses it; scope it as net-new Gym work. (The Agent-Skills *publishing framework* is also net-new, but it lives on the customer's agent platform, not Gym — see the Skills note above.)

**If you must name a single "closest" reference:** **verifiers** for the agent+training half (it already has working Gym code), **OpenEnv** for the MCP/Skills+ergonomics half. Harbor is the catalog/reference, not the architecture.
