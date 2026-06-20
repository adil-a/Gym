# NeMo-Gym — External Agent Integration: Decision Brief (#1396)

> **TL;DR / Recommendation.** This is *not* a "pick a framework" decision — it's a "build a clean external-agent HTTP contract on Gym's side and mine proven patterns from others" decision. Harbor, Prime Intellect verifiers, and OpenEnv all *invert control* (they own the orchestration loop / env-server), so adopting any wholesale is HIGH disruption and misses Gym's core asks (HTTP `/verify`, model-server access, 4k–65k concurrency). **Recommended: build the external contract natively — a thin Gym-side broker owns `/run`, session seeding, and `/verify`; the agent only generates — then graft verifiers' interception + token-capture for black-box RL and OpenEnv's MCP-bridge + skills for the Siemens surface.**

## 1. What customers asked for
- **Issue #1396 "epic: external agent integration"** (opened cwing-nvidia → ananthsub): let external agent harnesses integrate *without Gym's ceremony* — agent in an external repo, clean versioned HTTP contract, hit `/verify` without wrapping the loop, reach the model server with discovery+auth, training-compatible output, no Hydra/OmegaConf, cross-infra, 4k–65k concurrent `/run` + backpressure, incremental adoption.
- **Driving customer — Siemens "Fuse Agent":** bring-your-own agentic client (Claude Code / Cursor) → **MCP Tools + Agent Skills + access-controlled AI Data Lake** → per-product servers (Solido Characterizer/Analytics/Generator/Repair).
- Distilled into a 15-point rubric **R1–R15** used to score each option (see RECOMMENDATION.md).

## 2. The reframe (the one thing to land)
#1396 wants to **invert control**: an *external* orchestrator calls Gym's HTTP services (`/run`, a standalone `/verify`, `/v1/responses`) over a clean contract. All three candidate frameworks do the opposite — they own the loop/env-server and pull the agent into their world. So this is **build-the-contract, not adopt-a-framework.** Pattern extraction = LOW–MEDIUM effort; wholesale adoption = HIGH.

## 3. How the three stack up (coverage = primary criterion; all three = HIGH disruption to adopt wholesale)

| | Harbor | verifiers | OpenEnv |
|---|---|---|---|
| Coverage (yes / partial / no) | 5 / 6 / 4 | 6 / 7 / 2 | 7 / 6 / 2 |
| Strongest at | 28-agent BYO catalog; MCP + Skills config | interception + **free RL token capture**; ships working Gym interop | MCP-bridge + skills-install + incremental ramp |
| Core gap | no HTTP `/run` or `/verify` | keeps loop + scorer internal | env-server; cap-and-reject concurrency |

**Mine, don't adopt:** verifiers → interception proxy + token capture (it already ships a NeMo-Gym client + harness); OpenEnv → MCP bridge + skills-install + reward-provenance + `validate` conformance tool; Harbor → BYO-agent catalog + MCP/Skills config. **None** addresses R15 (multi-tenant data lake) — greenfield.

## 4. Recommended architecture

**The agent contract — draw a hard line** (paste in monospace / Courier New):

```
 FACING THE AGENT (the contract)      |  INTERNAL TO GYM (hidden)
  - task prompt (NO answer key)       |   - verifier logic + verifier_metadata
  - model endpoint /v1/responses      |   - policy weights / vLLM / trainer loop
  - tools (HTTP /<tool> or MCP)       |   - token_ids / logprobs / mask capture
  - session handle (or opt-out)       |   - session state, cookies, Ray, Hydra
  - /verify: submit output -> reward   |   - tool implementation + sandbox
  - capacity / backpressure signal     |   - metric aggregation -> training
```

- **Who seeds:** a thin **Gym-side broker** seeds the session and runs `/verify`; the external agent only generates. The agent receives `{prompt, model URL + token, session handle, tools}` — **never `verifier_metadata`.** (Today's internal `SimpleAgent` does seed + loop + verify all together; externally we split it so the answer key and session state stay on Gym's trusted side.)
- **Black-box RL:** you can't train weights you don't own. Point the black-box agent's model URL at a **trainable policy Gym hosts** and capture `token_ids`/`logprobs` at the model-call boundary via an **interception proxy** (rollout-tagged, mask-aware, on-policy). Gym already emits the token primitives; the proxy + contract are the net-new build. Caveat: you can only train tokens *your* policy generated; opaque agent loops can break on-policy attribution.

## 5. Decisions we need to make today
- **D1 — Build-native vs adopt-a-framework.** (Rec: build native + mine patterns.)
- **D2 — Control model:** Gym-side broker (agent only generates) vs external agent owns `/run`. (Rec: broker.)
- **D3 — Black-box RL scope (biggest call):** commit to the interception proxy now, or scope v1 to *cooperative/white-box* harnesses (agent reports its own token_ids/logprobs) + eval-only for true black boxes?
- **D4 — MCP + Skills:** adopt MCP as the tool/skill exposure for the Siemens surface — in this epic or a follow-on?
- **D5 — `verifier_metadata` trust split:** confirm Gym holds it and owns `/verify` (agent never sees it).
- **D6 — R15 (multi-tenant data lake / auth):** in this epic or a separate track?
- **D7 — Phasing & owners.**

_What we are NOT deciding today: proxy internals, weight-sync mechanics with NeMo-RL, exact schemas — take offline._

## 6. Proposed phasing (straw man for D7)
1. **v0:** clean versioned `/run` + standalone `/verify` + model-server discovery/auth + config-only entry (covers most of #1396).
2. **v1:** MCP tool/skill exposure (Siemens) + conformance `validate` tool.
3. **v2:** interception proxy for black-box RL token capture.
4. **Separate track:** multi-tenant data lake / auth (R15).

## Appendix — detailed artifacts
In `/opt/Gym/external-agent-eval/`: `RECOMMENDATION.md`, `harbor-review.md`, `verifiers-review.md`, `openenv-review.md`.
In `/opt/Gym/diagrams/`: per-framework ASCII call-flows `{harbor,verifiers,openenv}-flow.md`.
Requirements: `/opt/Gym/issue-1396.md`.
