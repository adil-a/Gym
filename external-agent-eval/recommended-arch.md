# Recommended NeMo-Gym external-agent architecture (#1396)

_The "build-the-contract, don't-adopt-a-framework" design. `◄── NEW` marks net-new components; everything else exists today. `▼`/`►` = one call, `▲` = return._

## 1. Components & trust boundary

```
  ┌─────────────────────────────────────────────────┐
  │  RL trainer  /  ng_collect_rollouts               │   owns policy weights θ
  └───────────────────────┬───────────────────────────┘
                          │  POST /run   (task row, incl verifier_metadata)
                          ▼
  ┌─────────────────────────────────────────────────┐
  │  GYM-SIDE BROKER                          ◄── NEW │   ── trusted / internal ──
  │   owns /run · seeds session · withholds           │
  │   verifier_metadata · calls /verify ·             │
  │   assembles the training trajectory               │
  └───────────────────────┬───────────────────────────┘
            scoped task    │  ▲  final output  →  reward + trajectory
       {prompt, model URL  ▼  │
        +token, session,      │
        tools}  (NO metadata) │
  ┌─────────────────────────────────────────────────┐
  │  EXTERNAL AGENT   (its own repo)                  │   ── untrusted ──
  │  black- or white-box · runs its OWN loop ·        │
  │  ONLY GENERATES                                   │
  └───┬─────────────────────────────────────┬─────────┘
      │ model calls                          │ tool calls (MCP, carry session)
      ▼                                      ▼
  ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │  MODEL SERVER  (trainable θ)  │    │  MCP GATEWAY  /mcp     ◄── NEW │
  │   2a direct, OR               │    │  Streamable HTTP · async ·     │
  │   2b via interception proxy   │    │  tools/list · tools/call       │
  │        ◄── NEW (black-box)    │    │  (or plain POST /<tool>)       │
  └──────────────────────────────┘    └──────────────┬───────────────┘
                                                      │ POST /<tool>
                                                      ▼ (internal · aiohttp)
                                       ┌──────────────────────────────┐
                                       │  RESOURCES SERVER (internal)  │
                                       │  /seed_session · /<tool>      │
                                       │  /verify (+ verifier_metadata)│
                                       └──────────────────────────────┘
```

## 2. Per-rollout flow

```
POST /run  (task row + verifier_metadata)                 [trainer → broker]
   │
   ├─ 1. /seed_session                                     [broker → resources]   ◄ GYM SEEDS
   │        ← session handle
   │
   ├─ 2. hand off a SCOPED task                            [broker → agent]
   │        {prompt, model URL + token, session, tools}    (NO verifier_metadata)
   │
   │   3. agent runs its OWN loop and GENERATES:           [agent]
   │        /v1/responses   ↔   tools (MCP /mcp or /<tool>)
   │        (2a direct,  or  2b through the proxy)
   │        ← final output
   │
   ├─ 4. /verify  (agent output + verifier_metadata)       [broker → resources]   ◄ GYM VERIFIES
   │        ← reward
   │
   └─ trajectory {tokens, logprobs, masks, reward}  →  trainer
```

The broker keeps **seeding** and **verifying** (and the answer key); only **generation** crosses the boundary to the agent.

## 3. Model access — two tiers

```
 2a  COOPERATIVE  (default — what #1396 actually asks for)
       agent ──/v1/responses──► model server (θ)
       agent REPORTS token_ids / logprobs / masks ──► broker        [no proxy]

 2b  BLACK-BOX fallback  (opaque agent that won't report tokens)     ◄── NEW
       agent ──base_url──► INTERCEPTION PROXY ──► model server (θ)
                           proxy CAPTURES token_ids / logprobs,
                           tags the rollout, rebuilds masks ──► broker
       limits: only trains tokens YOUR policy generated; agent must
               be redirectable to your endpoint, else eval-only
```

## 4. The contract (what each side sees)

```
 FACING THE AGENT (the contract)      |  INTERNAL TO GYM (hidden)
  - task prompt (NO answer key)       |   - verifier logic + verifier_metadata
  - model endpoint /v1/responses      |   - policy weights / vLLM / trainer loop
  - tools: MCP /mcp  or  plain /<tool> |   - token_ids / logprobs / mask capture
  - session handle (or opt-out)       |   - session state, cookies, Ray, Hydra
  - /verify: submit output -> reward   |   - tool implementation + sandbox
  - capacity / backpressure signal     |   - metric aggregation -> training
```

## 5. What's new vs. reused vs. mined

- **NEW (build):** the **Gym-side broker** (owns `/run`, seeds, withholds `verifier_metadata`, verifies, assembles trajectory); a **versioned external HTTP contract**; an **MCP gateway** (`/mcp`, Streamable HTTP, async — see §6); the **interception proxy** (only for tier 2b); model-server **discovery + auth** for external callers; HTTP-level **backpressure**.
- **REUSED (exists today):** model server (already emits `prompt_token_ids`/`generation_token_ids`/`generation_log_probs`), resources server (`/seed_session`, `/<tool>`, `/verify`), `ng_collect_rollouts` / aggregate-metrics path.
- **MINED (patterns, don't adopt wholesale):** verifiers → the interception + token-capture trick (it already ships a NeMo-Gym client/harness) for tier 2b; OpenEnv → `SessionMCPBridge` *logic* + `validate` conformance + reward-provenance guard; Harbor → BYO-agent catalog, MCP-server config, and `skills_dir` copy-in (the optional env-skill delivery below).
- **NOT Gym's job:** the **Agent-Skills publishing/auto-discovery framework** (second half of R14) belongs to the **customer's agent platform** (e.g. Siemens Fuse Agent), *not* Gym. Skills are content the agent loads. Gym's only skills touchpoint is **optional**: let an *environment author* bundle a `SKILL.md` for that env's own tools and have the contract deliver it (Harbor `skills_dir` copy-in).
- **GREENFIELD (truly net-new Gym work):** access-controlled data lake / multi-tenant (R15) — no candidate addresses it.

## 6. Exposing tools over MCP

The Siemens / Claude-Code / Cursor clients speak MCP, and they're remote — so tools are exposed over **MCP Streamable HTTP** (the `/mcp` endpoint). stdio is local-only; the older HTTP+SSE transport is deprecated.

```
 EXTERNAL AGENT (MCP client)
   │  Streamable HTTP → POST /mcp   (JSON-RPC: initialize · tools/list · tools/call)
   │  headers: Authorization: Bearer <token>,  Mcp-Session-Id ↔ Gym session handle
   ▼
 MCP GATEWAY            ◄── NEW   (on the broker / facing side)
   │  tools/list            → resources server's tool registry (THIS env's schemas)
   │  tools/call(name,args) → internal POST /<tool>   (Gym aiohttp client, carries session)
   ▼
 RESOURCES SERVER (internal; tool implementations unchanged)
```

- **It's a protocol translator, not a rewrite** — `tools/call` maps onto the existing `POST /<tool>`; implementations don't change.
- **On the facing side (broker/gateway), not the resources server directly** — keeps the internal server off the network and centralizes auth + MCP-session↔Gym-session mapping + per-rollout routing.
- **Async, NOT a stdlib `ThreadingHTTPServer`** — serve `/mcp` from FastAPI/Starlette and proxy the internal hop with Gym's global aiohttp client (httpx is banned), so it meets R11's 4k–65k concurrency. (OpenEnv's bridge uses a threaded stdlib server — mine its logic, not its server.)
- **Keep plain `POST /<tool>` too** — MCP is the front door for MCP-native clients; non-MCP harnesses use the plain HTTP tool contract. Same tools, two doors (satisfies R5 + R14 without coupling them).
