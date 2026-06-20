# External agents in NeMo-Gym — 3 integration approaches

High-level comparison for a team decision. Each approach is taken from one of the Tier-1
env+verifier frameworks we looked at. The question: **how should an external / black-box
agent (e.g. Claude Code) plug into NeMo-Gym?**

NeMo-Gym recap: **agent server** (orchestrates the model↔tool loop) · **model server**
(inference) · **resources server** (`verify()` → reward). The choice is *where the external
agent attaches* and *how we recover tokens for RL*.

---

## A. Per-agent adapter  — *Harbor model (and what NeMo-Gym does today)*

```
  ┌───────────────┐   spawn + prompt    ┌──────────────────────────────┐
  │ External agent│ ◄────────────────── │ NeMo-Gym AGENT SERVER        │
  │ binary        │                     │   adapter, 1 per agent        │
  │ (Claude Code) │ ──────────────────► │   (claude_code_agent,         │
  └──────┬────────┘   output/transcript │    harbor_agent, swe_agents)  │
         │ calls its OWN model          └───────────────┬──────────────┘
         ▼  (Anthropic API)                             │ parsed output
   [ BLACK BOX ]                                        ▼
                                            RESOURCES SERVER  verify() ─► reward

  RL tokens: the adapter must surface token_ids/logprobs — hard for a true black box.
```

## B. API interception / proxy  — *Prime Intellect verifiers model*

```
  ┌───────────────┐   LLM API calls     ┌──────────────────────────────┐
  │ External agent│ ──────────────────► │ Interception proxy            │
  │ (ANY agent,   │                     │  • looks like OpenAI/Anthropic │
  │  unmodified)  │ ◄────────────────── │  • each call = 1 env step      │
  └───────────────┘   completion        │  • captures tokens/logprobs    │
     BASE_URL = proxy                    └──────┬─────────────────┬──────┘
                                                │ forwards         │ trajectory
                                                ▼                  ▼
                                        MODEL SERVER         RESOURCES SERVER
                                        (the trained model)  rubric/verify ─► reward

  RL tokens: captured FOR FREE by the proxy. Zero per-agent code.
```

## C. MCP env-as-server  — *Meta/HF OpenEnv model*

```
  ┌───────────────┐   MCP tool calls    ┌──────────────────────────────┐
  │ External agent│ ──────────────────► │ NeMo-Gym env exposed as an    │
  │ (MCP client,  │                     │ MCP SERVER (bridge)           │
  │  e.g. Codex)  │ ◄────────────────── │   each tool call = env.step    │
  └──────┬────────┘   obs + reward      └───────────────┬──────────────┘
         │ calls its OWN model                          ▼
         ▼  (black box, for eval)             reward from env / resources

  RL training: separate WHITE-BOX path — model server owns sampling → tokens.
```

---

## Decision matrix

| | **A. Per-agent adapter** | **B. API interception** | **C. MCP env-server** |
|---|---|---|---|
| Upstream example | Harbor · *NeMo-Gym today* | PrimeIntellect verifiers | Meta/HF OpenEnv |
| Per-agent code | 1 wrapper **per agent** | none (agent-agnostic) | none if agent speaks MCP |
| Agent must… | have a CLI/entrypoint | use OpenAI/Anthropic base_url | be an MCP client |
| RL token capture | agent must return token_ids (hard) | **free** (proxy captures) | separate white-box path |
| Control / determinism | high | medium (turns inferred from calls) | high (explicit tool calls) |
| New infra in NeMo-Gym | none (extends agent servers) | a proxy model-server | an MCP bridge layer |
| Maintenance | O(#agents) | O(1) | O(1) + MCP upkeep |
| Strongest for | few agents, full control | **many agents + cheap RL tokens** | agents standardizing on MCP |

---

## Recommendation framing

- **Already have A.** `claude_code_agent` (shells out to `claude -p`) and `harbor_agent`
  prove the adapter path works. Lowest lift, but you pay per-agent maintenance and hit the
  token-capture problem for true black boxes in RL.
- **B is the scalable RL bet.** Agent-agnostic *and* captures tokens for free — best if the
  goal is "support any external agent and actually train against it." Cost: build a proxy
  model-server and rely on agents using a standard LLM API.
- **C is the cleanest for eval, weakest for RL today.** Great protocol boundary and
  future-facing as agents adopt MCP, but training needs a separate white-box path.

**Suggested decision:** keep A for breadth of eval coverage, and invest in **B** as the
RL-training integration. Revisit C when MCP-native agents become common.
