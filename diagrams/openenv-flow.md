# OpenEnv — external-agent call flow

_agent: Codex CLI (black-box) via MCP bridge · each ▼/► = one call, ▲ = return_

```
build_codex_runner()
   │ subprocess: codex exec --json …        # CODEX_HOME=tmp; codex mcp add <env> --url …
   ▼
[ Codex CLI = BLACK BOX, own model ]
   │ MCP tools/call  (HTTP POST /mcp)
   ▼
SessionMCPHttpServer.do_POST()
   │ bridge.handle_request()
   ▼
SessionMCPBridge   ("tools/call")
   │ call_tool(name, args)
   ▼
StepEnvSessionAdapter.call_tool()
   │ action_builder(name, args) ► client.step(action)
   ▼
EnvClient.step(action)                      # WS /ws  (http:// → ws://)
   │ → Environment.step()
   ▼
Environment
   ▲ StepResult(observation, reward, done)
   │ → ToolResult{data, done, metadata["reward"]}
   └─► returns up to Codex                   # loop
   │ run_black_box() returns HarnessRolloutResult
   ▼
_finalize_episode()
   │ session.verify() ► _resolve_env_reward()
   ▼
EpisodeRecord
   │ CollectRunner ► RolloutSerializer.write_episode()
   ▼
results.jsonl
```

**files** — `examples/browsergym_codex_eval.py` (`build_codex_runner`) · `examples/browsergym_harness_eval_common.py` (`SessionMCPHttpServer`,`run_black_box_episode`,`_finalize_episode`) · `core/harness/__init__.py` (`CLIHarnessAdapter.run_black_box`,`SessionMCPBridge`,`StepEnvSessionAdapter`,`MCPHarnessAdapter.run_white_box`,`ModelStep`,`_resolve_env_reward`) · `core/env_client.py` (`EnvClient.step`) · `core/env_server/interfaces.py` (`Environment`) · `core/client_types.py` (`StepResult`) · `core/harness/collect.py` (`CollectRunner`)

- **RL tokens:** none on this path (black box). Token_ids/logprobs only via the separate **white-box** `MCPHarnessAdapter.run_white_box(model_step)` → `build_harness_rollout_func` (TRL). Codex path = eval-only.
- **Reward:** stays in the env — `StepResult.reward` → `ToolResult.metadata["reward"]`; `_resolve_env_reward` forwards it (raises if synthesized outside env).
- **NeMo-Gym build:** MCP bridge re-exposing a resources server as `tools/list`+`tools/call` + subprocess CLI agent server; eval-only.
