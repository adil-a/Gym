# verifiers — external-agent call flow

_agent: any unmodified CLI binary, pointed at an interception proxy · each ▼/► = one call, ▲ = return_

```
load_environment(env_id)
   │ → CliAgentEnv(MultiTurnEnv)
   ▼
MultiTurnEnv.rollout()
   │ setup_state()
   ▼
CliAgentEnv.setup_state()
   │ interception_server.start()
   │ register_rollout(id) → request_id_queue
   │ create_sandbox()
   │ start_agent()                      # OPENAI_BASE_URL / ANTHROPIC_BASE_URL → proxy
   ▼
[ external CLI agent = BLACK BOX, unmodified ]
   │ POST /rollout/{id}/v1/{chat/completions|completions|responses|messages}
   ▼
InterceptionServer._handle_request()    # 1 intercepted request = 1 step
   │ _poll_next_request() ► queue.get()
   ▼
MultiTurnEnv.rollout loop
   │ get_model_response() ► Environment.get_model_response() ► state["client"].get_response()
   ▼
served model (vLLM / OpenAI)
   ▲ parse_tokens() → prompt/completion token_ids + logprobs
   │ deliver_response() | synthesize_stream()      # completion back to agent
   └─► agent continues … exits → @vf.stop agent_completed
   │ Rubric.score_rollout()
   ▼
state["reward"]  (+ trajectory + token_ids/logprobs)
   ▼
RL: prime-rl          eval: vf-eval
```

**files** — `utils/env_utils.py` (`load_environment`) · `envs/experimental/cli_agent_env.py` (`CliAgentEnv.setup_state`/`build_env_vars`) · `utils/interception_utils.py` (`InterceptionServer._handle_request`,`deliver_response`,`synthesize_stream`) · `envs/multiturn_env.py` (`MultiTurnEnv.rollout`) · `envs/environment.py` (`get_model_response`) · `clients/openai_completions_client.py:148` (`parse_tokens`) · `rubrics/rubric.py:317` (`Rubric.score_rollout`)

- **RL tokens:** free — `parse_tokens` reads `prompt_token_ids`/`token_ids`/`logprobs` off the served model (`None` → eval-only fallback).
- **Reward:** `Rubric.score_rollout()` = weighted sum of `funcs` over the trajectory → `state["reward"]`.
- **NeMo-Gym build:** interception-proxy model-server (forward to `vllm_model`, capture token_ids/logprobs) + queue-based agent server + sandbox/tunnel.
