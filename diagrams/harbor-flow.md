# Harbor — external-agent call flow

_agent: Claude Code (unmodified `claude` CLI) · each ▼/► = one call, ▲ = return_

```
hb tasks run --agent claude-code
   │ create_agent_from_config()
   ▼
AgentFactory
   │ create_agent_from_name() | create_agent_from_import_path()
   ▼
ClaudeCode  (BaseInstalledAgent → BaseAgent)
   │ Trial._prepare()
   ▼
Trial
   │ create_environment_from_config()
   ▼
BaseEnvironment  (Docker/Daytona/Modal/E2B/GKE/local)
   │ .start(force_build)
   │ agent.setup() ► install()                 # npm i -g @anthropic-ai/claude-code
   │ _run_agent_phase() ► agent.run(instruction)
   ▼
ClaudeCode.run()
   │ exec_as_agent()                           # (a) config setup
   │ exec_as_agent()                           # (b) claude --output-format=stream-json --print -- <instr>
   ▼
[ claude CLI = BLACK BOX ] ──own model @ ANTHROPIC_BASE_URL──┐
   │ issues shell commands                                   │
   ▼                                                         │
BaseEnvironment.exec(cmd)                                    │
   ▲ ExecResult{stdout, stderr, return_code} ────────────────┘ loop
   │ run() returns
   ▼
Trial._run_verifier()
   │ Verifier.verify() ► env.exec(test.sh)
   ▼
/logs/verifier/reward.json | reward.txt
   │ _parse_reward_json() | _parse_reward_text()
   ▼
VerifierResult.rewards["reward"]
   │ TrialResult
   ▼
eval: score        RL: HarborRolloutInterface.run() ► Rollout ► SkyRL
```

**files** — `cli/tasks.py`·`cli/jobs.py` (`--agent`/`--agent-import-path`) · `agents/factory.py` (`AgentFactory`,`_AGENT_MAP`) · `agents/installed/claude_code.py` (`ClaudeCode.run/install`) · `agents/installed/base.py` (`exec_as_agent`) · `environments/base.py` (`BaseEnvironment.exec`→`ExecResult`) · `trial/trial.py` (`_prepare`/`_run_agent_phase`/`_run_verifier`) · `verifier/verifier.py` (`verify`,`_parse_reward_*`) · `docs/.../rl.mdx` (`HarborRolloutInterface`)

- **RL tokens:** none native (black box) → need vLLM interception on `ANTHROPIC_BASE_URL`, or agent returns `token_ids`/`mask_ids` in `agent_result.metadata`.
- **Reward:** decoupled `test.sh` → `reward.json`-first/`.txt`-fallback → `VerifierResult`.
- **NeMo-Gym build:** container `exec` backend + CLI-shelling agent server + `verify()` runs the test script in-container.
