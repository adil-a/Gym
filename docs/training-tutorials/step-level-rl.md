(training-step-level-rl)=

# Step-Level RL with NeMo RL Primitives

Implements step-level REINFORCE on `step_arithmetic` by composing NeMo RL primitives directly (no `grpo_train` fork).

## Data contract

Env-side `verify()` extends `BaseVerifyResponse`:

```python
class StepArithmeticVerifyResponse(BaseVerifyResponse):
    reward: float
    step_rewards: list[float]   # aligned 1:1 with model-generated outputs
    correct_steps: int
    total_expected_steps: int
    final_answer: float | None
    submitted: bool
```

`step_rewards[i]` matches assistant entry `i` in NeMo-RL's `message_log`. The trainer zips assistant turns with `step_rewards`.

Trainer-side: standard `BatchedDataDict` consumed by `ClippedPGLossFn`. Per-token `advantages[t] = step_reward[turn(t)] - baseline` on assistant tokens, zero elsewhere.

## Loop structure

```text
for batch in dataloader:
    refit_policy_generation(policy, vLLM)
    rollout = run_async_nemo_gym_rollout(...)
    step_rewards = [r["step_rewards"] for r in rollout.final_batch["full_result"]]
    baseline = batch_mean(step_rewards)
    advantages = per-token from (step_reward - baseline) on assistant tokens
    train_data = BatchedDataDict(input_ids, token_mask, advantages, prev_logprobs, ...)
    policy.train(train_data, ClippedPGLossFn)
```

## Primitive mapping

| Verb | NeMo RL call |
|---|---|
| sample | `policy_generation.generate()` (via NeMo Gym agent server) |
| score | `policy.get_logprobs()`, `policy.get_reference_policy_logprobs()` |
| forward + backward + step | `policy.train(data, loss_fn)` |

Training backend (Megatron-Core or DTensor) and generation backend (vLLM or sglang) are selected by config inside `setup()`.

## Run

```bash
uv run examples/research/run_step_reinforce.py \
    --config examples/research/configs/step_reinforce_step_arithmetic.yaml
```

Config defaults: Qwen2.5-1.5B-Instruct, Megatron-Core + vLLM colocated, single GPU.

## Files

- Env: `resources_servers/step_arithmetic/` (NeMo Gym)
- Loop: `examples/research/run_step_reinforce.py` (NeMo RL)
- Config: `examples/research/configs/step_reinforce_step_arithmetic.yaml` (NeMo RL)
- Launcher: `examples/research/launch_step_reinforce.sh` (NeMo RL, slurm)
