# step_arithmetic

Step-level RL environment for arithmetic expression evaluation.

## Task

Given an expression like `(3+5)*2-1`, the agent issues an ordered sequence of tool calls and submits the result.

Tools:

- `add(a, b)` returns `a + b`
- `mul(a, b)` returns `a * b`
- `sub(a, b)` returns `a - b`
- `submit(answer)` records the final answer

Each problem has a canonical step sequence. For `(3+5)*2-1`: `[add(3,5), mul(8,2), sub(16,1), submit(15)]`.

## Reward signal

`verify()` returns:

- `reward: float`: 1.0 if `submit(answer)` matches `expected_answer`, else 0.0.
- `step_rewards: list[float]`: one entry per model-generated output (function_call or assistant message), aligned 1:1 with assistant entries in NeMo-RL's `message_log`. Each entry is 1.0 if the tool call matches the canonical step at the same position, else 0.0.

## Data

- `data/example.jsonl` (5 problems, committed).
- `data/train.jsonl`, `data/validation.jsonl` (gitignored). Generate:

```bash
python generate_data.py --count 4000 --out data/train.jsonl --seed 0
python generate_data.py --count 256  --out data/validation.jsonl --seed 1
```

Each problem records its canonical step sequence in `expected_steps`.

## Licensing

Code: Apache 2.0
Data: Apache 2.0

Dependencies:
- nemo_gym: Apache 2.0
