# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate arithmetic problems for the step_arithmetic env.

Usage:
    python generate_data.py --count 4000 --out data/train.jsonl
    python generate_data.py --count 256  --out data/validation.jsonl --seed 1
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a step-by-step arithmetic agent. You will be given an arithmetic expression. "
    "Evaluate it as an ordered sequence of binary operations using the provided tools, "
    "respecting parentheses (innermost first) and standard left-to-right order otherwise. "
    "Issue exactly one tool call per step. Use the result of each tool call as an operand for the next. "
    "When the expression is fully reduced, call submit(answer) with the final value."
)


def _tool(name, desc, props):
    return {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),
            "additionalProperties": False,
        },
        "strict": True,
    }


NUM = {"type": "number"}
TOOLS = [
    _tool("add", "Return a + b.", {"a": NUM, "b": NUM}),
    _tool("mul", "Return a * b.", {"a": NUM, "b": NUM}),
    _tool("sub", "Return a - b.", {"a": NUM, "b": NUM}),
    _tool("submit", "Submit the final answer.", {"answer": NUM}),
]

OP_NAMES = {"+": "add", "*": "mul", "-": "sub"}
OPS_3 = ["+", "*", "-"]


def _rand_small(rng):
    return rng.randint(1, 9)


def _make_2step(rng):
    """((a OP1 b) OP2 c)."""
    a, b, c = _rand_small(rng), _rand_small(rng), _rand_small(rng)
    op1, op2 = rng.choice(OPS_3), rng.choice(OPS_3)
    v1 = _eval(a, op1, b)
    v2 = _eval(v1, op2, c)
    expr = f"({a}{op1}{b}){op2}{c}"
    steps = [
        {"op": OP_NAMES[op1], "a": float(a), "b": float(b)},
        {"op": OP_NAMES[op2], "a": float(v1), "b": float(c)},
    ]
    return expr, steps, float(v2)


def _make_3step(rng):
    """(((a OP1 b) OP2 c) OP3 d)."""
    a, b, c, d = (_rand_small(rng) for _ in range(4))
    op1, op2, op3 = (rng.choice(OPS_3) for _ in range(3))
    v1 = _eval(a, op1, b)
    v2 = _eval(v1, op2, c)
    v3 = _eval(v2, op3, d)
    expr = f"(({a}{op1}{b}){op2}{c}){op3}{d}"
    steps = [
        {"op": OP_NAMES[op1], "a": float(a), "b": float(b)},
        {"op": OP_NAMES[op2], "a": float(v1), "b": float(c)},
        {"op": OP_NAMES[op3], "a": float(v2), "b": float(d)},
    ]
    return expr, steps, float(v3)


def _make_4step(rng):
    """((((a OP1 b) OP2 c) OP3 d) OP4 e)."""
    a, b, c, d, e = (_rand_small(rng) for _ in range(5))
    op1, op2, op3, op4 = (rng.choice(OPS_3) for _ in range(4))
    v1 = _eval(a, op1, b)
    v2 = _eval(v1, op2, c)
    v3 = _eval(v2, op3, d)
    v4 = _eval(v3, op4, e)
    expr = f"((({a}{op1}{b}){op2}{c}){op3}{d}){op4}{e}"
    steps = [
        {"op": OP_NAMES[op1], "a": float(a), "b": float(b)},
        {"op": OP_NAMES[op2], "a": float(v1), "b": float(c)},
        {"op": OP_NAMES[op3], "a": float(v2), "b": float(d)},
        {"op": OP_NAMES[op4], "a": float(v3), "b": float(e)},
    ]
    return expr, steps, float(v4)


def _eval(x, op, y):
    if op == "+":
        return x + y
    if op == "*":
        return x * y
    if op == "-":
        return x - y
    raise ValueError(op)


_MAKERS = [_make_2step, _make_3step, _make_4step]
_MAKER_WEIGHTS = [1, 2, 1]


def _make_one(rng, problem_id):
    maker = rng.choices(_MAKERS, weights=_MAKER_WEIGHTS, k=1)[0]
    expr, steps, answer = maker(rng)
    return {
        "problem_id": problem_id,
        "expression": expr,
        "expected_steps": steps,
        "expected_answer": answer,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "step_arithmetic_simple_agent",
        },
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Evaluate: {expr}"},
            ],
            "tools": TOOLS,
            "parallel_tool_calls": False,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=4000)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    written = 0
    with out.open("w") as f:
        i = 0
        while written < args.count:
            entry = _make_one(rng, problem_id=written)
            key = entry["expression"]
            # De-duplicate identical expressions to avoid trivial overfit.
            if key in seen:
                i += 1
                if i > args.count * 10:
                    break  # exhausted unique combinations
                continue
            seen.add(key)
            f.write(json.dumps(entry) + "\n")
            written += 1
            i += 1
    print(f"wrote {written} entries to {out}")


if __name__ == "__main__":
    main()
