# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare long_transduction benchmark dataset.

Five prompt variants per difficulty tier:

  Arithmetic chain summing (3 variants per max_operands × N_SAMPLES):
    - "unnumbered_streaming_sum" : plain expressions in order
    - "streaming_sum"            : "[N]<expr>" in order
    - "shuffled_streaming_sum"   : "[N]<expr>" shuffled in input

  Per-line UUID sort (2 variants per uuids_per_line × N_SAMPLES):
    - "streaming_uuid_sort"          : "[N](u),(u),..." in order; model sorts
                                       UUIDs within each line by hex order.
    - "shuffled_streaming_uuid_sort" : same but line order is shuffled in input.

Each row carries a `type` field so the resource server selects the right
parser. Sum variants share an `expressions` payload; uuid_sort variants share
a `uuid_lines` payload.

Usage:
    python prepare.py
    python prepare.py --force   # regenerate even if output already exists
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "long_transduction.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_UNNUMBERED_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Your task is to output each expression and the result of evaluating the expression.

Example Input:

5+6
2+4-1
4+8-3+2

Example Output:

5+6=11
2+4-1=5
4+8-3+2=11

Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the input:
{input}"""

PROMPT_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Each expression is preceded by a numeric index in brackets like [1], [2], [3], ...
Your task is to output each expression with its index and the result of evaluating the expression.

Example Input:

[1]5+6
[2]2+4-1
[3]4+8-3+2

Example Output:

[1]5+6=11
[2]2+4-1=5
[3]4+8-3+2=11

Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the input:
{input}"""

PROMPT_SHUFFLED_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Each expression is preceded by a numeric index in brackets like [1], [2], [3], ...
The input expressions are SHUFFLED — they appear in arbitrary order, not in numerical order.
Your task is to output each expression with its index and the result, IN ASCENDING ORDER OF INDEX, starting at [1].

Example Input (shuffled):

[2]2+4-1
[3]4+8-3+2
[1]5+6

Example Output (in ascending order of index):

[1]5+6=11
[2]2+4-1=5
[3]4+8-3+2=11

Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the input:
{input}"""

PROMPT_STREAMING_UUID_SORT = """You will be given a long list of numbered lines. Each line has the form:

[N]hex,hex,hex,...

where each token is an 8-character lowercase hex string (for example: a1b2c3d4).

Your task is to output each line with its index and the same hex tokens sorted in ASCENDING LEXICOGRAPHIC ORDER (compare them as plain strings).

Example Input:

[1]c0a8e1d2,a1b2c3d4,b1c2d3e4
[2]f0e1d2c3,01234567

Example Output:

[1]a1b2c3d4,b1c2d3e4,c0a8e1d2
[2]01234567,f0e1d2c3

Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the input:
{input}"""

PROMPT_SHUFFLED_STREAMING_UUID_SORT = """You will be given a long list of numbered lines. Each line has the form:

[N]hex,hex,hex,...

where each token is an 8-character lowercase hex string (for example: a1b2c3d4).

The input lines are SHUFFLED — they appear in arbitrary order, not in numerical order.

Your task is to output each line with its index and the same hex tokens sorted in ASCENDING LEXICOGRAPHIC ORDER (compare them as plain strings), AND emit the lines themselves in ASCENDING ORDER OF [N] starting at [1].

Example Input (lines shuffled):

[2]f0e1d2c3,01234567
[1]c0a8e1d2,a1b2c3d4,b1c2d3e4

Example Output (lines in ascending [N] order, hex tokens sorted per line):

[1]a1b2c3d4,b1c2d3e4,c0a8e1d2
[2]01234567,f0e1d2c3

Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the input:
{input}"""

PROMPT_TEMPLATES = {
    "unnumbered_streaming_sum":     PROMPT_UNNUMBERED_STREAMING_SUM,
    "streaming_sum":                PROMPT_STREAMING_SUM,
    "shuffled_streaming_sum":       PROMPT_SHUFFLED_STREAMING_SUM,
    "streaming_uuid_sort":          PROMPT_STREAMING_UUID_SORT,
    "shuffled_streaming_uuid_sort": PROMPT_SHUFFLED_STREAMING_UUID_SORT,
}

SUM_TYPES = ["unnumbered_streaming_sum",
             "streaming_sum", "shuffled_streaming_sum"]
UUID_SORT_TYPES = ["streaming_uuid_sort", "shuffled_streaming_uuid_sort"]
NUMBERED_TYPES = {
    "streaming_sum",
    "shuffled_streaming_sum",
    "streaming_uuid_sort",
    "shuffled_streaming_uuid_sort",
}

TARGET_TOKENS = 50_000
N_SAMPLES = 5
# Difficulty knob — interpreted as max operands per expression for sum types,
# and as exact UUIDs per line for uuid_sort types.
# MAX_OPERANDS_RANGE = [4, 6, 8]
MAX_OPERANDS_RANGE = [4]


def _get_encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic chain generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_expression(max_operands: int) -> tuple[str, int]:
    n = random.randint(2, max_operands)
    operands = [random.randint(0, 9) for _ in range(n)]
    operators = [random.choice(["+", "-"]) for _ in range(n - 1)]
    parts = [str(operands[0])]
    for op, operand in zip(operators, operands[1:]):
        parts.append(op + str(operand))
    expr = "".join(parts)
    result = operands[0]
    for op, operand in zip(operators, operands[1:]):
        result = result + operand if op == "+" else result - operand
    return expr, result


def _generate_expressions(max_operands: int, enc) -> tuple[list[dict], int]:
    """Generate as many expressions as fit a TARGET_TOKENS-budget prompt.

    Budget is computed against the longest sum-prompt header (shuffled).
    """
    longest_header = max(
        (PROMPT_TEMPLATES[t].split("{input}")[0] for t in SUM_TYPES),
        key=len,
    )
    overhead = len(enc.encode(longest_header))
    budget = TARGET_TOKENS - overhead

    expressions: list[dict] = []
    used_tokens = 0
    while True:
        expr, answer = _generate_expression(max_operands)
        line_tokens = len(enc.encode(f"[{len(expressions) + 1}]{expr}\n"))
        if used_tokens + line_tokens > budget:
            break
        expressions.append({"expr": expr, "answer": answer})
        used_tokens += line_tokens
    return expressions, overhead + used_tokens


def _build_sum_sample(
    expressions: list[dict],
    max_operands: int,
    sample_type: str,
    approx_prompt_tokens: int,
    rng: random.Random,
) -> dict:
    """Render one arithmetic-chain row."""
    numbered = list(enumerate(expressions, start=1))

    if sample_type == "unnumbered_streaming_sum":
        input_text = "\n".join(e["expr"] for _, e in numbered)
        expected_output = "\n".join(
            f"{e['expr']}={e['answer']}" for _, e in numbered
        )
    elif sample_type in {"streaming_sum", "shuffled_streaming_sum"}:
        if sample_type == "shuffled_streaming_sum":
            input_order = numbered.copy()
            rng.shuffle(input_order)
        else:
            input_order = numbered
        input_text = "\n".join(f"[{n}]{e['expr']}" for n, e in input_order)
        expected_output = "\n".join(
            f"[{n}]{e['expr']}={e['answer']}" for n, e in numbered
        )
    else:
        raise ValueError(f"unknown sum sample_type: {sample_type}")

    prompt = PROMPT_TEMPLATES[sample_type].replace("{input}", input_text)
    return {
        "type": sample_type,
        "prompt": prompt,
        "expected_output": expected_output,
        "expressions": expressions,
        "n_expressions": len(expressions),
        "max_operands": max_operands,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UUID-sort generation
# ─────────────────────────────────────────────────────────────────────────────

def _new_uuid(rng: random.Random) -> str:
    """Deterministic 8-char lowercase hex token drawn from the provided RNG.

    We use just the first segment of a uuid4 (32 bits, 8 hex chars) so the
    tokens are short enough to fit many lines per 100K-token prompt while
    still being unique enough across the dataset (~1 in 4B collision odds).
    """
    return f"{rng.getrandbits(32):08x}"


def _generate_uuid_lines(
    uuids_per_line: int,
    enc,
    seed_key: tuple,
) -> tuple[list[list[str]], int]:
    """Generate UUID lines that fit a TARGET_TOKENS-budget prompt.

    Each inner list is the canonical (input-presentation) UUIDs for that
    line. Token budget is computed against the longer (shuffled) header.
    """
    longest_header = max(
        (PROMPT_TEMPLATES[t].split("{input}")[0] for t in UUID_SORT_TYPES),
        key=len,
    )
    overhead = len(enc.encode(longest_header))
    budget = TARGET_TOKENS - overhead

    rng = random.Random(seed_key)
    lines: list[list[str]] = []
    used_tokens = 0
    while True:
        idx = len(lines) + 1
        uuids = [_new_uuid(rng) for _ in range(uuids_per_line)]
        line_str = f"[{idx}]" + ",".join(uuids) + "\n"
        line_tokens = len(enc.encode(line_str))
        if used_tokens + line_tokens > budget:
            break
        lines.append(uuids)
        used_tokens += line_tokens
    return lines, overhead + used_tokens


def _build_uuid_sample(
    uuid_lines: list[list[str]],
    uuids_per_line: int,
    sample_type: str,
    approx_prompt_tokens: int,
    rng: random.Random,
) -> dict:
    """Render one UUID-sort row.

    `uuid_lines` is the canonical (input-presentation) per-line UUIDs. The
    UUIDs are deliberately NOT pre-sorted — that's the model's task. For the
    shuffled variant, the LINE order in the input is shuffled but the model
    must still emit ascending [N].
    """
    numbered = list(enumerate(uuid_lines, start=1))

    if sample_type == "streaming_uuid_sort":
        input_order = numbered
    elif sample_type == "shuffled_streaming_uuid_sort":
        input_order = numbered.copy()
        rng.shuffle(input_order)
    else:
        raise ValueError(f"unknown uuid sample_type: {sample_type}")

    def _fmt_line(n: int, uuids: list[str]) -> str:
        return f"[{n}]" + ",".join(uuids)

    input_text = "\n".join(_fmt_line(n, uuids) for n, uuids in input_order)
    expected_output = "\n".join(
        _fmt_line(n, sorted(uuids)) for n, uuids in numbered
    )
    prompt = PROMPT_TEMPLATES[sample_type].replace("{input}", input_text)

    return {
        "type": sample_type,
        "prompt": prompt,
        "expected_output": expected_output,
        "uuid_lines": uuid_lines,
        "n_lines": len(uuid_lines),
        "uuids_per_line": uuids_per_line,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────

def generate(force: bool = False) -> None:
    if OUTPUT_FPATH.exists() and not force:
        count = sum(1 for line in OUTPUT_FPATH.open() if line.strip())
        print(
            f"long_transduction benchmark already exists: {count} examples in {OUTPUT_FPATH}"
        )
        return

    enc = _get_encoder()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUTPUT_FPATH.open("w") as out:
        # Arithmetic chain: 3 variants per (max_operands, sample_idx).
        for max_operands in MAX_OPERANDS_RANGE:
            print(
                f"Generating sum max_operands={max_operands} ({N_SAMPLES} pairs)...")
            for i in range(N_SAMPLES):
                expressions, n_tokens = _generate_expressions(
                    max_operands, enc)
                for sample_type in SUM_TYPES:
                    rng = random.Random((max_operands, i, sample_type))
                    sample = _build_sum_sample(
                        expressions, max_operands, sample_type, n_tokens, rng
                    )
                    out.write(json.dumps(sample) + "\n")
                    total += 1
                print(
                    f"  sum[{i + 1:2d}/{N_SAMPLES}] "
                    f"{len(expressions)} expressions, ~{n_tokens:,} tokens, "
                    f"emitted {len(SUM_TYPES)} variants"
                )

        # UUID sort: 2 variants per (uuids_per_line, sample_idx). The same
        # MAX_OPERANDS_RANGE knob is reused — for these types it is the exact
        # count of UUIDs per line.
        for uuids_per_line in MAX_OPERANDS_RANGE:
            print(
                f"Generating uuid_sort uuids_per_line={uuids_per_line} "
                f"({N_SAMPLES} pairs)..."
            )
            for i in range(N_SAMPLES):
                uuid_lines, n_tokens = _generate_uuid_lines(
                    uuids_per_line, enc, seed_key=("uuid", uuids_per_line, i)
                )
                for sample_type in UUID_SORT_TYPES:
                    rng = random.Random((uuids_per_line, i, sample_type))
                    sample = _build_uuid_sample(
                        uuid_lines, uuids_per_line, sample_type, n_tokens, rng
                    )
                    out.write(json.dumps(sample) + "\n")
                    total += 1
                print(
                    f"  uuid[{i + 1:2d}/{N_SAMPLES}] "
                    f"{len(uuid_lines)} lines × {uuids_per_line} uuids, "
                    f"~{n_tokens:,} tokens, emitted {len(UUID_SORT_TYPES)} variants"
                )

    print(f"Done. Wrote {total} examples to {OUTPUT_FPATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if output exists"
    )
    args = parser.parse_args()
    generate(force=args.force)


if __name__ == "__main__":
    main()
