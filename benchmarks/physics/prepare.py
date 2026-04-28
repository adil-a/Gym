# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare PHYSICS data for NeMo Gym.

Downloads ``desimfj/PHYSICS::test`` from HuggingFace — the same dataset
NeMo Skills' ``physics`` benchmark uses — and rewrites it into Gym JSONL
shape with one row per problem.

The transformation is byte-identical to NeMo Skills'
``nemo_skills/dataset/physics/prepare.py``:

  * ``question`` is the upstream ``question`` field unchanged.
  * ``expected_answer`` flattens the upstream nested ``answer`` list-of-lists
    by stripping any pre-existing ``\\boxed{...}`` wrappers and re-wrapping
    the comma-joined contents in a single ``\\boxed{...}``.  This is the same
    ``process_answer()`` Skills applies.
  * ``solution`` / ``answer_type`` / ``difficulty`` / ``language`` are passed
    through verbatim (Skills exposes them through its evaluator's row
    metadata; the physics_judge resource server doesn't read them, but they
    stay in the JSONL for downstream analysis).
  * ``domain`` is a verbatim copy of the upstream ``domain`` field. Skills
    re-labels it ``subset_for_metrics`` so its evaluator picks it up; here
    the physics_judge server reads ``domain`` directly via
    ``compute_subset_metrics(subset_key="domain")``.
"""

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "physics_benchmark.jsonl"

# Upstream HF identifier — must match the same source NeMo Skills uses
# (nemo_skills/dataset/physics/prepare.py:63).
_HF_REPO = "desimfj/PHYSICS"
_HF_SPLIT = "test"


def _strip_boxed(s: str) -> str:
    r"""Remove a single outer \boxed{...} wrapper if present.

    Mirrors Skills' nemo_skills/dataset/physics/prepare.py::strip_boxed.
    """
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[7:-1]
    return s


def _process_answer(answer) -> str:
    r"""Flatten a list-of-lists of \boxed{...}-wrapped answers and re-wrap.

    The upstream PHYSICS dataset stores ``answer`` as a nested list (multiple
    parts × multiple acceptable forms per part). Skills' prepare.py strips
    each entry's pre-existing ``\boxed{...}`` and joins them with ``, `` into
    a single ``\boxed{...}``. We replicate that exactly so the
    ``expected_answer`` strings are byte-identical to Skills'.
    """
    all_answers = [_strip_boxed(item) for sublist in answer for item in sublist]
    return f"\\boxed{{{', '.join(all_answers)}}}"


def _format_entry(entry: dict) -> dict:
    """Convert one upstream PHYSICS row to the Gym JSONL shape."""
    return {
        "question": entry["question"],
        "expected_answer": _process_answer(entry["answer"]),
        # Verifier-irrelevant metadata — passed through for downstream
        # analysis, parity with Skills' prepare.py output, and to support
        # `compute_subset_metrics(subset_key="domain")` in the metrics
        # aggregation.
        "domain": entry["domain"],
        "difficulty": entry["difficulty"],
        "answer_type": entry["answer_type"],
        "language": entry["language"],
        "solution": entry["solution"],
    }


def prepare() -> Path:
    """Download the dataset, convert to Gym JSONL, return the output file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {_HF_REPO}::{_HF_SPLIT} from HuggingFace ...")
    dataset = load_dataset(_HF_REPO, split=_HF_SPLIT)

    # Skills splits this further into `test` (en), `zh` (zh), and `en_zh`
    # (full); the `test` split is the default for `ns eval`. We mirror the
    # default split (English-only) so the parity comparison is on the same
    # set of problems Skills' default `physics:N` benchmark exercises.
    en_data = [entry for entry in dataset if entry["language"] == "en"]

    count = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as out:
        for entry in en_data:
            out.write(json.dumps(_format_entry(entry), ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
