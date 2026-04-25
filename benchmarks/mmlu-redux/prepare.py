# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare MMLU-Redux 2.0 benchmark data for NeMo Gym (mcqa).

Ports NeMo-Skills' ``mmlu-redux`` benchmark (``edinburgh-dawg/mmlu-redux-2.0``).
Strategy adapted from ZeroEval / NeMo-Skills ``format_entry`` logic.
"""

import json
import uuid
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm


# From https://github.com/hendrycks/test/blob/master/categories.py (NeMo-Skills / Hendrycks MMLU).
MMLU_SUBJECT_TO_CATEGORY: dict[str, list[str]] = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

MMLU_SUBJECTS: tuple[str, ...] = tuple(MMLU_SUBJECT_TO_CATEGORY.keys())

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "mmlu-redux_benchmark.jsonl"

HF_REPO = "edinburgh-dawg/mmlu-redux-2.0"


def format_entry(entry: dict, category: str) -> dict | None:
    if entry["error_type"] == "ok":
        final_answer = chr(65 + int(entry["answer"]))
    elif entry["error_type"] == "wrong_groundtruth" and entry["correct_answer"] in list("ABCD"):
        # NeMo-Skills had a typo ("correct_answer" literal); use the dataset's correct letter.
        final_answer = str(entry["correct_answer"]).strip().upper()
    else:
        return None

    choices = entry["choices"]
    if len(choices) != 4:
        return None
    letters = ["A", "B", "C", "D"]
    options = [{letters[i]: str(choices[i])} for i in range(4)]
    options_text = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(4))
    stem = (entry.get("question") or "").strip()
    seed = json.dumps({"category": category, "question": stem, "answer": final_answer}, sort_keys=True)
    row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
    subset = MMLU_SUBJECT_TO_CATEGORY[category][0]
    return {
        "question": stem,
        "options_text": options_text,
        "options": options,
        "expected_answer": final_answer,
        "subset_for_metrics": subset,
        "subject": category,
        "uuid": row_uuid,
    }


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for category in tqdm(MMLU_SUBJECTS, desc="MMLU-Redux categories"):
            dataset = load_dataset(HF_REPO, name=category, split="test")
            for entry in dataset:
                row = format_entry(entry, category)
                if row is None:
                    continue
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
