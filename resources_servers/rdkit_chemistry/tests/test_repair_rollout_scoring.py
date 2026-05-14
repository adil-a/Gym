# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RDKit rollout scoring repair utility."""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root

from resources_servers.rdkit_chemistry.scripts.repair_rollout_scoring import repair_rollout_directory


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_repair_rollout_directory_rescores_and_writes_metrics(tmp_path: Path):
    prompt = "Is there an amide group?\n\nCC(=O)N\n\nEnd your response with Answer Value: {0_or_1}."
    questions_path = tmp_path / "train.jsonl"
    _write_jsonl(
        questions_path,
        [
            {
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                "expected_answer": 1,
                "property_type": "presence",
                "property": "HasAmide",
                "method": "direct",
                "answer_format": "fmt_10",
                "chembl_id": "CHEMBLTEST",
                "smiles": "CC(=O)N",
            }
        ],
    )

    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    materialized_path = rollout_dir / "rollouts_chunk_000_materialized_inputs.jsonl"
    _write_jsonl(
        materialized_path,
        [
            {
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                "expected_answer": "1",
                "property_type": "presence",
                "property": "HasAmide",
                "method": "direct",
                "use_box_format": False,
                "chembl_id": "CHEMBLTEST",
                "smiles": "CC(=O)N",
                "agent_ref": {"name": "rdkit_chemistry_simple_agent"},
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
            }
        ],
    )
    rollouts_path = rollout_dir / "rollouts_chunk_000.jsonl"
    _write_jsonl(
        rollouts_path,
        [
            {
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "</think>\nAnswer Value: {1}"}],
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
                "reward": 0.0,
                "predicted_value": None,
                "correct": False,
                "property": "HasAmide",
                "property_type": "presence",
                "chembl_id": "CHEMBLTEST",
                "method": "direct",
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
            }
        ],
    )

    output_dir = tmp_path / "repaired"
    stats = repair_rollout_directory(
        questions_jsonl=questions_path,
        rollout_dir=rollout_dir,
        output_dir=output_dir,
    )

    repaired_rows = _read_jsonl(output_dir / "rollouts_chunk_000.jsonl")
    assert repaired_rows[0]["answer_format"] == "fmt_10"
    assert repaired_rows[0]["expected_answer"] == 1
    assert repaired_rows[0]["predicted_value"] == 1.0
    assert repaired_rows[0]["reward"] == 1.0
    assert repaired_rows[0]["correct"] is True

    repaired_materialized_rows = _read_jsonl(output_dir / "rollouts_chunk_000_materialized_inputs.jsonl")
    assert repaired_materialized_rows[0]["answer_format"] == "fmt_10"

    aggregate_metrics = json.loads((output_dir / "rollouts_chunk_000_aggregate_metrics.json").read_text())
    assert aggregate_metrics[0]["agent_ref"]["name"] == "rdkit_chemistry_simple_agent"
    assert aggregate_metrics[0]["agent_metrics"]["mean/reward"] == 1.0
    assert aggregate_metrics[0]["agent_metrics"]["direct"]["accuracy"] == 1.0

    summary = json.loads((output_dir / "repair_summary.json").read_text())
    assert summary["rollout_rows"] == 1
    assert summary["reward_changed"] == 1
    assert stats.corrected_rollouts == 1
