# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repair RDKit chemistry rollout scoring after answer-format data healing.

Run from a NeMo Gym checkout, for example:

    uv run python resources_servers/rdkit_chemistry/scripts/repair_rollout_scoring.py \
        --questions-jsonl /path/to/train.jsonl \
        --rollout-dir /path/to/results/direct \
        --output-dir /path/to/repaired/direct

The input rollout directory is not modified unless ``--overwrite`` is used with
an output directory that already contains repaired files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.reward_profile import compute_aggregate_metrics
from resources_servers.rdkit_chemistry.app import (
    RDKitChemistryResourcesServer,
    compute_reward,
    extract_predicted_value,
)


ROLL_TO_MATERIALIZED_RE = re.compile(r"^rollouts_chunk_\d+\.jsonl$")
REPAIRED_SCHEMA_VERSION = 1


@dataclass
class QuestionIndex:
    by_prompt: dict[str, dict[str, Any]]
    by_metadata: dict[tuple[str, ...], dict[str, Any]]
    duplicate_prompt_count: int = 0
    duplicate_metadata_count: int = 0


@dataclass
class RepairStats:
    chunks: int = 0
    materialized_rows: int = 0
    rollout_rows: int = 0
    missing_rollout_rows: int = 0
    reward_changed: int = 0
    predicted_value_changed: int = 0
    corrected_rollouts: int = 0
    parsed_rollouts: int = 0
    answer_format_counts: Counter[str] = field(default_factory=Counter)
    property_type_counts: Counter[str] = field(default_factory=Counter)
    missing_aggregate_metric_inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPAIRED_SCHEMA_VERSION,
            "chunks": self.chunks,
            "materialized_rows": self.materialized_rows,
            "rollout_rows": self.rollout_rows,
            "missing_rollout_rows": self.missing_rollout_rows,
            "reward_changed": self.reward_changed,
            "predicted_value_changed": self.predicted_value_changed,
            "corrected_rollouts": self.corrected_rollouts,
            "parsed_rollouts": self.parsed_rollouts,
            "answer_format_counts": dict(sorted(self.answer_format_counts.items())),
            "property_type_counts": dict(sorted(self.property_type_counts.items())),
            "missing_aggregate_metric_inputs": self.missing_aggregate_metric_inputs,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _extract_prompt(row: dict[str, Any]) -> str:
    params = row.get("responses_create_params") or {}
    input_value = params.get("input")
    if isinstance(input_value, str):
        return input_value
    if not isinstance(input_value, list):
        return ""

    parts: list[str] = []
    for item in input_value:
        if not isinstance(item, dict):
            continue
        if item.get("role") not in {None, "user"}:
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


def _metadata_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(key, ""))
        for key in ("chembl_id", "smiles", "property", "property_type", "expected_answer", "method")
    )


def _same_question(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a.get("answer_format") == b.get("answer_format")
        and str(a.get("expected_answer", "")) == str(b.get("expected_answer", ""))
        and a.get("property_type") == b.get("property_type")
    )


def load_question_index(questions_jsonl: Path) -> QuestionIndex:
    by_prompt: dict[str, dict[str, Any]] = {}
    by_metadata: dict[tuple[str, ...], dict[str, Any]] = {}
    ambiguous_metadata: set[tuple[str, ...]] = set()
    duplicate_prompt_count = 0
    duplicate_metadata_count = 0

    for row_number, row in enumerate(_read_jsonl(questions_jsonl), start=1):
        answer_format = row.get("answer_format")
        if not isinstance(answer_format, str) or not answer_format:
            raise ValueError(f"{questions_jsonl}:{row_number} is missing string field 'answer_format'")

        prompt = _extract_prompt(row)
        if prompt:
            existing = by_prompt.get(prompt)
            if existing is None:
                by_prompt[prompt] = row
            elif _same_question(existing, row):
                duplicate_prompt_count += 1
            else:
                raise ValueError(f"{questions_jsonl}:{row_number} has a duplicate prompt with conflicting metadata")

        metadata = _metadata_key(row)
        if any(metadata):
            if metadata in ambiguous_metadata:
                duplicate_metadata_count += 1
            else:
                existing = by_metadata.get(metadata)
                if existing is None:
                    by_metadata[metadata] = row
                elif _same_question(existing, row):
                    duplicate_metadata_count += 1
                else:
                    by_metadata.pop(metadata, None)
                    ambiguous_metadata.add(metadata)
                    duplicate_metadata_count += 1

    if not by_prompt and not by_metadata:
        raise ValueError(f"No usable questions found in {questions_jsonl}")

    return QuestionIndex(
        by_prompt=by_prompt,
        by_metadata=by_metadata,
        duplicate_prompt_count=duplicate_prompt_count,
        duplicate_metadata_count=duplicate_metadata_count,
    )


def _lookup_question(row: dict[str, Any], index: QuestionIndex, source: Path) -> dict[str, Any]:
    prompt = _extract_prompt(row)
    if prompt in index.by_prompt:
        return index.by_prompt[prompt]

    metadata = _metadata_key(row)
    if metadata in index.by_metadata:
        return index.by_metadata[metadata]

    task_idx = row.get(TASK_INDEX_KEY_NAME)
    rollout_idx = row.get(ROLLOUT_INDEX_KEY_NAME)
    raise KeyError(
        "Could not match materialized input to healed questions: "
        f"{source} task={task_idx!r} rollout={rollout_idx!r} "
        f"chembl_id={row.get('chembl_id')!r} property={row.get('property')!r}"
    )


def _rollout_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row[TASK_INDEX_KEY_NAME]), int(row[ROLLOUT_INDEX_KEY_NAME])


def _extract_response_text(row: dict[str, Any]) -> str:
    response = row.get("response") or {}
    output = response.get("output") or []
    texts: list[str] = []
    if not isinstance(output, list):
        return ""

    for output_item in output:
        if not isinstance(output_item, dict):
            continue
        if output_item.get("type") not in {None, "message"}:
            continue
        if output_item.get("role") not in {None, "assistant"}:
            continue
        content = output_item.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return "\n".join(texts).strip()


def _merge_question_fields(row: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    for key in (
        "answer_format",
        "expected_answer",
        "property_type",
        "property",
        "chembl_id",
        "method",
        "smiles",
    ):
        if key in question:
            merged[key] = question[key]
    return merged


def _repair_rollout(row: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    repaired = _merge_question_fields(row, question)
    property_type = str(repaired.get("property_type", ""))
    answer_format = repaired.get("answer_format")
    predicted = extract_predicted_value(
        _extract_response_text(repaired),
        property_type,
        answer_format=answer_format,
        use_box_format=bool(repaired.get("use_box_format", False)),
    )
    reward = compute_reward(predicted, float(repaired["expected_answer"]), property_type=property_type)
    repaired["predicted_value"] = predicted
    repaired["reward"] = reward
    repaired["correct"] = reward == 1.0
    return repaired


def _agent_name(row: dict[str, Any]) -> str:
    agent_ref = row.get(AGENT_REF_KEY_NAME) or {}
    if isinstance(agent_ref, dict) and agent_ref.get("name"):
        return str(agent_ref["name"])
    return "agent"


def _aggregate_metrics_for_agents(rows: list[dict[str, Any]], agent_by_key: dict[tuple[int, int], str]) -> list[dict]:
    server = RDKitChemistryResourcesServer.__new__(RDKitChemistryResourcesServer)
    rows_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_agent[agent_by_key.get(_rollout_key(row), "agent")].append(row)

    entries: list[dict[str, Any]] = []
    for agent_name, agent_rows in sorted(rows_by_agent.items()):
        aggregate = compute_aggregate_metrics(
            agent_rows,
            compute_metrics_fn=server.compute_metrics,
            get_key_metrics_fn=server.get_key_metrics,
        )
        entries.append(
            {
                AGENT_REF_KEY_NAME: {"name": agent_name},
                "agent_metrics": aggregate.agent_metrics,
                "key_metrics": aggregate.key_metrics,
                "group_level_metrics": aggregate.group_level_metrics,
            }
        )
    return entries


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.name != ".gitkeep"]
    if existing and not overwrite:
        raise FileExistsError(f"Output directory {output_dir} is not empty; pass --overwrite to replace files")


def _output_path(output_dir: Path, input_path: Path, overwrite: bool) -> Path:
    path = output_dir / input_path.name
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")
    return path


def repair_rollout_directory(
    *,
    questions_jsonl: Path,
    rollout_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    drop_use_box_format: bool = False,
) -> RepairStats:
    questions_jsonl = questions_jsonl.expanduser().resolve()
    rollout_dir = rollout_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not questions_jsonl.is_file():
        raise FileNotFoundError(f"Healed questions file does not exist: {questions_jsonl}")
    if not rollout_dir.is_dir():
        raise NotADirectoryError(f"Rollout directory does not exist: {rollout_dir}")
    if output_dir == rollout_dir and not overwrite:
        raise ValueError("Refusing to write into the input rollout directory without --overwrite")

    _prepare_output_dir(output_dir, overwrite=overwrite)
    question_index = load_question_index(questions_jsonl)
    stats = RepairStats()

    rollout_paths = sorted(path for path in rollout_dir.iterdir() if ROLL_TO_MATERIALIZED_RE.match(path.name))
    if not rollout_paths:
        raise FileNotFoundError(f"No rollouts_chunk_NNN.jsonl files found in {rollout_dir}")

    for rollout_path in rollout_paths:
        materialized_path = rollout_path.with_name(f"{rollout_path.stem}_materialized_inputs.jsonl")
        if not materialized_path.exists():
            raise FileNotFoundError(f"Missing materialized inputs for {rollout_path}: {materialized_path}")

        materialized_rows = _read_jsonl(materialized_path)
        healed_materialized_rows: list[dict[str, Any]] = []
        question_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        agent_by_key: dict[tuple[int, int], str] = {}
        for row in materialized_rows:
            question = _lookup_question(row, question_index, materialized_path)
            healed_row = _merge_question_fields(row, question)
            if drop_use_box_format:
                healed_row.pop("use_box_format", None)
            key = _rollout_key(healed_row)
            question_by_key[key] = question
            agent_by_key[key] = _agent_name(healed_row)
            healed_materialized_rows.append(healed_row)

        rollout_rows = _read_jsonl(rollout_path)
        repaired_rows: list[dict[str, Any]] = []
        for row in rollout_rows:
            key = _rollout_key(row)
            question = question_by_key.get(key)
            if question is None:
                raise KeyError(f"{rollout_path} contains rollout key {key} with no matching materialized input")

            old_reward = row.get("reward")
            old_predicted = row.get("predicted_value")
            repaired = _repair_rollout(row, question)
            if drop_use_box_format:
                repaired.pop("use_box_format", None)
            repaired_rows.append(repaired)

            stats.reward_changed += old_reward != repaired.get("reward")
            stats.predicted_value_changed += old_predicted != repaired.get("predicted_value")
            stats.corrected_rollouts += repaired.get("reward") == 1.0
            stats.parsed_rollouts += repaired.get("predicted_value") is not None
            stats.answer_format_counts.update([str(repaired.get("answer_format", ""))])
            stats.property_type_counts.update([str(repaired.get("property_type", ""))])

        stats.chunks += 1
        stats.materialized_rows += len(materialized_rows)
        stats.rollout_rows += len(rollout_rows)
        stats.missing_rollout_rows += max(0, len(materialized_rows) - len(rollout_rows))

        _write_jsonl(_output_path(output_dir, materialized_path, overwrite), healed_materialized_rows)
        _write_jsonl(_output_path(output_dir, rollout_path, overwrite), repaired_rows)
        aggregate_path = rollout_path.with_name(f"{rollout_path.stem}_aggregate_metrics.json")
        aggregate_output_path = _output_path(output_dir, aggregate_path, overwrite)
        if repaired_rows:
            _write_json(aggregate_output_path, _aggregate_metrics_for_agents(repaired_rows, agent_by_key))
        else:
            stats.missing_aggregate_metric_inputs.append(rollout_path.name)

    summary = stats.to_dict()
    summary["questions_jsonl"] = str(questions_jsonl)
    summary["rollout_dir"] = str(rollout_dir)
    summary["output_dir"] = str(output_dir)
    summary["duplicate_prompt_count"] = question_index.duplicate_prompt_count
    summary["duplicate_metadata_count"] = question_index.duplicate_metadata_count
    _write_json(output_dir / "repair_summary.json", summary)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-jsonl", type=Path, required=True, help="Healed questions JSONL, e.g. train.jsonl")
    parser.add_argument("--rollout-dir", type=Path, required=True, help="Directory containing rollouts_chunk_*.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for repaired rollout artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in the output directory")
    parser.add_argument(
        "--drop-use-box-format",
        action="store_true",
        help="Remove legacy use_box_format from repaired rollout and materialized rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = repair_rollout_directory(
        questions_jsonl=args.questions_jsonl,
        rollout_dir=args.rollout_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        drop_use_box_format=args.drop_use_box_format,
    )
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()
