# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pure logic for the rollout viewer.

This module has no Streamlit dependency so it can be unit-tested without a UI
runtime. It is responsible for locating a run's sibling artifact files, loading
them, turning a rollout into a classified transcript, diffing two rollouts of
the same task, and flattening rollouts into a metrics table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


# Sibling-file suffixes, relative to a rollouts file's stem. A rollouts file is
# ``<stem>.jsonl``; its siblings append these suffixes to ``<stem>``.
AGGREGATE_SUFFIX = "_aggregate_metrics.json"
MATERIALIZED_SUFFIX = "_materialized_inputs.jsonl"
FAILURES_SUFFIX = "_failures.jsonl"

# Suffixes that identify a ``.jsonl`` as a *sibling* rather than a rollouts file,
# so directory scans don't mistake them for runs of their own.
_NON_ROLLOUT_JSONL_SUFFIXES = (MATERIALIZED_SUFFIX, FAILURES_SUFFIX)

# Known trajectory item kinds.
KIND_MESSAGE = "message"
KIND_TOOL_CALL = "function_call"
KIND_TOOL_OUTPUT = "function_call_output"
KIND_REASONING = "reasoning"
KIND_UNKNOWN = "unknown"


@dataclass
class RunRef:
    """The set of files that make up one run, derived from a rollouts file.

    Siblings are optional; ``None`` means the file does not exist on disk.
    """

    run_id: str
    rollouts: Path
    aggregate: Optional[Path] = None
    materialized: Optional[Path] = None
    failures: Optional[Path] = None


@dataclass
class Run:
    """A loaded run: parsed rollouts plus whatever siblings were present."""

    run_id: str
    rollouts: list[dict[str, Any]]
    aggregate: Optional[Any] = None
    materialized: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TranscriptItem:
    """A single classified entry in a rollout transcript.

    Only the fields relevant to ``kind`` are populated; the rest keep their
    defaults. ``raw`` always holds the original item dict so the UI can offer a
    raw-JSON view and so unknown kinds degrade gracefully.
    """

    source: str  # "input" or "output"
    kind: str
    raw: dict[str, Any]
    role: Optional[str] = None
    text: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None  # parsed arguments object, when parseable
    tool_args_raw: Optional[str] = None  # original argument string
    tool_args_ok: bool = True  # False when arguments could not be parsed as JSON
    call_id: Optional[str] = None
    tool_output: Optional[str] = None
    tool_output_obj: Optional[Any] = None  # parsed output, when JSON-parseable
    tool_output_ok: bool = True
    reasoning_texts: list[str] = field(default_factory=list)
    has_encrypted_reasoning: bool = False


# --------------------------------------------------------------------------- #
# File discovery and loading
# --------------------------------------------------------------------------- #
def derive_run_ref(
    rollouts: Union[str, Path],
    *,
    aggregate: Union[str, Path, None] = None,
    materialized: Union[str, Path, None] = None,
    failures: Union[str, Path, None] = None,
) -> RunRef:
    """Build a :class:`RunRef` from a rollouts file, deriving missing siblings.

    Explicit sibling paths override the derived ones. A derived sibling is only
    recorded when the file actually exists; an explicitly-passed sibling is
    recorded as given (so the caller can surface a "you pointed me at a missing
    file" error).
    """
    rollouts = Path(rollouts)
    stem = _rollouts_stem(rollouts)
    run_id = stem.name

    def resolve(explicit: Union[str, Path, None], suffix: str) -> Optional[Path]:
        if explicit is not None:
            return Path(explicit)
        candidate = stem.with_name(stem.name + suffix)
        return candidate if candidate.exists() else None

    return RunRef(
        run_id=run_id,
        rollouts=rollouts,
        aggregate=resolve(aggregate, AGGREGATE_SUFFIX),
        materialized=resolve(materialized, MATERIALIZED_SUFFIX),
        failures=resolve(failures, FAILURES_SUFFIX),
    )


def _rollouts_stem(rollouts: Path) -> Path:
    """Return the run stem path (the rollouts path minus its ``.jsonl`` suffix)."""
    name = rollouts.name
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return rollouts.with_name(name)


def scan_dir(directory: Union[str, Path]) -> list[RunRef]:
    """Find candidate rollouts files in ``directory`` and build their RunRefs.

    Any ``*.jsonl`` that is not a known sibling (materialized inputs / failures)
    is treated as a rollouts file. Results are sorted by run id for stable UIs.
    """
    directory = Path(directory)
    refs: list[RunRef] = []
    for path in sorted(directory.glob("*.jsonl")):
        if any(path.name.endswith(suffix) for suffix in _NON_ROLLOUT_JSONL_SUFFIXES):
            continue
        refs.append(derive_run_ref(path))
    return refs


def load_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run(ref: RunRef) -> Run:
    """Load all present files for a run. Missing siblings degrade to empty."""
    aggregate: Optional[Any] = None
    if ref.aggregate is not None and Path(ref.aggregate).exists():
        with open(ref.aggregate, encoding="utf-8") as handle:
            aggregate = json.load(handle)

    materialized = (
        load_jsonl(ref.materialized) if ref.materialized is not None and Path(ref.materialized).exists() else []
    )
    failures = load_jsonl(ref.failures) if ref.failures is not None and Path(ref.failures).exists() else []

    return Run(
        run_id=ref.run_id,
        rollouts=load_jsonl(ref.rollouts),
        aggregate=aggregate,
        materialized=materialized,
        failures=failures,
    )


# --------------------------------------------------------------------------- #
# Per-rollout accessors
# --------------------------------------------------------------------------- #
def rollout_id(rollout: dict[str, Any]) -> str:
    """Return the ``"task:rollout"`` identity used across Gym artifacts."""
    return f"{rollout.get('_ng_task_index')}:{rollout.get('_ng_rollout_index')}"


def reward_of(rollout: dict[str, Any]) -> Optional[float]:
    reward = rollout.get("reward")
    return float(reward) if isinstance(reward, (int, float)) else None


def usage_of(rollout: dict[str, Any]) -> dict[str, Optional[int]]:
    """Extract token usage, tolerating a missing ``response``/``usage`` block."""
    usage = ((rollout.get("response") or {}).get("usage")) or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def agent_name_of(rollout: dict[str, Any]) -> Optional[str]:
    return (rollout.get("agent_ref") or {}).get("name")


def model_of(rollout: dict[str, Any]) -> Optional[str]:
    return (rollout.get("response") or {}).get("model")


# --------------------------------------------------------------------------- #
# Transcript classification
# --------------------------------------------------------------------------- #
def _safe_json(text: Optional[str]) -> tuple[Any, bool]:
    """Parse ``text`` as JSON. Returns ``(obj, True)`` or ``(None, False)``."""
    if text is None:
        return None, False
    try:
        return json.loads(text), True
    except (ValueError, TypeError):
        return None, False


def _message_text(item: dict[str, Any]) -> str:
    """Flatten a message item's content into plain text.

    Content is either a string or a list of blocks carrying ``text`` (the
    Responses API uses ``output_text`` / ``input_text`` blocks).
    """
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("text")]
        return "\n".join(parts)
    return ""


def classify_item(item: dict[str, Any], source: str) -> TranscriptItem:
    """Classify one raw trajectory item into a :class:`TranscriptItem`.

    Unknown item types fall back to ``KIND_UNKNOWN`` with only ``raw`` set, so
    the renderer can show them as labeled JSON rather than crashing.
    """
    item_type = item.get("type")

    if item_type == KIND_MESSAGE or (item_type is None and "role" in item and "content" in item):
        return TranscriptItem(
            source=source,
            kind=KIND_MESSAGE,
            raw=item,
            role=item.get("role"),
            text=_message_text(item),
        )

    if item_type == KIND_TOOL_CALL:
        args_raw = item.get("arguments")
        args, ok = _safe_json(args_raw)
        return TranscriptItem(
            source=source,
            kind=KIND_TOOL_CALL,
            raw=item,
            tool_name=item.get("name"),
            tool_args=args,
            tool_args_raw=args_raw,
            tool_args_ok=ok,
            call_id=item.get("call_id"),
        )

    if item_type == KIND_TOOL_OUTPUT:
        output = item.get("output")
        parsed, ok = _safe_json(output) if isinstance(output, str) else (output, False)
        return TranscriptItem(
            source=source,
            kind=KIND_TOOL_OUTPUT,
            raw=item,
            call_id=item.get("call_id"),
            tool_output=output if isinstance(output, str) else json.dumps(output),
            tool_output_obj=parsed,
            tool_output_ok=ok,
        )

    if item_type == KIND_REASONING:
        summaries = item.get("summary") or []
        texts = [s.get("text", "") for s in summaries if isinstance(s, dict)]
        return TranscriptItem(
            source=source,
            kind=KIND_REASONING,
            raw=item,
            reasoning_texts=texts,
            has_encrypted_reasoning=bool(item.get("encrypted_content")),
        )

    return TranscriptItem(source=source, kind=KIND_UNKNOWN, raw=item)


def iter_transcript(rollout: dict[str, Any]) -> list[TranscriptItem]:
    """Return the full ordered transcript: request input then response output."""
    items: list[TranscriptItem] = []
    request_input = (rollout.get("responses_create_params") or {}).get("input") or []
    response_output = (rollout.get("response") or {}).get("output") or []
    for raw in request_input:
        items.append(classify_item(raw, "input"))
    for raw in response_output:
        items.append(classify_item(raw, "output"))
    return items


def tools_of(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the tool schemas the rollout was given."""
    return (rollout.get("responses_create_params") or {}).get("tools") or []


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #
@dataclass
class DiffRow:
    """One aligned position when diffing two transcripts."""

    index: int
    left: Optional[TranscriptItem]
    right: Optional[TranscriptItem]
    differs: bool


def _item_signature(item: Optional[TranscriptItem]) -> Any:
    """A comparable, order-insensitive-to-ids signature of an item.

    Tool call/output ids vary between rollouts even when semantics match, so we
    compare on kind + role + text + tool name + parsed args/output instead.
    """
    if item is None:
        return None
    return (
        item.kind,
        item.role,
        item.text,
        item.tool_name,
        json.dumps(item.tool_args, sort_keys=True) if item.tool_args is not None else item.tool_args_raw,
        json.dumps(item.tool_output_obj, sort_keys=True) if item.tool_output_obj is not None else item.tool_output,
        tuple(item.reasoning_texts),
    )


def diff_rollouts(left: dict[str, Any], right: dict[str, Any]) -> list[DiffRow]:
    """Align two rollouts' transcripts by position and flag divergent rows."""
    left_items = iter_transcript(left)
    right_items = iter_transcript(right)
    rows: list[DiffRow] = []
    for index in range(max(len(left_items), len(right_items))):
        left_item = left_items[index] if index < len(left_items) else None
        right_item = right_items[index] if index < len(right_items) else None
        differs = _item_signature(left_item) != _item_signature(right_item)
        rows.append(DiffRow(index=index, left=left_item, right=right_item, differs=differs))
    return rows


def first_divergence(rows: list[DiffRow]) -> Optional[int]:
    """Return the index of the first differing row, or ``None`` if identical."""
    for row in rows:
        if row.differs:
            return row.index
    return None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rollouts_to_frame(run: Run) -> pd.DataFrame:
    """Flatten rollouts into a per-rollout scalar table for filtering/charts."""
    records: list[dict[str, Any]] = []
    for rollout in run.rollouts:
        usage = usage_of(rollout)
        records.append(
            {
                "rollout_id": rollout_id(rollout),
                "task_index": rollout.get("_ng_task_index"),
                "rollout_index": rollout.get("_ng_rollout_index"),
                "reward": reward_of(rollout),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "agent": agent_name_of(rollout),
                "model": model_of(rollout),
                "num_items": len(iter_transcript(rollout)),
            }
        )
    columns = [
        "rollout_id",
        "task_index",
        "rollout_index",
        "reward",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "agent",
        "model",
        "num_items",
    ]
    return pd.DataFrame(records, columns=columns)


def headline_metrics(run: Run) -> list[dict[str, Any]]:
    """Extract per-agent headline metrics from the aggregate file, if present.

    Returns a list of ``{"agent", "key_metrics", "agent_metrics"}`` entries. An
    absent or unexpectedly-shaped aggregate file yields an empty list, so the UI
    can fall back to self-computed metrics.
    """
    aggregate = run.aggregate
    if not isinstance(aggregate, list):
        return []
    entries: list[dict[str, Any]] = []
    for entry in aggregate:
        if not isinstance(entry, dict):
            continue
        entries.append(
            {
                "agent": (entry.get("agent_ref") or {}).get("name"),
                "key_metrics": entry.get("key_metrics") or {},
                "agent_metrics": entry.get("agent_metrics") or {},
            }
        )
    return entries


def task_indices(run: Run) -> list[Any]:
    """Sorted unique task indices present in the run's rollouts."""
    seen = {rollout.get("_ng_task_index") for rollout in run.rollouts}
    return sorted(value for value in seen if value is not None)


def rollouts_for_task(run: Run, task_index: Any) -> list[dict[str, Any]]:
    """All rollouts belonging to a given task index, in rollout-index order."""
    matches = [r for r in run.rollouts if r.get("_ng_task_index") == task_index]
    return sorted(matches, key=lambda r: (r.get("_ng_rollout_index") is None, r.get("_ng_rollout_index")))
