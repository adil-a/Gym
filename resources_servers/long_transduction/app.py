# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""long_transduction resource server — long-context input-to-output transduction.

Supported sample types (dispatched on the row's `type` field):

  Arithmetic chain summing:
    - "unnumbered_streaming_sum" : plain "<expr>=<answer>" lines
    - "streaming_sum"            : "[N]<expr>=<answer>" lines, in order
    - "shuffled_streaming_sum"   : same numbered format, input lines shuffled

  Per-line UUID sorting:
    - "streaming_uuid_sort"          : "[N](u1),(u2),..." lines, in order;
                                       model outputs each line's UUIDs in
                                       ascending lexicographic order.
    - "shuffled_streaming_uuid_sort" : same as above but input lines are
                                       shuffled; model must still emit in
                                       ascending [N] order.

verify() looks up the scorer in _SCORERS_BY_TYPE (raises on unknown). Rows
without a `type` are treated as the default ("unnumbered_streaming_sum") so
pre-typed rollouts continue to score.

compute_metrics() reports per-difficulty, per-type, and per-(type, difficulty)
accuracy. Difficulty for sum types = max_operands; for uuid_sort types =
uuids_per_line (stored as `max_operands` on the row for now, falling back to
None if absent).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

sys.path.insert(0, str(Path(__file__).parent))
from parse import (  # noqa: E402
    score_response,
    score_response_numbered,
    score_uuid_sort,
)


def _score_sum(generation: str, body) -> list[list[bool]]:
    """Adapter: pick the right arithmetic scorer for body.type and return
    each per-expression score as a 3-bool list [copy, answer, self_consistent].
    """
    if body.type == "unnumbered_streaming_sum" or body.type is None:
        triples = score_response(generation, body.expressions or [])
    else:
        triples = score_response_numbered(generation, body.expressions or [])
    return [list(t) for t in triples]


def _score_uuid(generation: str, body) -> list[list[bool]]:
    """Adapter: per-line (copy_correct, answer_correct, self_consistent) for UUID-sort."""
    return [list(t) for t in score_uuid_sort(generation, body.uuid_lines or [])]


def _generation(body) -> str:
    """Extract the model's generated text from a verify request body."""
    parts = [
        item.text
        for output in body.response.output
        if output.type == "message"
        for item in output.content
        if item.type == "output_text"
    ]
    return "".join(parts)


# Parser dispatch by `type`. Keys are the source of truth for which sample
# types this server understands; verify() raises ValueError on unknowns.
_SCORERS_BY_TYPE = {
    "unnumbered_streaming_sum":     _score_sum,
    "streaming_sum":                _score_sum,
    "shuffled_streaming_sum":       _score_sum,
    "streaming_uuid_sort":          _score_uuid,
    "shuffled_streaming_uuid_sort": _score_uuid,
}
# Rows with no `type` set keep scoring against the legacy unnumbered parser.
_DEFAULT_TYPE = "unnumbered_streaming_sum"


def _strip_reasoning(text: str) -> str:
    # Qwen3 / Nemotron / generic <think>...</think> reasoning models.
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].lstrip("\n")
    if "<think>" in text:
        return ""
    # gpt-oss harmony format: keep only the final channel.
    if "<|channel|>final<|message|>" in text:
        return text.rsplit("<|channel|>final<|message|>", 1)[1].lstrip("\n")
    if "<|channel|>analysis<|message|>" in text:
        return ""
    return text


class LongTransductionConfig(BaseResourcesServerConfig):
    strip_reasoning: bool = True


class LongTransductionRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    # Sample variant. None falls back to _DEFAULT_TYPE; recognized values are
    # the keys of _SCORERS_BY_TYPE.
    type: Optional[str] = None
    max_operands: Optional[int] = None
    n_expressions: Optional[int] = None
    # Arithmetic-chain payload (sum types).
    expressions: Optional[List[Dict[str, Any]]] = None
    # UUID-sort payload (uuid_sort types). Each inner list is one line's UUIDs
    # in their canonical (input-presentation) order; expected output for that
    # line is the same UUIDs sorted lexicographically.
    uuid_lines: Optional[List[List[str]]] = None
    uuids_per_line: Optional[int] = None
    n_lines: Optional[int] = None


class LongTransductionVerifyRequest(LongTransductionRunRequest, BaseVerifyRequest):
    pass


class LongTransductionVerifyResponse(LongTransductionVerifyRequest, BaseVerifyResponse):
    # Mean per-item correctness (used as the reward). For sum types this is
    # mean(answer_correct); for uuid_sort it's mean(position_correct) flattened
    # across all lines.
    answer_correct: float
    n_items_scored: int
    # Per-item [copy_correct, answer_correct, self_consistent]. The three
    # signals are defined slightly differently per task type — see the scorer
    # docstrings — but the shape and reward semantics (index 1) are uniform.
    item_scores: List[List[bool]]


class LongTransductionServer(SimpleResourcesServer):
    config: LongTransductionConfig

    async def verify(
        self, body: LongTransductionVerifyRequest
    ) -> LongTransductionVerifyResponse:
        sample_type = body.type or _DEFAULT_TYPE
        try:
            scorer = _SCORERS_BY_TYPE[sample_type]
        except KeyError as e:
            raise ValueError(
                f"Unsupported long_transduction sample type: {sample_type!r}. "
                f"Supported types: {sorted(_SCORERS_BY_TYPE)}."
            ) from e

        # Extract the model's generated text locally and (if configured) strip
        # any reasoning preamble for scoring. The saved response body is left
        # untouched so debugging tools can see the raw model output.
        generation = _generation(body)
        if self.config.strip_reasoning:
            generation = _strip_reasoning(generation)

        item_scores = scorer(generation, body)

        # Reward is the mean "answer_correct" signal (index 1) across items.
        # Both sum and uuid_sort scorers emit per-item 3-tuples of
        # (copy_correct, answer_correct, self_consistent), so a single rule
        # extracts the reward signal cleanly.
        flat = [row[1] for row in item_scores] if item_scores else []
        n_items = len(flat)
        reward = (sum(flat) / n_items) if n_items else 0.0

        return LongTransductionVerifyResponse(
            **body.model_dump(),
            reward=reward,
            answer_correct=reward,
            n_items_scored=n_items,
            item_scores=item_scores,
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        by_difficulty: Dict[Any, List[float]] = defaultdict(list)
        by_type: Dict[str, List[float]] = defaultdict(list)
        by_type_difficulty: Dict[tuple, List[float]] = defaultdict(list)
        for task_rollouts in tasks:
            for rollout in task_rollouts:
                if rollout.get("answer_correct") is None:
                    continue
                ttype = rollout.get("type") or _DEFAULT_TYPE
                # Pick the difficulty knob present on this row.
                diff = rollout.get("max_operands")
                if diff is None:
                    diff = rollout.get("uuids_per_line")
                if diff is None:
                    diff = "n/a"
                acc = rollout["answer_correct"]
                by_difficulty[diff].append(acc)
                by_type[ttype].append(acc)
                by_type_difficulty[(ttype, diff)].append(acc)

        metrics: Dict[str, Any] = {}
        all_vals: List[float] = []
        for diff in sorted(by_difficulty.keys(), key=lambda d: (d == "n/a", d)):
            vals = by_difficulty[diff]
            metrics[f"difficulty_{diff}"] = {
                "accuracy": sum(vals) / len(vals) if vals else None,
                "n": len(vals),
            }
            if vals:
                all_vals.extend(vals)

        for ttype in sorted(by_type.keys()):
            vals = by_type[ttype]
            metrics[f"type_{ttype}"] = {
                "accuracy": sum(vals) / len(vals) if vals else None,
                "n": len(vals),
            }

        for (ttype, diff) in sorted(
            by_type_difficulty.keys(), key=lambda k: (k[0], k[1] == "n/a", k[1])
        ):
            vals = by_type_difficulty[(ttype, diff)]
            metrics[f"type_{ttype}_difficulty_{diff}"] = {
                "accuracy": sum(vals) / len(vals) if vals else None,
                "n": len(vals),
            }

        if all_vals:
            metrics["overall_accuracy"] = sum(all_vals) / len(all_vals)
        return metrics

    def get_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v["accuracy"]
            for k, v in metrics.items()
            if isinstance(v, dict) and v.get("accuracy") is not None
        }


if __name__ == "__main__":
    LongTransductionServer.run_webserver()
