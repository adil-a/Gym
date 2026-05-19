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
"""GDPVal resources server.

Scores Stirrup agent deliverables for the GDPVal benchmark. Two modes,
selected via ``reward_mode`` config:

- ``rubric``: score deliverables against a per-task rubric using an LLM
  judge. Reward in [0.0, 1.0].
- ``comparison``: pairwise-judge the eval deliverable against a reference
  rollout's deliverable for the same ``task_id``. Reward in {0.0, 0.5, 1.0}.
  ``aggregate_metrics`` then reduces win/loss/tie counts into an ELO rating.

Scoring internals live in ``scoring.py`` (rubric) and ``comparison.py``
(pairwise judge + ELO math).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.server_utils import get_server_url


LOGGER = logging.getLogger(__name__)

_DEFAULT_JUDGE_PROMPT_FPATH = str(Path(__file__).parent / "prompts" / "judge_prompt.j2")
_DEFAULT_REFERENCE_ELO = 1000.0


def _iter_ref_repeat_dirs(task_dir: Path) -> List[Path]:
    """All reference deliverable dirs for a task, supporting both layouts.

    New: ``task_<id>/repeat_<n>/`` — return every repeat dir, sorted. Old:
    flat ``task_<id>/`` — return ``[task_dir]``. Missing → ``[]``.

    Returning every repeat lets the comparison verifier judge each eval
    rollout against *all* reference rollouts so the win rate (and ELO)
    averages over reference variance instead of being anchored to a single
    sample.
    """
    if not task_dir.is_dir():
        return []
    repeats = sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_"))
    return repeats or [task_dir]


def _safe_output_text(response: Any) -> str:
    """Extract concatenated assistant text from a response without relying on
    ``response.output_text`` — that property raises ``AttributeError`` when
    ``output[*].content`` contains raw strings (e.g. input messages carried
    through by the Stirrup agent)."""
    parts: List[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        if d.get("type") != "message":
            continue
        if d.get("role") and d.get("role") != "assistant":
            continue
        content = d.get("content") or []
        if isinstance(content, str):
            parts.append(content)
            continue
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(c.get("text") or "")
    return "\n".join(p for p in parts if p)


class ReferenceConfig(BaseModel):
    """One member of a committee of references for pairwise judging.

    Layout under ``deliverables_dir`` is the same as the legacy single-reference
    tree: ``<deliverables_dir>/task_<task_id>/[repeat_<n>/]``.
    """

    name: str
    deliverables_dir: str
    elo: float


class GDPValResourcesServerConfig(BaseResourcesServerConfig):
    reward_mode: Literal["rubric", "comparison"] = "rubric"

    # Committee of references for pairwise scoring. When non-empty, eval
    # deliverables are judged against every reference and ``aggregate_metrics``
    # fits one eval ELO jointly via Bradley-Terry MLE. Empty (default) means
    # rubric mode unless the legacy ``reference_deliverables_dir`` field is set
    # — in which case ``_promote_legacy_to_committee`` synthesizes a single-
    # element committee with ``name="default"``, so the rest of the server only
    # ever reads ``committee_references``.
    committee_references: List[ReferenceConfig] = Field(default_factory=list)

    # Legacy single-reference fields (deprecated, retained for back-compat).
    # When set without ``committee_references``, the model validator promotes
    # them to a one-element committee.
    reference_deliverables_dir: Optional[str] = None
    reference_elo: float = _DEFAULT_REFERENCE_ELO

    # Pairwise judge trials per (ref × task) pair. 4 is the historical default;
    # alternates swap/no-swap to debias position effects.
    num_comparison_trials: int = 4

    # Office→PDF preconversion for deliverables before pairwise judging.
    # Most office docs render poorly as raw text; PDFs let multimodal judges
    # read tables/charts. Costs ~5-30s per Office file.
    preconvert_office_to_pdf: bool = True
    preconvert_max_concurrent: int = 4

    judge_model_server: ModelServerRef
    judge_responses_create_params_overrides: Dict[str, Any] = {}
    judge_prompt_template_fpath: Optional[str] = None

    # Rubric-mode scoring backend:
    # - ``"binary"`` (default, legacy): judge emits a JSON ``{criteria_scores:
    #   [{score: 0|1, ...}], overall_score: float}``; reward is the overall
    #   score (0-1). Treats every criterion as equal weight.
    # - ``"structured"``: judge emits ``CRITERION_NUMBER[N]: GRADE[X] out of
    #   MAX_POSSIBLE_POINTS[Y]`` tagged output and ``FINAL_SCORE[…] / MAX_POSSIBLE_SCORE[…]``.
    #   Honors per-criterion point weights when the rubric carries them in
    #   ``rubric_json[i].score`` or ``rubric_json[i].weight``. For datasets
    #   without weights, every criterion contributes max-points 1, giving a
    #   signal equivalent to binary mode. Multi-trial averaged for stability.
    #   The tagged output is also more compact than the JSON-with-rationale
    #   format used by binary mode, so it rarely runs into the judge's
    #   ``finish_reason: length`` truncation on rubrics with many criteria.
    rubric_scoring_mode: Literal["binary", "structured"] = "binary"
    rubric_structured_num_trials: int = 2
    rubric_structured_formatting_retries: int = 3

    # When True, every judge call's raw response text is preserved on
    # ``verify_response.judge_response`` (per-trial in comparison mode under
    # ``per_ref_repeat[i].raw_responses``; under top-level ``raw_responses``
    # in rubric modes). Off by default — raw responses are 10-50 KB each and
    # multiply by num_trials × num_ref_repeats × num_tasks. Turn on for debug
    # runs to post-mortem judge verdicts.
    persist_raw_judge_responses: bool = False

    @model_validator(mode="after")
    def _promote_legacy_to_committee(self) -> "GDPValResourcesServerConfig":
        # If a committee is configured explicitly, that takes precedence.
        # Otherwise, when the deprecated ``reference_deliverables_dir`` is set,
        # synthesize a one-element committee so downstream code only ever has
        # to read ``committee_references``. The k=1 MLE coincides with the
        # legacy closed-form ELO, so this is behaviorally identical.
        if not self.committee_references and self.reference_deliverables_dir is not None:
            self.committee_references = [
                ReferenceConfig(
                    name="default",
                    deliverables_dir=self.reference_deliverables_dir,
                    elo=self.reference_elo,
                )
            ]
        return self


class GDPValVerifyRequest(BaseVerifyRequest):
    task_id: str
    sector: Optional[str] = None
    occupation: Optional[str] = None
    prompt: Optional[str] = None
    rubric_json: Optional[Any] = None
    rubric_pretty: Optional[str] = None
    reference_file_urls: Optional[List[str]] = None
    deliverables_dir: Optional[str] = None


class ReferenceRepeatResult(BaseModel):
    """One reference-repeat's trial outcome inside a ``ReferenceVerdict``."""

    repeat_dir: str
    win_count_a: int  # ref wins
    win_count_b: int  # eval wins
    tie_count: int
    task_count: int  # = num_comparison_trials
    raw_responses: Optional[List[str]] = None


class ReferenceVerdict(BaseModel):
    """Aggregated outcome for one committee reference on one task.

    ``wins`` / ``losses`` / ``ties`` sum across the reference's repeats (``eval``
    is submission B in ``run_trials``, so ``wins`` corresponds to
    ``win_count_b``). ``judged = wins + losses + ties = num_repeats × trials``.
    """

    name: str
    reference_elo: float
    wins: int
    losses: int
    ties: int
    judged: int
    win_rate: float  # (wins + 0.5*ties) / judged; 0.0 when judged == 0
    per_repeat: List[ReferenceRepeatResult] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


class GDPValVerifyResponse(GDPValVerifyRequest, BaseVerifyResponse):
    verify_mode: Literal["rubric", "comparison"] = "rubric"
    judge_response: Optional[Dict[str, Any]] = None
    invalid_judge_response: Optional[bool] = None
    # Per-reference verdicts populated in comparison mode. One entry per
    # committee member; failed refs (missing task dir, judge error) carry
    # ``success=False`` and zero counts. ``aggregate_metrics`` rolls these up
    # via Bradley-Terry MLE.
    per_reference_results: List[ReferenceVerdict] = Field(default_factory=list)
    # True when every committee reference failed for this task — the verify
    # caller should treat this row as a no-op for ELO aggregation.
    fully_unjudged: bool = False
    # Majority-decision flags across all (ref × repeat × trial) judge votes —
    # kept for back-compat with older verify responses (still bool-valued).
    win: Optional[bool] = None
    loss: Optional[bool] = None
    tie: Optional[bool] = None
    # Raw judge vote counts aggregated over every reference × repeat × trial.
    # Summed from ``per_reference_results`` in comparison mode. Retained so
    # ``aggregate_metrics`` can still consume rollouts produced before the
    # committee path landed.
    total_wins: Optional[int] = None
    total_losses: Optional[int] = None
    total_ties: Optional[int] = None


class GDPValResourcesServer(SimpleResourcesServer):
    config: GDPValResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._judge_prompt_fpath: str = self.config.judge_prompt_template_fpath or _DEFAULT_JUDGE_PROMPT_FPATH
        if self.config.reward_mode == "comparison" and not self.config.committee_references:
            # Validator already promoted ``reference_deliverables_dir`` if set,
            # so an empty committee_references here means neither field was
            # provided. Keep the "reference_deliverables_dir" mention so legacy
            # callers reading the error message still see a familiar key.
            raise ValueError(
                "reward_mode=comparison requires reference_deliverables_dir or committee_references to be set"
            )
        if self.config.preconvert_office_to_pdf:
            from resources_servers.gdpval.setup_libreoffice import ensure_libreoffice

            if not ensure_libreoffice() and self.config.reward_mode == "comparison":
                raise RuntimeError(
                    "preconvert_office_to_pdf=True and reward_mode='comparison' but libreoffice "
                    "could not be ensured on the host. Office deliverables would reach the multimodal "
                    "judge as filename-only stubs, biasing the win rate. Install libreoffice in the "
                    "deployment container, or set preconvert_office_to_pdf=false to opt out."
                )
        super().model_post_init(context)

    async def verify(self, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        if self.config.reward_mode == "comparison":
            return await self._verify_comparison(body)

        return await self._verify_rubric(body)

    async def _verify_rubric(self, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        if not (body.rubric_json or body.rubric_pretty):
            return GDPValVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verify_mode="rubric",
                invalid_judge_response=True,
            )

        overrides = dict(self.config.judge_responses_create_params_overrides or {})
        judge_base_url = get_server_url(self.config.judge_model_server.name) + "/v1"
        judge_model_name = overrides.pop("model", "judge")
        judge_api_key = overrides.pop("api_key", "dummy")
        # Anything left in `overrides` (max_tokens, temperature, top_p, …) is
        # merged into the judge's chat.completions.create kwargs.
        judge_create_overrides = overrides or None

        deliverable_text = _safe_output_text(body.response)
        deliverable_content_blocks: Optional[List[Dict[str, Any]]] = None

        if body.deliverables_dir and Path(body.deliverables_dir).is_dir():
            from responses_api_agents.stirrup_agent.file_reader import (
                convert_deliverables_to_content_blocks,
                read_deliverable_files,
            )

            read = read_deliverable_files(body.deliverables_dir)
            if read:
                deliverable_text = read
            blocks = convert_deliverables_to_content_blocks(body.deliverables_dir)
            if blocks:
                deliverable_content_blocks = blocks

        task_prompt = body.prompt or ""
        rubric_pretty = body.rubric_pretty or ""

        # Visual scoring when deliverable renders (PDFs/images) are available —
        # the judge model is expected to be multimodal (configured via
        # ``judge_model_server`` in the benchmark YAML). Falls back to text
        # scoring only when no content blocks could be built.
        if self.config.rubric_scoring_mode == "structured":
            from resources_servers.gdpval.scoring import score_with_rubric_structured

            reward, judge_result = await score_with_rubric_structured(
                deliverable_text=deliverable_text,
                rubric_json=body.rubric_json,
                rubric_pretty=rubric_pretty,
                task_prompt=task_prompt,
                model_base_url=judge_base_url,
                model_name=judge_model_name,
                api_key=judge_api_key,
                num_trials=self.config.rubric_structured_num_trials,
                formatting_retries=self.config.rubric_structured_formatting_retries,
                deliverable_content_blocks=deliverable_content_blocks,
                include_raw_responses=self.config.persist_raw_judge_responses,
            )
        elif deliverable_content_blocks:
            from resources_servers.gdpval.scoring import score_with_rubric_visual

            reward, judge_result = await score_with_rubric_visual(
                deliverable_content_blocks=deliverable_content_blocks,
                rubric_json=body.rubric_json,
                rubric_pretty=rubric_pretty,
                task_prompt=task_prompt,
                judge_prompt_template=self._judge_prompt_fpath,
                model_base_url=judge_base_url,
                model_name=judge_model_name,
                api_key=judge_api_key,
                create_overrides=judge_create_overrides,
                include_raw_responses=self.config.persist_raw_judge_responses,
            )
        else:
            from resources_servers.gdpval.scoring import score_with_rubric

            reward, judge_result = await score_with_rubric(
                deliverable_text=deliverable_text,
                rubric_json=body.rubric_json,
                rubric_pretty=rubric_pretty,
                task_prompt=task_prompt,
                judge_prompt_template=self._judge_prompt_fpath,
                model_base_url=judge_base_url,
                model_name=judge_model_name,
                api_key=judge_api_key,
                create_overrides=judge_create_overrides,
                include_raw_responses=self.config.persist_raw_judge_responses,
            )

        return GDPValVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            verify_mode="rubric",
            judge_response=judge_result,
            invalid_judge_response=(judge_result is None),
        )

    async def _preconvert_and_log(self, target_dir: Path, *, label: str) -> None:
        from resources_servers.gdpval.preconvert import preconvert_dir_async

        n_ok, n_fail, errors = await preconvert_dir_async(
            target_dir, max_concurrent=self.config.preconvert_max_concurrent
        )
        if n_ok or n_fail:
            LOGGER.info("preconvert %s: ok=%d fail=%d", label, n_ok, n_fail)
        if n_fail:
            for msg in errors[:5]:
                LOGGER.warning("preconvert %s: %s", label, msg)

    async def _verify_comparison(self, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        from openai import OpenAI

        from resources_servers.gdpval.comparison import (
            A_WIN_RESPONSE,
            B_WIN_RESPONSE,
            JUDGE_REQUEST_TIMEOUT_SECONDS,
            TIE_RESPONSE,
            build_file_section,
            clean_up_paths,
            run_trials,
            task_attempted,
        )

        # For each committee member, locate the per-repeat dirs that actually
        # have a completed ``finish_params.json`` (``task_attempted``). Refs
        # with no repeats for this task carry through to ``per_reference_results``
        # as ``success=False`` so aggregate metrics can still see the miss.
        ref_to_dirs: Dict[str, List[Path]] = {}
        for ref in self.config.committee_references:
            ref_root = Path(ref.deliverables_dir) / f"task_{body.task_id}"
            ref_to_dirs[ref.name] = [d for d in _iter_ref_repeat_dirs(ref_root) if task_attempted(str(d))]

        eval_task_dir = Path(body.deliverables_dir) if body.deliverables_dir else None

        if not any(ref_to_dirs.values()):
            print(f"[gdpval] no reference deliverable for task {body.task_id}", flush=True)
            return GDPValVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verify_mode="comparison",
                judge_response={"error": "reference_missing"},
                per_reference_results=[
                    ReferenceVerdict(
                        name=ref.name,
                        reference_elo=ref.elo,
                        wins=0,
                        losses=0,
                        ties=0,
                        judged=0,
                        win_rate=0.0,
                        success=False,
                        error_message="ref_missing",
                    )
                    for ref in self.config.committee_references
                ],
                fully_unjudged=True,
            )

        if eval_task_dir is None or not task_attempted(str(eval_task_dir)):
            print(f"[gdpval] eval deliverable missing for task {body.task_id}", flush=True)
            return GDPValVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verify_mode="comparison",
                judge_response={"error": "eval_missing"},
                loss=True,
            )

        if self.config.preconvert_office_to_pdf:
            await self._preconvert_and_log(eval_task_dir, label=f"eval/{body.task_id}")
            for ref in self.config.committee_references:
                for ref_dir in ref_to_dirs.get(ref.name, []):
                    await self._preconvert_and_log(ref_dir, label=f"ref/{ref.name}/{body.task_id}/{ref_dir.name}")

        overrides = dict(self.config.judge_responses_create_params_overrides or {})
        judge_base_url = get_server_url(self.config.judge_model_server.name) + "/v1"
        judge_model_name = overrides.get("model", "judge")
        judge_api_key = overrides.get("api_key", "dummy")
        client = OpenAI(
            base_url=judge_base_url,
            api_key=judge_api_key,
            timeout=JUDGE_REQUEST_TIMEOUT_SECONDS,
        )

        clean_up_list: List[Path] = []
        per_reference_results: List[ReferenceVerdict] = []
        try:
            eval_submission = build_file_section(str(eval_task_dir), clean_up_list)

            # Flat list of (ref, ref_dir, refs, ref_submission) so we can fan
            # out concurrently and reassemble per-ref afterwards. ``run_trials``
            # is blocking (it spins judge HTTP calls with retries), so each
            # entry is wrapped in ``asyncio.to_thread``; ``asyncio.gather``
            # gets us ref × repeat parallelism bounded only by the judge
            # server's ``max_concurrent_requests``.
            flat: List[tuple] = []  # (ref, ref_dir, refs_blocks, ref_submission_blocks)
            for ref in self.config.committee_references:
                for ref_dir in ref_to_dirs.get(ref.name, []):
                    refs_subdir = ref_dir / "reference_files"
                    refs_blocks = build_file_section(
                        str(refs_subdir) if refs_subdir.is_dir() else None,
                        clean_up_list,
                    )
                    ref_submission_blocks = build_file_section(str(ref_dir), clean_up_list)
                    flat.append((ref, ref_dir, refs_blocks, ref_submission_blocks))

            async def _one_trial(refs_blocks, ref_submission_blocks):
                return await asyncio.to_thread(
                    run_trials,
                    client=client,
                    model=judge_model_name,
                    task_prompt=body.prompt or "",
                    refs=refs_blocks,
                    submission_a=ref_submission_blocks,
                    submission_b=eval_submission,
                    num_trials=self.config.num_comparison_trials,
                    return_raw_responses=self.config.persist_raw_judge_responses,
                )

            results = await asyncio.gather(
                *[_one_trial(rfb, rsb) for (_r, _d, rfb, rsb) in flat],
                return_exceptions=True,
            )

            # Reassemble per ref. ``run_trials`` casts submission_a=ref,
            # submission_b=eval, so ``win_count_b`` is the eval's wins.
            ref_to_repeats: Dict[str, List[tuple]] = {}
            for (ref, ref_dir, _rfb, _rsb), result in zip(flat, results):
                ref_to_repeats.setdefault(ref.name, []).append((ref_dir, result))

            for ref in self.config.committee_references:
                entries = ref_to_repeats.get(ref.name, [])
                if not entries:
                    per_reference_results.append(
                        ReferenceVerdict(
                            name=ref.name,
                            reference_elo=ref.elo,
                            wins=0,
                            losses=0,
                            ties=0,
                            judged=0,
                            win_rate=0.0,
                            success=False,
                            error_message="ref_missing",
                        )
                    )
                    continue

                wins = losses = ties = 0
                per_repeat: List[ReferenceRepeatResult] = []
                first_error: Optional[str] = None
                for ref_dir, result in entries:
                    if isinstance(result, Exception):
                        if first_error is None:
                            first_error = f"{type(result).__name__}: {result}"
                        continue
                    wins += result["win_count_b"]
                    losses += result["win_count_a"]
                    ties += result["tie_count"]
                    per_repeat.append(
                        ReferenceRepeatResult(
                            repeat_dir=ref_dir.name,
                            win_count_a=result["win_count_a"],
                            win_count_b=result["win_count_b"],
                            tie_count=result["tie_count"],
                            task_count=result["task_count"],
                            raw_responses=result.get("raw_responses"),
                        )
                    )

                judged = wins + losses + ties
                success = judged > 0
                per_reference_results.append(
                    ReferenceVerdict(
                        name=ref.name,
                        reference_elo=ref.elo,
                        wins=wins,
                        losses=losses,
                        ties=ties,
                        judged=judged,
                        win_rate=((wins + 0.5 * ties) / judged) if judged > 0 else 0.0,
                        per_repeat=per_repeat,
                        success=success,
                        error_message=None if success else (first_error or "all_trials_failed"),
                    )
                )
        finally:
            clean_up_paths(clean_up_list)

        # Per-rollout reward: mean of per-ref majority-vote rewards across
        # refs that actually produced a verdict. Matches the bash_sandbox
        # committee pattern; for k=1 this is bit-for-bit the legacy reward.
        successful = [v for v in per_reference_results if v.success and v.judged > 0]
        if not successful:
            reward = 0.0
            fully_unjudged = True
        else:
            per_ref_rewards: List[float] = []
            for v in successful:
                if v.wins > v.losses:
                    per_ref_rewards.append(1.0)
                elif v.losses > v.wins:
                    per_ref_rewards.append(0.0)
                else:
                    per_ref_rewards.append(0.5)
            reward = sum(per_ref_rewards) / len(per_ref_rewards)
            fully_unjudged = False

        # Back-compat totals + flattened per-ref-repeat list so existing
        # consumers / saved evaluator_rollouts.jsonl rows keep parsing.
        total_wins = sum(v.wins for v in per_reference_results)
        total_losses = sum(v.losses for v in per_reference_results)
        total_ties = sum(v.ties for v in per_reference_results)
        total_judged = total_wins + total_losses + total_ties

        flat_per_ref_repeat: List[Dict[str, Any]] = []
        for v in per_reference_results:
            for rr in v.per_repeat:
                if rr.win_count_a > rr.win_count_b:
                    winner = A_WIN_RESPONSE
                elif rr.win_count_b > rr.win_count_a:
                    winner = B_WIN_RESPONSE
                else:
                    winner = TIE_RESPONSE
                entry: Dict[str, Any] = {
                    "ref_name": v.name,
                    "ref_repeat": rr.repeat_dir,
                    "winner": winner,
                    "win_count_a": rr.win_count_a,
                    "win_count_b": rr.win_count_b,
                    "tie_count": rr.tie_count,
                    "task_count": rr.task_count,
                }
                if rr.raw_responses is not None:
                    entry["raw_responses"] = rr.raw_responses
                flat_per_ref_repeat.append(entry)

        return GDPValVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verify_mode="comparison",
            per_reference_results=per_reference_results,
            fully_unjudged=fully_unjudged,
            judge_response={
                "per_ref_repeat": flat_per_ref_repeat,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "total_ties": total_ties,
                "total_judged": total_judged,
                "ref_repeat_count": len(flat_per_ref_repeat),
                "committee_size": len(self.config.committee_references),
            },
            win=reward == 1.0,
            loss=reward == 0.0,
            tie=reward == 0.5,
            total_wins=total_wins,
            total_losses=total_losses,
            total_ties=total_ties,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        if self.config.reward_mode != "comparison":
            return await super().aggregate_metrics(body)

        from resources_servers.gdpval.comparison import calculate_elo, calculate_elo_mle

        ref_elo_by_name: Dict[str, float] = {ref.name: ref.elo for ref in self.config.committee_references}
        # Per-ref vote totals across every verify response. Initialized to
        # zero for every configured committee member so refs that produced
        # no battles still show up in the diagnostic block.
        per_ref: Dict[str, Dict[str, int]] = {name: {"wins": 0, "losses": 0, "ties": 0} for name in ref_elo_by_name}

        # Fallback bucket for verify responses that lack ``per_reference_results``
        # but carry the legacy ``total_wins``/``total_losses``/``total_ties`` or
        # ``win``/``loss``/``tie`` fields. Attribute those votes to the first
        # committee member — the legacy single-ref path is the only producer
        # of that shape, and the validator already promoted it to a one-element
        # committee, so this is unambiguous.
        legacy_bucket = next(iter(ref_elo_by_name), None)

        for vr in body.verify_responses:
            if vr.get("fully_unjudged"):
                continue

            prr = vr.get("per_reference_results")
            if prr:
                for entry in prr:
                    if not entry.get("success"):
                        continue
                    name = entry.get("name") or legacy_bucket or "default"
                    stats = per_ref.setdefault(name, {"wins": 0, "losses": 0, "ties": 0})
                    stats["wins"] += int(entry.get("wins") or 0)
                    stats["losses"] += int(entry.get("losses") or 0)
                    stats["ties"] += int(entry.get("ties") or 0)
                continue

            if legacy_bucket is None:
                continue
            stats = per_ref.setdefault(legacy_bucket, {"wins": 0, "losses": 0, "ties": 0})
            tw, tl, tt = vr.get("total_wins"), vr.get("total_losses"), vr.get("total_ties")
            if tw is not None or tl is not None or tt is not None:
                stats["wins"] += int(tw or 0)
                stats["losses"] += int(tl or 0)
                stats["ties"] += int(tt or 0)
            else:
                stats["wins"] += int(bool(vr.get("win")))
                stats["losses"] += int(bool(vr.get("loss")))
                stats["ties"] += int(bool(vr.get("tie")))

        wins = sum(s["wins"] for s in per_ref.values())
        losses = sum(s["losses"] for s in per_ref.values())
        ties = sum(s["ties"] for s in per_ref.values())
        judged = wins + losses + ties

        if judged == 0:
            return await super().aggregate_metrics(body)

        overall_win_rate = (wins + 0.5 * ties) / judged

        # Joint Bradley-Terry MLE over refs with at least one battle. For k=1
        # this matches ``calculate_elo`` to within the bisection tolerance, so
        # legacy single-ref runs produce identical ``eval_elo`` to before.
        battles: List[tuple] = []  # (ref_elo, wins_plus_half_ties, n)
        for name, stats in per_ref.items():
            n = stats["wins"] + stats["losses"] + stats["ties"]
            if n == 0:
                continue
            anchor = ref_elo_by_name.get(name, _DEFAULT_REFERENCE_ELO)
            battles.append((anchor, stats["wins"] + 0.5 * stats["ties"], n))
        eval_elo, eval_elo_se, degenerate = calculate_elo_mle(battles)

        base = await super().aggregate_metrics(body)
        extra: Dict[str, Any] = {
            "comparison/wins": wins,
            "comparison/losses": losses,
            "comparison/ties": ties,
            "comparison/judged": judged,
            "comparison/win_rate": overall_win_rate,
            "comparison/eval_elo_degenerate": degenerate,
            "comparison/committee_size": len(ref_elo_by_name),
        }
        if eval_elo is not None:
            extra["comparison/eval_elo"] = eval_elo
            extra["comparison/normalized_elo"] = (eval_elo - 500.0) / 2000.0
        if eval_elo_se is not None:
            extra["comparison/eval_elo_se"] = eval_elo_se

        # ``comparison/reference_elo`` is now ambiguous with k>1; emit the mean
        # of committee anchors so the key keeps roughly its old meaning and
        # downstream consumers (W&B / mlflow) don't break.
        if ref_elo_by_name:
            extra["comparison/reference_elo"] = sum(ref_elo_by_name.values()) / len(ref_elo_by_name)

        # Per-ref diagnostics — each ref's standalone win rate + closed-form
        # ELO (computed against this ref alone). Useful for debugging which
        # committee member the eval over/under-performs against.
        for name, stats in per_ref.items():
            n = stats["wins"] + stats["losses"] + stats["ties"]
            if n == 0:
                continue
            wr = (stats["wins"] + 0.5 * stats["ties"]) / n
            anchor = ref_elo_by_name.get(name, _DEFAULT_REFERENCE_ELO)
            single_elo, _ = calculate_elo(wr, anchor)
            extra[f"comparison/{name}/wins"] = stats["wins"]
            extra[f"comparison/{name}/losses"] = stats["losses"]
            extra[f"comparison/{name}/ties"] = stats["ties"]
            extra[f"comparison/{name}/n"] = n
            extra[f"comparison/{name}/win_rate"] = wr
            extra[f"comparison/{name}/eval_elo_single"] = single_elo

        merged_agent = {**base.agent_metrics, **extra}
        merged_key = {**base.key_metrics, **extra}
        return AggregateMetrics(
            group_level_metrics=base.group_level_metrics,
            agent_metrics=merged_agent,
            key_metrics=merged_key,
        )


if __name__ == "__main__":
    GDPValResourcesServer.run_webserver()
