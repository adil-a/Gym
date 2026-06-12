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

"""Idempotent post-install patcher for the vendored OpenClaw JS bundle.

OpenClaw's auto-compaction ("context summarization") REPLACES the conversation history
mid-episode with a model-generated summary. That rewrite breaks the prompt_token_ids
prefix-contiguity chain that on-policy GRPO RL training relies on, so it must never fire
during a training rollout.

The compaction triggers are baked into the minified bundle (node_modules/openclaw/dist)
and -- in the pinned version -- are NOT reachable by any openclaw.json / pi settings.json
lever (the documented `compaction.enabled` key is rejected as unknown, and the model
context-window fields never reach the guard's budget). We therefore neutralize the
triggers with surgical source patches applied right after `npm install` and re-applied on
every reinstall (npm regenerates dist/).

Design constraints:
- Each patch is anchored on a UNIQUE marker substring and rewrites only that site.
- Idempotent: an already-patched site contains SENTINEL, so re-running is a no-op.
- Fails loudly if a marker is missing (e.g. OpenClaw was upgraded and the code shape
  changed) so the neutralization is never silently dropped.
- Touches nothing but the compaction-trigger sites: benign per-result tool-output
  truncation and all other behaviour are left exactly as shipped.
"""

import sys
from pathlib import Path


SENTINEL = "/*nemo-gym-no-compaction*/"

# Separate sentinel for the unknown-tool-surfacing patches (below). Kept distinct from the
# compaction SENTINEL so each family is independently greppable / countable.
SENTINEL_SURFACE_TOOL = "/*nemo-gym-surface-unknown-tool*/"

# A "nothing to compact" result. This is the exact shape OpenClaw's compaction code already
# returns when there is nothing to compact (see compact-Be1VaHAE.js: `{ ok: true, compacted: false }`),
# so every caller already handles it gracefully (proceed; the bounded overflow/timeout recovery
# loops treat compacted:false as "did not reduce" and fall through -- no infinite loop).
_NOOP_RESULT = '{ ok: true, compacted: false, reason: "compaction disabled for RL token-contiguity" }'

# Explicit, greppable training-health signal emitted (to stderr -> per-rollout openclaw_stderr.log)
# whenever an automatic compaction is requested and suppressed. The guard path is silenced upstream
# (it never reaches an executor), so this fires for the model-overflow / precheck / timeout /
# post-turn paths -- i.e. it marks genuine context-pressure events without the harmful history
# rewrite. Grep `[nemo-gym] auto-compaction suppressed` across rollouts to count them.
_SUPPRESS_LOG = 'console.error("[nemo-gym] auto-compaction suppressed (RL token-contiguity)");'
_NOOP_BODY = _SUPPRESS_LOG + " return " + _NOOP_RESULT + ";"

# OpenClaw auto-compaction summarizes and REPLACES conversation history mid-episode, which breaks
# the prompt_token_ids prefix-contiguity that on-policy RL training requires. There is no config
# lever to disable it in the pinned version, so we neutralize it at three code chokepoints that
# together cover EVERY automatic-compaction path (verified by source-tracing the v2026.5.6 bundle):
#
#   1. preemptive-overflow-guard  -- the PREEMPTIVE TOOL-LOOP guard. `maxContextChars` is used ONLY
#      to decide whether to throw PREEMPTIVE_CONTEXT_OVERFLOW_MESSAGE (the embedded runner catches
#      it and recovers via branch=compact). Forcing the budget to +Infinity means the throw never
#      fires, so episodes grow without this trigger. The separate per-result truncation budget
#      (`maxSingleToolResultChars`) is untouched -- benign tool-output trimming still works.
#
#   2. compaction-executor  -- `compactEmbeddedPiSessionDirect` is the embedded-pi compaction
#      EXECUTOR that `contextEngine.compact(...)` routes through (overflow recovery, timeout
#      recovery, and the post-turn entry fallback all call it). Early-returning the benign
#      "nothing to compact" result here means NO automatic path ever produces/installs a summary.
#      This is OpenClaw's OWN compaction (distinct from pi's internal compaction, which is not on
#      the embedded execution path). Pruning (pruneHistoryForContextShare) is only reachable from
#      inside this executor, so it is covered too.
#
#   3. harness-compaction-entry  -- `maybeCompactAgentHarnessSession` is the agent-harness
#      compaction entry called once per turn. Early-returning the benign result covers the harness
#      path as well, regardless of whether the pi harness defines its own compact().
#
# Each patch only ever runs in a compaction code path, so nothing else regresses. Manual `/compact`
# is also disabled (it IS compaction; unused during training rollouts).
PATCHES: list[dict[str, str]] = [
    {
        "name": "preemptive-overflow-guard",
        "find": (
            "const maxContextChars = Math.max(1024, Math.floor(contextWindowTokens * 4 * PREEMPTIVE_OVERFLOW_RATIO));"
        ),
        "replace": "const maxContextChars = Number.POSITIVE_INFINITY;" + SENTINEL,
    },
    {
        "name": "compaction-executor",
        "find": "async function compactEmbeddedPiSessionDirect(params) {",
        "replace": "async function compactEmbeddedPiSessionDirect(params) {" + SENTINEL + " " + _NOOP_BODY,
    },
    {
        "name": "harness-compaction-entry",
        "find": "async function maybeCompactAgentHarnessSession(params) {",
        "replace": "async function maybeCompactAgentHarnessSession(params) {" + SENTINEL + " " + _NOOP_BODY,
    },
]

# When the model emits a tool call whose name is NOT a registered tool (e.g. a hallucinated `stop`/
# `submit`/`finish`), OpenClaw's history sanitizers DROP the toolCall block from the model-facing
# history -- at session record time AND at replay time -- keyed purely on tool-name allowlist
# membership. The generated "Tool <name> not found" error result is then orphaned (its matching
# toolCall is gone) and never reaches the model. Two harms: (1) the model never learns the call
# failed, so it re-emits the same call -> unbounded loop; (2) the tokens the model GENERATED for
# that call are absent from the next turn's prompt, breaking the prompt_token_ids prefix contiguity
# that on-policy GRPO RL requires (gen[n] must appear verbatim in prompt[n+1]).
#
# We want the unknown call to behave like a known-tool call with a bad argument (which OpenClaw
# already keeps WITH its error result): keep the toolCall in history, let it pair with the existing
# "not found" error result, and replay BOTH to the model -- so the model sees the error as a normal
# function_call_output, can recover in-episode, and the turn stays on-policy and trainable.
#
# Two surgical edits, each removing ONLY the allowlist-membership gate while preserving every
# structural check (name format regex, length cap, and the id / arguments presence checks -- a call
# with no id or no arguments is still dropped, because it cannot be safely paired). Verified by
# source-tracing the v2026.5.6 bundle; both name-acceptance predicates funnel every drop site:
#
#   1. surface-unknown-tool-record  -- `isAllowedToolCallName` (tool-call-id-*.js) is the predicate
#      used by `repairToolCallInputs` at the per-block record-time strip and the thinking-turn
#      replay-safety checks. Its final line gates on `allowedToolNames.has(...)`. Returning `true`
#      (after the format checks above it) keeps any structurally-valid name at record + the
#      compaction-replay path.
#
#   2. surface-unknown-tool-replay  -- `resolveReplayToolCallName` (selection-*.js) is the predicate
#      for the MODEL-FACING replay (`sanitizeReplayToolCallInputs`, both its per-block and
#      thinking-turn branches). It returns `resolveExactAllowedToolName(...)` which is null for an
#      unknown name -> block dropped. Falling back to `?? trimmed` keeps the (already format- and
#      whitespace-validated) name; near-matches of real tools still canonicalize via
#      resolveExactAllowedToolName, so legitimate normalization is unchanged.
#
# Both edits are name-agnostic (any hallucinated name is surfaced, not just `stop`) and model-/
# chat-template-agnostic (they only stop a history drop in OpenClaw's JS; downstream rendering is
# unchanged). Idempotent via SENTINEL_SURFACE_TOOL; fail loud if a marker shape changes.
SURFACE_TOOL_PATCHES: list[dict[str, str]] = [
    {
        "name": "surface-unknown-tool-record",
        "find": "return allowedToolNames.has(normalizeLowercaseStringOrEmpty(trimmed));",
        "replace": "return true;" + SENTINEL_SURFACE_TOOL,
    },
    {
        "name": "surface-unknown-tool-replay",
        "find": "return resolveExactAllowedToolName(trimmed, allowedToolNames);",
        "replace": "return resolveExactAllowedToolName(trimmed, allowedToolNames) ?? trimmed;" + SENTINEL_SURFACE_TOOL,
    },
]
PATCHES += SURFACE_TOOL_PATCHES

_RANK = {"missing": 0, "already": 1, "applied": 2}


def apply_patch_to_text(text: str, patch: dict[str, str]) -> tuple[str, str]:
    """Apply one patch to `text`. Returns (new_text, status).

    status is one of: "already" (sentinel-bearing replacement already present),
    "applied" (marker found and rewritten), "missing" (marker not in this text).
    Raises RuntimeError if the marker is present more than once (ambiguous site).
    """
    if patch["replace"] in text:
        return text, "already"
    count = text.count(patch["find"])
    if count == 0:
        return text, "missing"
    if count != 1:
        raise RuntimeError(f"patch {patch['name']!r}: marker is not unique ({count} matches)")
    return text.replace(patch["find"], patch["replace"]), "applied"


def patch_dist(dist_dir: Path) -> dict[str, str]:
    """Apply every patch to the top-level *.js bundles under `dist_dir` (non-recursive,
    so the plugin-sdk *.d.ts type stubs are never touched). Returns {patch_name: status}.
    Raises SystemExit if no bundles are found or any patch's marker is missing everywhere.
    """
    js_files = sorted(dist_dir.glob("*.js"))
    if not js_files:
        raise SystemExit(f"[patch_openclaw] no dist/*.js bundles under {dist_dir}")

    status = {p["name"]: "missing" for p in PATCHES}
    for jf in js_files:
        text = jf.read_text(encoding="utf-8")
        new_text = text
        for patch in PATCHES:
            updated, st = apply_patch_to_text(new_text, patch)
            if _RANK[st] > _RANK[status[patch["name"]]]:
                status[patch["name"]] = st
            if st == "applied":
                new_text = updated
        if new_text != text:
            jf.write_text(new_text, encoding="utf-8")

    missing = sorted(n for n, s in status.items() if s == "missing")
    if missing:
        raise SystemExit(
            f"[patch_openclaw] required patch marker(s) NOT FOUND: {missing}. OpenClaw may have "
            f"been upgraded; revisit the markers in patch_openclaw.py against {dist_dir}."
        )
    return status


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: patch_openclaw.py <openclaw/dist dir>", file=sys.stderr)
        return 2
    status = patch_dist(Path(argv[1]))
    for name, st in status.items():
        print(f"[patch_openclaw] {name}: {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
