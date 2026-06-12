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
"""Streamlit shell for the rollout viewer.

This module is intentionally thin: all parsing/diff/metric logic lives in
``core.py`` (unit-tested), while this file only wires those outputs to widgets.
It is launched via ``ng_view_rollouts`` (which runs ``streamlit run`` on it).

Streamlit puts this file's directory on ``sys.path``, so ``import core`` resolves
to the sibling module when run as a script.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import altair as alt
import core
import pandas as pd
import streamlit as st


# Roles get a chat avatar; unknown roles fall back to a neutral one.
_ROLE_AVATARS = {"user": "🧑", "assistant": "🤖", "developer": "⚙️", "system": "⚙️", "tool": "🔧"}


def _parse_args() -> argparse.Namespace:
    """Parse args passed after Streamlit's ``--`` separator."""
    parser = argparse.ArgumentParser(prog="ng_view_rollouts")
    parser.add_argument("--dir", default="results", help="Directory to scan for rollout files.")
    parser.add_argument("--rollouts", default=None, help="Explicit path to a rollouts JSONL.")
    parser.add_argument("--metrics", default=None, help="Explicit path to an aggregate metrics JSON.")
    parser.add_argument("--materialized", default=None, help="Explicit path to a materialized inputs JSONL.")
    parser.add_argument("--failures", default=None, help="Explicit path to a failures JSONL.")
    # Streamlit may inject its own args; ignore unknowns.
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def _load_run_cached(
    rollouts: str, aggregate: str | None, materialized: str | None, failures: str | None, _mtime: float
):
    """Load a run, memoized on file paths + the rollouts file's mtime."""
    ref = core.derive_run_ref(rollouts, aggregate=aggregate, materialized=materialized, failures=failures)
    return core.load_run(ref)


def _mtime(path: str | None) -> float:
    try:
        return os.path.getmtime(path) if path else 0.0
    except OSError:
        return 0.0


def _select_run(args: argparse.Namespace) -> core.RunRef | None:
    """Build the sidebar run picker and return the chosen RunRef."""
    st.sidebar.header("Run")

    if args.rollouts:
        # Explicit file paths take precedence over directory scanning.
        st.sidebar.caption(f"Explicit rollouts file:\n`{args.rollouts}`")
        rollouts_path = args.rollouts
    else:
        scan_dir = st.sidebar.text_input("Results directory", value=args.dir)
        refs = core.scan_dir(scan_dir) if scan_dir and Path(scan_dir).is_dir() else []
        if not refs:
            st.sidebar.warning(f"No rollout files found in `{scan_dir}`.")
            return None
        labels = {ref.run_id: ref for ref in refs}
        chosen = st.sidebar.selectbox("Run", options=list(labels.keys()))
        rollouts_path = str(labels[chosen].rollouts)

    with st.sidebar.expander("Override sibling files"):
        aggregate = st.text_input("Aggregate metrics", value=args.metrics or "") or None
        materialized = st.text_input("Materialized inputs", value=args.materialized or "") or None
        failures = st.text_input("Failures", value=args.failures or "") or None

    return core.derive_run_ref(rollouts_path, aggregate=aggregate, materialized=materialized, failures=failures)


def _render_item(item: core.TranscriptItem, *, key: str) -> None:
    """Render a single transcript item with kind-appropriate styling."""
    if item.kind == core.KIND_MESSAGE:
        with st.chat_message(item.role or "assistant", avatar=_ROLE_AVATARS.get(item.role or "", "💬")):
            st.markdown(f"**{item.role or 'message'}**")
            st.markdown(item.text or "_(empty)_")

    elif item.kind == core.KIND_TOOL_CALL:
        with st.expander(f"🔧 tool call · `{item.tool_name}`", expanded=True):
            if item.tool_args_ok:
                st.json(item.tool_args)
            else:
                st.warning("Arguments are not valid JSON; showing raw string.")
                st.code(item.tool_args_raw or "", language="text")

    elif item.kind == core.KIND_TOOL_OUTPUT:
        with st.expander(f"📤 tool result · `{item.call_id}`", expanded=True):
            if item.tool_output_ok:
                st.json(item.tool_output_obj)
            else:
                st.code(item.tool_output or "", language="text")

    elif item.kind == core.KIND_REASONING:
        label = "🧠 reasoning"
        if not any(item.reasoning_texts) and item.has_encrypted_reasoning:
            label += " (encrypted; no summary)"
        with st.expander(label, expanded=False):
            if any(item.reasoning_texts):
                for text in item.reasoning_texts:
                    st.markdown(f"> {text}")
            else:
                st.caption("No summary text available.")

    else:  # KIND_UNKNOWN
        with st.expander(f"❓ unknown item · type=`{item.raw.get('type')}`", expanded=False):
            st.json(item.raw)


def _render_transcript(rollout: dict, *, key: str) -> None:
    """Render a rollout's full transcript plus a raw-JSON toggle."""
    for index, item in enumerate(core.iter_transcript(rollout)):
        _render_item(item, key=f"{key}-{index}")
    with st.expander("Show raw rollout JSON", expanded=False):
        st.json(rollout)


def _conversation_view(run: core.Run) -> None:
    st.subheader("Conversation")
    if not run.rollouts:
        st.info("This run has no rollouts.")
        return

    group_by_task = st.checkbox("Group by task (compare repeated rollouts)", value=False)

    if not group_by_task:
        labels = [f"{core.rollout_id(r)} · reward={core.reward_of(r)}" for r in run.rollouts]
        choice = st.selectbox("Rollout", options=list(range(len(run.rollouts))), format_func=lambda i: labels[i])
        _render_transcript(run.rollouts[choice], key="single")
        return

    # Grouped: pick a task, then either view one rollout or diff two.
    task = st.selectbox("Task", options=core.task_indices(run))
    task_rollouts = core.rollouts_for_task(run, task)
    labels = [f"rollout {r.get('_ng_rollout_index')} · reward={core.reward_of(r)}" for r in task_rollouts]

    if len(task_rollouts) >= 2 and st.checkbox("Diff two rollouts side by side", value=False):
        col_a, col_b = st.columns(2)
        idx_a = col_a.selectbox("Left", options=range(len(task_rollouts)), format_func=lambda i: labels[i], key="da")
        idx_b = col_b.selectbox(
            "Right",
            options=range(len(task_rollouts)),
            format_func=lambda i: labels[i],
            index=min(1, len(labels) - 1),
            key="db",
        )
        _diff_view(task_rollouts[idx_a], task_rollouts[idx_b])
    else:
        choice = st.selectbox("Rollout", options=range(len(task_rollouts)), format_func=lambda i: labels[i])
        _render_transcript(task_rollouts[choice], key="grouped")


def _diff_view(left: dict, right: dict) -> None:
    rows = core.diff_rollouts(left, right)
    diverge = core.first_divergence(rows)
    if diverge is None:
        st.success("Transcripts are identical.")
    else:
        st.warning(f"First divergence at transcript position {diverge}.")
    col_a, col_b = st.columns(2)
    for row in rows:
        marker = "🔺 " if row.differs else ""
        with col_a:
            st.markdown(f"{marker}**[{row.index}]**")
            if row.left is not None:
                _render_item(row.left, key=f"L{row.index}")
        with col_b:
            st.markdown(f"{marker}**[{row.index}]**")
            if row.right is not None:
                _render_item(row.right, key=f"R{row.index}")


def _metrics_view(run: core.Run) -> None:
    st.subheader("Metrics")
    frame = core.rollouts_to_frame(run)

    headline = core.headline_metrics(run)
    if headline:
        st.markdown("**Aggregate (from metrics file)**")
        for entry in headline:
            st.markdown(f"Agent `{entry['agent']}`")
            if entry["key_metrics"]:
                st.dataframe(pd.DataFrame([entry["key_metrics"]]), hide_index=True)
    else:
        st.caption("No aggregate metrics file; showing self-computed metrics only.")

    if frame["reward"].notna().any():
        st.markdown("**Reward distribution**")
        hist = (
            alt.Chart(frame.dropna(subset=["reward"]))
            .mark_bar()
            .encode(x=alt.X("reward:Q", bin=alt.Bin(maxbins=20)), y="count()")
        )
        st.altair_chart(hist, width="stretch")

    if frame[["total_tokens", "reward"]].notna().all(axis=None) and not frame.empty:
        st.markdown("**Tokens vs. reward**")
        scatter = (
            alt.Chart(frame)
            .mark_circle(size=80)
            .encode(
                x="total_tokens:Q",
                y="reward:Q",
                tooltip=["rollout_id", "task_index", "total_tokens", "reward"],
            )
        )
        st.altair_chart(scatter, width="stretch")

    st.markdown("**Per-rollout table**")
    st.dataframe(frame, hide_index=True, width="stretch")


def _failures_view(run: core.Run) -> None:
    st.subheader("Failures")
    if not run.failures:
        st.success("No failures recorded for this run.")
        return
    st.caption(f"{len(run.failures)} failed rollout(s).")
    for index, failure in enumerate(run.failures):
        with st.expander(f"Failure {index}", expanded=False):
            st.json(failure)


def main() -> None:
    st.set_page_config(page_title="NeMo Gym Rollout Viewer", layout="wide")
    st.title("NeMo Gym Rollout Viewer")

    args = _parse_args()
    ref = _select_run(args)
    if ref is None:
        st.info("Pick a results directory or pass --rollouts to get started.")
        return

    run = _load_run_cached(
        str(ref.rollouts),
        str(ref.aggregate) if ref.aggregate else None,
        str(ref.materialized) if ref.materialized else None,
        str(ref.failures) if ref.failures else None,
        _mtime(str(ref.rollouts)),
    )
    st.sidebar.metric("Rollouts", len(run.rollouts))

    conversation, metrics, failures = st.tabs(["Conversation", "Metrics", "Failures"])
    with conversation:
        _conversation_view(run)
    with metrics:
        _metrics_view(run)
    with failures:
        _failures_view(run)


if __name__ == "__main__":
    main()
