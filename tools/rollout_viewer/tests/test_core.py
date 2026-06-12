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
from pathlib import Path

import core
import pytest


FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROLLOUTS = FIXTURES / "sample_rollouts.jsonl"


@pytest.fixture
def run():
    return core.load_run(core.derive_run_ref(ROLLOUTS))


# --------------------------------------------------------------------------- #
# Discovery & loading
# --------------------------------------------------------------------------- #
def test_derive_run_ref_discovers_siblings():
    ref = core.derive_run_ref(ROLLOUTS)
    assert ref.run_id == "sample_rollouts"
    assert ref.aggregate is not None and ref.aggregate.name.endswith("_aggregate_metrics.json")
    assert ref.materialized is not None
    assert ref.failures is not None


def test_derive_run_ref_missing_siblings(tmp_path):
    lonely = tmp_path / "lonely_rollouts.jsonl"
    lonely.write_text("{}\n", encoding="utf-8")
    ref = core.derive_run_ref(lonely)
    assert ref.aggregate is None
    assert ref.materialized is None
    assert ref.failures is None


def test_derive_run_ref_explicit_override(tmp_path):
    rollouts = tmp_path / "r.jsonl"
    rollouts.write_text("{}\n", encoding="utf-8")
    ref = core.derive_run_ref(rollouts, aggregate="/custom/metrics.json")
    assert ref.aggregate == Path("/custom/metrics.json")


def test_rollouts_stem_without_jsonl_suffix(tmp_path):
    # A path that does not end in .jsonl keeps its full name as the stem.
    weird = tmp_path / "noext"
    weird.write_text("{}\n", encoding="utf-8")
    ref = core.derive_run_ref(weird)
    assert ref.run_id == "noext"


def test_scan_dir_ignores_siblings():
    refs = core.scan_dir(FIXTURES)
    run_ids = [ref.run_id for ref in refs]
    assert run_ids == ["sample_rollouts"]  # siblings (_failures, _materialized) excluded


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert core.load_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_load_run_full(run):
    assert len(run.rollouts) == 3
    assert isinstance(run.aggregate, list)
    assert len(run.failures) == 1
    assert len(run.materialized) == 1


def test_load_run_missing_siblings(tmp_path):
    rollouts = tmp_path / "solo_rollouts.jsonl"
    rollouts.write_text('{"_ng_task_index": 0, "_ng_rollout_index": 0}\n', encoding="utf-8")
    loaded = core.load_run(core.derive_run_ref(rollouts))
    assert loaded.aggregate is None
    assert loaded.failures == []
    assert loaded.materialized == []


def test_load_run_handles_explicit_missing_aggregate(tmp_path):
    rollouts = tmp_path / "solo_rollouts.jsonl"
    rollouts.write_text("{}\n", encoding="utf-8")
    ref = core.derive_run_ref(rollouts, aggregate=tmp_path / "nope.json", failures=tmp_path / "nope.jsonl")
    loaded = core.load_run(ref)
    assert loaded.aggregate is None
    assert loaded.failures == []


# --------------------------------------------------------------------------- #
# Per-rollout accessors
# --------------------------------------------------------------------------- #
def test_rollout_accessors(run):
    first = run.rollouts[0]
    assert core.rollout_id(first) == "0:0"
    assert core.reward_of(first) == 1.0
    assert core.usage_of(first) == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert core.agent_name_of(first) == "agent"
    assert core.model_of(first) == "claude-x"


def test_reward_of_missing(run):
    assert core.reward_of(run.rollouts[2]) is None


def test_usage_of_empty(run):
    assert core.usage_of(run.rollouts[2]) == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }


def test_accessors_tolerate_empty_dict():
    assert core.usage_of({}) == {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    assert core.agent_name_of({}) is None
    assert core.model_of({}) is None


# --------------------------------------------------------------------------- #
# JSON + message helpers
# --------------------------------------------------------------------------- #
def test_safe_json():
    assert core._safe_json('{"a": 1}') == ({"a": 1}, True)
    assert core._safe_json("not json") == (None, False)
    assert core._safe_json(None) == (None, False)


def test_message_text_variants():
    assert core._message_text({"content": "hi"}) == "hi"
    assert core._message_text({"content": [{"text": "a"}, {"text": "b"}, {"no": "text"}]}) == "a\nb"
    assert core._message_text({"content": None}) == ""


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def test_classify_message_with_explicit_type():
    item = core.classify_item({"type": "message", "role": "user", "content": "hi"}, "input")
    assert item.kind == core.KIND_MESSAGE
    assert item.role == "user"
    assert item.text == "hi"


def test_classify_message_without_type():
    # The last fixture output message has role+content but no "type".
    item = core.classify_item({"role": "assistant", "content": [{"text": "done"}]}, "output")
    assert item.kind == core.KIND_MESSAGE
    assert item.text == "done"


def test_classify_tool_call_valid_and_invalid():
    good = core.classify_item(
        {"type": "function_call", "name": "f", "arguments": '{"x": 1}', "call_id": "c"}, "output"
    )
    assert good.kind == core.KIND_TOOL_CALL
    assert good.tool_args == {"x": 1}
    assert good.tool_args_ok is True

    bad = core.classify_item({"type": "function_call", "name": "f", "arguments": "oops", "call_id": "c"}, "output")
    assert bad.tool_args_ok is False
    assert bad.tool_args_raw == "oops"


def test_classify_tool_output_json_and_plain():
    js = core.classify_item({"type": "function_call_output", "call_id": "c", "output": '{"k": 1}'}, "output")
    assert js.tool_output_obj == {"k": 1}
    assert js.tool_output_ok is True

    plain = core.classify_item({"type": "function_call_output", "call_id": "c", "output": "hello"}, "output")
    assert plain.tool_output_ok is False
    assert plain.tool_output == "hello"


def test_classify_tool_output_non_string():
    item = core.classify_item({"type": "function_call_output", "call_id": "c", "output": {"k": 1}}, "output")
    assert item.tool_output_ok is False
    assert item.tool_output == '{"k": 1}'
    assert item.tool_output_obj == {"k": 1}


def test_classify_reasoning_with_and_without_summary():
    with_summary = core.classify_item(
        {"type": "reasoning", "summary": [{"text": "think"}], "encrypted_content": "sig"}, "output"
    )
    assert with_summary.reasoning_texts == ["think"]
    assert with_summary.has_encrypted_reasoning is True

    empty = core.classify_item({"type": "reasoning", "summary": [], "encrypted_content": "sig"}, "output")
    assert empty.reasoning_texts == []
    assert empty.has_encrypted_reasoning is True


def test_classify_unknown():
    item = core.classify_item({"type": "web_search_call", "id": "ws"}, "output")
    assert item.kind == core.KIND_UNKNOWN
    assert item.raw["type"] == "web_search_call"


def test_iter_transcript_order_and_counts(run):
    items = core.iter_transcript(run.rollouts[0])
    # 2 input messages + 5 output items
    assert len(items) == 7
    assert items[0].source == "input"
    assert items[-1].source == "output"
    kinds = [i.kind for i in items]
    assert kinds == [
        core.KIND_MESSAGE,
        core.KIND_MESSAGE,
        core.KIND_REASONING,
        core.KIND_MESSAGE,
        core.KIND_TOOL_CALL,
        core.KIND_TOOL_OUTPUT,
        core.KIND_MESSAGE,
    ]


def test_tools_of(run):
    assert core.tools_of(run.rollouts[0])[0]["name"] == "get_weather"
    assert core.tools_of(run.rollouts[2]) == []


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #
def test_diff_detects_divergence(run):
    rows = core.diff_rollouts(run.rollouts[0], run.rollouts[1])
    diverge = core.first_divergence(rows)
    # Identical up to the tool result (cold vs warm) at transcript index 5.
    assert diverge == 5
    assert rows[5].differs is True
    assert rows[0].differs is False


def test_diff_identical():
    rows = core.diff_rollouts({}, {})
    assert core.first_divergence(rows) is None


def test_diff_unequal_lengths(run):
    rows = core.diff_rollouts(run.rollouts[0], {})
    # Right side runs out; every row differs and right items are None.
    assert all(row.differs for row in rows)
    assert rows[0].right is None


def test_item_signature_none():
    assert core._item_signature(None) is None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_rollouts_to_frame(run):
    frame = core.rollouts_to_frame(run)
    assert list(frame["rollout_id"]) == ["0:0", "0:1", "1:0"]
    assert frame.loc[0, "reward"] == 1.0
    assert frame.loc[0, "total_tokens"] == 120
    assert frame.loc[0, "num_items"] == 7
    assert "agent" in frame.columns


def test_rollouts_to_frame_empty():
    frame = core.rollouts_to_frame(core.Run(run_id="x", rollouts=[]))
    assert frame.empty
    assert "reward" in frame.columns


def test_headline_metrics_present(run):
    headline = core.headline_metrics(run)
    assert len(headline) == 1
    assert headline[0]["agent"] == "agent"
    assert headline[0]["key_metrics"]["mean/reward"] == 0.5


def test_headline_metrics_absent():
    assert core.headline_metrics(core.Run(run_id="x", rollouts=[], aggregate=None)) == []


def test_headline_metrics_non_list():
    assert core.headline_metrics(core.Run(run_id="x", rollouts=[], aggregate={"unexpected": True})) == []


def test_headline_metrics_skips_non_dict_entries():
    run = core.Run(run_id="x", rollouts=[], aggregate=["junk", {"agent_ref": {"name": "a"}}])
    headline = core.headline_metrics(run)
    assert len(headline) == 1
    assert headline[0]["agent"] == "a"


def test_task_indices(run):
    assert core.task_indices(run) == [0, 1]


def test_task_indices_skips_none():
    run = core.Run(run_id="x", rollouts=[{"_ng_task_index": None}, {"_ng_task_index": 3}])
    assert core.task_indices(run) == [3]


def test_rollouts_for_task_ordered(run):
    task0 = core.rollouts_for_task(run, 0)
    assert [r["_ng_rollout_index"] for r in task0] == [0, 1]
    assert core.rollouts_for_task(run, 99) == []
