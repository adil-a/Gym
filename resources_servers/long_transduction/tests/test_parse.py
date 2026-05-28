# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for long_transduction parsers."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `parse` importable without installing this server as a package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from parse import (  # noqa: E402
    _eval_expr,
    _normalize_expr,
    _parse_line,
    _parse_numbered_line,
    _parse_numbered_uuid_line,
    score_response,
    score_response_numbered,
    score_uuid_sort,
)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeExpr:
    def test_strips_internal_and_edge_whitespace(self):
        assert _normalize_expr("5 + 6 - 3") == "5+6-3"
        assert _normalize_expr("  5+6  ") == "5+6"

    def test_already_normalized(self):
        assert _normalize_expr("5+6") == "5+6"


class TestParseLine:
    def test_basic(self):
        assert _parse_line("5+6=11") == ("5+6", 11.0)

    def test_with_whitespace(self):
        assert _parse_line(" 5 + 6 = 11 ") == ("5+6", 11.0)

    def test_negative_answer(self):
        assert _parse_line("5-9=-4") == ("5-9", -4.0)

    def test_float_answer(self):
        assert _parse_line("5+6=11.0") == ("5+6", 11.0)

    def test_missing_equals(self):
        assert _parse_line("5+6") == (None, None)

    def test_garbage(self):
        assert _parse_line("hello world") == (None, None)

    def test_empty(self):
        assert _parse_line("") == (None, None)


class TestParseNumberedLine:
    def test_basic(self):
        assert _parse_numbered_line("[1]5+6=11") == (1, "5+6", 11.0)

    def test_multidigit_index(self):
        assert _parse_numbered_line("[42]5-3=2") == (42, "5-3", 2.0)

    def test_whitespace_after_bracket_is_tolerated(self):
        assert _parse_numbered_line("[3] 5 + 6 = 11") == (3, "5+6", 11.0)

    def test_missing_bracket(self):
        assert _parse_numbered_line("1]5+6=11") == (None, None, None)

    def test_missing_answer(self):
        assert _parse_numbered_line("[1]5+6") == (None, None, None)

    def test_empty(self):
        assert _parse_numbered_line("") == (None, None, None)


class TestEvalExpr:
    def test_chain(self):
        assert _eval_expr("5+6-3") == 8

    def test_single_operand(self):
        assert _eval_expr("5") == 5

    def test_negative_result(self):
        assert _eval_expr("0-9") == -9

    def test_long_chain(self):
        assert _eval_expr("5+6+7+8-1-1") == 24


# 8-char hex UUID constants used throughout uuid tests.
# Lex order: "1..." < "3..." < "a..." so UUID_A < UUID_C < UUID_B.
UUID_A = "11111111"
UUID_B = "aaaaaaaa"
UUID_C = "33333333"


class TestParseNumberedUuidLine:
    def test_canonical_format(self):
        # Canonical format: "[N]hex,hex,...".
        idx, uuids = _parse_numbered_uuid_line(f"[7]{UUID_A},{UUID_B}")
        assert idx == 7
        assert uuids == [UUID_A, UUID_B]

    def test_parens_tolerated(self):
        # Parens around tokens still parse — \b boundaries pick them out.
        idx, uuids = _parse_numbered_uuid_line(f"[1]({UUID_A}),({UUID_B})")
        assert idx == 1
        assert uuids == [UUID_A, UUID_B]

    def test_uppercase_lowercased(self):
        idx, uuids = _parse_numbered_uuid_line(f"[3]{UUID_B.upper()}")
        assert idx == 3
        assert uuids == [UUID_B]

    def test_no_uuids_after_index(self):
        idx, uuids = _parse_numbered_uuid_line("[5]xyz")  # no 8-hex tokens
        assert idx == 5
        assert uuids == []

    def test_partial_hex_not_matched(self):
        # Only 7 hex chars — must not match.
        idx, uuids = _parse_numbered_uuid_line("[1]abcdef0,12345678")
        assert idx == 1
        assert uuids == ["12345678"]  # only the 8-char one

    def test_no_bracket_prefix(self):
        assert _parse_numbered_uuid_line("just some text") == (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# score_response (unnumbered_streaming_sum)
# ─────────────────────────────────────────────────────────────────────────────

SUM_EXPRS = [
    {"expr": "5+6",   "answer": 11},
    {"expr": "2-3",   "answer": -1},
    {"expr": "1+2+3", "answer": 6},
]


class TestScoreResponse:
    def test_all_correct(self):
        out = "5+6=11\n2-3=-1\n1+2+3=6"
        assert score_response(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_missing_last_line(self):
        out = "5+6=11\n2-3=-1"
        scores = score_response(out, SUM_EXPRS)
        assert scores == [
            (True, True, True),
            (True, True, True),
            (False, False, False),
        ]

    def test_unparseable_lines_are_skipped(self):
        # Garbage lines drop out; remaining parseable lines align with expected.
        out = "5+6=11\nhello world\n2-3=-1\n!!\n1+2+3=6"
        assert score_response(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_wrong_expr_but_self_consistent(self):
        # Model wrote a different expression but did its OWN arithmetic right.
        out = "5+6=11\n2+3=5\n1+2+3=6"
        scores = score_response(out, SUM_EXPRS)
        assert scores[0] == (True, True, True)
        # Line 2: copy=False (2+3 != 2-3), answer=False (5 != -1),
        #         self_consistent=True (2+3 = 5 is correct arithmetic).
        assert scores[1] == (False, False, True)
        assert scores[2] == (True, True, True)

    def test_wrong_arithmetic_breaks_self_consistent(self):
        out = "5+6=99\n2-3=-1\n1+2+3=6"
        scores = score_response(out, SUM_EXPRS)
        # Line 0: copy=True, answer=False, self_consistent=False (5+6 != 99).
        assert scores[0] == (True, False, False)

    def test_empty_output(self):
        assert score_response("", SUM_EXPRS) == [(False, False, False)] * 3


# ─────────────────────────────────────────────────────────────────────────────
# score_response_numbered (streaming_sum + shuffled_streaming_sum)
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreResponseNumbered:
    def test_in_order(self):
        out = "[1]5+6=11\n[2]2-3=-1\n[3]1+2+3=6"
        assert score_response_numbered(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_out_of_order_matches_by_number(self):
        out = "[3]1+2+3=6\n[1]5+6=11\n[2]2-3=-1"
        assert score_response_numbered(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_missing_number_marks_only_that_index(self):
        out = "[1]5+6=11\n[3]1+2+3=6"  # [2] absent
        scores = score_response_numbered(out, SUM_EXPRS)
        assert scores[0] == (True, True, True)
        assert scores[1] == (False, False, False)
        assert scores[2] == (True, True, True)

    def test_duplicate_number_first_wins(self):
        out = "[1]5+6=11\n[1]9+9=999\n[2]2-3=-1\n[3]1+2+3=6"
        scores = score_response_numbered(out, SUM_EXPRS)
        assert scores[0] == (True, True, True)  # first [1] won

    def test_unparseable_lines_dont_shift_alignment(self):
        out = "[1]5+6=11\ngarbage\n[2]2-3=-1\n[3]1+2+3=6"
        assert score_response_numbered(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_empty_output(self):
        assert score_response_numbered("", SUM_EXPRS) == [(False, False, False)] * 3


# ─────────────────────────────────────────────────────────────────────────────
# score_uuid_sort (streaming_uuid_sort + shuffled_streaming_uuid_sort)
# ─────────────────────────────────────────────────────────────────────────────

# uuid_lines = the canonical input-order UUIDs per line (NOT pre-sorted).
UUID_LINES = [
    [UUID_B, UUID_A, UUID_C],  # line 1 input
    [UUID_C, UUID_A],          # line 2 input
]
EXPECTED_LINE_1 = sorted(UUID_LINES[0])  # [A, C, B]
EXPECTED_LINE_2 = sorted(UUID_LINES[1])  # [A, C]


def _render_uuid_line(idx: int, uuids: list[str]) -> str:
    """Canonical render: [N]hex,hex,..."""
    return f"[{idx}]" + ",".join(uuids)


class TestScoreUuidSort:
    """score_uuid_sort returns per-line (copy_correct, answer_correct, self_consistent).

      copy_correct    : set of model's emitted UUIDs for this line equals
                        the set of UUIDs in the input for this line.
      answer_correct  : strict positional+length match against the expected
                        sorted list.
      self_consistent : model's emitted UUIDs are in non-decreasing lex order.
    """

    def test_all_correct(self):
        out = "\n".join([
            _render_uuid_line(1, EXPECTED_LINE_1),
            _render_uuid_line(2, EXPECTED_LINE_2),
        ])
        assert score_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_input_order_right_uuids_wrong_sort(self):
        # Model echoed input order (didn't sort). Set matches input,
        # answer_correct=False (wrong order), self_consistent=False (not sorted).
        out = "\n".join([
            _render_uuid_line(1, UUID_LINES[0]),  # [B, A, C] — set OK, not sorted
            _render_uuid_line(2, EXPECTED_LINE_2),
        ])
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (True, False, False)
        assert scores[1] == (True, True, True)

    def test_wrong_uuids_but_sorted_self_consistent_only(self):
        # Model emitted entirely different UUIDs but in lex order.
        # copy_correct=False (wrong set), answer_correct=False,
        # self_consistent=True (the wrong list is itself sorted).
        wrong_sorted = sorted(["bbbbbbbb", "cccccccc", "dddddddd"])
        out = "\n".join([
            _render_uuid_line(1, wrong_sorted),
            _render_uuid_line(2, EXPECTED_LINE_2),
        ])
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (False, False, True)
        assert scores[1] == (True, True, True)

    def test_missing_line_all_false(self):
        out = _render_uuid_line(1, EXPECTED_LINE_1)
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (True, True, True)
        assert scores[1] == (False, False, False)

    def test_short_line_breaks_copy_and_answer(self):
        # Model emitted only 2 of 3 expected UUIDs. Set differs (missing UUID_B),
        # so copy_correct=False. Length mismatch -> answer_correct=False.
        # The two emitted are in lex order so self_consistent=True.
        out = "\n".join([
            _render_uuid_line(1, EXPECTED_LINE_1[:2]),
            _render_uuid_line(2, EXPECTED_LINE_2),
        ])
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (False, False, True)
        assert scores[1] == (True, True, True)

    def test_extra_uuids_break_copy_and_answer(self):
        # Model added an UUID not in input. Set differs -> copy_correct=False.
        # Length differs -> answer_correct=False. Appended "ff..." keeps sort.
        out = "\n".join([
            _render_uuid_line(1, EXPECTED_LINE_1 + ["ffffffff"]),
            _render_uuid_line(2, EXPECTED_LINE_2),
        ])
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (False, False, True)
        assert scores[1] == (True, True, True)

    def test_answer_correct_implies_copy_and_self_consistent(self):
        # When answer_correct is True, copy_correct and self_consistent must
        # also be True by construction (expected = sorted(input)).
        out = "\n".join([
            _render_uuid_line(1, EXPECTED_LINE_1),
            _render_uuid_line(2, EXPECTED_LINE_2),
        ])
        for triple in score_uuid_sort(out, UUID_LINES):
            copy, ans, sc = triple
            if ans:
                assert copy and sc

    def test_uppercase_uuids_match(self):
        out = "\n".join([
            _render_uuid_line(1, [u.upper() for u in EXPECTED_LINE_1]),
            _render_uuid_line(2, EXPECTED_LINE_2),
        ])
        assert score_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_out_of_order_lines_matched_by_index(self):
        out = "\n".join([
            _render_uuid_line(2, EXPECTED_LINE_2),
            _render_uuid_line(1, EXPECTED_LINE_1),
        ])
        assert score_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_parens_tolerated_in_model_output(self):
        line_1 = "[1]" + ",".join(f"({u})" for u in EXPECTED_LINE_1)
        out = "\n".join([line_1, _render_uuid_line(2, EXPECTED_LINE_2)])
        assert score_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_empty_output(self):
        assert score_uuid_sort("", UUID_LINES) == [
            (False, False, False),
            (False, False, False),
        ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
