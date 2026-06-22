# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure grading helpers shared by harnesses + the verifier server.

These functions never touch a sandbox; they decide ``resolved`` from parsed
test status and map a report to a (non-nullable) reward.
"""

from __future__ import annotations

from collections.abc import Iterable

from responses_api_agents.swe_env.harness import SweEvalReport


def compute_resolved(
    *,
    fail_to_pass: Iterable[str],
    pass_to_pass: Iterable[str],
    passed: Iterable[str],
) -> bool:
    """SWE-bench resolution rule: every FAIL_TO_PASS and PASS_TO_PASS test passes.

    Ports the ``check_tests_passed`` semantics (swe_agents/app.py:670).
    """
    passed_set = set(passed)
    required = list(fail_to_pass) + list(pass_to_pass)
    if not required:
        return False
    return all(test in passed_set for test in required)


def reward_from_report(report: SweEvalReport) -> float:
    """Map a report to a reward. Always a ``float`` (the wire field is non-nullable).

    An infra/eval failure (``error_kind`` set) yields ``0.0`` and is masked via
    the flag downstream — never ``None``.
    """
    if report.error_kind is not None:
        return 0.0
    return 1.0 if report.resolved else 0.0
