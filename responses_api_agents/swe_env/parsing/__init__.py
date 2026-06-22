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

"""Vendored SWE-Bench-Ext test-output parser, relocated into swe_env.

Copied verbatim from ``responses_api_agents/swe_agents/swe_bench_ext/``:
the per-framework parsers (``parsing.py``), framework output config
(``frameworks.py``), and the resolution helper (``utils.py``).

This ``__init__`` re-exports the public symbols that SWE harnesses use for
host-side grading so callers can import them from a single location, e.g.::

    from responses_api_agents.swe_env.parsing import (
        parse_and_check_tests,
        get_framework_config,
        get_test_command_with_output,
    )
"""

from responses_api_agents.swe_env.parsing.frameworks import (
    FRAMEWORK_CONFIGS,
    get_framework_config,
    get_test_command_with_output,
)
from responses_api_agents.swe_env.parsing.parsing import (
    normalize_test_id,
    parse_test_output,
)
from responses_api_agents.swe_env.parsing.utils import parse_and_check_tests


__all__ = [
    # utils.py — high-level grading entry point (F2P/P2P resolution)
    "parse_and_check_tests",
    # frameworks.py — framework output config + command augmentation
    "FRAMEWORK_CONFIGS",
    "get_framework_config",
    "get_test_command_with_output",
    # parsing.py — framework dispatcher + test-id normalization
    "parse_test_output",
    "normalize_test_id",
]
