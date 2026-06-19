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

"""SWE dataset-family harnesses. Importing this package registers them.

Phase 1 implements the ``swe-bench-ext`` reference family end-to-end (flat,
host-graded). The other 5 families (swe-bench/multilingual/r2e-gym nested;
nv-internal/swe-rebench flat) are scaffolded in the plan and tracked as TODOs
in SWE_ENV_DECOUPLE_STATUS.md.
"""

from responses_api_agents.swe_env.harnesses.swe_bench_ext import SweBenchExtHarness
from responses_api_agents.swe_env.registry import register_harness


def register_builtin_harnesses() -> None:
    from responses_api_agents.swe_env.registry import list_harnesses

    if "swe-bench-ext" not in list_harnesses():
        register_harness(SweBenchExtHarness())


register_builtin_harnesses()


__all__ = ["SweBenchExtHarness", "register_builtin_harnesses"]
