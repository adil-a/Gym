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

"""SWE dataset-family harnesses. Importing this package registers all 6 families.

Flat host-graded (run on any exec provider incl. docker): ``swe-bench-ext``,
``nv-internal-1``, ``swe-rebench``. Nested-harness (apptainer-only; run the
vendored ``run_local_evaluation`` in-container): ``swe-bench``,
``swe-bench-multilingual``, ``r2e-gym`` — these fail-fast on exec-only providers
and are validated on an apptainer/`.sif` cluster (see SWE_ENV_DECOUPLE_STATUS.md).
"""

from responses_api_agents.swe_env.harnesses.nv_internal import NVInternalHarness
from responses_api_agents.swe_env.harnesses.r2egym import R2EGymHarness
from responses_api_agents.swe_env.harnesses.swe_bench_ext import SweBenchExtHarness
from responses_api_agents.swe_env.harnesses.swe_rebench import SweRebenchHarness
from responses_api_agents.swe_env.harnesses.swebench import SweBenchHarness
from responses_api_agents.swe_env.registry import list_harnesses, register_harness


def register_builtin_harnesses() -> None:
    builtins = [
        SweBenchExtHarness(),
        NVInternalHarness(),
        SweRebenchHarness(),
        SweBenchHarness("swe-bench"),
        SweBenchHarness("swe-bench-multilingual"),
        R2EGymHarness(),
    ]
    existing = set(list_harnesses())
    for harness in builtins:
        if harness.name not in existing:
            register_harness(harness)


register_builtin_harnesses()


__all__ = [
    "NVInternalHarness",
    "R2EGymHarness",
    "SweBenchExtHarness",
    "SweBenchHarness",
    "SweRebenchHarness",
    "register_builtin_harnesses",
]
