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

"""Name -> harness registry (replaces the 3 stringly-typed dataset-dispatch
sites in swe_agents/app.py: :1702, :1850, :2059)."""

from __future__ import annotations

from responses_api_agents.swe_env.harness import SweTaskHarness


_HARNESSES: dict[str, SweTaskHarness] = {}


def register_harness(harness: SweTaskHarness, *, override: bool = False) -> None:
    if not harness.name:
        raise ValueError("Harness must define a non-empty 'name'")
    if not override and harness.name in _HARNESSES:
        raise ValueError(f"Harness {harness.name!r} is already registered")
    _HARNESSES[harness.name] = harness


def get_harness(name: str) -> SweTaskHarness:
    try:
        return _HARNESSES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_HARNESSES)) or "(none)"
        raise KeyError(f"Unknown SWE harness {name!r}. Registered: {available}") from exc


def list_harnesses() -> list[str]:
    return sorted(_HARNESSES)
