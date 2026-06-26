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

"""Name-to-harness registry for dispatching tasks to their SWE harness."""

from __future__ import annotations

from responses_api_agents.swe_env.harness import SweTaskHarness


_HARNESSES: dict[str, SweTaskHarness] = {}


def register_harness(harness: SweTaskHarness, *, override: bool = False) -> None:
    """Register a harness under its ``name``.

    Args:
        harness (SweTaskHarness): The harness to register. Its ``name`` must be
            non-empty.
        override (bool): If ``True``, replace an existing harness with the same
            name instead of raising.

    Raises:
        ValueError: If the harness name is empty, or a harness with the same name
            is already registered and ``override`` is ``False``.
    """
    if not harness.name:
        raise ValueError("Harness must define a non-empty 'name'")
    if not override and harness.name in _HARNESSES:
        raise ValueError(f"Harness {harness.name!r} is already registered")
    _HARNESSES[harness.name] = harness


# HuggingFace dataset names don't match registry keys; map by substring (most-specific first)
# so callers can pass a raw ``dataset_name`` (e.g. "princeton-nlp/SWE-bench_Verified").
_HF_NAME_ALIASES: list[tuple[str, str]] = [
    ("SWE-bench_Multilingual", "swe-bench-multilingual"),
    ("R2E-Gym", "r2e-gym"),
    ("SWE-rebench", "swe-rebench"),
    ("SWE-bench", "swe-bench"),
]


def _ensure_registered() -> None:
    """Lazily register the built-in harnesses if the registry is empty.

    Importing ``responses_api_agents.swe_env.harnesses`` registers all families, but a fresh
    process (e.g. a Ray worker running the decoupled agent) may call ``get_harness`` before that
    import has run. Registering on demand keeps lookups robust regardless of import order.
    """
    if _HARNESSES:
        return
    from responses_api_agents.swe_env.harnesses import register_builtin_harnesses

    register_builtin_harnesses()


def get_harness(name: str) -> SweTaskHarness:
    """Look up a harness by registry key, or by HuggingFace dataset-name substring.

    Built-in harnesses are registered on first use (robust to import order). An exact key match
    wins; otherwise a HuggingFace ``dataset_name`` substring is resolved to its key (e.g.
    ``"princeton-nlp/SWE-bench_Verified"`` -> ``"swe-bench"``).

    Args:
        name (str): The registry key, or a HuggingFace dataset name.

    Returns:
        SweTaskHarness: The registered harness.

    Raises:
        KeyError: If no harness matches ``name``.
    """
    _ensure_registered()
    if name in _HARNESSES:
        return _HARNESSES[name]
    for needle, key in _HF_NAME_ALIASES:
        if needle in name and key in _HARNESSES:
            return _HARNESSES[key]
    available = ", ".join(sorted(_HARNESSES)) or "(none)"
    raise KeyError(f"Unknown SWE harness {name!r}. Registered: {available}")


def list_harnesses() -> list[str]:
    """List the names of all registered harnesses.

    Returns:
        list[str]: The registered harness names, sorted alphabetically.
    """
    return sorted(_HARNESSES)
