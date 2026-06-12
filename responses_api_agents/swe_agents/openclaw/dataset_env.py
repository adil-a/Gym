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

"""Dataset-name-driven resolution of the agent's in-container workspace dir and the
Python-runtime env bin that the agent's ``python`` must resolve to.

Mirrors OpenHands' single ``DATASET_TYPE`` model (``set_dataset_type`` +
``_get_workspace_path`` + the per-dataset ``instance_swe_entry_*.sh`` env activation in
``evaluation/benchmarks/swe_bench``), so OpenClaw supports every dataset OpenHands does,
driven only by the dataset name.

Two deliberate differences from OpenHands:
  * Workspace resolves to the **in-place** repo dir (e.g. ``/testbed``), skipping the
    ``cp -al /testbed -> /workspace/{name}`` copy — that copy is an OpenHands-runtime
    artifact (its agent operates out of a ``/workspace`` root and keeps ``/testbed``
    pristine within one long-lived container); OpenClaw runs agent and eval in separate
    container invocations with ``patch.diff`` as the handoff and the eval independently
    resets the repo, so working in place is correct and simpler.
  * Activation rides on OpenClaw's ``tools.exec.pathPrepend`` (the env's ``bin`` dir)
    rather than ``conda activate``. OpenClaw's ``exec`` rebuilds the command PATH from a
    sanitized base (conda *base*, no deps) and discards any inherited/activated PATH, then
    applies ``pathPrepend`` on top — so the env bin on ``pathPrepend`` (after the
    security-wrapper dir, which stays first) is the only lever that reaches the agent.

If a dataset's env is absent the prepended dir simply doesn't resolve and the agent falls
back to the image's default interpreter — i.e. it fails **silently**, matching OpenHands'
``if [ -d /opt/miniconda3 ]; then conda activate testbed; fi`` guard.
"""

from __future__ import annotations

from typing import Any, Optional


# SWE-bench-family SIFs install the repo + its deps into the /opt/miniconda3 'testbed'
# conda env; R2E-Gym uses an in-repo venv at /testbed/.venv.
_CONDA_TESTBED_BIN = "/opt/miniconda3/envs/testbed/bin"


def resolve_dataset_type(dataset_name: Optional[str]) -> str:
    """Map a dataset name to OpenHands' canonical DATASET_TYPE (same substring rules)."""
    name = (dataset_name or "").lower()
    if "nv-internal-1" in name:
        return "nv-internal-1"
    if "swe-rebench-v2" in name or "swe-rebench_v2" in name:
        return "SWE-rebench-V2"
    if "swe-gym" in name:
        return "SWE-Gym"
    if "r2e-gym" in name:
        return "R2E-Gym"
    if "swe-bench-live" in name:
        return "SWE-bench-Live"
    if "swe-rebench" in name:
        return "SWE-rebench"
    if "multimodal" in name:
        return "Multimodal"
    if "multilingual" in name:
        return "SWE-bench_Multilingual"
    if "swe-bench-ext" in name:
        return "swe-bench-ext"
    return "SWE-bench"


def resolve_workspace_path(dataset_type: str, instance_dict: Optional[dict[str, Any]] = None) -> str:
    """In-place repo dir for the dataset (OpenHands ``_get_workspace_path`` without the
    ``/workspace`` copy). Defaults to ``/testbed`` (SWE-bench-family SIFs); nv-internal-1
    (``/app``), swe-bench-ext (``/workspace/repo``), and SWE-rebench-V2 (``/{repo_name}``)
    override."""
    instance_dict = instance_dict or {}
    if dataset_type == "nv-internal-1":
        return "/app"
    if dataset_type == "swe-bench-ext":
        return "/workspace/repo"
    if dataset_type == "SWE-rebench-V2":
        repo = instance_dict.get("repo", "") or ""
        repo_name = repo.split("/")[1] if "/" in repo else repo
        if not repo_name:
            raise ValueError("SWE-rebench-V2 requires the instance 'repo' field to resolve the workspace path")
        return f"/{repo_name}"
    # SWE-bench, SWE-bench_Multilingual, SWE-rebench, SWE-Gym, R2E-Gym, SWE-bench-Live, Multimodal
    # (SWE-bench is the default / most common case)
    return "/testbed"


def resolve_agent_env_bin(dataset_name: Optional[str]) -> Optional[str]:
    """The env ``bin`` dir to prepend onto openclaw ``tools.exec.pathPrepend`` so the
    agent's ``python`` is the repo interpreter (with deps), not the dep-less base. Returns
    ``None`` for datasets that use the image's default interpreter (no activation)."""
    dataset_type = resolve_dataset_type(dataset_name)
    if dataset_type in ("SWE-bench", "SWE-bench_Multilingual", "SWE-rebench", "SWE-Gym", "Multimodal"):
        return _CONDA_TESTBED_BIN
    if dataset_type == "R2E-Gym":
        return "/testbed/.venv/bin"
    # SWE-bench-Live, nv-internal-1, SWE-rebench-V2, swe-bench-ext: image default interpreter.
    return None
