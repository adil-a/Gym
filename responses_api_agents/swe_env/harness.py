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

"""Task model + harness contract for the SWE environment library.

The harness is the agent-agnostic re-expression of the legacy
``BaseDatasetHarnessProcessor`` ``setup/get_run_command/postprocess_after_run``
triad (swe_agents/app.py:313-323). The contract is intentionally split across a
trust boundary:

* ``build_spec`` / ``supports_provider`` / ``materialize`` are **provisioning**
  methods imported and called by *agents* (and the verifier).
* ``reset_repo`` / ``run_eval`` / ``grade`` are **server-private grading**
  methods used **only** by ``resources_servers/swe_env/verify_task.py``. A test
  asserts agent adapters never reference them (see plan §2 "single-class variant
  with the enforcement test").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox import SandboxSpec


if TYPE_CHECKING:
    from responses_api_agents.swe_env.environment import AsyncSweEnvironment


@dataclass
class SweTask:
    """A single SWE task to provision and/or verify.

    Mirrors the fields the legacy dataset processors read off
    ``problem_info['instance_dict']`` (base_commit app.py:601-602, test_patch
    :760, FAIL_TO_PASS/PASS_TO_PASS :637-638/:764-765, split :374).
    """

    instance_id: str
    image: str | None = None
    base_commit: str | None = None
    repo_workdir: str = "/testbed"
    test_command: str = ""
    test_framework: str = ""
    model_patch: str = ""
    test_patch: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    benchmark: str = "swe-bench-ext"
    split: str = "test"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalArtifacts:
    """Raw evaluation output retrieved from the sandbox, before grading."""

    test_output: str = ""
    return_code: int = 0
    patch_applied: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweEvalReport:
    """Graded result of a single task. ``error_kind`` masks a sample.

    ``error_kind`` is ``None`` for a clean grade. A non-``None`` value (e.g.
    ``"sandbox"`` / ``"eval_error"``) marks an infra failure: the sample is
    masked via this flag and ``reward_from_report`` returns ``0.0`` — **never**
    ``None`` (the wire ``reward`` field is a non-nullable ``float``).
    """

    instance_id: str
    resolved: bool = False
    patch_applied: bool = False
    patch_exists: bool = False
    error_kind: str | None = None
    tests_status: dict[str, Any] = field(default_factory=dict)


class SweTaskHarness(ABC):
    """Per-family provisioning + (server-private) grading recipe."""

    #: registry key, e.g. ``"swe-bench-ext"``.
    name: str = ""
    #: ``"flat-host-grade"`` (parse host-side) or ``"nested-harness"`` (in-container grader).
    grade_strategy: str = "flat-host-grade"

    # --- provisioning (agent-facing + verifier) ------------------------------

    @abstractmethod
    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec (image/workdir/env/ttl/provider_options) for a task."""

    def supports_provider(self, provider_name: str) -> bool:
        """Capability gate. Nested-Docker families override to reject exec-only providers."""
        return True

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Upload the model patch (+ test patch) into the started sandbox."""
        if task.model_patch:
            await env.write_text("/root/patch.diff", _ensure_trailing_newline(task.model_patch))
        if task.test_patch:
            await env.write_text("/root/test_patch.diff", _ensure_trailing_newline(task.test_patch))

    # --- server-private grading (verifier only) ------------------------------

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Reset the in-sandbox checkout to ``base_commit`` for hermetic grading.

        Only ``git reset --hard`` (matches legacy swe-bench-ext, app.py:943). We do
        NOT ``git clean -fdx``: verification runs in a FRESH sandbox (no agent edits
        to scrub), and clean would delete the image's prebuilt artifacts (compiled
        C extensions, installed env) and break the tests.
        """
        if task.base_commit:
            await env.execute(f"git reset --hard {task.base_commit}", cwd=task.repo_workdir)

    @abstractmethod
    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Apply the patch(es) and run the evaluation; return raw artifacts."""

    @abstractmethod
    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Host-side parse of the artifacts into a graded report."""


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
