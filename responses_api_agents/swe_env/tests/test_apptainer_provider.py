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

"""Apptainer provider tests (mocked subprocess — apptainer not installed here)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nemo_gym.sandbox import SandboxSpec
from responses_api_agents.swe_env.providers.apptainer_provider import ApptainerSandboxProvider


def _patch_run(provider, scripted):
    """Replace ``_run`` with a recorder returning scripted (rc, out, err) per call."""
    calls: list[list[str]] = []

    async def fake_run(*args, timeout_s=None):
        calls.append(list(args))
        return scripted(list(args))

    provider._run = fake_run  # type: ignore[assignment]
    return calls


def test_resolve_sif_direct_path(tmp_path: Path):
    sif = tmp_path / "image.sif"
    sif.write_text("x")
    provider = ApptainerSandboxProvider()
    spec = SandboxSpec(image=str(sif))
    assert provider._resolve_sif(spec) == str(sif)


def test_resolve_sif_glob(tmp_path: Path):
    (tmp_path / "myrepo__inst.sif").write_text("x")
    provider = ApptainerSandboxProvider(image_root=str(tmp_path))
    spec = SandboxSpec(image="inst", provider_options={"image_glob": "*.sif"})
    assert provider._resolve_sif(spec).endswith("myrepo__inst.sif")


def test_create_and_exec_issue_expected_argv(tmp_path: Path):
    sif = tmp_path / "image.sif"
    sif.write_text("x")
    provider = ApptainerSandboxProvider()
    calls = _patch_run(provider, lambda args: (0, "out", ""))

    handle = asyncio.run(
        provider.create(SandboxSpec(image=str(sif), workdir="/testbed", metadata={"instance_id": "i"}))
    )
    assert handle.provider_name == "apptainer"
    start_argv = calls[0]
    assert start_argv[:2] == ["instance", "start"]
    assert str(sif) in start_argv

    asyncio.run(provider.exec(handle, "echo hi", cwd="/testbed"))
    exec_argv = calls[-1]
    assert exec_argv[0] == "exec"
    assert any(a.startswith("instance://") for a in exec_argv)
    assert "--pwd" in exec_argv and "/testbed" in exec_argv


def test_exec_timeout_returns_typed_result(tmp_path: Path):
    provider = ApptainerSandboxProvider()

    async def timeout_run(*args, timeout_s=None):
        raise asyncio.TimeoutError

    provider._run = timeout_run  # type: ignore[assignment]
    from nemo_gym.sandbox import SandboxHandle

    handle = SandboxHandle(sandbox_id="x", provider_name="apptainer", raw={"workdir": "/t", "scratch": str(tmp_path)})
    result = asyncio.run(provider.exec(handle, "sleep 100", timeout_s=1))
    assert result.return_code == 124
    assert result.error_type == "timeout"
