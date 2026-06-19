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

"""Apptainer-backed ``SandboxProvider`` (ports the legacy ``.sif`` path).

Structural port of ``swe_agents/app.py`` ``_build_apptainer_command`` (:1800)
and ``_find_container`` (:1702) onto the #1377 provider Protocol, using a
long-lived ``apptainer instance`` so repo edits persist across exec calls and a
bind-mounted host scratch dir for file transfer.

NOTE: apptainer is not installed on the dev box this was authored on, so this
provider is exercised only via a mocked-subprocess unit test. Validate on an
apptainer/`.sif` cluster before relying on it (see SWE_ENV_DECOUPLE_STATUS.md).
"""

from __future__ import annotations

import asyncio
import glob
import os
import posixpath
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


_IO_MOUNT = "/sandbox_io"


class ApptainerSandboxProvider:
    """Run sandboxes as ``apptainer instance`` processes from ``.sif`` images."""

    name = "apptainer"

    def __init__(
        self,
        *,
        apptainer_bin: str = "apptainer",
        image_root: str | None = None,
        scratch_root: str | None = None,
        instance_args: list[str] | None = None,
        exec_args: list[str] | None = None,
        **_: Any,
    ) -> None:
        self._bin = apptainer_bin
        self._image_root = image_root
        self._scratch_root = scratch_root
        self._instance_args = list(instance_args or ["--writable-tmpfs", "--cleanenv"])
        self._exec_args = list(exec_args or [])

    async def _run(self, *args: str, timeout_s: int | float | None = None) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            self._bin, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            proc.kill()
            await proc.wait()
            raise
        rc = proc.returncode if proc.returncode is not None else -1
        return rc, out.decode(errors="replace"), err.decode(errors="replace")

    def _resolve_sif(self, spec: SandboxSpec) -> str:
        """Resolve a ``.sif`` path from provider_options or by glob (ports _find_container)."""
        sif = spec.provider_options.get("sif_path") or spec.image
        if sif and os.path.isfile(sif):
            return sif
        # Fuzzy glob under image_root, mirroring the legacy id-munging lookup.
        pattern = spec.provider_options.get("image_glob")
        roots = [r for r in (self._image_root, spec.provider_options.get("image_root")) if r]
        candidates: list[str] = []
        for root in roots:
            if pattern:
                candidates += glob.glob(os.path.join(root, pattern))
            elif sif:
                candidates += glob.glob(os.path.join(root, f"*{sif}*"))
                candidates += glob.glob(os.path.join(root, f"*{sif}*.sif"))
        if not candidates:
            raise SandboxCreateError(f"No .sif found for image={spec.image!r} (roots={roots}, glob={pattern!r})")
        return sorted(candidates)[-1]

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        sif = self._resolve_sif(spec)
        scratch = tempfile.mkdtemp(prefix="swe-apptainer-io-", dir=self._scratch_root)
        instance_name = f"swe-{(spec.metadata.get('instance_id') or 'task')[:24]}-{uuid.uuid4().hex[:8]}"
        args = ["instance", "start", *self._instance_args, "--bind", f"{scratch}:{_IO_MOUNT}"]
        for key, value in (spec.env or {}).items():
            args += ["--env", f"{key}={value}"]
        args += spec.provider_options.get("instance_args", [])
        args += [sif, instance_name]
        try:
            rc, out, err = await self._run(*args, timeout_s=spec.ready_timeout_s or 600)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            shutil.rmtree(scratch, ignore_errors=True)
            raise SandboxCreateError(f"apptainer instance start timed out for {sif!r}") from exc
        if rc != 0:
            shutil.rmtree(scratch, ignore_errors=True)
            raise SandboxCreateError(f"apptainer instance start failed (rc={rc}): {err.strip() or out.strip()}")
        return SandboxHandle(
            sandbox_id=instance_name,
            provider_name=self.name,
            raw={"sif": sif, "scratch": scratch, "workdir": spec.workdir},
        )

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        args = ["exec", *self._exec_args]
        workdir = cwd or handle.raw.get("workdir")
        if workdir:
            args += ["--pwd", workdir]
        for key, value in (env or {}).items():
            args += ["--env", f"{key}={value}"]
        args += [f"instance://{handle.sandbox_id}", "bash", "-c", command]
        try:
            rc, out, err = await self._run(*args, timeout_s=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            return SandboxExecResult(
                stdout=None, stderr=f"command timed out after {timeout_s}s", return_code=124, error_type="timeout"
            )
        return SandboxExecResult(stdout=out, stderr=err, return_code=rc)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        scratch = handle.raw["scratch"]
        base = posixpath.basename(target_path)
        shutil.copy(str(source_path), os.path.join(scratch, base))
        parent = posixpath.dirname(target_path)
        mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
        result = await self.exec(handle, f"{mkdir}cp {_IO_MOUNT}/{shlex.quote(base)} {shlex.quote(target_path)}")
        if result.return_code != 0:
            raise RuntimeError(f"apptainer upload copy failed: {result.stderr}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        scratch = handle.raw["scratch"]
        base = posixpath.basename(source_path)
        result = await self.exec(handle, f"cp {shlex.quote(source_path)} {_IO_MOUNT}/{shlex.quote(base)}")
        if result.return_code != 0:
            raise RuntimeError(f"apptainer download copy failed: {result.stderr}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(scratch, base), str(target))

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        rc, out, _ = await self._run("instance", "list", handle.sandbox_id)
        if rc != 0:
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if handle.sandbox_id in out else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        try:
            await self._run("instance", "stop", handle.sandbox_id)
        finally:
            shutil.rmtree(handle.raw.get("scratch", ""), ignore_errors=True)

    async def aclose(self) -> None:
        return None
