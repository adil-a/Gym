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

"""Local Docker-backed ``SandboxProvider`` implementation.

Implements the ``nemo_gym.sandbox`` provider Protocol via the ``docker`` CLI so
SWE environments can be provisioned and graded on any box with Docker — no
apptainer or opensandbox cluster required. This is what makes a real
end-to-end SWE-bench verification runnable on a single workstation.
"""

from __future__ import annotations

import asyncio
import posixpath
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


class DockerSandboxProvider:
    """Run sandboxes as long-lived Docker containers via the ``docker`` CLI."""

    name = "docker"

    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        default_user: str | int | None = None,
        network: str | None = None,
        run_args: list[str] | None = None,
        keep_alive_command: str = "sleep infinity",
        **_: Any,
    ) -> None:
        self._bin = docker_bin
        self._default_user = default_user
        self._network = network
        self._run_args = list(run_args or [])
        self._keep_alive = keep_alive_command

    async def _run(self, *args: str, timeout_s: int | float | None = None) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            self._bin,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            proc.kill()
            await proc.wait()
            raise
        return (
            proc.returncode if proc.returncode is not None else -1,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    @staticmethod
    def _resources(spec: SandboxSpec) -> SandboxResources:
        if isinstance(spec.resources, SandboxResources):
            return spec.resources
        return SandboxResources.from_mapping(spec.resources if isinstance(spec.resources, Mapping) else {})

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if not spec.image:
            raise SandboxCreateError("DockerSandboxProvider requires spec.image")
        args = ["run", "-d", "--init"]
        if self._network:
            args += ["--network", self._network]
        res = self._resources(spec)
        if res.memory_mib:
            args.append(f"--memory={int(res.memory_mib)}m")
        if res.cpu:
            args.append(f"--cpus={res.cpu}")
        if res.gpu:
            args.append("--gpus=all")
        if spec.workdir:
            args += ["-w", spec.workdir]
        for key, value in (spec.env or {}).items():
            args += ["-e", f"{key}={value}"]
        args += self._run_args
        args += [spec.image, "bash", "-c", self._keep_alive]
        try:
            rc, out, err = await self._run(*args, timeout_s=spec.ready_timeout_s or 600)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise SandboxCreateError(f"docker run timed out for image {spec.image!r}") from exc
        if rc != 0:
            raise SandboxCreateError(f"docker run failed (rc={rc}) for {spec.image!r}: {err.strip() or out.strip()}")
        container_id = out.strip().splitlines()[-1].strip()
        if not container_id:
            raise SandboxCreateError("docker run did not return a container id")
        return SandboxHandle(
            sandbox_id=container_id,
            provider_name=self.name,
            raw={"image": spec.image, "workdir": spec.workdir},
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
        args = ["exec"]
        workdir = cwd or handle.raw.get("workdir")
        if workdir:
            args += ["-w", workdir]
        eff_user = user if user is not None else self._default_user
        if eff_user is not None:
            args += ["-u", str(eff_user)]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        args += [handle.sandbox_id, "bash", "-c", command]
        try:
            rc, out, err = await self._run(*args, timeout_s=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            return SandboxExecResult(
                stdout=None,
                stderr=f"command timed out after {timeout_s}s",
                return_code=124,
                error_type="timeout",
            )
        # docker exec returns 125/126/127 for docker-level failures (container gone, not executable).
        error_type = "sandbox" if rc in (125, 126, 127) and not out else None
        return SandboxExecResult(stdout=out, stderr=err, return_code=rc, error_type=error_type)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        parent = posixpath.dirname(target_path)
        if parent:
            await self.exec(handle, f"mkdir -p {shlex.quote(parent)}")
        rc, out, err = await self._run("cp", str(source_path), f"{handle.sandbox_id}:{target_path}")
        if rc != 0:
            raise RuntimeError(f"docker cp upload failed: {err.strip() or out.strip()}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rc, out, err = await self._run("cp", f"{handle.sandbox_id}:{source_path}", str(target))
        if rc != 0:
            raise RuntimeError(f"docker cp download failed: {err.strip() or out.strip()}")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        rc, out, _ = await self._run("inspect", "-f", "{{.State.Running}}", handle.sandbox_id)
        if rc != 0:
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if out.strip() == "true" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        await self._run("rm", "-f", handle.sandbox_id)

    async def aclose(self) -> None:
        return None
