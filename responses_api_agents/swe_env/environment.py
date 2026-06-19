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

"""Async SWE environment adapter over ``nemo_gym.sandbox`` (generalizes
``mini_swe_agent_2/sandbox_environment.py`` for any agent and the verifier)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, SandboxSpec


class AsyncSweEnvironment:
    """Thin async wrapper around a started ``AsyncSandbox``.

    Agents drive their own loop with ``execute``/``upload``/``download``; the
    verifier uses the same surface to run eval recipes. The environment never
    owns trajectory capture or grading logic — only sandbox I/O.
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        self._sandbox = sandbox
        self._closed = False

    @classmethod
    async def start(
        cls,
        provider: Mapping[str, Any] | SandboxProvider,
        spec: SandboxSpec,
    ) -> "AsyncSweEnvironment":
        """Create + start a fresh sandbox and return the environment."""
        sandbox = AsyncSandbox(provider, spec)
        await sandbox.start()
        return cls(sandbox)

    @property
    def sandbox(self) -> AsyncSandbox:
        return self._sandbox

    async def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        user: str | int | None = "root",
        timeout_s: int | float | None = None,
        is_eval: bool = False,
    ) -> dict[str, Any]:
        """Run a command; return a normalized dict (output/returncode/streams)."""
        result = await self._sandbox.exec(command, cwd=cwd, env=None, timeout_s=timeout_s, user=user)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = "\n".join(part for part in (stdout, stderr) if part)
        return {
            "output": output,
            "returncode": result.return_code,
            "stdout": stdout,
            "stderr": stderr,
            "error_type": result.error_type,
        }

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        await self._sandbox.upload(local_path, remote_path)

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        await self._sandbox.download(remote_path, local_path)

    async def write_text(self, remote_path: str, content: str) -> None:
        """Write a string to a file inside the sandbox (via a temp upload)."""
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()
            await self._sandbox.upload(tmp.name, remote_path)
        finally:
            os.unlink(tmp.name)

    async def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._sandbox.stop()

    async def __aenter__(self) -> "AsyncSweEnvironment":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()
