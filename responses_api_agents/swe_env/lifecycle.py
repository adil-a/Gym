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

"""Sandbox lifecycle: durable registry, create-admission, and an acquire context
manager (plan §9).

#1377 ships no reaper / no durable registry / ``ttl_s`` defaults to ``None``, so
these are net-new #1249 components. The registry is a directory of one-JSON-file-
per-sandbox records (atomic temp+rename writes) so a separate process (the reaper)
can see sandboxes created by Ray workers and reap orphans on owner-pid death.
``acquire_sandbox`` records the sandbox immediately after create and always
stops + evicts it on exit (incl. cancellation).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

from nemo_gym.sandbox import SandboxProvider, SandboxSpec
from responses_api_agents.swe_env.environment import AsyncSweEnvironment


#: Per-process boot nonce — distinguishes a recycled PID from the original owner.
BOOT_NONCE = uuid.uuid4().hex


def content_key(*, instance_id: str, patch: str, harness: str, run_golden: bool = False) -> str:
    """Stable idempotency key for a verify request (instance + patch + harness)."""
    digest = hashlib.sha256()
    for part in (instance_id, patch or "", harness, "golden" if run_golden else "model"):
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass
class SandboxRecord:
    sandbox_id: str
    provider: str
    instance_id: str = ""
    run_id: str = ""
    attempt: int = 0
    created_at: float = 0.0
    ttl_s: float | None = None
    owner_pid: int = 0
    boot_nonce: str = ""
    content_key: str = ""


class SandboxRegistry:
    """Durable filesystem registry of live sandboxes (one atomic JSON per sandbox)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sandbox_id: str) -> Path:
        safe = sandbox_id.replace("/", "_")
        return self.root / f"{safe}.json"

    def record(self, rec: SandboxRecord) -> None:
        tmp = self.root / f".{rec.sandbox_id.replace('/', '_')}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(asdict(rec)), encoding="utf-8")
        tmp.replace(self._path(rec.sandbox_id))

    def evict(self, sandbox_id: str) -> None:
        self._path(sandbox_id).unlink(missing_ok=True)

    def list_records(self) -> list[SandboxRecord]:
        records: list[SandboxRecord] = []
        for path in self.root.glob("*.json"):
            try:
                records.append(SandboxRecord(**json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return records


class CreateAdmission:
    """Bounds concurrent ``provider.create()`` calls.

    In-process bound via an ``asyncio.Semaphore``; cross-process/worker visibility
    is provided by the registry (the documented ceiling ``M`` is per deployment —
    pin the verifier to a single worker or share one admission per plan §9).
    """

    def __init__(self, max_concurrent: int = 16) -> None:
        self.max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self) -> "CreateAdmission":
        await self._sem.acquire()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._sem.release()


@asynccontextmanager
async def acquire_sandbox(
    provider: Mapping[str, Any] | SandboxProvider,
    spec: SandboxSpec,
    *,
    registry: SandboxRegistry | None = None,
    admission: CreateAdmission | None = None,
    instance_id: str = "",
    run_id: str = "",
    attempt: int = 0,
    key: str = "",
) -> AsyncIterator[AsyncSweEnvironment]:
    """Admit, create, register, yield, then always stop + evict the sandbox."""
    admitted = False
    if admission is not None:
        await admission.__aenter__()
        admitted = True
    env: AsyncSweEnvironment | None = None
    sandbox_id: str | None = None
    try:
        env = await AsyncSweEnvironment.start(provider, spec)
        sandbox_id = env.sandbox_id
        if registry is not None and sandbox_id:
            registry.record(
                SandboxRecord(
                    sandbox_id=sandbox_id,
                    provider=env.provider_name or "",
                    instance_id=instance_id,
                    run_id=run_id,
                    attempt=attempt,
                    created_at=time.time(),
                    ttl_s=spec.ttl_s,
                    owner_pid=os.getpid(),
                    boot_nonce=BOOT_NONCE,
                    content_key=key,
                )
            )
        yield env
    finally:
        if env is not None:
            try:
                await env.cleanup()
            except Exception:
                pass
        if registry is not None and sandbox_id:
            registry.evict(sandbox_id)
        if admitted and admission is not None:
            await admission.__aexit__(None, None, None)
