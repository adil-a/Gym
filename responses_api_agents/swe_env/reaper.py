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

"""Sandbox reaper (plan §9): stops sandboxes whose TTL expired or whose owning
process is dead, and supports an atexit/SIGTERM bulk-stop.

Two reap predicates, so it is safe across processes/workers:
* ``ttl_s`` elapsed since ``created_at`` (verifier eval sandboxes get a short TTL), and
* ``owner_pid`` no longer alive (a crashed Ray worker / serving process).

It never reaps a record whose owner PID is still alive (a live sibling's sandbox).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Callable

from nemo_gym.sandbox import SandboxHandle, SandboxProvider
from responses_api_agents.swe_env.lifecycle import SandboxRecord, SandboxRegistry


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SandboxReaper:
    """Scans a :class:`SandboxRegistry` and stops dead/expired sandboxes."""

    def __init__(
        self,
        registry: SandboxRegistry,
        provider_resolver: Callable[[str], SandboxProvider],
    ) -> None:
        self.registry = registry
        self._resolve = provider_resolver
        self._task: asyncio.Task | None = None

    def reapable(self, now: float | None = None) -> list[SandboxRecord]:
        now = time.time() if now is None else now
        out: list[SandboxRecord] = []
        for rec in self.registry.list_records():
            ttl_expired = rec.ttl_s is not None and rec.created_at and (now - rec.created_at) > rec.ttl_s
            owner_dead = not pid_alive(rec.owner_pid)
            if ttl_expired or owner_dead:
                out.append(rec)
        return out

    async def _stop(self, rec: SandboxRecord) -> None:
        try:
            provider = self._resolve(rec.provider)
            await provider.close(SandboxHandle(sandbox_id=rec.sandbox_id, provider_name=rec.provider, raw={}))
        except Exception:
            pass
        self.registry.evict(rec.sandbox_id)

    async def reap_once(self, now: float | None = None) -> list[str]:
        reaped: list[str] = []
        for rec in self.reapable(now):
            await self._stop(rec)
            reaped.append(rec.sandbox_id)
        return reaped

    async def run_forever(self, interval_s: float = 60.0) -> None:
        while True:
            try:
                await self.reap_once()
            except Exception:
                pass
            await asyncio.sleep(interval_s)

    def start(self, interval_s: float = 60.0) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_forever(interval_s))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def stop_all_owned(self) -> list[str]:
        """Bulk-stop sandboxes owned by THIS process (atexit/SIGTERM backstop)."""
        stopped: list[str] = []
        for rec in self.registry.list_records():
            if rec.owner_pid == os.getpid():
                await self._stop(rec)
                stopped.append(rec.sandbox_id)
        return stopped
