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

"""Lifecycle (registry / admission / acquire), reaper, and verify_task idempotency."""

from __future__ import annotations

import asyncio
import os
import time

import responses_api_agents.swe_env.harnesses  # noqa: F401  (register harnesses)
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxStatus
from resources_servers.swe_env.verify_task import clear_idempotency_cache, verify_task
from responses_api_agents.swe_env.harness import SweTask
from responses_api_agents.swe_env.lifecycle import (
    BOOT_NONCE,
    CreateAdmission,
    SandboxRecord,
    SandboxRegistry,
    acquire_sandbox,
    content_key,
)
from responses_api_agents.swe_env.reaper import SandboxReaper, pid_alive


class _CountingProvider:
    """Provider INSTANCE (passed directly) so we can count create/close/exec."""

    name = "fake-life"

    def __init__(self, *, exec_sleep=0.0, test_output="PASSED t::a\n"):
        self.create_count = 0
        self.close_count = 0
        self._exec_sleep = exec_sleep
        self._test_output = test_output

    async def create(self, spec):
        self.create_count += 1
        return SandboxHandle(
            sandbox_id=f"sb-{self.create_count}", provider_name=self.name, raw={"workdir": spec.workdir}
        )

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if self._exec_sleep:
            await asyncio.sleep(self._exec_sleep)
        if "pytest" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, *a, **k):
        return None

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        self.close_count += 1

    async def aclose(self):
        return None


def _task(**kw) -> SweTask:
    base = dict(
        instance_id="inst-1",
        image="img:tag",
        base_commit="HEAD",
        test_command="python -m pytest -rA -q",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::a"],
        benchmark="swe-bench-ext",
    )
    base.update(kw)
    return SweTask(**base)


# ---- content key ------------------------------------------------------------


def test_content_key_stable_and_sensitive():
    a = content_key(instance_id="i", patch="p", harness="h")
    assert a == content_key(instance_id="i", patch="p", harness="h")
    assert a != content_key(instance_id="i", patch="p2", harness="h")
    assert a != content_key(instance_id="i", patch="p", harness="h", run_golden=True)


# ---- registry ---------------------------------------------------------------


def test_registry_record_list_evict(tmp_path):
    reg = SandboxRegistry(tmp_path)
    reg.record(SandboxRecord(sandbox_id="sb1", provider="docker", owner_pid=os.getpid()))
    assert [r.sandbox_id for r in reg.list_records()] == ["sb1"]
    reg.evict("sb1")
    assert reg.list_records() == []


# ---- acquire_sandbox: records during, evicts + stops after ------------------


def test_acquire_sandbox_records_and_cleans_up(tmp_path):
    reg = SandboxRegistry(tmp_path)
    provider = _CountingProvider()
    spec_seen = {}

    async def run():
        from responses_api_agents.swe_env.harnesses.swe_bench_ext import SweBenchExtHarness

        spec = SweBenchExtHarness().build_spec(_task())
        async with acquire_sandbox(provider, spec, registry=reg, instance_id="inst-1", key="k") as env:
            spec_seen["records_during"] = len(reg.list_records())
            assert env.sandbox_id is not None
        spec_seen["records_after"] = len(reg.list_records())

    asyncio.run(run())
    assert spec_seen["records_during"] == 1
    assert spec_seen["records_after"] == 0
    assert provider.close_count == 1


def test_admission_is_async_context_manager():
    adm = CreateAdmission(2)
    assert adm.max_concurrent == 2

    async def run():
        async with adm:
            async with adm:
                pass

    asyncio.run(run())


# ---- reaper -----------------------------------------------------------------


def test_pid_alive():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(2_147_483_646) is False
    assert pid_alive(0) is False


def test_reaper_reaps_dead_and_expired_not_live(tmp_path):
    reg = SandboxRegistry(tmp_path)
    # alive owner, no ttl -> keep
    reg.record(SandboxRecord(sandbox_id="live", provider="fake-life", owner_pid=os.getpid(), boot_nonce=BOOT_NONCE))
    # dead owner -> reap
    reg.record(SandboxRecord(sandbox_id="orphan", provider="fake-life", owner_pid=2_147_483_646))
    # expired ttl (alive owner) -> reap
    reg.record(
        SandboxRecord(
            sandbox_id="expired", provider="fake-life", owner_pid=os.getpid(), created_at=time.time() - 100, ttl_s=1
        )
    )
    provider = _CountingProvider()
    reaper = SandboxReaper(reg, lambda name: provider)

    reaped = asyncio.run(reaper.reap_once())
    assert set(reaped) == {"orphan", "expired"}
    assert [r.sandbox_id for r in reg.list_records()] == ["live"]
    assert provider.close_count == 2


def test_reaper_stop_all_owned(tmp_path):
    reg = SandboxRegistry(tmp_path)
    reg.record(SandboxRecord(sandbox_id="mine", provider="fake-life", owner_pid=os.getpid()))
    reg.record(SandboxRecord(sandbox_id="other", provider="fake-life", owner_pid=2_147_483_646))
    provider = _CountingProvider()
    reaper = SandboxReaper(reg, lambda name: provider)
    stopped = asyncio.run(reaper.stop_all_owned())
    assert stopped == ["mine"]
    assert {r.sandbox_id for r in reg.list_records()} == {"other"}


# ---- verify_task idempotency (coalesce concurrent -> ONE create) ------------


def test_verify_task_idempotency_coalesces_concurrent(tmp_path):
    clear_idempotency_cache()
    reg = SandboxRegistry(tmp_path)
    provider = _CountingProvider(exec_sleep=0.05)

    async def run():
        # Two concurrent verifies of the SAME task -> same content key -> one create.
        return await asyncio.gather(
            verify_task(provider, _task(), registry=reg),
            verify_task(provider, _task(), registry=reg),
        )

    reports = asyncio.run(run())
    assert all(r.resolved for r in reports)
    assert provider.create_count == 1  # coalesced


def test_verify_task_eval_timeout_masks(tmp_path):
    clear_idempotency_cache()
    reg = SandboxRegistry(tmp_path)
    provider = _CountingProvider(exec_sleep=0.5)
    report = asyncio.run(verify_task(provider, _task(), registry=reg, eval_timeout_s=0.05, idempotent=False))
    assert report.error_kind == "eval_timeout"
