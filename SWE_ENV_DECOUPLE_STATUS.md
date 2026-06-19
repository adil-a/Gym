# SWE Env Decoupling (#1249) — Live Status

**Last updated:** items 2–5 complete; **CI all-green** (Lint/Test/copyright/secrets) on PR head `7ca0a987`.
**Goal:** Implement the plan to decouple SWE environment infra from agent harnesses (issue #1249), on top of the Sandbox API PR #1377; unit-test it; do a real SWE-bench sanity run with a small Qwen on the 2 local GPUs; open a PR (based off #1377) and get CI green.

Plan file: `/home/adasif/.claude/plans/https-github-com-nvidia-nemo-gym-issues-lazy-donut.md`

---

## TL;DR for when you wake up
**Plan items 2, 3, 4, 5 implemented; 87 tests pass + a real model→sandbox→verifier run.** PR: **https://github.com/adil-a/Gym/pull/1** (based off #1377).

What's done this session (on top of the earlier swe-bench-ext foundation):
- **Item 2 — all 6 families:** relocated the 1606-line vendored parser into `swe_env/parsing/`; added `nv-internal-1` + `swe-rebench` (flat, docker-runnable) and `swe-bench` + `swe-bench-multilingual` + `r2e-gym` (nested, apptainer-only, fail-fast on exec-only providers). All registered.
- **Item 3 — lifecycle/reaper/idempotency:** durable `SandboxRegistry`, `CreateAdmission`, always-teardown `acquire_sandbox`, `SandboxReaper` (ttl + owner-pid, never reaps a live sibling, atexit bulk-stop), and content-key idempotency in `verify_task` (coalesces unbounded ServerClient retries → one create) + per-call eval timeout.
- **Item 4 — wire-ownership + swe_agents cutover path:** the full `verify()` HTTP path is proven end-to-end (agent POSTs `BaseVerifyRequest` → non-nullable `reward` + `mask_sample`). Added `responses_api_agents/swe_agents/swe_env_adapter.py` — an **additive, tested SELF_DRIVING adapter** that provisions the OpenHands working container via `swe_env.lifecycle`, injects model-server egress, self-drives, extracts the patch, and scores it through the verifier (so `swe_agents` now *consumes* the decoupled env). It's additive (legacy `run()` untouched → test_app.py's 2010 mocked lines stay green); flipping `run()` to call it + deleting the legacy in-worker eval after a dual-run parity window is the final **apptainer/OpenHands-gated** step.
- **Item 5 — cross-cutting:** `model_endpoint` egress primitive (§6), reaper wired into the verifier server, verifier config + data-gate fixtures so `ng_test_all` passes upstream.
- Earlier-session proof still stands: **vLLM `Qwen2.5-Coder-3B-Instruct`** generated a patch → verifier scored `reward=1.0` in a real docker sandbox.

**apptainer is now installed + validated** (you ran the sudo install). Both sandbox providers are proven end-to-end with a real model:
- **docker provider:** Qwen-generated patch → fresh docker sandbox → `reward=1.0`.
- **apptainer provider:** built a `.sif` from the docker image; Qwen-generated patch → `apptainer instance` sandbox → `reward=1.0` (`test_apptainer_itest.py`, env-gated). The 3 nested families' provider gate is satisfied; their real-instance grading still needs published SWE-bench `.sif` images.

**The one genuinely-remaining item — the legacy `run()` flip:** replacing `SWEBenchWrapper.run()`'s two-container apptainer path with `acquire_sandbox` + verifier, and deleting the legacy code. I did **not** do this blind because it (a) rewrites the 2190-line `app.py` + the 2010-line **mocked** `test_app.py`, and (b) cannot be regression-tested here — a real OpenHands rollout needs the OpenHands harness **and a real SWE-bench instance `.sif` image** (not present; apptainer alone doesn't provide it). Doing it blind would risk the green, mergeable PR for un-validatable OpenHands-integration code. **Recommendation:** do the flip in an environment with the OpenHands harness + a real instance image (small, well-scoped — the decoupled path it targets is already validated on both providers + a real model). The additive `swe_env_adapter.py` is the ready migration entry point.

Other: commits are **DCO-signed but NOT GPG-signed** (headless pinentry) — re-sign if branch protection requires.

**Deliberately NOT done (to keep the PR mergeable / CI green):** rewiring the *legacy OpenHands `swe_agents`* and *`mini_swe_agent_2`* to call the new verifier — that cutover needs apptainer/opensandbox + their runtimes to validate, and doing it blind would risk breaking their CI. The env is fully consumable (contract proven in item 4); the cutover is the documented apptainer/opensandbox-gated follow-up.

---

## Environment (discovered)
- **GPUs:** 2× NVIDIA RTX 6000 Ada, 49 GB each, idle. ✅ plenty for a small Qwen.
- **Tooling:** `uv` 0.11.21 ✅, `docker` 29.6 ✅, `git`/`gh` ✅ (gh auth = `adil-a`).
- **`apptainer` / `singularity`: MISSING ❌** — this is the key constraint. The *legacy* `swe_agents` eval path and the 3 *nested* SWE-bench families require apptainer + on-Lustre `.sif` images, neither of which exist on this box.
- **Consequence for testing:** a full legacy-style end-to-end (OpenHands apptainer agent + apptainer eval) is **not runnable here**. opensandbox needs a k8s/opensandbox service (also not local). So real end-to-end testing uses a **docker/local sandbox provider** I implement, plus vLLM for the model half. Unit tests use a FakeSandbox. The full apptainer/opensandbox run command is documented for a box that has that infra.

## Branching
- Checked out PR #1377 head (`hemil/sandbox-api-part-1`) as local `pr-1377` (via `git fetch origin pull/1377/head`).
- Working branch: **`feat/swe-env-decouple-1249`** (off `pr-1377`).
- PR will be opened **based off #1377** (fork-internal base = a copy of the #1377 branch) so the diff is only my changes on top of the sandbox API. Will be retargeted to upstream `main` once #1377 merges.

## Decisions made (autonomously)
- **Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (fits one 49GB GPU comfortably; better code ability than 3B for a meaningful patch attempt). Served via Gym's `vllm_model` server.
- **Scope tonight:** implement the plan's *first coherent, unit-tested increment* — the `swe_env` library (provisioner / grading recipe / registry / grading / environment / parsing / providers / minimal lifecycle), the **swe-bench-ext** reference family end-to-end, and the **required `resources_servers/swe_env/` verifier** with a server-private `verify_task` + FakeSandbox tests. Other 5 families, full reaper/idempotency depth, and config consolidation are scaffolded/deferred with clear TODOs. (Rationale: this is the heart of the decoupling and is fully testable without GPUs/apptainer.)
- **venv:** root `.venv` in the Gym dir (already exists); synced with the `[sandbox]` extra.

---

## Progress log
- [done] Recon; branch `feat/swe-env-decouple-1249` off `pr-1377`.
- [done] Implemented `responses_api_agents/swe_env/` library: `harness.py` (SweTask/EvalArtifacts/SweEvalReport + `SweTaskHarness` ABC with the provisioning/grading trust split), `environment.py` (`AsyncSweEnvironment` over `nemo_gym.sandbox`), `grading.py` (`compute_resolved`/`reward_from_report`), `registry.py`, `providers/` (`DockerSandboxProvider` — real/local; `ApptainerSandboxProvider` — ports the legacy `.sif` path, mocked-tested), `harnesses/swe_bench_ext.py` (reference flat family).
- [done] Implemented `resources_servers/swe_env/`: server-private `verify_task.py` orchestrator (fresh-only: acquire → reset → materialize → run_eval → grade → teardown) + `app.py` (`SweEnvVerifier(SimpleResourcesServer).verify`, patch extraction, masking via `reward=0.0`+flag never `None`).
- [done] **25 tests pass** (lib + apptainer-mocked + verifier). **Real docker e2e PASSED** (`SWE_ENV_DOCKER_ITEST=1`): gold patch → resolved/reward 1.0; empty → 0.0.
- [done] Added `--recount` to the swe-bench-ext apply (mirrors legacy app.py:989) so model-generated diffs with imperfect `@@` counts still apply.
- [done] **vLLM `Qwen2.5-Coder-3B-Instruct` served** (docker, GPU 0). **Model-driven e2e: reward 1.0** (model fixed the bug; verifier graded it in a real docker sandbox). Demo script: `/tmp/swe_env_local_demo.py` (not committed; reproducible).
- [done] PR opened off #1377: https://github.com/adil-a/Gym/pull/1.
- [done] **CI GREEN** ✅ — Test (per-server `ng_test`, 1m12s), Lint, copyright-check, secrets-detector all **pass** (`request` skips on forks).

## Current status
**DONE for this increment: implemented, tested (25 + real docker e2e + real model-driven e2e), PR open off #1377, CI green.** Remaining work is the deferred follow-ups below (other 5 families, data-gate fixtures, lifecycle depth, rewiring legacy swe_agents, retarget to upstream after #1377 merges).

### How to reproduce the model-driven run
```bash
# 1) serve the model (GPU 0)
docker run -d --name swe-vllm --gpus '"device=0"' -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8000:8000 vllm/vllm-openai:latest --model Qwen/Qwen2.5-Coder-3B-Instruct --max-model-len 8192
# 2) run the loop (from repo root)
.venv-swe/bin/python /tmp/swe_env_local_demo.py
# 3) the swe_env test suite
.venv-swe/bin/python -m pytest responses_api_agents/swe_env/tests resources_servers/swe_env/tests \
  -o addopts="" -q --import-mode=importlib            # add SWE_ENV_DOCKER_ITEST=1 for the real-docker e2e
```

## What's tested
- `responses_api_agents/swe_env/tests/test_swe_env.py` — parse/grade/reward/registry + `verify_task` resolved/unresolved/empty-patch/infra-masked/golden/patch-not-applied/unsupported-provider (FakeSandbox).
- `responses_api_agents/swe_env/tests/test_apptainer_provider.py` — sif resolve (direct + glob), create/exec argv, timeout (mocked subprocess; apptainer absent).
- `resources_servers/swe_env/tests/test_verify.py` — verify() adapter (`build_task`/`extract_patch`/`_as_list`), reward correctness (FakeSandbox), + **env-gated real docker e2e** (`SWE_ENV_DOCKER_ITEST=1`, never runs in CI).
- Run locally: `.venv-swe/bin/python -m pytest responses_api_agents/swe_env/tests resources_servers/swe_env/tests -o addopts="" -q --import-mode=importlib` → 24 passed, 1 skipped; add `SWE_ENV_DOCKER_ITEST=1` for the 25th (real docker).

## What landed in this PR (scope of the increment)
- `responses_api_agents/swe_env/` library: `harness.py`, `environment.py`, `grading.py`, `registry.py`, `providers/{docker,apptainer}_provider.py`, `harnesses/swe_bench_ext.py`, `requirements.txt`, tests.
- `resources_servers/swe_env/` verifier: `app.py` (`SweEnvVerifier.verify`), server-private `verify_task.py`, `requirements.txt`, tests.
- Proven: 25 tests green; real docker e2e (gold patch → 1.0); model-driven e2e with Qwen2.5-Coder-3B (→ 1.0).

## Known constraints (environment, not bugs)
- **No apptainer / no opensandbox cluster** on this box → the legacy OpenHands path and the 3 nested SWE-bench families can't run here. Validated the architecture with a **docker** provider instead. The apptainer provider is written + mocked-tested; validate it on a `.sif` cluster.
- Commits **DCO-signed but not GPG-signed** (headless pinentry). Re-sign if branch protection requires.

## Follow-ups (deferred, in rough priority order)
1. **Retarget the PR to upstream `main` once #1377 merges** (it's currently based off a copy of the #1377 branch in the fork). Until then it carries #1377's commits underneath.
2. **Data-gate fixtures** for `resources_servers/swe_env` (`data/example.jsonl` ×5 + `example_metrics.json` + `example_rollouts.jsonl` ×5) so the **full** suite / `ng_test_all` passes. Per the plan these must be a real agent `/run` output (use a gold/patch-injecting agent), not raw patches. *(Not needed for the current server-only CI; needed before merge to upstream where `ng_test_all` runs.)*
3. **Remaining 5 families**: nested swe-bench/multilingual/r2e-gym (apptainer-only) + flat nv-internal/swe-rebench; relocate the full vendored `swe_bench_ext` parser (1606 lines) for real published instances.
4. **Lifecycle/reaper + idempotency** depth (durable registry, owner-pid reaper, content-key idempotency, `ClientTimeout`) — see plan §9.
5. **Rewire the legacy OpenHands `swe_agents`** (and `mini_swe_agent_2`) to consume `swe_env` + POST to the verifier (plan §7 step 7/8); dual-run reward parity before deleting the legacy in-worker eval.
6. **Config consolidation** (`swe_env_base.yaml` + `${inherit_from:...}`) and the cross-tree packaging note (plan §2/§5).

## Cleanup notes
- Local venv: `.venv-swe/` (owned by you; the repo `.venv`/`vllm_venv` are root-owned from your container and unusable here).
- vLLM container `swe-vllm` (GPU 0) + image `swe-env-itest:local` were created for testing; stop/remove with `docker rm -f swe-vllm` and `docker rmi swe-env-itest:local` if you want the GPU/space back.
