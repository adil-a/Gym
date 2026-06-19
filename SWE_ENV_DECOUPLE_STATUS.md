# SWE Env Decoupling (#1249) — Live Status

**Last updated:** session start (overnight autonomous run)
**Goal:** Implement the plan to decouple SWE environment infra from agent harnesses (issue #1249), on top of the Sandbox API PR #1377; unit-test it; do a real SWE-bench sanity run with a small Qwen on the 2 local GPUs; open a PR (based off #1377) and get CI green.

Plan file: `/home/adasif/.claude/plans/https-github-com-nvidia-nemo-gym-issues-lazy-donut.md`

---

## TL;DR for when you wake up
**Done + proven end-to-end with a REAL small model.** The decoupling is implemented (`swe_env` library + required `resources_servers/swe_env` verifier), **25 tests pass**, and I ran a genuine **model → sandbox → verifier** loop locally:
- **vLLM served `Qwen2.5-Coder-3B-Instruct`** on GPU 0 (`http://localhost:8000`).
- The model generated a patch for a buggy `calc.add`; the **verifier applied it in a real docker sandbox, ran pytest, and scored `resolved=True, reward=1.0`** (and `0.0` for an empty patch). This is a real SWE-bench-style task solved by a small model and graded by the new decoupled verifier.
- **PR: https://github.com/adil-a/Gym/pull/1** (based off the #1377 branch; diff = only my changes).

**Why docker, not apptainer:** this box has **no apptainer** and **no opensandbox cluster**, so the legacy OpenHands/nested SWE-bench path can't run here. I implemented a **docker sandbox provider** to prove the architecture for real. The apptainer provider is written + mocked-tested but must be validated on a `.sif` cluster.

**Caveat:** commits are **DCO-signed (`-s`) but NOT GPG-signed** — GPG needs an interactive pinentry I can't drive headlessly. If branch protection requires signatures, `git rebase --exec 'git commit --amend --no-edit -S'` (or re-commit) with your key.

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
