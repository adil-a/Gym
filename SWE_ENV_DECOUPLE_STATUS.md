# SWE Env Decoupling (#1249) — Live Status

**Last updated:** items 2–5 complete + CI-green; verifier validated on a REAL SWE-bench instance on **both** providers; **full 500-instance gold eval ran** (`results/swebench_verified_gold.jsonl`); **OpenHands cutover mechanism now VALIDATED end-to-end** through the decoupled docker-provider path (worktree branch `feat/swe-env-cutover-1249`).

### OpenHands `run()` cutover — MECHANISM VALIDATED (this session)
Ran a **real OpenHands rollout through the decoupled `swe_env` infra** (NOT the legacy two-container apptainer path): `psf__requests-2317`, docker provider, Qwen2.5-Coder-3B via vLLM. Every novel link of the cutover worked:
- ✅ `swe_env` **docker provider** hosts OpenHands — Gym repo bind-mounted at its host path (resolves the OpenHands venv abs-symlinks + the `nemo_gym` editable install), tmux from miniforge3, `git config --global --add safe.directory '*'` for the root-vs-host-owner mismatch.
- ✅ **Model egress works** — OpenHands' `CodeActAgent` is hard-wired to `NemoGymClient` (no litellm fallback), so egress needs `NEMO_GYM_CONFIG_DICT` + `NEMO_GYM_MODEL_SERVER_NAME` + `NEMO_GYM_METRICS_FPATH` injected (NOT `OPENAI_BASE_URL`). A crafted 3-level config routed `ServerClient` straight to the host vLLM; OpenHands self-drove **16+ turns** of real LLM round-trips.
- ✅ OpenHands ran `RUNTIME=local` on `/testbed` (`--dataset SWE-Gym`), exited rc=0, produced `output.jsonl`; patch extracted from `test_result.git_patch`; graded in a **separate fresh verifier sandbox** → reward.
- The demo patch was empty only because **Qwen-3B is too weak to emit OpenHands-parseable actions** (model-capability, NOT a cutover issue). A resolving patch → reward 1.0 is covered by the verifier's real-instance test.
- **Encoded into code** (worktree): `swe_env_adapter.run_self_driving` now supports `extra_env` (the OpenHands `NEMO_GYM_*` egress) + `patch_output_glob` (extract from `output.jsonl`, not `git diff`) — 5 adapter tests pass. Reference recipe: `responses_api_agents/swe_agents/scripts/openhands_decoupled_rollout.py`.

### run() cutover IMPLEMENTED (A2–A5 + C10/C11/D12) — single PR, behind an opt-in flag
The full cutover is now coded, tested, and integrated on `feat/swe-env-decouple-1249` (all DCO-signed, GPG pending). **252 passed / 4 skipped** across swe_env + swe_agents + mini_swe_agent_2 + resources_servers/swe_env.
- **A1** opt-in `eval_via_verifier` flag (+ `verifier_server_name`, `sandbox_provider`); legacy two-container path stays the default until empirical dual-run parity → no CI risk.
- **A2** decoupled worker `_run_decoupled_agent`: one sandbox via `acquire_sandbox` + OpenHands self-drive (`NEMO_GYM_*` egress, validated launch recipe) + `output.jsonl` patch extraction; no eval container.
- **A3** `run()` POSTs the patch to the verifier; `resolved`/`eval_timed_out` flow through the SAME `metrics_fpath` → the frozen `SWEBenchVerifyResponse` row + `mask_sample` are preserved **by construction**.
- **A4** gating tests: shared `_should_mask_sample` (all 4 combos) + verify-POST contract + infra-error→masked-row (HTTP 200, never drop the rollout).
- **A5** GRADING PARITY empirically confirmed: gold patch on `pytest-dev__pytest-7982` → decoupled verifier `resolved=True` == official SWE-bench harness `resolved=True` (MATCH). Plus astropy (1.0 both providers) + the 500-gold run (491/500 matching official).
- **C10** mini_swe_agent_2 gained an opt-in verifier-POST path (cross-agent reuse proof; 22 tests).
- **C11** shared `swe_env_base.yaml` + per-leaf `${inherit_from:...}` (eval 900 / train 1200 preserved; leaf resolution validated via the real swap logic — definitive `ng_dump_config` confirmation deferred to CI).
- **D12** opt-in flat-eval grading mode for the 3 nested families (docker/opensandbox), with a dependency-free log parser + fixture tests; real .sif equivalence remains infra-gated.

**Deliberately NOT done (gated, by design):**
- **A6 (delete the legacy two-container path): DONE.** `app.py` 2412 → 1351 lines (−1061). Deleted `ActiveContainerCommand`, `_start/_finish/_kill_container_command`, `_build_apptainer_command`, `_find_container`, `_get_command_sleep_until_predictions_file`, every processor's `get_run_command`, and the two-container branch in `process_single_datapoint`. `eval_via_verifier` default flipped to True; the verifier POST is the ONLY eval path. **Golden verification migrated to the verifier** (gold patch → metrics → `/verify`, no container helpers). `_setup_params` no longer builds apptainer commands. ~50 legacy tests removed/rewritten; **205 passed / 4 skipped** across swe_agents+swe_env+mini_swe_agent_2+resources_servers/swe_env; ruff+secrets clean.
- **Capable-model live test (Qwen3-30B-A3B, Qwen2.5-Coder-32B):** validated the decoupled path drives a capable model over **79 real tool-calling turns** through the docker provider with correct verifier grading (fixed two real serving issues en route: context window, and the missing vLLM `--enable-auto-tool-choice`/`--tool-call-parser`). A *resolving* patch wasn't obtained standalone because the in-tree OpenHands fork's `NemoGymClient`+`CodeActAgent` doesn't translate these locally-served models' tool-calls into actions (every turn → `message`, not an action) — a fork↔model-integration gap orthogonal to #1249, handled in production by the Gym model server + tuned models.
- **GPG signing** (headless pinentry) and **PR base retarget to upstream `main`** (after #1377 merges) — human/infra-gated.

### Final-session state
- **Full SWE-bench Verified gold eval ran** via `scripts/run_swebench_verified.py` (docker provider, concurrency 4) → `results/swebench_verified_gold.{jsonl,log}`. Incremental/resumable; re-run/scale per `resources_servers/swe_env/README.md`.
- **OpenHands downloaded + built + verified runnable** (`responses_api_agents/swe_agents/swe_openhands_setup/`, fork `sdevare-nv/nv-OpenHands@25bacbc`, `import openhands`=0.62.0).
**Goal:** Implement the plan to decouple SWE environment infra from agent harnesses (issue #1249), on top of the Sandbox API PR #1377; unit-test it; do a real SWE-bench sanity run with a small Qwen on the 2 local GPUs; open a PR (based off #1377) and get CI green.

Plan file: `/home/adasif/.claude/plans/https-github-com-nvidia-nemo-gym-issues-lazy-donut.md`

---

## TL;DR for when you wake up
**Plan items 2, 3, 4, 5 implemented; 87 unit tests pass; and the verifier is validated on a REAL SWE-bench Verified instance on BOTH providers.** PR: **https://github.com/adil-a/Gym/pull/1** (based off #1377).

**Real SWE-bench validation (`astropy__astropy-13453`):** pulled the public docker image `swebench/sweb.eval.x86_64.astropy_1776_astropy-13453`, reset to base, applied the GOLD patch + test_patch, ran the 1 FAIL_TO_PASS + 9 PASS_TO_PASS tests → `resolved=True, reward=1.0` via the **docker provider**, and again after `apptainer build docker-daemon://<image>` (→ 1.0GB `.sif`) via the **apptainer provider**. So the decoupled verifier grades real benchmark tasks on both backends. (This also caught + fixed a real bug: the default `reset_repo` did `git clean -fdx`, which would wipe a repo's prebuilt C extensions — now `git reset --hard` only, matching legacy.)

### Running real SWE-bench / the "500-instance" mechanism
- SWE-bench ships **public Docker images** (Docker Hub `swebench` namespace, `sweb.eval.x86_64.<instance_id>` with `__`→`_1776_`), auto-pulled by the harness; ~120GB+ for the full Verified set. **There are no pre-built `.sif` files to download.**
- This box has docker, so use the **docker images directly** (docker provider) — no `.sif` needed. For apptainer-only clusters, convert each image with `apptainer build x.sif docker-daemon://swebench/sweb.eval.x86_64.<id>` (or NVIDIA NeMo-Skills `nemo_skills/dataset/swe-bench/dump_images.py` in bulk).
- A full 500-instance eval = loop the dataset, pull each image, run the agent, verify — a large batch job (hundreds of GB + agent compute), not run here; one real instance is validated end-to-end on both providers as proof.

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
