# NeMo Gym dogfood: skill-eval checkpoint

> **Superseded by** [`skill-eval-infra-v1.md`](skill-eval-infra-v1.md) (infrastructure + gotchas) and [`skill-eval-findings.md`](skill-eval-findings.md) (skill-specific claims, gated on v7). A PM review correctly flagged that this document conflated the "infra we built" narrative with the "skills findings" narrative, and several of the skill-specific claims rested on contaminated data. The split preserves the infra/gotchas material unchanged and parks the findings section until a clean measurement run is complete.

**Period:** ~2 days · **Owner:** Lawrence Lane (llane@nvidia.com) · **Status:** ongoing

## What we built

A three-server NeMo Gym harness that grades `.claude/skills/*/SKILL.md` content via **paired rollouts** (with-skill vs without-skill system prompt). Wraps an LLM-as-judge that returns per-assertion binary grades with evidence, plus a sandboxed per-session workspace exposing `run_bash` and `read_file` tools to the policy model.

Three servers: `skill_workspace` (tools + sandbox), `skill_judge` (per-assertion grader), `skill_eval_agent` (orchestrator). Scripts on top: `build_skill_eval_jsonl.py` (provenance-stamped input builder), `diff_skill_scoreboards.py` (delta-of-delta with attribution). All in-tree, all Hydra-composed, `ng_run`-bootable.

Ran ~1,500 rollouts across 7 iterations (v1 → v6 + shape probe A/B) on 8 skills × 3 scenarios × n=5.

## What we learned about skills themselves

- **Reward-only scoring is lossy.** Every skill reduced tool calls by **0.8–4.8 calls per rollout** in the with-arm, even skills with flat reward deltas. `gym-data` at Δreward = −0.013 (looked dead) saved 3.27 tool calls per rollout. Skills teach efficiency separately from accuracy.
- **One skill is actively misleading.** `gym-profile` has Δreward = −0.144 AND Δtools = −3.13: the skill makes the model faster *and* worse. Root cause is Diátaxis-shaped — the skill is a how-to, but the failing assertions test conceptual noun recall (explanation-mode). The without-arm reasons from first principles and names the right field; the with-arm follows the recipe and generalizes over it.
- **Only one skill measurement is actually clean.** `gym-run` at Δreward = +0.487 is the only trustworthy delta we have. Every other number was confounded by seeded-artifact contamination (see below).
- **Noise floor at n=5 is ~±0.05 skill-level, ~±0.20 per-cell.** Measured against v4↔v5 zero-edit runs. Judge non-determinism at temperature=0 on the NVIDIA inference API is the dominant source.

## Gotchas we hit that the team might want to know about

**NeMo Gym sharp edges:**
1. `ng_run` runs `python app.py`, not `python -m`. Relative imports (`from .schemas import X`) break; must use absolute imports from project root.
2. Trailing slashes in `policy_base_url` produce double-slash 404s on some providers.
3. Some `/v1/responses` providers return `object: "chat.completion"`; `NeMoGymResponse` validation fails on the literal. Normalize in the model server.
4. OpenAI `FunctionToolParam` requires explicit `strict: False` or validation fails at the model server.
5. Host NeMo Gym `.venv/bin` leaks into subprocess PATH via inherited env; rollouts can see `ng_*` binaries, Ray sockets, HF/MLflow creds. Needed a sandbox env strip.
6. Sandbox PATH strip removes `python` alias on macOS — rollouts silently failed with empty output until we added a workspace-local `python → python3` symlink.
7. Rollout JSONL only persists the *final turn's* token usage. Multi-turn tool loops lose intermediate-turn tokens.
8. Workspace cleanup must be inside a `finally` block — tool failures, judge failures, cancellation all leak tmpdirs otherwise.

**Methodology gotchas (likely generalize):**
9. **Every artifact seeded into a workspace contaminates the control arm.** We measured 100% peek rate on SKILL.md before we fixed that (`ls && cat SKILL.md` was the model's default first tool call). References directory has the same problem and we're still debugging it.
10. **Content-hashing provenance is only as clean as its boundary.** We hash inputs at JSONL-build time. If server code is edited between build and `ng_collect_rollouts`, the hash lies. Need rollout-time re-hashing.
11. **"Same-sha" is a necessary but not sufficient attribution claim.** The model and judge endpoints' non-determinism still moves deltas by up to 0.20 on bit-identical runs.

## What NeMo Gym could absorb upstream

1. **Assertion-grade LLM-as-judge as a reusable base class.** Every benchmark using an LLM judge re-invents this (we did, `code_gen` did, `equivalence_llm_judge` did). Common shape: `(response, tool_calls, assertions[])` → `grades[]`.
2. **Paired-arm pattern as a framework primitive.** A `pairing_key` field + a post-hoc `ng_pair_rollouts` CLI that joins and diffs. Small change, generic.
3. **Framework-level provenance stamping.** Auto-hash any file referenced by Hydra config into `verifier_metadata` at collection time.
4. **Per-turn token aggregation in the agent base class.** Currently downstream consumers have to sum across the tool loop themselves.

## Open questions we'd love input on

- **Multi-judge voting for calibration.** At n=5, judge drift is larger than most real signals. Has anyone looked at ensemble judges?
- **"Doc-eval" as a first-class NeMo Gym category.** Skill-eval is one instance of a broader pattern: "can a reader accomplish JTBD X after consulting doc Y?" The same harness works for any markdown-plus-scenarios. Worth a blessed template under `resources_servers/`?
- **Non-`verify()` resources server base class.** Our `skill_workspace` exposes tools but doesn't grade. Today we subclass `SimpleResourcesServer` and return a dummy `verify()`. Cleaner separation would help.

## Next up

- Fix the references-directory contamination (mirror the SKILL.md fix).
- Split `with_skill` into two independent flags (`with_skill` / `with_references`) for a 4-cell 2×2: blind / docs-only / skill-only / skill+docs.
- Ship multi-axis diff (Δreward, Δtools, Δtokens).
- Build a doc-defect classifier over the rollouts to triage content gaps automatically.
