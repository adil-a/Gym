# Skill-eval harness v1: infrastructure and sharp edges

**Period:** ~2 days · **Owner:** Lawrence Lane (llane@nvidia.com) · **Status:** infra shipped; measurements in-flight

This is the infrastructure-and-methodology artifact from a short NeMo Gym dogfood sprint. It covers **what we built and the framework gotchas we hit**, deliberately separated from claims about which specific skills help. Skill-specific findings live in `skill-eval-findings.md`, which is gated on a clean post-contamination-fix measurement run.

## What we built

A three-server NeMo Gym environment that grades `.claude/skills/*/SKILL.md` content via paired rollouts. All in-tree, Hydra-composed, `ng_run`-bootable.

- `resources_servers/skill_workspace` — per-session sandbox tmpdir. Exposes `run_bash` and `read_file` tools. Seeds scenario fixtures and, per `with_references` / `with_scripts` flags, the skill's supporting artifacts.
- `resources_servers/skill_judge` — LLM-as-judge returning per-assertion binary grades with evidence. Reward = fraction of assertions satisfied.
- `responses_api_agents/skill_eval_agent` — orchestrator. Seeds the workspace, optionally prepends SKILL.md as a system message per `with_skill`, runs a model↔tool loop, forwards the transcript to the judge.

Tooling:

- `scripts/build_skill_eval_jsonl.py` — emits a 4-cell 2×2 per scenario with five content-hash provenance fields (`skill_md_sha`, `evals_sha`, `fixtures_sha`, `judge_prompt_sha`, `harness_version`).
- `scripts/diff_skill_scoreboards.py` — auto-detects 2×2 vs legacy input; renders multi-axis deltas (Δreward, Δtools, Δtokens) with per-field provenance attribution.
- `scripts/diagnose_doc_defects.py` — planned (Phase 3; not yet built).

Scale: ~1,500 rollouts across six iterations (v1–v6) on 8 skills × 3 scenarios. Calibration runs (v4↔v5, zero-edit reruns) established a skill-level noise floor of ~±0.05 and a per-cell noise floor of ~±0.20 at n=5.

## The 2×2 cells

Each scenario is now rolled out in four cells, toggled by two independent flags:

|  | `with_references=False` | `with_references=True` |
|---|---|---|
| `with_skill=False` | **blind** — no skill in prompt, no docs on disk (model priors only) | **docs-only** — realistic reader without the skill pack |
| `with_skill=True`  | **skill-only** — skill prompted, no supporting artifacts | **skill+docs** — realistic reader with the skill pack |

The diff tool surfaces four named marginal effects:

- `skill | refs=T` — skill+docs − docs-only (realistic marginal value of the skill)
- `skill | refs=F` — skill-only − blind (skill as standalone doc)
- `refs  | skill=T` — skill+docs − skill-only (do docs still matter when skill is prompted?)
- `refs  | skill=F` — docs-only − blind (marginal value of references alone)

## NeMo Gym sharp edges we hit

These are small, reproducible, and cheap fixes. The most concrete output of the dogfood.

1. `ng_run` runs `python app.py`, not `python -m`. Relative imports (`from .schemas import X`) break; must use absolute imports from project root.
2. Trailing slashes in `policy_base_url` produce double-slash 404s on some providers.
3. Some `/v1/responses` providers return `object: "chat.completion"`; `NeMoGymResponse` validation fails on the literal. Normalize in the model server.
4. OpenAI `FunctionToolParam` requires explicit `strict: False` or validation fails at the model server.
5. Host NeMo Gym `.venv/bin` leaks into subprocess PATH via inherited env; rollouts can see `ng_*` binaries, Ray sockets, HF/MLflow creds. Fixed locally with a sandbox env strip.
6. Sandbox PATH strip removes `python` alias on macOS — rollouts silently failed with empty output until we added a workspace-local `python → python3` symlink.
7. Rollout JSONL only persists the *final turn's* token usage. Multi-turn tool loops lose intermediate-turn tokens.
8. Workspace cleanup must be inside a `finally` block — tool failures, judge failures, cancellation all leak tmpdirs otherwise.

## Methodology gotchas (likely generalize)

9. **Every artifact seeded into a workspace contaminates the control arm.** We measured 100% peek rate on SKILL.md before fixing it (`ls && cat SKILL.md` was the model's default first tool call). Then discovered `references/*.md` had the same problem — 100% of `without_skill` rollouts for `gym-profile` were reading `references/metrics-guide.md`, which contains every noun the assertions test for. Fix landed via the 2×2 split.
10. **Content-hash provenance is only as clean as its boundary.** We hash inputs at JSONL-build time. If server code is edited between build and `ng_collect_rollouts`, the hash lies. Known limitation; rollout-time re-hashing queued.
11. **"Same-sha" is necessary but not sufficient for attribution.** Model and judge endpoints' non-determinism moves deltas by up to 0.20 per cell at n=5 on bit-identical runs. Measurement trustworthiness is currently dominated by judge drift, not by input variance.

## Upstream candidates (two, with existing receipts)

1. **Assertion-grade LLM-as-judge base class.** Three existing implementations in-tree today: `skill_judge`, `code_gen`, `equivalence_llm_judge`. Common shape is `(response, tool_calls, assertions[]) → grades[]`. A shared base class saves the next team re-inventing it.
2. **Per-turn token aggregation in the agent base class.** Today only the final response's `usage` reaches the output JSONL. Every multi-turn agent eventually needs this; we hit it while trying to measure cost impact of skills.

Two other candidates (paired-arm as a framework primitive; framework-level provenance stamping) are real ideas but lack a second concrete customer yet. Keeping them as local tools until demand shows up.

## Not yet validated upstream asks (parked)

- **"Doc-eval as first-class category"** — premature. Methodology is still stabilizing and we've only tested on one doc type (SKILL.md). Revisit after at least one non-skill doc set has been evaluated.
- **Non-`verify()` resources server base class** — nice-to-have but not a blocker. Our workspace subclasses `SimpleResourcesServer` and returns a dummy `verify()`; it works.

## Open measurement questions

- **Judge drift is the dominant noise source.** Multi-judge voting, higher n, different judge models, stronger rubric — each has a cost. We haven't run the experiment yet.
- **Power analysis.** What n gets us below judge drift for a target effect size? Should lead the next checkpoint.

## What's shipped in this iteration

Branch `lbliii/prague-v2`, commits `b072cfb1..825d9ac7`:

- Three-server harness + scripts + tutorials (base)
- 2×2 split: `with_references` / `with_scripts` flags on workspace; agent forwards; builder emits 4 cells; 99 tests pass
- Multi-axis diff tool: auto-detects 2×2 vs legacy; reports Δreward/Δtools/Δtokens

## What's not yet shipped

- Rollout-time provenance (`/version` endpoints, runtime hash embedded in output)
- Per-turn token aggregation in the agent loop
- Wall-clock per rollout
- `scripts/diagnose_doc_defects.py` (Phase 3 classifier)
- A completed post-fix measurement run (v7 — in flight)
