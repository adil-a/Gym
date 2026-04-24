(skill-eval-scoreboard)=

# Evaluating Agent Skills, Part 2: Running the Scoreboard

Part 1 ({ref}`skill-eval-harness`) built the infrastructure. This part covers what you do with it: write scenarios, generate JSONL, run the scoreboard, and read the result with the right amount of skepticism.

By the end, you will:

- Understand **with-skill vs without-skill delta** — what it measures and why it beats absolute scoring.
- Write a good `evals.json` and know how assertions fail you.
- Generate input JSONL, run the full scoreboard, and compute per-skill deltas.
- Know when to trust a delta and when to ignore it.

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

< Back to Building Environments
:::

---

## Prerequisites

- {ref}`skill-eval-harness` — Part 1, the infrastructure this part runs on
- `ng_run` is up and `ng_status` shows all four servers healthy

---

## The methodology: a 2×2 grid, not a single A/B

Absolute reward scores lie. If a skill scores 0.90 across 3 scenarios, is it good? Depends entirely on what the model does *without* the skill. A 0.90 skill on top of a 0.90 baseline is noise; a 0.90 skill on top of a 0.40 baseline is +0.50 of real lift.

But "without the skill" hides a decision. A skill in a repo ships two kinds of material: the `SKILL.md` that might go into the system prompt, and `references/` and `scripts/` that live on disk and get copied into the workspace. A reader who "doesn't have the skill pack" might have the repo checked out or might not. Those are two different controls, and they give different answers. We learned this the painful way — our first five iterations measured a confound (see the "contamination" sidebar below).

The harness runs every scenario across **four cells**, toggled by two independent flags:

|  | `with_references=False` | `with_references=True` |
|---|---|---|
| `with_skill=False` | **blind** — model priors only | **docs-only** — realistic reader without the skill overlay |
| `with_skill=True`  | **skill-only** — SKILL.md in prompt, nothing on disk | **skill+docs** — realistic reader with the skill pack |

Four named marginal effects come out of this:

- **`skill | refs=T`** (skill+docs − docs-only) — realistic-deployment value of the skill overlay
- **`skill | refs=F`** (skill-only − blind) — skill as a standalone doc
- **`refs  | skill=T`** (skill+docs − skill-only) — do references still matter when the skill is prompted?
- **`refs  | skill=F`** (docs-only − blind) — marginal value of references alone

The "realistic deployment" effect is the one most users care about: does installing the skill overlay help a reader who already has the repo? But the other three are diagnostic — they tell you whether a skill is doing unique work, competing with its own references, or just redundant.

:::{admonition} Contamination sidebar — why the 2×2 exists
:class: warning

Earlier versions of this harness copied `SKILL.md` into the workspace unconditionally, then gated the system-prompt prepend on a single `with_skill` flag. In the "without" arm, the model's first tool call was `ls && cat SKILL.md`: it read the skill off disk 100% of the time. After that was fixed, `references/` was still being seeded in both arms — and for skills whose references happened to be well-structured reference docs (e.g. gym-profile's `metrics-guide.md`), the "without skill" arm was still seeing the skill-author's prose.

The 2×2 exists because "what the reader has" is a decision the harness has to make explicit. Gating `with_references` separately from `with_skill` lets the control arm be a genuinely cold start when you want that contrast, and a realistic-reader baseline when you want the other.
:::

---

## Step 1: Write `evals.json`

Each skill has a `.claude/skills/<skill-name>/evals/evals.json` with 3–5 scenarios:

```json
{
  "evals": [
    {
      "id": 1,
      "prompt": "Review the benchmark at evals/files/sample_benchmark/ and produce a merge-readiness report.",
      "files": ["evals/files/sample_benchmark/configs/foo.yaml",
                "evals/files/sample_benchmark/app.py"],
      "assertions": [
        "The agent runs scripts/review.py against the files",
        "verified-true WARN is reported for the YAML config",
        "The report mentions that verified should be false for new unbaselined servers"
      ],
      "expected_output": null
    }
  ]
}
```

### What a good assertion looks like

- **Specific** — "verified-true WARN is reported" > "the agent mentions verified status".
- **Testable from the transcript** — the judge sees the model's output + tool calls, nothing else. Don't assert on filesystem state.
- **Function-oriented, not phrasing-oriented.** Assertions that require a literal string are fragile.

### Cautionary tale

One of our assertions read *"Handoff to gym-profile mentioned for analyzing results after rollout collection"*. The model correctly recommended `ng_reward_profile` — the right *function* — but didn't name the skill `gym-profile` verbatim. The judge correctly flagged it as unsatisfied. The grade was right; the assertion was over-literal.

**Rewrite before running**: *"Mentions using ng_reward_profile (or a handoff to the gym-profile skill) to analyze rollout results"*.

### What about `expected_output`?

Optional. If you have a gold-standard answer, put it here and the judge will see it. For most skills, leave it `null` — the assertions are the spec.

---

## Step 2: Generate the input JSONL

`scripts/build_skill_eval_jsonl.py` walks `.claude/skills/*/evals/evals.json` and emits **four records per scenario** — one for each cell of the 2×2:

```bash
python scripts/build_skill_eval_jsonl.py \
    --skills-dir .claude/skills \
    --output responses_api_agents/skill_eval_agent/data/example.jsonl
# --cells=blind,skill+docs to restrict to a subset
```

Each record:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "<scenario prompt>"}]
  },
  "verifier_metadata": {
    "skill_path": "/abs/path/to/skill",
    "skill_name": "my-skill",
    "skill_md_sha": "cd125b6470be",
    "evals_sha": "ef847045784b",
    "fixtures_sha": "c03a0ca4ddda",
    "judge_prompt_sha": "665408560ff4",
    "harness_version": "e28ce8330e99",
    "scenario_id": 1,
    "files": ["evals/files/..."],
    "cell": "skill+docs",
    "with_skill": true,
    "with_references": true,
    "with_scripts": true,
    "skill_md": "<contents of SKILL.md>",
    "assertions": [...],
    "expected_output": null
  }
}
```

`with_skill` / `with_references` are the two independent flags whose cross product defines the 2×2. The `cell` label is emitted for convenience so downstream tooling can bucket on a single field. Five content-hash provenance fields (`*_sha`, `harness_version`) ride through to the output for attribution — see Step 7.

---

## Step 3: Run the scoreboard

Start with `num_repeats=1` to sanity-check end-to-end; then bump to 5 for a scoreboard you can interpret.

```bash
ng_collect_rollouts \
    +agent_name=skill_eval_agent \
    +input_jsonl_fpath=responses_api_agents/skill_eval_agent/data/example.jsonl \
    +output_jsonl_fpath=results/scoreboard.jsonl \
    +num_repeats=5 \
    +num_samples_in_parallel=6 \
    "+responses_create_params={max_output_tokens: 8192}"
```

`num_samples_in_parallel` is bounded by your endpoint's rate limit more than your local machine. 6-way parallel on the NVIDIA inference-api produced zero flakes in our runs.

The output JSONL contains one line per rollout with `reward`, per-assertion `grades[]`, and the full `verifier_metadata` preserved.

---

## Step 4: Read the deltas

`ng_collect_rollouts` prints a single `mean/reward` across everything — that number mixes with-skill and without-skill rollouts and is **not** what you want. Bucket by skill and by the `with_skill` flag:

Use the diff tool:

```bash
python scripts/diff_skill_scoreboards.py results/scoreboard.jsonl
```

This auto-detects whether the JSONL is 4-cell (new) or 2-arm (legacy) and renders the appropriate scoreboard — for a 4-cell file, a per-skill block with all four cells plus the four named marginal effects. It also reports on three axes in every row: **Δreward** (accuracy), **Δtools** (tool-call count — efficiency), and **Δtokens** (output length).

A real scoreboard from v7 (n=5 per cell, 480 rollouts total, post-contamination-fix):

| skill | `skill \| refs=T` Δreward | `skill \| refs=F` Δreward | `refs \| skill=F` Δreward | note |
|---|---|---|---|---|
| **gym-run** | **+0.436** | **+0.480** | −0.013 | no references/ dir; the only skill whose numbers never had a contamination asterisk |
| add-benchmark | +0.141 | +0.446 | +0.313 | skill+refs both carry signal; skill adds real value on top |
| gym-config | +0.111 | +0.173 | +0.062 | skill does most of the work; refs add little |
| gym-debug | +0.080 | +0.267 | +0.187 | skill strong standalone; refs substantial; skill+refs near ceiling |
| gym-review | +0.029 | +0.298 | +0.602 | refs dominate; skill mostly redundant once refs present |
| gym-data | +0.013 | +0.040 | +0.013 | ceiling-clipped every cell; can't measure |
| gym-scaffold-agent | −0.040 | +0.213 | +0.333 | skill useful standalone; refs alone do more |
| **gym-profile** | **−0.107** | **+0.278** | +0.342 | skill competes with its own references when both are present |

Bold = effect outside the ~±0.10 noise floor at n=5.

### How to read this table

**The "realistic deployment" column is `skill | refs=T`.** That's the question "does adding the skill overlay help a reader who already has the repo?" Everything else is a diagnostic breakdown of *why* the realistic effect is what it is:

- **Big `skill | refs=F`, small `skill | refs=T`** (e.g. gym-review at +0.298 vs +0.029) → the skill is mostly a compressed restatement of its references. If the reader has docs, they don't need the skill. **Prescription:** either shrink the skill to only cover what docs don't, or promote references.
- **Big positive on both `skill | refs=T` AND `skill | refs=F`** (gym-run, add-benchmark) → the skill adds real unique value regardless of whether references are present. **Keep.**
- **Negative `skill | refs=T`, positive `skill | refs=F`** (gym-profile, gym-scaffold-agent) → the skill is useful in isolation but *competes* with its own references when both are loaded. The two are narrating the same content in conflicting ways. **Prescription:** rewrite SKILL.md to cross-reference the references rather than duplicate them.
- **Everything ~0** (gym-data) → ceiling-clipped; either the JTBD is solvable without any scaffold, or the scenarios are too easy. Harder scenarios needed.

### The tool-call axis is a second signal

Every skill in v7 reduces tool calls on the realistic contrast — even skills with tiny or negative Δreward:

| skill | Δtools (skill \| refs=T) | Δreward (skill \| refs=T) |
|---|---|---|
| gym-review | **−4.87** | +0.029 |
| gym-debug | **−4.33** | +0.080 |
| gym-data | **−3.27** | +0.013 |
| gym-profile | **−2.87** | −0.107 |
| gym-run | −1.87 | +0.436 |
| gym-config | −1.60 | +0.111 |
| add-benchmark | −1.27 | +0.141 |
| gym-scaffold-agent | −0.73 | −0.040 |

gym-review is the clearest example: no meaningful accuracy effect, but 4.87 fewer tool calls per rollout with the skill. At deployment scale (many rollouts × production latency + tool cost), that's material value the reward-only scoreboard erases. **Always read both axes. "Flat reward, fewer tools" is a real skill contribution.**

---

## Pitfalls — receipts from our own runs

### Pitfall 1: Ceiling effects

Example from v7: `gym-data` scored 0.96–1.00 across every cell of the 2×2. That is **not** "this skill has no effect" — it's "this scenario is solvable without the skill, so we can't measure." Don't shrug at +0.000; go write harder scenarios.

**Rule of thumb:** if `docs-only` ≥ 0.95, treat both `skill | refs=T` and `skill | refs=F` as *inconclusive*. Add one adversarial scenario and rerun.

### Pitfall 2: Low-n noise and judge drift

At n=1 per bucket, a single wrong grade moves the delta by `1/num_assertions` (often 0.15–0.25). At n=5, judge non-determinism at temperature=0 on the NVIDIA inference API moves deltas by up to ~0.05 at the skill level and up to ~0.20 at the per-cell level even with bit-identical inputs.

**Rule of thumb:**

- n=1 gives you rank ordering, not magnitudes.
- n=5 gives you detectable effects ≥ ~0.10 per cell, ≥ ~0.05 skill-level.
- Bump to n=10–20 on a cell if you need to resolve a smaller effect.

Don't claim a skill "helps by 8%" at n=5 on a single cell. Do claim it on consistent evidence across cells with effects outside the noise floor.

### Pitfall 3: Negative deltas on the realistic contrast are diagnostic, not proof of misleading content

A negative `skill | refs=T` doesn't mean the skill is bad in isolation. It means the skill competes poorly *with its own references on disk*. Check `skill | refs=F` before concluding anything:

- **Negative on both** → the skill is genuinely misleading. Read the failing `grades[].evidence`, find the specific noun being missed.
- **Negative on `refs=T`, positive on `refs=F`** → the skill is a good standalone doc but redundant-with-conflicting-framing alongside its references. Rewrite SKILL.md to cross-reference the references rather than duplicate them.

gym-profile in v7 is an example of the second pattern: `skill | refs=F` = +0.278, `skill | refs=T` = −0.107.

### Pitfall 4: Over-literal assertions

See the `gym-profile` example in Step 1. If an assertion-level grade looks wrong, check the assertion *before* blaming the judge. Our spot-check found 6/7 judge decisions defensible; the one "miss" was assertion phrasing.

---

## Step 5: Break out by scenario

Skill-level means average across all scenarios for that skill, which hides exactly the information you need when chasing a delta. A skill with three scenarios and deltas `[+0.20, +0.00, −0.20]` scores 0.00 at the skill level — indistinguishable from a boringly flat skill. When you inspect it, you'll find one great scenario, one ceiling-clipped scenario, and one adversarial scenario where the skill actively misleads the model.

Bucket by `(skill, scenario_id, with_skill)`:

```python
import json
from collections import defaultdict

by_cell = defaultdict(list)
for line in open("results/scoreboard.jsonl"):
    r = json.loads(line)
    md = r.get("verifier_metadata", {})
    key = (md["skill_name"], md["scenario_id"], "with" if md.get("with_skill") else "without")
    by_cell[key].append(r["reward"])

for key in sorted(by_cell):
    rewards = by_cell[key]
    skill, sid, arm = key
    print(f"{skill:22s} sc{sid} {arm:7s}  mean={sum(rewards)/len(rewards):.2f}  n={len(rewards)}")
```

This is also what turns a puzzling skill-level delta into a debugging lead. When `gym-profile` moved from Δ = +0.038 (v2) to Δ = −0.053 (v3) — a change of −0.091 on the *same* SKILL.md — the skill-level number was a dead end. Breaking out by scenario pointed straight at scenario 2's `with_skill` arm: 0.84 in v2, 0.56 in v3. Every other cell in the breakdown was flat. That alone narrowed the root cause to one rollout, which we then opened and found spinning on `python: command not found` — a harness-level side effect we fix in Step 7.

**Rule of thumb:** if the delta-of-deltas between two runs is larger than the noise floor, go straight to the scenario breakdown before doing anything else. The skill-level number is a pointer; the scenario breakdown is the address.

---

## Step 6: Spot-check the judge

Before trusting the scoreboard, verify the judge is actually measuring what you think. Sample a few partial-credit rollouts and read the evidence:

```python
import json, random

random.seed(7)
rows = [json.loads(l) for l in open("results/scoreboard.jsonl")]
partial = [r for r in rows if 0 < r["reward"] < 1.0]
for r in random.sample(partial, k=min(5, len(partial))):
    md = r["verifier_metadata"]
    print(f"\n{md['skill_name']} sc{md['scenario_id']} with={md['with_skill']} r={r['reward']:.2f}")
    for a, g in zip(md["assertions"], r["grades"]):
        mark = "✓" if g["satisfied"] else "✗"
        print(f"  {mark} {a}")
        print(f"     → {g['evidence'][:150]}")
```

What you're looking for:

- **Evidence strings cite real text** from the response or tool calls — not fabricated.
- **Consistent grading on duplicate scenarios** (`with_skill=True` and `=False` should grade the same assertion the same way when the response is substantively similar).
- **Misses should be visibly absent** in the transcript, not judge hallucinations.

Our 7-sample audit: 6/7 clean, 1/7 was assertion phrasing, not judge error. Once you see that pattern, you can trust the deltas.

---

## Step 7: Iterate — version the inputs, rerun, diff

One scoreboard tells you *where* a skill is hurting. It does not tell you whether an edit *fixed* anything. For that you need two runs side-by-side with an unambiguous answer to "is what I ran in v2 actually different from v1 — and different in *which* way?"

### What provenance covers (and what it still doesn't)

`scripts/build_skill_eval_jsonl.py` embeds five content hashes in every record's `verifier_metadata`. They ride through the full pipeline — build → agent → judge → output JSONL — so every scoreboard tells you exactly which inputs it was measuring:

| field | hashes | changes when… | attribution note |
|---|---|---|---|
| `skill_md_sha` | `SKILL.md` | skill prose edited | the only field a *skill* change moves |
| `evals_sha` | `evals/evals.json` | scenarios or assertions edited | assertion-phrasing debt shows up here |
| `fixtures_sha` | listed fixtures for this scenario | a fixture is added/renamed/edited | scenario-scoped, not skill-scoped |
| `judge_prompt_sha` | `skill_judge/prompt_templates/skill_judge.txt` | judge prompt template edited | a stack-wide change; moves every skill |
| `harness_version` | concatenated bytes of `skill_workspace/app.py` + `skill_judge/app.py` + `skill_eval_agent/app.py` | any server code edited | same: stack-wide, moves every skill |

No version bumps to maintain, no drift: if you edit a file that matters, its hash changes; if you don't, it doesn't.

What's still *not* hashed: the policy model version, the judge model version, model temperature, and anything those endpoints do non-deterministically (Opus at `temperature=0` is not bitwise-deterministic on our inference API). When every hash matches and deltas still move, that is your noise floor plus judge drift — both real, neither your fault.

### The diff tool

`scripts/diff_skill_scoreboards.py` runs in two modes:

```bash
# single-file: per-skill scoreboard with all 5 provenance columns
python scripts/diff_skill_scoreboards.py results/v1/rollouts.jsonl

# two-file: v1 vs v2 delta-of-deltas, with per-field provenance diff
python scripts/diff_skill_scoreboards.py \
    results/v1/rollouts.jsonl --v2 results/v2/rollouts.jsonl
```

The `provenance diff` column in two-file mode tells you which inputs actually changed:

| tag | meaning |
|---|---|
| `—` + `same-all` | every hash matches. Any movement is noise / judge drift. |
| `—` + `partial(N/5)` | only N of 5 fields known. Can't distinguish "unchanged" from "untracked". |
| `md` | only SKILL.md changed. Read `with_skill` to see effect. |
| `evals` | only scenarios/assertions changed. `with_skill` AND `without_skill` can both move — same prompt, different judge target. |
| `harness` or `judge` | stack-wide change. Expect correlated movement across every skill's `without_skill` column. |
| `md+evals` | skill body AND scenarios edited in the same run. **Stop.** You cannot attribute the delta cleanly — rerun with one edit at a time. |

**Attribution rule:** a delta-of-delta is only attributable to what the provenance diff points at. `same-all` plus a big delta change is a noise or drift finding, not a "the skill got worse" finding. Don't report it as the latter.

### Worked example: contamination fix, measured

The most instructive run we've produced is v6 → v7: the v7 harness fixed a contamination bug in which the control arm was seeing the skill's `references/` on disk (see the contamination sidebar in the methodology section). Nothing else changed — no skill edits, no assertion edits, no fixture edits. Just the gating flag.

The diff tool correctly flags the change as `harness`-attributable, and the movements fall into two camps:

| skill | v6 "Δreward" (contaminated) | v7 `skill \| refs=T` (clean) | interpretation |
|---|---|---|---|
| gym-run | +0.487 | +0.436 | holds — no references/ dir, so v6 was already clean |
| gym-review | +0.152 | +0.029 | mostly reference value, not skill value |
| gym-debug | +0.133 | +0.080 | shrinks but survives |
| add-benchmark | +0.110 | +0.141 | holds |
| gym-config | +0.089 | +0.111 | holds |
| gym-scaffold-agent | +0.040 | −0.040 | was within noise either direction |
| gym-data | −0.013 | +0.013 | ceiling-clipped both iterations |
| gym-profile | −0.144 | −0.107 | still negative, but smaller and now decomposable |

**Takeaways:**

1. **gym-run's number survived.** Predictable in hindsight: it's the only skill with no `references/` directory, so its control arm was never contaminated. The pre-fix number measured the right thing by accident.
2. **gym-review's apparent value was mostly its references' value.** The v6 read was "this skill adds +0.152 on top of baseline." The v7 read is "the skill adds +0.029 on top of a reader who already has the repo." Same skill, two very different product claims — and only one is defensible.
3. **Attribution requires both the provenance diff AND a correct measurement design.** The provenance tool can tell you "harness changed"; it can't tell you whether that change fixed a measurement bug or created a new one. You still need to think about what the control arm represents.

### Judge drift is a real noise source — document what n buys you

Separately, we ran two "zero-edit" reruns (v4 and v5 — identical inputs, identical harness, just re-collected). The diff tool correctly reported `same-all` across every skill, but deltas still moved by up to 0.053 at the skill level and 0.20 at the per-cell level. That's judge non-determinism: Opus at temperature=0 on the NVIDIA inference API is not bitwise-deterministic, and at n=5 the stochasticity dominates subtle effects.

**The practical implication:** at n=5, claims below ~±0.10 on a single cell are not distinguishable from noise. If you need smaller effects reliably, bump `num_repeats` — every doubling roughly halves the per-cell stderr. Budget accordingly.

Provenance is a necessary prerequisite for attribution, but it isn't sufficient. `same-all` is a cache-hit check, not a p-value.

### The iteration loop

```
1. Pick the worst (or most interesting) skill from the scoreboard.
2. Break it out by scenario (Step 5) to localize which cell moved.
3. Read the partial-credit rollouts' grades[].evidence to find the failure mode.
4. Edit ONE input — SKILL.md, evals.json, a fixture, or harness code.
5. Regenerate the input JSONL (every relevant hash updates automatically).
6. Rerun ng_collect_rollouts into a fresh output path.
7. Diff v2 vs v1 with diff_skill_scoreboards.py.
8. If the provenance diff points at the thing you changed AND the delta
   change is larger than the noise floor, ship it.
```

The "one input at a time" rule is load-bearing. We once edited both a SKILL.md and the sandbox harness in one run, and had to burn a third run to tell them apart.

---

## What's next

Follow-on investments that are already paying off or that the receipts point at:

- **Harder scenarios for ceiling-clipped skills** — until `docs-only` drops below ~0.90, you can't measure improvement. `gym-data`, `gym-review`, and some cells of `gym-debug` are currently stuck here.
- **Rewrite SKILL.md for skills where `skill | refs=T` < `refs | skill=F`** — those skills are competing with their own references. The fix is to narrate *to* the references, not duplicate them (see the Diátaxis-flavored reading in `skill-eval-findings.md`).
- **Calibration run at n=20 on one cell** — to pin down the real detectable effect size and answer the power-analysis question honestly.
- **Doc-defect classifier** — `scripts/diagnose_doc_defects.py` (planned) will automate the per-assertion triage: content-present-but-not-recalled (isolated info), content-missing (info gap), actively-misleading, etc.

For the actual per-skill findings from v7 and the retraction log for earlier claims, see [`skill-eval-findings.md`](skill-eval-findings.md). For a sharable infra/gotchas artifact, see [`skill-eval-infra-v1.md`](skill-eval-infra-v1.md).

:::{button-ref} skill-eval-harness
:color: secondary
:outline:
:ref-type: doc

← Back to Part 1: Build the Harness
:::
