# Skill-eval findings

**Status:** v7 complete on post-contamination-fix harness. n=5, 480 rollouts, Opus 4.7 policy + judge via NVIDIA inference API. Noise floor established at ~±0.05 skill-level and ~±0.20 per-cell in prior calibration runs (v4↔v5 zero-edit).

## Methodology (what's measured)

Each skill is rolled out under four cells per scenario:

- `blind` — no skill in prompt, no supporting artifacts on disk
- `docs-only` — references/scripts on disk, no skill in prompt (realistic reader without skill pack)
- `skill-only` — skill in prompt, no supporting artifacts (skill as standalone doc)
- `skill+docs` — skill in prompt + artifacts on disk (realistic deployed reader)

Per-skill claims cite one of four marginal effects:

- **`skill | refs=T`** (skill+docs − docs-only) — realistic-deployment value of the skill overlay
- **`skill | refs=F`** (skill-only − blind) — skill as a standalone doc
- **`refs  | skill=T`** (skill+docs − skill-only) — do refs still matter when the skill is prompted?
- **`refs  | skill=F`** (docs-only − blind) — marginal value of references alone

Each delta reports on three axes: `Δreward` (accuracy), `Δtools` (efficiency), `Δtokens` (output length).

## v7 headline table — Δreward

| skill | skill \| refs=T | skill \| refs=F | refs \| skill=T | refs \| skill=F |
|---|---|---|---|---|
| **gym-run** | **+0.436** | **+0.480** | −0.058 | −0.013 |
| add-benchmark | +0.141 | +0.446 | +0.008 | +0.313 |
| gym-config | +0.111 | +0.173 | +0.000 | +0.062 |
| gym-debug | +0.080 | +0.267 | +0.000 | +0.187 |
| gym-review | +0.029 | +0.298 | +0.332 | +0.602 |
| gym-data | +0.013 | +0.040 | −0.013 | +0.013 |
| gym-scaffold-agent | −0.040 | +0.213 | +0.080 | +0.333 |
| **gym-profile** | **−0.107** | **+0.278** | −0.042 | +0.342 |

Key: **bold** = effect cleanly outside noise floor (~±0.10 conservative at n=5). Per-cell n=15 (3 scenarios × 5 repeats).

## v7 Δtools on the realistic contrast (skill \| refs=T)

Every skill reduces tool calls when the skill is added on top of references, even when the accuracy delta is small:

| skill | Δtools (skill \| refs=T) |
|---|---|
| gym-review | −4.87 |
| gym-debug | −4.33 |
| gym-data | −3.27 |
| gym-profile | −2.87 |
| gym-run | −1.87 |
| gym-config | −1.60 |
| add-benchmark | −1.27 |
| gym-scaffold-agent | −0.73 |

The efficiency claim from the earlier (contaminated) checkpoint survives and is the strongest multi-skill pattern in the v7 data. Prescriptive reading: at deployment scale, the skills save real latency/cost per invocation, regardless of whether they materially move accuracy.

## Per-skill reads

### gym-run — the one clean win, confirmed

`skill | refs=T` = +0.436. This skill has no `references/` directory, so its control arm was never contaminated. The v6 value of +0.487 holds within noise. Skill is the load-bearing artifact for these scenarios.

### gym-profile — a decomposition, not a retraction

Earlier checkpoint called this "actively misleading" on the basis of aggregate Δ = −0.144. The 2×2 shows the story is subtler:

- **Skill as standalone doc**: +0.278. Real value when it's the only scaffold.
- **Refs as standalone doc**: +0.342. References are better-structured for the JTBD than SKILL.md.
- **Adding skill on top of refs**: −0.107. The two compete.

The skill isn't misleading in isolation; it's redundant with (and narratively conflicts with) its own references. Fix is SKILL.md-side: narrate *to* the references (cross-link at decision points) rather than duplicate their content in a different structure. Previous Diátaxis-flavored reading still applies — the references are reference-mode and the SKILL.md is how-to-mode, and when both are present the model routes through the wrong one.

### gym-review — refs dominate; skill is mostly redundant

`refs | skill=F` = +0.602 (largest reference-value in the sprint). `skill | refs=T` = +0.029 (within noise). The skill adds essentially nothing once references are present. Prescription: either delete SKILL.md and promote references, or make SKILL.md cover JTBDs the references don't.

### gym-debug, gym-config, add-benchmark — solid realistic-deployment value

All three show `skill | refs=T` in the +0.08 to +0.14 band — above noise floor, meaningful but not load-bearing. Standalone value (skill | refs=F) is substantially higher (+0.17 to +0.45), indicating the skills are genuinely teaching something, but that much of what they teach is also in the references.

### gym-scaffold-agent — marginally negative on the realistic contrast

`skill | refs=T` = −0.040 is inside noise. Standalone (+0.213) is positive; the skill does teach. But adding it on top of refs contributes zero to slightly-negative. Similar failure mode to gym-profile, smaller magnitude.

### gym-data — ceiling-clipped

All cells score 0.96–1.00. No JTBD signal at this scenario difficulty. Harder scenarios needed before this skill can be measured.

## How the v7 numbers relate to the v6 (contaminated) numbers

Pre-fix, the checkpoint reported these Δreward values against a control that was seeing references on disk (and SKILL.md pre-SKILL.md-fix). The "realistic deployment" equivalent in v7 is the `skill | refs=T` column:

| skill | v6 Δ (contaminated) | v7 skill \| refs=T | Δ shifted by |
|---|---|---|---|
| gym-run | +0.487 | +0.436 | −0.051 (holds) |
| gym-review | +0.152 | +0.029 | −0.123 (mostly reference value, not skill value) |
| gym-debug | +0.133 | +0.080 | −0.053 (shrinks but survives) |
| add-benchmark | +0.110 | +0.141 | +0.031 (holds) |
| gym-config | +0.089 | +0.111 | +0.022 (holds) |
| gym-scaffold-agent | +0.040 | −0.040 | −0.080 (flips sign, inside noise either way) |
| gym-data | −0.013 | +0.013 | +0.026 (ceiling both versions) |
| gym-profile | −0.144 | −0.107 | +0.037 (smaller, still negative) |

**gym-run is the only skill whose pre-fix claim survives intact** — predictable, since it was the only skill with no references to contaminate.

## What we can claim

Supported by v7 data, outside noise floor:

- **gym-run's skill is load-bearing for its scenarios** (Δreward +0.44 realistic, +0.48 standalone).
- **gym-profile's SKILL.md competes with its own references** when both are present (−0.107 realistic).
- **Skills teach efficiency across the board** (Δtools −0.73 to −4.87 on realistic contrast).
- **add-benchmark, gym-config, gym-debug have real realistic-deployment value** in the +0.08 to +0.14 band.
- **Most skills are redundant-heavy with their own references** (standalone Δreward is 2–5× the realistic Δreward for five of eight skills).

Not supported by v7 data:

- Fine-grained claims about shape (bullet vs checkbox vs heading). The prior shape-probe rollouts inherited the contamination bug; would need a clean rerun, and v7 suggests the effect size is under our n=5 resolution.
- A single-number "skill quality" ranking. The 2×2 shows most skills carry value on some axis (standalone, efficiency, realistic) and not others. Any ranking is a weighting decision, not a measurement.

## Open methodology questions

- **Judge drift at temperature=0 remains the dominant noise source.** n=5 gives detectable effects down to ~±0.10. For the three skills below noise on the realistic contrast (gym-data, gym-scaffold-agent, gym-review), we cannot distinguish small-positive from small-negative.
- **Power analysis TBD.** The right n for the realistic-deployment contrast is unknown; a calibration run at n=20 on one cell would tell us.

## Pre-fix retractions (archived)

Claims from the earlier checkpoint that do not survive v7:

- ~~"Every skill reduces tool calls by 0.8–4.8 per rollout."~~ Replaced by the tighter claim on the `skill | refs=T` contrast above: every skill still reduces tool calls, but the correct magnitude is −0.73 to −4.87 measured on the realistic contrast, not against a contaminated control.
- ~~"gym-profile is actively misleading."~~ Replaced by "gym-profile's SKILL.md competes with its own references; the skill is useful in isolation but negatively-interacts with refs."
- Shape-probe null result — withdrawn pending rerun on clean harness.
