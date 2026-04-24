# Skill-eval findings

**Status:** gated on v7 (post-contamination-fix) rollout.

This document captures claims about which specific skills help, hurt, or are neutral. Findings land here only after:

1. The underlying measurement is taken on a harness with the known-contamination bugs fixed (SKILL.md peek, references peek — fixed in `lbliii/prague-v2` commit `8bec001e`).
2. The delta is outside the empirical noise floor for the configuration used (n=5 gives ~±0.05 skill-level, ~±0.20 per-cell; a claim at ±0.10 needs n≥10).
3. The provenance diff confirms that the only variable changed is the one the claim attributes to.

**Findings below this line are only valid if all three gates are green.**

---

## Methodology (what's measured)

Each skill is rolled out under four cells per scenario:

- `blind` — no skill in prompt, no supporting artifacts on disk
- `docs-only` — references/scripts on disk, no skill in prompt (realistic reader without skill pack)
- `skill-only` — skill in prompt, no supporting artifacts (skill as standalone doc)
- `skill+docs` — skill in prompt + artifacts on disk (realistic deployed reader)

Per-skill claims cite one of four marginal effects:

- **`skill | refs=T`** (skill+docs − docs-only) — "does adding the skill overlay help a reader who already has the repo?" This is the realistic-deployment claim.
- **`skill | refs=F`** (skill-only − blind) — "is the skill a good standalone doc?"
- **`refs | skill=T`** (skill+docs − skill-only) — "do references still matter when the skill is prompted?"
- **`refs | skill=F`** (docs-only − blind) — "marginal value of references alone"

Each delta is reported on three axes: `Δreward` (accuracy), `Δtools` (efficiency), `Δtokens` (output length).

## v7 results

*Pending — v7 rollout in progress. Once complete, this section will list per-skill marginal effects with confidence qualifiers.*

| skill | `skill \| refs=T` Δreward | `skill \| refs=F` Δreward | Δtools (refs=T) | notes |
|---|---|---|---|---|
| gym-run | _pending_ | _pending_ | _pending_ | |
| gym-review | _pending_ | _pending_ | _pending_ | |
| gym-debug | _pending_ | _pending_ | _pending_ | |
| gym-config | _pending_ | _pending_ | _pending_ | |
| gym-profile | _pending_ | _pending_ | _pending_ | |
| gym-scaffold-agent | _pending_ | _pending_ | _pending_ | |
| gym-data | _pending_ | _pending_ | _pending_ | ceiling-clipped in v6; expected to remain so |
| add-benchmark | _pending_ | _pending_ | _pending_ | |

## Power analysis

*Pending — planned calibration run at n=20 on one skill+cell to quantify the detectable effect size at n=5 vs n=10 vs n=20.*

## Pre-fix results (archived, not defensible)

The following were reported in an earlier checkpoint against measurements with known contamination (SKILL.md on disk in the control arm, references/*.md on disk in both arms). Retained here for comparison once the clean v7 results are available; they should not be cited as findings:

| skill | Δreward (v6, contaminated) |
|---|---|
| gym-run | +0.487 |
| gym-review | +0.152 |
| gym-debug | +0.133 |
| add-benchmark | +0.110 |
| gym-config | +0.089 |
| gym-scaffold-agent | +0.040 |
| gym-data | −0.013 |
| gym-profile | −0.144 |

Of these, only `gym-run` had no `references/` directory and therefore a partially-uncontaminated control. Every other row has a contamination asterisk.

**Claims retracted pending v7:**

- "Every skill reduces tool calls by 0.8–4.8 per rollout." Measurement is against a contaminated control where the without-arm was `cat SKILL.md`-ing on turn 1. Real Δtools is unknown until v7.
- "gym-profile is actively misleading." The aggregate delta (−0.144) was inside per-cell noise; the per-scenario evidence (sc3 with=0.80 / without=1.00 zero-variance) is a stronger claim about *one scenario*, not the skill. Revisit post-fix.
- "Shape of the 'Before you answer' checklist matters" — null result on clean baseline, but that baseline was itself contaminated. Claim is withdrawn pending a post-fix rerun.

The only pre-fix claim that survives: **when a skill has no `references/` directory**, its with-vs-without comparison is partially uncontaminated. `gym-run` is the single data point in that category. v7 will tell us whether its Δreward = +0.487 holds on a fully clean baseline.
