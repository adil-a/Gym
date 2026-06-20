# Brief — Independent Assessment: reproduce SWE-bench Verified "golden" on GB200/ARM (oci-hsg)

## Your task (please read first)

There is a **working** SWE-bench Verified **golden** run on an x86 cluster (**cw-dfw**) that scores
**492/500**. We are trying to reproduce that *same* run on an **ARM (GB200) cluster, `oci-hsg`**, and
have **run into problems**.

We want your **independent, unbiased** opinion on the viable **directions** to get the golden run
reproducing 492/500 on `oci-hsg` ARM — with tradeoffs, risks, and a recommended first experiment.

> **Important:** investigate and form your **own** diagnosis. We are *deliberately not* sharing our
> `oci-hsg` findings or the fixes we've tried, so your assessment isn't anchored to ours. We do have
> detailed notes — ask the human for them **only after** you've formed an initial independent view.

You have working `ssh` access to **both** clusters from your shell (`ssh cw-dfw`, `ssh oci-hsg`) and
can allocate nodes and run experiments yourself.

---

## What the "golden" run is

A **known-answer baseline**: instead of a model generating patches, each SWE-bench Verified instance
is scored by feeding its **gold (ground-truth) patch** through the test suite. No model, no vLLM, no
GPU inference. It validates that the eval harness + per-instance sandboxes are working: a correct
gold patch should make the instance **resolve**.

**`resolved` (per instance), computed inside the per-instance x86_64 `.sif` eval sandbox:**
1. reset the repo (`/testbed`) to `base_commit`
2. apply the **gold patch** (as the "model prediction")
3. apply the instance's **`test_patch`** (adds/updates the tests)
4. run the instance's **`FAIL_TO_PASS`** and **`PASS_TO_PASS`** tests
5. `resolved = (all FAIL_TO_PASS now pass) AND (all PASS_TO_PASS still pass)`

---

## The working baseline — cw-dfw (substantive; dig in here)

**Result:** **492/500 resolved**, `patch_exists` 500/500. The 8 non-resolved are **environment/flaky**
test failures (documented upstream gold-failures), not harness bugs — so ~492 is the expected gold
ceiling, not a defect. The 8: `astropy-7606`, `astropy-8707`, `astropy-8872`, `django-10097`,
`pylint-6528`, `pylint-7277`, `sphinx-8595`, `sphinx-9711`.

**How it runs:** the NVIDIA NeMo Gym `swe_agents` flow, put into "gold mode" by a small runtime
patch (`gold_patch_app.py`) that sets the prediction `model_patch = instance_dict["patch"]` and skips
the OpenHands/vLLM agent. The per-instance x86_64 `.sif` then does the eval/scoring above.

**cw-dfw environment facts:**
- Nodes are **x86_64**, and have **only `enroot`** (no `docker`, no `apptainer` on the host). The Gym
  flow installs `apptainer` at runtime *inside* an amd64 Gym container, and that runs the `.sif`
  sandboxes.
- The `.sif` eval images are the official SWE-bench `sweb.eval.x86_64.*` images (naming maps the
  instance-id `__` → `_1776_`, e.g. `astropy__astropy-13033` → `swebench_sweb.eval.x86_64.astropy_1776_astropy-13033.sif`).

**Where things live on cw-dfw (`ssh cw-dfw`):**
- Golden build + scripts + full runbook:
  `/lustre/fsw/portfolios/coreai/users/adasif/nel/nvidia-eval-factory-benchmarking/`
  - `reviewed_configs/vpr/2026/nemotron-3dot5-nano/swe-bench/GOLDEN_RUNBOOK.md` ← **read this**; it has
    the exact commands, files, monitoring, the 8 misses, and the gotchas.
  - `scripts/swe/gold_patch_app.py` (gold-mode runtime patch), plus `run_golden.sh`,
    `golden_worker.sh`, `run_gym_pair.sh` in the swe-bench dir.
- **Prepped dataset (all 500 instances with full gold `patch`, `test_patch`, `FAIL_TO_PASS`,
  `PASS_TO_PASS`, `base_commit`, `version`):**
  `/lustre/fsw/portfolios/llmservice/users/sdevare/repos/nano/dataset/rl/part_[0-4].jsonl`
  (each line is one instance; the SWE-bench fields are inside a nested `instance_dict`).
- x86_64 `.sif` eval images: `/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench/`

These `.sif` images are the **same** x86_64 images used on oci-hsg — so cw-dfw is your reference for
what "correct" looks like, and you can diff behavior directly.

---

## oci-hsg — the bare minimum to get started (investigate the rest yourself)

- **Access:** `ssh oci-hsg` (works from your shell, user `adasif`).
- **Hardware:** ARM / `aarch64` GB200 nodes.
- **Reserved node for this work:** `nvl72171-T06`, via reservation
  `sla_res_id_174_nemo_3_ultra_swe_Bench`, account `llmservice_nemotron_ultra`, partition `batch`,
  QOS `interactive`. **Node-setup steps are in the "Node setup" subsection below.**
- **Assets a colleague staged** (under `bxyu`):
  - A container image meant to help run the x86 images on ARM:
    `/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/bxyu/nemo-gym/results/container_with_qemu.sqsh`
    (launch via pyxis: `srun ... --container-image=<that.sqsh> --container-mounts=/dev/kvm:/dev/kvm,/lustre:/lustre ...`)
  - The 500 x86_64 `.sif` eval images:
    `/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/bxyu/nemo-gym/results/swebench_verified_containers/`
- **Writable scratch:**
  `/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/adasif/`
- **Goal here:** get the golden gold-patch eval running on this ARM node and reproduce cw-dfw's
  **492/500** (same images, same gold patches). It does **not** currently work end-to-end — figure
  out **why**, then propose directions.

> Ops note (logistics only, not a hint): the `oci-hsg` slurm controller is sometimes slow / times out
> (`Socket timed out`) — just retry.

### Node setup (plumbing we're giving you, so you can actually run the x86 images)

This is just *infrastructure* to get a working x86-on-ARM sandbox. The **diagnosis is still yours**.

**1. Hold the node** (detached, so you can drive it over multiple `ssh`/`srun` calls):
```bash
setsid bash -c 'salloc --no-shell -J hold -A llmservice_nemotron_ultra -p batch -q interactive \
  --reservation=sla_res_id_174_nemo_3_ultra_swe_Bench -w nvl72171-T06 -N1 --gres=gpu:4 -t 04:00:00' \
  > salloc.log 2>&1 </dev/null &
sleep 10; grep -o 'job allocation [0-9]*' salloc.log     # note the JOBID
```
(For a quick *manual* poke instead, run a plain `srun` with the reservation/account flags plus
`--container-image=<qemu.sqsh> --container-mounts=/dev/kvm:/dev/kvm,/lustre:/lustre --pty bash` to get
a shell straight inside the container.)

**2. Run work inside the qemu container** (one step at a time, via `--overlap` into the held alloc):
```bash
QSQSH=/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/bxyu/nemo-gym/results/container_with_qemu.sqsh
srun --jobid=<JOBID> --overlap --container-image="$QSQSH" \
  --container-mounts=/dev/kvm:/dev/kvm,/lustre:/lustre --no-container-mount-home  bash <your-script>
```

**3. Register the x86 qemu binfmt handler** — do this *inside* the container as the first thing each
session (a freshly-booted node may not have it). You're root-in-userns in the container:
```bash
umount /proc/sys/fs/binfmt_misc 2>/dev/null
mount -t binfmt_misc none /proc/sys/fs/binfmt_misc
REG=':qemu-x86_64:M::\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x3e\x00:\xff\xff\xff\xff\xff\xfe\xfe\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff:/usr/libexec/qemu-binfmt/x86_64-binfmt-P:POF'
printf '%s\n' "$REG" > /proc/sys/fs/binfmt_misc/register   # use %s (LITERAL) — do NOT let printf interpret the \x escapes
cat /proc/sys/fs/binfmt_misc/qemu-x86_64                   # should print: enabled ...
```

**4. Sanity check that x86 emulation works:**
```bash
SIFDIR=/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/bxyu/nemo-gym/results/swebench_verified_containers
SIF=$(ls "$SIFDIR"/*.sif | head -1)
apptainer exec "$SIF" uname -m                             # -> x86_64
```
Inside each `.sif`: the test-env python is `/opt/miniconda3/envs/testbed/bin/python` and the repo is
at `/testbed`. For an eval that writes into `/testbed`, add `apptainer exec --writable-tmpfs`.

> Now you have a working x86-on-ARM sandbox. **Reproducing the golden eval and finding out why it
> doesn't reach 492/500 is what we want you to investigate** — we've deliberately withheld our root
> cause and our attempted fixes so your assessment stays independent.

---

## What we want back

1. A clear **root-cause diagnosis** of why the golden run doesn't reproduce on oci-hsg ARM (with the
   evidence you gathered).
2. A ranked set of **directions** to reach 492/500 on oci-hsg, each with: approach, whether it would
   faithfully reproduce the *exact* x86 score, effort/risk, and the single **first experiment** to
   de-risk it.
3. Call out anything we may be missing, and **challenge assumptions** (including the premise that the
   x86 images must run unchanged). Be candid about tradeoffs.

When you have an initial independent view, you can ask the human for our own notes to compare.
