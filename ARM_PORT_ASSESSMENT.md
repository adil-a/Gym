# SWE-bench Verified golden run on oci-hsg (GB200/ARM) — independent assessment

*adasif + Claude, 2026-06-10. Companion to `ARM_PORT_AGENT_BRIEF.md`. All experiments on `nvl72171-T06`
via bxyu's `container_with_qemu.sqsh`, the staged 500 x86_64 `.sif` images, and gold-patch evals
generated with the same SWE-bench fork (`HeyyyyyyG/SWE-bench`) the cw-dfw golden run uses.*

## TL;DR

The x86 `.sif` sandboxes mostly *run* under qemu-user on the GB200 nodes — but the nodes boot a
**64K-page kernel** (`6.8.0-*-nvidia-64k`), and qemu-user cannot faithfully emulate the 4K-granularity
memory operations x86 Linux binaries assume. That one platform fact produces every failure class we
measured. An unmitigated full-500 gold census scores **321/500** (cw-dfw: 492/500). A stack of
userspace mitigations (no admin, no image changes, no eval.sh changes) recovers it to
**334/500**. The rest — 86 instances that die at `import numpy` plus ~72 whose test runs crash in
multithreaded x86 processes — are both manifestations of the page-size mismatch and are physically
unfixable from userspace; they need a **4K-page kernel** (admin reboot of one canary node) or a
**KVM 4K-page guest VM** (admin: kvm group membership). Both asks are below.

## 1. Evidence chain (what fails and exactly why)

### 1a. Deterministic ImportError — 86 instances (astropy 22, matplotlib 34, xarray 22, seaborn 2, sklearn ~6)

`ld.so` fails to map ELF segments of vendored manylinux libs:

```
mmap(0xfffff6875000, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_DENYWRITE|MAP_FIXED, fd=3, 0x275000)
  = -1 errno=14 (Bad address)            ← qemu 8.2.2 AND 10.0.8 (both tested)
OSError: …/numpy.libs/libgfortran-040039e1.so.5.0.0: failed to map segment from shared object
```

On a 64K-page host, a file mapping needs `offset ≡ addr (mod 64K)`; these libraries are linked at 4K
granularity (`libgfortran` LOAD segments are congruent mod 4K but **not** mod 64K), and qemu's
partial-page fallback EFAULTs into the loader's PROT_NONE reservation. Anything importing the numpy
stack dies. **Newer qemu does not fix this** — we extracted qemu 10.0.8 static from Debian trixie and
reproduced identically.

### 1b. SIGSEGV in multithreaded guest processes — ~160 instances tagged in census

Three flavors, one mechanism (qemu TCG races on the 64K host):

- **git** (~153 instances' first crash): `git status/diff/checkout` segfaults in eval.sh. Deterministic
  *per image* — sphinx-8721 and django-10914 crash 50/50, xarray-2905 0/50 — driven by whether the
  image's `.git` index is big enough to trigger git's **threaded index preload** (strace shows the
  crash after `madvise(MADV_DONTNEED)` in the preload thread). This is why sphinx scored **0/44**.
- **OpenBLAS**: imports spawn one worker thread per visible CPU (=144) → segv at teardown or mid-run
  (deterministic on sklearn images).
- **tox / python**: sphinx's eval command (`tox --current-env`) crashes in its threaded output pumps;
  the same tests pass when invoked directly (verified: sphinx-8721's full F2P+P2P set passes with the
  tox wrapper bypassed).

Two negative results pin the mechanism: the crashes persist under `QEMU_ONE_INSN_PER_TB=1`
(deterministic single-instruction TBs, verified engaged via its ~11x slowdown) and under single-CPU
`taskset` — so they are NOT TCG execution races. They are structural: thread stacks and their guard
pages are 4K-granularity `mmap`/`mprotect`/`madvise` operations that cannot be honored inside shared
64K host pages. i.e. the thread-crash class has the same page-size root cause as 1a.

### 1c. Environment breakage: locale

`locale-gen` claims success under qemu but `en_US.UTF-8` never registers → older django eval.sh
(which doesn't export `PYTHONIOENCODING`) dies with `UnicodeEncodeError` writing `…` to stdout.

### 1d. Emulation slowness (minor)

~5–10× on CPU-bound python. p50 per instance 58s, p99 452s at 32-way parallelism — full 500 census
ran in ~50 min on one node. Only ~4 instances (3 long sympy + sklearn-14710) exceeded the 1800s test
timeout. Two pytest instances fail a single *timing-sensitive race test*
(`test_crash_on_closing_tmpfile_py27`) — emulation-timing artifact, not a crash.

## 2. Measured scores

| configuration | resolved | notes |
|---|---|---|
| cw-dfw x86 (reference) | 492/500 | 8 known env-flaky misses (upstream-documented) |
| oci-hsg unmitigated | **321/500** | sphinx 0/44, astropy 0/22, matplotlib 0/34, xarray 0/22, seaborn 0/2, sklearn 12/32, django 205/231 |
| + git de-thread config + BLAS pinning | 332/500 | +11 sklearn |
| + PYTHONIOENCODING=utf8 | 334/500 | +2 django (locale class) |
| + QEMU_ONE_INSN_PER_TB=1 on remainder | no change (0/38 graded mid-wave) | proves crashes are structural, not TCG races |
| **userspace ceiling** | **334/500** | remaining gap = 86 mmap-import + ~72 thread-crash (both page-size) + 8 known-x86 + ~4 timeouts + 2 timing-sensitive tests |

## 3. The userspace mitigation stack ("fix on our end")

All injectable from the Gym `swe_agents` harness (apptainer `--env` flags in
`_build_apptainer_command`, plus a git-config preamble before `eval.sh`); zero changes to the
`.sif` images or eval.sh definitions:

```bash
# container env (apptainer --env …)
OPENBLAS_NUM_THREADS=1  OMP_NUM_THREADS=1  MKL_NUM_THREADS=1   # kills the 144-thread BLAS pools
PYTHONIOENCODING=utf8                                          # locale machinery broken under qemu
# in-container preamble (before eval.sh)
git config --global core.preloadIndex false
git config --global index.threads 1
git config --global pack.threads 1
git config --global grep.threads 1                             # kills git's crashing preload/pack threads
```

Verified: git config 20/20 vs 0/20 on deterministic crashers; BLAS pinning 3/3 vs 0/3; utf8 env
+2 django. Tested and NOT sufficient: `taskset` single-CPU pinning, `QEMU_ONE_INSN_PER_TB=1`, and
qemu 10.0.8 — the in-test thread crashes survive all three (they avoid nothing; the mitigations that
work do so by preventing thread/mapping creation, not by fixing qemu).

Also required for the end-to-end Gym run on oci-hsg (independent of all the above): the eval inside
each `.sif` invokes a bind-mounted **x86_64** SWE-bench harness venv
(`swe_agents/swe_swebench_setup/SWE-bench/venv`). It must be staged on oci-hsg lustre (copy from
cw-dfw, or build once under the qemu sandbox). bxyu's May-20 attempt failed on exactly this before
ever reaching the qemu issues.

## 4. Why this fully explains the oci-hsg failure

- Pure-python repos (requests 8/8, flask 1/1, sympy 72/75, pytest 17/19, django 205/231 unmitigated)
  pass — the harness, image resolution, patch application, parsers all work on ARM.
- Every unresolved instance carries one of the above signatures (153 git-first-crash, 24
  import-mmap-first, remainder BLAS/tox/timeout) — there were **zero** unexplained clean test
  failures in the census.
- The 8 cw-dfw misses reproduce as misses here too (same env-flaky tests), consistent with ~492
  being the ceiling once the platform issue is gone.

## 5. Ranked directions to 492/500

| # | direction | faithful to x86 score? | effort / risk | first experiment |
|---|---|---|---|---|
| 1 | **4K-page kernel on a canary node** (`linux-image-…-nvidia` non-`-64k` flavor exists in Ubuntu noble for the same kernel version already installed) | Yes — same images, same eval, mmap constraint vanishes by construction | Admin ticket + reboot of 1 node. Risk: Lustre-client/NVIDIA/MOFED modules must exist for the 4K flavor; GPU perf on that node may regress (irrelevant for this CPU-only eval) | Reboot `nvl72171-T06` to 4K; run staged `git status ×50` + `import numpy` probes (minutes), then the staged full census (~1h) — expect 484/500-ish (492 minus residual timing/timeout cases) without ANY of the mitigations above |
| 2 | **KVM aarch64 guest VM with a 4K-page guest kernel** (host kernel untouched; run the whole qemu-user pipeline inside the VM) | Yes — same property as #1 | Admin: add user to `kvm` group / udev rule on 1 node (we verified `/dev/kvm` is currently permission-denied). Then ~1 day plumbing (cloud image, virtiofs for /lustre) | After kvm access: boot Ubuntu arm64 cloud image with 4K kernel, mount /lustre via virtiofs, re-run the staged probes inside |
| 3 | **Userspace mitigation stack (available today, no admin)** | Partial — 334/500 ceiling; numpy-mmap and in-test threading classes cannot pass | Zero admin. Already implemented + measured in `armport/scripts/run_goldeval_*.sh` | Done (this report) |
| 4 | **Hybrid images**: x86 sifs for pure-python repos + **arm64 sifs** for the 86 numpy-stack instances | Near-faithful per-instance, but not "same images"; arm64 gold ceiling must be re-baselined (the 8 misses may differ) | Upstream arm64 images exist for only **281/500** (and notably NOT for most of the missing classes — sklearn/matplotlib/xarray largely absent); building ~219 locally on ARM is real work, old envs may not build | Pull `sweb.eval.arm64.astropy_1776_astropy-12907`, convert to sif, run its gold eval natively on ARM — if resolved, extend to the 86 |
| 5 | **Full-system x86 VM under TCG** (`qemu-system-x86_64`, no KVM possible for x86-on-ARM) | Yes in correctness; timing differs (10–30×) → test timeouts must be raised (fidelity caveat) | No admin, but slowest option; throughput likely OK only for nightly golden runs | Boot one x86 VM (cloud image), virtiofs /lustre, run the sklearn + astropy evals inside, measure wall-clock |

**Recommendation:** file the admin ticket for #1 (canary node) and #2 (kvm group) together — #2 is the
fallback if the 4K kernel flavor is missing Lustre modules. Use #3 *now* to unblock partial-coverage
golden runs and CI. Treat #4/#5 as contingencies.

Confidence note for #1/#2: the mmap class (86) is provably page-size-caused. The thread-crash class
is structural to 64K hosts with very high confidence: it survives deterministic execution mode and
CPU pinning (ruling out races), localizes to 4K-granular thread-stack/guard-page operations, and
these qemu paths run crash-free at industrial scale on 4K hosts (docker buildx, CI cross-builds).
The canary run is the experiment that confirms it; if any residue survives on 4K, the mitigation
stack from #3 composes cleanly on top.

## 6. Admin ticket (ready to paste)

> **Subject:** 4K-page kernel on one GB200 node (canary) + kvm group access — needed for SWE-bench eval reproduction
>
> We need to run x86_64 evaluation containers under qemu-user on oci-hsg. The current
> `6.8.0-1046-nvidia-64k` kernel's 64K page size breaks qemu's emulation of 4K-granularity x86 mmaps
> (ELF loading and git index mapping fail; measured 321/500 vs 492/500 on the x86 reference).
>
> **Ask 1 (preferred):** on node `nvl72171-T06` (reservation `sla_res_id_174_nemo_3_ultra_swe_Bench`),
> install and boot the 4K flavor of the already-installed kernel: `linux-image-6.8.0-1046-nvidia` +
> `linux-headers-6.8.0-1046-nvidia` (no `-64k` suffix; same Ubuntu noble archive). Please ensure the
> Lustre client (and MOFED/NVIDIA DKMS, if applicable) are built for that flavor before reboot.
> Verification after boot: `getconf PAGESIZE` → 4096, and `/lustre` mounts. GPU workloads are not
> needed on this node for our eval.
>
> **Ask 2 (cheaper fallback):** add user `adasif` to the `kvm` group (or udev rule mode 0666 on
> `/dev/kvm`) on one GB200 node, so we can boot a 4K-page aarch64 guest VM under KVM instead of
> changing the host kernel.

## 7. Reproduction assets

Everything is staged and resumable on oci-hsg under
`/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/adasif/armport/`:
`qemu-x86_64-10.0.8` (+ binfmt registration snippet — note: binfmt_misc is per-container-namespace on
this kernel; re-register per srun), `goldeval500/` (per-instance `eval.sh` + `gold.patch`),
`scripts/run_goldeval*.sh` (resumable runners: baseline / mitigated / one-insn),
`results500*/` (raw test outputs for every configuration), probe scripts 2–13.
Grading runs locally against the same fork: `/tmp/armport/{grade500.py,grade_retry.py}` with
`swb-venv`. Census artifacts: `/tmp/armport/results500*/` + `_census.json`.
