# swe_env

Shared library for provisioning and grading SWE (software-engineering) task
environments. It is imported by `anyswe_agent` (and usable by any other Gym
agent that self-drives inside a SWE task sandbox); it is **not** a runnable
server and has no config or entrypoint.

Everything is provider-neutral, running over the `nemo_gym.sandbox` providers
(docker / apptainer / opensandbox):

- **`harnesses/`** — per-dataset-family recipes (SWE-bench, SWE-bench
  Multilingual, SWE-bench-ext, R2E-Gym, SWE-rebench, NV-internal). Each builds
  the task sandbox spec, materializes the model patch, runs the evaluation, and
  grades the result host-side via the official per-repo parser (falling back to
  a generic parser only where the official one is unavailable).
- **`sandbox.py`** — async sandbox lifecycle (`AsyncSweEnvironment`,
  `acquire_sandbox`) with always-teardown semantics.
- **`self_drive.py`** — provision a writable sandbox, inject a sandbox-reachable
  model endpoint / egress env, run an opaque agent launch command, and extract
  the resulting `git diff` patch.
- **`verify_task.py`** — grade a patch inline in a fresh sandbox (no separate
  `/verify` server), returning a mask-aware reward.
- **`parsing/`** — test-log parsers (relocated verbatim from `swe_agents`).

## Reward-profiling baseline — SWE-bench Verified gold patches

A gold-patch census validates the grader end-to-end: feed each instance's ground-truth
patch through the flat grader and it should resolve. On the **docker** provider
(pull-on-demand images, host-side flat grading) the full 500-instance census resolves
**486 / 500** (`patch_exists` 500/500, **0** infra errors), in line with the
apptainer / `.sif` nested reference of **492 / 500** (whose 8 misses are documented
upstream env-flaky gold-failures). An empty patch resolves **0 / 500**, as expected.

The remaining ~14 docker misses are instance-specific — the documented astropy/django
upstream flaky gold-failures, plus a few where a single required test does not run in the
container — not a systematic grader defect. (An earlier census surfaced one that *was*
systematic: swebench 4.1.0's sphinx/sklearn eval command runs pytest without `-rA`, hiding
passing tests from the host-side parser; fixed by forcing `PYTEST_ADDOPTS=-rA` in the flat
eval — see `harnesses/swebench.py::_flat_eval_script` — which recovered ~45 instances, 445→486.)

### Reproduce

Grade gold patches on docker (no model, no agent — pure grader validation). Bound concurrency
with a semaphore and `docker rmi` each image after grading to cap disk:

```python
import asyncio, dataclasses, json
from datasets import load_dataset
from responses_api_agents.anyswe_agent.app import _build_swetask
from responses_api_agents.swe_env.verify_task import verify_task

async def gold_resolves(inst) -> bool:
    pinfo = {
        "instance_id": inst["instance_id"],
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "container_formatter": "docker://swebench/sweb.eval.x86_64.{instance_id}",
        "instance_dict": json.dumps(dict(inst)),
    }
    task = _build_swetask(pinfo, flat_eval=True)
    report = await verify_task({"docker": {}}, dataclasses.replace(task, model_patch=inst["patch"]))
    return bool(report.resolved)  # error_kind is None on a clean (non-infra) grade

rows = list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))
```

Swap `{"docker": {}}` for `{"apptainer": {}}` (with a `.sif` `container_formatter`) to grade on
apptainer instead.

## Tests

The unit tests run against a scripted fake `SandboxProvider`, so they need no
Docker/apptainer and execute in CI:

```bash
ng_test +entrypoint=responses_api_agents/swe_env
```
