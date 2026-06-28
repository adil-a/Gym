# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gold-patch census — validate the swe_env grader end-to-end (reward-profiling evidence).

Feeds each SWE-bench (Verified, by default) instance's GOLD patch through the flat grader on the
chosen sandbox provider. No model and no agent run — a correct grader should resolve ~all instances
(modulo a handful of documented upstream env-flaky gold-failures). This is what produces the
``responses_api_agents/swe_env/README.md`` baseline (docker 493/500, apptainer/.sif 492/500).

Resumable + checkpointed to ``--out``; with ``--rmi`` each instance's docker image is removed after
grading so the full run stays within disk on a single host.

Examples:
    # docker (images pull on demand; --rmi caps disk by removing each after grading)
    HF_HOME=/tmp/hf_cache python responses_api_agents/anyswe_agent/gold_census.py --rmi
    python responses_api_agents/anyswe_agent/gold_census.py --limit 50 --concurrency 8 --rmi
    # apptainer (point the formatter at pre-built local .sif images)
    python responses_api_agents/anyswe_agent/gold_census.py --provider apptainer \\
        --container-formatter 'data/sifs/{instance_id}.sif'
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from pathlib import Path


# Run as a script from the repo root (python responses_api_agents/anyswe_agent/gold_census.py):
# put the repo root on sys.path so the first-party imports below resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import load_dataset  # noqa: E402

from responses_api_agents.anyswe_agent.app import _build_swetask  # noqa: E402
from responses_api_agents.swe_env.verify_task import verify_task  # noqa: E402


def main() -> None:
    """Parse arguments, grade every instance's gold patch, and print the resolve tally."""
    parser = argparse.ArgumentParser(description="Gold-patch census for the swe_env grader.")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--provider", default="docker", help="sandbox provider: docker or apptainer")
    parser.add_argument("--container-formatter", default="docker://swebench/sweb.eval.x86_64.{instance_id}")
    parser.add_argument("--limit", type=int, default=0, help="0 = all instances")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--out", default="gold_census_results.json")
    parser.add_argument("--rmi", action="store_true", help="docker rmi each image after grading (bounds disk)")
    args = parser.parse_args()

    # Mirror anyswe's grading provider (app.py `_grading_provider`): apptainer's base image is
    # read-only, so the eval script's git checkout / patch apply / pytest writes to /testbed need a
    # writable overlay (--writable-tmpfs -> disk-backed overlay); and the host $HOME must NOT be bound
    # in (--no-mount home), or host dotfiles/caches leak into the eval and change test outcomes vs
    # docker (matplotlib image-comparison tests fail on the host font cache). docker containers are
    # writable + host-isolated by default, so no change there.
    provider_cfg: dict = {args.provider: {}}
    if args.provider == "apptainer":
        provider_cfg = {"apptainer": {"create": {"extra_start_args": ["--writable-tmpfs", "--no-mount", "home"]}}}

    rows = list(load_dataset(args.dataset, split=args.split))
    by_id = {r["instance_id"]: r for r in (rows[: args.limit] if args.limit else rows)}
    out = Path(args.out)
    results: dict = json.loads(out.read_text()) if out.exists() else {}
    todo = [i for i in by_id if i not in results]
    sem = asyncio.Semaphore(args.concurrency)
    done = [len(results)]

    async def grade_one(instance_id: str) -> None:
        inst = by_id[instance_id]
        problem_info = {
            "instance_id": instance_id,
            "dataset_name": args.dataset,
            "container_formatter": args.container_formatter,
            "instance_dict": json.dumps(dict(inst)),
        }
        task = dataclasses.replace(_build_swetask(problem_info, flat_eval=True), model_patch=inst["patch"])
        async with sem:
            try:
                report = await verify_task(provider_cfg, task)
                results[instance_id] = {"resolved": bool(report.resolved), "error_kind": report.error_kind}
            except Exception as exc:  # keep the census going; record the failure for this row
                results[instance_id] = {"resolved": "ERR", "error": repr(exc)[:200]}
            if args.rmi and args.provider == "docker":
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "rmi",
                    "-f",
                    task.image,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
        done[0] += 1
        if done[0] % 10 == 0:
            resolved = sum(1 for v in results.values() if v.get("resolved") is True)
            print(f"  {done[0]}/{len(by_id)} graded, resolved {resolved}", flush=True)
            out.write_text(json.dumps(results, indent=2))

    async def run() -> None:
        await asyncio.gather(*(grade_one(i) for i in todo))

    if todo:
        asyncio.run(run())
    out.write_text(json.dumps(results, indent=2))
    resolved = sorted(i for i, v in results.items() if v.get("resolved") is True)
    not_resolved = sorted(i for i, v in results.items() if v.get("resolved") is not True)
    print(f"\ngold resolved {len(resolved)}/{len(results)} on {args.provider}")
    print(f"not resolved ({len(not_resolved)}): {not_resolved}")


if __name__ == "__main__":
    main()
