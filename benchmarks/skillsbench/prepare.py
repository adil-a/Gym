# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare the SkillsBench benchmark for the `benchflow_agent`.

Clones the SkillsBench task-definition repo at a pinned commit into ``data/`` (this
checkout is the runtime ``tasks_dir`` for the agent) and writes one Gym JSONL row
per task. Each row carries an ``instance_id`` of the form ``skillsbench::<task>``;
the agent runs that single task through BenchFlow. The task instruction lives inside
the task's container, so ``responses_create_params.input`` is empty.

Run via ``ng_prepare_benchmark "+config_paths=[benchmarks/skillsbench/config.yaml]"``,
which calls ``prepare()`` with no arguments (the pinned defaults below).

To use a different repo/commit or exclude tasks, run this script directly with CLI
flags (``ng_prepare_benchmark`` does not forward arguments to ``prepare()``)::

    python benchmarks/skillsbench/prepare.py --commit <sha> --exclude task-a --exclude task-b

It writes the same JSONL path, so the subsequent ``ng_run`` / ``ng_collect_rollouts``
steps are unaffected by how it was invoked.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
REPO_DIR = DATA_DIR / "skillsbench_repo"
OUTPUT_FPATH = DATA_DIR / "skillsbench_benchmark.jsonl"

DEFAULT_REPO_URL = "https://github.com/benchflow-ai/skillsbench.git"
DEFAULT_COMMIT = "312d07e15e5398f6eda32ee1bb86e492ab18edd1"  # pragma: allowlist secret

# Top-level config key of the agent that serves these rows (see config.yaml). Used as
# the `agent_ref.name` so `ng_collect_rollouts` routes each row to the right server.
# Not configurable here — it must match config.yaml.
AGENT_INSTANCE_NAME = "skillsbench_benchflow_agent"


def _ensure_repo(repo_dir: Path, repo_url: str, commit: str) -> None:
    """
    Clones SkillsBench at `repo_url` and `commit` into `repo_dir`.
    If `repo_dir` is already checked out to `commit`, reuses it, otherwise removes and clones again.
    """
    if repo_dir.exists():
        head = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if head.returncode == 0 and head.stdout.strip() == commit:
            print(f"SkillsBench already at {commit}; reusing {repo_dir}")
            return
        print(f"Removing stale SkillsBench checkout at {repo_dir}")
        shutil.rmtree(repo_dir)

    print(f"Cloning {repo_url} at {commit}...")
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", commit], check=True)


def _discover_task_names(repo_dir: Path, excluded_tasks: set[str]) -> list[str]:
    """Returns sorted SkillsBench task directory names (those with a task.toml)."""
    tasks_root = repo_dir / "tasks"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"No tasks/ directory found in SkillsBench checkout at {tasks_root}")
    return sorted(
        task_dir.name
        for task_dir in tasks_root.iterdir()
        if task_dir.is_dir() and (task_dir / "task.toml").exists() and task_dir.name not in excluded_tasks
    )


def prepare(
    repo: str = DEFAULT_REPO_URL,
    commit: str = DEFAULT_COMMIT,
    excluded_tasks: set[str] | None = None,
) -> Path:
    """Clones SkillsBench and creates a JSONL file with one row per task. Returns the JSONL path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_repo(REPO_DIR, repo, commit)

    task_names = _discover_task_names(REPO_DIR, excluded_tasks or {})
    if not task_names:
        raise RuntimeError(f"No SkillsBench tasks found under {REPO_DIR / 'tasks'}")

    with open(OUTPUT_FPATH, "w", encoding="utf-8") as f:
        for task_name in task_names:
            row = {
                "instance_id": f"skillsbench::{task_name}",
                "responses_create_params": {"input": []},
                "agent_ref": {"name": AGENT_INSTANCE_NAME},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(task_names)} SkillsBench tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares the SkillsBench benchmark dataset.")
    parser.add_argument("--repo", default=DEFAULT_REPO_URL, help="SkillsBench git repo URL.")
    parser.add_argument("--commit", default=DEFAULT_COMMIT, help="SkillsBench commit to check out.")
    parser.add_argument(
        "--exclude", action="append", default=None, metavar="TASK", help="Task to exclude (repeatable)."
    )
    args = parser.parse_args()
    prepare(
        repo=args.repo,
        commit=args.commit,
        excluded_tasks=set(args.exclude) if args.exclude is not None else None,
    )
