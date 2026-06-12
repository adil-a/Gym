# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass


VERSION_TARGET = "nemo_gym.cli.general:version"


@dataclass(frozen=True)
class Command:
    # What to run: either a "module:function" string (lazily imported and called with no args),
    # or a callable(args, overrides) that owns dispatch (e.g. picks the target from parsed flags).
    target: str | Callable[[argparse.Namespace, list[str]], None]
    # One-line help shown in the parent listing and atop this command's own --help.
    summary: str | None = None
    # Hook to add this command's own flags to its subparser.
    configure: Callable[[argparse.ArgumentParser], None] | None = None


# Flags shared by all commands are added here once and attached via `parents=[COMMON]`.
COMMON = argparse.ArgumentParser(add_help=False)


def dispatch(target: str, overrides: list[str]) -> None:
    module_path, func_name = target.split(":")
    # Drop the parsed command tokens so the downstream Hydra parsing only sees overrides.
    sys.argv = [sys.argv[0], *overrides]
    func = getattr(importlib.import_module(module_path), func_name)
    func()


def _toggle_command(flag: str, flag_help: str, *, off_command: str, on_command: str, summary: str) -> Command:
    """Build a command whose target switches from `off` (default) to `on` when --<flag> is given."""
    dest = flag.replace("-", "_")

    def configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(f"--{flag}", action="store_true", help=flag_help)

    def target(args: argparse.Namespace, overrides: list[str]) -> None:
        dispatch(on_command if getattr(args, dest) else off_command, overrides)

    return Command(target=target, configure=configure, summary=summary)


def _choice_command(flag: str, flag_help: str, *, targets: dict[str, str], default: str, summary: str) -> Command:
    """Build a command whose target is selected by --<flag> {choices} (default `default`)."""
    dest = flag.replace("-", "_")

    def configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(f"--{flag}", choices=tuple(targets), default=default, help=flag_help)

    def target(args: argparse.Namespace, overrides: list[str]) -> None:
        dispatch(targets[getattr(args, dest)], overrides)

    return Command(target=target, configure=configure, summary=summary)


# One-line help for each command group, shown in `gym --help`.
GROUPS = {
    "list": "List available components. As of now, only benchmarks are available.",
    "dataset": "Manage datasets.",
    "env": "Develop and run environments.",
    "eval": "Run evaluations.",
    "dev": "Contributor helpers.",
}

COMMANDS = {
    "list benchmarks": Command(target="nemo_gym.cli.eval:list_benchmarks", summary="List available benchmarks."),
    "dataset upload": _choice_command(
        "target",
        "Destination backend (default: hf).",
        targets={
            "hf": "nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_cli",
            "gitlab": "nemo_gym.cli.dataset:upload_jsonl_dataset_cli",
        },
        default="hf",
        summary="Upload a prepared dataset to HF (default) or GitLab.",
    ),
    "dataset download": _choice_command(
        "source",
        "Source backend (default: hf).",
        targets={
            "hf": "nemo_gym.cli.dataset:download_jsonl_dataset_from_hf_cli",
            "gitlab": "nemo_gym.cli.dataset:download_jsonl_dataset_cli",
        },
        default="hf",
        summary="Download a dataset from HF (default) or GitLab.",
    ),
    "dataset rm": Command(
        target="nemo_gym.cli.dataset:delete_jsonl_dataset_from_gitlab_cli",
        summary="Delete a dataset from GitLab.",
    ),
    "dataset migrate": Command(
        target="nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_and_delete_gitlab_cli",
        summary="Transfer a dataset from GitLab to HF.",
    ),
    "dataset render": Command(
        target="nemo_gym.cli.dataset:materialize_prompts_cli",
        summary="Generate a dataset preview.",
    ),
    "dataset collate": Command(
        target="nemo_gym.cli.dataset:prepare_data",
        summary="Validate and collate the dataset.",
    ),
    "env init": Command(
        target="nemo_gym.cli.env:init_resources_server",
        summary="Scaffold config for a new server, benchmark, or agent.",
    ),
    "env resolve": Command(
        target="nemo_gym.cli.env:dump_config",
        summary="Resolve the final config from configs, flags, and overrides.",
    ),
    "env packages": Command(
        target="nemo_gym.cli.env:pip_list",
        summary="Print pip packages for the selected resource server.",
    ),
    "env test": Command(target="nemo_gym.cli.env:test", summary="Test the resource server(s)."),
    "env run": Command(target="nemo_gym.cli.env:run", summary="Start the servers."),
    "env status": Command(target="nemo_gym.cli.env:status", summary="Print the server status."),
    "eval prepare": Command(
        target="nemo_gym.cli.eval:prepare_benchmark",
        summary="Prepare benchmark data and dump it to disk.",
    ),
    "eval run": _toggle_command(
        "no-serve",
        "Collect against already-running servers instead of starting them.",
        off_command="nemo_gym.cli.eval:e2e_rollout_collection",
        on_command="nemo_gym.cli.eval:collect_rollouts",
        summary="Collate data, start servers, and collect rollouts.",
    ),
    "eval aggregate": Command(
        target="nemo_gym.cli.eval:aggregate_rollouts",
        summary="Aggregate sharded rollout results.",
    ),
    "eval profile": Command(
        target="nemo_gym.cli.eval:reward_profile",
        summary="Compute a reward profile from rollouts.",
    ),
    "dev test": Command(target="nemo_gym.cli.dev:dev_test", summary="Run NeMo Gym's unit tests."),
}


def _add_leaf(subparsers: argparse._SubParsersAction, name: str, command: Command) -> None:
    leaf = subparsers.add_parser(name, parents=[COMMON], help=command.summary, description=command.summary)
    leaf.set_defaults(_command=command)
    if command.configure is not None:
        command.configure(leaf)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gym", add_help=True)
    parser.add_argument("--version", action="store_true", help="Show the NeMo Gym version and exit.")
    parser.set_defaults(_parser=parser)

    subparsers = parser.add_subparsers()
    groups: dict[str, argparse._SubParsersAction] = {}

    for command_name, command in COMMANDS.items():
        parts = command_name.split()
        if len(parts) == 1:
            _add_leaf(subparsers, parts[0], command)
            continue

        group_name, action_name = parts
        if group_name not in groups:
            group_parser = subparsers.add_parser(
                group_name, help=GROUPS.get(group_name), description=GROUPS.get(group_name)
            )
            group_parser.set_defaults(_parser=group_parser)
            groups[group_name] = group_parser.add_subparsers()
        _add_leaf(groups[group_name], action_name, command)

    return parser


def main() -> None:
    parser = build_parser()
    args, overrides = parser.parse_known_args()

    if args.version:
        dispatch(VERSION_TARGET, overrides)
        return

    command = getattr(args, "_command", None)
    if command is None:
        args._parser.print_help()
        sys.exit(1)

    if callable(command.target):
        command.target(args, overrides)
    else:
        dispatch(command.target, overrides)
