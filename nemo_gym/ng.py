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
"""NeMo Gym CLI — `ng` entry point.

Single binary wrapping the core ng_* commands with a cleaner UX:

    ng start <env> [<env> ...] [-m vllm|openai|local_vllm]
    ng rollouts <env> [-n limit] [-r repeats] [-o output]
    ng prepare <env>
    ng list
    ng status
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import typer
from omegaconf import OmegaConf

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

# Directories scanned when resolving an env by name (in order).
_SCAN_DIRS = [
    "training_environments",
    "benchmarks",
    "example_environments",
    "resources_servers",
]

_MODEL_CONFIGS: Dict[str, str] = {
    "openai": "responses_api_models/openai_model/configs/openai_model.yaml",
    "vllm": "responses_api_models/vllm_model/configs/vllm_model.yaml",
    "local_vllm": "responses_api_models/local_vllm_model/configs/local_vllm_model.yaml",
    "azure_openai": "responses_api_models/azure_openai_model/configs/azure_openai_model.yaml",
}


# ============================================================================
# Helpers
# ============================================================================


def _resolve_env(name_or_path: str) -> Path:
    """Resolve an env name or direct path to a config.yaml."""
    p = Path(name_or_path)

    # Direct yaml path
    if p.suffix == ".yaml" and p.exists():
        return p

    # Directory containing config.yaml
    if (p / "config.yaml").exists():
        return p / "config.yaml"

    # Name scan across known directories
    for scan_dir in _SCAN_DIRS:
        d = Path(scan_dir)
        if not d.exists():
            continue
        for candidate in d.glob("**/config.yaml"):
            if candidate.parent.name == name_or_path:
                return candidate
        # Also match configs/*.yaml for resources_servers style
        for candidate in d.glob(f"**/{name_or_path}/configs/*.yaml"):
            return candidate

    raise typer.BadParameter(
        f"Cannot find environment '{name_or_path}'. "
        f"Pass a path or a name found under: {', '.join(_SCAN_DIRS)}."
    )


def _agent_name_from_config(config_path: Path) -> Optional[str]:
    """Return the first responses_api_agents instance name from the config."""
    try:
        cfg = OmegaConf.load(config_path)
        for key in cfg:
            block = cfg[key]
            if isinstance(block, dict) and "responses_api_agents" in block:
                return key
    except Exception:
        pass
    return None


def _example_data_path_from_config(config_path: Path) -> Optional[str]:
    """Return the example dataset jsonl_fpath from the config, if present."""
    try:
        cfg = OmegaConf.load(config_path)
        for key in cfg:
            block = cfg[key]
            if not isinstance(block, dict) or "responses_api_agents" not in block:
                continue
            agent_cfg = list(block["responses_api_agents"].values())[0]
            for ds in agent_cfg.get("datasets", []):
                if ds.get("type") == "example":
                    return ds.get("jsonl_fpath")
    except Exception:
        pass
    return None


def _hydra_call(fn, config_paths: List[str], extra_args: Optional[List[str]] = None) -> None:
    """Set sys.argv for Hydra and call the underlying ng_* function."""
    paths_str = ",".join(config_paths)
    sys.argv = [fn.__name__, f"+config_paths=[{paths_str}]"] + (extra_args or [])
    fn()


# ============================================================================
# Commands
# ============================================================================


@app.command()
def start(
    envs: List[str] = typer.Argument(..., help="One or more environment names or paths."),
    model: str = typer.Option("openai", "-m", "--model", help="Model server: openai, vllm, local_vllm, azure_openai."),
) -> None:
    """Start environment servers."""
    config_paths: List[str] = []
    for env in envs:
        config_paths.append(str(_resolve_env(env)))

    model_config = _MODEL_CONFIGS.get(model)
    if not model_config:
        typer.echo(f"Unknown model '{model}'. Choose from: {', '.join(_MODEL_CONFIGS)}", err=True)
        raise typer.Exit(1)
    config_paths.append(model_config)

    from nemo_gym.cli import run

    _hydra_call(run, config_paths)


@app.command()
def rollouts(
    env: str = typer.Argument(..., help="Environment name or path."),
    limit: Optional[int] = typer.Option(None, "-n", "--limit", help="Max number of examples to run."),
    repeats: Optional[int] = typer.Option(None, "-r", "--repeats", help="Number of rollouts per example."),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output JSONL path. Defaults to results/<env>_rollouts.jsonl."),
    input_path: Optional[str] = typer.Option(None, "-i", "--input", help="Input JSONL path. Defaults to env's example dataset."),
    agent: Optional[str] = typer.Option(None, "-a", "--agent", help="Agent name override."),
) -> None:
    """Collect rollouts for an environment."""
    config_path = _resolve_env(env)

    agent_name = agent or _agent_name_from_config(config_path)
    if not agent_name:
        typer.echo(f"Could not infer agent name from {config_path}. Use --agent to specify.", err=True)
        raise typer.Exit(1)

    resolved_input = input_path or _example_data_path_from_config(config_path)
    if not resolved_input:
        typer.echo(f"Could not infer input data path from {config_path}. Use --input to specify.", err=True)
        raise typer.Exit(1)

    env_name = Path(env).name if Path(env).exists() else env
    resolved_output = output or f"results/{env_name}_rollouts.jsonl"

    extra: List[str] = [
        f"+agent_name={agent_name}",
        f"+input_jsonl_fpath={resolved_input}",
        f"+output_jsonl_fpath={resolved_output}",
    ]
    if limit is not None:
        extra.append(f"+limit={limit}")
    if repeats is not None:
        extra.append(f"+num_repeats={repeats}")

    from nemo_gym.rollout_collection import collect_rollouts

    sys.argv = ["ng rollouts"] + extra
    collect_rollouts()


@app.command()
def prepare(
    envs: List[str] = typer.Argument(..., help="One or more environment names or paths."),
) -> None:
    """Run prepare.py for one or more environments."""
    import importlib.util
    import runpy

    for env in envs:
        config_path = _resolve_env(env)
        prepare_script = config_path.parent / "prepare.py"
        if not prepare_script.exists():
            typer.echo(f"No prepare.py found for '{env}' (looked at {prepare_script})", err=True)
            continue
        typer.echo(f"Preparing {env}...")
        runpy.run_path(str(prepare_script), run_name="__main__")


@app.command(name="list")
def list_envs() -> None:
    """List all discoverable environments."""
    import rich

    for scan_dir in _SCAN_DIRS:
        d = Path(scan_dir)
        if not d.exists():
            continue
        configs = sorted(c for c in d.glob("**/config.yaml") if ".venv" not in c.parts)
        if not configs:
            continue
        rich.print(f"\n[bold]{scan_dir}/[/bold]")
        for c in configs:
            rich.print(f"  {c.parent.name}  [dim]{c.parent}[/dim]")


@app.command()
def status() -> None:
    """Check server health."""
    sys.argv = ["ng status"]
    from nemo_gym.cli import status as _status

    _status()


if __name__ == "__main__":
    app()
