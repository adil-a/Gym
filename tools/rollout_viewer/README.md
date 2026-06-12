# Rollout Viewer

An offline [Streamlit](https://streamlit.io/) app for reading and analyzing the rollout
artifacts NeMo Gym produces during evaluation and reward profiling. It renders each rollout's
full trajectory — conversation, tool calls, tool results, and reasoning/thinking traces — and
shows per-run metrics.

See [`DESIGN_NOTE.md`](DESIGN_NOTE.md) for the design and trade-offs.

## Install

The viewer is behind an optional `viewer` extra (Streamlit + pandas) so the rest of NeMo Gym
stays headless and lean:

```bash
uv sync --extra viewer
# or
pip install -e ".[viewer]"
```

## Run

```bash
# Scan a results directory and pick a run from the dropdown (default dir: results/)
ng_view_rollouts

# Scan a specific directory
ng_view_rollouts --dir path/to/results

# Open a specific rollouts file (siblings auto-discovered by stem)
ng_view_rollouts --rollouts results/my_run_rollouts.jsonl

# Override any individual sibling file
ng_view_rollouts --rollouts results/my_run_rollouts.jsonl \
  --metrics results/my_run_aggregate_metrics.json \
  --failures results/my_run_failures.jsonl
```

`ng_view_rollouts` wraps `streamlit run`; if the `viewer` extra isn't installed it prints an
install hint and exits.

## What it shows

A run is the set of sibling files sharing a stem:

| File | Contents |
|------|----------|
| `<stem>.jsonl` | one rollout per line (the rollouts file) |
| `<stem>_aggregate_metrics.json` | per-agent / per-task metric rollup |
| `<stem>_materialized_inputs.jsonl` | resolved task inputs |
| `<stem>_failures.jsonl` | rollouts that errored out |

Only the rollouts file is required; missing siblings degrade gracefully.

Three tabs:

- **Conversation** — the full transcript (`responses_create_params.input` followed by
  `response.output`) rendered top to bottom: chat messages, collapsible tool-call and tool-result
  cards (with JSON pretty-printing and `call_id` shown), and collapsed reasoning blocks. Unknown
  item types fall back to a labeled raw-JSON card. A "group by task" toggle surfaces the repeated
  rollouts of one task and lets you **diff two rollouts side by side**, highlighting the first
  point where they diverge. Every rollout has a "show raw JSON" toggle.
- **Metrics** — headline numbers from the aggregate file (when present), plus a self-computed
  reward histogram, a tokens-vs-reward scatter, and a sortable per-rollout table.
- **Failures** — failed rollouts rendered as raw-JSON cards.

## Layout

```
tools/rollout_viewer/
├── core.py        # pure logic: load, classify transcript, diff, metrics (no Streamlit)
├── app.py         # thin Streamlit UI shell
├── DESIGN_NOTE.md
├── README.md
└── tests/         # unit tests for core.py (run with the commands below)
```

`core.py` holds all the parsing/diff/metric logic and has no Streamlit dependency, so it is
unit-tested directly; `app.py` is a thin rendering shell. The `ng_view_rollouts` launcher lives
in `nemo_gym/view_rollouts.py` (it must be an importable package module to be a console script).

## Test

`tools/` is outside the default pytest discovery path, so run the viewer tests explicitly:

```bash
cd tools/rollout_viewer
python -m pytest tests/ -o addopts="" --cov=core --cov-report=term-missing
```
