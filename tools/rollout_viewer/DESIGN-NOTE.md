# Rollout Viewer Design Note

`rollout_viewer` is an offline Streamlit app for inspecting the rollout artifacts NeMo Gym
produces during evaluation and reward profiling. It exists because those artifacts are JSONL
files of OpenAI Responses API objects — faithful but unreadable by hand — and there is no way
today to *read a trajectory* (conversation, tool calls, reasoning) or *analyze a run's metrics*
without writing throwaible code each time.

It is an analysis tool, not a server. It does not follow the FastAPI/`ServerClient` server
conventions; it reads files from disk and renders them.

## Scope

In scope for v1:

* Load a single run (or switch between runs in a directory) and render each rollout's full
  trajectory as a readable transcript.
* Slice and analyze a run's rollouts: filterable table, group-by-task, reward/token charts,
  and the precomputed aggregate metrics.
* Diff two rollouts of the *same task* side by side.

Explicitly deferred (structured for, not built):

* Cross-run / model-vs-model comparison (load two runs, diff the same task across them).
* Free-text search across a transcript.
* A persistent SQLite catalog of many runs queried over time.

## Artifacts It Reads

A reward-profiling / rollout-collection run writes a set of sibling files sharing a stem
`<stem>`:

| File | Contents | Required |
|------|----------|----------|
| `<stem>.jsonl` | one rollout per line (the `_rollouts.jsonl`) | yes |
| `<stem>_aggregate_metrics.json` | per-agent + per-task rollup, `rollout_infos` | no |
| `<stem>_materialized_inputs.jsonl` | resolved task inputs | no |
| `<stem>_failures.jsonl` | rollouts that errored out | no |

Each rollout line has the shape (fields omitted for brevity):

```json
{
  "responses_create_params": { "input": [ ... ], "tools": [ ... ] },
  "response": { "output": [ ... ], "usage": { "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 } },
  "reward": 1.0,
  "_ng_task_index": 0,
  "_ng_rollout_index": 0,
  "agent_ref": { "name": "..." }
}
```

The full transcript of a rollout is `responses_create_params.input` (the task setup) followed by
`response.output` (the generated turns). Both are ordered lists of the same item types.

### Trajectory Item Types

The transcript is a heterogeneous ordered list. The renderer classifies each item and styles it:

| Item type | Key fields | Rendered as |
|-----------|-----------|-------------|
| `message` | `role`, `content[].text` | role-colored chat block |
| `function_call` | `name`, `arguments` (JSON string), `call_id` | collapsible tool-call card, args pretty-printed, paired to its output by `call_id` |
| `function_call_output` | `output` (string, often JSON), `call_id` | collapsible tool-result card, JSON pretty-printed when parseable |
| `reasoning` | `summary[].text`, `encrypted_content` | distinct, collapsed-by-default "thinking" block |
| _anything else_ | — | labeled raw-JSON card (no crash on unknown types) |

## Architecture

Two modules, split so the value-bearing logic is testable without a Streamlit runtime:

* **`core.py`** — pure functions, zero Streamlit imports:
  * `discover_run(path_or_dir) -> RunRef` and `scan_dir(dir) -> list[RunRef]`: find a
    `_rollouts.jsonl` and derive its siblings by stem; tolerate missing siblings.
  * `load_run(RunRef) -> Run`: parse rollouts + aggregate + failures; cached by the caller on
    `(path, mtime)`.
  * `rollouts_to_frame(Run) -> DataFrame`: flatten the per-rollout scalar columns
    (`_ng_task_index`, `_ng_rollout_index`, `reward`, `input/output/total_tokens`, agent name)
    via `json_normalize`, for the table and charts.
  * `iter_transcript(rollout) -> list[Item]`: concatenate input + output and classify each item
    into a typed view with `call_id` pairing resolved.
  * `diff_rollouts(a, b) -> Divergence`: align two same-task transcripts and mark the first
    differing item / differing tool calls.
  * Everything is keyed by a `run_id` (derived from the stem) so a second run is additive later.
* **`app.py`** — thin Streamlit shell: sidebar (dir picker + dropdown + per-file overrides),
  three views (Conversation, Metrics, Failures), wiring `core.py` outputs to widgets. Holds no
  business logic worth unit-testing; uses `@st.cache_data` keyed on `(path, mtime)` to memoize
  `load_run`.

### Why pandas in-memory, not SQLite

The JSONL is the canonical artifact. A DB would add an ETL step, a schema, and a staleness
failure mode (re-collecting a run invalidates the DB), to solve a performance problem that does
not exist at this scale (a heavy run is tens of thousands of small rows; pandas filters it in
well under a second). The nested `output` array also does not map cleanly to relational tables —
only the flat scalar columns are needed for tables/charts, and the conversation view reads the
nested structure directly from the selected row. `@st.cache_data` keyed on `(path, mtime)` gives
the only caching this tool needs. SQLite is revisited only if a persistent multi-run catalog
(the deferred item) is built.

## Views

* **Conversation** — pick a rollout (driven by the metrics table selection or a direct picker),
  render its transcript per the item-type table above. Tool cards and reasoning blocks are
  collapsible; a per-rollout "show raw JSON" toggle exposes ground truth. A "group by task"
  toggle surfaces the repeated rollouts of one task; within a task, two rollouts can be selected
  for a side-by-side diff with divergence highlighting.
* **Metrics** — headline stats from `_aggregate_metrics.json` when present (`key_metrics`,
  per-agent mean/median/std), plus self-computed Altair visuals from the rollouts: reward
  distribution histogram, token-vs-reward scatter, and a sortable per-task table. Falls back to
  purely self-computed metrics when the aggregate file is absent. Altair is used because it ships
  bundled with Streamlit (no added dependency) and supports interactive hover/zoom.
* **Failures** — lists rows from `_failures.jsonl` as labeled raw-JSON cards. The file's exact
  schema is not assumed, so it is rendered raw rather than over-parsed.

## Packaging & Invocation

* Lives under `tools/rollout_viewer/` with its own `README.md`.
* Streamlit + pandas are an **optional extra** (`pip install -e ".[viewer]"` /
  `uv sync --extra viewer`), not core dependencies — the framework mostly runs headless and these
  libs are heavy. Altair arrives transitively with Streamlit.
* An `ng_view_rollouts` console-script entrypoint wraps launch: it checks the imports (printing a
  friendly "install the `viewer` extra" message on `ImportError`), then exec's
  `streamlit run tools/rollout_viewer/app.py -- <args>`. Arguments: optional `--dir`, and
  explicit `--rollouts/--metrics/--materialized/--failures` overrides for any individual file.

## Testing

`core.py` is unit-tested to the repo's ≥96% bar against a small committed fixture run that
exercises every branch: all four known item types, an unknown item type (raw-JSON fallback), a
task with multiple rollouts (grouping + diff), a missing sibling file (graceful degradation), and
a malformed `arguments` / non-JSON tool output (pretty-print fallback). `app.py` is a thin
rendering shell excluded from the coverage target; it is at most smoke-imported.

## Trade-Offs

* **Streamlit over a static HTML generator or TUI.** The stated need is *interact* + *analyze
  metrics*, which is a dashboard. Streamlit gives the file picker, filtering, selection-driven
  panes, and charts with the least code. Cost: a separate process and an optional heavy
  dependency — fenced behind the `viewer` extra so non-users pay nothing.
* **Pure-core + thin-UI split.** Makes the parsing/pairing/diff logic — where correctness lives —
  testable without a Streamlit runtime, at the cost of a small indirection between `core.py` and
  `app.py`. This is what lets the tool meet the coverage bar without fighting `AppTest`.
* **Within-run only, keyed by `run_id`.** Building cross-run diff now would add a two-run data
  model and selection complexity before it is needed. Keying everything by `run_id` from the
  start makes the later jump to cross-run comparison a UI addition rather than a data-model
  rewrite.
* **Raw-JSON fallback for unknown item types and the failures file.** The Responses API surface
  evolves; rendering unknowns raw keeps the viewer robust to formats it has not enumerated
  instead of crashing or silently dropping them.
