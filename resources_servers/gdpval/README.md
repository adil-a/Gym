# GDPVal resources server

Scores deliverables produced by the Stirrup agent on the GDPVal benchmark.

Two modes via `reward_mode` config:

- `rubric` (default) — LLM judge scores each deliverable against a per-task
  rubric, reward in `[0.0, 1.0]`.
- `comparison` — pairwise judge compares eval deliverable vs. one or more
  reference rollouts, reward in `{0.0, 0.5, 1.0}` per ref (mean across refs).
  `aggregate_metrics` fits one eval ELO jointly via Bradley-Terry MLE.

Canonical entry point is the benchmark at `benchmarks/gdpval/`:

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/gdpval/config.yaml]"
ng_e2e_collect_rollouts \
  "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,benchmarks/gdpval/config.yaml]" \
  ++split=benchmark
```

See `benchmarks/gdpval/README.md` for the full run recipe.

## Committee-of-references (comparison mode)

Set `committee_references` to a list of `{name, deliverables_dir, elo}` triples
to score the eval against multiple reference models at once. Each reference
keeps its own anchor ELO; the aggregate metric `comparison/eval_elo` is the
joint Bradley-Terry MLE over all per-ref battles.

```yaml
gdpval_resources_server:
  resources_servers:
    gdpval:
      reward_mode: comparison
      committee_references:
        - {name: glm51,        deliverables_dir: /gdpval/refs/glm51,        elo: 1535}
        - {name: kimi_k25,     deliverables_dir: /gdpval/refs/kimi_k25,     elo: 1290}
        - {name: qwen35_397b,  deliverables_dir: /gdpval/refs/qwen35_397b,  elo: 1213}
```

Each `deliverables_dir` should follow the same `task_<id>/[repeat_<n>/]`
layout the Stirrup agent persists. The legacy `reference_deliverables_dir` +
`reference_elo` pair is auto-promoted to a one-element committee, so existing
single-reference configs run unchanged.

**Emitted aggregate metrics** (under `comparison/`):

| key | meaning |
|---|---|
| `eval_elo` | BT-MLE eval rating jointly over all refs |
| `eval_elo_se` | standard error from the Fisher information |
| `eval_elo_degenerate` | `true` when eval beats/loses every ref (MLE diverges) |
| `committee_size` | number of refs declared |
| `<ref>/n` `<ref>/wins` `<ref>/losses` `<ref>/ties` `<ref>/win_rate` | per-ref totals |
| `<ref>/eval_elo_single` | closed-form ELO using only this ref (diagnostic) |
| `wins` `losses` `ties` `judged` `win_rate` | summed across refs (back-compat) |

**Concurrency sizing.** Each /verify fans out across `committee_size × num_repeats`
judge calls per task via `asyncio.gather`. Peak judge QPS scales by
`committee_size` vs. the single-ref baseline. Rough rule of thumb:

```
judge_model_server.max_concurrent_requests
    ≥ agent_concurrency × committee_size × num_comparison_trials / 4
```

Bump the judge endpoint's `max_concurrent_requests` accordingly when
introducing additional committee members, or wall-clock per task scales
linearly with `committee_size`.

**Setting `committee_references` from the command line.** Hydra's basic-grammar
list-element overrides do not work against this field (the dotted-index form
`++…committee_references.0.elo=…` fails with `Cannot merge DictConfig with
ListConfig`). The only working form is a full list replacement:

```bash
++gdpval_resources_server.resources_servers.gdpval.committee_references='[
  {name: glm51,        deliverables_dir: /gdpval/refs/glm51,        elo: 1535},
  {name: kimi_k25,     deliverables_dir: /gdpval/refs/kimi_k25,     elo: 1290},
  {name: qwen35_397b,  deliverables_dir: /gdpval/refs/qwen35_397b,  elo: 1213}
]'
```

Verify it lands before launch with `ng_dump_config` — see
`~/code/idea/reports/plans/20260515-113133-gdpval-sed-num-repeats.md` for the
same gotcha applied to `num_repeats`.
