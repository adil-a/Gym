# CritPt Benchmark

Benchmark wrapper for [CritPt](https://huggingface.co/datasets/CritPt-Benchmark/CritPt), a
70-problem research-level physics benchmark. Each problem has a description and a Python
code template. The model must produce a precise numerical answer.

- **Tasks**: 70 physics problems
- **Reward**: binary; scored by the [Artificial Analysis API](https://artificialanalysis.ai/documentation#critpt-api) — private test cases run server-side
- **Metrics**: `pass@1/accuracy` — fraction of problems the AA API accepts

The agent runs two LLM turns per problem:
1. **Turn 1** (`prompts/turn1.yaml`): step-by-step derivation ending in `Final Answer:`
2. **Turn 2**: populate the code template with the answer (the model sees its Turn 1 reasoning)

Turn 2's output is submitted to the AA API by `CritPtResourcesServer.verify()`.

## API key

The Artificial Analysis API key is read from `env.yaml`:

```yaml
artificial_analysis_api_key: <your-key>
```

The resources server config interpolates this via `${artificial_analysis_api_key}` — no key in any committed file.

## Prepare benchmark data

`CritPt-Benchmark/CritPt` is a public HuggingFace dataset (no auth required).

```bash
python benchmarks/critpt/prepare.py
```

Writes:
- `benchmarks/critpt/data/critpt_benchmark.jsonl` — full 70-problem dataset (gitignored)
- `benchmarks/critpt/data/example.jsonl` — first 5 problems (committed)

## Running servers

```bash
ng_run "+config_paths=[benchmarks/critpt/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` and
`artificial_analysis_api_key` in `env.yaml`.

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=critpt_benchmark_agent \
    +input_jsonl_fpath=benchmarks/critpt/data/critpt_benchmark.jsonl \
    +output_jsonl_fpath=results/critpt_rollouts.jsonl \
    +num_repeats=1 \
    "++responses_create_params={temperature: 0.0}"
```

Use `temperature: 0.0` to match the nemo-skills baseline and ensure reproducible scores.

## Metrics

`pass@1/accuracy` is the headline metric. Target: match the nemo-skills CritPt score on the
same model within ~1% variance.
