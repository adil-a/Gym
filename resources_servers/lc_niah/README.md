# lc_niah

Long-context needle-in-a-haystack (NIAH) rule-based resources server that grades a
response on two signals at once:

1. **Answer correctness** (`answer_score`) — the final answer (the assistant's
   `output_text`) is parsed for a `Final Answer: [..]` node list and scored as F1
   against `expected_answer` (a JSON list of nodes).
2. **Reasoning/input overlap** — the model's reasoning (the `reasoning` item's
   summary text) should have *small* overlap with the input message, i.e. it should
   not just copy the prompt back into its chain of thought. Three independent
   overlap signals are computed (each in `[0, 1]`, higher = more copying):
   - `overlap_seq_match` — `difflib.SequenceMatcher` ratio (global similarity)
   - `overlap_ngram16` — fraction of the reasoning's 16-grams that appear in the input
   - `overlap_lcs` — longest common substring length / reasoning length

These signals are combined into a single `reasoning_overlap` penalty in `verify()`
(default: `max(...)` — the most conservative; tweak the combination there), then
gated by the answer:

```
reasoning_overlap = max(overlap_seq_match, overlap_ngram16, overlap_lcs)
reward            = answer_score * (1 - reasoning_overlap)
```

- A wrong answer scores `0` regardless of reasoning.
- A correct answer scores `1 - reasoning_overlap`, so verbatim copying of the prompt
  into the reasoning is penalized even when the answer is right.

The verify response exposes `answer_score`, the three `overlap_*` signals, and the
combined `reasoning_overlap` for inspection.

## Config

See `configs/lc_niah.yaml`.

## Dataset

Each JSONL row needs `responses_create_params.input` (the prompt) and an
`expected_answer` (passed to the verifier). See `data/example.jsonl`.

# Licensing information
Code: Apache-2.0
Data: example data is synthetic / illustrative.

Dependencies
- nemo_gym: Apache 2.0
