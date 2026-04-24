(skill-eval-harness)=

# Evaluating Agent Skills, Part 1: Build the Harness

Build a NeMo Gym benchmark that grades **agent skills** — the `SKILL.md`-style prompts shipped in `.claude/skills/`. This tutorial walks through the three Gym servers that compose the harness and how they orchestrate a full rollout: seed a scoped workspace, let the model drive tools, judge the transcript, clean up.

By the end, you will:

- Understand the microservice boundary between the **workspace**, **judge**, and **orchestrator** servers.
- Scaffold and wire all three via Hydra so `ng_run` brings the stack up.
- Know the five integration gotchas that cost us a full afternoon of debugging.

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

< Back to Building Environments
:::

:::{tip}
Part 2 of this tutorial ({ref}`skill-eval-scoreboard`) covers **running** the harness: writing `evals.json`, the 2×2 cell methodology, reading the scoreboard, and judge calibration. This part is the infrastructure; Part 2 is the methodology.
:::

---

## What we're building

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
  J[JSONL] --> A[skill_eval_agent<br/>orchestrator]
  A -->|/seed_session + /run_bash + /read_file + /close| W[skill_workspace<br/>tools]
  A -->|/v1/responses| M[policy_model<br/>inference]
  A -->|/verify| JG[skill_judge<br/>grader]
  JG -->|/v1/responses| M
  JG --> A
  A --> R[rollout JSONL<br/>reward + grades]
```

Four servers cooperate per rollout:

- **`skill_workspace`** (resources server) — an ephemeral per-session tmpdir seeded with the skill's `scripts/` and `references/` (gated by per-request flags) plus scenario fixtures. Exposes `run_bash` and `read_file` tools to the policy model. **`SKILL.md` is never seeded** — see the contamination gotcha below.
- **`skill_judge`** (resources server) — LLM-as-judge. Receives the final model transcript plus the captured tool-call log and returns per-assertion binary grades with evidence.
- **`policy_model`** (model server) — the model being evaluated.
- **`skill_eval_agent`** (responses API agent) — orchestrates everything: seeds a workspace, runs the rollout loop, dispatches tool calls, forwards the transcript to the judge, closes the workspace.

Data flow: JSONL → `/run` → (model ↔ workspace tools loop) → `/verify` → reward + per-assertion grades → output JSONL.

---

## Prerequisites

- {ref}`core-components` — Model, Resources, and Agent servers
- {ref}`configuration-concepts` — Hydra composition
- {ref}`llm-as-judge-verification` — the judge pattern we build on

---

## Step 1: Scaffold three server directories

```
resources_servers/skill_workspace/
├── app.py
├── schemas.py
├── configs/skill_workspace.yaml
├── tests/test_app.py
└── requirements.txt

resources_servers/skill_judge/
├── app.py
├── schemas.py
├── configs/skill_judge.yaml
├── prompt_templates/skill_judge.txt
├── tests/test_app.py
└── requirements.txt

responses_api_agents/skill_eval_agent/
├── app.py
├── schemas.py
├── configs/skill_eval_agent.yaml
├── data/example.jsonl
├── tests/test_app.py
└── requirements.txt
```

Each server extends the appropriate base class:

```python
# resources_servers/skill_workspace/app.py
from nemo_gym.base_resources_server import SimpleResourcesServer

class SkillWorkspaceResourcesServer(SimpleResourcesServer):
    ...

# responses_api_agents/skill_eval_agent/app.py
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

class SkillEvalAgent(SimpleResponsesAPIAgent):
    ...
```

---

## Step 2: The workspace — tools-as-a-service

The workspace server exposes four endpoints: one to seed a session, two tools, one to tear down.

### `/seed_session`

Accepts `skill_path`, `scenario_id`, `files`, and two independent gating flags (`with_references`, `with_scripts`). Creates a fresh tmpdir, optionally copies `scripts/` and `references/`, copies the requested fixture files, and returns an `env_id` the agent will use for subsequent tool calls.

```python
async def seed_session(self, body: SkillWorkspaceSeedSessionRequest) -> SkillWorkspaceSeedSessionResponse:
    skill_src = Path(body.skill_path).resolve()
    workspace = Path(tempfile.mkdtemp(
        prefix=f"skill_ws_{body.scenario_id}_",
        dir=self.config.workspace_root,
    ))
    env_id = str(uuid.uuid4())
    # SKILL.md intentionally NOT seeded — see "Gotcha: workspace contamination" below.
    seed_subdirs = []
    if body.with_scripts:
        seed_subdirs.append("scripts")
    if body.with_references:
        seed_subdirs.append("references")
    for subdir in seed_subdirs:
        src = skill_src / subdir
        if src.is_dir():
            shutil.copytree(src, workspace / subdir)
    for rel in body.files:
        shutil.copy2(skill_src / rel, workspace / rel)
    self.env_id_to_workspace[env_id] = workspace
    return SkillWorkspaceSeedSessionResponse(env_id=env_id)
```

### Gotcha: workspace contamination

Anything you seed into the workspace is readable by the model through its tools — that's the whole point. But if the artifact is *about the skill being evaluated*, the control arm ("no skill") is no longer a clean control: the first thing the model does in every rollout is `ls && cat SKILL.md`, and if that file exists it happily grabs the skill body off disk.

We learned this the hard way. Pre-fix, 100% of without-skill rollouts were reading `SKILL.md` off disk. After we removed `SKILL.md`, 100% of without-skill `gym-profile` rollouts fell back to reading `references/metrics-guide.md`, which (being the skill author's carefully-written reference doc) contains every noun the assertions test for. Both findings produced the same prescription: gate the seeding of skill-payload artifacts on explicit flags, and default the "without skill" arm to a true blind state.

That's what `with_references` / `with_scripts` are for. The skill-eval harness exercises them as part of its 2×2 grid (Part 2); other callers can set either to `False` when they want a stricter control.

### `/run_bash` and `/read_file`

Both resolve paths against the workspace root (path-escape check!), bound concurrency with `asyncio.Semaphore`, cap output size, and decode with `errors="replace"` to survive non-UTF-8 output.

```python
async def run_bash(self, body: RunBashRequest) -> RunBashResponse:
    workspace = self.env_id_to_workspace.get(body.env_id)
    async with self._get_bash_semaphore():
        proc = await asyncio.create_subprocess_shell(
            body.cmd, cwd=str(workspace),
            env=_build_sandbox_env(workspace),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    # … cap and decode …
```

`_build_sandbox_env` strips the host's NeMo Gym venv from `PATH` — the bash subprocess sees only `/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin` plus a minimal set of locale/terminal variables. The model-under-test can't accidentally discover host `ng_*` binaries, Ray state, or HF/MLflow credentials that Uvicorn's parent environment inherited. A workspace-local `.sandbox_bin/python → python3` symlink covers the common case of prompts that assume `python` exists (which macOS sandboxes otherwise don't expose).

### `/close`

Removes the tmpdir. The orchestrator calls this in a `finally` block so workspaces never leak.

### Why ephemeral?

Each rollout gets its own isolated filesystem. No state leaks between scenarios; concurrent rollouts can't step on each other. Under 4-way parallelism with 48 scenarios we saw zero flakes.

---

## Step 3: The judge — per-assertion LLM grading

A single judge call grades a list of assertions. Binary per-assertion `satisfied: bool` with short `evidence` is much easier to debug than a single scalar rubric.

```python
# resources_servers/skill_judge/app.py
async def _call_judge(self, req: JudgeRequest) -> JudgeResponse:
    user_prompt = self._judge_prompt_template.format(
        prompt=req.prompt,
        expected_output=req.expected_output or "(not provided)",
        response=req.response,
        tool_calls=_format_tool_calls(req.tool_calls),
        assertions=_format_assertions(req.assertions),
        n=len(req.assertions),
    )
    raw = await self.server_client.post(
        server_name=self.config.judge_model_server.name,
        url_path="/v1/responses",
        json=create_params,
    )
    judge_text = _last_assistant_text(NeMoGymResponse.model_validate(await get_response_json(raw)))
    raw_grades = _extract_json_array(judge_text)
    grades = _normalize_grades(raw_grades, len(req.assertions))
    return JudgeResponse(grades=grades, reward=sum(g.satisfied for g in grades)/len(grades))
```

Key design choices:

- **JSON array output, with code-fence fallback parsing** (`_extract_json_array`) — models love to wrap output in ` ```json `.
- **`_normalize_grades` projects onto exactly N grades by id** — a judge that returns only 3 of 5 gets zero-filled with `"judge did not return a grade"` evidence, not a crash.
- **`/verify` forwards tool calls into the judge prompt** — the judge sees what the agent *did*, not just what it said.

See the [existing `llm-as-judge-verification` tutorial](llm-as-judge-verification) for the broader pattern.

---

## Step 4: The orchestrator agent

This is where it all comes together. `/run` has five jobs:

1. Seed a workspace (get `env_id`), forwarding the `with_references` / `with_scripts` flags from `verifier_metadata`.
2. Optionally prepend `SKILL.md` to the input, gated by `verifier_metadata.with_skill`.
3. Run the rollout loop: model generates → dispatch tool calls → feed outputs back → repeat until done or `max_steps`.
4. Forward the transcript + captured `tool_calls` to the judge's `/verify`.
5. `/close` the workspace — always, even on exceptions.

The two flags (`with_skill`, `with_references`) are what let the harness produce the 2×2 control structure described in Part 2.

```python
# responses_api_agents/skill_eval_agent/app.py
async def run(self, request, body: SkillEvalAgentRunRequest) -> SkillEvalAgentVerifyResponse:
    metadata = dict(body.verifier_metadata or {})
    with_skill = bool(metadata.get("with_skill", False))
    # Both flags default to with_skill for back-compat with pre-2×2 JSONLs.
    with_references = bool(metadata.get("with_references", with_skill))
    with_scripts = bool(metadata.get("with_scripts", with_skill))
    seed_resp = await self.server_client.post(
        server_name=self.config.workspace_server.name,
        url_path="/seed_session",
        json={"skill_path": metadata["skill_path"],
              "scenario_id": metadata["scenario_id"],
              "files": list(metadata.get("files") or []),
              "with_references": with_references,
              "with_scripts": with_scripts},
    )
    env_id = (await get_response_json(seed_resp))["env_id"]

    try:
        params = body.responses_create_params.model_copy(deep=True)
        if with_skill and metadata.get("skill_md"):
            params.input.insert(0, NeMoGymEasyInputMessage(
                role="system",
                content=f"{_SKILL_SYSTEM_PREFIX}{metadata['skill_md']}",
            ))
        if self.config.inject_tools and not params.tools:
            params.tools = _TOOLS_SCHEMA

        model_response, tool_calls = await self._rollout_loop(params, env_id)

        verify_metadata = dict(metadata)
        verify_metadata["tool_calls"] = tool_calls
        judge_resp = await self.server_client.post(
            server_name=self.config.judge_server.name,
            url_path="/verify",
            json={"responses_create_params": params.model_dump(mode="json"),
                  "response": model_response.model_dump(mode="json"),
                  "verifier_metadata": verify_metadata},
        )
        return SkillEvalAgentVerifyResponse.model_validate(await get_response_json(judge_resp))
    finally:
        await self.server_client.post(
            server_name=self.config.workspace_server.name,
            url_path="/close",
            json={"env_id": env_id},
        )
```

The rollout loop is a vanilla tool-call loop — poll the model, dispatch function calls against the workspace, accumulate outputs until the model produces an assistant message with no tool calls (or we hit `max_steps`). See `_rollout_loop` and `_dispatch_tool_call` in `responses_api_agents/skill_eval_agent/app.py` for the full implementation.

:::{note}
`/v1/responses` on this agent stays a **pure proxy** — it has no `env_id` context and can't orchestrate anything. All the interesting logic lives in `/run`.
:::

---

## Step 5: Wire everything in Hydra

```yaml
# responses_api_agents/skill_eval_agent/configs/skill_eval_agent.yaml
skill_eval_agent:
  responses_api_agents:
    skill_eval_agent:
      entrypoint: app.py
      domain: other
      workspace_server: {type: resources_servers, name: skill_workspace}
      judge_server:     {type: resources_servers, name: skill_judge}
      model_server:     {type: responses_api_models, name: policy_model}
      max_steps: 8
      inject_tools: true
      datasets:
      - name: skill_eval
        type: example
        jsonl_fpath: responses_api_agents/skill_eval_agent/data/example.jsonl
```

Credentials for the policy model go in `env.yaml` at the repo root:

```yaml
policy_base_url: https://your-endpoint/v1
policy_api_key: your-key
policy_model_name: your-model
```

Bring the stack up:

```bash
ng_run "+config_paths=[
  resources_servers/skill_workspace/configs/skill_workspace.yaml,
  resources_servers/skill_judge/configs/skill_judge.yaml,
  responses_api_models/openai_model/configs/openai_model.yaml,
  responses_api_agents/skill_eval_agent/configs/skill_eval_agent.yaml
]" +skip_venv_if_present=true

ng_status  # should show 4 healthy
```

---

## The five gotchas

These each cost real time the first time we ran the stack live. Keep them handy.

**1. `ng_run` runs `python app.py`, not `python -m`.**
Relative imports (`from .schemas import X`) work in unit tests but blow up under `ng_run` with `ImportError: attempted relative import with no known parent package`. Use absolute imports:

```python
# bad
from .schemas import SkillWorkspaceSeedSessionRequest
# good — works because nemo_gym/__init__.py adds project root to sys.path
from resources_servers.skill_workspace.schemas import SkillWorkspaceSeedSessionRequest
```

**2. Trailing slashes in `policy_base_url` produce 404s.**
`https://endpoint/v1/` + `/responses` → `https://endpoint/v1//responses`. Some providers 404 on the double slash. Strip the trailing slash in `env.yaml`.

**3. Some providers return `object: "chat.completion"` on `/v1/responses`.**
`NeMoGymResponse` requires `object: "response"` and raises a Pydantic literal error. Normalize in your model server's `responses()`:

```python
openai_response_dict = await self._client.create_response(**body_dict)
if isinstance(openai_response_dict, dict) and openai_response_dict.get("object") != "response":
    openai_response_dict["object"] = "response"
return NeMoGymResponse.model_validate(openai_response_dict)
```

**4. OpenAI's `FunctionToolParam` requires `strict: Optional[bool]`.**
Tool schemas that omit `strict` fail validation at the model server. Add `"strict": False` to each function tool definition.

**5. Always `/close` in a `finally` block.**
Tool errors, judge failures, cancellation — any of these will leak tmpdirs if `/close` is inside the `try`. Workspace cleanup belongs in `finally`.

---

## Versioning: provenance in verifier_metadata

Every input-JSONL record carries five content hashes in `verifier_metadata`:

| field | hashes |
|---|---|
| `skill_md_sha` | `SKILL.md` |
| `evals_sha` | `evals/evals.json` |
| `fixtures_sha` | listed fixture files for this scenario |
| `judge_prompt_sha` | judge prompt template |
| `harness_version` | concatenated bytes of the three server `app.py` files |

All five are computed at JSONL-build time by `scripts/build_skill_eval_jsonl.py`, persisted through `verify()`, and emitted on the output rollout. Every scoreboard is self-describing: it tells you exactly which inputs it measured.

Why five and not one: "did the skill get worse?" is usually the wrong question. The right question is "which input change moved the delta?" Without field-level provenance, a harness edit and a skill edit look identical in the output. With it, `scripts/diff_skill_scoreboards.py` can tell you `md+evals` (both edited — stop and disambiguate), `harness` (stack change — expect correlated movement across all cells of every skill), or `same-all` (no tracked input changed — you're looking at noise or judge drift, not a skill effect). Part 2 walks through how this attribution shapes the 2×2-cell interpretation.

---

## Verify the stack

```bash
# 1. All four servers healthy
ng_status  # expect "4 healthy"

# 2. Policy model responds sanely
PORT=$(ng_status 2>&1 | grep -A 5 policy_model | grep port | head -1 | awk -F: '{print $2}' | tr -d ' ,')
curl -sS -X POST "http://127.0.0.1:$PORT/v1/responses" \
     -H 'Content-Type: application/json' \
     -d '{"input": [{"role": "user", "content": "hi"}], "max_output_tokens": 32}'
# expect: {"id": "resp_...", "object": "response", ...}

# 3. One smoke rollout
ng_collect_rollouts \
     +agent_name=skill_eval_agent \
     +input_jsonl_fpath=/tmp/one_scenario.jsonl \
     +output_jsonl_fpath=/tmp/smoke.jsonl \
     +num_repeats=1 \
     +num_samples_in_parallel=2 \
     "+responses_create_params={max_output_tokens: 8192}"
# expect: "Key metrics for skill_eval_agent: {mean/reward: ...}"
```

If all three pass, the harness is wired correctly.

---

## What's next

You have working infrastructure. Now you need **scenarios** (per-skill `evals.json`) and a **methodology** for turning rollouts into a scoreboard you can trust.

:::{button-ref} skill-eval-scoreboard
:color: primary
:ref-type: doc

Continue to Part 2: Running the Scoreboard →
:::
