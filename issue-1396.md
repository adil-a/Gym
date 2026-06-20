# NeMo-Gym issue #1396 — epic: external agent integration

_Extracted from https://github.com/NVIDIA-NeMo/Gym/issues/1396 (opened by cwing-nvidia, 2026-05-22; assignee ananthsub). This is a model-extracted summary — treat the wording as approximate, verify specifics against the live issue if a claim is load-bearing._

## Goal
Make external-agent integration **spectrum-based** — from zero-code config-only hookup to deep framework integration — so teams get value immediately and deepen integration incrementally. Today, integrating requires heavy ceremony: FastAPI wrapper servers, subclassing `SimpleResponsesAPIAgent`, understanding internal plumbing (`ServerClient`, cookie propagation, `NeMoGymResponse` schema, session middleware), Hydra/OmegaConf YAML, and running wrappers inside Gym's managed process tree via `ng_run`.

## Customer-requested capabilities / 12 user considerations (PRIMARY scoring rubric)
- **R1** Agent code runs in an EXTERNAL repo, outside Gym's directory tree.
- **R2** Access Gym's verification layer (`/verify`) WITHOUT re-implementing agent abstractions / wrapping the whole loop.
- **R3** Support custom output formats beyond Messages/Chat (Messages, Chat, custom).
- **R4** Fast-path validation tool / conformance test suite to test integration before scaling.
- **R5** Clean HTTP contract — versioned `/run` request/response with examples, no requirement to adopt Gym's internal libraries.
- **R6** External harnesses can reach Gym's model servers (`/v1/responses`) with URL discovery + auth.
- **R7** Output compatible with Gym's metric aggregation + downstream TRAINING pipelines (implies token-level data).
- **R8** Simple configuration (endpoint URL) without the Hydra/OmegaConf learning curve.
- **R9** Stateful-harness support: clear session mapping OR opt-out of cookie-based session management.
- **R10** Cross-network/cross-infra operation (different clusters, clouds, developer laptops) with timeout/retry/auth.
- **R11** Thousands of concurrent rollouts (4k–65k concurrent `/run`) with backpressure + capacity signaling.
- **R12** Incremental adoption path: progressive framework-integration steps.
- (also) consistent inference during eval vs production usage.

## Customer architecture signal (Siemens "Fuse Agent", from the provided slides)
- **R13** Bring-your-own external agentic client (Cursor, Claude Code, VS Code).
- **R14** Expose MCP **Tools** + Agent **Skills** (per-product MCP+Skills servers: Solido Characterizer/Analytics/Generator/Repair).
- **R15** Access-controlled, secure **AI Data Lake** / RAG context injection; multi-tenant; auto-discover skills/toolboxes.

## Status
Epic with multiple sub-issues ("0/20 of 2 issues completed" indicator); no linked PRs/branches at extraction time. No comment thread captured.
