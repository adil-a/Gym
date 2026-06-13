import argparse
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_TOOLKIT = REPO_ROOT / ".claude/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py"
CODEX_TOOLKIT = REPO_ROOT / ".codex/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py"


def load_blade_toolkit():
    spec = importlib.util.spec_from_file_location("blade_toolkit", CLAUDE_TOOLKIT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blade_toolkit_copies_stay_in_sync():
    assert CLAUDE_TOOLKIT.read_text() == CODEX_TOOLKIT.read_text()


def test_make_shallow_keeps_only_high_level_metric_tables(tmp_path):
    toolkit = load_blade_toolkit()
    source = tmp_path / "golden_report.md"
    output = tmp_path / "shallow.md"
    source.write_text(
        """# Example BLADE Report

## 1. Aggregate Metrics

| Model | Pass@1 |
|---|---:|
| strong | 80.0% |

### Task-Level Aggregate Detail

| Task | Evidence |
|---|---|
| task_nested_leak | should not appear |

```text
code block with task_nested_leak and diagnostic evidence
```

- Diagnostic bullet with task_nested_leak should be dropped.

## Workflow Funnel and Phase Distribution

| Phase | Count |
|---|---:|
| scored | 10 |

## Dominant Failure Modes

| Failure | Count |
|---|---:|
| wrong_tool | 7 |

### task_123

| Task | Tool |
|---|---|
| task_123 | createSecretEvidence |
""",
    )

    toolkit.cmd_make_shallow(argparse.Namespace(input=str(source), output=str(output)))

    shallow = output.read_text()
    assert "| Model | Pass@1 |" in shallow
    assert "| Phase | Count |" in shallow
    assert "## Dominant Failure Modes" in shallow
    assert "wrong_tool" not in shallow
    assert "task_nested_leak" not in shallow
    assert "Diagnostic bullet" not in shallow
    assert "code block with task_nested_leak" not in shallow
    assert "task_123" not in shallow
    assert "createSecretEvidence" not in shallow
