# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a Jupyter notebook (.ipynb) to JSON for the Fern NotebookViewer component.

Output schema (matches NotebookViewerProps in fern/components/NotebookViewer.tsx):

  {
    "cells": [
      {"type": "markdown", "source": "..."},
      {"type": "code", "source": "...", "language": "python", "source_html": "<span ...>",
       "outputs": [{"type": "text", "data": "..."},
                   {"type": "image", "data": "<base64>"},
                   {"type": "text", "format": "html", "data": "<table ...>"}]}
    ]
  }

If Pygments is installed, code cells are pre-rendered to syntax-highlighted HTML
(stored in `source_html`) so the component can drop them in without a client-side
highlighter pass. Otherwise, code is plain text and the component escapes it.

Usage:
  python3 scripts/ipynb_to_fern_json.py <input.ipynb> <output.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from pygments import highlight
    from pygments.formatters.html import HtmlFormatter
    from pygments.lexers import get_lexer_by_name
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False


def render_code_html(source: str, language: str) -> str | None:
    if not HAS_PYGMENTS:
        return None
    try:
        lexer = get_lexer_by_name(language, stripall=True)
    except Exception:
        return None
    formatter = HtmlFormatter(nowrap=True, noclasses=False)
    return highlight(source, lexer, formatter)


def convert_outputs(outputs: list[dict]) -> list[dict]:
    """Convert nbformat outputs to the NotebookViewer's CellOutput schema."""
    result: list[dict] = []
    for out in outputs:
        otype = out.get("output_type")
        if otype == "stream":
            text = "".join(out.get("text", []))
            if text.strip():
                result.append({"type": "text", "data": text})
        elif otype in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "image/png" in data:
                img = data["image/png"]
                if isinstance(img, list):
                    img = "".join(img)
                result.append({"type": "image", "data": img.strip()})
            elif "text/html" in data:
                html = data["text/html"]
                if isinstance(html, list):
                    html = "".join(html)
                result.append({"type": "text", "format": "html", "data": html})
            elif "text/plain" in data:
                text = data["text/plain"]
                if isinstance(text, list):
                    text = "".join(text)
                if text.strip():
                    result.append({"type": "text", "data": text})
        elif otype == "error":
            tb = "\n".join(out.get("traceback", []))
            if tb:
                result.append({"type": "text", "data": tb})
    return result


def convert(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text())
    cells: list[dict] = []
    default_language = (
        nb.get("metadata", {}).get("kernelspec", {}).get("language")
        or nb.get("metadata", {}).get("language_info", {}).get("name")
        or "python"
    )

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if ctype == "markdown":
            if source.strip():
                cells.append({"type": "markdown", "source": source})
        elif ctype == "code":
            entry: dict = {
                "type": "code",
                "source": source,
                "language": default_language,
            }
            html = render_code_html(source, default_language)
            if html:
                entry["source_html"] = html
            outs = convert_outputs(cell.get("outputs", []))
            if outs:
                entry["outputs"] = outs
            cells.append(entry)
        # Skip raw / unknown cell types

    return {"cells": cells}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .ipynb file")
    parser.add_argument(
        "output",
        type=Path,
        help="Output file. Use .ts to emit a TypeScript module with default export "
        "(works with Fern's MDX import pipeline). Use .json for raw JSON.",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = convert(args.input)

    if args.output.suffix == ".ts":
        ts = (
            f"// Generated from {args.input}\n"
            "// Regenerate with: python3 scripts/ipynb_to_fern_json.py "
            f"{args.input} {args.output}\n\n"
            'import type { NotebookData } from "../NotebookViewer";\n\n'
            "const notebook: NotebookData = "
            + json.dumps(data, indent=2)
            + ";\n\nexport default notebook;\n"
        )
        args.output.write_text(ts)
    else:
        args.output.write_text(json.dumps(data, indent=2) + "\n")

    n_md = sum(1 for c in data["cells"] if c["type"] == "markdown")
    n_code = sum(1 for c in data["cells"] if c["type"] == "code")
    pygments = "with Pygments" if HAS_PYGMENTS else "without Pygments (no syntax highlighting)"
    print(f"{args.input} → {args.output}: {n_md} markdown + {n_code} code cells {pygments}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
