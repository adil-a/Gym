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
"""Console-script launcher for the rollout viewer Streamlit app.

The viewer itself lives under ``tools/rollout_viewer/`` (outside the packaged
``nemo_gym`` modules) and depends on the optional ``viewer`` extra. This thin
launcher checks that the extra is installed, then hands off to ``streamlit run``.
"""

from __future__ import annotations

import sys
from pathlib import Path


_INSTALL_HINT = (
    "The rollout viewer requires the optional 'viewer' extra.\n"
    "Install it with:\n"
    "    uv sync --extra viewer\n"
    "or\n"
    '    pip install -e ".[viewer]"'
)


def _app_path() -> Path:
    """Resolve the path to the viewer's Streamlit entry script."""
    return Path(__file__).resolve().parents[1] / "tools" / "rollout_viewer" / "app.py"


def view_rollouts() -> None:
    """Entry point for the ``ng_view_rollouts`` console script.

    Forwards any CLI args (e.g. ``--dir``, ``--rollouts``) to the app after
    Streamlit's ``--`` separator.
    """
    try:
        from streamlit.web import cli as stcli  # noqa: F401
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        raise SystemExit(1)

    app = _app_path()
    if not app.exists():
        print(f"Viewer app not found at {app}", file=sys.stderr)
        raise SystemExit(1)

    # Rebuild argv as: streamlit run <app> -- <user args>
    sys.argv = ["streamlit", "run", str(app), "--", *sys.argv[1:]]
    from streamlit.web import cli

    sys.exit(cli.main())
