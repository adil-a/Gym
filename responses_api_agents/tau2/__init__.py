# Vendored runtime patch for tau2-bench:
# vLLM >= 0.16.0 renamed the assistant-message response field
# `reasoning_content` -> `reasoning`. Upstream tau2-bench (pinned at
# e6e23241) only reads `reasoning_content`, so the cross-turn replay in
# `to_litellm_messages()` silently no-ops on modern vLLM. We override
# `tau2.utils.llm_utils.generate` with a copy that reads either field.
#
# Self-contained — no external fork needed; lives only in this Gym branch.

import importlib.util
import sys
from pathlib import Path

_PATCH_PATH = Path(__file__).parent / "_patches" / "llm_utils.py"
if _PATCH_PATH.exists():
    # Force-load the upstream module first so its dependencies (data_model,
    # ModelResponse, etc.) are wired in the normal way.
    import tau2.utils.llm_utils as _llm_utils

    _spec = importlib.util.spec_from_file_location(
        "tau2.utils.llm_utils_patched", _PATCH_PATH
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["tau2.utils.llm_utils_patched"] = _mod
    _spec.loader.exec_module(_mod)

    # Swap in the patched generate (+ to_litellm_messages just in case).
    for _name in ("generate", "to_litellm_messages"):
        if hasattr(_mod, _name):
            setattr(_llm_utils, _name, getattr(_mod, _name))
