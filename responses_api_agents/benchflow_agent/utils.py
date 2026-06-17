# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Helpers for converting BenchFlow rollout results into NeMo Gym responses."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)


@dataclass
class BenchFlowAgentUtils:
    @staticmethod
    def get_default_response_object() -> Dict[str, Any]:
        return {
            "id": f"resp_{str(uuid4())}",
            "created_at": int(time.time()),
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "background": False,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": {
                "effort": None,
                "generate_summary": None,
                "summary": None,
            },
            "service_tier": "default",
            "status": "completed",
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "top_logprobs": 0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 0,
            },
            "user": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
            "store": True,
        }

    @staticmethod
    def extract_reward(rewards: Optional[Dict[str, Any]]) -> float:
        """
        Extracts the scalar reward from a BenchFlow `RolloutResult.rewards` dict.
        Returns 0.0 if the reward is missing or invalid.
        """
        if not isinstance(rewards, dict):
            return 0.0
        reward = rewards.get("reward")
        if not isinstance(reward, (int, float)):
            return 0.0
        return float(reward)

    @staticmethod
    def extract_usage(result: Any) -> Dict[str, Any]:
        """Builds the `usage` dict from a BenchFlow `RolloutResult`'s token counts."""
        input_tokens = getattr(result, "n_input_tokens", None) or 0
        output_tokens = getattr(result, "n_output_tokens", None) or 0
        cached_tokens = getattr(result, "n_cache_read_tokens", None) or 0
        total_tokens = getattr(result, "total_tokens", None) or (input_tokens + output_tokens)
        return {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": total_tokens,
        }

    @staticmethod
    def trajectory_to_output(trajectory: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Converts BenchFlow's ACP trajectory events into NeMo Gym output items.

        BenchFlow captures a flat, ordered list of ACP events (see the fork's `trajectories/_capture.py`).
        Each event is one of:
          - `{"type": "agent_message"|"agent_thought"|"user_message", "text": str}`
          - `{"type": "tool_call", "tool_call_id", "kind", "title", "status", "content"}`

        This is a best-effort, eval-only conversion (no token ids / logprobs):
          - `agent_message`  -> assistant message
          - `agent_thought`  -> assistant message wrapped in `<think>...</think>`
          - `tool_call`      -> `function_call` (+ `function_call_output` when `content` is present)
        `user_message` and unknown event types are skipped (the task instruction lives inside the container, not in the request).
        Returns `[]` when there is no usable trajectory.
        """
        output_items: List[Dict[str, Any]] = []
        if not isinstance(trajectory, list):
            return output_items

        for event in trajectory:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")

            if event_type in ("agent_message", "agent_thought"):
                text = event.get("text") or ""
                if event_type == "agent_thought":
                    text = f"<think>{text}</think>"
                message = NeMoGymResponseOutputMessage(
                    id=f"cht_{uuid4().hex[:12]}",
                    content=[
                        NeMoGymResponseOutputText(
                            annotations=[],
                            text=text,
                            type="output_text",
                            logprobs=None,
                        ),
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
                output_items.append(message.model_dump())

            elif event_type == "tool_call":
                call_id = event.get("tool_call_id") or f"call_{uuid4().hex[:8]}"
                name = event.get("kind") or event.get("title") or "tool"
                arguments = {"title": event.get("title", ""), "status": event.get("status", "")}
                function_call = NeMoGymResponseFunctionToolCall(
                    arguments=json.dumps(arguments),
                    call_id=call_id,
                    name=str(name),
                    type="function_call",
                    id=f"fc_{uuid4().hex[:8]}",
                    status="completed",
                )
                output_items.append(function_call.model_dump())

                content = event.get("content")
                if content:
                    output = content if isinstance(content, str) else json.dumps(content, default=str)
                    function_call_output = NeMoGymFunctionCallOutput(
                        call_id=call_id,
                        output=output,
                        type="function_call_output",
                        id=f"fco_{uuid4().hex[:8]}",
                        status="completed",
                    )
                    output_items.append(function_call_output.model_dump())

        return output_items

    @staticmethod
    def apply_task_config_overrides(task_dir: Path, overrides: Dict[str, Any]) -> None:
        """Deep-merges `overrides` into the task's `task.md` YAML frontmatter, in place."""
        if not overrides:
            return
        import yaml

        md_path = Path(task_dir) / "task.md"
        if not md_path.is_file():
            raise FileNotFoundError(
                f"apply_task_config_overrides supports only task.md, none found in {task_dir}"
            )

        frontmatter, body = BenchFlowAgentUtils._split_frontmatter(md_path.read_text(encoding="utf-8"))
        merged = BenchFlowAgentUtils._deep_merge(yaml.safe_load(frontmatter) or {}, overrides)
        new_frontmatter = yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)
        md_path.write_text(f"---\n{new_frontmatter}---\n{body}", encoding="utf-8")

    @staticmethod
    def _split_frontmatter(text: str) -> Tuple[str, str]:
        """Splits a `task.md` document into `(frontmatter_yaml, body)`.
        Returns `("", text)` when there is no leading `---` frontmatter fence.
        """
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            return "", text
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "".join(lines[1:i]), "".join(lines[i + 1 :])
        return "", text

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges `overrides` into `base` and returns a new dict.

        Nested dicts merge key-by-key; any non-dict value (including lists) in
        `overrides` replaces the corresponding value in `base`.
        """
        merged = dict(base)
        for key, value in overrides.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = BenchFlowAgentUtils._deep_merge(existing, value)
            else:
                merged[key] = value
        return merged
