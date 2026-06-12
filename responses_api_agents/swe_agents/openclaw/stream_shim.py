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

"""In-container streaming proxy: terminates OpenClaw's stream:true requests,
forwards stream:false to NeMo-Gym's vllm_model proxy, re-emits SSE, and logs
each (request, response) pair to JSONL for host-side trajectory reconstruction.

Run as a subprocess from run_openclaw.sh. Lifecycle ends with SIGTERM at which
point the JSONL is flushed."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web


class StreamShim:
    def __init__(
        self,
        *,
        upstream_base_url: str,
        port_file: str,
        pid_file: str,
        jsonl_log: str,
        max_turns: int,
        top_p: float | None = None,
    ):
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.port_file = port_file
        self.pid_file = pid_file
        self.jsonl_log_path = jsonl_log
        self.max_turns = max_turns
        # top_p is injected onto every forwarded request because OpenClaw's openai-responses
        # transport doesn't wire top_p (it only forwards temperature/max_output_tokens), so it
        # can't be set via openclaw.json. None => leave the agent's request untouched.
        self.top_p = top_p
        self.port: int | None = None
        self._runner: web.AppRunner | None = None
        self._jsonl_fp = None
        self._session: aiohttp.ClientSession | None = None
        self._turns_used = 0
        self._session_cookie: str = ""  # held outside aiohttp's jar to keep this fully under our control
        # Prior-turn prefix token IDs, re-attached to the next request's last assistant
        # item — see _inject_prefix_token_ids.
        self._last_prefix_token_ids: dict | None = None

    async def start(self) -> None:
        self._jsonl_fp = open(self.jsonl_log_path, "a", buffering=1)  # line-buffered
        self._session = aiohttp.ClientSession()
        app = web.Application()
        app.router.add_route("GET", "/v1/models", self._handle_passthrough)
        app.router.add_route("POST", "/v1/responses", self._handle_inference)
        app.router.add_route("POST", "/v1/chat/completions", self._handle_inference)
        app.router.add_route("*", "/{tail:.*}", self._handle_404)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host="127.0.0.1", port=0)
        await site.start()
        sock = next(iter(self._runner.sites))._server.sockets[0]
        self.port = sock.getsockname()[1]
        Path(self.port_file).write_text(str(self.port))
        Path(self.pid_file).write_text(str(os.getpid()))

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._jsonl_fp is not None:
            self._jsonl_fp.close()
            self._jsonl_fp = None

    def _cookie_for_outbound(self) -> str:
        return self._session_cookie

    def _absorb_set_cookie(self, set_cookie_header: str) -> None:
        # Expect a single Set-Cookie like "SESSION_ID=<val>; Path=/; …".
        # We only persist the SESSION_ID=… name=value pair.
        if not set_cookie_header:
            return
        first = set_cookie_header.split(";", 1)[0].strip()
        if first.startswith("SESSION_ID="):
            self._session_cookie = first

    def _inject_prefix_token_ids(self, body: dict) -> None:
        """Re-attach the previous turn's token IDs onto this request's last assistant item.

        OpenClaw's Pi agent re-serializes history to role/content/tool_calls and drops the
        prompt_token_ids/generation_token_ids/generation_log_probs that vllm_model attached to
        the previous response. Without them on the request, NeMo-RL's vLLM server cannot form
        required_prefix_token_ids and its on-policy token-ID splice no-ops — multi-turn GRPO
        then trains off-policy. We restore them on the last item of the prior assistant turn
        (the message/function_call just before the trailing tool result), mirroring where
        vllm_model placed them on the response. No-op until a token-id-bearing response is seen.
        """
        prefix = self._last_prefix_token_ids
        if prefix is None:
            return
        inp = body.get("input")
        if not isinstance(inp, list):
            return
        for item in reversed(inp):
            if not isinstance(item, dict):
                return
            if item.get("type") == "function_call_output":
                continue  # tool results are environment inputs, not the model's prior output
            itype = item.get("type")
            is_assistant_generation = (
                itype == "function_call"
                or itype == "reasoning"
                or ((itype == "message" or itype is None) and item.get("role") == "assistant")
            )
            if is_assistant_generation:
                item.update(prefix)
            return  # first non-tool-result item from the end is the prior turn's last item

    async def _handle_404(self, request: web.Request) -> web.Response:
        return web.Response(status=404, text=f"shim: {request.method} {request.path} not allowed")

    async def _handle_passthrough(self, request: web.Request) -> web.Response:
        # Append the relative tail after /v1
        url = f"{self.upstream_base_url}/{request.path[len('/v1/') :]}"
        async with self._session.get(url, headers=dict(request.headers)) as up:
            body_text = await up.text()
            return web.Response(status=up.status, text=body_text, content_type="application/json")

    async def _handle_inference(self, request: web.Request) -> web.StreamResponse:
        # No lock guards _turns_used / _session_cookie across the upstream await below:
        # this shim serves exactly one openclaw agent (Pi issues model calls serially,
        # one turn at a time) and each rollout gets its own shim process, so requests
        # never overlap. If that invariant ever changes, add an asyncio.Lock here.
        if self._turns_used >= self.max_turns:
            entry = {
                "turn": self._turns_used,
                "endpoint": request.path,
                "request": None,
                "response": None,
                "upstream_status": None,
                "started_at": time.time(),
                "ended_at": time.time(),
                "session_cookie_in": None,
                "session_cookie_out": None,
                "error": "max_iteration",
            }
            self._write_entry(entry)
            return web.Response(
                status=400,
                text=json.dumps(
                    {
                        "error": {
                            "type": "max_iteration",
                            "message": f"agent_max_turns ({self.max_turns}) exceeded",
                        }
                    }
                ),
                content_type="application/json",
            )
        body = await request.json()
        body.pop("stream", None)  # strip stream:true → forward non-streamed
        if self.top_p is not None:
            body["top_p"] = self.top_p  # inject top_p (not settable via openclaw.json)
        # Re-attach the prior turn's ground-truth token IDs so NeMo-RL's on-policy splice can fire.
        self._inject_prefix_token_ids(body)
        cookie_in = self._cookie_for_outbound()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        if cookie_in:
            headers["Cookie"] = cookie_in
        endpoint = request.path
        url = f"{self.upstream_base_url}/{endpoint[len('/v1/') :]}"
        started_at = time.time()
        entry = {
            "turn": self._turns_used,
            "endpoint": endpoint,
            "request": body,
            "response": None,
            "upstream_status": None,
            "started_at": started_at,
            "ended_at": None,
            "session_cookie_in": cookie_in,
            "session_cookie_out": None,
            "error": None,
        }
        try:
            async with self._session.post(url, json=body, headers=headers) as up:
                up_body = await up.json(content_type=None)
                entry["upstream_status"] = up.status
                entry["response"] = up_body
                entry["session_cookie_out"] = up.headers.get("Set-Cookie", "")
                self._absorb_set_cookie(up.headers.get("Set-Cookie", ""))
        except Exception as e:
            entry["error"] = repr(e)
            entry["ended_at"] = time.time()
            self._write_entry(entry)
            return web.Response(status=502, text=json.dumps({"error": {"type": "proxy_error", "message": repr(e)}}))
        finally:
            entry["ended_at"] = time.time()

        if not (200 <= up.status < 300):
            # Propagate the body and status verbatim; classification happens host-side.
            self._write_entry(entry)
            return web.Response(
                status=up.status,
                text=json.dumps(up_body) if isinstance(up_body, (dict, list)) else str(up_body),
                content_type="application/json",
            )

        self._turns_used += 1
        # Remember this turn's ground-truth token IDs for the next request's splice prefix.
        self._last_prefix_token_ids = _extract_prefix_token_ids(up_body)
        self._write_entry(entry)

        # Re-emit as SSE so OpenClaw's streaming client materialises the response.
        if endpoint == "/v1/responses":
            payload = _responses_sse_payload(up_body)
        else:  # /v1/chat/completions
            payload = f"data: {json.dumps(up_body)}\n\ndata: [DONE]\n\n"
        resp = web.StreamResponse(
            status=up.status,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        await resp.write(payload.encode("utf-8"))
        await resp.write_eof()
        return resp

    def _write_entry(self, entry: dict) -> None:
        self._jsonl_fp.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._jsonl_fp.flush()


_PREFIX_TOKEN_ID_KEYS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")


def _extract_prefix_token_ids(up_body: dict) -> dict | None:
    """Pull the token-id triple off the response's last token-id-bearing output item.

    vllm_model attaches the triple to a turn's LAST output item only (when
    return_token_id_information is on). Returns None when absent (eval mode, or an
    empty/context-overflow response), which leaves the next turn's injection a no-op."""
    if not isinstance(up_body, dict):
        return None
    output = up_body.get("output")
    if not isinstance(output, list):
        return None
    for item in reversed(output):
        if isinstance(item, dict) and all(k in item for k in _PREFIX_TOKEN_ID_KEYS):
            return {k: item[k] for k in _PREFIX_TOKEN_ID_KEYS}
    return None


def _responses_sse_payload(up_body: dict) -> str:
    """Synthesize the full Responses streaming event sequence from a complete
    (non-streamed) response body.

    OpenClaw's streaming client materialises output items from the granular
    events (``output_item.added``/``.done``, ``output_text.delta``,
    ``function_call_arguments.delta``/``.done``), not from the terminal
    ``response.completed`` snapshot. Emitting only created+completed left it
    with zero items ("payloads=0" → incomplete-turn FailoverError).
    """
    events: list[tuple[str, dict]] = []
    seq = 0

    def emit(event_type: str, data: dict) -> None:
        nonlocal seq
        events.append((event_type, {"type": event_type, "sequence_number": seq, **data}))
        seq += 1

    framing = {**up_body, "output": [], "status": "in_progress"}
    emit("response.created", {"response": framing})
    emit("response.in_progress", {"response": framing})

    for output_index, item in enumerate(up_body.get("output", [])):
        item_id = item.get("id", f"item_{output_index}")
        itype = item.get("type")
        if itype == "message":
            emit("response.output_item.added", {"output_index": output_index, "item": {**item, "content": []}})
            for content_index, part in enumerate(item.get("content", [])):
                if part.get("type") != "output_text":
                    continue  # our config only produces output_text parts
                text = part.get("text", "")
                base = {"item_id": item_id, "output_index": output_index, "content_index": content_index}
                emit(
                    "response.content_part.added",
                    {**base, "part": {"type": "output_text", "text": "", "annotations": part.get("annotations", [])}},
                )
                emit("response.output_text.delta", {**base, "delta": text})
                emit("response.output_text.done", {**base, "text": text})
                emit("response.content_part.done", {**base, "part": part})
            emit("response.output_item.done", {"output_index": output_index, "item": item})
        elif itype == "function_call":
            emit("response.output_item.added", {"output_index": output_index, "item": {**item, "arguments": ""}})
            args = item.get("arguments", "")
            base = {"item_id": item_id, "output_index": output_index}
            emit("response.function_call_arguments.delta", {**base, "delta": args})
            emit("response.function_call_arguments.done", {**base, "arguments": args})
            emit("response.output_item.done", {"output_index": output_index, "item": item})
        else:
            # Unknown item types (reasoning, function_call_output, …): surface as a
            # plain added/done pair so nothing is silently dropped from the stream.
            emit("response.output_item.added", {"output_index": output_index, "item": item})
            emit("response.output_item.done", {"output_index": output_index, "item": item})

    emit("response.completed", {"response": up_body})

    return "".join(f"event: {event_type}\ndata: {json.dumps(data)}\n\n" for event_type, data in events)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-base-url", required=True)
    p.add_argument("--port-file", required=True)
    p.add_argument("--pid-file", required=True)
    p.add_argument("--jsonl-log", required=True)
    p.add_argument("--max-turns", type=int, required=True)
    p.add_argument(
        "--top-p", default=None, help="nucleus sampling top_p to inject per request (empty/unset = leave untouched)"
    )
    return p


async def _main_async(args: argparse.Namespace) -> None:
    shim = StreamShim(
        upstream_base_url=args.upstream_base_url,
        port_file=args.port_file,
        pid_file=args.pid_file,
        jsonl_log=args.jsonl_log,
        max_turns=args.max_turns,
        top_p=float(args.top_p) if args.top_p else None,
    )
    stop_event = asyncio.Event()

    def _handle_signal():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    await shim.start()
    try:
        await stop_event.wait()
    finally:
        await shim.stop()


def main() -> int:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(_main_async(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
