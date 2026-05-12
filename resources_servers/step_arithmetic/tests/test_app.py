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
import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.openai_utils import NeMoGymResponseOutputText
from nemo_gym.server_utils import ServerClient
from resources_servers.step_arithmetic.app import (
    BinaryOpRequest,
    ExpectedStep,
    StepArithmeticResourcesServer,
    StepArithmeticResourcesServerConfig,
    StepArithmeticVerifyRequest,
    SubmitRequest,
)


def _function_call(name: str, args: dict, call_id: str = "c") -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        arguments=json.dumps(args), call_id=call_id, name=name
    )


def _function_call_output(call_id: str = "c", output: str = '{"result": 0}') -> NeMoGymFunctionCallOutput:
    return NeMoGymFunctionCallOutput(call_id=call_id, output=output)


def _message(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_response(output: list) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="test_model",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class TestApp:
    @pytest.fixture
    def config(self) -> StepArithmeticResourcesServerConfig:
        return StepArithmeticResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        )

    @pytest.fixture
    def server(self, config) -> StepArithmeticResourcesServer:
        return StepArithmeticResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    def _verify_request(self, response, expected_steps, expected_answer):
        return StepArithmeticVerifyRequest(
            problem_id=0,
            expression="(3+5)*2-1",
            expected_steps=[ExpectedStep(**s) for s in expected_steps],
            expected_answer=expected_answer,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=response,
        )

    def test_sanity(self, server) -> None:
        assert server is not None

    @pytest.mark.asyncio
    async def test_tools(self, server) -> None:
        assert (await server.add(BinaryOpRequest(a=3, b=5))).result == 8
        assert (await server.mul(BinaryOpRequest(a=4, b=2))).result == 8
        assert (await server.sub(BinaryOpRequest(a=10, b=7))).result == 3
        assert (await server.submit(SubmitRequest(answer=15))).success is True

    @pytest.mark.asyncio
    async def test_verify_perfect_trajectory(self, server) -> None:
        # (3+5)*2-1 = 15; all steps correct, submit correct.
        response = _make_response([
            _function_call("add", {"a": 3, "b": 5}),
            _function_call_output(),
            _function_call("mul", {"a": 8, "b": 2}),
            _function_call_output(),
            _function_call("sub", {"a": 16, "b": 1}),
            _function_call_output(),
            _function_call("submit", {"answer": 15}),
        ])
        expected_steps = [
            {"op": "add", "a": 3, "b": 5},
            {"op": "mul", "a": 8, "b": 2},
            {"op": "sub", "a": 16, "b": 1},
        ]
        body = self._verify_request(response, expected_steps, 15.0)
        result = await server.verify(body)

        assert result.reward == 1.0
        assert result.step_rewards == [1.0, 1.0, 1.0, 1.0]
        assert result.correct_steps == 3
        assert result.total_expected_steps == 3
        assert result.final_answer == 15.0
        assert result.submitted is True

    @pytest.mark.asyncio
    async def test_verify_partial_trajectory(self, server) -> None:
        # First step correct, second wrong (used wrong operands), third uses wrong intermediate, submit wrong.
        response = _make_response([
            _function_call("add", {"a": 3, "b": 5}),       # correct → 1.0
            _function_call("mul", {"a": 7, "b": 2}),       # wrong operand → 0.0
            _function_call("sub", {"a": 14, "b": 1}),      # wrong (canonical was sub(16,1)) → 0.0
            _function_call("submit", {"answer": 13}),      # wrong final → 0.0
        ])
        expected_steps = [
            {"op": "add", "a": 3, "b": 5},
            {"op": "mul", "a": 8, "b": 2},
            {"op": "sub", "a": 16, "b": 1},
        ]
        body = self._verify_request(response, expected_steps, 15.0)
        result = await server.verify(body)

        assert result.reward == 0.0
        assert result.step_rewards == [1.0, 0.0, 0.0, 0.0]
        assert result.correct_steps == 1
        assert result.final_answer == 13.0
        assert result.submitted is True

    @pytest.mark.asyncio
    async def test_verify_no_submit(self, server) -> None:
        response = _make_response([
            _function_call("add", {"a": 3, "b": 5}),
            _function_call("mul", {"a": 8, "b": 2}),
            _function_call("sub", {"a": 16, "b": 1}),
        ])
        expected_steps = [
            {"op": "add", "a": 3, "b": 5},
            {"op": "mul", "a": 8, "b": 2},
            {"op": "sub", "a": 16, "b": 1},
        ]
        body = self._verify_request(response, expected_steps, 15.0)
        result = await server.verify(body)

        assert result.reward == 0.0
        assert result.submitted is False
        assert result.final_answer is None
        assert result.step_rewards == [1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_verify_extra_op_calls(self, server) -> None:
        # Agent issues more ops than canonical sequence has; extras score 0.
        response = _make_response([
            _function_call("add", {"a": 3, "b": 5}),
            _function_call("mul", {"a": 8, "b": 2}),
            _function_call("sub", {"a": 16, "b": 1}),
            _function_call("add", {"a": 15, "b": 0}),  # extra
            _function_call("submit", {"answer": 15}),
        ])
        expected_steps = [
            {"op": "add", "a": 3, "b": 5},
            {"op": "mul", "a": 8, "b": 2},
            {"op": "sub", "a": 16, "b": 1},
        ]
        body = self._verify_request(response, expected_steps, 15.0)
        result = await server.verify(body)

        assert result.step_rewards == [1.0, 1.0, 1.0, 0.0, 1.0]
        assert result.correct_steps == 3
        assert result.reward == 1.0  # final answer still correct

    @pytest.mark.asyncio
    async def test_verify_message_output_aligns(self, server) -> None:
        # An assistant text message in the middle should produce a 0 step_reward
        # without consuming an expected_step slot.
        response = _make_response([
            _function_call("add", {"a": 3, "b": 5}),
            _message("Let me think..."),
            _function_call("mul", {"a": 8, "b": 2}),
            _function_call("sub", {"a": 16, "b": 1}),
            _function_call("submit", {"answer": 15}),
        ])
        expected_steps = [
            {"op": "add", "a": 3, "b": 5},
            {"op": "mul", "a": 8, "b": 2},
            {"op": "sub", "a": 16, "b": 1},
        ]
        body = self._verify_request(response, expected_steps, 15.0)
        result = await server.verify(body)

        # step_rewards length == count of model-generated outputs (4 fn_calls + 1 msg = 5)
        assert result.step_rewards == [1.0, 0.0, 1.0, 1.0, 1.0]
        assert result.correct_steps == 3

    @pytest.mark.asyncio
    async def test_verify_bad_args_and_unknown_op(self, server) -> None:
        response = _make_response([
            NeMoGymResponseFunctionToolCall(
                arguments="not-json", call_id="c", name="add"
            ),
            _function_call("divide", {"a": 8, "b": 2}),     # unknown op
            _function_call("mul", {"a": 8, "b": 2}),        # also wrong slot now (idx=2 expects sub)
        ])
        expected_steps = [
            {"op": "add", "a": 3, "b": 5},
            {"op": "mul", "a": 8, "b": 2},
            {"op": "sub", "a": 16, "b": 1},
        ]
        body = self._verify_request(response, expected_steps, 15.0)
        result = await server.verify(body)

        # Bad args → 0 (advances idx). Unknown op → 0 (no idx advance). Mul at idx=1 → match → 1.0.
        assert result.step_rewards == [0.0, 0.0, 1.0]
        assert result.reward == 0.0
