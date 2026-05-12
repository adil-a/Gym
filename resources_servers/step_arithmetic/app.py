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
import math
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


_TOL = 1e-6
_BINARY_OPS = {"add", "mul", "sub"}


class StepArithmeticResourcesServerConfig(BaseResourcesServerConfig):
    pass


class BinaryOpRequest(BaseModel):
    a: float
    b: float


class BinaryOpResponse(BaseModel):
    result: float


class SubmitRequest(BaseModel):
    answer: float


class SubmitResponse(BaseModel):
    success: bool


class ExpectedStep(BaseModel):
    op: str
    a: float
    b: float


class StepArithmeticRunRequest(BaseRunRequest):
    problem_id: int
    expression: str
    expected_steps: List[ExpectedStep]
    expected_answer: float


class StepArithmeticVerifyRequest(StepArithmeticRunRequest, BaseVerifyRequest):
    pass


class StepArithmeticVerifyResponse(BaseVerifyResponse):
    step_rewards: List[float]
    correct_steps: int
    total_expected_steps: int
    final_answer: Optional[float]
    submitted: bool


class StepArithmeticResourcesServer(SimpleResourcesServer):
    config: StepArithmeticResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/add")(self.add)
        app.post("/mul")(self.mul)
        app.post("/sub")(self.sub)
        app.post("/submit")(self.submit)

        return app

    async def add(self, body: BinaryOpRequest) -> BinaryOpResponse:
        return BinaryOpResponse(result=body.a + body.b)

    async def mul(self, body: BinaryOpRequest) -> BinaryOpResponse:
        return BinaryOpResponse(result=body.a * body.b)

    async def sub(self, body: BinaryOpRequest) -> BinaryOpResponse:
        return BinaryOpResponse(result=body.a - body.b)

    async def submit(self, body: SubmitRequest) -> SubmitResponse:
        return SubmitResponse(success=True)

    async def verify(self, body: StepArithmeticVerifyRequest) -> StepArithmeticVerifyResponse:
        expected_steps = body.expected_steps
        expected_answer = body.expected_answer

        step_rewards: List[float] = []
        correct_steps = 0
        expected_idx = 0
        final_answer: Optional[float] = None
        submitted = False

        for output_item in body.response.output:
            # Skip tool responses (not model-generated; not in NeMo-RL message_log).
            if output_item.type == "function_call_output":
                continue

            if output_item.type == "message":
                step_rewards.append(0.0)
                continue

            if output_item.type != "function_call":
                step_rewards.append(0.0)
                continue

            try:
                args = json.loads(output_item.arguments)
            except (json.JSONDecodeError, TypeError):
                step_rewards.append(0.0)
                if output_item.name in _BINARY_OPS:
                    expected_idx += 1
                continue

            if output_item.name == "submit":
                ans = args.get("answer")
                final_answer = float(ans) if isinstance(ans, (int, float)) else None
                submitted = True
                ok = final_answer is not None and math.isclose(
                    final_answer, expected_answer, abs_tol=_TOL
                )
                step_rewards.append(1.0 if ok else 0.0)
                continue

            if output_item.name not in _BINARY_OPS:
                step_rewards.append(0.0)
                continue

            if expected_idx >= len(expected_steps):
                # Agent issued more ops than the canonical sequence has.
                step_rewards.append(0.0)
                expected_idx += 1
                continue

            exp = expected_steps[expected_idx]
            a = args.get("a")
            b = args.get("b")
            match = (
                output_item.name == exp.op
                and isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and math.isclose(float(a), exp.a, abs_tol=_TOL)
                and math.isclose(float(b), exp.b, abs_tol=_TOL)
            )
            step_rewards.append(1.0 if match else 0.0)
            if match:
                correct_steps += 1
            expected_idx += 1

        final_correct = (
            submitted
            and final_answer is not None
            and math.isclose(final_answer, expected_answer, abs_tol=_TOL)
        )
        return StepArithmeticVerifyResponse(
            **body.model_dump(),
            reward=float(final_correct),
            step_rewards=step_rewards,
            correct_steps=correct_steps,
            total_expected_steps=len(expected_steps),
            final_answer=final_answer,
            submitted=submitted,
        )


if __name__ == "__main__":
    StepArithmeticResourcesServer.run_webserver()
