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
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    MCPResourcesServer,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


class TestBaseResourcesServer:
    def test_sanity(self) -> None:
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")

        class TestSimpleResourcesServer(SimpleResourcesServer):
            async def verify(self, body):
                pass

        agent = TestSimpleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        agent.setup_webserver()


class TestMCPResourcesServer:
    def test_mounts_mcp_endpoint_with_normal_gym_endpoints(self) -> None:
        pytest.importorskip("mcp")
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                @mcp.tool()
                def ping() -> str:
                    return "pong"

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        paths = {getattr(route, "path", None) for route in app.routes}

        assert "/seed_session" in paths
        assert "/verify" in paths
        assert "/aggregate_metrics" in paths
        assert "/mcp" in paths

    def test_build_mcp_session_metadata_maps_token_to_session_id(self) -> None:
        pytest.importorskip("mcp")
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                pass

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))
        request = MagicMock(spec=Request)
        request.session = {SESSION_ID_KEY: "gym-session-1"}

        metadata = server.build_mcp_session_metadata(request)
        token = metadata.headers["X-NeMo-Gym-Session-Token"]

        assert metadata.server_name == "test_mcp_resources_server"
        assert metadata.url_path == "/mcp"
        assert server._mcp_session_id_by_token[token] == "gym-session-1"

    def test_missing_mcp_session_token_raises_401(self) -> None:
        pytest.importorskip("mcp")
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                pass

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))

        with pytest.raises(HTTPException) as exc_info:
            server.require_mcp_session_id()

        assert exc_info.value.status_code == 401

    def test_invalid_mcp_session_token_raises_401(self) -> None:
        pytest.importorskip("mcp")
        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="test_mcp_resources_server")

        class TestMCPServer(MCPResourcesServer):
            def register_mcp_tools(self, mcp):
                pass

            async def verify(self, body):
                pass

        server = TestMCPServer(config=config, server_client=MagicMock(spec=ServerClient))

        from nemo_gym.base_resources_server import _MCP_SESSION_TOKEN

        context_token = _MCP_SESSION_TOKEN.set("bad-token")
        try:
            with pytest.raises(HTTPException) as exc_info:
                server.require_mcp_session_id()
        finally:
            _MCP_SESSION_TOKEN.reset(context_token)

        assert exc_info.value.status_code == 401
