# Example MCP Weather

This resources server demonstrates a Gym-owned Streamable HTTP MCP endpoint mounted at `/mcp`.
`/seed_session` returns hidden MCP metadata for Claude Code, the `get_weather` MCP tool records calls
against the Gym session, and `/verify` rewards only answers that used the tool in that same session.
