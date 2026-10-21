"""MCP server exposing the diagnostic tools to any MCP client
(Claude Desktop, Claude Code, or the agent in this repo).

Run:  python -m autodiag.mcp_server
"""
from mcp.server.mcpserver import MCPServer

from .tools import TOOLS

server = MCPServer(
    "autodiag",
    instructions="Automotive diagnostics: DTC lookup, service-manual search, CAN frame decoding, vehicle snapshots.",
)

for fn in TOOLS:
    server.tool()(fn)


if __name__ == "__main__":
    server.run("stdio")
