from pointblank.mcp.server import mcp

# Expose the low-level MCP Server for introspection tools (e.g., Great Docs)
_server = mcp._mcp_server

__all__ = ["mcp"]
