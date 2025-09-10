# verticapy/mcp/__main__.py
"""
Entry point for running the VerticaPy MCP server as a module.

Usage:
    python -m verticapy.mcp
    python -m verticapy.mcp.server --host localhost --port 8765
"""

from .server import main

if __name__ == "__main__":
    main()