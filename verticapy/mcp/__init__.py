"""
VerticaPy MCP (Model Context Protocol) Server

This module provides an MCP-compliant server that exposes VerticaPy functionality
to Large Language Model clients like Claude Desktop.

Usage:
    from verticapy import mcp
    mcp.start()

Or from command line:
    python -m verticapy.mcp
"""

from .server import start, MCPServer

__version__ = "0.1.0"
__all__ = ["start", "MCPServer"]