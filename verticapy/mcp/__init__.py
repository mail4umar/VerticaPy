# verticapy/mcp/__init__.py
"""
VerticaPy MCP (Model Context Protocol) Integration

This module provides MCP server functionality for VerticaPy,
allowing LLMs to interact with Vertica through VerticaPy functions.
"""

from .server import MCPServer
from .tools import ToolRegistry
from .client import start_mcp_server, stop_mcp_server, is_server_running

__version__ = "0.1.0"
__all__ = ["MCPServer", "ToolRegistry", "start_mcp_server", "stop_mcp_server", "is_server_running"]


# Convenience function for quick start
def start(host: str = "localhost", port: int = 8765):
    """
    Quick start function for MCP server
    
    Args:
        host: Host to bind to (default: localhost)
        port: Port to bind to (default: 8765)
    """
    return start_mcp_server(host=host, port=port)


def stop():
    """Stop the MCP server"""
    stop_mcp_server()


def get_config(host: str = "localhost", port: int = 8765):
    """
    Get Claude Desktop configuration for the MCP server
    
    Args:
        host: Host to bind to (default: localhost)
        port: Port to bind to (default: 8765)
        
    Returns:
        dict: Configuration dictionary
    """
    return {
        "mcpServers": {
            "verticapy": {
                "command": "python",
                "args": ["-m", "verticapy.mcp.server", "--host", host, "--port", str(port)]
            }
        }
    }


def print_config(host: str = "localhost", port: int = 8765):
    """Print the configuration instructions"""
    print("\n📋 Add this to your Claude Desktop config:")
    print('"mcpServers": {')
    print('  "verticapy": {')
    print('    "command": "python",')
    print(f'    "args": ["-m", "verticapy.mcp.server", "--host", "{host}", "--port", "{port}"]')
    print('  }')
    print('}')