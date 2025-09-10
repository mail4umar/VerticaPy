# verticapy/mcp/exceptions.py  
"""
Custom exceptions for VerticaPy MCP server.
"""


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when there's no active VerticaPy connection."""
    pass


class MCPToolError(MCPError):
    """Raised when a tool encounters an error.""" 
    pass


class MCPResourceError(MCPError):
    """Raised when a resource cannot be accessed."""
    pass


class MCPSessionError(MCPError):
    """Raised when session state is invalid."""
    pass