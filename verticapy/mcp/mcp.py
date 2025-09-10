from verticapy.mcp.server import MCPServer
from verticapy.mcp.config import MCPConfig

# Singleton instance for simple usage
_server_instance = None

class MCP:
    """Simple interface for starting VerticaPy MCP server."""
    
    @classmethod
    def start(cls, config: MCPConfig = None):
        """Start the MCP server with current VerticaPy session."""
        global _server_instance
        
        if _server_instance is None:
            _server_instance = MCPServer(config or MCPConfig())
        
        _server_instance.start()
        return _server_instance

__all__ = ["MCPServer", "MCPConfig", "MCP"]