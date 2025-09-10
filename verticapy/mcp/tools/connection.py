"""
Connection management tools for VerticaPy MCP server.
"""

from typing import Any, Dict
import logging
import verticapy as vp
from .base import VerticaPyMCPTool
from ..exceptions import MCPToolError

logger = logging.getLogger(__name__)


class ListConnectionsTool(VerticaPyMCPTool):
    """Tool for listing available VerticaPy connections."""
    
    def __init__(self):
        super().__init__(
            name="list_connections",
            description="List all available VerticaPy database connections"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """List available connections."""
        try:
            from verticapy.connection import available_connections
            
            connections = available_connections()
            
            return {
                "success": True,
                "connections": connections,
                "connection_count": len(connections),
                "active_connection": vp.current_cursor().connection if vp.current_cursor() else None
            }
            
        except Exception as e:
            logger.error(f"Error listing connections: {e}")
            raise MCPToolError(f"Failed to list connections: {str(e)}")


class GetConnectionInfoTool(VerticaPyMCPTool):
    """Tool for getting connection information."""
    
    def __init__(self):
        super().__init__(
            name="get_connection_info", 
            description="Get detailed information about the current database connection"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "connection_name": {
                    "type": "string",
                    "description": "Specific connection name (optional)"
                }
            }
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Get connection information."""
        try:
            cursor = vp.current_cursor()
            if not cursor:
                return {
                    "success": False,
                    "error": "No active connection"
                }
            
            # Get connection details
            connection = cursor.connection
            
            info = {
                "success": True,
                "host": getattr(connection, 'host', 'unknown'),
                "port": getattr(connection, 'port', 'unknown'),
                "database": getattr(connection, 'database', 'unknown'),
                "user": getattr(connection, 'user', 'unknown'),
                "status": "connected" if connection else "disconnected",
                "server_version": None
            }
            
            # Try to get server version
            try:
                cursor.execute("SELECT version()")
                result = cursor.fetchone()
                if result:
                    info["server_version"] = result[0]
            except:
                pass
                
            return info
            
        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            raise MCPToolError(f"Failed to get connection info: {str(e)}")