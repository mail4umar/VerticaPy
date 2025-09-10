# verticapy/mcp/tools.py
"""
VerticaPy MCP Tools Registry

This module defines and manages the tools available through the MCP server.
Each tool corresponds to a VerticaPy function that can be called by LLMs.
"""

from typing import Dict, Any, Callable, Optional
import json
import verticapy as vp
from verticapy.core.vdataframe.base import vDataFrame


class ToolRegistry:
    """Registry for MCP tools"""
    
    def __init__(self):
        self.tools = {}
        self._register_core_tools()
    
    def _register_core_tools(self):
        """Register core VerticaPy tools"""
        
        # Create DataFrame tool
        self.register_tool(
            name="create_dataframe",
            description="Create a vDataFrame from dictionary data",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Dictionary of column names to lists of values"
                    },
                    "name": {
                        "type": "string", 
                        "description": "Optional name for the dataframe",
                        "default": "temp_df"
                    }
                },
                "required": ["data"]
            },
            handler=self._handle_create_dataframe
        )
        
        # Get available connections tool
        self.register_tool(
            name="get_connections",
            description="Get list of available VerticaPy connections",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._handle_get_connections
        )
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable):
        """Register a new tool"""
        self.tools[name] = {
            "description": description,
            "inputSchema": input_schema,
            "handler": handler
        }
    
    def get_tool_list(self) -> list:
        """Get list of all registered tools for MCP tools/list"""
        tools_list = []
        for name, tool_def in self.tools.items():
            tools_list.append({
                "name": name,
                "description": tool_def["description"],
                "inputSchema": tool_def["inputSchema"]
            })
        return tools_list
    
    def has_tool(self, name: str) -> bool:
        """Check if tool exists"""
        return name in self.tools
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name"""
        if not self.has_tool(name):
            raise ValueError(f"Unknown tool: {name}")
        
        handler = self.tools[name]["handler"]
        return await handler(arguments)
    
    # Tool handlers
    async def _handle_create_dataframe(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_dataframe tool"""
        data = arguments.get("data")
        name = arguments.get("name", "temp_df")
        
        if not data or not isinstance(data, dict):
            raise ValueError("Data must be a dictionary of column names to lists")
        
        # Validate that all values are lists and same length
        lengths = [len(v) if isinstance(v, list) else 1 for v in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All columns must have the same number of rows")
        
        try:
            # Create vDataFrame
            df = vp.vDataFrame(data)
            
            # Get basic info about the dataframe  
            result = {
                "success": True,
                "dataframe_name": name,
                "shape": {
                    "rows": df.shape()[0],
                    "columns": df.shape()[1]
                },
                "columns": df.get_columns(),
                "dtypes": {col: str(df[col].dtype()) for col in df.get_columns()},
                "sample_data": self._get_sample_data(df),
                "message": f"Successfully created vDataFrame '{name}' with {df.shape()[0]} rows and {df.shape()[1]} columns"
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create vDataFrame: {str(e)}"
            }
    
    async def _handle_get_connections(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_connections tool"""
        try:
            from verticapy.connection import available_connections
            
            connections = available_connections()
            
            result = {
                "success": True,
                "connections": connections,
                "count": len(connections),
                "message": f"Found {len(connections)} available connection(s)"
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "connections": [],
                "count": 0,
                "message": f"Failed to get connections: {str(e)}"
            }
    
    def _get_sample_data(self, df: vDataFrame, n: int = 3) -> Dict[str, Any]:
        """Get sample data from vDataFrame"""
        try:
            # Try to get head data
            head_data = df.head(n)
            
            # Convert to dictionary format if possible
            if hasattr(head_data, 'to_dict'):
                return head_data.to_dict()
            else:
                # Fallback: get basic representation
                return {
                    "preview": str(head_data),
                    "note": "Sample data converted to string representation"
                }
        except Exception as e:
            return {
                "error": f"Could not retrieve sample data: {str(e)}"
            }


# Global registry instance
_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry