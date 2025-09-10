# verticapy/mcp/tools.py
"""
MCP tools for VerticaPy functionality.
"""

import verticapy as vp
from typing import Dict, Any, List

class VerticaPyTools:
    """Collection of MCP tools for VerticaPy"""
    
    @staticmethod
    def get_tool_definitions() -> List[Dict[str, Any]]:
        """Return list of available MCP tool definitions"""
        return [
            {
                "name": "create_dataframe",
                "description": "Create a VerticaPy vDataFrame from dictionary data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Dictionary containing column names as keys and lists of values",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": ["number", "string", "boolean", "null"]}
                            }
                        },
                        "name": {
                            "type": "string",
                            "description": "Optional name for the dataframe",
                            "default": "temp_df"
                        }
                    },
                    "required": ["data"]
                }
            },
            {
                "name": "dataframe_info",
                "description": "Get information about a vDataFrame including shape, columns, and basic statistics",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Dictionary containing column names as keys and lists of values"
                        }
                    },
                    "required": ["data"]
                }
            }
        ]
    
    @staticmethod
    async def create_dataframe(arguments: Dict[str, Any]) -> str:
        """Tool to create a VerticaPy vDataFrame"""
        data = arguments.get("data")
        name = arguments.get("name", "temp_df")
        
        if not data or not isinstance(data, dict):
            raise ValueError("Data must be a non-empty dictionary")
            
        # Validate that all columns have the same length
        lengths = [len(values) for values in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All columns must have the same number of rows")
            
        try:
            # Create vDataFrame using VerticaPy
            df = vp.vDataFrame(data)
            
            # Get basic info about the dataframe
            shape = df.shape()
            columns = list(data.keys())
            
            result = f"✅ Successfully created vDataFrame '{name}'\n"
            result += f"📊 Shape: {shape}\n"
            result += f"📋 Columns: {columns}\n\n"
            result += "🔍 First few rows:\n"
            result += str(df.head(5))
            
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to create vDataFrame: {str(e)}")
    
    @staticmethod
    async def dataframe_info(arguments: Dict[str, Any]) -> str:
        """Tool to get detailed information about a vDataFrame"""
        data = arguments.get("data")
        
        if not data or not isinstance(data, dict):
            raise ValueError("Data must be a non-empty dictionary")
            
        try:
            # Create vDataFrame
            df = vp.vDataFrame(data)
            
            # Get comprehensive info
            shape = df.shape()
            columns = df.get_columns()
            dtypes = {col: str(df[col].dtype()) for col in columns}
            
            result = f"📊 DataFrame Information\n"
            result += f"{'='*40}\n"
            result += f"Shape: {shape}\n"
            result += f"Memory Usage: {df.memory_usage()} bytes\n\n"
            
            result += f"📋 Column Information:\n"
            for col in columns:
                result += f"  • {col}: {dtypes[col]}\n"
            
            result += f"\n📈 Basic Statistics:\n"
            try:
                stats = df.describe()
                result += str(stats)
            except Exception as e:
                result += f"Could not compute statistics: {str(e)}"
                
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to analyze vDataFrame: {str(e)}")

# Tool registry
TOOLS = {
    "create_dataframe": VerticaPyTools.create_dataframe,
    "dataframe_info": VerticaPyTools.dataframe_info,
}