"""
vDataFrame operation tools for VerticaPy MCP server.
"""

from typing import Any, Dict, List
import logging
import verticapy as vp
from .base import VerticaPyMCPTool
from ..exceptions import MCPToolError, MCPSessionError
from ..utils import serialize_vdataframe_info

logger = logging.getLogger(__name__)


class CreateVDataFrameTool(VerticaPyMCPTool):
    """Tool for creating VerticaPy vDataFrame objects."""
    
    def __init__(self):
        super().__init__(
            name="create_vdataframe",
            description="Create a VerticaPy vDataFrame from a table or SQL query"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "input_relation": {
                    "type": "string",
                    "description": "Table name or SQL query to create vDataFrame from"
                },
                "name": {
                    "type": "string",
                    "description": "Name to assign to the vDataFrame in session"
                },
                "schema": {
                    "type": "string", 
                    "description": "Schema name for table (optional)"
                }
            },
            "required": ["input_relation", "name"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Create vDataFrame."""
        try:
            input_relation = args["input_relation"]
            name = args["name"]
            schema = args.get("schema")
            
            # Check session limits
            if len(session.dataframes) >= session.config.max_dataframes:
                raise MCPSessionError(f"Maximum dataframes limit ({session.config.max_dataframes}) reached")
            
            # Create vDataFrame
            if schema:
                full_relation = f"{schema}.{input_relation}"
            else:
                full_relation = input_relation
                
            vdf = vp.vDataFrame(full_relation)
            
            # Store in session
            session.dataframes[name] = vdf
            
            # Get vDataFrame info
            vdf_info = serialize_vdataframe_info(vdf)
            
            return {
                "success": True,
                "name": name,
                "vdataframe_info": vdf_info,
                "message": f"vDataFrame '{name}' created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating vDataFrame: {e}")
            raise MCPToolError(f"Failed to create vDataFrame: {str(e)}")


class DescribeDataTool(VerticaPyMCPTool):
    """Tool for describing vDataFrame data."""
    
    def __init__(self):
        super().__init__(
            name="describe_data",
            description="Get descriptive statistics for a vDataFrame"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "dataframe_name": {
                    "type": "string",
                    "description": "Name of the vDataFrame to describe"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to describe (optional)"
                },
                "include_categorical": {
                    "type": "boolean",
                    "description": "Include categorical variable descriptions",
                    "default": True
                }
            },
            "required": ["dataframe_name"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Describe vDataFrame data."""
        try:
            dataframe_name = args["dataframe_name"]
            columns = args.get("columns")
            include_categorical = args.get("include_categorical", True)
            
            # Get vDataFrame from session
            if dataframe_name not in session.dataframes:
                raise MCPSessionError(f"vDataFrame '{dataframe_name}' not found in session")
            
            vdf = session.dataframes[dataframe_name]
            
            # Get description
            if columns:
                description = vdf[columns].describe()
            else:
                description = vdf.describe()
            
            # Convert to serializable format
            desc_dict = description.to_dict() if hasattr(description, 'to_dict') else str(description)
            
            return {
                "success": True,
                "dataframe_name": dataframe_name,
                "description": desc_dict,
                "columns_described": columns or vdf.get_columns()
            }
            
        except Exception as e:
            logger.error(f"Error describing data: {e}")
            raise MCPToolError(f"Failed to describe data: {str(e)}")


class GetColumnsTool(VerticaPyMCPTool):
    """Tool for getting vDataFrame column information."""
    
    def __init__(self):
        super().__init__(
            name="get_columns",
            description="Get column names and data types for a vDataFrame"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object", 
            "properties": {
                "dataframe_name": {
                    "type": "string",
                    "description": "Name of the vDataFrame"
                },
                "include_dtypes": {
                    "type": "boolean",
                    "description": "Include data type information",
                    "default": True
                }
            },
            "required": ["dataframe_name"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Get column information."""
        try:
            dataframe_name = args["dataframe_name"]
            include_dtypes = args.get("include_dtypes", True)
            
            # Get vDataFrame from session
            if dataframe_name not in session.dataframes:
                raise MCPSessionError(f"vDataFrame '{dataframe_name}' not found in session")
            
            vdf = session.dataframes[dataframe_name]
            
            # Get columns
            columns = vdf.get_columns()
            
            result = {
                "success": True,
                "dataframe_name": dataframe_name,
                "columns": columns,
                "column_count": len(columns)
            }
            
            # Add data types if requested
            if include_dtypes:
                dtypes = vdf.dtypes()
                result["dtypes"] = dict(zip(columns, [str(dtype) for dtype in dtypes]))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting columns: {e}")
            raise MCPToolError(f"Failed to get columns: {str(e)}")


class AggregateDataTool(VerticaPyMCPTool):
    """Tool for aggregating vDataFrame data."""
    
    def __init__(self):
        super().__init__(
            name="aggregate_data",
            description="Perform aggregation operations on vDataFrame"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "dataframe_name": {
                    "type": "string",
                    "description": "Name of the vDataFrame to aggregate"
                },
                "groupby_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to group by"
                },
                "aggregations": {
                    "type": "object",
                    "description": "Aggregation functions per column (e.g., {'col1': 'sum', 'col2': 'avg'})"
                },
                "result_name": {
                    "type": "string",
                    "description": "Name for the resulting aggregated vDataFrame"
                }
            },
            "required": ["dataframe_name", "groupby_columns", "aggregations", "result_name"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Perform aggregation."""
        try:
            dataframe_name = args["dataframe_name"]
            groupby_columns = args["groupby_columns"]
            aggregations = args["aggregations"]
            result_name = args["result_name"]
            
            # Get vDataFrame from session
            if dataframe_name not in session.dataframes:
                raise MCPSessionError(f"vDataFrame '{dataframe_name}' not found in session")
            
            vdf = session.dataframes[dataframe_name]
            
            # Perform aggregation
            grouped = vdf.groupby(groupby_columns)
            result_vdf = grouped.agg(aggregations)
            
            # Store result in session
            session.dataframes[result_name] = result_vdf
            
            # Get result info
            result_info = serialize_vdataframe_info(result_vdf)
            
            return {
                "success": True,
                "original_dataframe": dataframe_name,
                "result_dataframe": result_name,
                "groupby_columns": groupby_columns,
                "aggregations": aggregations,
                "result_info": result_info
            }
            
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            raise MCPToolError(f"Failed to aggregate data: {str(e)}")
