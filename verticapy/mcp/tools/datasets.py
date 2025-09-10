"""
Dataset loading tools for VerticaPy MCP server.
"""

from typing import Any, Dict
import logging
import verticapy as vp
from .base import VerticaPyMCPTool
from ..exceptions import MCPToolError, MCPSessionError
from ..utils import serialize_vdataframe_info

logger = logging.getLogger(__name__)


class ListDatasetsTool(VerticaPyMCPTool):
    """Tool for listing available built-in datasets."""
    
    def __init__(self):
        super().__init__(
            name="list_datasets",
            description="List all available built-in VerticaPy datasets"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """List available datasets."""
        try:
            # Built-in VerticaPy datasets
            datasets = {
                "titanic": {
                    "description": "Titanic passenger survival dataset",
                    "rows": 1234,
                    "columns": 12,
                    "type": "classification"
                },
                "iris": {
                    "description": "Iris flower dataset",
                    "rows": 150,
                    "columns": 5,
                    "type": "classification"
                },
                "winequality": {
                    "description": "Wine quality dataset",
                    "rows": 6497,
                    "columns": 13,
                    "type": "regression"
                },
                "amazon": {
                    "description": "Amazon product reviews dataset",
                    "rows": 1000,
                    "columns": 10,
                    "type": "text_analysis"
                },
                "commodities": {
                    "description": "Commodities prices dataset",
                    "rows": 1000,
                    "columns": 8,
                    "type": "time_series"
                }
            }
            
            return {
                "success": True,
                "datasets": datasets,
                "dataset_count": len(datasets)
            }
            
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            raise MCPToolError(f"Failed to list datasets: {str(e)}")


class LoadDatasetTool(VerticaPyMCPTool):
    """Tool for loading built-in VerticaPy datasets."""
    
    def __init__(self):
        super().__init__(
            name="load_dataset",
            description="Load a built-in VerticaPy dataset"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "enum": ["titanic", "iris", "winequality", "amazon", "commodities"],
                    "description": "Name of the dataset to load"
                },
                "vdf_name": {
                    "type": "string",
                    "description": "Name to assign to the loaded vDataFrame"
                }
            },
            "required": ["dataset_name", "vdf_name"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Load dataset."""
        try:
            dataset_name = args["dataset_name"]
            vdf_name = args["vdf_name"]
            
            # Check session limits
            if len(session.dataframes) >= session.config.max_dataframes:
                raise MCPSessionError(f"Maximum dataframes limit ({session.config.max_dataframes}) reached")
            
            # Load dataset using VerticaPy functions
            load_functions = {
                "titanic": vp.load_titanic,
                "iris": vp.load_iris,
                "winequality": vp.load_winequality,
                "amazon": vp.load_amazon,
                "commodities": vp.load_commodities
            }
            
            if dataset_name not in load_functions:
                raise MCPToolError(f"Unknown dataset: {dataset_name}")
            
            # Load the dataset
            vdf = load_functions[dataset_name]()
            
            # Store in session
            session.dataframes[vdf_name] = vdf
            
            # Get dataset info
            vdf_info = serialize_vdataframe_info(vdf)
            
            return {
                "success": True,
                "dataset_name": dataset_name,
                "vdf_name": vdf_name,
                "vdataframe_info": vdf_info,
                "message": f"Dataset '{dataset_name}' loaded as '{vdf_name}'"
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise MCPToolError(f"Failed to load dataset: {str(e)}")
