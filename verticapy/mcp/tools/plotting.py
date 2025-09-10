"""
Plotting tools for VerticaPy MCP server.
"""

from typing import Any, Dict
import logging
import verticapy as vp
from .base import VerticaPyMCPTool
from ..exceptions import MCPToolError, MCPSessionError
from ..utils import encode_plot

logger = logging.getLogger(__name__)


class CreatePlotTool(VerticaPyMCPTool):
    """Tool for creating plots with VerticaPy."""
    
    def __init__(self):
        super().__init__(
            name="create_plot",
            description="Create visualizations using VerticaPy plotting functions"
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": ["histogram", "scatter", "boxplot", "bar", "line", "pie"],
                    "description": "Type of plot to create"
                },
                "dataframe_name": {
                    "type": "string",
                    "description": "Name of the vDataFrame"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column names for the plot"
                },
                "x_column": {
                    "type": "string",
                    "description": "X-axis column (for scatter, line plots)"
                },
                "y_column": {
                    "type": "string",
                    "description": "Y-axis column (for scatter, line plots)"
                },
                "groupby": {
                    "type": "string",
                    "description": "Column to group by (optional)"
                },
                "title": {
                    "type": "string",
                    "description": "Plot title (optional)"
                },
                "save_format": {
                    "type": "string",
                    "enum": ["png", "svg", "html"],
                    "default": "png",
                    "description": "Output format for the plot"
                }
            },
            "required": ["plot_type", "dataframe_name"]
        }
    
    async def execute(self, args: Dict[str, Any], session: Any) -> Dict[str, Any]:
        """Create plot."""
        try:
            plot_type = args["plot_type"]
            dataframe_name = args["dataframe_name"]
            columns = args.get("columns", [])
            x_column = args.get("x_column")
            y_column = args.get("y_column")
            groupby = args.get("groupby")
            title = args.get("title")
            save_format = args.get("save_format", "png")
            
            # Get vDataFrame from session
            if dataframe_name not in session.dataframes:
                raise MCPSessionError(f"vDataFrame '{dataframe_name}' not found in session")
            
            vdf = session.dataframes[dataframe_name]
            
            # Create plot based on type
            if plot_type == "histogram":
                column = columns[0] if columns else vdf.get_columns()[0]
                plot = vdf[column].hist(title=title)
                
            elif plot_type == "scatter":
                if not x_column or not y_column:
                    raise MCPToolError("scatter plot requires x_column and y_column")
                plot = vdf.scatter([x_column, y_column], by=groupby, title=title)
                
            elif plot_type == "boxplot":
                column = columns[0] if columns else vdf.get_columns()[0]
                plot = vdf[column].boxplot(by=groupby, title=title)
                
            elif plot_type == "bar":
                column = columns[0] if columns else vdf.get_columns()[0]
                plot = vdf[column].bar(by=groupby, title=title)
                
            elif plot_type == "line":
                if not x_column or not y_column:
                    raise MCPToolError("line plot requires x_column and y_column")
                plot = vdf.plot([x_column, y_column], kind="line", title=title)
                
            elif plot_type == "pie":
                column = columns[0] if columns else vdf.get_columns()[0]
                plot = vdf[column].pie(title=title)
                
            else:
                raise MCPToolError(f"Unsupported plot type: {plot_type}")
            
            # Encode plot
            encoded_plot = encode_plot(plot, save_format)
            
            return {
                "success": True,
                "plot_type": plot_type,
                "dataframe_name": dataframe_name,
                "columns": columns,
                "format": save_format,
                "plot_data": encoded_plot,
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            raise MCPToolError(f"Failed to create plot: {str(e)}")