# verticapy/mcp/config.py
"""
Configuration for VerticaPy MCP Server.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class MCPConfig:
    """Configuration class for MCP server."""
    
    name: str = "verticapy"
    version: str = "1.0.0"
    description: str = "VerticaPy MCP Server for advanced Vertica database analytics"
    
    # Server settings
    transport: str = "stdio"  # MCP standard transport
    
    # VerticaPy tools
    tools: List[str] = field(default_factory=lambda: [
        # Connection management
        "list_connections",
        "get_connection_info",
        
        # DataFrame operations
        "create_vdataframe",
        "describe_data", 
        "get_columns",
        "filter_data",
        "aggregate_data",
        
        # Machine learning
        "train_model",
        "evaluate_model",
        "predict_model",
        "cross_validate",
        
        # Visualization
        "create_plot",
        "correlation_matrix",
        
        # Datasets
        "list_datasets", 
        "load_dataset",
        
        # Performance
        "query_profiler"
    ])
    
    # Available resources
    resources: List[str] = field(default_factory=lambda: [
        "connections",
        "dataframes", 
        "models"
    ])
    
    # Available prompts
    prompts: List[str] = field(default_factory=lambda: [
        "data_science_assistant",
        "ml_model_advisor"
    ])
    
    # Session management
    max_dataframes: int = 10
    max_models: int = 5
    cache_plots: bool = True
    auto_cleanup: bool = True
    
    # Plotting settings
    default_plot_format: str = "png"
    plot_width: int = 800
    plot_height: int = 600