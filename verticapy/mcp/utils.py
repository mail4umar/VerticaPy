# verticapy/mcp/utils.py
"""
Utility functions for VerticaPy MCP server.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
import base64
import io

logger = logging.getLogger(__name__)


def serialize_vdataframe_info(vdf) -> Dict[str, Any]:
    """
    Serialize vDataFrame information for MCP responses.
    
    Args:
        vdf: VerticaPy vDataFrame object
        
    Returns:
        Serialized vDataFrame information
    """
    try:
        return {
            "shape": vdf.shape(),
            "columns": vdf.get_columns(),
            "dtypes": dict(zip(vdf.get_columns(), [str(dtype) for dtype in vdf.dtypes()])),
            "memory_usage": vdf.memory_usage() if hasattr(vdf, 'memory_usage') else None,
            "table_info": {
                "name": vdf._VERTICAPY_VARIABLES_["input_relation"],
                "schema": getattr(vdf, '_schema', None)
            }
        }
    except Exception as e:
        logger.warning(f"Could not serialize vDataFrame info: {e}")
        return {"error": str(e)}


def serialize_model_info(model) -> Dict[str, Any]:
    """
    Serialize ML model information for MCP responses.
    
    Args:
        model: VerticaPy ML model object
        
    Returns:
        Serialized model information
    """
    try:
        info = {
            "model_type": model.__class__.__name__,
            "parameters": getattr(model, 'parameters', {}),
        }
        
        # Add model-specific information
        if hasattr(model, 'score_'):
            info["score"] = model.score_
        if hasattr(model, 'features_importance_'):
            info["feature_importance"] = model.features_importance_
        if hasattr(model, 'summary'):
            info["summary"] = str(model.summary())
            
        return info
    except Exception as e:
        logger.warning(f"Could not serialize model info: {e}")
        return {"error": str(e)}


def encode_plot(plot_obj, format: str = "png") -> str:
    """
    Encode plot object as base64 string.
    
    Args:
        plot_obj: Plot object from VerticaPy
        format: Output format (png, svg, html)
        
    Returns:
        Base64 encoded plot
    """
    try:
        buffer = io.BytesIO()
        
        if format.lower() == "png":
            plot_obj.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
            
        elif format.lower() == "svg":
            plot_obj.savefig(buffer, format='svg', bbox_inches='tight')
            buffer.seek(0)
            svg_content = buffer.read().decode('utf-8')
            return f"data:image/svg+xml;base64,{base64.b64encode(svg_content.encode()).decode()}"
            
        else:
            return str(plot_obj)
            
    except Exception as e:
        logger.error(f"Error encoding plot: {e}")
        return f"Error encoding plot: {str(e)}"