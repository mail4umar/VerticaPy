"""
Main MCP Server implementation for VerticaPy.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict

from .config import MCPConfig
from .exceptions import MCPConnectionError, MCPToolError, MCPResourceError, MCPSessionError
from .tools import (
    ListConnectionsTool, GetConnectionInfoTool,
    CreateVDataFrameTool, DescribeDataTool, GetColumnsTool, AggregateDataTool,
    TrainModelTool, EvaluateModelTool, PredictModelTool,
    ListDatasetsTool, LoadDatasetTool,
    CreatePlotTool
)
from .utils import format_error_response

logger = logging.getLogger(__name__)


class MCPSession:
    """Session state management for VerticaPy MCP server."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.dataframes: Dict[str, Any] = {}  # vDataFrame objects
        self.models: Dict[str, Any] = {}      # Trained ML models
        self.plots: Dict[str, Any] = {}       # Generated plots cache
        
    def cleanup(self):
        """Cleanup session resources."""
        if self.config.auto_cleanup:
            # Clear old objects if limits exceeded
            if len(self.dataframes) > self.config.max_dataframes:
                # Remove oldest dataframes
                excess = len(self.dataframes) - self.config.max_dataframes
                keys_to_remove = list(self.dataframes.keys())[:excess]
                for key in keys_to_remove:
                    del self.dataframes[key]
                    logger.info(f"Removed dataframe '{key}' due to limit")
            
            if len(self.models) > self.config.max_models:
                # Remove oldest models
                excess = len(self.models) - self.config.max_models
                keys_to_remove = list(self.models.keys())[:excess]
                for key in keys_to_remove:
                    del self.models[key]
                    logger.info(f"Removed model '{key}' due to limit")


class MCPServer:
    """VerticaPy MCP Server implementation."""
    
    def __init__(self, config: Optional[MCPConfig] = None):
        """
        Initialize MCP Server.
        
        Args:
            config: Server configuration
        """
        self.config = config or MCPConfig()
        self.session = MCPSession(self.config)
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        self._initialize_tools()
        self._initialize_resources()
        self._initialize_prompts()
    
    def _initialize_tools(self):
        """Initialize available tools."""
        tool_classes = {
            # Connection tools
            "list_connections": ListConnectionsTool,
            "get_connection_info": GetConnectionInfoTool,
            
            # DataFrame tools
            "create_vdataframe": CreateVDataFrameTool,
            "describe_data": DescribeDataTool,
            "get_columns": GetColumnsTool,
            "aggregate_data": AggregateDataTool,
            
            # ML tools
            "train_model": TrainModelTool,
            "evaluate_model": EvaluateModelTool,
            "predict_model": PredictModelTool,
            
            # Dataset tools
            "list_datasets": ListDatasetsTool,
            "load_dataset": LoadDatasetTool,
            
            # Plotting tools
            "create_plot": CreatePlotTool
        }
        
        for tool_name in self.config.tools:
            if tool_name in tool_classes:
                self.tools[tool_name] = tool_classes[tool_name]()
                logger.debug(f"Initialized tool: {tool_name}")
    
    def _initialize_resources(self):
        """Initialize available resources."""
        # Resources will be dynamically generated based on session state
        self.resources = {
            "connections": self._get_connections_resource,
            "dataframes": self._get_dataframes_resource,
            "models": self._get_models_resource
        }
    
    def _initialize_prompts(self):
        """Initialize available prompts."""
        self.prompts = {
            "data_science_assistant": {
                "name": "VerticaPy Data Science Assistant", 
                "description": "Helps with data science workflows using VerticaPy",
                "get_prompt": self._get_data_science_prompt
            },
            "ml_model_advisor": {
                "name": "ML Model Advisor",
                "description": "Provides guidance on machine learning model selection and evaluation",
                "get_prompt": self._get_ml_advisor_prompt
            }
        }
    
    # Resource handlers
    def _get_connections_resource(self) -> Dict[str, Any]:
        """Get connections resource."""
        try:
            import verticapy as vp
            from verticapy.connection import available_connections
            
            connections = available_connections()
            current_cursor = vp.current_cursor()
            
            return {
                "uri": "verticapy://connections",
                "name": "VerticaPy Connections",
                "description": "Available database connections",
                "mimeType": "application/json",
                "content": {
                    "available_connections": connections,
                    "active_connection": str(current_cursor.connection) if current_cursor else None,
                    "connection_count": len(connections)
                }
            }
        except Exception as e:
            logger.error(f"Error getting connections resource: {e}")
            return {"error": str(e)}
    
    def _get_dataframes_resource(self) -> Dict[str, Any]:
        """Get dataframes resource."""
        try:
            from .utils import serialize_vdataframe_info
            
            dataframes_info = {}
            for name, vdf in self.session.dataframes.items():
                dataframes_info[name] = serialize_vdataframe_info(vdf)
            
            return {
                "uri": "verticapy://dataframes",
                "name": "VerticaPy DataFrames", 
                "description": "Currently loaded vDataFrame objects",
                "mimeType": "application/json",
                "content": {
                    "dataframes": dataframes_info,
                    "dataframe_count": len(self.session.dataframes)
                }
            }
        except Exception as e:
            logger.error(f"Error getting dataframes resource: {e}")
            return {"error": str(e)}
    
    def _get_models_resource(self) -> Dict[str, Any]:
        """Get models resource."""
        try:
            from .utils import serialize_model_info
            
            models_info = {}
            for name, model in self.session.models.items():
                models_info[name] = serialize_model_info(model)
            
            return {
                "uri": "verticapy://models",
                "name": "VerticaPy ML Models",
                "description": "Trained machine learning models", 
                "mimeType": "application/json",
                "content": {
                    "models": models_info,
                    "model_count": len(self.session.models)
                }
            }
        except Exception as e:
            logger.error(f"Error getting models resource: {e}")
            return {"error": str(e)}
    
    # Prompt handlers
    def _get_data_science_prompt(self, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get data science assistant prompt."""
        # Get current session context
        dataframes_summary = list(self.session.dataframes.keys())
        models_summary = list(self.session.models.keys())
        
        context = f"""
You are a VerticaPy Data Science Assistant. You have access to the following VerticaPy tools and session state:

CURRENT SESSION:
- Active vDataFrames: {dataframes_summary}
- Trained Models: {models_summary}

AVAILABLE TOOLS:
- create_vdataframe: Create vDataFrame from tables/queries
- describe_data: Get statistical summaries
- load_dataset: Load built-in datasets (titanic, iris, winequality, etc.)
- train_model: Train ML models (LinearRegression, RandomForest, XGB, etc.)
- evaluate_model: Calculate model metrics (accuracy, AUC, R2, etc.)
- create_plot: Generate visualizations (histogram, scatter, boxplot, etc.)
- aggregate_data: Group by and aggregate operations

WORKFLOW RECOMMENDATIONS:
1. Start by loading or creating a dataset
2. Explore the data with describe_data and create_plot
3. Train appropriate models based on your problem type
4. Evaluate model performance with various metrics
5. Create visualizations to understand results

Always use VerticaPy's native functions rather than raw SQL when possible.
"""
        
        return {
            "name": "data_science_assistant",
            "description": "VerticaPy Data Science Assistant",
            "content": context
        }
    
    def _get_ml_advisor_prompt(self, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get ML model advisor prompt."""
        context = """
You are a Machine Learning Model Advisor for VerticaPy. Help users select and evaluate the right ML algorithms.

AVAILABLE MODELS:
- LinearRegression: For continuous target variables, linear relationships
- LogisticRegression: For binary/categorical classification
- RandomForestRegressor: For non-linear regression, handles missing values well
- RandomForestClassifier: For non-linear classification, good feature importance
- XGBRegressor: Gradient boosting for regression, often highest performance
- XGBClassifier: Gradient boosting for classification
- KMeans: For clustering/unsupervised learning
- DBSCAN: For density-based clustering

EVALUATION METRICS:
- Classification: accuracy, precision, recall, f1, auc
- Regression: r2, rmse, mae, aic, bic
- Model Selection: Use cross_validate for robust evaluation

RECOMMENDATIONS:
- Start with simple models (Linear/Logistic Regression) for baseline
- Try ensemble methods (RandomForest, XGB) for better performance
- Use appropriate metrics for your problem type
- Always validate with holdout or cross-validation
"""
        
        return {
            "name": "ml_model_advisor",
            "description": "ML Model Selection and Evaluation Advisor",
            "content": context
        }
    
    # MCP Protocol handlers
    async def handle_initialize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": self.config.name,
                    "version": self.config.version,
                    "description": self.config.description
                }
            }
        }
    
    async def handle_tools_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools list request."""
        tools_list = []
        
        for tool_name, tool in self.tools.items():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.get_schema()
            })
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": tools_list
            }
        }
    
    async def handle_tools_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request."""
        try:
            import verticapy as vp
            
            # Check if we have an active connection
            cursor = vp.current_cursor()
            if not cursor:
                return format_error_response(
                    MCPConnectionError("No active VerticaPy connection. Please call vp.connect() first."),
                    request.get("id")
                )
            
            params = request.get("params", {})
            tool_name = params.get("name")
            args = params.get("arguments", {})
            
            if tool_name not in self.tools:
                return format_error_response(
                    MCPToolError(f"Unknown tool: {tool_name}"),
                    request.get("id")
                )
            
            # Execute tool
            tool = self.tools[tool_name]
            result = await tool.execute(args, self.session)
            
            # Cleanup session if needed
            self.session.cleanup()
            
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in tool call: {e}")
            return format_error_response(e, request.get("id"))
    
    async def handle_resources_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources list request."""
        resources_list = []
        
        for resource_name, resource_handler in self.resources.items():
            if resource_name in self.config.resources:
                resource_info = resource_handler()
                if "error" not in resource_info:
                    resources_list.append({
                        "uri": resource_info["uri"],
                        "name": resource_info["name"],
                        "description": resource_info["description"],
                        "mimeType": resource_info["mimeType"]
                    })
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "resources": resources_list
            }
        }
    
    async def handle_resources_read(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource read request."""
        try:
            params = request.get("params", {})
            uri = params.get("uri", "")
            
            # Parse URI to get resource type
            if uri.startswith("verticapy://"):
                resource_type = uri.replace("verticapy://", "")
                
                if resource_type in self.resources:
                    resource_data = self.resources[resource_type]()
                    
                    if "error" in resource_data:
                        return format_error_response(
                            MCPResourceError(resource_data["error"]),
                            request.get("id")
                        )
                    
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "contents": [
                                {
                                    "uri": resource_data["uri"],
                                    "mimeType": resource_data["mimeType"],
                                    "text": json.dumps(resource_data["content"], indent=2)
                                }
                            ]
                        }
                    }
            
            return format_error_response(
                MCPResourceError(f"Unknown resource URI: {uri}"),
                request.get("id")
            )
            
        except Exception as e:
            logger.error(f"Error reading resource: {e}")
            return format_error_response(e, request.get("id"))
    
    async def handle_prompts_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts list request."""
        prompts_list = []
        
        for prompt_name, prompt_info in self.prompts.items():
            if prompt_name in self.config.prompts:
                prompts_list.append({
                    "name": prompt_name,
                    "description": prompt_info["description"]
                })
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "prompts": prompts_list
            }
        }
    
    async def handle_prompts_get(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompt get request."""
        try:
            params = request.get("params", {})
            prompt_name = params.get("name")
            args = params.get("arguments", {})
            
            if prompt_name not in self.prompts:
                return format_error_response(
                    MCPToolError(f"Unknown prompt: {prompt_name}"),
                    request.get("id")
                )
            
            # Get prompt content
            prompt_info = self.prompts[prompt_name]
            prompt_data = prompt_info["get_prompt"](args)
            
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "description": prompt_data["description"],
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": prompt_data["content"]
                            }
                        }
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting prompt: {e}")
            return format_error_response(e, request.get("id"))
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests."""
        method = request.get("method", "")
        
        # Route requests to appropriate handlers
        if method == "initialize":
            return await self.handle_initialize(request)
        elif method == "tools/list":
            return await self.handle_tools_list(request)
        elif method == "tools/call":
            return await self.handle_tools_call(request)
        elif method == "resources/list":
            return await self.handle_resources_list(request)
        elif method == "resources/read":
            return await self.handle_resources_read(request)
        elif method == "prompts/list":
            return await self.handle_prompts_list(request)
        elif method == "prompts/get":
            return await self.handle_prompts_get(request)
        else:
            return format_error_response(
                MCPToolError(f"Unknown method: {method}"),
                request.get("id")
            )
    
    async def run_stdio(self):
        """Run server using stdio transport."""
        logger.info("Starting VerticaPy MCP server with stdio transport")
        
        while True:
            try:
                # Read JSON-RPC request from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                
                # Write JSON-RPC response to stdout
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response), flush=True)
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error_response = {
                    "jsonrpc": "2.0", 
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": "Internal error"
                    }
                }
                print(json.dumps(error_response), flush=True)
    
    def start(self):
        """Start the MCP server."""
        import verticapy as vp
        
        # Check if VerticaPy is connected
        cursor = vp.current_cursor()
        if not cursor:
            print("Warning: No active VerticaPy connection detected.")
            print("Please run vp.connect(...) before starting the MCP server.")
            print()
        
        # Print configuration instructions
        print(f"Starting VerticaPy MCP server (version {self.config.version})")
        print("=" * 50)
        print()
        print("Add this to your Claude Desktop configuration:")
        print()
        print('"mcpServers": {')
        print(f'  "verticapy": {{')
        print(f'    "command": "python",')
        print(f'    "args": ["-m", "verticapy.mcp.server"]')
        print(f'  }}')
        print('}')
        print()
        print("Available tools:")
        for tool_name in self.config.tools:
            if tool_name in self.tools:
                print(f"  - {tool_name}: {self.tools[tool_name].description}")
        print()
        
        # Run the server
        try:
            asyncio.run(self.run_stdio())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise


# CLI entry point
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]  # Log to stderr to avoid interfering with stdio
    )
    
    # Create and start server
    server = MCPServer()
    server.start()