# verticapy/mcp/server.py
import asyncio
import json
import logging
from typing import Dict, Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol
import verticapy as vp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.connection = None
        
    def set_connection(self, conn):
        """Set the VerticaPy connection to use for queries"""
        self.connection = conn
        
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket client connections"""
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            
    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming JSON-RPC messages"""
        try:
            request = json.loads(message)
            response = await self.process_request(request)
            await websocket.send(json.dumps(response))
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            }
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = {
                "jsonrpc": "2.0", 
                "id": request.get("id") if 'request' in locals() else None,
                "error": {"code": -32603, "message": str(e)}
            }
            await websocket.send(json.dumps(error_response))
            
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON-RPC request and return response"""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")
        
        logger.info(f"Processing method: {method}")
        
        if method == "initialize":
            return await self.handle_initialize(req_id, params)
        elif method == "tools/list":
            return await self.handle_tools_list(req_id)
        elif method == "tools/call":
            return await self.handle_tool_call(req_id, params)
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
            
    async def handle_initialize(self, req_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "verticapy-mcp",
                    "version": "0.1.0"
                }
            }
        }
        
    async def handle_tools_list(self, req_id: int) -> Dict[str, Any]:
        """Handle tools/list request - return available tools"""
        tools = [
            {
                "name": "create_dataframe",
                "description": "Create a VerticaPy vDataFrame from dictionary data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Dictionary containing column names as keys and lists of values"
                        },
                        "name": {
                            "type": "string",
                            "description": "Optional name for the dataframe",
                            "default": "temp_df"
                        }
                    },
                    "required": ["data"]
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": tools
            }
        }
        
    async def handle_tool_call(self, req_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request - execute the requested tool"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "create_dataframe":
                result = await self.create_dataframe_tool(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)}
            }
            
    async def create_dataframe_tool(self, arguments: Dict[str, Any]) -> str:
        """Tool to create a VerticaPy vDataFrame"""
        data = arguments.get("data")
        name = arguments.get("name", "temp_df")
        
        if not data or not isinstance(data, dict):
            raise ValueError("Data must be a non-empty dictionary")
            
        try:
            # Create vDataFrame using VerticaPy
            df = vp.vDataFrame(data)
            
            # Get basic info about the dataframe
            shape = df.shape()
            columns = list(data.keys())
            
            result = f"Successfully created vDataFrame '{name}'\n"
            result += f"Shape: {shape}\n"
            result += f"Columns: {columns}\n"
            result += f"First few rows:\n{df.head(5)}"
            
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to create vDataFrame: {str(e)}")
    
    async def start(self):
        """Start the MCP server"""
        logger.info(f"Starting VerticaPy MCP server on ws://{self.host}:{self.port}")
        
        # Print configuration instructions
        print("\n" + "="*60)
        print("VerticaPy MCP Server Started!")
        print(f"Server running on ws://{self.host}:{self.port}")
        print("\nTo connect with Claude Desktop, add this to your config:")
        print('"mcpServers": {')
        print('  "verticapy": {')
        print('    "command": "python",')
        print('    "args": ["-m", "verticapy.mcp"]')
        print('  }')
        print('}')
        print("="*60)
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever

# Global server instance
_server = MCPServer()

def start(host: str = "localhost", port: int = 8765, connection=None):
    """Start the MCP server (convenience function)"""
    global _server
    _server = MCPServer(host, port)
    if connection:
        _server.set_connection(connection)
    asyncio.run(_server.start())

if __name__ == "__main__":
    start()