"""
VerticaPy MCP Server - Refactored

A minimal Model Context Protocol server for VerticaPy integration.
This version uses the ToolRegistry for better modularity.
"""

import json
import asyncio
import websockets
import logging
from typing import Dict, Any
import traceback

from .tools import get_tool_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPServer:
    """Minimal MCP Server for VerticaPy"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.tool_registry = get_tool_registry()
        
    async def handle_request(self, websocket, path):
        """Handle incoming WebSocket requests"""
        logger.info(f"New connection from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    response = await self._process_request(request)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                        "id": None
                    }
                    await websocket.send(json.dumps(error_response))
                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}")
                    logger.error(traceback.format_exc())
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                        "id": request.get("id") if 'request' in locals() else None
                    }
                    await websocket.send(json.dumps(error_response))
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
    
    async def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP request and return response"""
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {})
        
        if method == "initialize":
            return await self._handle_initialize(req_id, params)
        elif method == "tools/list":
            return await self._handle_tools_list(req_id)
        elif method == "tools/call":
            return await self._handle_tools_call(req_id, params)
        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": req_id
            }
    
    async def _handle_initialize(self, req_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "verticapy-mcp-server",
                    "version": "0.1.0"
                }
            },
            "id": req_id
        }
    
    async def _handle_tools_list(self, req_id: str) -> Dict[str, Any]:
        """Handle tools/list request"""
        tools_list = self.tool_registry.get_tool_list()
        
        return {
            "jsonrpc": "2.0",
            "result": {"tools": tools_list},
            "id": req_id
        }
    
    async def _handle_tools_call(self, req_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not self.tool_registry.has_tool(tool_name):
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
                "id": req_id
            }
        
        try:
            result = await self.tool_registry.execute_tool(tool_name, arguments)
            
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                },
                "id": req_id
            }
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Tool execution failed: {str(e)}"},
                "id": req_id
            }
    
    async def start_server(self):
        """Start the MCP server"""
        logger.info(f"Starting VerticaPy MCP server on ws://{self.host}:{self.port}")
        
        # Print configuration instructions
        self._print_startup_info()
        
        # Start WebSocket server
        async with websockets.serve(
            self.handle_request,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20
        ):
            await asyncio.Future()  # Run forever
    
    def _print_startup_info(self):
        """Print startup information and configuration"""
        print(f"\n🚀 VerticaPy MCP Server starting on ws://{self.host}:{self.port}")
        print(f"📊 Available tools: {len(self.tool_registry.tools)}")
        for tool_name in self.tool_registry.tools.keys():
            print(f"  • {tool_name}")
        
        print("\n📋 Add this to your Claude Desktop config:")
        print('"mcpServers": {')
        print('  "verticapy": {')
        print('    "command": "python",')
        print(f'    "args": ["-m", "verticapy.mcp.server", "--host", "{self.host}", "--port", "{self.port}"]')
        print('  }')
        print('}')
        print("\n✅ Server ready for connections...")


def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VerticaPy MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    server = MCPServer(host=args.host, port=args.port)
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")


if __name__ == "__main__":
    main()