# verticapy/mcp/client.py
"""
VerticaPy MCP Client Helper

This module provides convenience functions for starting and managing
the MCP server from within VerticaPy.
"""

import asyncio
import threading
import logging
from typing import Optional
from .server import MCPServer

logger = logging.getLogger(__name__)


class MCPServerThread:
    """Thread wrapper for running MCP server"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.loop = None
        self._running = False
    
    def start(self):
        """Start the MCP server in a separate thread"""
        if self._running:
            logger.warning("MCP server is already running")
            return
        
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        self._running = True
        
        # Give server a moment to start
        import time
        time.sleep(1)
        
        logger.info(f"MCP server started on ws://{self.host}:{self.port}")
    
    def _run_server(self):
        """Run the server in the thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.server = MCPServer(host=self.host, port=self.port)
            self.loop.run_until_complete(self.server.start_server())
        except Exception as e:
            logger.error(f"MCP server error: {str(e)}")
        finally:
            self._running = False
    
    def stop(self):
        """Stop the MCP server"""
        if not self._running:
            logger.warning("MCP server is not running")
            return
        
        if self.loop:
            # Schedule the loop to stop
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        self._running = False
        logger.info("MCP server stopped")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running


# Global server instance
_server_instance: Optional[MCPServerThread] = None


def start_mcp_server(host: str = "localhost", port: int = 8765) -> MCPServerThread:
    """
    Start the VerticaPy MCP server
    
    Args:
        host: Host to bind to (default: localhost)
        port: Port to bind to (default: 8765)
        
    Returns:
        MCPServerThread instance
    """
    global _server_instance
    
    if _server_instance and _server_instance.is_running():
        logger.warning(f"MCP server already running on ws://{_server_instance.host}:{_server_instance.port}")
        return _server_instance
    
    _server_instance = MCPServerThread(host=host, port=port)
    _server_instance.start()
    
    return _server_instance


def stop_mcp_server():
    """Stop the VerticaPy MCP server"""
    global _server_instance
    
    if _server_instance:
        _server_instance.stop()
        _server_instance = None
    else:
        logger.warning("No MCP server instance to stop")


def get_mcp_server() -> Optional[MCPServerThread]:
    """Get the current MCP server instance"""
    return _server_instance


def is_server_running() -> bool:
    """Check if MCP server is currently running"""
    global _server_instance
    return _server_instance is not None and _server_instance.is_running()