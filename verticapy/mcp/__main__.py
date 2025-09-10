# verticapy/mcp/__main__.py
"""
Command-line entry point for VerticaPy MCP server.

Usage:
    python -m verticapy.mcp [--host HOST] [--port PORT]
"""

import argparse
import sys
from .server import start

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Start VerticaPy MCP server",
        prog="python -m verticapy.mcp"
    )
    
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="VerticaPy MCP Server 0.1.0"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Starting VerticaPy MCP server...")
        start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down VerticaPy MCP server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()