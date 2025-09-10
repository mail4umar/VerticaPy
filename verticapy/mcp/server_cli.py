"""
CLI module for running VerticaPy MCP server.
"""

import argparse
import logging
import sys
from typing import Optional

from .server import MCPServer
from .config import MCPConfig


def create_config_from_args(args) -> MCPConfig:
    """Create MCPConfig from command line arguments."""
    config = MCPConfig()
    
    if args.name:
        config.name = args.name
    if args.tools:
        config.tools = args.tools.split(',')
    if args.max_dataframes:
        config.max_dataframes = args.max_dataframes
    if args.max_models:
        config.max_models = args.max_models
    
    return config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VerticaPy MCP Server - Expose VerticaPy functionality via Model Context Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default configuration
  python -m verticapy.mcp.server
  
  # Start with custom tools
  python -m verticapy.mcp.server --tools "create_vdataframe,train_model,create_plot"
  
  # Start with resource limits
  python -m verticapy.mcp.server --max-dataframes 5 --max-models 3

Claude Desktop Configuration:
  Add this to your Claude config file:
  
  "mcpServers": {
    "verticapy": {
      "command": "python",
      "args": ["-m", "verticapy.mcp.server"]
    }
  }
        """
    )
    
    parser.add_argument(
        "--name",
        default="verticapy",
        help="Server name (default: verticapy)"
    )
    
    parser.add_argument(
        "--tools", 
        help="Comma-separated list of tools to enable (default: all)"
    )
    
    parser.add_argument(
        "--max-dataframes",
        type=int,
        default=10,
        help="Maximum number of vDataFrames in session (default: 10)"
    )
    
    parser.add_argument(
        "--max-models",
        type=int,
        default=5,
        help="Maximum number of ML models in session (default: 5)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path (default: stderr)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_handlers = []
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))
    else:
        log_handlers.append(logging.StreamHandler(sys.stderr))
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Create and start server
    server = MCPServer(config)
    server.start()


if __name__ == "__main__":
    main()