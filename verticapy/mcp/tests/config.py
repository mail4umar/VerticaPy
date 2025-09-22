"""
MCP Test Configuration

Configuration settings for MCP server tests.
"""

import os
import sys
import logging
from configparser import ConfigParser

# Add the parent directory to Python path to access connection.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default test configuration - matches MCP connection syntax
CONN_INFO = {
   "host": "127.0.0.1",      # or "host.docker.internal" if using Docker Desktop
   "port": 5433,
   "database": "demo",
   "user": "dbadmin",
   "password": "",
   "tls_verify": False
}

DEFAULT_TEST_CONFIG = {
    **CONN_INFO,
    "log_level": logging.INFO,
    "test_schema": "public",
    "test_table": "titanic",
}

def load_test_config():
    """Load test configuration from environment variables or config file."""
    config = DEFAULT_TEST_CONFIG.copy()
    
    # Try to load from config file if it exists
    config_file = os.path.join(os.path.dirname(__file__), "test_config.conf")
    if os.path.exists(config_file):
        parser = ConfigParser()
        parser.read(config_file)
        if parser.has_section("mcp_test_config"):
            for key, value in parser.items("mcp_test_config"):
                if key in config:
                    # Convert to appropriate type
                    if key == "port":
                        config[key] = str(value)  # Keep port as string to match MCP
                    elif key == "log_level":
                        config[key] = getattr(logging, value.upper(), logging.INFO)
                    else:
                        config[key] = value
    
    # Override with environment variables if they exist (matching MCP pattern)
    env_mappings = {
        "VERTICA_HOST": "host",
        "VERTICA_PORT": "port", 
        "VERTICA_DATABASE": "database",
        "VERTICA_USER": "user",
        "VERTICA_PASSWORD": "password",
    }
    
    for env_var, config_key in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            config[config_key] = env_value
    
    return config

def get_conn_info():
    """Get connection info in the exact format used by MCP"""
    config = load_test_config()
    return {
        "host": config["host"],
        "port": config["port"],
        "database": config["database"],
        "password": config["password"],
        "user": config["user"],
        "tls_verify": config.get("tls_verify", False)
    }

def create_test_connection_manager():
    """Create a VerticaPyConnection instance configured for testing"""
    try:
        from connection import VerticaPyConnection
        return VerticaPyConnection()
    except ImportError as e:
        raise ImportError(f"Could not import VerticaPyConnection: {e}. Make sure connection.py is available.")