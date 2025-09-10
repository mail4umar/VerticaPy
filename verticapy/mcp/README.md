# VerticaPy MCP Server Setup Guide

This guide will help you set up and test the VerticaPy MCP (Model Context Protocol) server.

## Prerequisites

- Python 3.8+
- VerticaPy installed
- `websockets` library: `pip install websockets`
- Access to a Vertica database (optional for basic testing)

## Directory Structure

Add these files to your VerticaPy installation:

```
verticapy/
├── mcp/
│   ├── __init__.py          # Module initialization
│   ├── server.py            # Main MCP server
│   ├── tools.py             # Tool registry and handlers
│   └── client.py            # Client helper functions
└── ...
```

## Installation Steps

### 1. Create the MCP Module Directory

```bash
cd /path/to/verticapy
mkdir mcp
```

### 2. Add the Module Files

Copy the provided files into the `verticapy/mcp/` directory:
- `__init__.py`
- `server.py` 
- `tools.py`
- `client.py`

### 3. Install Dependencies

```bash
pip install websockets
```

### 4. Update Main VerticaPy __init__.py

Add this line to `verticapy/__init__.py`:

```python
# Add MCP module
from . import mcp
```

## Testing the Installation

### Option 1: Use the Test Notebook

1. Copy the `test_mcp_components.ipynb` content into a Jupyter notebook
2. Run each cell individually to test components
3. This will help you identify any issues before starting the full server

### Option 2: Manual Testing

```python
# Test basic functionality
import verticapy as vp
from verticapy.mcp.tools import ToolRegistry

# Create tool registry
registry = ToolRegistry()
print("Available tools:", list(registry.tools.keys()))

# Test creating a dataframe
import asyncio

async def test():
    result = await registry.execute_tool("create_dataframe", {
        "data": {
            "y_true": [1, 1.5, 3, 2, 5],
            "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
        }
    })
    print("Result:", result)

asyncio.run(test())
```

## Running the MCP Server

### Option 1: Command Line

```bash
# Start the server
python -m verticapy.mcp.server

# With custom host/port
python -m verticapy.mcp.server --host localhost --port 8765
```

### Option 2: From Python/Jupyter

```python
import verticapy.mcp as mcp

# Quick start
mcp.start()

# Or with custom settings
from verticapy.mcp.client import start_mcp_server
server = start_mcp_server(host="localhost", port=8765)
```

## Claude Desktop Configuration

When you start the MCP server, it will display the configuration to add to Claude Desktop:

```json
"mcpServers": {
  "verticapy": {
    "command": "python",
    "args": ["-m", "verticapy.mcp.server", "--host", "localhost", "--port", "8765"]
  }
}
```

Add this to your Claude Desktop configuration file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

## Available Tools

### 1. create_dataframe

Creates a VerticaPy vDataFrame from dictionary data.

**Input:**
- `data` (object): Dictionary of column names to lists of values
- `name` (string, optional): Name for the dataframe

**Example:**
```json
{
  "data": {
    "y_true": [1, 1.5, 3, 2, 5],
    "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5]
  },
  "name": "my_dataframe"
}
```

### 2. get_connections

Lists available VerticaPy connections.

**Input:** None required

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all files are in the correct `verticapy/mcp/` directory
   - Check that `__init__.py` files are present
   - Verify VerticaPy is properly installed

2. **WebSocket Connection Issues**
   - Check if port 8765 is available
   - Try a different port: `--port 8766`
   - Ensure firewall allows localhost connections

3. **VerticaPy Connection Issues**
   - The server doesn't require a Vertica connection to start
   - Connection-dependent tools will fail gracefully if no connection exists

### Debug Mode

Run with debug logging:

```bash
python -m verticapy.mcp.server --log-level DEBUG
```

### Testing Individual Components

Use the provided test notebook to isolate and test specific components:

1. Tool registry functionality
2. DataFrame creation
3. Server start/stop operations
4. Error handling

## Next Steps

Once the basic server is working:

1. **Add More Tools**: Extend the ToolRegistry with additional VerticaPy functions
2. **Add SQL Support**: Implement `verticapy._executeSQL()` integration
3. **Add Analytics Functions**: Include metrics like `aic_score()`, etc.
4. **Performance Tools**: Add QueryProfiler integration

## Example Usage in Claude

Once configured, you can ask Claude:

> "Create a dataframe with columns y_true and y_pred containing the values [1, 1.5, 3, 2, 5] and [1.1, 1.55, 2.9, 2.01, 4.5] respectively"

Claude will use the MCP server to execute the `create_dataframe` tool and return the results.

## Support

If you encounter issues:

1. Check the server logs for error messages
2. Run the test notebook to isolate problems
3. Verify VerticaPy installation and configuration
4. Test with a minimal configuration first