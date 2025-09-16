# server.py

from mcp.server.fastmcp import FastMCP
import verticapy as vp

# Initialize FastMCP server
mcp = FastMCP("verticapy")

# -----------------------------
# Connection setup (run once)
# -----------------------------
CONN_INFO = {
    "host": "10.10.10.235",
    "port": "34101",
    "database": "verticadb21477",
    "password": "",
    "user": "ughumman",
}

class VerticaPyConnection:
    """Manages VerticaPy database connections."""
    
    def __init__(self):
        self.is_connected = False
        self.connection_name = "VerticaDSN"
    
    def connect(self) -> tuple[bool, str]:
        """Establish connection to Vertica database."""
        try:
            vp.new_connection(
                CONN_INFO,
                name=self.connection_name,
                auto=True,
                overwrite=True,
            )
            self.is_connected = True
            return True, "Successfully connected to Vertica database"
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            print(error_msg)
            self.is_connected = False
            return False, error_msg
    
    def ensure_connected(self) -> tuple[bool, str]:
        """Ensure we have an active connection."""
        if not self.is_connected:
            return self.connect()
        return True, "Already connected"
    
    def disconnect(self) -> tuple[bool, str]:
        """Disconnect from Vertica database."""
        try:
            if self.is_connected:
                vp.close_connection(self.connection_name)
                self.is_connected = False
                return True, "Successfully disconnected from Vertica database"
            else:
                return True, "Already disconnected"
        except Exception as e:
            error_msg = f"Disconnect failed: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def get_connection_status(self) -> dict:
        """Get current connection status and info."""
        return {
            "is_connected": self.is_connected,
            "connection_name": self.connection_name,
            "host": CONN_INFO["host"],
            "port": CONN_INFO["port"],
            "database": CONN_INFO["database"],
            "user": CONN_INFO["user"]
        }

# Global connection manager
connection_manager = VerticaPyConnection()

@mcp.tool()
def connect_to_vertica() -> dict:
    """
    Connect to Vertica database using the configured credentials.
    
    Returns:
        dict: Connection status with success/failure message and connection details
    """
    try:
        success, message = connection_manager.connect()
        
        result = {
            "success": success,
            "message": message,
            "connection_info": connection_manager.get_connection_status()
        }
        
        if success:
            # Test the connection by getting version info
            try:
                version_info = vp.version()
                result["verticapy_version"] = version_info
            except Exception as e:
                result["warning"] = f"Connected but couldn't retrieve version info: {str(e)}"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error during connection: {str(e)}",
            "connection_info": connection_manager.get_connection_status()
        }

@mcp.tool()
def disconnect_from_vertica() -> dict:
    """
    Disconnect from Vertica database.
    
    Returns:
        dict: Disconnection status with success/failure message
    """
    try:
        success, message = connection_manager.disconnect()
        return {
            "success": success,
            "message": message,
            "connection_info": connection_manager.get_connection_status()
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error during disconnection: {str(e)}",
            "connection_info": connection_manager.get_connection_status()
        }

@mcp.tool()
def get_connection_status() -> dict:
    """
    Get current connection status and details.
    
    Returns:
        dict: Current connection status and configuration details
    """
    return connection_manager.get_connection_status()


# -----------------------------
# Data Exploration Tools
# -----------------------------
@mcp.tool()
def list_tables(schema: str = "public"):
    """List tables in the given schema."""
    return {"tables": ["placeholder_table1", "placeholder_table2"]}

@mcp.tool()
def describe_table(table: str):
    """Describe schema of a table."""
    return {"table": table, "columns": ["col1", "col2"]}

@mcp.tool()
def sample_data(table: str, n: int = 5):
    """Return sample rows from a table."""
    return {"table": table, "rows": ["row1", "row2", "..."]}

@mcp.tool()
def summary_stats(table: str, column: str):
    """Return basic statistics for a column."""
    return {"table": table, "column": column, "mean": 0, "min": 0, "max": 0}


# -----------------------------
# Modeling Tools
# -----------------------------
@mcp.tool()
def train_model(table: str, target: str, model_type: str = "logistic_reg"):
    """Train a model in Vertica (placeholder)."""
    return {
        "table": table,
        "target": target,
        "model_type": model_type,
        "metrics": {"accuracy": 0.0, "auc": 0.0},
    }

@mcp.tool()
def predict(table: str, model_name: str):
    """Apply model to table (placeholder)."""
    return {"table": table, "model_name": model_name, "predictions": ["..."]}

@mcp.tool()
def list_models():
    """List trained models."""
    return {"models": ["placeholder_model1", "placeholder_model2"]}


# -----------------------------
# Query Profiling Tools
# -----------------------------
@mcp.tool()
def run_query(query: str):
    """Execute SQL query (placeholder)."""
    return {"query": query, "rows": ["row1", "row2"]}

@mcp.tool()
def profile_query(query: str):
    """Profile query using VerticaPy QueryProfiler (placeholder)."""
    return {
        "query": query,
        "duration_sec": 0,
        "operators": [
            {"name": "Scan", "time_sec": 0, "rows": 0},
        ],
    }

@mcp.tool()
def get_query_plan(query: str):
    """Get query plan (placeholder)."""
    return {"query": query, "plan": ["Step1", "Step2"]}

@mcp.tool()
def get_query_metrics(query: str):
    """Get query metrics summary (placeholder)."""
    return {"query": query, "metrics": {"cpu": 0, "io": 0}}


# -----------------------------
# Run MCP Server
# -----------------------------
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")