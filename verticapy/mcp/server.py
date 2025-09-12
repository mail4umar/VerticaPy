# verticapy_mcp_server.py

import json
from typing import Any, Dict
import verticapy as vp
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("verticapy")

# Global connection info - you can modify this as needed
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
    
    def connect(self) -> bool:
        """Establish connection to Vertica database."""
        try:
            vp.new_connection(
                CONN_INFO,
                name=self.connection_name,
                auto=True,
                overwrite=True,
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.is_connected = False
            return False
    
    def ensure_connected(self) -> bool:
        """Ensure we have an active connection."""
        if not self.is_connected:
            return self.connect()
        return True

# Global connection manager
connection_manager = VerticaPyConnection()

@mcp.tool()
async def connect_to_vertica() -> str:
    """Connect to Vertica database using VerticaPy.
    
    Returns:
        Status message indicating success or failure
    """
    try:
        success = connection_manager.connect()
        if success:
            return "Successfully connected to Vertica database"
        else:
            return "Failed to connect to Vertica database"
    except Exception as e:
        return f"Error connecting to database: {str(e)}"

@mcp.tool()
async def create_dataframe(data_dict: str,
                           name: str = None,
                           relation_type: str = "view",
                           usecols: list = None,
                           overwrite: bool = False) -> str:
    """Create a VerticaPy vDataFrame from a dictionary of data and optionally store it in the database.
    
    Args:
        data_dict: JSON string representation of dictionary containing column data
        name: Optional fully qualified name (e.g. '"public"."mytable"') to save in database
        relation_type: Relation type ("view", "temporary", "table", "local", "insert")
        usecols: Optional list of columns to save
        overwrite: If True, overwrite the existing table/view if it already exists
    
    Returns:
        Information about the created dataframe and (if applicable) storage result
    """
    try:
        # Ensure connection
        if not connection_manager.ensure_connected():
            return "Error: Could not establish database connection"
        
        # Parse JSON
        try:
            data = json.loads(data_dict)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON data: {str(e)}"
        
        # Create vDataFrame
        df = vp.vDataFrame(data)
        info = [f"Created vDataFrame with shape: {df.shape()}"]
        info.append(f"Columns: {df.get_columns()}")
        
        # Store to DB if requested
        if name:
            try:
                df.to_db(
                    name=name,
                    usecols=usecols,
                    relation_type=relation_type,
                    inplace=overwrite,   # key part
                )
                action = "Overwritten" if overwrite else "Saved"
                info.append(f"{action} vDataFrame in database as {name} ({relation_type})")
            except Exception as e:
                err_str = str(e)
                if "already exists" in err_str or "DuplicateObject" in err_str:
                    if overwrite:
                        info.append(f"Attempted overwrite failed: {err_str}")
                    else:
                        info.append(f"Table/View {name} already exists. Use overwrite=True to replace it.")
                else:
                    info.append(f"Error saving to database: {err_str}")
        
        # Show preview
        head_data = df.head(5)
        info.append(f"First 5 rows:\n{head_data}")
        
        return "\n".join(info)
    except Exception as e:
        return f"Error creating dataframe: {str(e)}"

@mcp.tool()
async def summarize_dataframe(table_name: str) -> str:
    """Summarize a vDataFrame by computing descriptive statistics.
    
    Args:
        table_name: Fully qualified table/view name (e.g. '"public"."cool"')
    
    Returns:
        JSON string with statistical summary of columns
    """
    try:
        if not connection_manager.ensure_connected():
            return "Error: Could not establish database connection"
        
        # Load the vDataFrame
        vdf = vp.vDataFrame(table_name)
        
        # Compute summary statistics
        summary = vdf.describe().values
        
        # Return JSON string
        return json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error summarizing dataframe: {str(e)}"


@mcp.tool()
async def fetch_dataframe(table_name: str) -> str:
    """Fetch a VerticaPy vDataFrame from the database by table name.
    
    Args:
        table_name: Fully qualified table/view name (e.g. '"public"."data_normalized"')
    
    Returns:
        Information about the dataframe (shape, columns, preview)
    """
    try:
        if not connection_manager.ensure_connected():
            return "Error: Could not establish database connection"
        
        df = vp.vDataFrame(table_name)
        info = []
        info.append(f"Fetched vDataFrame from {table_name}")
        info.append(f"Shape: {df.shape()}")
        info.append(f"Columns: {df.get_columns()}")
        info.append(f"Data types: {df.dtypes()}")
        head_data = df.head(5)
        info.append(f"First 5 rows:\n{head_data}")
        
        return "\n".join(info)
    except Exception as e:
        return f"Error fetching dataframe: {str(e)}"


@mcp.tool()
async def get_dataframe_info(table_name: str = None) -> str:
    """Get information about existing dataframes or tables in the database.
    
    Args:
        table_name: Optional name of specific table to get info about
    
    Returns:
        Information about available tables/dataframes
    """
    try:
        if not connection_manager.ensure_connected():
            return "Error: Could not establish database connection"
        
        if table_name:
            # Get info about specific table
            try:
                df = vp.vDataFrame(table_name)
                info = []
                info.append(f"Table: {table_name}")
                info.append(f"Shape: {df.shape()}")
                info.append(f"Columns: {df.get_columns()}")
                info.append(f"Data types: {df.dtypes()}")
                return "\n".join(info)
            except Exception as e:
                return f"Error accessing table '{table_name}': {str(e)}"
        else:
            # List available tables (this might need adjustment based on VerticaPy version)
            try:
                # This is a basic approach - you might need to adjust based on your VerticaPy version
                result = vp._executeSQL("SELECT table_name FROM tables WHERE table_schema = CURRENT_SCHEMA()")
                return f"Available tables: {result}"
            except Exception as e:
                return f"Error listing tables: {str(e)}"
    
    except Exception as e:
        return f"Error getting dataframe info: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')