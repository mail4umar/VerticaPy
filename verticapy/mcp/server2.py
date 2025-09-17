# server.py

from mcp.server.fastmcp import FastMCP
import verticapy as vp
from verticapy._utils._sql._sys import _executeSQL
from connection import VerticaPyConnection

# Initialize FastMCP server
mcp = FastMCP("verticapy")

# -----------------------------
# Connection setup (run once)
# -----------------------------

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
def list_tables(schema: str = "public") -> dict:
    """
    List all tables in the specified schema.
    
    Args:
        schema (str): Schema name to list tables from. Defaults to "public".
    
    Returns:
        dict: Dictionary containing tables list, count, and metadata
    """
    try:
        # Ensure we have an active connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {
                "success": False,
                "error": f"Connection failed: {message}",
                "tables": [],
                "count": 0
            }
        
        # Method 1: Using VerticaPy's built-in function (preferred)
        try:
            # This uses vp.get_data_types() internally but filters for tables
            tables_info = vp.get_data_types(schema_name=schema, table_name='*')
            
            # Extract unique table names
            table_names = []
            if tables_info:
                # tables_info is usually a list of tuples: (schema, table, column, type, ...)
                table_names = list(set([row[1] for row in tables_info if row[0].lower() == schema.lower()]))
                table_names.sort()
            
            return {
                "success": True,
                "schema": schema,
                "tables": table_names,
                "count": len(table_names),
                "method": "verticapy_builtin"
            }
            
        except Exception as builtin_error:
            # Method 2: Fallback to direct SQL query
            try:
                sql_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = ? 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
                
                result = _executeSQL(
                    query=sql_query,
                    parameters=[schema],
                    method="fetchall"
                )
                
                table_names = [row[0] for row in result]
                
                return {
                    "success": True,
                    "schema": schema,
                    "tables": table_names,
                    "count": len(table_names),
                    "method": "direct_sql",
                    "warning": f"Used fallback method due to: {str(builtin_error)}"
                }
                
            except Exception as sql_error:
                # Method 3: Alternative SQL approach
                try:
                    alt_query = f"""
                    SELECT table_name 
                    FROM v_catalog.tables 
                    WHERE table_schema = '{schema}' 
                    ORDER BY table_name
                    """
                    
                    result = _executeSQL(
                        query=alt_query,
                        method="fetchall"
                    )
                    
                    table_names = [row[0] for row in result]
                    
                    return {
                        "success": True,
                        "schema": schema,
                        "tables": table_names,
                        "count": len(table_names),
                        "method": "v_catalog",
                        "warning": f"Used alternative method due to previous errors"
                    }
                    
                except Exception as final_error:
                    return {
                        "success": False,
                        "error": f"All methods failed. Last error: {str(final_error)}",
                        "schema": schema,
                        "tables": [],
                        "count": 0,
                        "attempted_methods": ["verticapy_builtin", "direct_sql", "v_catalog"]
                    }
                    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "schema": schema,
            "tables": [],
            "count": 0
        }

@mcp.tool()
def list_all_schemas() -> dict:
    """
    List all available schemas in the database.
    
    Returns:
        dict: Dictionary containing schemas list and count
    """
    try:
        # Ensure we have an active connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {
                "success": False,
                "error": f"Connection failed: {message}",
                "schemas": [],
                "count": 0
            }
        
        try:
            # Query to get all schemas
            sql_query = """
            SELECT schema_name 
            FROM information_schema.schemata 
            ORDER BY schema_name
            """
            
            result = _executeSQL(
                query=sql_query,
                method="fetchall"
            )
            
            schema_names = [row[0] for row in result]
            
            return {
                "success": True,
                "schemas": schema_names,
                "count": len(schema_names)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to retrieve schemas: {str(e)}",
                "schemas": [],
                "count": 0
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "schemas": [],
            "count": 0
        }

@mcp.tool()
def describe_table(table: str) -> dict:
    """
    Describe schema of a table using VerticaPy vDataFrame.
    
    Args:
        table (str): Table name to describe. Can be schema.table or just table name.
    
    Returns:
        dict: Dictionary containing table schema information including columns, types, and basic stats.
    """
    try:
        # Ensure we have an active connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {
                "success": False,
                "error": f"Connection failed: {message}",
                "table": table
            }
        
        # Create vDataFrame from table
        try:
            vdf = vp.vDataFrame(table)
            
            # Get basic information
            columns = vdf.get_columns()
            dtypes_info = vdf.dtypes()
            shape = vdf.shape()
            
            # Get detailed column information with data types
            column_details = []
            for col in columns:
                col_clean = col.strip('"')  # Remove quotes from column names
                try:
                    # Get data type for this column
                    col_dtype = dtypes_info.get(col, "unknown")
                    
                    column_details.append({
                        "column_name": col_clean,
                        "data_type": str(col_dtype),
                        "quoted_name": col  # Keep original quoted name if needed
                    })
                except Exception as col_error:
                    # If we can't get dtype for a specific column, still include it
                    column_details.append({
                        "column_name": col_clean,
                        "data_type": "unknown",
                        "quoted_name": col,
                        "error": str(col_error)
                    })
            
            # Try to get additional table information
            table_info = {}
            try:
                # Get basic statistics if the table is not too large
                if shape[0] < 1000000:  # Only for tables with less than 1M rows
                    describe_df = vdf.describe()
                    table_info["has_statistics"] = True
                    table_info["describe_available"] = True
                else:
                    table_info["has_statistics"] = False
                    table_info["describe_available"] = False
                    table_info["note"] = "Statistics skipped for large table"
            except Exception:
                table_info["has_statistics"] = False
                table_info["describe_available"] = False
            
            return {
                "success": True,
                "table": table,
                "columns": column_details,
                "column_count": len(columns),
                "row_count": shape[0],
                "shape": {"rows": shape[0], "columns": shape[1]},
                "dtypes": dtypes_info,
                "table_info": table_info,
                "method": "verticapy_vdataframe"
            }
            
        except Exception as vdf_error:
            # If vDataFrame creation fails, the table might not exist or we don't have access
            error_msg = str(vdf_error)
            
            # Check if it's a "relation does not exist" error
            if "does not exist" in error_msg.lower() or "relation" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"Table '{table}' does not exist or is not accessible",
                    "table": table,
                    "detailed_error": error_msg
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create vDataFrame for table '{table}': {error_msg}",
                    "table": table,
                    "detailed_error": error_msg
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "table": table
        }
    
@mcp.tool()
def sample_data(table: str, n: int = 5) -> dict:
    """
    Return sample rows from a table using VerticaPy vDataFrame.
    
    Args:
        table (str): Table name to sample from. Can be schema.table or just table name.
        n (int): Number of rows to sample. Defaults to 5.
    
    Returns:
        dict: Dictionary containing sampled data, columns, and metadata.
    """
    try:
        # Ensure we have an active connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {
                "success": False,
                "error": f"Connection failed: {message}",
                "table": table,
                "data": []
            }
        
        # Create vDataFrame from table
        try:
            vdf = vp.vDataFrame(table)
            
            # Get basic info
            shape = vdf.shape()
            columns = vdf.get_columns()
            
            # Determine sampling method based on table size and requested sample size
            if n >= shape[0]:
                # If we want more rows than exist, just get all rows
                sample_vdf = vdf.head(shape[0])
                sampling_method = "head_all_rows"
            elif n <= 10:
                # For small samples, use head() for consistency
                sample_vdf = vdf.head(n)
                sampling_method = "head"
            else:
                # For larger samples, use random sampling
                try:
                    sample_vdf = vdf.sample(n=n, method="random")
                    sampling_method = "random"
                except Exception:
                    # Fallback to head if sampling fails
                    sample_vdf = vdf.head(n)
                    sampling_method = "head_fallback"
            
            # Convert to pandas for easy JSON serialization
            try:
                pandas_df = sample_vdf.to_pandas()
                
                # Convert to JSON-serializable format
                data_records = pandas_df.to_dict('records')
                
                # Clean up column names (remove quotes)
                clean_columns = [col.strip('"') for col in columns]
                
                return {
                    "success": True,
                    "table": table,
                    "columns": clean_columns,
                    "data": data_records,
                    "sample_size": len(data_records),
                    "requested_size": n,
                    "total_rows": shape[0],
                    "total_columns": shape[1],
                    "sampling_method": sampling_method,
                    "method": "verticapy_vdataframe"
                }
                
            except Exception as pandas_error:
                # If pandas conversion fails, try to get data as list
                try:
                    # Get data as lists
                    data_list = sample_vdf.to_list()
                    
                    # Create records manually
                    clean_columns = [col.strip('"') for col in columns]
                    data_records = []
                    
                    for row in data_list:
                        if len(row) == len(clean_columns):
                            record = dict(zip(clean_columns, row))
                            data_records.append(record)
                    
                    return {
                        "success": True,
                        "table": table,
                        "columns": clean_columns,
                        "data": data_records,
                        "sample_size": len(data_records),
                        "requested_size": n,
                        "total_rows": shape[0],
                        "total_columns": shape[1],
                        "sampling_method": sampling_method,
                        "method": "verticapy_vdataframe_list",
                        "warning": f"Used list conversion due to pandas error: {str(pandas_error)}"
                    }
                    
                except Exception as list_error:
                    return {
                        "success": False,
                        "error": f"Failed to convert sample data: pandas error: {str(pandas_error)}, list error: {str(list_error)}",
                        "table": table,
                        "data": []
                    }
            
        except Exception as vdf_error:
            error_msg = str(vdf_error)
            
            if "does not exist" in error_msg.lower() or "relation" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"Table '{table}' does not exist or is not accessible",
                    "table": table,
                    "data": []
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create vDataFrame for table '{table}': {error_msg}",
                    "table": table,
                    "data": []
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "table": table,
            "data": []
        }
    
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