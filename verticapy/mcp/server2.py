# server.py

from mcp.server.fastmcp import FastMCP
from typing import Any
import verticapy as vp
import numpy as np
from decimal import Decimal
import datetime
from verticapy._utils._sql._sys import _executeSQL
from connection import VerticaPyConnection
from verticapy.core.tablesample.base import TableSample

# Initialize FastMCP server
mcp = FastMCP("verticapy")

# -----------------------------
# Connection setup (run once)
# -----------------------------

# Global connection manager
connection_manager = VerticaPyConnection()

def _to_json_serializable(obj: Any):
    """
    Convert VerticaPy / Python objects into JSON-serializable primitives.
    - Handles TableSample (uses .values)
    - Handles dicts/lists recursively
    - Handles numpy, Decimal, datetime
    """
    # None
    if obj is None:
        return None

    # VerticaPy TableSample
    if isinstance(obj, TableSample):
        return _to_json_serializable(obj.values)

    # dict
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]

    # numpy array
    if isinstance(obj, np.ndarray):
        return _to_json_serializable(obj.tolist())

    # numpy scalar types
    if isinstance(obj, (np.integer, np.int_, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_, np.float64)):
        return float(obj)

    # Decimal
    if isinstance(obj, Decimal):
        try:
            f = float(obj)
            return int(f) if f.is_integer() else f
        except Exception:
            return str(obj)

    # datetime
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    # pandas / numpy scalars with .item()
    try:
        if hasattr(obj, "item"):
            return _to_json_serializable(obj.item())
    except Exception:
        pass

    # fallback for primitives
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # fallback to str
    return str(obj)

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
    Describe a table using VerticaPy vDataFrame.
    Includes row count, column names + types, and stats for numeric columns.
    
    Args:
        table (str): Table name to describe. Can be schema.table or just table name.
    
    Returns:
        dict: Dictionary with row count, columns info, and numeric stats.
    """
    try:
        # Ensure connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {"success": False, "error": f"Connection failed: {message}"}
        
        # Create vDataFrame
        vdf = vp.vDataFrame(table)
        
        # Row/column counts
        row_count, col_count = vdf.shape()
        
        # Column names + types
        dtypes_info = vdf.dtypes()
        column_details = {
            "index": [col.strip('"') for col in dtypes_info.keys()],
            "dtype": list(dtypes_info.values())
        }
        
        # Stats for numeric columns
        stats = {}
        try:
            stats = vdf.describe().values
            stats["index"] = [col.strip('"') for col in stats["index"]]
        except Exception:
            # Some tables may not have numerical columns
            stats = {}
        
        return {
            "success": True,
            "table": table,
            "row_count": row_count,
            "column_count": col_count,
            "columns": column_details,
            "stats": stats,
            "method": "vdf.describe().values + vdf.dtypes()"
        }
    
    except Exception as e:
        return {"success": False, "error": str(e), "table": table}

    
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
    
# Supported metrics
AVAILABLE_METRICS = [
    "describe", "sum", "var", "std", "avg", "mean",
    "count", "max", "min", "median", "mode", "nunique",
    "topk", "nlargest", "nsmallest", "distinct", "aggregate"
]

@mcp.tool()
def column_stats(table: str, column: str, metric: str = "describe", **kwargs) -> dict:
    """
    MCP tool: return JSON-friendly statistics for a single column using VerticaPy vDataColumn.

    Args:
        table (str): Table name
        column (str): Column name
        metric (str): One of:
            describe, sum, var, std, avg, mean, count, max, min,
            median, mode, nunique, topk, nlargest, nsmallest,
            distinct, aggregate.

        nlargest -> get the n highest values from the table based on the select column
        nsmallest -> get the n lowest values from the table based on the select column
        topk -> get the top k repeated values along with their percentage
        nunique -> get the number of unique values in a column
        
        kwargs: Extra parameters, e.g.:
            - topk: {"k": 5}
            - nlargest: {"n": 5}
            - nsmallest: {"n": 5}
            - aggregate: {"func": ["min", "approx_50%", "max"]}
    Returns:
        dict: { success: bool, table, column, metric, result: <json-serializable> | error }
    """
    try:
        # validate metric
        metric = (metric or "describe").lower()
        if metric not in AVAILABLE_METRICS:
            return {
                "success": False,
                "table": table,
                "column": column,
                "metric": metric,
                "error": f"Unsupported metric '{metric}'. Choose from: {AVAILABLE_METRICS}"
            }

        # ensure connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {"success": False, "error": f"Connection failed: {message}", "table": table, "column": column}

        # build vDataFrame
        vdf = vp.vDataFrame(table)

        # resolve actual column name from vdf.get_columns() (handles quoted names)
        available_cols = vdf.get_columns()  # e.g. ['"date"', '"unit_price"', ...]
        # map to stripped names
        stripped_map = {c.strip('"'): c for c in available_cols}
        col_key = column.strip('"')
        actual_col = stripped_map.get(col_key)
        if actual_col is None:
            # try case-insensitive match
            for k, v in stripped_map.items():
                if k.lower() == col_key.lower():
                    actual_col = v
                    break

        if actual_col is None:
            return {
                "success": False,
                "table": table,
                "column": column,
                "error": f"Column '{column}' not found. Available columns: {[_c for _c in available_cols]}"
            }
        # access vDataColumn
        col = vdf[actual_col]

        # compute metric (selective)
        try:
            if metric == "describe":
                raw = col.describe().values
            elif metric in ("avg", "mean"):
                # try avg(), fallback to mean()
                if hasattr(col, "avg"):
                    raw = col.avg()
                elif hasattr(col, "mean"):
                    raw = col.mean()
                else:
                    raise AttributeError("No mean/avg method available on vDataColumn")
            elif metric == "sum":
                raw = col.sum()
            elif metric == "var":
                raw = col.var()
            elif metric == "std":
                raw = col.std()
            elif metric == "count":
                raw = col.count()
            elif metric == "max":
                raw = col.max()
            elif metric == "min":
                raw = col.min()
            elif metric == "median":
                # some vDataColumn objects may have median()
                if hasattr(col, "median"):
                    raw = col.median()
                else:
                    # fallback to aggregate approx_50%
                    raw = col.aggregate(func=["approx_50%"]).values
            elif metric == "mode":
                # mode may return a single value or list
                raw = col.mode()
            elif metric == "nunique":
                raw = col.nunique()
            elif metric == "topk":
                k = int(kwargs.get("k", 3))
                raw = col.topk(k).values
            elif metric == "nlargest":
                n = int(kwargs.get("n", 5))
                raw = col.nlargest(n).values
            elif metric == "nsmallest":
                n = int(kwargs.get("n", 5))
                raw = col.nsmallest(n).values
            elif metric == "distinct":
                # distinct returns an iterable/list of distinct values
                raw = list(col.distinct())
            elif metric == "aggregate":
                func_list = kwargs.get("func", ["min", "approx_10%", "approx_50%", "approx_90%", "max"])
                raw = col.aggregate(func=func_list).values
            else:
                # should not happen due to earlier validation
                return {"success": False, "error": f"Unhandled metric '{metric}'", "table": table, "column": column}
        except Exception as metric_exc:
            return {
                "success": False,
                "table": table,
                "column": column,
                "metric": metric,
                "error": f"Failed to compute metric '{metric}': {str(metric_exc)}"
            }

        # serialize result
        result = _to_json_serializable(raw)

        return {
            "success": True,
            "table": table,
            "column": col_key,           # return stripped / human-friendly name
            "metric": metric,
            "result": result,
            "method": "vDataColumn",
        }

    except Exception as e:
        return {"success": False, "table": table, "column": column, "error": str(e)}


@mcp.tool()
def table_stats(table: str, metric: str = "describe", columns: list = None, **kwargs) -> dict:
    """
    MCP tool: return JSON-friendly statistics for an entire table using VerticaPy vDataFrame.

    Args:
        table (str): Table name
        metric (str): One of:
            describe, sum, var, std, avg, mean, count, max, min,
            median, nunique, aggregate.
        columns (list, optional): List of columns to analyze. If None, all numeric columns are used.
        
        kwargs: Extra parameters for specific metrics:
            - aggregate: {"func": ["min", "approx_50%", "max"]}
    
    Returns:
        dict: { success: bool, table, metric, result: <json-serializable> | error }
    """
    try:
        # Validate metric - only include metrics that work at table level
        table_level_metrics = [
            "describe", "sum", "var", "std", "avg", "mean",
            "count", "max", "min", "median", "nunique", "aggregate"
        ]
        
        metric = (metric or "describe").lower()
        if metric not in table_level_metrics:
            return {
                "success": False,
                "table": table,
                "metric": metric,
                "error": f"Unsupported table-level metric '{metric}'. Choose from: {table_level_metrics}"
            }

        # Ensure connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {"success": False, "error": f"Connection failed: {message}", "table": table}

        # Build vDataFrame
        vdf = vp.vDataFrame(table)

        # Handle column selection
        if columns is not None:
            # Validate and resolve column names
            available_cols = vdf.get_columns()
            stripped_map = {c.strip('"'): c for c in available_cols}
            
            resolved_columns = []
            for col in columns:
                col_key = col.strip('"')
                actual_col = stripped_map.get(col_key)
                if actual_col is None:
                    # Try case-insensitive match
                    for k, v in stripped_map.items():
                        if k.lower() == col_key.lower():
                            actual_col = v
                            break
                
                if actual_col is None:
                    return {
                        "success": False,
                        "table": table,
                        "metric": metric,
                        "error": f"Column '{col}' not found. Available columns: {list(stripped_map.keys())}"
                    }
                resolved_columns.append(actual_col)
            
            # Use specific columns
            columns_param = resolved_columns
        else:
            # Let vDataFrame determine appropriate columns (usually numeric ones)
            columns_param = None

        # Compute metric
        try:
            if metric == "describe":
                if columns_param:
                    raw = vdf.describe(columns=columns_param).values
                else:
                    raw = vdf.describe().values
            elif metric in ("avg", "mean"):
                if columns_param:
                    raw = vdf.avg(columns=columns_param)
                else:
                    raw = vdf.avg()
            elif metric == "sum":
                if columns_param:
                    raw = vdf.sum(columns=columns_param)
                else:
                    raw = vdf.sum()
            elif metric == "var":
                if columns_param:
                    raw = vdf.var(columns=columns_param)
                else:
                    raw = vdf.var()
            elif metric == "std":
                if columns_param:
                    raw = vdf.std(columns=columns_param)
                else:
                    raw = vdf.std()
            elif metric == "count":
                if columns_param:
                    raw = vdf.count(columns=columns_param)
                else:
                    raw = vdf.count()
            elif metric == "max":
                if columns_param:
                    raw = vdf.max(columns=columns_param)
                else:
                    raw = vdf.max()
            elif metric == "min":
                if columns_param:
                    raw = vdf.min(columns=columns_param)
                else:
                    raw = vdf.min()
            elif metric == "median":
                if columns_param:
                    raw = vdf.median(columns=columns_param)
                else:
                    raw = vdf.median()
            elif metric == "nunique":
                if columns_param:
                    raw = vdf.nunique(columns=columns_param)
                else:
                    raw = vdf.nunique()
            elif metric == "aggregate":
                func_list = kwargs.get("func", ["min", "approx_10%", "approx_50%", "approx_90%", "max"])
                if columns_param:
                    raw = vdf.aggregate(func=func_list, columns=columns_param).values
                else:
                    raw = vdf.aggregate(func=func_list).values
            else:
                # Should not happen due to earlier validation
                return {"success": False, "error": f"Unhandled metric '{metric}'", "table": table}
        
        except Exception as metric_exc:
            return {
                "success": False,
                "table": table,
                "metric": metric,
                "error": f"Failed to compute table metric '{metric}': {str(metric_exc)}"
            }

        # Serialize result
        result = _to_json_serializable(raw)

        return {
            "success": True,
            "table": table,
            "metric": metric,
            "columns_analyzed": columns if columns else "auto_selected",
            "result": result,
            "method": "vDataFrame",
        }

    except Exception as e:
        return {"success": False, "table": table, "metric": metric, "error": str(e)}


# -----------------------------
# Data Transformation Tools
# -----------------------------

# Global storage for transformed vDataFrames
_vdf_cache = {}

@mcp.tool()
def transform_data(
    table: str, 
    operation: str, 
    vdf_id: str = None,
    show_preview: bool = True,
    **kwargs
) -> dict:
    """
    Transform data using VerticaPy vDataFrame operations like groupby, join, pivot, etc.
    
    Args:
        table (str): Source table name or existing vdf_id from cache
        operation (str): Type of transformation:
            - "groupby": Group data and aggregate
            - "join": Join with another table
            - "pivot": Create pivot table
            - "filter": Filter rows based on conditions
            - "select": Select specific columns
            - "sort": Sort data
        vdf_id (str, optional): Unique ID to store the result vDataFrame for later use
        show_preview (bool): Whether to show first 10 rows of result
        
        **kwargs: Operation-specific parameters:
        
        For groupby:
            - columns (list): Columns to group by
            - expr (list): Aggregation expressions, e.g., ["sum(revenue) AS daily_total"]
        
        For join:
            - right_table (str): Table or vdf_id to join with
            - on (str or list): Join condition(s)
            - how (str): Join type ("inner", "left", "right", "full")
        
        For pivot:
            - columns (list): Columns to pivot on
            - values (str): Column containing values
            - aggfunc (str): Aggregation function (default: "sum")
        
        For filter:
            - condition (str): SQL WHERE condition
        
        For select:
            - columns (list): Columns to select
        
        For sort:
            - columns (list): Columns to sort by
            - ascending (bool or list): Sort order (default: True)
    
    Returns:
        dict: Transformation result with preview data and vdf_id for reuse
    """
    try:
        # Ensure connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {"success": False, "error": f"Connection failed: {message}"}

        # Get source vDataFrame
        if table in _vdf_cache:
            # Use cached vDataFrame
            source_vdf = _vdf_cache[table]
            source_info = f"cached_vdf_{table}"
        else:
            # Create new vDataFrame from table
            try:
                source_vdf = vp.vDataFrame(table)
                source_info = f"table_{table}"
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create vDataFrame from table '{table}': {str(e)}"
                }

        # Perform transformation based on operation
        operation = operation.lower()
        
        try:
            if operation == "groupby":
                columns = kwargs.get("columns", [])
                expr = kwargs.get("expr", [])
                
                if not columns:
                    return {
                        "success": False,
                        "error": "groupby operation requires 'columns' parameter"
                    }
                
                if not expr:
                    return {
                        "success": False,
                        "error": "groupby operation requires 'expr' parameter with aggregation expressions"
                    }
                
                result_vdf = source_vdf.groupby(columns=columns, expr=expr)
                operation_info = f"groupby(columns={columns}, expr={expr})"
            
            elif operation == "join":
                right_table = kwargs.get("right_table")
                on = kwargs.get("on")
                how = kwargs.get("how", "inner")
                
                if not right_table or not on:
                    return {
                        "success": False,
                        "error": "join operation requires 'right_table' and 'on' parameters"
                    }
                
                # Get right vDataFrame
                if right_table in _vdf_cache:
                    right_vdf = _vdf_cache[right_table]
                else:
                    right_vdf = vp.vDataFrame(right_table)
                
                result_vdf = source_vdf.join(right_vdf, on=on, how=how)
                operation_info = f"join(right_table={right_table}, on={on}, how={how})"
            
            elif operation == "pivot":
                columns = kwargs.get("columns", [])
                values = kwargs.get("values")
                aggfunc = kwargs.get("aggfunc", "sum")
                
                if not columns or not values:
                    return {
                        "success": False,
                        "error": "pivot operation requires 'columns' and 'values' parameters"
                    }
                
                result_vdf = source_vdf.pivot(columns=columns, values=values, aggfunc=aggfunc)
                operation_info = f"pivot(columns={columns}, values={values}, aggfunc={aggfunc})"
            
            elif operation == "filter":
                condition = kwargs.get("condition")
                
                if not condition:
                    return {
                        "success": False,
                        "error": "filter operation requires 'condition' parameter"
                    }
                
                result_vdf = source_vdf.filter(condition)
                operation_info = f"filter(condition={condition})"
            
            elif operation == "select":
                columns = kwargs.get("columns", [])
                
                if not columns:
                    return {
                        "success": False,
                        "error": "select operation requires 'columns' parameter"
                    }
                
                result_vdf = source_vdf[columns]
                operation_info = f"select(columns={columns})"
            
            elif operation == "sort":
                columns = kwargs.get("columns", [])
                ascending = kwargs.get("ascending", True)
                
                if not columns:
                    return {
                        "success": False,
                        "error": "sort operation requires 'columns' parameter"
                    }
                
                result_vdf = source_vdf.sort(columns, ascending=ascending)
                operation_info = f"sort(columns={columns}, ascending={ascending})"
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation '{operation}'. Supported: groupby, join, pivot, filter, select, sort"
                }
        
        except Exception as op_error:
            return {
                "success": False,
                "error": f"Failed to execute {operation} operation: {str(op_error)}",
                "operation": operation,
                "kwargs": kwargs
            }

        # Generate vdf_id if not provided
        if not vdf_id:
            import time
            vdf_id = f"{operation}_{int(time.time())}"
        
        # Store result in cache
        _vdf_cache[vdf_id] = result_vdf
        
        # Get result info
        try:
            result_shape = result_vdf.shape()
            result_columns = result_vdf.get_columns()
            clean_columns = [col.strip('"') for col in result_columns]
        except Exception:
            result_shape = (0, 0)
            clean_columns = []
        
        # Prepare response
        response = {
            "success": True,
            "source": source_info,
            "operation": operation_info,
            "vdf_id": vdf_id,
            "result_shape": result_shape,
            "result_columns": clean_columns,
            "cached": True
        }
        
        # Add preview if requested
        if show_preview and result_shape[0] > 0:
            try:
                # Get first 10 rows as JSON
                preview_vdf = result_vdf[:10]
                preview_json = preview_vdf.to_json()
                
                # Parse JSON string to object for cleaner output
                import json
                preview_data = json.loads(preview_json)
                
                response["preview"] = {
                    "rows_shown": len(preview_data),
                    "data": preview_data,
                    "format": "json"
                }
            except Exception as preview_error:
                # Fallback to basic info if preview fails
                response["preview_error"] = f"Could not generate preview: {str(preview_error)}"
        
        return response
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error in transform_data: {str(e)}",
            "operation": operation
        }


@mcp.tool()
def list_cached_vdfs() -> dict:
    """
    List all cached vDataFrames available for use in other operations.
    
    Returns:
        dict: Information about all cached vDataFrames
    """
    try:
        vdf_info = {}
        
        for vdf_id, vdf in _vdf_cache.items():
            try:
                shape = vdf.shape()
                columns = vdf.get_columns()
                clean_columns = [col.strip('"') for col in columns]
                
                vdf_info[vdf_id] = {
                    "shape": shape,
                    "columns": clean_columns,
                    "column_count": len(clean_columns)
                }
            except Exception as e:
                vdf_info[vdf_id] = {
                    "error": f"Could not get info: {str(e)}"
                }
        
        return {
            "success": True,
            "cached_vdfs": vdf_info,
            "count": len(_vdf_cache)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to list cached vDataFrames: {str(e)}"
        }


@mcp.tool()
def clear_vdf_cache(vdf_id: str = None) -> dict:
    """
    Clear cached vDataFrames.
    
    Args:
        vdf_id (str, optional): Specific vDataFrame ID to remove. If None, clears all.
    
    Returns:
        dict: Operation result
    """
    try:
        global _vdf_cache
        
        if vdf_id:
            if vdf_id in _vdf_cache:
                del _vdf_cache[vdf_id]
                return {
                    "success": True,
                    "message": f"Cleared vDataFrame '{vdf_id}' from cache",
                    "remaining_count": len(_vdf_cache)
                }
            else:
                return {
                    "success": False,
                    "error": f"vDataFrame '{vdf_id}' not found in cache"
                }
        else:
            cleared_count = len(_vdf_cache)
            _vdf_cache.clear()
            return {
                "success": True,
                "message": f"Cleared all {cleared_count} vDataFrames from cache",
                "remaining_count": 0
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to clear cache: {str(e)}"
        }


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