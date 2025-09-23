# server.py

from mcp.server.fastmcp import FastMCP
from typing import Any
import verticapy as vp
import numpy as np
from decimal import Decimal
import datetime
import json
import time
import verticapy as vp
import verticapy as vp
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._gen import gen_name
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
                verticapy_version = vp.__version__
                vertica_db_version = vp.vertica_version()
                result["verticapy_version"] = verticapy_version
                result["vertica_db_version"] = vertica_db_version
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
            FROM v_catalog.schemata 
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
        
        # Handle different dtypes return formats
        if hasattr(dtypes_info, 'values'):
            # If dtypes returns a TableSample-like object, convert to dict
            dtypes_dict = _to_json_serializable(dtypes_info.values)
            if isinstance(dtypes_dict, dict) and "index" in dtypes_dict and "dtype" in dtypes_dict:
                column_details = {
                    "index": [col.strip('"') for col in dtypes_dict["index"]],
                    "dtype": dtypes_dict["dtype"]
                }
            else:
                # Fallback if the structure is unexpected
                try:
                    columns = vdf.get_columns()
                    column_details = {
                        "index": [col.strip('"') for col in columns],
                        "dtype": ["unknown"] * len(columns)
                    }
                except Exception:
                    column_details = {"index": [], "dtype": []}
        elif isinstance(dtypes_info, dict):
            # If dtypes returns a regular dict
            column_details = {
                "index": [col.strip('"') for col in dtypes_info.keys()],
                "dtype": list(dtypes_info.values())
            }
        else:
            # Fallback: try to get column info differently
            try:
                columns = vdf.get_columns()
                column_details = {
                    "index": [col.strip('"') for col in columns],
                    "dtype": ["unknown"] * len(columns)
                }
            except Exception:
                column_details = {"index": [], "dtype": []}
        
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
def column_stats(table: str, column: str, metric: str = "describe", **extra_kwargs) -> dict:
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
        
        Extra parameters for specific metrics:
            - topk: k (int) - number of top values
            - nlargest: n (int) - number of largest values  
            - nsmallest: n (int) - number of smallest values
            - aggregate: func (list) - aggregation functions
    Returns:
        dict: { success: bool, table, column, metric, result: <json-serializable> | error }
    """
    try:
        # Parse extra_kwargs - MCP sends it as a JSON string in extra_kwargs parameter
        import json
        kwargs = {}
        
        # Check if extra_kwargs contains a JSON string to parse
        if 'extra_kwargs' in extra_kwargs:
            extra_kwargs_value = extra_kwargs['extra_kwargs']
            if isinstance(extra_kwargs_value, str):
                try:
                    kwargs = json.loads(extra_kwargs_value)
                except json.JSONDecodeError:
                    pass
            elif isinstance(extra_kwargs_value, dict):
                kwargs = extra_kwargs_value
        
        # Also add any other direct parameters
        for key, value in extra_kwargs.items():
            if key != 'extra_kwargs':
                kwargs[key] = value
        
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

        # Get vDataFrame - check cache first, then create from table
        if table in _vdf_cache:
            vdf = _vdf_cache[table]
        else:
            # build vDataFrame from table
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
def table_stats(table: str, metric: str = "describe", columns: list = None, **extra_kwargs) -> dict:
    """
    MCP tool: return JSON-friendly statistics for an entire table using VerticaPy vDataFrame.

    Args:
        table (str): Table name
        metric (str): One of:
            describe, sum, var, std, avg, mean, count, max, min,
            median, nunique, aggregate.
        columns (list, optional): List of columns to analyze. If None, all numeric columns are used.
        **extra_kwargs: Additional parameters for specific metrics (e.g., func for aggregate)
    
    Returns:
        dict: { success: bool, table, metric, result: <json-serializable> | error }
    """
    try:
        # Parse extra_kwargs - MCP sends it as a JSON string in extra_kwargs parameter
        import json
        kwargs = {}
        
        # Check if extra_kwargs contains a JSON string to parse
        if 'extra_kwargs' in extra_kwargs:
            extra_kwargs_value = extra_kwargs['extra_kwargs']
            if isinstance(extra_kwargs_value, str):
                try:
                    kwargs = json.loads(extra_kwargs_value)
                except json.JSONDecodeError:
                    pass
            elif isinstance(extra_kwargs_value, dict):
                kwargs = extra_kwargs_value
        
        # Also add any other direct parameters
        for key, value in extra_kwargs.items():
            if key != 'extra_kwargs':
                kwargs[key] = value
        
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

        # Get vDataFrame - check cache first, then create from table
        if table in _vdf_cache:
            vdf = _vdf_cache[table]
        else:
            # Build vDataFrame from table
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
    **extra_kwargs
) -> dict:
    """
    Transform data using VerticaPy vDataFrame operations like groupby, join, pivot, etc.
    
    Args:
        table (str): Source table name or existing vdf_id from cache
        operation (str): Type of transformation:
            - "groupby": Group data and aggregate
            - "join": Join with another table
            - "pivot": Create pivot table
            - "search": Search/filter rows based on conditions
            - "select": Select specific columns
            - "sort": Sort data
        vdf_id (str, optional): Unique ID to store the result vDataFrame for later use
        show_preview (bool): Whether to show first 10 rows of result
        
        kwargs (str): JSON string containing operation-specific parameters, or dict:
        
        For groupby:
            - columns (list): Columns to group by
            - expr (list, optional): Aggregation expressions, e.g., ["sum(revenue) AS daily_total"]
            - rollup (bool or list, optional): Whether to use ROLLUP (default: False)
            - having (str, optional): HAVING clause condition
        
        For join:
            - right_table (str): Table or vdf_id to join with
            - on (tuple/dict/list, optional): Join condition(s)
            - on_interpolate (dict, optional): Interpolation conditions
            - how (str): Join type ("left", "right", "cross", "full", "natural", "self", "inner", default: "natural")
            - expr1 (str/list, optional): Expressions for left table
            - expr2 (str/list, optional): Expressions for right table
        
        For pivot:
            - index (str): Column to use as index
            - columns (str): Column to pivot on
            - values (str): Column containing values to aggregate
            - aggr (str): Aggregation function (default: "sum")
            - prefix (str, optional): Prefix for new column names
        
        For search:
            - conditions (str): SQL WHERE conditions
            - usecols (str/list, optional): Columns to include in result
            - expr (str/list, optional): Additional expressions
            - order_by (str/dict/list, optional): Ordering specification
        
        For select:
            - columns (list): Columns to select
        
        For sort:
            - columns (dict/str/list): Sort specification
              - Dict format: {"column": "asc"/"desc"}
              - String/list with ascending param for backward compatibility
            - ascending (bool/list, optional): Sort order when using string/list format (default: True)
    
    Returns:
        dict: Transformation result with preview data and vdf_id for reuse
    """
    try:
        # Parse extra_kwargs - MCP sends it as a JSON string in extra_kwargs parameter
        import json
        kwargs = {}
        
        # Check if extra_kwargs contains a JSON string to parse
        if 'extra_kwargs' in extra_kwargs:
            extra_kwargs_value = extra_kwargs['extra_kwargs']
            if isinstance(extra_kwargs_value, str):
                try:
                    kwargs = json.loads(extra_kwargs_value)
                except json.JSONDecodeError:
                    pass
            elif isinstance(extra_kwargs_value, dict):
                kwargs = extra_kwargs_value
        
        # Also add any other direct parameters
        for key, value in extra_kwargs.items():
            if key != 'extra_kwargs':
                kwargs[key] = value
        
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
                rollup = kwargs.get("rollup", False)
                having = kwargs.get("having", None)
                
                if not columns:
                    return {
                        "success": False,
                        "error": "groupby operation requires 'columns' parameter"
                    }
                
                result_vdf = source_vdf.groupby(columns=columns, expr=expr, rollup=rollup, having=having)
                operation_info = f"groupby(columns={columns}, expr={expr}, rollup={rollup}, having={having})"
            
            elif operation == "join":
                input_relation = kwargs.get("right_table")
                on = kwargs.get("on", None)
                on_interpolate = kwargs.get("on_interpolate", None)
                how = kwargs.get("how", "natural")
                expr1 = kwargs.get("expr1", None)
                expr2 = kwargs.get("expr2", None)
                
                if not input_relation:
                    return {
                        "success": False,
                        "error": "join operation requires 'right_table' parameter"
                    }
                
                # Get right vDataFrame
                if input_relation in _vdf_cache:
                    right_vdf = _vdf_cache[input_relation]
                else:
                    right_vdf = vp.vDataFrame(input_relation)
                
                result_vdf = source_vdf.join(
                    input_relation=right_vdf, 
                    on=on, 
                    on_interpolate=on_interpolate,
                    how=how, 
                    expr1=expr1, 
                    expr2=expr2
                )
                operation_info = f"join(input_relation={input_relation}, on={on}, how={how}, expr1={expr1}, expr2={expr2})"
            
            elif operation == "pivot":
                index = kwargs.get("index")
                columns = kwargs.get("columns")
                values = kwargs.get("values")
                aggr = kwargs.get("aggr", "sum")
                prefix = kwargs.get("prefix", None)
                
                if not index or not columns or not values:
                    return {
                        "success": False,
                        "error": "pivot operation requires 'index', 'columns', and 'values' parameters"
                    }
                
                result_vdf = source_vdf.pivot(index=index, columns=columns, values=values, aggr=aggr, prefix=prefix)
                operation_info = f"pivot(index={index}, columns={columns}, values={values}, aggr={aggr}, prefix={prefix})"
            
            elif operation == "search":
                conditions = kwargs.get("conditions", "")
                usecols = kwargs.get("usecols", None)
                expr = kwargs.get("expr", None)
                order_by = kwargs.get("order_by", None)
                
                if not conditions:
                    return {
                        "success": False,
                        "error": f"search operation requires 'conditions' parameter. Received kwargs: {kwargs}"
                    }
                
                # Check if expr contains aggregate functions - suggest groupby instead
                if expr and isinstance(expr, list):
                    aggregate_funcs = ['sum(', 'count(', 'avg(', 'min(', 'max(', 'median(']
                    for e in expr:
                        if any(func in str(e).lower() for func in aggregate_funcs):
                            return {
                                "success": False,
                                "error": f"Cannot use aggregate functions in search operation. Use 'groupby' operation instead for expressions like: {expr}",
                                "suggestion": "Use operation='groupby' with appropriate 'columns' and 'expr' parameters"
                            }
                
                result_vdf = source_vdf.search(conditions=conditions, usecols=usecols, expr=expr, order_by=order_by)
                operation_info = f"search(conditions={conditions}, usecols={usecols}, expr={expr}, order_by={order_by})"
            
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
                columns = kwargs.get("columns")
                
                if not columns:
                    return {
                        "success": False,
                        "error": f"sort operation requires 'columns' parameter. Received kwargs: {kwargs}"
                    }
                
                # Handle both dict format {"column": "asc"} and list/string format
                if isinstance(columns, dict):
                    result_vdf = source_vdf.sort(columns)
                    operation_info = f"sort({columns})"
                else:
                    # For backward compatibility, convert to dict format if ascending is provided
                    ascending = kwargs.get("ascending", True)
                    if isinstance(columns, list):
                        if isinstance(ascending, list):
                            sort_dict = {col: "asc" if asc else "desc" for col, asc in zip(columns, ascending)}
                        else:
                            sort_dict = {col: "asc" if ascending else "desc" for col in columns}
                    else:
                        sort_dict = {columns: "asc" if ascending else "desc"}
                    
                    result_vdf = source_vdf.sort(sort_dict)
                    operation_info = f"sort({sort_dict})"
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation '{operation}'. Supported: groupby, join, pivot, search, select, sort"
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

# Import necessary ML models
from verticapy.machine_learning.vertica.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from verticapy.machine_learning.vertica.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
)
from verticapy.machine_learning.vertica.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from verticapy.machine_learning.vertica.cluster import (
    KMeans,
    DBSCAN,
)
from verticapy.machine_learning.vertica.model_management import load_model

# Available models mapping
AVAILABLE_MODELS = {
    # Classification models
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "xgb_classifier": XGBClassifier,
    "decision_tree_classifier": DecisionTreeClassifier,
    
    # Regression models  
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "random_forest_regressor": RandomForestRegressor,
    "xgb_regressor": XGBRegressor,
    "decision_tree_regressor": DecisionTreeRegressor,
    
    # Clustering models
    "kmeans": KMeans,
    "dbscan": DBSCAN,
}

@mcp.tool()
def train_model(
    table: str, 
    model_type: str, 
    model_name: str = None,
    target: str = None,
    features: list = None,
    test_table: str = None,
    model_params: str = None,
    **extra_kwargs
) -> dict:
    """
    Train a machine learning model using VerticaPy with comprehensive performance metrics.
    
    Args:
        table (str): Table name or vdf_id from cache to use for training
        model_type (str): Type of model to train. Available options:
            Classification: logistic_regression, random_forest_classifier, 
                          xgb_classifier, decision_tree_classifier
            Regression: linear_regression, ridge, lasso, elastic_net,
                       random_forest_regressor, xgb_regressor, decision_tree_regressor  
            Clustering: kmeans, dbscan
        model_name (str, optional): Name to save the model. If None, auto-generated.
        target (str, optional): Target column name (required for supervised models)
        features (list, optional): List of feature column names. If None, auto-detected.
        test_table (str, optional): Test table for evaluation
        model_params (str, optional): JSON string or dict containing model parameters.
            Examples: 
            - '{"max_iter": 100, "tol": 1e-6}' for LogisticRegression
            - '{"max_depth": 5, "n_estimators": 100}' for RandomForest
            - '{"n_cluster": 3, "init_method": "kmeanspp"}' for KMeans
        **extra_kwargs: Additional direct model parameters (alternative to model_params)
    
    Returns:
        dict: Comprehensive training results including:
            - model_info: Basic model information (name, type, features, target)
            - feature_importance: Feature importance scores and signs (for supported models)
            - performance_metrics: Model evaluation metrics (R², RMSE, MAE, AIC, BIC, etc.)
            - model_summary: Detailed coefficients with statistical significance
            - model_score: Overall model score
            - classification_report: Classification metrics (for classifiers)
            - confusion_matrix: Confusion matrix (for classifiers)
            - centroids: Cluster centers (for clustering models)
    """
    try:
        # Parse model_params if provided (can be JSON string or dict)
        parsed_model_params = {}
        
        if model_params:
            if isinstance(model_params, str):
                if model_params.strip():  # Only parse non-empty strings
                    try:
                        parsed_model_params = json.loads(model_params)
                    except json.JSONDecodeError as e:
                        return {
                            "success": False,
                            "error": f"Invalid JSON in model_params: {str(e)}"
                        }
            elif isinstance(model_params, dict):
                parsed_model_params = model_params
        
        # Filter out train_model function parameters from extra_kwargs
        # These should not be passed to the model constructor
        train_model_params = {
            'table', 'model_type', 'model_name', 'target', 'features', 
            'test_table', 'model_params', 'extra_kwargs'
        }
        
        # Filter extra_kwargs to only include actual model parameters
        filtered_extra_kwargs = {
            k: v for k, v in extra_kwargs.items() 
            if k not in train_model_params
        }
        
        # Merge with any additional kwargs (after filtering)
        final_model_params = {**parsed_model_params, **filtered_extra_kwargs}
        
        # Ensure connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {"success": False, "error": f"Connection failed: {message}"}
        
        # Validate model type
        if model_type not in AVAILABLE_MODELS:
            return {
                "success": False,
                "error": f"Unsupported model type '{model_type}'. Available types: {list(AVAILABLE_MODELS.keys())}"
            }
        
        # Determine if it's a supervised model (early validation)
        is_supervised = model_type not in ["kmeans", "dbscan"]
        is_clustering = model_type in ["kmeans", "dbscan"]
        
        # Validate target requirement for supervised models (before expensive operations)
        if is_supervised and not target:
            return {
                "success": False, 
                "error": "Target column is required for supervised learning models"
            }
        
        # Get model class
        ModelClass = AVAILABLE_MODELS[model_type]
        
        # Generate model name if not provided
        if not model_name:
            model_name = f"mcp_{model_type}_{gen_name([model_type, 'model'])}"
        
        # Get training data (from cache or table)
        if table in _vdf_cache:
            train_vdf = _vdf_cache[table]
            table_info = f"cached vDataFrame '{table}'"
        else:
            try:
                train_vdf = vp.vDataFrame(table)
                table_info = f"table '{table}'"
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to load training data from {table}: {str(e)}"
                }
        
        # Get available columns and create a mapping for both quoted and unquoted names
        available_columns = train_vdf.get_columns()
        # Create a mapping: both quoted and unquoted names -> actual column name
        column_mapping = {}
        for col in available_columns:
            stripped = col.strip('"')
            column_mapping[col] = col  # exact match
            column_mapping[stripped] = col  # unquoted -> quoted
            column_mapping[stripped.lower()] = col  # case-insensitive
            column_mapping[col.lower()] = col  # quoted case-insensitive
        
        # Validate and resolve target column name (for supervised models)
        actual_target = None
        if is_supervised:
            if target in column_mapping:
                actual_target = column_mapping[target]
            else:
                available_cols_clean = [col.strip('"') for col in available_columns]
                return {
                    "success": False,
                    "error": f"Target column '{target}' not found. Available: {available_cols_clean}"
                }
        
        # Auto-detect features if not provided
        if features is None:
            if is_supervised:
                # Remove target from features for supervised models
                features = [col for col in available_columns if col != actual_target]
            else:
                # Use all columns for clustering
                features = available_columns
        else:
            # Validate and resolve feature column names
            resolved_features = []
            missing_features = []
            
            for feature in features:
                if feature in column_mapping:
                    resolved_features.append(column_mapping[feature])
                else:
                    missing_features.append(feature)
            
            if missing_features:
                available_cols_clean = [col.strip('"') for col in available_columns]
                return {
                    "success": False,
                    "error": f"Features not found in data: {missing_features}. Available: {available_cols_clean}"
                }
            
            features = resolved_features
        
        # Use resolved names
        if is_supervised:
            target = actual_target
        
        # Create model instance
        try:
            model = ModelClass(name=model_name, **final_model_params)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create model instance: {str(e)}"
            }
        
        # Train the model
        try:
            if is_clustering:
                # Clustering models
                fit_result = model.fit(train_vdf, features)
            else:
                # Supervised models
                test_relation = test_table if test_table else ""
                fit_result = model.fit(train_vdf, features, target, test_relation, return_report=True)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to train model: {str(e)}"
            }
        
        # Get model information
        try:
            model_info = {
                "model_name": model_name,
                "model_type": model_type,
                "features": features,
                "n_features": len(features),
                "training_table": table_info,
                "model_stored": True
            }
            
            if is_supervised:
                model_info["target"] = target
                model_info["model_category"] = "supervised"
                
                # Get basic model attributes if available
                try:
                    if hasattr(model, 'shape'):
                        training_shape = model.shape
                        model_info["training_samples"] = training_shape[0] if training_shape else "unknown"
                except:
                    pass
                    
            elif is_clustering:
                model_info["model_category"] = "clustering"
                
                # Get clustering-specific info
                try:
                    if hasattr(model, 'n_cluster_'):
                        model_info["n_clusters"] = model.n_cluster_
                    elif hasattr(model, 'get_params'):
                        params = model.get_params()
                        if 'n_cluster' in params:
                            model_info["n_clusters"] = params['n_cluster']
                except:
                    pass
            
            # Get model parameters
            try:
                model_info["parameters"] = model.get_params()
            except:
                model_info["parameters"] = final_model_params
            
            response = {
                "success": True,
                "model_info": model_info,
                "training_summary": str(fit_result) if fit_result else "Training completed successfully"
            }
            
            # Add comprehensive performance metrics
            try:
                # Get feature importance (for supervised models that support it)
                if is_supervised and hasattr(model, 'features_importance'):
                    try:
                        feature_importance = model.features_importance(show=False)
                        response["feature_importance"] = _to_json_serializable(feature_importance.values)
                    except Exception as e:
                        response["feature_importance_note"] = f"Could not get feature importance: {str(e)}"
                
                # Get performance report (for supervised models)
                if is_supervised and hasattr(model, 'report'):
                    try:
                        performance_report = model.report()
                        response["performance_metrics"] = _to_json_serializable(performance_report.values)
                    except Exception as e:
                        response["performance_metrics_note"] = f"Could not get performance report: {str(e)}"
                
                # Get model summary (detailed coefficients, etc.)
                if hasattr(model, 'summarize'):
                    try:
                        model_summary = model.summarize()
                        response["model_summary"] = model_summary
                    except Exception as e:
                        response["model_summary_note"] = f"Could not get model summary: {str(e)}"
                
                # Get additional model-specific metrics
                if is_supervised:
                    # Classification-specific metrics
                    if 'classifier' in model_type or 'logistic' in model_type:
                        try:
                            if hasattr(model, 'classification_report'):
                                classification_report = model.classification_report()
                                response["classification_report"] = _to_json_serializable(classification_report.values)
                        except Exception as e:
                            response["classification_report_note"] = f"Could not get classification report: {str(e)}"
                        
                        try:
                            if hasattr(model, 'confusion_matrix'):
                                confusion_matrix = model.confusion_matrix()
                                response["confusion_matrix"] = _to_json_serializable(confusion_matrix.values)
                        except Exception as e:
                            response["confusion_matrix_note"] = f"Could not get confusion matrix: {str(e)}"
                    
                    # Try to get model score
                    try:
                        if hasattr(model, 'score'):
                            score = model.score()
                            response["model_score"] = _to_json_serializable(score)
                    except Exception as e:
                        response["model_score_note"] = f"Could not get model score: {str(e)}"
                
                # Clustering-specific metrics
                elif is_clustering:
                    try:
                        if hasattr(model, 'get_params'):
                            cluster_params = model.get_params()
                            response["cluster_parameters"] = cluster_params
                        
                        # Try to get cluster centers or other clustering metrics
                        if hasattr(model, 'cluster_centers_'):
                            response["cluster_centers"] = _to_json_serializable(model.cluster_centers_)
                        elif hasattr(model, 'centroids'):
                            centroids = model.centroids()
                            response["centroids"] = _to_json_serializable(centroids.values)
                    except Exception as e:
                        response["clustering_metrics_note"] = f"Could not get clustering metrics: {str(e)}"
                        
            except Exception as e:
                response["metrics_collection_error"] = f"Error collecting performance metrics: {str(e)}"
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Model trained but failed to retrieve information: {str(e)}",
                "model_name": model_name
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error in train_model: {str(e)}"
        }


@mcp.tool()
def predict(
    table: str, 
    model_name: str, 
    output_name: str = None,
    features: list = None,
    prediction_type: str = "prediction"
) -> dict:
    """
    Make predictions using a trained model.
    
    Args:
        table (str): Table name or vdf_id from cache to make predictions on
        model_name (str): Name of the trained model to use
        output_name (str, optional): Name for the prediction column. If None, auto-generated.
        features (list, optional): List of feature columns to use. If None, uses model's features.
        prediction_type (str): Type of prediction for classification models:
            - "prediction": Class predictions (default)
            - "probability": Prediction probabilities
    
    Returns:
        dict: Prediction results with sample data and metadata
    """
    try:
        # Ensure connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {"success": False, "error": f"Connection failed: {message}"}
        
        # Check if model exists
        try:
            model = load_model(model_name)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load model '{model_name}': {str(e)}"
            }
        
        # Get prediction data (from cache or table)
        if table in _vdf_cache:
            pred_vdf = _vdf_cache[table]
            table_info = f"cached vDataFrame '{table}'"
        else:
            try:
                pred_vdf = vp.vDataFrame(table)
                table_info = f"table '{table}'"
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to load prediction data from {table}: {str(e)}"
                }
        
        # Use model's features if not specified
        if features is None:
            if hasattr(model, 'X'):
                features = model.X
            else:
                return {
                    "success": False,
                    "error": "Could not determine features from model and no features provided"
                }
        
        # Validate features exist in data
        available_columns = pred_vdf.get_columns()
        missing_features = [f for f in features if f not in available_columns]
        if missing_features:
            return {
                "success": False,
                "error": f"Features not found in prediction data: {missing_features}. Available: {available_columns}"
            }
        
        # Generate output name if not provided
        if not output_name:
            output_name = f"prediction_{gen_name()}"
        
        try:
            # Make predictions based on model type and prediction type
            is_classifier = 'classifier' in model._model_type.lower() if hasattr(model, '_model_type') else False
            
            if is_classifier and prediction_type == "probability":
                # For classification probability predictions
                if hasattr(model, 'predict_proba'):
                    result_vdf = model.predict_proba(pred_vdf, features, output_name)
                else:
                    return {
                        "success": False,
                        "error": f"Model '{model_name}' does not support probability predictions"
                    }
            else:
                # Regular predictions (classification or regression)
                result_vdf = model.predict(pred_vdf, features, output_name)
            
            # Get prediction results info
            result_shape = result_vdf.shape()
            result_columns = result_vdf.get_columns()
            
            # Get sample of predictions
            sample_size = min(10, result_shape[0])
            if sample_size > 0:
                try:
                    sample_data = result_vdf.head(sample_size).to_pandas().to_dict('records')
                    sample_data = _to_json_serializable(sample_data)
                except Exception:
                    sample_data = []
            else:
                sample_data = []
            
            return {
                "success": True,
                "model_name": model_name,
                "prediction_table": table_info,
                "prediction_column": output_name,
                "prediction_type": prediction_type,
                "total_predictions": result_shape[0],
                "result_columns": [col.strip('"') for col in result_columns],
                "sample_predictions": sample_data,
                "model_info": {
                    "model_type": getattr(model, '_model_type', 'unknown'),
                    "features_used": [f.strip('"') for f in features]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to make predictions: {str(e)}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error in predict: {str(e)}"
        }


@mcp.tool()
def list_models(model_type_filter: str = None, limit: int = 50) -> dict:
    """
    List all trained models stored in the Vertica database.
    
    Args:
        model_type_filter (str, optional): Filter by model type (e.g., 'LINEAR_REGRESSION', 'RF_CLASSIFIER')
        limit (int): Maximum number of models to return (default: 50)
    
    Returns:
        dict: List of models with their details
    """
    try:
        # Ensure connection
        success, message = connection_manager.ensure_connected()
        if not success:
            return {"success": False, "error": f"Connection failed: {message}"}
        
        # Build query to get models from the MODELS system table
        base_query = """
        SELECT 
            schema_name,
            model_name,
            model_type,
            category,
            owner_name,
            create_time
        FROM MODELS
        """
        
        conditions = []
        if model_type_filter:
            conditions.append(f"UPPER(model_type) LIKE UPPER('%{model_type_filter}%')")
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        base_query += f" ORDER BY create_time DESC LIMIT {limit}"
        
        try:
            result = _executeSQL(
                query=base_query,
                method="fetchall"
            )
            
            models = []
            for row in result:
                model_info = {
                    "schema_name": row[0],
                    "model_name": row[1], 
                    "full_name": f"{row[0]}.{row[1]}",
                    "model_type": row[2],
                    "category": row[3],
                    "owner": row[4],
                    "created_at": str(row[5]) if row[5] else None
                }
                
                # Try to get additional model details
                try:
                    full_model_name = f"{row[0]}.{row[1]}"
                    # Get model summary for additional info
                    summary_query = f"""
                    SELECT GET_MODEL_SUMMARY(USING PARAMETERS model_name = '{full_model_name}')
                    """
                    summary_result = _executeSQL(
                        query=summary_query,
                        method="fetchfirstelem"
                    )
                    if summary_result:
                        # Extract useful info from summary (this is model-specific)
                        summary_text = str(summary_result)
                        model_info["summary_available"] = True
                        if "predictor" in summary_text.lower():
                            model_info["has_predictors"] = True
                except Exception:
                    # If we can't get summary, that's okay
                    model_info["summary_available"] = False
                
                models.append(model_info)
            
            return {
                "success": True,
                "models": models,
                "count": len(models),
                "filter_applied": model_type_filter,
                "available_model_types": list(AVAILABLE_MODELS.keys())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to query models: {str(e)}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error in list_models: {str(e)}"
        }


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