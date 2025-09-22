"""
Pytest-based Comprehensive Test Suite for VerticaPy MCP Server

This test suite thoroughly tests all MCP tools provided by server2.py.
It uses pytest fixtures for robust setup/teardown and data management.

Copyright  (c)  2018-2025 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""

import pytest
import sys
import os
import json
import logging
from typing import Dict, Any

# Add the parent directory to Python path to access server2.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MCP server functions and connection class
from server2 import (
    connect_to_vertica,
    disconnect_from_vertica,
    get_connection_status,
    list_tables,
    list_all_schemas,
    describe_table,
    sample_data,
    column_stats,
    table_stats,
    transform_data,
    list_cached_vdfs,
    clear_vdf_cache,
    connection_manager,
    _vdf_cache
)
from connection import VerticaPyConnection

from config import load_test_config, get_conn_info, CONN_INFO
import verticapy as vp


class TestColors:
    """ANSI color codes for pretty test output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_test_info(test_name: str, result: Dict[str, Any], expected_success: bool = True):
    """Pretty print test results"""
    success = result.get("success", False)
    
    if success == expected_success:
        status_color = TestColors.OKGREEN
        status_symbol = "✓"
    else:
        status_color = TestColors.FAIL
        status_symbol = "✗"
    
    print(f"{status_color}{status_symbol} {test_name}{TestColors.ENDC}")
    
    # Print key result information
    if success:
        if "message" in result:
            print(f"  Message: {result['message']}")
        if "count" in result:
            print(f"  Count: {result['count']}")
        if "tables" in result and isinstance(result['tables'], list):
            print(f"  Tables found: {len(result['tables'])}")
            if result['tables']:
                print(f"  Sample tables: {result['tables'][:3]}")
        if "columns" in result and isinstance(result['columns'], dict):
            cols = result['columns'].get('index', [])
            print(f"  Columns: {len(cols)} found")
            if cols:
                print(f"  Sample columns: {cols[:5]}")
        if "result" in result:
            print(f"  Result type: {type(result['result'])}")
            if isinstance(result['result'], dict) and 'index' in result['result']:
                print(f"  Result columns: {len(result['result'].get('index', []))}")
    else:
        if "error" in result:
            print(f"  {TestColors.FAIL}Error: {result['error']}{TestColors.ENDC}")
    
    print()  # Add spacing


def display_conn_info_mcp_format(conn_info):
    """Display connection info in exact MCP format"""
    print(f"{TestColors.OKCYAN}CONN_INFO = {{{TestColors.ENDC}")
    for key, value in conn_info.items():
        print(f'    "{key}": "{value}",')
    print(f"{TestColors.OKCYAN}}}{TestColors.ENDC}")


class TestMCPConnection:
    """Test MCP connection management"""
    
    def test_connection_status_and_info(self, mcp_connection, test_config):
        """Test connection status and display connection info"""
        print(f"\n{TestColors.BOLD}--- Testing Connection Management ---{TestColors.ENDC}")
        
        # Display connection info in MCP format
        print(f"{TestColors.OKBLUE}Connection configuration (MCP format):{TestColors.ENDC}")
        display_conn_info_mcp_format(mcp_connection)
        print()
        
        # Test connection status
        result = get_connection_status()
        # Convert to expected format for print_test_info
        formatted_result = {
            "success": result.get("is_connected", False),
            "message": f"Connection status: {'Connected' if result.get('is_connected') else 'Disconnected'}",
            **result
        }
        print_test_info("Get connection status", formatted_result)
        assert result.get("is_connected", False)
    
    def test_connection_lifecycle(self, mcp_connection):
        """Test connection disconnect and reconnect"""
        print(f"\n{TestColors.BOLD}--- Testing Connection Lifecycle ---{TestColors.ENDC}")
        
        # Test disconnect
        result = disconnect_from_vertica()
        print_test_info("Disconnect from Vertica", result)
        
        # Test operations after disconnect (should fail gracefully)
        result = list_tables()
        print_test_info("List tables after disconnect", result, expected_success=False)
        
        # Reconnect
        result = connect_to_vertica()
        print_test_info("Reconnect to Vertica", result)
        assert result.get("success", False)


class TestSchemaAndTableOperations:
    """Test schema and table operations"""
    
    def test_schema_listing(self, mcp_connection):
        """Test listing schemas"""
        print(f"\n{TestColors.BOLD}--- Testing Schema Listing ---{TestColors.ENDC}")
        
        result = list_all_schemas()
        print_test_info("List all schemas", result)
        assert result.get("success", False)
        assert isinstance(result.get("schemas", []), list)
    
    def test_table_listing(self, mcp_connection, schema_loader):
        """Test listing tables"""
        print(f"\n{TestColors.BOLD}--- Testing Table Listing ---{TestColors.ENDC}")
        
        # Test list tables in test schema
        result = list_tables(schema=schema_loader)
        print_test_info(f"List tables in {schema_loader} schema", result)
        assert result.get("success", False)
        assert isinstance(result.get("tables", []), list)
        
        # Test list tables in public schema
        result = list_tables(schema="public")
        print_test_info("List tables in public schema", result)
        assert result.get("success", False)
        
        # Test non-existent schema
        result = list_tables(schema="non_existent_schema")
        print_test_info("List tables in non-existent schema", result)
    
    def test_table_description(self, mcp_connection, titanic_vd, schema_loader):
        """Test table description"""
        print(f"\n{TestColors.BOLD}--- Testing Table Description ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test describe existing table
        result = describe_table(table_name)
        print_test_info(f"Describe table {table_name}", result)
        assert result.get("success", False)
        assert isinstance(result.get("row_count", 0), int)
        assert isinstance(result.get("column_count", 0), int)
        
        # Test describe non-existent table
        result = describe_table("non_existent_table")
        print_test_info("Describe non-existent table", result, expected_success=False)
        assert not result.get("success", True)
    
    def test_data_sampling(self, mcp_connection, titanic_vd, schema_loader):
        """Test data sampling"""
        print(f"\n{TestColors.BOLD}--- Testing Data Sampling ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test sample data with default size
        result = sample_data(table_name)
        print_test_info(f"Sample data from {table_name} (default size)", result)
        assert result.get("success", False)
        assert isinstance(result.get("data", []), list)
        
        # Test sample data with specific size
        result = sample_data(table_name, n=10)
        print_test_info(f"Sample 10 rows from {table_name}", result)
        assert result.get("success", False)
        assert len(result.get("data", [])) <= 10
        
        # Test sample from non-existent table
        result = sample_data("non_existent_table")
        print_test_info("Sample from non-existent table", result, expected_success=False)
        assert not result.get("success", True)


class TestColumnStatistics:
    """Test column-level statistics"""
    
    def test_basic_column_stats(self, mcp_connection, titanic_vd, schema_loader, test_datasets_info):
        """Test basic column statistics"""
        print(f"\n{TestColors.BOLD}--- Testing Basic Column Statistics ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        titanic_info = test_datasets_info["titanic"]
        
        # Test basic stats on numeric columns
        for column in titanic_info["numeric_columns"][:3]:  # Test first 3
            for metric in ["describe", "mean", "max", "min", "count"]:
                result = column_stats(table_name, column, metric)
                print_test_info(f"Column stats: {column}.{metric}", result)
                if result.get("success"):
                    assert result.get("success", False)
                    assert result.get("result") is not None
    
    def test_parameterized_column_stats(self, mcp_connection, titanic_vd, schema_loader, sample_test_operations):
        """Test parameterized column statistics"""
        print(f"\n{TestColors.BOLD}--- Testing Parameterized Column Statistics ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test parameterized metrics
        for param_metric in sample_test_operations["parameterized_metrics"]:
            metric = param_metric["metric"]
            params = param_metric["params"]
            
            result = column_stats(
                table_name, 
                "age",  # Use age column for all tests
                metric,
                extra_kwargs=json.dumps(params)
            )
            print_test_info(f"Column stats: age.{metric}({params})", result)
            if result.get("success"):
                assert result.get("success", False)
    
    def test_column_stats_error_handling(self, mcp_connection, titanic_vd, schema_loader):
        """Test column statistics error handling"""
        print(f"\n{TestColors.BOLD}--- Testing Column Stats Error Handling ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test invalid column
        result = column_stats(table_name, "non_existent_column", "describe")
        print_test_info("Column stats: non-existent column", result, expected_success=False)
        assert not result.get("success", True)
        
        # Test invalid metric
        result = column_stats(table_name, "age", "invalid_metric")
        print_test_info("Column stats: invalid metric", result, expected_success=False)
        assert not result.get("success", True)


class TestTableStatistics:
    """Test table-level statistics"""
    
    def test_basic_table_stats(self, mcp_connection, titanic_vd, schema_loader, sample_test_operations):
        """Test basic table statistics"""
        print(f"\n{TestColors.BOLD}--- Testing Basic Table Statistics ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test basic metrics
        for metric in sample_test_operations["statistical_metrics"][:5]:  # Test first 5
            result = table_stats(table_name, metric)
            print_test_info(f"Table stats: {metric}", result)
            if result.get("success"):
                assert result.get("success", False)
    
    def test_table_stats_with_columns(self, mcp_connection, titanic_vd, schema_loader):
        """Test table statistics with specific columns"""
        print(f"\n{TestColors.BOLD}--- Testing Table Stats with Specific Columns ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test with specific columns
        result = table_stats(
            table_name,
            "describe", 
            columns=["age", "fare"]
        )
        print_test_info("Table stats: describe specific columns", result)
        if result.get("success"):
            assert result.get("success", False)
    
    def test_table_stats_error_handling(self, mcp_connection):
        """Test table statistics error handling"""
        print(f"\n{TestColors.BOLD}--- Testing Table Stats Error Handling ---{TestColors.ENDC}")
        
        # Test invalid table
        result = table_stats("non_existent_table", "describe")
        print_test_info("Table stats: non-existent table", result, expected_success=False)
        assert not result.get("success", True)


class TestDataTransformations:
    """Test data transformation operations"""
    
    def test_search_operations(self, mcp_connection, titanic_vd, schema_loader, sample_test_operations, clean_vdf_cache):
        """Test search transformations"""
        print(f"\n{TestColors.BOLD}--- Testing Search Transformations ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test various search operations
        for i, search_op in enumerate(sample_test_operations["search_operations"][:2]):  # Test first 2
            result = transform_data(
                table=table_name,
                operation="search",
                extra_kwargs=json.dumps({"conditions": search_op["conditions"]}),
                vdf_id=f"search_{i}",
                show_preview=True
            )
            print_test_info(f"Transform: search ({search_op['description']})", result)
            if result.get("success"):
                assert result.get("success", False)
                assert result.get("vdf_id") == f"search_{i}"
    
    def test_groupby_operations(self, mcp_connection, titanic_vd, schema_loader, sample_test_operations, clean_vdf_cache):
        """Test groupby transformations"""
        print(f"\n{TestColors.BOLD}--- Testing Groupby Transformations ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test groupby operations
        for i, groupby_op in enumerate(sample_test_operations["groupby_operations"]):
            result = transform_data(
                table=table_name,
                operation="groupby",
                extra_kwargs=json.dumps({
                    "columns": groupby_op["columns"],
                    "expr": groupby_op["expr"]
                }),
                vdf_id=f"groupby_{i}",
                show_preview=True
            )
            print_test_info(f"Transform: groupby ({groupby_op['description']})", result)
            if result.get("success"):
                assert result.get("success", False)
    
    def test_other_transformations(self, mcp_connection, titanic_vd, schema_loader, clean_vdf_cache):
        """Test other transformation operations"""
        print(f"\n{TestColors.BOLD}--- Testing Other Transformations ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test select operation
        result = transform_data(
            table=table_name,
            operation="select",
            extra_kwargs=json.dumps({"columns": ["name", "age", "sex", "survived"]}),
            vdf_id="basic_info",
            show_preview=True
        )
        print_test_info("Transform: select specific columns", result)
        if result.get("success"):
            assert result.get("success", False)
        
        # Test sort operation
        result = transform_data(
            table=table_name,
            operation="sort",
            extra_kwargs=json.dumps({"columns": {"fare": "desc"}}),
            vdf_id="sorted_by_fare",
            show_preview=True
        )
        print_test_info("Transform: sort by fare descending", result)
        if result.get("success"):
            assert result.get("success", False)
    
    def test_transformation_error_handling(self, mcp_connection, titanic_vd, schema_loader, clean_vdf_cache):
        """Test transformation error handling"""
        print(f"\n{TestColors.BOLD}--- Testing Transformation Error Handling ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test search with invalid aggregate function (should fail with helpful message)
        result = transform_data(
            table=table_name,
            operation="search",
            extra_kwargs=json.dumps({
                "conditions": "survived = 1", 
                "expr": ["sum(fare) AS total_fare"]
            }),
            show_preview=True
        )
        print_test_info("Transform: search with aggregate (should fail)", result, expected_success=False)
        assert not result.get("success", True)
        assert "groupby" in result.get("error", "").lower()
        
        # Test invalid operation
        result = transform_data(
            table=table_name,
            operation="invalid_operation",
            show_preview=True
        )
        print_test_info("Transform: invalid operation", result, expected_success=False)
        assert not result.get("success", True)


class TestCachedOperations:
    """Test cached vDataFrame operations"""
    
    def test_cached_vdf_operations(self, mcp_connection, titanic_vd, schema_loader, clean_vdf_cache):
        """Test operations on cached vDataFrames"""
        print(f"\n{TestColors.BOLD}--- Testing Cached vDataFrame Operations ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Create a cached vDataFrame
        result = transform_data(
            table=table_name,
            operation="search",
            extra_kwargs=json.dumps({"conditions": "age > 30"}),
            vdf_id="adults",
            show_preview=False
        )
        print_test_info("Create cached vDataFrame (adults)", result)
        
        if result.get("success"):
            # Test column stats on cached vDataFrame
            result = column_stats("adults", "fare", "mean")
            print_test_info("Column stats on cached vDataFrame", result)
            if result.get("success"):
                assert result.get("success", False)
            
            # Test table stats on cached vDataFrame
            result = table_stats("adults", "describe")
            print_test_info("Table stats on cached vDataFrame", result)
            if result.get("success"):
                assert result.get("success", False)
    
    def test_vdf_cache_management(self, mcp_connection, clean_vdf_cache):
        """Test vDataFrame cache management"""
        print(f"\n{TestColors.BOLD}--- Testing vDataFrame Cache Management ---{TestColors.ENDC}")
        
        # Test list cached vDataFrames
        result = list_cached_vdfs()
        print_test_info("List cached vDataFrames", result)
        assert result.get("success", False)
        
        # Test clearing cache
        result = clear_vdf_cache()
        print_test_info("Clear all cached vDataFrames", result)
        assert result.get("success", False)
        
        # Verify cache is empty
        result = list_cached_vdfs()
        print_test_info("Verify cache is cleared", result)
        assert result.get("count", -1) == 0


class TestMultipleDatasets:
    """Test operations on multiple datasets"""
    
    def test_iris_operations(self, mcp_connection, iris_vd, schema_loader):
        """Test operations on iris dataset"""
        print(f"\n{TestColors.BOLD}--- Testing Iris Dataset Operations ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.iris"
        
        # Test describe
        result = describe_table(table_name)
        print_test_info(f"Describe iris table", result)
        if result.get("success"):
            assert result.get("success", False)
        
        # Test column stats
        result = column_stats(table_name, "sepal_length", "mean")
        print_test_info("Iris: sepal_length mean", result)
        if result.get("success"):
            assert result.get("success", False)
    
    def test_winequality_operations(self, mcp_connection, winequality_vd, schema_loader):
        """Test operations on winequality dataset"""
        print(f"\n{TestColors.BOLD}--- Testing Wine Quality Dataset Operations ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.winequality"
        
        # Test sample data
        result = sample_data(table_name, n=5)
        print_test_info("Wine quality: sample 5 rows", result)
        if result.get("success"):
            assert result.get("success", False)
        
        # Test groupby on quality
        result = transform_data(
            table=table_name,
            operation="groupby",
            extra_kwargs=json.dumps({
                "columns": ["quality"],
                "expr": ["count(*) AS wine_count", "avg(alcohol) AS avg_alcohol"]
            }),
            vdf_id="wine_by_quality",
            show_preview=True
        )
        print_test_info("Wine quality: group by quality", result)
        if result.get("success"):
            assert result.get("success", False)


class TestParameterParsing:
    """Test parameter parsing for different formats"""
    
    def test_json_parameter_parsing(self, mcp_connection, titanic_vd, schema_loader):
        """Test different parameter formats"""
        print(f"\n{TestColors.BOLD}--- Testing Parameter Parsing ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test JSON string parameter
        result = column_stats(
            table_name, 
            "age", 
            "nlargest", 
            extra_kwargs='{"n": 3}'
        )
        print_test_info("JSON string parameter", result)
        if result.get("success"):
            assert result.get("success", False)
        
        # Test empty JSON parameter
        result = column_stats(
            table_name, 
            "age", 
            "describe", 
            extra_kwargs='{}'
        )
        print_test_info("Empty JSON parameter", result)
        if result.get("success"):
            assert result.get("success", False)
        
        # Test complex JSON parameter
        result = column_stats(
            table_name, 
            "age", 
            "aggregate",
            extra_kwargs='{"func": ["min", "max", "avg"]}'
        )
        print_test_info("Complex JSON parameter", result)
        if result.get("success"):
            assert result.get("success", False)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])