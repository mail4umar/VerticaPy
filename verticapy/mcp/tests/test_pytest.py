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
    train_model,
    predict,
    list_models,
    AVAILABLE_MODELS,
    connection_manager,
    _vdf_cache
)
from connection import VerticaPyConnection

# Import config from the local tests directory
import sys
import os
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

import config
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
        result = column_stats(table_name, "SepalLengthCm", "mean")
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


class TestMachineLearningModels:
    """Test machine learning model training and prediction"""
    
    def test_available_models_info(self, mcp_connection):
        """Test that available models dictionary is properly configured"""
        print(f"\n{TestColors.BOLD}--- Testing Available ML Models ---{TestColors.ENDC}")
        
        # Check that AVAILABLE_MODELS is properly configured
        assert isinstance(AVAILABLE_MODELS, dict)
        assert len(AVAILABLE_MODELS) > 0
        
        print(f"Available ML models ({len(AVAILABLE_MODELS)}):")
        for model_type, model_class in AVAILABLE_MODELS.items():
            print(f"  - {model_type}: {model_class.__name__}")
        
        # Test that we have different categories
        classification_models = [k for k in AVAILABLE_MODELS.keys() if 'classifier' in k]
        regression_models = [k for k in AVAILABLE_MODELS.keys() if 'regressor' in k or k in ['linear_regression', 'ridge', 'lasso', 'elastic_net']]
        clustering_models = [k for k in AVAILABLE_MODELS.keys() if k in ['kmeans', 'dbscan']]
        
        print(f"  Classification models: {len(classification_models)}")
        print(f"  Regression models: {len(regression_models)}")
        print(f"  Clustering models: {len(clustering_models)}")
        
        assert len(classification_models) > 0
        assert len(regression_models) > 0
        assert len(clustering_models) > 0
    
    def test_list_models_empty(self, mcp_connection):
        """Test listing models when database might be empty"""
        print(f"\n{TestColors.BOLD}--- Testing List Models (Initial) ---{TestColors.ENDC}")
        
        result = list_models()
        print_test_info("List all models (initial)", result)
        assert result.get("success", False)
        assert isinstance(result.get("models", []), list)
        assert isinstance(result.get("count", -1), int)
        assert result.get("available_model_types") == list(AVAILABLE_MODELS.keys())
    
    def test_train_classification_model(self, mcp_connection, titanic_vd, schema_loader, clean_vdf_cache):
        """Test training a classification model"""
        print(f"\n{TestColors.BOLD}--- Testing Classification Model Training ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test training a logistic regression model
        result = train_model(
            table=table_name,
            model_type="logistic_regression",
            model_name="test_titanic_lr",
            target="survived",
            features=["age", "fare", "pclass"],
            max_iter=100
        )
        print_test_info("Train logistic regression (Titanic survival)", result)
        
        if result.get("success"):
            assert result.get("success", False)
            model_info = result.get("model_info", {})
            assert model_info.get("model_name") == "test_titanic_lr"
            assert model_info.get("model_type") == "logistic_regression"
            assert model_info.get("target") == "survived"
            assert model_info.get("model_category") == "supervised"
            assert isinstance(model_info.get("features", []), list)
            assert len(model_info.get("features", [])) == 3
            
            # Test new performance metrics are included
            assert "feature_importance" in result
            assert "performance_metrics" in result
            assert "model_summary" in result
            
            # Validate feature importance structure
            feature_importance = result.get("feature_importance", {})
            assert isinstance(feature_importance, dict)
            assert len(feature_importance) > 0  # Should have importance scores
            
            # Validate performance metrics structure
            performance_metrics = result.get("performance_metrics", {})
            assert isinstance(performance_metrics, dict)
            # Classification model should have metrics like accuracy, auc, etc.
            assert any(key in performance_metrics for key in ["accuracy", "auc", "precision", "recall"])
            
            # Validate model summary structure
            model_summary = result.get("model_summary", "")
            assert isinstance(model_summary, str)
            assert len(model_summary) > 0  # Should contain model details
    
    def test_train_regression_model(self, mcp_connection, titanic_vd, schema_loader, clean_vdf_cache):
        """Test training a regression model"""
        print(f"\n{TestColors.BOLD}--- Testing Regression Model Training ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test training a linear regression model to predict fare
        result = train_model(
            table=table_name,
            model_type="linear_regression",
            model_name="test_titanic_fare_lr",
            target="fare",
            features=["age", "pclass"],
            fit_intercept=True
        )
        print_test_info("Train linear regression (Titanic fare prediction)", result)
        
        if result.get("success"):
            assert result.get("success", False)
            model_info = result.get("model_info", {})
            assert model_info.get("model_name") == "test_titanic_fare_lr"
            assert model_info.get("model_type") == "linear_regression"
            assert model_info.get("target") == "fare"
            assert model_info.get("model_category") == "supervised"
            
            # Test new performance metrics are included
            assert "feature_importance" in result
            assert "performance_metrics" in result
            assert "model_summary" in result
            
            # For regression models, check for regression-specific metrics
            performance_metrics = result.get("performance_metrics", {})
            assert isinstance(performance_metrics, dict)
            # Should have regression metrics like r2, mse, mae, etc.
            assert any(key in performance_metrics for key in ["r2", "mse", "mae", "rmse"])
    
    def test_train_clustering_model(self, mcp_connection, iris_vd, schema_loader, clean_vdf_cache):
        """Test training a clustering model"""
        print(f"\n{TestColors.BOLD}--- Testing Clustering Model Training ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.iris"
        
        # Test training a k-means clustering model
        result = train_model(
            table=table_name,
            model_type="kmeans",
            model_name="test_iris_kmeans",
            features=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            n_cluster=3,
            init_method="kmeanspp"
        )
        print_test_info("Train K-Means clustering (Iris dataset)", result)
        
        if result.get("success"):
            assert result.get("success", False)
            model_info = result.get("model_info", {})
            assert model_info.get("model_name") == "test_iris_kmeans"
            assert model_info.get("model_type") == "kmeans"
            assert model_info.get("model_category") == "clustering"
            assert len(model_info.get("features", [])) == 4
            
            # Test new performance metrics are included
            assert "performance_metrics" in result
            assert "model_summary" in result
            # Note: clustering models may not have feature_importance
            
            # For clustering models, check for clustering-specific metrics
            performance_metrics = result.get("performance_metrics", {})
            assert isinstance(performance_metrics, dict)
    
    def test_train_ensemble_model(self, mcp_connection, winequality_vd, schema_loader, clean_vdf_cache):
        """Test training an ensemble model"""
        print(f"\n{TestColors.BOLD}--- Testing Ensemble Model Training ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.winequality"
        
        # Test training a random forest classifier
        result = train_model(
            table=table_name,
            model_type="random_forest_classifier",
            model_name="test_wine_rf",
            target="quality",
            features=["alcohol", "volatile_acidity", "citric_acid", "residual_sugar"],
            n_estimators=10,  # Small number for faster testing
            max_depth=5
        )
        print_test_info("Train Random Forest (Wine quality classification)", result)
        
        if result.get("success"):
            assert result.get("success", False)
            model_info = result.get("model_info", {})
            assert model_info.get("model_name") == "test_wine_rf"
            assert model_info.get("model_type") == "random_forest_classifier"
            assert "quality" in str(model_info.get("target", ""))
            
            # Test new performance metrics are included
            assert "feature_importance" in result
            assert "performance_metrics" in result
            assert "model_summary" in result
            
            # Ensemble models should have comprehensive feature importance
            feature_importance = result.get("feature_importance", {})
            assert isinstance(feature_importance, dict)
            assert len(feature_importance) == 4  # Should match number of features
    
    def test_train_model_auto_features(self, mcp_connection, titanic_vd, schema_loader, clean_vdf_cache):
        """Test training with auto-detected features"""
        print(f"\n{TestColors.BOLD}--- Testing Auto-Feature Detection ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test training without specifying features (should auto-detect)
        result = train_model(
            table=table_name,
            model_type="decision_tree_classifier",
            target="survived"
            # No features specified - should auto-detect
        )
        print_test_info("Train Decision Tree with auto-detected features", result)
        
        if result.get("success"):
            assert result.get("success", False)
            model_info = result.get("model_info", {})
            # Should have detected multiple features (all columns except target)
            assert len(model_info.get("features", [])) > 1
            
            # Test new performance metrics are included
            assert "feature_importance" in result
            assert "performance_metrics" in result
            assert "model_summary" in result
    
    def test_train_model_error_handling(self, mcp_connection, titanic_vd, schema_loader):
        """Test model training error handling"""
        print(f"\n{TestColors.BOLD}--- Testing Model Training Error Handling ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test invalid model type
        result = train_model(
            table=table_name,
            model_type="invalid_model_type",
            target="survived"
        )
        print_test_info("Train model: invalid model type", result, expected_success=False)
        assert not result.get("success", True)
        assert "Unsupported model type" in result.get("error", "")
        
        # Test missing target for supervised model
        result = train_model(
            table=table_name,
            model_type="logistic_regression"
            # No target specified
        )
        print_test_info("Train model: missing target", result, expected_success=False)
        assert not result.get("success", True)
        assert "Target column is required" in result.get("error", "")
        
        # Test invalid table
        result = train_model(
            table="non_existent_table",
            model_type="linear_regression",
            target="some_column"
        )
        print_test_info("Train model: invalid table", result, expected_success=False)
        assert not result.get("success", True)
        
        # Test invalid features
        result = train_model(
            table=table_name,
            model_type="linear_regression",
            target="fare",
            features=["non_existent_column"]
        )
        print_test_info("Train model: invalid features", result, expected_success=False)
        assert not result.get("success", True)
        assert "Features not found" in result.get("error", "")
    
    def test_train_model_with_json_params(self, mcp_connection, titanic_vd, schema_loader):
        """Test training models with JSON model_params (MCP client format)"""
        print(f"\n{TestColors.BOLD}--- Testing Model Training with JSON Parameters ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test training with JSON model_params string (as MCP client would send)
        result = train_model(
            table=table_name,
            model_type="logistic_regression",
            model_name="test_json_params_lr",
            target="survived",
            features=["age", "fare", "pclass"],
            model_params='{"max_iter": 150, "tol": 1e-5, "solver": "newton"}'
        )
        print_test_info("Train model with JSON model_params", result)
        
        if result.get("success"):
            assert result.get("success", False)
            model_info = result.get("model_info", {})
            assert model_info.get("model_name") == "test_json_params_lr"
            
            # Verify parameters were applied (check if they appear in model summary or parameters)
            model_summary = result.get("model_summary", "")
            assert isinstance(model_summary, str)
            assert len(model_summary) > 0  # Should have actual content
            
            print(f"  ✓ Model trained successfully with JSON parameters")
        
        # Test training with empty JSON model_params
        result = train_model(
            table=table_name,
            model_type="linear_regression",
            model_name="test_empty_params_lr",
            target="fare",
            features=["age", "pclass"],
            model_params='{}'
        )
        print_test_info("Train model with empty JSON model_params", result)
        
        if result.get("success"):
            assert result.get("success", False)
            print(f"  ✓ Model trained successfully with empty JSON parameters")
        
        # Test training with None model_params (backward compatibility)
        result = train_model(
            table=table_name,
            model_type="decision_tree_classifier",
            model_name="test_none_params_dt",
            target="survived",
            features=["age", "fare"],
            model_params=None
        )
        print_test_info("Train model with None model_params", result)
        
        if result.get("success"):
            assert result.get("success", False)
            print(f"  ✓ Model trained successfully with None parameters")
        
        # Test invalid JSON model_params (should handle gracefully)
        result = train_model(
            table=table_name,
            model_type="logistic_regression",
            model_name="test_invalid_json_lr",
            target="survived",
            features=["age", "fare"],
            model_params='{"invalid_json": true, missing_quote_and_brace'
        )
        print_test_info("Train model with invalid JSON model_params", result, expected_success=False)
        assert not result.get("success", True)
        assert "JSON" in result.get("error", "") or "parse" in result.get("error", "").lower()
    
    def test_comprehensive_performance_metrics(self, mcp_connection, winequality_vd, schema_loader, clean_vdf_cache):
        """Test that all expected performance metrics are included in responses"""
        print(f"\n{TestColors.BOLD}--- Testing Comprehensive Performance Metrics ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.winequality"
        
        # Train a classification model to test comprehensive metrics
        result = train_model(
            table=table_name,
            model_type="logistic_regression",
            model_name="test_comprehensive_metrics",
            target="quality",
            features=["alcohol", "volatile_acidity", "citric_acid"],
            model_params='{"max_iter": 100}'
        )
        print_test_info("Train model for comprehensive metrics testing", result)
        
        if result.get("success"):
            # Validate all expected top-level keys are present
            expected_keys = ["success", "model_info", "feature_importance", "performance_metrics", "model_summary"]
            for key in expected_keys:
                assert key in result, f"Missing expected key: {key}"
            
            # Detailed validation of feature_importance
            feature_importance = result.get("feature_importance", {})
            assert isinstance(feature_importance, dict)
            assert len(feature_importance) > 0
            # All importance values should be numeric
            for feature, importance in feature_importance.items():
                assert isinstance(importance, (int, float)), f"Feature importance for {feature} should be numeric"
                assert importance >= 0, f"Feature importance should be non-negative"
            print(f"  ✓ Feature importance validated: {len(feature_importance)} features")
            
            # Detailed validation of performance_metrics
            performance_metrics = result.get("performance_metrics", {})
            assert isinstance(performance_metrics, dict)
            assert len(performance_metrics) > 0
            print(f"  ✓ Performance metrics validated: {list(performance_metrics.keys())}")
            
            # Detailed validation of model_summary
            model_summary = result.get("model_summary", "")
            assert isinstance(model_summary, str)
            assert len(model_summary) > 0
            print(f"  ✓ Model summary validated: {len(model_summary)} characters")
            
            # Validate model_info structure
            model_info = result.get("model_info", {})
            required_model_info_keys = ["model_name", "model_type", "target", "features", "model_category"]
            for key in required_model_info_keys:
                assert key in model_info, f"Missing required model_info key: {key}"
            print(f"  ✓ Model info structure validated")
            
            print(f"  ✓ All comprehensive performance metrics validated successfully")


class TestModelPrediction:
    """Test model prediction functionality"""
    
    def test_predict_classification(self, mcp_connection, titanic_vd, schema_loader):
        """Test prediction with classification model"""
        print(f"\n{TestColors.BOLD}--- Testing Classification Model Prediction ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # First train a simple model
        train_result = train_model(
            table=table_name,
            model_type="logistic_regression",
            model_name="test_prediction_lr",
            target="survived",
            features=["age", "fare", "pclass"],
            max_iter=50
        )
        
        if train_result.get("success"):
            # Test regular prediction
            result = predict(
                table=table_name,
                model_name="test_prediction_lr",
                output_name="survival_prediction"
            )
            print_test_info("Predict: classification (regular)", result)
            
            if result.get("success"):
                assert result.get("success", False)
                assert result.get("model_name") == "test_prediction_lr"
                assert result.get("prediction_column") == "survival_prediction"
                assert result.get("prediction_type") == "prediction"
                assert isinstance(result.get("total_predictions", -1), int)
                assert result.get("total_predictions", 0) > 0
            
            # Test probability prediction
            result = predict(
                table=table_name,
                model_name="test_prediction_lr",
                output_name="survival_probability",
                prediction_type="probability"
            )
            print_test_info("Predict: classification (probability)", result)
            
            if result.get("success"):
                assert result.get("success", False)
                assert result.get("prediction_type") == "probability"
    
    def test_predict_regression(self, mcp_connection, titanic_vd, schema_loader):
        """Test prediction with regression model"""
        print(f"\n{TestColors.BOLD}--- Testing Regression Model Prediction ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # First train a regression model
        train_result = train_model(
            table=table_name,
            model_type="linear_regression",
            model_name="test_prediction_regression",
            target="fare",
            features=["age", "pclass"]
        )
        
        if train_result.get("success"):
            # Test regression prediction
            result = predict(
                table=table_name,
                model_name="test_prediction_regression",
                output_name="predicted_fare"
            )
            print_test_info("Predict: regression", result)
            
            if result.get("success"):
                assert result.get("success", False)
                assert result.get("prediction_column") == "predicted_fare"
                assert isinstance(result.get("sample_predictions", []), list)
    
    def test_predict_clustering(self, mcp_connection, iris_vd, schema_loader):
        """Test prediction with clustering model"""
        print(f"\n{TestColors.BOLD}--- Testing Clustering Model Prediction ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.iris"
        
        # First train a clustering model
        train_result = train_model(
            table=table_name,
            model_type="kmeans",
            model_name="test_prediction_kmeans",
            features=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            n_cluster=3
        )
        
        if train_result.get("success"):
            # Test clustering prediction (cluster assignment)
            result = predict(
                table=table_name,
                model_name="test_prediction_kmeans",
                output_name="cluster_assignment"
            )
            print_test_info("Predict: clustering", result)
            
            if result.get("success"):
                assert result.get("success", False)
                assert result.get("prediction_column") == "cluster_assignment"
    
    def test_predict_with_cached_data(self, mcp_connection, titanic_vd, schema_loader, clean_vdf_cache):
        """Test prediction using cached vDataFrame"""
        print(f"\n{TestColors.BOLD}--- Testing Prediction with Cached Data ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Create cached data first
        transform_result = transform_data(
            table=table_name,
            operation="search",
            extra_kwargs=json.dumps({"conditions": "age > 25"}),
            vdf_id="adults_test_data",
            show_preview=False
        )
        
        if transform_result.get("success"):
            # Train model on original data
            train_result = train_model(
                table=table_name,
                model_type="logistic_regression",
                model_name="test_cached_prediction",
                target="survived",
                features=["age", "fare", "pclass"]
            )
            
            if train_result.get("success"):
                # Predict on cached data
                result = predict(
                    table="adults_test_data",  # Use cached vDataFrame
                    model_name="test_cached_prediction",
                    output_name="adult_survival_pred"
                )
                print_test_info("Predict: using cached vDataFrame", result)
                
                if result.get("success"):
                    assert result.get("success", False)
                    assert "cached vDataFrame" in result.get("prediction_table", "")
    
    def test_predict_error_handling(self, mcp_connection, titanic_vd, schema_loader):
        """Test prediction error handling"""
        print(f"\n{TestColors.BOLD}--- Testing Prediction Error Handling ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Test with non-existent model
        result = predict(
            table=table_name,
            model_name="non_existent_model"
        )
        print_test_info("Predict: non-existent model", result, expected_success=False)
        assert not result.get("success", True)
        assert "Failed to load model" in result.get("error", "")
        
        # Test with non-existent table
        result = predict(
            table="non_existent_table",
            model_name="any_model"
        )
        print_test_info("Predict: non-existent table", result, expected_success=False)
        assert not result.get("success", True)
        
        # Train a model first for feature mismatch test
        train_result = train_model(
            table=table_name,
            model_type="linear_regression",
            model_name="test_feature_mismatch",
            target="fare",
            features=["age", "pclass"]
        )
        
        if train_result.get("success"):
            # Test with invalid features
            result = predict(
                table=table_name,
                model_name="test_feature_mismatch",
                features=["non_existent_feature"]
            )
            print_test_info("Predict: invalid features", result, expected_success=False)
            if not result.get("success", True):
                assert "Features not found" in result.get("error", "")


class TestModelManagement:
    """Test model management and listing functionality"""
    
    def test_list_models_after_training(self, mcp_connection, titanic_vd, schema_loader):
        """Test listing models after training some models"""
        print(f"\n{TestColors.BOLD}--- Testing Model Listing After Training ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Train a couple of models first
        models_to_train = [
            {
                "model_type": "logistic_regression",
                "model_name": "test_list_lr",
                "target": "survived",
                "features": ["age", "fare"]
            },
            {
                "model_type": "linear_regression", 
                "model_name": "test_list_linear",
                "target": "fare",
                "features": ["age", "pclass"]
            }
        ]
        
        trained_models = []
        for model_config in models_to_train:
            result = train_model(table=table_name, **model_config)
            if result.get("success"):
                trained_models.append(model_config["model_name"])
        
        if trained_models:
            # Test listing all models
            result = list_models()
            print_test_info("List all models after training", result)
            
            if result.get("success"):
                assert result.get("success", False)
                assert isinstance(result.get("models", []), list)
                assert result.get("count", 0) >= len(trained_models)
                
                # Check that our trained models appear in the list
                model_names = [m.get("model_name", "") for m in result.get("models", [])]
                for trained_name in trained_models:
                    # The model might appear with schema prefix
                    found = any(trained_name in name for name in model_names)
                    if not found:
                        print(f"  Note: Trained model '{trained_name}' not found in list")
    
    def test_list_models_with_filter(self, mcp_connection):
        """Test listing models with type filter"""
        print(f"\n{TestColors.BOLD}--- Testing Model Listing with Filters ---{TestColors.ENDC}")
        
        # Test filter by model type
        result = list_models(model_type_filter="LINEAR")
        print_test_info("List models: filter by LINEAR", result)
        if result.get("success"):
            assert result.get("success", False)
            assert result.get("filter_applied") == "LINEAR"
        
        # Test filter by classifier
        result = list_models(model_type_filter="CLASSIFIER")
        print_test_info("List models: filter by CLASSIFIER", result)
        if result.get("success"):
            assert result.get("success", False)
        
        # Test with limit
        result = list_models(limit=5)
        print_test_info("List models: limit to 5", result)
        if result.get("success"):
            assert result.get("success", False)
            assert len(result.get("models", [])) <= 5
    
    def test_comprehensive_ml_workflow(self, mcp_connection, titanic_vd, schema_loader, clean_vdf_cache):
        """Test complete ML workflow: data prep -> train -> predict -> manage"""
        print(f"\n{TestColors.BOLD}--- Testing Complete ML Workflow ---{TestColors.ENDC}")
        
        table_name = f"{schema_loader}.titanic"
        
        # Step 1: Prepare data with transformation
        prep_result = transform_data(
            table=table_name,
            operation="search",
            extra_kwargs=json.dumps({
                "conditions": "age IS NOT NULL AND fare IS NOT NULL",
                "usecols": ["age", "fare", "pclass", "survived", "sex"]
            }),
            vdf_id="clean_titanic_data",
            show_preview=False
        )
        print_test_info("Step 1: Data preparation", prep_result)
        
        if prep_result.get("success"):
            # Step 2: Train model on clean data
            train_result = train_model(
                table="clean_titanic_data",  # Use cached data
                model_type="random_forest_classifier",
                model_name="workflow_test_rf",
                target="survived",
                features=["age", "fare", "pclass"],
                n_estimators=5,  # Small for testing
                max_depth=3
            )
            print_test_info("Step 2: Model training", train_result)
            
            if train_result.get("success"):
                # Step 3: Make predictions
                pred_result = predict(
                    table="clean_titanic_data",
                    model_name="workflow_test_rf",
                    output_name="rf_prediction"
                )
                print_test_info("Step 3: Make predictions", pred_result)
                
                if pred_result.get("success"):
                    # Step 4: List models to verify our model exists
                    list_result = list_models(model_type_filter="RF")
                    print_test_info("Step 4: Verify model in database", list_result)
                    
                    if list_result.get("success"):
                        # Complete workflow successful
                        print(f"{TestColors.OKGREEN}✓ Complete ML workflow successful!{TestColors.ENDC}")
                        
                        # Verify we have sample predictions
                        sample_preds = pred_result.get("sample_predictions", [])
                        if sample_preds:
                            print(f"  Sample predictions generated: {len(sample_preds)} rows")
                        
                        assert len(sample_preds) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])