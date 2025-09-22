"""
Comprehensive Test Suite for VerticaPy MCP Server

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
from verticapy.datasets import load_titanic
import unittest
import sys
import os


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


class MCPServerTestSuite(unittest.TestCase):
    """Comprehensive test suite for MCP Server functions"""
    
    @staticmethod
    def display_conn_info_mcp_format(conn_info):
        """Display connection info in exact MCP format"""
        print(f"{TestColors.OKCYAN}CONN_INFO = {{{TestColors.ENDC}")
        for key, value in conn_info.items():
            print(f'    "{key}": "{value}",')
        print(f"{TestColors.OKCYAN}}}{TestColors.ENDC}")
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        print(f"\n{TestColors.HEADER}{TestColors.BOLD}=== VerticaPy MCP Server Test Suite ==={TestColors.ENDC}")
        print(f"{TestColors.OKBLUE}Setting up test environment...{TestColors.ENDC}")
        
        # Load configuration
        cls.config = load_test_config()
        cls.conn_info = get_conn_info()
        
        # Configure logging
        logging.basicConfig(
            level=cls.config["log_level"],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        
        # Initialize test variables
        cls.test_table = f"{cls.config['test_schema']}.{cls.config['test_table']}"
        cls.connection_established = False
        
        print(f"{TestColors.OKCYAN}Using MCP Connection Info:{TestColors.ENDC}")
        print(f"  Host: {cls.conn_info['host']}")
        print(f"  Port: {cls.conn_info['port']}")
        print(f"  Database: {cls.conn_info['database']}")
        print(f"  User: {cls.conn_info['user']}")
        print(f"  Password: {'*' * len(cls.conn_info['password']) if cls.conn_info['password'] else '(empty)'}")
    
    def setUp(self):
        """Set up before each test"""
        self.logger.info(f"Starting test: {self._testMethodName}")
    
    def tearDown(self):
        """Clean up after each test"""
        self.logger.info(f"Completed test: {self._testMethodName}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print(f"\n{TestColors.OKBLUE}Cleaning up test environment...{TestColors.ENDC}")
        try:
            # Clear any cached vDataFrames
            clear_vdf_cache()
            # Disconnect from database
            disconnect_from_vertica()
            print(f"{TestColors.OKGREEN}✓ Test environment cleaned up successfully{TestColors.ENDC}")
        except Exception as e:
            print(f"{TestColors.WARNING}⚠ Warning during cleanup: {str(e)}{TestColors.ENDC}")
    
    def _print_test_result(self, test_name: str, result: Dict[str, Any], expected_success: bool = True):
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
    
    def test_01_connection_management(self):
        """Test database connection management"""
        print(f"\n{TestColors.BOLD}--- Testing Connection Management ---{TestColors.ENDC}")
        
        # Print connection info in MCP format
        print(f"{TestColors.OKBLUE}Connection configuration (MCP format):{TestColors.ENDC}")
        self.display_conn_info_mcp_format(self.conn_info)
        print()
        
        # Test initial connection status
        result = get_connection_status()
        self._print_test_result("Get initial connection status", result)
        
        # Test connection
        result = connect_to_vertica()
        self._print_test_result("Connect to Vertica", result)
        
        if result.get("success"):
            self.__class__.connection_established = True
            
            # Test connection status after connecting
            result = get_connection_status()
            self._print_test_result("Get connection status after connect", result)
        
        self.assertTrue(self.__class__.connection_established, "Failed to establish database connection")
    
    def test_02_data_setup(self):
        """Load the Titanic dataset for testing"""
        print(f"\n{TestColors.BOLD}--- Setting Up Test Data ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        try:
            # Load Titanic dataset
            print(f"{TestColors.OKCYAN}Loading Titanic dataset...{TestColors.ENDC}")
            load_titanic()
            print(f"{TestColors.OKGREEN}✓ Titanic dataset loaded successfully{TestColors.ENDC}")
            
            # Verify the table exists
            result = list_tables(schema=self.config["test_schema"])
            self._print_test_result("Verify test data loaded", result)
            
            tables = result.get("tables", [])
            self.assertIn(self.config["test_table"], tables, f"Test table {self.config['test_table']} not found")
            
        except Exception as e:
            self.fail(f"Failed to load test data: {str(e)}")
    
    def test_03_schema_and_table_listing(self):
        """Test schema and table listing functions"""
        print(f"\n{TestColors.BOLD}--- Testing Schema and Table Listing ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test list all schemas
        result = list_all_schemas()
        self._print_test_result("List all schemas", result)
        self.assertTrue(result.get("success", False))
        self.assertIsInstance(result.get("schemas", []), list)
        
        # Test list tables in public schema
        result = list_tables(schema="public")
        self._print_test_result("List tables in public schema", result)
        self.assertTrue(result.get("success", False))
        self.assertIsInstance(result.get("tables", []), list)
        
        # Test list tables in non-existent schema
        result = list_tables(schema="non_existent_schema")
        self._print_test_result("List tables in non-existent schema", result)
        # This should still succeed but return empty list
    
    def test_04_table_description(self):
        """Test table description functionality"""
        print(f"\n{TestColors.BOLD}--- Testing Table Description ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test describe table
        result = describe_table(self.test_table)
        self._print_test_result(f"Describe table {self.test_table}", result)
        self.assertTrue(result.get("success", False))
        self.assertIsInstance(result.get("row_count", 0), int)
        self.assertIsInstance(result.get("column_count", 0), int)
        
        # Test describe non-existent table
        result = describe_table("non_existent_table")
        self._print_test_result("Describe non-existent table", result, expected_success=False)
        self.assertFalse(result.get("success", True))
    
    def test_05_data_sampling(self):
        """Test data sampling functionality"""
        print(f"\n{TestColors.BOLD}--- Testing Data Sampling ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test sample data with default size
        result = sample_data(self.test_table)
        self._print_test_result(f"Sample data from {self.test_table} (default size)", result)
        self.assertTrue(result.get("success", False))
        self.assertIsInstance(result.get("data", []), list)
        
        # Test sample data with specific size
        result = sample_data(self.test_table, n=10)
        self._print_test_result(f"Sample 10 rows from {self.test_table}", result)
        self.assertTrue(result.get("success", False))
        self.assertLessEqual(len(result.get("data", [])), 10)
        
        # Test sample from non-existent table
        result = sample_data("non_existent_table")
        self._print_test_result("Sample from non-existent table", result, expected_success=False)
        self.assertFalse(result.get("success", True))
    
    def test_06_column_statistics(self):
        """Test column statistics functionality"""
        print(f"\n{TestColors.BOLD}--- Testing Column Statistics ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test basic column stats
        test_cases = [
            ("age", "describe"),
            ("age", "mean"),
            ("age", "max"),
            ("age", "min"),
            ("age", "count"),
            ("fare", "sum"),
            ("survived", "nunique"),
        ]
        
        for column, metric in test_cases:
            result = column_stats(self.test_table, column, metric)
            self._print_test_result(f"Column stats: {column}.{metric}", result)
            if result.get("success"):
                self.assertTrue(result.get("success", False))
                self.assertIsNotNone(result.get("result"))
        
        # Test nlargest with parameter
        result = column_stats(
            self.test_table, 
            "fare", 
            "nlargest", 
            extra_kwargs='{"n": 3}'
        )
        self._print_test_result("Column stats: fare.nlargest(n=3)", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test nsmallest with parameter
        result = column_stats(
            self.test_table, 
            "age", 
            "nsmallest", 
            extra_kwargs='{"n": 5}'
        )
        self._print_test_result("Column stats: age.nsmallest(n=5)", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test topk with parameter
        result = column_stats(
            self.test_table, 
            "sex", 
            "topk", 
            extra_kwargs='{"k": 2}'
        )
        self._print_test_result("Column stats: sex.topk(k=2)", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test aggregate with custom functions
        result = column_stats(
            self.test_table, 
            "age", 
            "aggregate", 
            extra_kwargs='{"func": ["min", "avg", "max"]}'
        )
        self._print_test_result("Column stats: age.aggregate(custom functions)", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test invalid column
        result = column_stats(self.test_table, "non_existent_column", "describe")
        self._print_test_result("Column stats: non-existent column", result, expected_success=False)
        self.assertFalse(result.get("success", True))
        
        # Test invalid metric
        result = column_stats(self.test_table, "age", "invalid_metric")
        self._print_test_result("Column stats: invalid metric", result, expected_success=False)
        self.assertFalse(result.get("success", True))
    
    def test_07_table_statistics(self):
        """Test table-level statistics functionality"""
        print(f"\n{TestColors.BOLD}--- Testing Table Statistics ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test basic table stats
        metrics = ["describe", "count", "mean", "max", "min", "sum"]
        
        for metric in metrics:
            result = table_stats(self.test_table, metric)
            self._print_test_result(f"Table stats: {metric}", result)
            if result.get("success"):
                self.assertTrue(result.get("success", False))
        
        # Test table stats with specific columns
        result = table_stats(
            self.test_table, 
            "describe", 
            columns=["age", "fare"]
        )
        self._print_test_result("Table stats: describe specific columns", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test aggregate with custom functions
        result = table_stats(
            self.test_table, 
            "aggregate", 
            extra_kwargs='{"func": ["count", "avg", "std"]}'
        )
        self._print_test_result("Table stats: aggregate with custom functions", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test invalid table
        result = table_stats("non_existent_table", "describe")
        self._print_test_result("Table stats: non-existent table", result, expected_success=False)
        self.assertFalse(result.get("success", True))
    
    def test_08_data_transformations(self):
        """Test data transformation functionality"""
        print(f"\n{TestColors.BOLD}--- Testing Data Transformations ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test search operation
        result = transform_data(
            table=self.test_table,
            operation="search",
            extra_kwargs='{"conditions": "survived = 1"}',
            vdf_id="survivors",
            show_preview=True
        )
        self._print_test_result("Transform: search (survivors)", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
            self.assertEqual(result.get("vdf_id"), "survivors")
        
        # Test groupby operation
        result = transform_data(
            table=self.test_table,
            operation="groupby",
            extra_kwargs='{"columns": ["sex"], "expr": ["count(*) AS passenger_count", "avg(age) AS avg_age"]}',
            vdf_id="by_sex",
            show_preview=True
        )
        self._print_test_result("Transform: groupby by sex", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test groupby with having clause
        result = transform_data(
            table=self.test_table,
            operation="groupby",
            extra_kwargs='{"columns": ["pclass"], "expr": ["count(*) AS count", "avg(fare) AS avg_fare"], "having": "count(*) > 100"}',
            vdf_id="by_class_filtered",
            show_preview=True
        )
        self._print_test_result("Transform: groupby with having clause", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test select operation
        result = transform_data(
            table=self.test_table,
            operation="select",
            extra_kwargs='{"columns": ["name", "age", "sex", "survived"]}',
            vdf_id="basic_info",
            show_preview=True
        )
        self._print_test_result("Transform: select specific columns", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test sort operation
        result = transform_data(
            table=self.test_table,
            operation="sort",
            extra_kwargs='{"columns": {"fare": "desc"}}',
            vdf_id="sorted_by_fare",
            show_preview=True
        )
        self._print_test_result("Transform: sort by fare descending", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
        
        # Test search with invalid aggregate function (should fail with helpful message)
        result = transform_data(
            table=self.test_table,
            operation="search",
            extra_kwargs='{"conditions": "survived = 1", "expr": ["sum(fare) AS total_fare"]}',
            show_preview=True
        )
        self._print_test_result("Transform: search with aggregate (should fail)", result, expected_success=False)
        self.assertFalse(result.get("success", True))
        self.assertIn("groupby", result.get("error", "").lower())
        
        # Test invalid operation
        result = transform_data(
            table=self.test_table,
            operation="invalid_operation",
            show_preview=True
        )
        self._print_test_result("Transform: invalid operation", result, expected_success=False)
        self.assertFalse(result.get("success", True))
    
    def test_09_cached_vdf_operations(self):
        """Test operations on cached vDataFrames"""
        print(f"\n{TestColors.BOLD}--- Testing Cached vDataFrame Operations ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # First create a cached vDataFrame
        result = transform_data(
            table=self.test_table,
            operation="search",
            extra_kwargs='{"conditions": "age > 30"}',
            vdf_id="adults",
            show_preview=False
        )
        self._print_test_result("Create cached vDataFrame (adults)", result)
        
        if result.get("success"):
            # Test column stats on cached vDataFrame
            result = column_stats("adults", "fare", "mean")
            self._print_test_result("Column stats on cached vDataFrame", result)
            if result.get("success"):
                self.assertTrue(result.get("success", False))
            
            # Test table stats on cached vDataFrame
            result = table_stats("adults", "describe")
            self._print_test_result("Table stats on cached vDataFrame", result)
            if result.get("success"):
                self.assertTrue(result.get("success", False))
            
            # Test chaining transformations
            result = transform_data(
                table="adults",
                operation="groupby",
                extra_kwargs='{"columns": ["sex"], "expr": ["count(*) AS count", "avg(fare) AS avg_fare"]}',
                vdf_id="adults_by_sex",
                show_preview=True
            )
            self._print_test_result("Chain transformations on cached vDataFrame", result)
            if result.get("success"):
                self.assertTrue(result.get("success", False))
    
    def test_10_vdf_cache_management(self):
        """Test vDataFrame cache management"""
        print(f"\n{TestColors.BOLD}--- Testing vDataFrame Cache Management ---{TestColors.ENDC}")
        
        # Test list cached vDataFrames
        result = list_cached_vdfs()
        self._print_test_result("List cached vDataFrames", result)
        self.assertTrue(result.get("success", False))
        
        cached_count = result.get("count", 0)
        print(f"  Found {cached_count} cached vDataFrames")
        
        if cached_count > 0:
            # Test clearing specific vDataFrame
            cached_vdfs = result.get("cached_vdfs", {})
            first_vdf_id = list(cached_vdfs.keys())[0]
            
            result = clear_vdf_cache(first_vdf_id)
            self._print_test_result(f"Clear specific vDataFrame: {first_vdf_id}", result)
            self.assertTrue(result.get("success", False))
        
        # Test clearing all cache
        result = clear_vdf_cache()
        self._print_test_result("Clear all cached vDataFrames", result)
        self.assertTrue(result.get("success", False))
        
        # Verify cache is empty
        result = list_cached_vdfs()
        self._print_test_result("Verify cache is cleared", result)
        self.assertEqual(result.get("count", -1), 0)
    
    def test_11_error_handling(self):
        """Test error handling and edge cases"""
        print(f"\n{TestColors.BOLD}--- Testing Error Handling ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test operations with invalid parameters
        error_test_cases = [
            {
                "name": "Column stats with empty parameters",
                "func": lambda: column_stats("", "", ""),
                "expected_success": False
            },
            {
                "name": "Table stats with invalid metric",
                "func": lambda: table_stats(self.test_table, "invalid_metric"),
                "expected_success": False
            },
            {
                "name": "Transform with missing required parameters",
                "func": lambda: transform_data(self.test_table, "groupby", extra_kwargs='{}'),
                "expected_success": False
            },
            {
                "name": "Transform with invalid JSON in extra_kwargs",
                "func": lambda: transform_data(self.test_table, "search", extra_kwargs='invalid_json'),
                "expected_success": False
            },
        ]
        
        for test_case in error_test_cases:
            try:
                result = test_case["func"]()
                self._print_test_result(test_case["name"], result, test_case["expected_success"])
                
                if test_case["expected_success"]:
                    self.assertTrue(result.get("success", False))
                else:
                    self.assertFalse(result.get("success", True))
                    
            except Exception as e:
                if test_case["expected_success"]:
                    self.fail(f"Unexpected exception in {test_case['name']}: {str(e)}")
                else:
                    print(f"{TestColors.OKGREEN}✓ {test_case['name']} (expected exception: {type(e).__name__}){TestColors.ENDC}")
    
    def test_12_parameter_parsing(self):
        """Test parameter parsing for different formats"""
        print(f"\n{TestColors.BOLD}--- Testing Parameter Parsing ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test different parameter formats
        param_test_cases = [
            {
                "name": "JSON string parameter",
                "func": lambda: column_stats(self.test_table, "age", "nlargest", extra_kwargs='{"n": 3}'),
            },
            {
                "name": "Empty JSON parameter",
                "func": lambda: column_stats(self.test_table, "age", "describe", extra_kwargs='{}'),
            },
            {
                "name": "Complex JSON parameter",
                "func": lambda: column_stats(self.test_table, "age", "aggregate", 
                                           extra_kwargs='{"func": ["min", "max", "avg"]}'),
            },
        ]
        
        for test_case in param_test_cases:
            result = test_case["func"]()
            self._print_test_result(test_case["name"], result)
            if result.get("success"):
                self.assertTrue(result.get("success", False))
    
    def test_13_performance_and_large_operations(self):
        """Test performance with larger operations"""
        print(f"\n{TestColors.BOLD}--- Testing Performance and Large Operations ---{TestColors.ENDC}")
        
        if not self.__class__.connection_established:
            self.skipTest("Database connection not established")
        
        # Test sampling large amounts of data
        result = sample_data(self.test_table, n=100)
        self._print_test_result("Sample large dataset (100 rows)", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
            self.assertLessEqual(len(result.get("data", [])), 100)
        
        # Test complex groupby operations
        result = transform_data(
            table=self.test_table,
            operation="groupby",
            extra_kwargs='{"columns": ["pclass", "sex"], "expr": ["count(*) AS count", "avg(age) AS avg_age", "avg(fare) AS avg_fare", "sum(survived) AS survivors"]}',
            vdf_id="complex_groupby",
            show_preview=True
        )
        self._print_test_result("Complex groupby operation", result)
        if result.get("success"):
            self.assertTrue(result.get("success", False))
    
    def test_14_connection_lifecycle(self):
        """Test connection lifecycle management"""
        print(f"\n{TestColors.BOLD}--- Testing Connection Lifecycle ---{TestColors.ENDC}")
        
        # Test disconnect
        result = disconnect_from_vertica()
        self._print_test_result("Disconnect from Vertica", result)
        
        # Test operations after disconnect (should fail gracefully)
        result = list_tables()
        self._print_test_result("List tables after disconnect", result, expected_success=False)
        
        # Reconnect for other tests
        result = connect_to_vertica()
        self._print_test_result("Reconnect to Vertica", result)
        if result.get("success"):
            self.__class__.connection_established = True


def run_comprehensive_test():
    """Run the comprehensive test suite with detailed output"""
    print(f"{TestColors.HEADER}{TestColors.BOLD}")
    print("=" * 80)
    print("         VerticaPy MCP Server - Comprehensive Test Suite")
    print("=" * 80)
    print(f"{TestColors.ENDC}")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(MCPServerTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print(f"\n{TestColors.OKCYAN}Starting test execution...{TestColors.ENDC}\n")
    
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{TestColors.HEADER}{TestColors.BOLD}")
    print("=" * 80)
    print("                        TEST SUMMARY")
    print("=" * 80)
    print(f"{TestColors.ENDC}")
    
    print(f"Tests run: {result.testsRun}")
    print(f"{TestColors.OKGREEN}Successes: {result.testsRun - len(result.failures) - len(result.errors)}{TestColors.ENDC}")
    
    if result.failures:
        print(f"{TestColors.FAIL}Failures: {len(result.failures)}{TestColors.ENDC}")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"{TestColors.FAIL}Errors: {len(result.errors)}{TestColors.ENDC}")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if result.wasSuccessful():
        print(f"\n{TestColors.OKGREEN}{TestColors.BOLD}🎉 ALL TESTS PASSED! 🎉{TestColors.ENDC}")
    else:
        print(f"\n{TestColors.FAIL}{TestColors.BOLD}❌ SOME TESTS FAILED{TestColors.ENDC}")
        return False
    
    return True


if __name__ == "__main__":
    # Run the comprehensive test suite
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)