# VerticaPy MCP Server Test Suite

This directory contains a comprehensive test suite for the VerticaPy MCP (Model Context Protocol) server.

## Overview

The test suite thoroughly tests all MCP tools provided by `server2.py`, including:

- **Connection Management**: Database connection, disconnection, and status checking
- **Data Exploration**: Schema listing, table listing, table description, and data sampling
- **Statistical Analysis**: Column-level and table-level statistical computations
- **Data Transformations**: Search, groupby, join, pivot, select, and sort operations
- **Cache Management**: vDataFrame caching and management
- **Error Handling**: Comprehensive error handling and edge case testing
- **Parameter Parsing**: Testing different parameter formats and JSON parsing


## Files

- `test_pytest.py` - Main comprehensive test suite (pytest-based)
- `config.py` - Test configuration management
- `test_config.conf.example` - Example configuration file
- `run_tests.py` - Simple test runner script
- `README.md` - This documentation

## Setup

### 1. Database Connection

Create a configuration file from the example:

```bash
copy test_config.conf.example test_config.conf
```

Edit `test_config.conf` with your Vertica database connection details (matching MCP CONN_INFO format):

```ini
[mcp_test_config]
host = 10.10.10.69
port = 32796
database = verticadb21477
user = ughumman
password = 
log_level = INFO
test_schema = public
test_table = titanic
```

### 2. Environment Variables (Alternative)

You can also set environment variables instead of using the config file (matching MCP pattern):

```bash
export VERTICA_HOST=10.10.10.69
export VERTICA_PORT=32796
export VERTICA_DATABASE=verticadb21477
export VERTICA_USER=ughumman
export VERTICA_PASSWORD=your_password
```

## Running Tests

### Method 1: Using the Test Runner (Recommended)

```bash
cd verticapy/mcp/tests
python run_tests.py
```

The test runner automatically detects pytest and falls back to unittest if needed.


### Method 2: Using pytest directly (Recommended)

```bash
cd verticapy/mcp/tests
pytest test_pytest.py -v --tb=short --color=yes
```

### Method 3: Using specific framework

```bash
# Force pytest
python run_tests.py --framework pytest --verbose

# Force unittest
python run_tests.py --framework unittest
```


### Method 4: Using unittest directly

```bash
# Not applicable (no unittest-based test file retained)
```

## Test Structure


The test suite is organized into the following test categories (all in `test_pytest.py`):

1. **Connection Management** (`test_01_connection_management`)
   - Tests database connection establishment
   - Verifies connection status checking
   - Tests connection lifecycle

2. **Data Setup** (`test_02_data_setup`)
   - Loads the Titanic dataset using `load_titanic()`
   - Verifies test data is available

3. **Schema and Table Listing** (`test_03_schema_and_table_listing`)
   - Tests schema enumeration
   - Tests table listing by schema
   - Handles non-existent schemas gracefully

4. **Table Description** (`test_04_table_description`)
   - Tests table metadata retrieval
   - Verifies column information and data types
   - Tests error handling for invalid tables

5. **Data Sampling** (`test_05_data_sampling`)
   - Tests data sampling with various sizes
   - Verifies sample data structure
   - Tests error handling

6. **Column Statistics** (`test_06_column_statistics`)
   - Tests all statistical metrics (mean, max, min, etc.)
   - Tests parameterized operations (nlargest, nsmallest, topk)
   - Tests custom aggregation functions
   - Verifies parameter parsing

7. **Table Statistics** (`test_07_table_statistics`)
   - Tests table-level statistical operations
   - Tests column-specific statistics
   - Tests custom aggregation functions

8. **Data Transformations** (`test_08_data_transformations`)
   - Tests search operations with conditions
   - Tests groupby operations with aggregations
   - Tests select, sort, and other transformations
   - Tests error handling for invalid operations

9. **Cached vDataFrame Operations** (`test_09_cached_vdf_operations`)
   - Tests operations on cached transformed data
   - Tests chaining of transformations
   - Verifies cache functionality

10. **Cache Management** (`test_10_vdf_cache_management`)
    - Tests listing cached vDataFrames
    - Tests selective and complete cache clearing
    - Verifies cache state management

11. **Error Handling** (`test_11_error_handling`)
    - Tests graceful handling of invalid parameters
    - Tests error messages and response structure
    - Verifies system stability under error conditions

12. **Parameter Parsing** (`test_12_parameter_parsing`)
    - Tests JSON parameter parsing
    - Tests different parameter formats
    - Verifies complex parameter structures

13. **Performance Testing** (`test_13_performance_and_large_operations`)
    - Tests operations with larger datasets
    - Tests complex multi-column operations
    - Verifies performance characteristics

14. **Connection Lifecycle** (`test_14_connection_lifecycle`)
    - Tests disconnect functionality
    - Tests operations after disconnection
    - Tests reconnection capabilities

## Test Output

The test suite provides colorized, detailed output showing:

- ✓ Successful operations with key metrics
- ✗ Failed operations with error details
- Summary statistics (row counts, column counts, etc.)
- Performance information
- Comprehensive final summary

Example output:
```
=== VerticaPy MCP Server Test Suite ===
Setting up test environment...
Configuration loaded: Host=localhost, Database=vmart

--- Testing Connection Management ---
✓ Get initial connection status
  Message: Not connected to database
✓ Connect to Vertica
  Message: Successfully connected to Vertica database
...
```

## Requirements

- Python 3.7+
- VerticaPy
- Access to a Vertica database
- Required Python packages:
  - `pytest` (recommended: `pip install pytest`)
  - `unittest` (built-in, fallback)
  - `json` (built-in)
  - `logging` (built-in)
  - `configparser` (built-in)

## Troubleshooting

### Connection Issues

1. Verify your database connection details in `test_config.conf`
2. Ensure the Vertica database is running and accessible
3. Check firewall settings and network connectivity
4. Verify user permissions for database access

### Test Data Issues

1. Ensure the `load_titanic()` function works in your environment
2. Verify you have permissions to create tables in the test schema
3. Check if the Titanic dataset is available in your VerticaPy installation

### Import Issues

1. Ensure VerticaPy is properly installed
2. Verify the MCP server files are in the correct location
3. Check Python path configuration

### Test Failures

1. Review the detailed error output from failed tests
2. Check the log files if logging is enabled
3. Verify database connectivity during test execution
4. Ensure sufficient database resources are available

## Contributing

When adding new tests:

1. Follow the existing naming convention (`test_##_descriptive_name`)
2. Add detailed print statements showing test progress
3. Use the `_print_test_result()` helper for consistent output formatting
4. Test both success and failure scenarios
5. Update this README with new test descriptions

## License

This test suite is part of the VerticaPy project and follows the same licensing terms.