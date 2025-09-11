# test_verticapy_components.py

import json
import sys
import os

# Add the current directory to Python path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import verticapy as vp

# Test configuration
CONN_INFO = {
    "host": "10.10.10.235",
    "port": "34101",
    "database": "verticadb21477",
    "password": "",
    "user": "ughumman",
}

def test_connection():
    """Test basic VerticaPy connection."""
    print("Testing VerticaPy connection...")
    try:
        vp.new_connection(
            CONN_INFO,
            name="VerticaDSN",
            auto=True,
            overwrite=True,
        )
        print("✓ Connection successful")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_dataframe_creation():
    """Test creating a vDataFrame with sample data."""
    print("\nTesting vDataFrame creation...")
    
    sample_data = {
        "y_true": [1, 1.5, 3, 2, 5],
        "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
    }
    
    try:
        df = vp.vDataFrame(sample_data)
        print("✓ vDataFrame created successfully")
        
        # Test basic operations
        print(f"  Shape: {df.shape()}")
        print(f"  Columns: {df.get_columns()}")
        print(f"  Data types: {df.dtypes()}")
        
        # Show first few rows
        print("  First 3 rows:")
        print(df.head(3))
        
        return True
    except Exception as e:
        print(f"✗ vDataFrame creation failed: {e}")
        return False

def test_json_parsing():
    """Test JSON parsing for data input."""
    print("\nTesting JSON parsing...")
    
    test_json = '{"y_true": [1, 1.5, 3, 2, 5], "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5]}'
    
    try:
        data = json.loads(test_json)
        print("✓ JSON parsing successful")
        print(f"  Parsed data: {data}")
        return True
    except Exception as e:
        print(f"✗ JSON parsing failed: {e}")
        return False

def test_mcp_tools_logic():
    """Test the logic that will be used in MCP tools."""
    print("\nTesting MCP tool logic...")
    
    try:
        # Test the connection logic
        print("Testing connection logic...")
        connection_success = test_connection()
        
        if not connection_success:
            print("✗ Cannot proceed with dataframe tests without connection")
            return False
        
        # Test dataframe creation logic
        print("Testing dataframe creation logic...")
        test_json = '{"y_true": [1, 1.5, 3, 2, 5], "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5]}'
        
        # Parse JSON
        data = json.loads(test_json)
        
        # Create dataframe
        df = vp.vDataFrame(data)
        
        # Generate info response (similar to what MCP tool will return)
        info = []
        info.append(f"Created vDataFrame with shape: {df.shape()}")
        info.append(f"Columns: {df.get_columns()}")
        head_data = df.head(5)
        info.append(f"First 5 rows:\n{head_data}")
        
        response = "\n".join(info)
        print("✓ MCP tool logic test successful")
        print("Response that would be returned:")
        print(response)
        
        return True
        
    except Exception as e:
        print(f"✗ MCP tool logic test failed: {e}")
        return False

def test_error_handling():
    """Test error handling scenarios."""
    print("\nTesting error handling...")
    
    # Test invalid JSON
    try:
        invalid_json = '{"y_true": [1, 1.5, 3, 2, 5], "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5'  # Missing closing brace
        data = json.loads(invalid_json)
        print("✗ Should have failed on invalid JSON")
    except json.JSONDecodeError:
        print("✓ Invalid JSON handling works")
    
    # Test empty data
    try:
        empty_data = {}
        df = vp.vDataFrame(empty_data)
        print("✗ Should have failed on empty data")
    except Exception:
        print("✓ Empty data handling works")
    
    return True

def run_all_tests():
    """Run all component tests."""
    print("=" * 50)
    print("VERTICAPY MCP SERVER COMPONENT TESTS")
    print("=" * 50)
    
    tests = [
        test_json_parsing,
        test_connection,
        test_dataframe_creation,
        test_mcp_tools_logic,
        test_error_handling,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Your MCP server components are ready.")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    run_all_tests()