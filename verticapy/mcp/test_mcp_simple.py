#!/usr/bin/env python3
"""
Simple test script for VerticaPy MCP Server components
Run this to quickly test if everything is working before starting the full server.
"""

import sys
import asyncio
import json
import verticapy as vp

def setup_connection():

    # Creating a vertica_python connection directory.
    conn_info = {
    "host": "10.10.10.235", # ex: 127.0.0.1
    "port": "34101",
    "database": "verticadb21477", # ex: testdb
    "password": "",
    "user": "ughumman", # ex: dbadmin
    }

    # Creating a new auto connection.
    vp.new_connection(
    conn_info,
    name = "VerticaDSN",
    auto = True,
    overwrite = True,
    )

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import verticapy as vp
        print("  ✅ VerticaPy imported successfully")
    except ImportError as e:
        print(f"  ❌ VerticaPy import failed: {e}")
        return False
    
    try:
        from verticapy.mcp.tools import ToolRegistry
        print("  ✅ ToolRegistry imported successfully")
    except ImportError as e:
        print(f"  ❌ ToolRegistry import failed: {e}")
        return False
    
    try:
        from verticapy.mcp.client import start_mcp_server, stop_mcp_server
        print("  ✅ MCP client imported successfully")
    except ImportError as e:
        print(f"  ❌ MCP client import failed: {e}")
        return False
    
    try:
        import websockets
        print("  ✅ WebSockets imported successfully")
    except ImportError as e:
        print(f"  ❌ WebSockets import failed: {e}")
        print("  💡 Install with: pip install websockets")
        return False
    
    return True

def test_tool_registry():
    """Test tool registry functionality"""
    print("\n🔧 Testing Tool Registry...")
    
    try:
        from verticapy.mcp.tools import ToolRegistry
        registry = ToolRegistry()
        
        tools = list(registry.tools.keys())
        print(f"  ✅ Registry created with {len(tools)} tools: {tools}")
        
        # Test tool list format
        tools_list = registry.get_tool_list()
        print(f"  ✅ Tool list generated successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Tool registry test failed: {e}")
        return False

async def test_create_dataframe():
    """Test create_dataframe tool"""
    print("\n📊 Testing create_dataframe tool...")
    
    try:
        from verticapy.mcp.tools import ToolRegistry
        registry = ToolRegistry()
        
        # Test with sample data
        test_data = {
            "y_true": [1, 1.5, 3, 2, 5],
            "y_pred": [1.1, 1.55, 2.9, 2.01, 4.5],
        }
        
        result = await registry.execute_tool("create_dataframe", {
            "data": test_data,
            "name": "test_df"
        })
        
        if result.get("success"):
            print(f"  ✅ DataFrame created successfully!")
            print(f"     Shape: {result.get('shape')}")
            print(f"     Columns: {result.get('columns')}")
        else:
            print(f"  ❌ DataFrame creation failed: {result.get('error')}")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ create_dataframe test failed: {e}")
        return False

def test_direct_verticapy():
    """Test direct VerticaPy functionality"""
    print("\n🐍 Testing direct VerticaPy vDataFrame creation...")
    
    try:
        import verticapy as vp
        
        test_data = {
            "col1": [1, 2, 3],
            "col2": [1.1, 2.2, 3.3],
        }
        
        df = vp.vDataFrame(test_data)
        print(f"  ✅ Direct vDataFrame creation successful!")
        print(f"     Shape: {df.shape()}")
        print(f"     Columns: {df.get_columns()}")
        
        return True
    except Exception as e:
        print(f"  ❌ Direct VerticaPy test failed: {e}")
        return False

def test_server_creation():
    """Test MCP server creation (without starting)"""
    print("\n🌐 Testing MCP Server creation...")
    
    try:
        from verticapy.mcp.server import MCPServer
        
        server = MCPServer(host="localhost", port=8765)
        print("  ✅ MCP Server instance created successfully")
        
        # Test tool registry integration
        tools = server.tool_registry.get_tool_list()
        print(f"  ✅ Server has access to {len(tools)} tools")
        
        return True
    except Exception as e:
        print(f"  ❌ MCP Server creation failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 VerticaPy MCP Server Component Test")
    print("=" * 60)
    
    # Setup connection first
    setup_connection()
    
    tests = [
        ("Import Test", test_imports),
        ("Tool Registry Test", test_tool_registry),
        ("Direct VerticaPy Test", test_direct_verticapy),
        ("Create DataFrame Tool Test", test_create_dataframe),
        ("MCP Server Creation Test", test_server_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! You can now start the MCP server:")
        print("   python -m verticapy.mcp.server")
        print("\nOr directly:")
        print("   python verticapy/mcp/server.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   If only import tests failed, you can still try starting the server:")
        print("   python verticapy/mcp/server.py")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())