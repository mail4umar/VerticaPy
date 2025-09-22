#!/usr/bin/env python3
"""
Test Runner for VerticaPy MCP Server

Script to run the comprehensive test suite using either pytest or unittest.
"""

import sys
import os
import argparse

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_pytest_tests(verbose=True):
    """Run tests using pytest"""
    try:
        import pytest
        
        args = ["test_pytest.py", "-v", "--tb=short", "--color=yes"]
        if verbose:
            args.extend(["-s", "--capture=no"])
        
        print("🧪 Running MCP Server tests with pytest...")
        result = pytest.main(args)
        return result == 0
        
    except ImportError:
        print("❌ pytest not available. Please install pytest: pip install pytest")
        return False


def run_unittest_tests():
    """Run tests using unittest"""
    try:
        from test import run_comprehensive_test
        print("🧪 Running MCP Server tests with unittest...")
        return run_comprehensive_test()
    except ImportError as e:
        print(f"❌ Could not import unittest tests: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run VerticaPy MCP Server tests")
    parser.add_argument(
        "--framework", 
        choices=["pytest", "unittest", "auto"], 
        default="auto",
        help="Test framework to use (default: auto)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("VerticaPy MCP Server Test Runner")
    print("=" * 40)
    
    success = False
    
    try:
        if args.framework == "pytest":
            success = run_pytest_tests(args.verbose)
        elif args.framework == "unittest":
            success = run_unittest_tests()
        else:  # auto
            # Try pytest first, fallback to unittest
            try:
                import pytest
                print("📋 Auto-detected pytest, using pytest framework...")
                success = run_pytest_tests(args.verbose)
            except ImportError:
                print("📋 pytest not available, falling back to unittest...")
                success = run_unittest_tests()
        
        if success:
            print(f"\n🎉 All tests completed successfully!")
        else:
            print(f"\n❌ Some tests failed or encountered errors.")
            
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error during test execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()