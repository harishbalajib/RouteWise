#!/usr/bin/env python3
"""
Test runner for Smart US Travel Planner
"""

import sys
import os
import subprocess
import time

# Add parent directory to path so tests can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test(test_file):
    """Run a specific test file"""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("Test completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print("Test failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running test: {e}")
        return False

def run_all_tests():
    """Run all test files"""
    test_files = [
        "test_mock_data.py",
        "test_all_algorithms.py", 
        "test_api_integration.py",
        "test_place_search.py",
        "test_preferred_timing.py",
        "test_api_key.py",
        "debug_api.py"
    ]
    
    print("Smart US Travel Planner - Test Suite")
    print("="*60)
    print(f"Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            if run_test(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠ Test file not found: {test_file}")
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed")
        return False

def run_specific_test(test_name):
    """Run a specific test by name"""
    test_file = f"test_{test_name}.py"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    return run_test(test_file)

def show_available_tests():
    """Show available test files"""
    test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
    debug_files = [f for f in os.listdir('.') if f.startswith('debug_') and f.endswith('.py')]
    
    print("Available tests:")
    print("-" * 30)
    
    for test_file in sorted(test_files + debug_files):
        name = test_file.replace('.py', '').replace('test_', '').replace('debug_', '')
        print(f"  {name}")
    
    print("\nUsage:")
    print("  python run_tests.py                    # Run all tests")
    print("  python run_tests.py <test_name>        # Run specific test")
    print("  python run_tests.py --list             # Show available tests")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--list" or sys.argv[1] == "-l":
            show_available_tests()
        else:
            # Run specific test
            test_name = sys.argv[1]
            success = run_specific_test(test_name)
            sys.exit(0 if success else 1)
    else:
        print("Usage: python run_tests.py [test_name|--list]")
        sys.exit(1) 