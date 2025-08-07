#!/usr/bin/env python3
"""
Comprehensive Test Script for ML Preprocessing Profiler
Tests all possible import methods and provides detailed installation information.
"""

import sys
import os

print("🔍 ML Preprocessing Profiler - Comprehensive Test")
print("=" * 60)

# Test 1: Check Python environment
print("\n1. Python Environment Check:")
print(f"   Python Version: {sys.version}")
print(f"   Python Executable: {sys.executable}")
print(f"   Current Working Directory: {os.getcwd()}")

# Test 2: Check Python path
print("\n2. Python Path Check:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    print(f"   {i+1}. {path}")
if len(sys.path) > 5:
    print(f"   ... and {len(sys.path) - 5} more paths")

# Test 3: Check if package is installed
print("\n3. Package Installation Check:")
try:
    import pkg_resources
    installed_packages = [d for d in pkg_resources.working_set if 'ml_preprocessing' in d.project_name.lower()]
    if installed_packages:
        for pkg in installed_packages:
            print(f"   ✓ Found: {pkg.project_name} {pkg.version}")
            print(f"     Location: {pkg.location}")
    else:
        print("   ❌ No ml_preprocessing_profiler found in installed packages")
except ImportError:
    print("   ⚠️  Could not check installed packages")

# Test 4: Test different import methods
print("\n4. Import Method Tests:")

# Method 1: Direct package import
try:
    import ml_preprocessing_profiler
    print("   ✓ Method 1: import ml_preprocessing_profiler")
    print(f"     Version: {ml_preprocessing_profiler.__version__}")
    print(f"     Author: {ml_preprocessing_profiler.__author__}")
except ImportError as e:
    print(f"   ❌ Method 1 failed: {e}")

# Method 2: From package import specific function
try:
    from ml_preprocessing_profiler import evaluate_preprocessors
    print("   ✓ Method 2: from ml_preprocessing_profiler import evaluate_preprocessors")
except ImportError as e:
    print(f"   ❌ Method 2 failed: {e}")

# Method 3: From package.core import function
try:
    from ml_preprocessing_profiler.core import evaluate_preprocessors
    print("   ✓ Method 3: from ml_preprocessing_profiler.core import evaluate_preprocessors")
except ImportError as e:
    print(f"   ❌ Method 3 failed: {e}")

# Method 4: Import all utilities
try:
    from ml_preprocessing_profiler import (
        validate_input_data,
        detect_problem_type,
        get_data_summary,
        print_data_summary
    )
    print("   ✓ Method 4: Imported utility functions")
except ImportError as e:
    print(f"   ❌ Method 4 failed: {e}")

# Test 5: Functional test
print("\n5. Functional Test:")
try:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    
    # Load data
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=42)
    
    # Test the main function
    results = evaluate_preprocessors(X, y, model, test_size=0.2, random_state=42)
    
    print("   ✓ Functional test passed!")
    print(f"   ✓ Generated {len(results)} preprocessing combinations")
    
    if results:
        best_result = max(results, key=lambda x: x['score'])
        print(f"   ✓ Best score: {best_result['score']:.4f}")
    
except Exception as e:
    print(f"   ❌ Functional test failed: {e}")

# Test 6: Check package structure
print("\n6. Package Structure Check:")
try:
    import ml_preprocessing_profiler
    package_dir = os.path.dirname(ml_preprocessing_profiler.__file__)
    print(f"   Package directory: {package_dir}")
    
    if os.path.exists(package_dir):
        files = [f for f in os.listdir(package_dir) if f.endswith('.py')]
        print(f"   Python files in package: {files}")
    else:
        print("   ❌ Package directory not found")
        
except Exception as e:
    print(f"   ❌ Could not check package structure: {e}")

print("\n" + "=" * 60)
print("📋 Summary:")
print("If all tests show ✓, your library is working correctly!")
print("If you see ❌, there might be an installation issue.")
print("\n💡 Usage Examples:")
print("""
# Method 1 (Recommended):
from ml_preprocessing_profiler import evaluate_preprocessors

# Method 2:
from ml_preprocessing_profiler.core import evaluate_preprocessors

# Method 3:
import ml_preprocessing_profiler
results = ml_preprocessing_profiler.evaluate_preprocessors(X, y, model)
""")

print("\n🔧 Troubleshooting:")
print("If you still get import errors:")
print("1. Make sure you're in the correct directory")
print("2. Run: pip install -e .")
print("3. Check if you're using the same Python environment")
print("4. Try: python -c 'import ml_preprocessing_profiler'")
