#!/usr/bin/env python3
"""
Simple test script to verify the ml_preprocessing_profiler installation and basic functionality.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from ml_preprocessing_profiler.core import evaluate_preprocessors
        from ml_preprocessing_profiler.preprocessors import get_scalers, get_encoders, get_imputers
        from ml_preprocessing_profiler.report import generate_report
        from ml_preprocessing_profiler.utils import print_data_summary
        print("✓ All modules imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with a simple dataset."""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from ml_preprocessing_profiler.core import evaluate_preprocessors
        
        # Create a simple dataset
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(y, name='target')
        
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Run evaluation
        results = evaluate_preprocessors(X, y, model, test_size=0.3, random_state=42)
        
        # Check results
        if isinstance(results, list) and len(results) > 0:
            print("✓ Basic functionality test passed!")
            print(f"  - Generated {len(results)} preprocessing combinations")
            print(f"  - Best score: {max(r['score'] for r in results):.4f}")
            return True
        else:
            print("✗ Basic functionality test failed: No results generated")
            return False
            
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_with_categorical_data():
    """Test with categorical data."""
    print("\nTesting with categorical data...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from ml_preprocessing_profiler.core import evaluate_preprocessors
        
        # Create dataset with categorical features
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5] * 20,
            'feature_2': ['A', 'B', 'A', 'C', 'B'] * 20,
            'feature_3': [0.1, 0.2, 0.3, 0.4, 0.5] * 20
        })
        y = pd.Series([0, 1, 0, 1, 0] * 20, name='target')
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = evaluate_preprocessors(X, y, model, test_size=0.3, random_state=42)
        
        if isinstance(results, list) and len(results) > 0:
            print("✓ Categorical data test passed!")
            return True
        else:
            print("✗ Categorical data test failed")
            return False
            
    except Exception as e:
        print(f"✗ Categorical data test failed: {e}")
        return False

def test_with_missing_data():
    """Test with missing data."""
    print("\nTesting with missing data...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from ml_preprocessing_profiler.core import evaluate_preprocessors
        
        # Create dataset with missing values
        X = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5] * 20,
            'feature_2': ['A', 'B', 'A', np.nan, 'B'] * 20,
            'feature_3': [0.1, 0.2, 0.3, 0.4, np.nan] * 20
        })
        y = pd.Series([0, 1, 0, 1, 0] * 20, name='target')
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = evaluate_preprocessors(X, y, model, test_size=0.3, random_state=42)
        
        if isinstance(results, list) and len(results) > 0:
            print("✓ Missing data test passed!")
            return True
        else:
            print("✗ Missing data test failed")
            return False
            
    except Exception as e:
        print(f"✗ Missing data test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ML Preprocessing Profiler - Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_with_categorical_data,
        test_with_missing_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The library is working correctly.")
        print("\nYou can now use the library:")
        print("""
from ml_preprocessing_profiler.core import evaluate_preprocessors
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
results = evaluate_preprocessors(X, y, model)
        """)
    else:
        print("❌ Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
