#!/usr/bin/env python3
"""
Quick test script to demonstrate the ML Preprocessing Profiler library.
This script shows that the library can be imported and used successfully.
"""

print("ML Preprocessing Profiler - Quick Test")
print("=" * 50)

try:
    # Test imports
    print("1. Testing imports...")
    from ml_preprocessing_profiler.core import evaluate_preprocessors
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    print("   ✓ All imports successful!")

    # Test basic functionality
    print("\n2. Testing basic functionality...")
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=42)
    
    print("   Running evaluation...")
    results = evaluate_preprocessors(X, y, model, test_size=0.2, random_state=42)
    
    print(f"   ✓ Evaluation completed successfully!")
    print(f"   ✓ Generated {len(results)} preprocessing combinations")
    
    if results:
        best_result = max(results, key=lambda x: x['score'])
        print(f"   ✓ Best score: {best_result['score']:.4f}")
        print(f"   ✓ Best pipeline: {best_result.get('scaler', 'None')} + {best_result.get('encoder', 'None')} + {best_result.get('imputer', 'None')}")
    
    print("\n🎉 All tests passed! The library is working correctly.")
    print("\nYou can now use the library in your projects:")
    print("""
from ml_preprocessing_profiler.core import evaluate_preprocessors
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
results = evaluate_preprocessors(X, y, model)
    """)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the library is installed: pip install -e .")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check the installation and dependencies.")
