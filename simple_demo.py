#!/usr/bin/env python3
"""
Simple Demo Script for ML Preprocessing Profiler
आसान डेमो स्क्रिप्ट जो library का उपयोग दिखाती है
"""

print("🚀 ML Preprocessing Profiler - Simple Demo")
print("=" * 50)

# Method 1: सबसे आसान तरीका (Recommended)
print("\n📝 Method 1: सबसे आसान तरीका")
print("from ml_preprocessing_profiler import evaluate_preprocessors")

try:
    from ml_preprocessing_profiler import evaluate_preprocessors
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    
    print("✅ Import successful!")
    
    # Load data
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=42)
    
    print(f"📊 Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print("🤖 Model: RandomForestClassifier")
    
    # Run evaluation
    print("\n🔄 Running preprocessing evaluation...")
    results = evaluate_preprocessors(X, y, model, test_size=0.2, random_state=42)
    
    print(f"✅ Evaluation completed!")
    print(f"📈 Generated {len(results)} preprocessing combinations")
    
    if results:
        best_result = max(results, key=lambda x: x['score'])
        print(f"🏆 Best score: {best_result['score']:.4f}")
        print(f"🔧 Best pipeline: {best_result.get('scaler', 'None')} + {best_result.get('encoder', 'None')} + {best_result.get('imputer', 'None')}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Method 2: Alternative import method
print("\n" + "=" * 50)
print("📝 Method 2: Alternative import method")
print("from ml_preprocessing_profiler.core import evaluate_preprocessors")

try:
    from ml_preprocessing_profiler.core import evaluate_preprocessors
    print("✅ Alternative import successful!")
except Exception as e:
    print(f"❌ Error: {e}")

# Method 3: Direct package import
print("\n" + "=" * 50)
print("📝 Method 3: Direct package import")
print("import ml_preprocessing_profiler")

try:
    import ml_preprocessing_profiler
    print(f"✅ Package import successful!")
    print(f"📦 Version: {ml_preprocessing_profiler.__version__}")
    print(f"👨‍💻 Author: {ml_preprocessing_profiler.__author__}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("🎉 Demo completed successfully!")
print("\n💡 अब आप library का उपयोग कर सकते हैं:")
print("""
# सबसे आसान तरीका:
from ml_preprocessing_profiler import evaluate_preprocessors
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
results = evaluate_preprocessors(X, y, model)
""")

print("\n🔧 अगर अभी भी error आए तो:")
print("1. सही directory में हैं या नहीं चेक करें")
print("2. pip install -e . चलाएं")
print("3. Python environment सही है या नहीं चेक करें")
print("4. python -c 'import ml_preprocessing_profiler' टेस्ट करें")
