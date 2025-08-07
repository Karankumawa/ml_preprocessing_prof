#!/usr/bin/env python3
"""
ML Preprocessing Profiler - Usage Demo

This script demonstrates how to use the ml_preprocessing_profiler library
to compare different preprocessing techniques on machine learning datasets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_iris, load_boston, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import pandas as pd
import numpy as np

from ml_preprocessing_profiler.core import evaluate_preprocessors
from ml_preprocessing_profiler.utils import print_data_summary

def demo_classification():
    """Demonstrate preprocessing comparison for classification problems."""
    print("\n" + "="*60)
    print("CLASSIFICATION PROBLEM DEMO")
    print("="*60)
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    y = pd.Series(y, name='target')
    
    # Print dataset summary
    print_data_summary(X, y)
    
    # Test with different models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42)
    }
    
    for model_name, model in models.items():
        print(f"\nTesting with {model_name}...")
        try:
            results = evaluate_preprocessors(
                X, y, model, 
                problem_type='classification',
                test_size=0.2,
                random_state=42
            )
            print(f"✓ {model_name} evaluation completed successfully!")
        except Exception as e:
            print(f"✗ {model_name} evaluation failed: {e}")

def demo_regression():
    """Demonstrate preprocessing comparison for regression problems."""
    print("\n" + "="*60)
    print("REGRESSION PROBLEM DEMO")
    print("="*60)
    
    # Load Boston housing dataset (or California housing if Boston is not available)
    try:
        from sklearn.datasets import fetch_california_housing
        print("Loading California housing dataset...")
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.Series(housing.target, name='target')
    except:
        # Fallback to synthetic data
        print("Loading synthetic regression dataset...")
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                                 n_redundant=2, n_classes=2, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y, name='target')
    
    # Print dataset summary
    print_data_summary(X, y)
    
    # Test with different models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'SVR': SVR()
    }
    
    for model_name, model in models.items():
        print(f"\nTesting with {model_name}...")
        try:
            results = evaluate_preprocessors(
                X, y, model, 
                problem_type='regression',
                test_size=0.2,
                random_state=42
            )
            print(f"✓ {model_name} evaluation completed successfully!")
        except Exception as e:
            print(f"✗ {model_name} evaluation failed: {e}")

def demo_with_missing_data():
    """Demonstrate preprocessing with missing data."""
    print("\n" + "="*60)
    print("MISSING DATA DEMO")
    print("="*60)
    
    # Create dataset with missing values
    print("Creating dataset with missing values...")
    np.random.seed(42)
    X = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.choice(['A', 'B', 'C'], 100),
        'feature_4': np.random.randn(100)
    })
    
    # Introduce missing values
    X.loc[np.random.choice(X.index, 20), 'feature_1'] = np.nan
    X.loc[np.random.choice(X.index, 15), 'feature_2'] = np.nan
    X.loc[np.random.choice(X.index, 10), 'feature_3'] = np.nan
    
    y = pd.Series(np.random.choice([0, 1], 100), name='target')
    
    # Print dataset summary
    print_data_summary(X, y)
    
    # Test with Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print(f"\nTesting with RandomForest (handling missing data)...")
    
    try:
        results = evaluate_preprocessors(
            X, y, model, 
            problem_type='classification',
            test_size=0.2,
            random_state=42
        )
        print(f"✓ Missing data evaluation completed successfully!")
    except Exception as e:
        print(f"✗ Missing data evaluation failed: {e}")

def demo_simple_usage():
    """Demonstrate the simplest usage pattern."""
    print("\n" + "="*60)
    print("SIMPLE USAGE DEMO")
    print("="*60)
    
    print("This is the simplest way to use the library:")
    print("""
from ml_preprocessing_profiler.core import evaluate_preprocessors
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y = load_iris(return_X_y=True)

# Create model
model = RandomForestClassifier()

# Run evaluation (one line!)
results = evaluate_preprocessors(X, y, model)
    """)
    
    # Actually run it
    print("Running the simple example...")
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    try:
        results = evaluate_preprocessors(X, y, model)
        print("✓ Simple usage demo completed successfully!")
    except Exception as e:
        print(f"✗ Simple usage demo failed: {e}")

def main():
    """Run all demonstration examples."""
    print("ML Preprocessing Profiler - Usage Demonstrations")
    print("="*60)
    
    # Run all demos
    demo_simple_usage()
    demo_classification()
    demo_regression()
    demo_with_missing_data()
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETED!")
    print("="*60)
    print("\nKey takeaways:")
    print("1. The library automatically detects and applies appropriate preprocessing")
    print("2. It compares multiple combinations of scalers, encoders, and imputers")
    print("3. It generates comprehensive reports with visualizations")
    print("4. It works with any scikit-learn compatible model")
    print("5. It handles both classification and regression problems")
    print("6. It can process datasets with missing values and categorical features")

if __name__ == "__main__":
    main()
