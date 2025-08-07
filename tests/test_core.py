"""
Tests for the core module of ml_preprocessing_profiler.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_preprocessing_profiler.core import evaluate_preprocessors, calculate_score, clone_model

class TestCore(unittest.TestCase):
    """Test cases for core functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create test datasets
        X_clf, y_clf = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=42)
        
        self.X_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(5)])
        self.y_clf = pd.Series(y_clf, name='target')
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(5)])
        self.y_reg = pd.Series(y_reg, name='target')
        
        # Create test models
        self.clf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    def test_calculate_score_classification(self):
        """Test score calculation for classification."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        score = calculate_score(y_true, y_pred, 'classification')
        expected_score = 0.6  # 3 correct out of 5
        
        self.assertAlmostEqual(score, expected_score, places=5)
    
    def test_calculate_score_regression(self):
        """Test score calculation for regression."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        score = calculate_score(y_true, y_pred, 'regression')
        
        # R² score should be close to 1 for good predictions
        self.assertGreater(score, 0.9)
    
    def test_calculate_score_invalid_type(self):
        """Test that invalid problem type raises error."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        with self.assertRaises(ValueError):
            calculate_score(y_true, y_pred, 'invalid_type')
    
    def test_clone_model(self):
        """Test model cloning functionality."""
        original_model = RandomForestClassifier(n_estimators=10, random_state=42)
        cloned_model = clone_model(original_model)
        
        # Check that they are different objects
        self.assertIsNot(original_model, cloned_model)
        
        # Check that they have the same parameters
        self.assertEqual(original_model.get_params(), cloned_model.get_params())
    
    def test_evaluate_preprocessors_classification(self):
        """Test preprocessing evaluation for classification."""
        results = evaluate_preprocessors(
            self.X_clf, self.y_clf, self.clf_model,
            problem_type='classification',
            test_size=0.3,
            random_state=42
        )
        
        # Check that results are returned
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for result in results:
            self.assertIn('scaler', result)
            self.assertIn('encoder', result)
            self.assertIn('imputer', result)
            self.assertIn('score', result)
            self.assertIn('time', result)
            self.assertIn('combination', result)
            
            # Check score is valid
            self.assertGreaterEqual(result['score'], 0.0)
            self.assertLessEqual(result['score'], 1.0)
            
            # Check time is positive
            self.assertGreaterEqual(result['time'], 0.0)
    
    def test_evaluate_preprocessors_regression(self):
        """Test preprocessing evaluation for regression."""
        results = evaluate_preprocessors(
            self.X_reg, self.y_reg, self.reg_model,
            problem_type='regression',
            test_size=0.3,
            random_state=42
        )
        
        # Check that results are returned
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for result in results:
            self.assertIn('scaler', result)
            self.assertIn('encoder', result)
            self.assertIn('imputer', result)
            self.assertIn('score', result)
            self.assertIn('time', result)
            self.assertIn('combination', result)
            
            # Check time is positive
            self.assertGreaterEqual(result['time'], 0.0)
    
    def test_evaluate_preprocessors_with_numpy_arrays(self):
        """Test that the function works with numpy arrays."""
        X_np = self.X_clf.values
        y_np = self.y_clf.values
        
        results = evaluate_preprocessors(
            X_np, y_np, self.clf_model,
            problem_type='classification',
            test_size=0.3,
            random_state=42
        )
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
    
    def test_evaluate_preprocessors_custom_test_size(self):
        """Test with custom test size."""
        results = evaluate_preprocessors(
            self.X_clf, self.y_clf, self.clf_model,
            problem_type='classification',
            test_size=0.4,
            random_state=42
        )
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main()
