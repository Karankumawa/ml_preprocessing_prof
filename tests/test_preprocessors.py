"""
Tests for the preprocessors module of ml_preprocessing_profiler.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_preprocessing_profiler.preprocessors import (
    get_scalers, get_encoders, get_imputers, 
    is_classification, get_preprocessing_summary
)

class TestPreprocessors(unittest.TestCase):
    """Test cases for preprocessors functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create test datasets
        self.X_numerical = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [10, 20, 30, 40, 50],
            'feature_3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        self.X_categorical = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': ['A', 'B', 'A', 'C', 'B'],
            'feature_3': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        self.X_missing = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],
            'feature_2': ['A', 'B', 'A', np.nan, 'B'],
            'feature_3': [0.1, 0.2, 0.3, 0.4, np.nan]
        })
    
    def test_get_scalers(self):
        """Test that get_scalers returns expected scalers."""
        scalers = get_scalers()
        
        # Check that expected scalers are present
        expected_scalers = ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None']
        for scaler_name in expected_scalers:
            self.assertIn(scaler_name, scalers)
        
        # Check that scalers are actual scaler objects or None
        for scaler_name, scaler in scalers.items():
            if scaler_name != 'None':
                self.assertIsNotNone(scaler)
                # Check that it has fit_transform method
                self.assertTrue(hasattr(scaler, 'fit_transform'))
            else:
                self.assertIsNone(scaler)
    
    def test_get_encoders_numerical_data(self):
        """Test get_encoders with numerical data."""
        encoders = get_encoders(self.X_numerical)
        
        # Should only have 'None' option for numerical data
        self.assertIn('None', encoders)
        self.assertEqual(len(encoders), 1)
    
    def test_get_encoders_categorical_data(self):
        """Test get_encoders with categorical data."""
        encoders = get_encoders(self.X_categorical)
        
        # Should have encoders for categorical data
        expected_encoders = ['OneHotEncoder', 'OrdinalEncoder', 'None']
        for encoder_name in expected_encoders:
            self.assertIn(encoder_name, encoders)
        
        # Check that encoders are actual encoder objects or None
        for encoder_name, encoder in encoders.items():
            if encoder_name != 'None':
                self.assertIsNotNone(encoder)
                # Check that it has fit_transform method
                self.assertTrue(hasattr(encoder, 'fit_transform'))
            else:
                self.assertIsNone(encoder)
    
    def test_get_imputers_no_missing_data(self):
        """Test get_imputers with no missing data."""
        imputers = get_imputers(self.X_numerical)
        
        # Should only have 'None' option for data without missing values
        self.assertIn('None', imputers)
        self.assertEqual(len(imputers), 1)
    
    def test_get_imputers_with_missing_data(self):
        """Test get_imputers with missing data."""
        imputers = get_imputers(self.X_missing)
        
        # Should have imputers for data with missing values
        expected_imputers = [
            'SimpleImputer_Mean', 'SimpleImputer_Median', 
            'SimpleImputer_MostFrequent', 'KNNImputer', 'None'
        ]
        for imputer_name in expected_imputers:
            self.assertIn(imputer_name, imputers)
        
        # Check that imputers are actual imputer objects or None
        for imputer_name, imputer in imputers.items():
            if imputer_name != 'None':
                self.assertIsNotNone(imputer)
                # Check that it has fit_transform method
                self.assertTrue(hasattr(imputer, 'fit_transform'))
            else:
                self.assertIsNone(imputer)
    
    def test_is_classification(self):
        """Test is_classification function."""
        # Numerical data should return False
        self.assertFalse(is_classification(self.X_numerical))
        
        # Categorical data should return True
        self.assertTrue(is_classification(self.X_categorical))
    
    def test_get_preprocessing_summary(self):
        """Test get_preprocessing_summary function."""
        summary = get_preprocessing_summary(self.X_categorical)
        
        # Check that summary contains expected keys
        expected_keys = ['shape', 'dtypes', 'missing_values', 'categorical_columns', 'numerical_columns', 'memory_usage']
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check specific values
        self.assertEqual(summary['shape'], (5, 3))
        self.assertEqual(summary['missing_values'], 0)
        self.assertEqual(len(summary['categorical_columns']), 1)  # feature_2
        self.assertEqual(len(summary['numerical_columns']), 2)    # feature_1, feature_3
        self.assertGreater(summary['memory_usage'], 0)
    
    def test_get_preprocessing_summary_with_missing_data(self):
        """Test get_preprocessing_summary with missing data."""
        summary = get_preprocessing_summary(self.X_missing)
        
        # Should detect missing values
        self.assertGreater(summary['missing_values'], 0)
        
        # Should have both categorical and numerical columns
        self.assertGreater(len(summary['categorical_columns']), 0)
        self.assertGreater(len(summary['numerical_columns']), 0)
    
    def test_scaler_functionality(self):
        """Test that scalers actually work."""
        scalers = get_scalers()
        
        for scaler_name, scaler in scalers.items():
            if scaler_name != 'None':
                # Test that scaler can fit and transform
                X_scaled = scaler.fit_transform(self.X_numerical)
                
                # Check that output has same shape
                self.assertEqual(X_scaled.shape, self.X_numerical.shape)
                
                # Check that output is numpy array
                self.assertIsInstance(X_scaled, np.ndarray)
    
    def test_encoder_functionality(self):
        """Test that encoders actually work."""
        encoders = get_encoders(self.X_categorical)
        
        for encoder_name, encoder in encoders.items():
            if encoder_name != 'None':
                # Test that encoder can fit and transform
                X_encoded = encoder.fit_transform(self.X_categorical)
                
                # Check that output is numpy array
                self.assertIsInstance(X_encoded, np.ndarray)
    
    def test_imputer_functionality(self):
        """Test that imputers actually work."""
        imputers = get_imputers(self.X_missing)
        
        for imputer_name, imputer in imputers.items():
            if imputer_name != 'None':
                # Test that imputer can fit and transform
                X_imputed = imputer.fit_transform(self.X_missing)
                
                # Check that output has same shape
                self.assertEqual(X_imputed.shape, self.X_missing.shape)
                
                # Check that output is numpy array
                self.assertIsInstance(X_imputed, np.ndarray)
                
                # Check that no NaN values remain
                self.assertFalse(np.isnan(X_imputed).any())

if __name__ == '__main__':
    unittest.main()
