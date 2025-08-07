import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer

def get_scalers():
    """Return dictionary of available scalers."""
    return {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'None': None
    }

def get_encoders(X):
    """Return dictionary of available encoders based on data types."""
    encoders = {}
    
    # Check for categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        encoders.update({
            'OneHotEncoder': OneHotEncoder(handle_unknown='ignore', sparse=False),
            'OrdinalEncoder': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        })
    else:
        # If no categorical columns, return empty dict
        encoders = {}
    
    # Always include None option
    encoders['None'] = None
    
    return encoders

def get_imputers(X):
    """Return dictionary of available imputers based on data characteristics."""
    imputers = {
        'None': None
    }
    
    # Check for missing values
    has_missing = X.isnull().any().any()
    
    if has_missing:
        imputers.update({
            'SimpleImputer_Mean': SimpleImputer(strategy='mean'),
            'SimpleImputer_Median': SimpleImputer(strategy='median'),
            'SimpleImputer_MostFrequent': SimpleImputer(strategy='most_frequent'),
            'KNNImputer': KNNImputer(n_neighbors=5)
        })
    
    return imputers

def is_classification(X):
    """Check if the dataset is suitable for classification (has categorical target)."""
    # This is a simple heuristic - in practice, the user should specify
    return X.select_dtypes(include=['object', 'category']).shape[1] > 0

def get_preprocessing_summary(X):
    """Generate a summary of the dataset for preprocessing recommendations."""
    summary = {
        'shape': X.shape,
        'dtypes': X.dtypes.value_counts().to_dict(),
        'missing_values': X.isnull().sum().sum(),
        'categorical_columns': X.select_dtypes(include=['object', 'category']).columns.tolist(),
        'numerical_columns': X.select_dtypes(include=[np.number]).columns.tolist(),
        'memory_usage': X.memory_usage(deep=True).sum()
    }
    
    return summary
