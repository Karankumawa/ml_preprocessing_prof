import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
import warnings

def validate_input_data(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> tuple:
    """
    Validate and prepare input data for preprocessing evaluation.
    
    Parameters:
    -----------
    X : Union[pd.DataFrame, np.ndarray]
        Feature matrix
    y : Union[pd.Series, np.ndarray]
        Target variable
        
    Returns:
    --------
    tuple : (X, y) as pandas DataFrame and Series
    """
    # Convert X to DataFrame if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    elif not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame or numpy array")
    
    # Convert y to Series if needed
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            y = pd.Series(y, name='target')
        else:
            y = pd.Series(y.flatten(), name='target')
    elif not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series or numpy array")
    
    # Check shapes
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    # Check for empty data
    if X.empty or y.empty:
        raise ValueError("X and y cannot be empty")
    
    return X, y

def detect_problem_type(y: pd.Series) -> str:
    """
    Automatically detect if the problem is classification or regression.
    
    Parameters:
    -----------
    y : pd.Series
        Target variable
        
    Returns:
    --------
    str : 'classification' or 'regression'
    """
    # Check if target is categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    
    # Check if target has integer values with few unique values (likely classification)
    if y.dtype in ['int64', 'int32'] and len(y.unique()) <= 20:
        return 'classification'
    
    # Default to regression
    return 'regression'

def get_data_summary(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Generate comprehensive summary of the dataset.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
        
    Returns:
    --------
    Dict[str, Any] : Summary statistics
    """
    summary = {
        'dataset_shape': X.shape,
        'target_info': {
            'dtype': str(y.dtype),
            'unique_values': len(y.unique()),
            'missing_values': y.isnull().sum(),
            'problem_type': detect_problem_type(y)
        },
        'features_info': {
            'numerical_features': len(X.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(X.select_dtypes(include=['object', 'category']).columns),
            'missing_values': X.isnull().sum().sum(),
            'memory_usage_mb': X.memory_usage(deep=True).sum() / 1024 / 1024
        }
    }
    
    return summary

def print_data_summary(X: pd.DataFrame, y: pd.Series):
    """
    Print a formatted summary of the dataset.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    """
    summary = get_data_summary(X, y)
    
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Dataset shape: {summary['dataset_shape']}")
    print(f"Problem type: {summary['target_info']['problem_type']}")
    print(f"Target dtype: {summary['target_info']['dtype']}")
    print(f"Target unique values: {summary['target_info']['unique_values']}")
    print(f"Numerical features: {summary['features_info']['numerical_features']}")
    print(f"Categorical features: {summary['features_info']['categorical_features']}")
    print(f"Missing values: {summary['features_info']['missing_values']}")
    print(f"Memory usage: {summary['features_info']['memory_usage_mb']:.2f} MB")
    print("=" * 60)

def check_model_compatibility(model: Any) -> bool:
    """
    Check if the model is compatible with the evaluation framework.
    
    Parameters:
    -----------
    model : Any
        The model to check
        
    Returns:
    --------
    bool : True if compatible, False otherwise
    """
    required_methods = ['fit', 'predict']
    
    for method in required_methods:
        if not hasattr(model, method):
            warnings.warn(f"Model missing required method: {method}")
            return False
    
    return True

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
        
    Returns:
    --------
    str : Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def create_preprocessing_pipeline(scaler_name: str, encoder_name: str, imputer_name: str) -> Dict[str, Any]:
    """
    Create a preprocessing pipeline configuration.
    
    Parameters:
    -----------
    scaler_name : str
        Name of the scaler
    encoder_name : str
        Name of the encoder
    imputer_name : str
        Name of the imputer
        
    Returns:
    --------
    Dict[str, Any] : Pipeline configuration
    """
    from .preprocessors import get_scalers, get_encoders, get_imputers
    
    # Create dummy DataFrame to get encoders and imputers
    dummy_X = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    
    scalers = get_scalers()
    encoders = get_encoders(dummy_X)
    imputers = get_imputers(dummy_X)
    
    pipeline = {
        'scaler': scalers.get(scaler_name),
        'encoder': encoders.get(encoder_name),
        'imputer': imputers.get(imputer_name),
        'scaler_name': scaler_name,
        'encoder_name': encoder_name,
        'imputer_name': imputer_name
    }
    
    return pipeline
