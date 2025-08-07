"""
ML Preprocessing Profiler

A Python library to compare different preprocessing techniques and their effects on machine learning model performance.
"""

__version__ = "0.1.0"
__author__ = "karan kumawat"
__email__ = "karankumawat303@gmail.com"

# Import main functions for easy access
from .core import evaluate_preprocessors
from .utils import (
    validate_input_data,
    detect_problem_type,
    get_data_summary,
    print_data_summary,
    check_model_compatibility,
    format_time
)
from .preprocessors import (
    get_scalers,
    get_encoders,
    get_imputers,
    get_preprocessing_summary
)

__all__ = [
    'evaluate_preprocessors',
    'validate_input_data',
    'detect_problem_type',
    'get_data_summary',
    'print_data_summary',
    'check_model_compatibility',
    'format_time',
    'get_scalers',
    'get_encoders',
    'get_imputers',
    'get_preprocessing_summary'
]
