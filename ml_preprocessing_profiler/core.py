import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from .preprocessors import get_scalers, get_encoders, get_imputers
from .report import generate_report

def evaluate_preprocessors(X, y, model, problem_type='classification', test_size=0.2, random_state=42):
    """
    Evaluate different preprocessing combinations on a given dataset and model.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    model : sklearn estimator
        The model to evaluate
    problem_type : str, default='classification'
        Type of problem: 'classification' or 'regression'
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict : Results summary with preprocessing combinations and their performance
    """
    # Convert to pandas DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    results = []
    
    # Get preprocessing options
    scalers = get_scalers()
    encoders = get_encoders(X)
    imputers = get_imputers(X)
    
    print(f"Evaluating {len(scalers)} scalers, {len(encoders)} encoders, and {len(imputers)} imputers...")
    print(f"Total combinations to test: {len(scalers) * len(encoders) * len(imputers)}")
    
    # Test baseline (no preprocessing)
    try:
        start_time = time.time()
        model_copy = clone_model(model)
        model_copy.fit(X_train, y_train)
        y_pred = model_copy.predict(X_test)
        
        baseline_score = calculate_score(y_test, y_pred, problem_type)
        baseline_time = time.time() - start_time
        
        results.append({
            'scaler': 'None',
            'encoder': 'None', 
            'imputer': 'None',
            'score': baseline_score,
            'time': baseline_time,
            'combination': 'Baseline (No Preprocessing)'
        })
        print(f"Baseline score: {baseline_score:.4f}")
    except Exception as e:
        print(f"Baseline failed: {e}")
    
    # Test all preprocessing combinations
    for scaler_name, scaler in scalers.items():
        for encoder_name, encoder in encoders.items():
            for imputer_name, imputer in imputers.items():
                try:
                    start_time = time.time()
                    
                    # Apply preprocessing
                    X_train_processed = X_train.copy()
                    X_test_processed = X_test.copy()
                    
                    # Apply imputer
                    if imputer is not None:
                        X_train_processed = imputer.fit_transform(X_train_processed)
                        X_test_processed = imputer.transform(X_test_processed)
                    
                    # Apply encoder
                    if encoder is not None:
                        X_train_processed = encoder.fit_transform(X_train_processed)
                        X_test_processed = encoder.transform(X_test_processed)
                    
                    # Apply scaler
                    if scaler is not None:
                        X_train_processed = scaler.fit_transform(X_train_processed)
                        X_test_processed = scaler.transform(X_test_processed)
                    
                    # Train and evaluate model
                    model_copy = clone_model(model)
                    model_copy.fit(X_train_processed, y_train)
                    y_pred = model_copy.predict(X_test_processed)
                    
                    score = calculate_score(y_test, y_pred, problem_type)
                    processing_time = time.time() - start_time
                    
                    results.append({
                        'scaler': scaler_name,
                        'encoder': encoder_name,
                        'imputer': imputer_name,
                        'score': score,
                        'time': processing_time,
                        'combination': f"{scaler_name} + {encoder_name} + {imputer_name}"
                    })
                    
                except Exception as e:
                    print(f"Skipping {scaler_name}+{encoder_name}+{imputer_name} due to error: {e}")
    
    # Generate and display report
    generate_report(results, problem_type)
    
    return results

def calculate_score(y_true, y_pred, problem_type):
    """Calculate appropriate score based on problem type."""
    if problem_type == 'classification':
        return accuracy_score(y_true, y_pred)
    elif problem_type == 'regression':
        return r2_score(y_true, y_pred)
    else:
        raise ValueError("problem_type must be 'classification' or 'regression'")

def clone_model(model):
    """Create a copy of the model to avoid modifying the original."""
    from sklearn.base import clone
    return clone(model)
