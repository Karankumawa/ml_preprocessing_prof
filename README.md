# ML Preprocessing Profiler

A comprehensive Python library for comparing different preprocessing techniques and their effects on machine learning model performance. Automatically test multiple combinations of scalers, encoders, and imputers to find the optimal preprocessing pipeline for your dataset.

## 🚀 Features

- **Automatic Preprocessing Comparison**: Test multiple combinations of scalers, encoders, and imputers
- **Comprehensive Reporting**: Generate detailed reports with visualizations and performance metrics
- **Flexible Input**: Works with any scikit-learn compatible model and dataset
- **Smart Detection**: Automatically detects data types and applies appropriate preprocessing
- **Missing Data Handling**: Built-in support for datasets with missing values
- **Classification & Regression**: Supports both problem types with appropriate metrics
- **Easy Integration**: Simple one-line function call for complete analysis

## 📦 Installation

### Option 1: Install from GitHub (Recommended)

```bash
pip install git+https://github.com/Karankumawa/ml_preprocessing_prof.git
```

### Option 2: Clone and Install Locally

```bash
git clone https://github.com/Karankumawa/ml_preprocessing_prof.git
cd ml_preprocessing_profiler
pip install -e .
```

### Option 3: Use Installation Script

```bash
git clone https://github.com/Karankumawa/ml_preprocessing_prof.git
cd ml_preprocessing_profiler
python install.py
```

### Option 4: Manual Installation

```bash
# Install dependencies first
pip install scikit-learn pandas matplotlib seaborn numpy

# Clone and install
git clone https://github.com/Karankumawa/ml_preprocessing_prof.git
cd ml_preprocessing_profiler
pip install -e .
```

**Note:** The library is not yet published to PyPI, so `pip install ml-preprocessing-profiler` won't work yet. Use one of the methods above instead.

For detailed installation instructions and troubleshooting, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md).

## 🎯 Quick Start

```python
from ml_preprocessing_profiler.core import evaluate_preprocessors
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load your data
X, y = load_iris(return_X_y=True)

# Create your model
model = RandomForestClassifier()

# Run the evaluation (one line!)
results = evaluate_preprocessors(X, y, model)
```

That's it! The library will automatically:
- Detect your data types (numerical/categorical)
- Apply appropriate preprocessing techniques
- Train and evaluate your model with each combination
- Generate comprehensive reports and visualizations

## 📊 What You Get

### 1. Performance Comparison Table
```
Top 10 Preprocessing Combinations (Classification):
--------------------------------------------------------------------------------
Preprocessing Pipeline                    Accuracy  Time (s)
StandardScaler + OneHotEncoder + None    0.9667    0.045
MinMaxScaler + OneHotEncoder + None      0.9667    0.043
RobustScaler + OneHotEncoder + None      0.9667    0.044
StandardScaler + None + None             0.9333    0.032
...
```

### 2. Visual Reports
- Score distribution across all combinations
- Top performing preprocessing pipelines
- Performance comparison by scaler type
- Score vs. processing time analysis
- Processing time distribution

### 3. Summary Statistics
- Best and worst performing combinations
- Mean and standard deviation of scores
- Processing time analysis

## 🔧 Advanced Usage

### Custom Problem Type
```python
# For regression problems
results = evaluate_preprocessors(X, y, model, problem_type='regression')
```

### Custom Test Split
```python
# Use 30% for testing
results = evaluate_preprocessors(X, y, model, test_size=0.3)
```

### Reproducible Results
```python
# Set random seed for reproducibility
results = evaluate_preprocessors(X, y, model, random_state=42)
```

### With Missing Data
```python
import numpy as np
import pandas as pd

# Create dataset with missing values
X = pd.DataFrame({
    'feature_1': [1, 2, np.nan, 4, 5],
    'feature_2': ['A', 'B', 'A', np.nan, 'B'],
    'feature_3': [0.1, 0.2, 0.3, 0.4, np.nan]
})
y = pd.Series([0, 1, 0, 1, 0])

# The library will automatically handle missing values
results = evaluate_preprocessors(X, y, model)
```

## 🛠️ Supported Preprocessing Techniques

### Scalers
- **StandardScaler**: Standardize features by removing the mean and scaling to unit variance
- **MinMaxScaler**: Scale features to a given range (default: [0, 1])
- **RobustScaler**: Scale features using statistics that are robust to outliers
- **None**: No scaling applied

### Encoders
- **OneHotEncoder**: Encode categorical features as a one-hot numeric array
- **OrdinalEncoder**: Encode categorical features as an integer array
- **None**: No encoding applied (for numerical data only)

### Imputers
- **SimpleImputer_Mean**: Replace missing values using the mean along each column
- **SimpleImputer_Median**: Replace missing values using the median along each column
- **SimpleImputer_MostFrequent**: Replace missing values using the most frequent value
- **KNNImputer**: Impute missing values using k-Nearest Neighbors
- **None**: No imputation applied (for data without missing values)

## 📈 Supported Models

The library works with any scikit-learn compatible model that has:
- `fit(X, y)` method
- `predict(X)` method

Examples:
- **Classification**: RandomForestClassifier, LogisticRegression, SVC, etc.
- **Regression**: RandomForestRegressor, LinearRegression, SVR, etc.

## 🧪 Examples

Check out the `examples/usage_demo.py` file for comprehensive examples including:

- Classification problems
- Regression problems
- Datasets with missing values
- Different model types
- Custom configurations

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=ml_preprocessing_profiler
```

## 📚 API Reference

### Main Function

#### `evaluate_preprocessors(X, y, model, problem_type='classification', test_size=0.2, random_state=42)`

Evaluate different preprocessing combinations on a given dataset and model.

**Parameters:**
- `X`: Feature matrix (pandas DataFrame or numpy array)
- `y`: Target variable (pandas Series or numpy array)
- `model`: scikit-learn compatible model
- `problem_type`: 'classification' or 'regression' (default: 'classification')
- `test_size`: Proportion of dataset for testing (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)

**Returns:**
- List of dictionaries containing results for each preprocessing combination

### Utility Functions

#### `print_data_summary(X, y)`
Print a formatted summary of the dataset.

#### `get_data_summary(X, y)`
Generate comprehensive summary statistics of the dataset.

#### `detect_problem_type(y)`
Automatically detect if the problem is classification or regression.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install in development mode: `pip install -e ".[dev]"`
5. Make your changes
6. Run tests: `pytest`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the excellent [scikit-learn](https://scikit-learn.org/) library
- Inspired by the need for systematic preprocessing comparison in ML workflows
- Thanks to the open-source community for the amazing tools that made this possible

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the [documentation](https://ml-preprocessing-profiler.readthedocs.io/)
2. Search existing [issues](https://github.com/Karankumawa/ml_preprocessing_prof.git/issues)
3. Create a new issue with a minimal reproducible example

## 🔄 Changelog

### Version 0.1.0
- Initial release
- Support for classification and regression problems
- Automatic preprocessing combination testing
- Comprehensive reporting and visualization
- Missing data handling
- Support for all major scikit-learn models
