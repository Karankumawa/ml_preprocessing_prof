# Installation Guide for ML Preprocessing Profiler

## Why the error occurs

The error `ERROR: Could not find a version that satisfies the requirement ml-preprocessing-profiler` occurs because the library hasn't been published to PyPI (Python Package Index) yet. Here are the different ways to install and use the library:

## Installation Methods

### Method 1: Install from GitHub (Recommended)

```bash
pip install git+https://github.com/Karankumawa/ml_preprocessing_prof.git
```

### Method 2: Clone and Install Locally

```bash
# Clone the repository
git clone https://github.com/Karankumawa/ml_preprocessing_prof.git

# Navigate to the directory
cd ml_preprocessing_profiler

# Install in development mode
pip install -e .
```

### Method 3: Download and Install

```bash
# Download the ZIP file from GitHub
# Extract it to a folder
# Navigate to the extracted folder
cd ml_preprocessing_profiler

# Install in development mode
pip install -e .
```

### Method 4: Manual Installation

If you have the files locally:

```bash
# Navigate to the project directory
cd path/to/ml_preprocessing_profiler

# Install dependencies first
pip install scikit-learn pandas matplotlib seaborn numpy

# Install the library
pip install -e .
```

## Verification

After installation, verify it works:

```python
# Test import
from ml_preprocessing_profiler import evaluate_preprocessors
print("✅ Installation successful!")

# Test functionality
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
results = evaluate_preprocessors(X, y, model)
print(f"✅ Functionality test passed! Generated {len(results)} combinations")
```

## Troubleshooting

### If you get network errors:
1. Check your internet connection
2. Try using a VPN if GitHub is blocked
3. Use Method 2 or 3 instead

### If you get permission errors:
```bash
# On Windows, run as administrator
# On Linux/Mac, use sudo
sudo pip install -e .
```

### If you get dependency errors:
```bash
# Install dependencies manually
pip install scikit-learn>=1.0.0
pip install pandas>=1.3.0
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0
pip install numpy>=1.20.0
```

### If you're using a virtual environment:
```bash
# Create and activate virtual environment first
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Usage Examples

Once installed, you can use the library:

```python
# Method 1: Direct import (Recommended)
from ml_preprocessing_profiler import evaluate_preprocessors

# Method 2: From specific module
from ml_preprocessing_profiler.core import evaluate_preprocessors

# Method 3: Import entire package
import ml_preprocessing_profiler
results = ml_preprocessing_profiler.evaluate_preprocessors(X, y, model)
```

## Future: PyPI Publication

The library will be published to PyPI soon, after which you'll be able to install it with:

```bash
pip install ml-preprocessing-profiler
```

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Create an issue on GitHub: https://github.com/Karankumawa/ml_preprocessing_prof.git/issues
3. Contact: karankumawat303@gmail.com
