#!/usr/bin/env python3
"""
Installation Script for ML Preprocessing Profiler
This script helps users install the library and its dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def install_dependencies():
    """Install required dependencies."""
    dependencies = [
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "numpy>=1.20.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    return True

def install_library():
    """Install the library in development mode."""
    return run_command("pip install -e .", "Installing ML Preprocessing Profiler")

def test_installation():
    """Test if the installation was successful."""
    print("🧪 Testing installation...")
    try:
        # Test import
        import ml_preprocessing_profiler
        print("✅ Library import successful!")
        
        # Test main function
        from ml_preprocessing_profiler import evaluate_preprocessors
        print("✅ Main function import successful!")
        
        # Test with sample data
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(random_state=42)
        results = evaluate_preprocessors(X, y, model, test_size=0.2, random_state=42)
        
        print(f"✅ Functionality test passed! Generated {len(results)} combinations")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main installation process."""
    print("🚀 ML Preprocessing Profiler - Installation Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists("setup.py"):
        print("❌ Please run this script from the ml_preprocessing_profiler directory!")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    print("✅ Running from correct directory!")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies!")
        sys.exit(1)
    
    # Install library
    if not install_library():
        print("❌ Failed to install library!")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Installation completed successfully!")
    print("\n💡 You can now use the library:")
    print("""
from ml_preprocessing_profiler import evaluate_preprocessors
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
results = evaluate_preprocessors(X, y, model)
    """)
    
    print("\n📚 For more information, see:")
    print("- README.md")
    print("- INSTALLATION_GUIDE.md")
    print("- examples/usage_demo.py")

if __name__ == "__main__":
    main()
