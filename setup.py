from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml_preprocessing_profiler",
    version="0.1.0",
                    author="karan kumawat",
                author_email="karankumawat303@gmail.com",
                description="A tool to compare preprocessing effects on ML performance",
                long_description=long_description,
                long_description_content_type="text/markdown",
                url="https://github.com/Karankumawa/ml_preprocessing_prof.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine learning, preprocessing, scikit-learn, data science, ml",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ml_preprocessing_profiler/issues",
        "Source": "https://github.com/yourusername/ml_preprocessing_profiler",
        "Documentation": "https://ml-preprocessing-profiler.readthedocs.io/",
    },
)
