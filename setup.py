#!/usr/bin/env python3
"""Setup script for AutoMLPipeline package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="automl-pipeline",
    version="1.0.0",
    author="AutoML Pipeline Team",
    author_email="support@automlpipeline.com",
    description="Enterprise-Grade Automated Machine Learning Pipeline with AI-Powered Insights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/automl-pipeline/automl-pipeline",
    project_urls={
        "Bug Tracker": "https://github.com/automl-pipeline/automl-pipeline/issues",
        "Documentation": "https://automl-pipeline.readthedocs.io/",
        "Source Code": "https://github.com/automl-pipeline/automl-pipeline",
        "Changelog": "https://github.com/automl-pipeline/automl-pipeline/blob/main/CHANGELOG.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.20.0",
            "tox>=4.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "myst-parser>=0.18.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-mock>=3.8.0",
        ],
        "full": [
            "google-generativeai>=0.3.0",
            "plotly>=5.0.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "automl-pipeline=automl_pipeline.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "automl_pipeline": [
            "examples/data/*.csv",
            "templates/*.html",
            "templates/*.css",
        ],
    },
    keywords=[
        "machine learning",
        "automl",
        "automated machine learning",
        "data science",
        "artificial intelligence",
        "pipeline",
        "classification",
        "regression",
        "model selection",
        "hyperparameter tuning",
    ],
    zip_safe=False,
)