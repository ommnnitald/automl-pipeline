"""
AutoMLPipeline - Automated Machine Learning Pipeline
==================================================

A comprehensive automated machine learning pipeline with AI-powered insights.
"""

from .core.pipeline import AutoMLPipeline
from .core.config import PipelineConfig
from .utils.validation import load_model
from .examples.classification_example import run_iris_classification
from .examples.regression_example import run_housing_regression

__version__ = "1.0.0"
__author__ = "AutoML Pipeline Team"

__all__ = [
    "AutoMLPipeline",
    "PipelineConfig", 
    "load_model",
    "run_iris_classification",
    "run_housing_regression"
]