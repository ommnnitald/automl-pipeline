"""
Configuration classes for AutoMLPipeline
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class PipelineConfig:
    """Configuration class for AutoMLPipeline."""

    # Model selection
    models_to_try: Optional[List[str]] = None
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42

    # Feature engineering
    handle_missing_values: bool = True
    scale_features: bool = True
    encode_categorical: bool = True

    # Model training
    max_training_time: Optional[int] = None  # seconds
    early_stopping: bool = True

    # Output
    save_models: bool = True
    generate_report: bool = True
    verbose: bool = True

    # AI integration
    use_ai_insights: bool = True
    ai_api_key: Optional[str] = None

    def __post_init__(self):
        """Set default models if none specified."""
        if self.models_to_try is None:
            self.models_to_try = [
                'RandomForest',
                'LogisticRegression',
                'SVM',
                'GradientBoosting',
                'XGBoost'
            ]