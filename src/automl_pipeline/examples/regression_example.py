"""
Simple Housing Regression Example
"""

import pandas as pd
import numpy as np
from automl_pipeline import AutoMLPipeline

def run_housing_regression():
    """Run housing regression example"""
    print("🏠 Running Housing Regression Example...")
    
    # Create sample housing data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'sqft': np.random.randint(800, 4000, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'location': np.random.choice(['urban', 'suburban', 'rural'], n_samples)
    }
    
    # Create realistic price based on features
    df = pd.DataFrame(data)
    df['price'] = (
        df['bedrooms'] * 50000 +
        df['bathrooms'] * 30000 +
        df['sqft'] * 100 +
        (50 - df['age']) * 1000 +
        df['location'].map({'urban': 100000, 'suburban': 50000, 'rural': 0}) +
        np.random.normal(0, 20000, n_samples)
    )
    
    # Create and run pipeline
    pipeline = AutoMLPipeline()
    results = pipeline.fit(df, target_column='price')
    
    print(f"✅ Best model: {results.best_model_name}")
    print(f"✅ R² Score: {results.best_score:.4f}")
    
    return results

if __name__ == "__main__":
    run_housing_regression()