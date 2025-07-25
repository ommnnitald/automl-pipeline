"""
Simple Iris Classification Example
"""

import pandas as pd
from sklearn.datasets import load_iris
from automl_pipeline import AutoMLPipeline

def run_iris_classification():
    """Run iris classification example"""
    print("🌸 Running Iris Classification Example...")
    
    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    # Create and run pipeline
    pipeline = AutoMLPipeline()
    results = pipeline.fit(df, target_column='species')
    
    print(f"✅ Best model: {results.best_model_name}")
    print(f"✅ Accuracy: {results.best_score:.4f}")
    
    return results

if __name__ == "__main__":
    run_iris_classification()