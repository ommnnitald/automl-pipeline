"""
Command Line Interface for AutoMLPipeline
"""

import argparse
import pandas as pd
from automl_pipeline import AutoMLPipeline

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='AutoMLPipeline CLI')
    parser.add_argument('data_file', help='Path to CSV data file')
    parser.add_argument('target_column', help='Name of target column')
    parser.add_argument('--output', '-o', default='automl_results', help='Output directory')
    parser.add_argument('--ai', action='store_true', help='Enable AI insights')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📊 Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    
    # Create pipeline
    pipeline = AutoMLPipeline(enable_ai_insights=args.ai)
    
    # Run analysis
    results = pipeline.fit(df, target_column=args.target_column)
    
    # Save results
    results.save_model(f"{args.output}/best_model.pkl")
    print(f"✅ Results saved to {args.output}/")

if __name__ == "__main__":
    main()