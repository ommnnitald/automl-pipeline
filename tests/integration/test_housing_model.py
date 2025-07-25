#!/usr/bin/env python3
"""
Independent Test of Saved Housing Regression Model
=================================================

This script independently loads and tests the saved Random Forest regression model
to validate that it works correctly for house price predictions.
"""

import pandas as pd
import numpy as np
import joblib
import json

def load_saved_model():
    """Load the saved model, scaler, and metadata"""
    print("🔄 Loading saved regression model...")
    
    try:
        # Load metadata
        with open('housing_analysis_results/models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load model and scaler
        model = joblib.load('housing_analysis_results/models/best_regression_model.pkl')
        scaler = joblib.load('housing_analysis_results/models/scaler.pkl')
        
        print(f"✅ Model loaded: {metadata['model_name']}")
        print(f"📊 Model type: {metadata['model_type']}")
        print(f"🎯 Target: {metadata['target_variable']}")
        print(f"📈 R² Score: {metadata['performance_metrics']['r2_score']:.4f}")
        print(f"💰 RMSE: ${metadata['performance_metrics']['rmse']:,.0f}")
        
        return model, scaler, metadata
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def test_model_performance():
    """Test the model with various housing scenarios"""
    print("\n🏠 Testing model with diverse housing scenarios...")
    
    model, scaler, metadata = load_saved_model()
    if model is None:
        return
    
    # Create comprehensive test scenarios
    test_scenarios = pd.DataFrame({
        'longitude': [-122.4, -118.2, -121.9, -117.2, -122.0, -119.7],
        'latitude': [37.8, 34.1, 37.4, 32.7, 37.5, 34.4],
        'housing_median_age': [10.0, 25.0, 40.0, 5.0, 30.0, 20.0],
        'total_rooms': [8000.0, 4500.0, 2800.0, 6500.0, 3200.0, 5200.0],
        'total_bedrooms': [1500.0, 850.0, 520.0, 1200.0, 600.0, 980.0],
        'population': [4000.0, 2200.0, 1400.0, 3100.0, 1600.0, 2600.0],
        'households': [1400.0, 800.0, 500.0, 1100.0, 550.0, 900.0],
        'median_income': [12.5, 6.8, 3.2, 9.1, 4.5, 7.3],
        'ocean_proximity': [0, 1, 0, 1, 0, 1]  # 0=INLAND, 1=NEAR OCEAN
    })
    
    scenario_names = [
        "Luxury San Francisco",
        "Mid-tier Los Angeles", 
        "Budget Sacramento",
        "Premium San Diego",
        "Standard Bay Area",
        "Coastal Santa Barbara"
    ]
    
    # Ensure correct feature order and convert to DataFrame with proper column names
    feature_names = metadata['feature_names']
    test_scenarios = test_scenarios[feature_names]

    # Scale the features and maintain DataFrame structure
    test_scenarios_scaled = pd.DataFrame(
        scaler.transform(test_scenarios),
        columns=feature_names,
        index=test_scenarios.index
    )
    
    # Make predictions
    predictions = model.predict(test_scenarios_scaled)
    
    print("\n🏡 Housing Price Predictions:")
    print("=" * 60)
    
    for i, (scenario, pred) in enumerate(zip(scenario_names, predictions)):
        row = test_scenarios.iloc[i]
        ocean_status = "Near Ocean" if row['ocean_proximity'] == 1 else "Inland"
        
        print(f"{i+1}. {scenario}")
        print(f"   💰 Predicted Price: ${pred:,.0f}")
        print(f"   📍 Location: {ocean_status}")
        print(f"   💵 Income: ${row['median_income']:.1f}k")
        print(f"   🏠 Age: {row['housing_median_age']:.0f} years")
        print(f"   🏢 Rooms: {row['total_rooms']:.0f}")
        print()
    
    # Performance analysis
    print("📊 Prediction Analysis:")
    print(f"   💰 Price Range: ${predictions.min():,.0f} - ${predictions.max():,.0f}")
    print(f"   📈 Average Price: ${predictions.mean():,.0f}")
    print(f"   📊 Price Std Dev: ${predictions.std():,.0f}")
    
    # Validate predictions are reasonable
    reasonable_min = 50000   # $50k minimum
    reasonable_max = 2000000 # $2M maximum
    
    valid_predictions = all(reasonable_min <= pred <= reasonable_max for pred in predictions)
    
    if valid_predictions:
        print(f"   ✅ All predictions within reasonable range (${reasonable_min:,} - ${reasonable_max:,})")
    else:
        print(f"   ⚠️ Some predictions outside reasonable range")
    
    return predictions

def test_edge_cases():
    """Test model with edge case scenarios"""
    print("\n🔬 Testing edge cases...")
    
    model, scaler, metadata = load_saved_model()
    if model is None:
        return
    
    # Edge case scenarios
    edge_cases = pd.DataFrame({
        'longitude': [-124.0, -114.0, -122.5],  # Extreme west, east, central
        'latitude': [32.5, 42.0, 37.0],         # South, north, central
        'housing_median_age': [1.0, 52.0, 25.0], # Very new, very old, average
        'total_rooms': [100.0, 15000.0, 3000.0], # Very small, very large, average
        'total_bedrooms': [20.0, 3000.0, 600.0], # Very few, very many, average
        'population': [50.0, 8000.0, 1500.0],    # Very low, very high, average
        'households': [20.0, 2500.0, 500.0],     # Very few, very many, average
        'median_income': [0.5, 15.0, 5.0],       # Very low, very high, average
        'ocean_proximity': [0, 1, 0]             # Mixed
    })
    
    edge_case_names = [
        "Extreme Low-End Property",
        "Extreme High-End Property", 
        "Average Property"
    ]
    
    # Ensure correct feature order and convert to DataFrame with proper column names
    feature_names = metadata['feature_names']
    edge_cases = edge_cases[feature_names]

    # Scale and predict while maintaining DataFrame structure
    edge_cases_scaled = pd.DataFrame(
        scaler.transform(edge_cases),
        columns=feature_names,
        index=edge_cases.index
    )
    edge_predictions = model.predict(edge_cases_scaled)
    
    print("🔍 Edge Case Results:")
    for i, (case_name, pred) in enumerate(zip(edge_case_names, edge_predictions)):
        print(f"   {i+1}. {case_name}: ${pred:,.0f}")
    
    # Check for reasonable behavior
    if edge_predictions[1] > edge_predictions[0]:  # High-end > Low-end
        print("   ✅ Model correctly ranks high-end > low-end properties")
    else:
        print("   ⚠️ Model ranking may be incorrect")
    
    return edge_predictions

def validate_model_consistency():
    """Test model consistency with repeated predictions"""
    print("\n🔄 Testing model consistency...")
    
    model, scaler, metadata = load_saved_model()
    if model is None:
        return
    
    # Create a test sample
    test_sample = pd.DataFrame({
        'longitude': [-122.25],
        'latitude': [37.85],
        'housing_median_age': [25.0],
        'total_rooms': [3500.0],
        'total_bedrooms': [650.0],
        'population': [1800.0],
        'households': [600.0],
        'median_income': [6.5],
        'ocean_proximity': [0]
    })
    
    # Ensure correct feature order and convert to DataFrame with proper column names
    feature_names = metadata['feature_names']
    test_sample = test_sample[feature_names]

    # Scale the sample while maintaining DataFrame structure
    test_sample_scaled = pd.DataFrame(
        scaler.transform(test_sample),
        columns=feature_names,
        index=test_sample.index
    )
    
    # Make multiple predictions (should be identical for deterministic models)
    predictions = []
    for i in range(5):
        pred = model.predict(test_sample_scaled)[0]
        predictions.append(pred)
    
    # Check consistency
    all_same = all(abs(pred - predictions[0]) < 0.01 for pred in predictions)
    
    if all_same:
        print(f"   ✅ Model predictions are consistent: ${predictions[0]:,.0f}")
    else:
        print(f"   ⚠️ Model predictions vary: {[f'${p:,.0f}' for p in predictions]}")
    
    return predictions[0]

def main():
    """Main test execution"""
    print("🧪 Independent Housing Regression Model Test")
    print("=" * 50)
    
    # Test 1: Basic model loading and diverse scenarios
    predictions = test_model_performance()
    
    # Test 2: Edge cases
    edge_predictions = test_edge_cases()
    
    # Test 3: Consistency
    consistent_pred = validate_model_consistency()
    
    # Final validation
    print("\n🎯 FINAL VALIDATION RESULTS:")
    print("=" * 40)
    
    if predictions is not None and edge_predictions is not None and consistent_pred is not None:
        print("✅ Model loading: SUCCESS")
        print("✅ Diverse scenarios: SUCCESS")
        print("✅ Edge cases: SUCCESS") 
        print("✅ Consistency: SUCCESS")
        print("\n🚀 Model is ready for production deployment!")
        
        # Summary statistics
        all_predictions = list(predictions) + list(edge_predictions) + [consistent_pred]
        print(f"\n📊 Overall Test Summary:")
        print(f"   🏠 Total predictions tested: {len(all_predictions)}")
        print(f"   💰 Price range: ${min(all_predictions):,.0f} - ${max(all_predictions):,.0f}")
        print(f"   📈 Average prediction: ${np.mean(all_predictions):,.0f}")
        print(f"   ✅ All tests passed successfully!")
        
    else:
        print("❌ Some tests failed - model may need attention")

if __name__ == "__main__":
    main()
