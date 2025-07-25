"""
Main AutoMLPipeline class for end users
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_squared_error
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AutoMLPipeline:
    """
    Automated Machine Learning Pipeline
    
    A simple, user-friendly interface for automated machine learning
    that handles both classification and regression tasks.
    """
    
    def __init__(self, enable_ai_insights=False, config=None):
        """
        Initialize the AutoML Pipeline
        
        Args:
            enable_ai_insights (bool): Enable AI-powered insights (requires GEMINI_API_KEY)
            config (PipelineConfig): Configuration object
        """
        self.enable_ai_insights = enable_ai_insights
        self.config = config or self._default_config()
        self.results = None
        
    def _default_config(self):
        """Default configuration"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'max_models': 6,
            'output_dir': 'automl_results'
        }
    
    def fit(self, data, target_column):
        """
        Fit the AutoML pipeline to data
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Name of target column
            
        Returns:
            AutoMLResults: Results object with best model and metrics
        """
        print("🚀 Starting AutoML Pipeline...")
        
        # Prepare data
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Detect problem type
        problem_type = self._detect_problem_type(y)
        print(f"🎯 Problem type detected: {problem_type}")
        
        # Preprocess data
        X_processed, preprocessors = self._preprocess_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Train models
        models = self._train_models(X_train, X_test, y_train, y_test, problem_type)
        
        # Select best model
        best_model_name, best_model_info = self._select_best_model(models, problem_type)
        
        # Create results
        self.results = AutoMLResults(
            best_model_name=best_model_name,
            best_model=best_model_info['model'],
            best_score=best_model_info['score'],
            problem_type=problem_type,
            preprocessors=preprocessors,
            feature_names=X.columns.tolist(),
            target_column=target_column,
            all_models=models
        )
        
        print(f"✅ Pipeline complete! Best model: {best_model_name} (Score: {best_model_info['score']:.4f})")
        
        return self.results
    
    def _detect_problem_type(self, y):
        """Detect if problem is classification or regression"""
        if y.dtype == 'object' or len(y.unique()) < 20:
            return 'classification'
        else:
            return 'regression'
    
    def _preprocess_data(self, X):
        """Preprocess the data"""
        print("⚙️ Preprocessing data...")

        preprocessors = {}
        X_processed = X.copy()

        # Handle missing values first
        print(f"   Missing values found: {X_processed.isnull().sum().sum()}")
        if X_processed.isnull().sum().sum() > 0:
            # For numerical columns: fill with median
            numerical_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                if X_processed[col].isnull().sum() > 0:
                    median_val = X_processed[col].median()
                    X_processed[col] = X_processed[col].fillna(median_val)
                    preprocessors[f'median_{col}'] = median_val
                    print(f"   Filled {col} missing values with median: {median_val}")

            # For categorical columns: fill with mode
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if X_processed[col].isnull().sum() > 0:
                    mode_val = X_processed[col].mode()[0] if len(X_processed[col].mode()) > 0 else 'unknown'
                    X_processed[col] = X_processed[col].fillna(mode_val)
                    preprocessors[f'mode_{col}'] = mode_val
                    print(f"   Filled {col} missing values with mode: {mode_val}")

        # Verify no missing values remain
        remaining_missing = X_processed.isnull().sum().sum()
        print(f"   Missing values after preprocessing: {remaining_missing}")

        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            preprocessors[f'label_encoder_{col}'] = le

        # Scale numerical features
        scaler = StandardScaler()
        X_processed = pd.DataFrame(
            scaler.fit_transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        preprocessors['scaler'] = scaler

        # Final check for any remaining NaN values
        final_missing = X_processed.isnull().sum().sum()
        if final_missing > 0:
            print(f"   WARNING: {final_missing} NaN values still remain!")
            # Force fill any remaining NaN with 0
            X_processed = X_processed.fillna(0)
            print(f"   Filled remaining NaN values with 0")

        print(f"   ✅ Preprocessing complete. Final shape: {X_processed.shape}")
        return X_processed, preprocessors
    
    def _train_models(self, X_train, X_test, y_train, y_test, problem_type):
        """Train multiple models"""
        print("🤖 Training models...")
        
        if problem_type == 'classification':
            models_to_try = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier()
            }
            score_func = accuracy_score
        else:
            models_to_try = {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'SVR': SVR(),
                'K-Nearest Neighbors': KNeighborsRegressor()
            }
            score_func = r2_score
        
        results = {}
        for name, model in models_to_try.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = score_func(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'score': score,
                'predictions': y_pred
            }
            print(f"     ✅ {name}: {score:.4f}")
        
        return results
    
    def _select_best_model(self, models, problem_type):
        """Select the best performing model"""
        best_name = max(models.keys(), key=lambda k: models[k]['score'])
        return best_name, models[best_name]


class AutoMLResults:
    """Results from AutoML pipeline"""
    
    def __init__(self, best_model_name, best_model, best_score, problem_type, 
                 preprocessors, feature_names, target_column, all_models):
        self.best_model_name = best_model_name
        self.best_model = best_model
        self.best_score = best_score
        self.problem_type = problem_type
        self.preprocessors = preprocessors
        self.feature_names = feature_names
        self.target_column = target_column
        self.all_models = all_models
    
    def predict(self, new_data):
        """Make predictions on new data"""
        # Preprocess new data
        X_processed = new_data.copy()
        
        # Apply same preprocessing
        for col in X_processed.select_dtypes(include=['object']).columns:
            if f'label_encoder_{col}' in self.preprocessors:
                le = self.preprocessors[f'label_encoder_{col}']
                X_processed[col] = le.transform(X_processed[col].astype(str))
        
        # Scale features
        X_processed = self.preprocessors['scaler'].transform(X_processed)
        
        # Make predictions
        return self.best_model.predict(X_processed)
    
    def save_model(self, filepath):
        """Save the model and preprocessors"""
        model_data = {
            'model': self.best_model,
            'preprocessors': self.preprocessors,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'problem_type': self.problem_type,
            'model_name': self.best_model_name,
            'score': self.best_score
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")