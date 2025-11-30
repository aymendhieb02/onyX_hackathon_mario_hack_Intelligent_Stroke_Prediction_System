"""
Model Wrapper Classes for Stroke Prediction
These classes are required to load the pickled models
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def complete_preprocessing(df, expected_features=None):
    """
    Complete preprocessing pipeline for stroke prediction models.
    The model expects: age_30_45, age_45_60, age_55_plus, age_60_75, Residence_type_encoded, etc.
    """
    df = df.copy()
    
    # Step 1: Handle missing values
    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].fillna(df['bmi'].median() if len(df) > 0 else 25.0)
    if 'avg_glucose_level' in df.columns:
        df['avg_glucose_level'] = df['avg_glucose_level'].fillna(df['avg_glucose_level'].median() if len(df) > 0 else 100.0)
    
    # Step 2: Create age features (all possible age features)
    if 'age' in df.columns:
        df['age'] = df['age'].astype(float)
        # Create all possible age features
        df['age_30_45'] = ((df['age'] >= 30) & (df['age'] < 45)).astype(int)
        df['age_45_60'] = ((df['age'] >= 45) & (df['age'] < 60)).astype(int)
        df['age_60_75'] = ((df['age'] >= 60) & (df['age'] < 75)).astype(int)
        df['age_55_plus'] = (df['age'] >= 55).astype(int)
        df['age_75_plus'] = (df['age'] >= 75).astype(int)
        df['age_65_plus'] = (df['age'] >= 65).astype(int)
        df['age_80_plus'] = (df['age'] >= 80).astype(int)
        
        # Additional age features that might be in scaler
        df['age_squared'] = df['age'] ** 2
        df['age_cubed'] = df['age'] ** 3
        df['age_log'] = np.log(df['age'] + 1)
        df['age_bin_young'] = (df['age'] < 45).astype(int)
        df['age_bin_middle'] = ((df['age'] >= 45) & (df['age'] < 65)).astype(int)
        df['age_bin_elderly'] = (df['age'] >= 65).astype(int)
        
        # Age interactions
        if 'hypertension' in df.columns:
            df['age_hypertension_interaction'] = df['age'] * df['hypertension']
        if 'heart_disease' in df.columns:
            df['age_heart_disease_interaction'] = df['age'] * df['heart_disease']
        if 'avg_glucose_level' in df.columns:
            df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level'] / 100
        
        # Age dominated risk
        df['age_dominated_risk'] = (
            (df['age'] >= 65) * 3 +
            (df['age'] >= 55) * 2 +
            (df['age'] >= 45) * 1 +
            (df['hypertension'] if 'hypertension' in df.columns else 0) * 1 +
            (df['heart_disease'] if 'heart_disease' in df.columns else 0) * 2
        )
    
    # Step 3: Encode ALL categorical variables with _encoded suffix
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categorical_cols:
        if col in df.columns:
            # Use Categorical codes for encoding (creates _encoded columns)
            df[col + '_encoded'] = pd.Categorical(df[col]).codes
    
    # Step 4: Remove original categorical columns (keep only _encoded versions)
    for col in categorical_cols:
        if col in df.columns:
            df = df.drop(columns=[col], errors='ignore')
    
    # Step 5: If we have expected_features, use them to filter
    if expected_features is not None:
        # Add missing expected features with 0
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        
        # Select ONLY the expected features in the correct order
        df = df[expected_features]
    else:
        # If we don't know expected features, keep all created features
        # Remove any remaining non-numeric columns that aren't encoded
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols]
    
    return df


class StrokeBinaryPredictor(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for binary predictions (0 or 1)
    """
    def __init__(self, model, scaler, preprocessing_func, expected_features=None):
        self.model = model
        self.scaler = scaler
        self.preprocessing_func = preprocessing_func
        self.expected_features = expected_features
        # Store model's expected feature count
        self.model_n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    
    def predict(self, X):
        """
        Returns binary predictions: 0 (no stroke) or 1 (stroke)
        """
        # Apply complete preprocessing
        X_processed = self.preprocessing_func(X)
        
        # Ensure DataFrame
        if not isinstance(X_processed, pd.DataFrame):
            X_processed = pd.DataFrame(X_processed)
        
        # If we have expected_features, ensure we use only those
        if self.expected_features is not None:
            # Add missing features
            for col in self.expected_features:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            # Select only expected features
            X_processed = X_processed[self.expected_features]
        
        # Scale features - but scaler might expect more features
        # So we need to match scaler's expected features first
        if hasattr(self.scaler, 'feature_names_in_') and self.scaler.feature_names_in_ is not None:
            scaler_features = list(self.scaler.feature_names_in_)
            # Add missing features for scaler
            for col in scaler_features:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            # Reorder to match scaler
            X_processed = X_processed[scaler_features]
        
        X_scaled = self.scaler.transform(X_processed)
        
        # Now select only the features the model expects
        if self.model_n_features is not None:
            if X_scaled.shape[1] > self.model_n_features:
                # Model expects fewer features - use expected_features if available
                if self.expected_features is not None and len(self.expected_features) == self.model_n_features:
                    # Convert to DataFrame to select columns
                    X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_features)
                    X_scaled = X_scaled_df[self.expected_features].values
                else:
                    # Take first N features
                    X_scaled = X_scaled[:, :self.model_n_features]
            elif X_scaled.shape[1] < self.model_n_features:
                # Model expects more features - pad with zeros
                padding = np.zeros((X_scaled.shape[0], self.model_n_features - X_scaled.shape[1]))
                X_scaled = np.hstack([X_scaled, padding])
        
        # Get binary predictions
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def fit(self, X, y):
        """For compatibility with sklearn"""
        return self


class StrokeProbabilityPredictor(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for probability predictions (0.0 to 1.0)
    """
    def __init__(self, model, scaler, preprocessing_func, expected_features=None):
        self.model = model
        self.scaler = scaler
        self.preprocessing_func = preprocessing_func
        self.expected_features = expected_features
        # Store model's expected feature count
        self.model_n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    
    def predict(self, X):
        """
        Returns probability/risk percentage: 0.0 to 1.0
        """
        # Apply complete preprocessing
        X_processed = self.preprocessing_func(X)
        
        # Ensure DataFrame
        if not isinstance(X_processed, pd.DataFrame):
            X_processed = pd.DataFrame(X_processed)
        
        # If we have expected_features, ensure we use only those
        if self.expected_features is not None:
            # Add missing features
            for col in self.expected_features:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            # Select only expected features
            X_processed = X_processed[self.expected_features]
        
        # Scale features - but scaler might expect more features
        if hasattr(self.scaler, 'feature_names_in_') and self.scaler.feature_names_in_ is not None:
            scaler_features = list(self.scaler.feature_names_in_)
            # Add missing features for scaler
            for col in scaler_features:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            # Reorder to match scaler
            X_processed = X_processed[scaler_features]
        
        X_scaled = self.scaler.transform(X_processed)
        
        # Now select only the features the model expects
        if self.model_n_features is not None:
            if X_scaled.shape[1] > self.model_n_features:
                if self.expected_features is not None and len(self.expected_features) == self.model_n_features:
                    X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_features)
                    X_scaled = X_scaled_df[self.expected_features].values
                else:
                    X_scaled = X_scaled[:, :self.model_n_features]
            elif X_scaled.shape[1] < self.model_n_features:
                padding = np.zeros((X_scaled.shape[0], self.model_n_features - X_scaled.shape[1]))
                X_scaled = np.hstack([X_scaled, padding])
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return probabilities
    
    def predict_proba(self, X):
        """
        Returns full probability array
        """
        X_processed = self.preprocessing_func(X)
        if not isinstance(X_processed, pd.DataFrame):
            X_processed = pd.DataFrame(X_processed)
        
        # Same preprocessing as predict method
        if self.expected_features is not None:
            for col in self.expected_features:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            X_processed = X_processed[self.expected_features]
        
        if hasattr(self.scaler, 'feature_names_in_') and self.scaler.feature_names_in_ is not None:
            scaler_features = list(self.scaler.feature_names_in_)
            for col in scaler_features:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            X_processed = X_processed[scaler_features]
        
        X_scaled = self.scaler.transform(X_processed)
        
        if self.model_n_features is not None:
            if X_scaled.shape[1] > self.model_n_features:
                if self.expected_features is not None and len(self.expected_features) == self.model_n_features:
                    X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_features)
                    X_scaled = X_scaled_df[self.expected_features].values
                else:
                    X_scaled = X_scaled[:, :self.model_n_features]
            elif X_scaled.shape[1] < self.model_n_features:
                padding = np.zeros((X_scaled.shape[0], self.model_n_features - X_scaled.shape[1]))
                X_scaled = np.hstack([X_scaled, padding])
        
        return self.model.predict_proba(X_scaled)
    
    def fit(self, X, y):
        """For compatibility with sklearn"""
        return self

