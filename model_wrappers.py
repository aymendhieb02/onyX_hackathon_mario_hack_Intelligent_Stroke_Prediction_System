"""
Model Wrapper Classes for Stroke Prediction
These classes are required to load the pickled models
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


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

