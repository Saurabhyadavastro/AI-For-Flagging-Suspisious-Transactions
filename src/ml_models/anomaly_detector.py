"""Anomaly detection models for transaction analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime


class AnomalyDetector:
    """Main anomaly detection class using multiple algorithms."""
    
    def __init__(self, model_path: str = "data/models"):
        """Initialize the anomaly detector."""
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        features_df = df.copy()
        
        # Time-based features
        if 'timestamp' in features_df.columns:
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            features_df['hour'] = features_df['timestamp'].dt.hour
            features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        if 'amount' in features_df.columns:
            features_df['log_amount'] = np.log1p(features_df['amount'])
            features_df['amount_zscore'] = (features_df['amount'] - features_df['amount'].mean()) / features_df['amount'].std()
        
        # Frequency features (if multiple transactions per account)
        if 'sender_account' in features_df.columns:
            sender_counts = features_df['sender_account'].value_counts()
            features_df['sender_frequency'] = features_df['sender_account'].map(sender_counts)
        
        if 'receiver_account' in features_df.columns:
            receiver_counts = features_df['receiver_account'].value_counts()
            features_df['receiver_frequency'] = features_df['receiver_account'].map(receiver_counts)
        
        # Select numerical features for modeling
        numerical_features = [
            'amount', 'log_amount', 'amount_zscore', 'hour', 'day_of_week', 
            'is_weekend', 'sender_frequency', 'receiver_frequency'
        ]
        
        # Filter to only include available features
        available_features = [col for col in numerical_features if col in features_df.columns]
        
        self.feature_columns = available_features
        return features_df[available_features].fillna(0)
    
    def train_isolation_forest(self, df: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """Train Isolation Forest model."""
        features_df = self.prepare_features(df)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        # Train Isolation Forest
        isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        isolation_forest.fit(scaled_features)
        
        # Store models
        self.models['isolation_forest'] = isolation_forest
        self.scalers['isolation_forest'] = scaler
        
        # Save models
        joblib.dump(isolation_forest, os.path.join(self.model_path, 'isolation_forest.pkl'))
        joblib.dump(scaler, os.path.join(self.model_path, 'isolation_forest_scaler.pkl'))
        
        # Get predictions for training data
        predictions = isolation_forest.predict(scaled_features)
        anomaly_scores = isolation_forest.decision_function(scaled_features)
        
        return {
            'model_type': 'isolation_forest',
            'contamination': contamination,
            'n_samples': len(features_df),
            'n_anomalies': np.sum(predictions == -1),
            'anomaly_scores': anomaly_scores,
            'predictions': predictions
        }
    
    def predict_anomalies(self, df: pd.DataFrame, model_type: str = 'isolation_forest') -> Dict[str, Any]:
        """Predict anomalies using trained models."""
        if model_type not in self.models:
            # Try to load the model
            self.load_model(model_type)
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Please train the model first.")
        
        features_df = self.prepare_features(df)
        
        # Scale features using the saved scaler
        scaler = self.scalers[model_type]
        scaled_features = scaler.transform(features_df)
        
        # Get predictions
        model = self.models[model_type]
        predictions = model.predict(scaled_features)
        anomaly_scores = model.decision_function(scaled_features)
        
        # Convert to binary labels (1 for normal, 0 for anomaly)
        binary_predictions = (predictions == 1).astype(int)
        
        return {
            'predictions': binary_predictions,
            'anomaly_scores': anomaly_scores,
            'is_anomaly': predictions == -1,
            'n_anomalies': np.sum(predictions == -1)
        }
    
    def load_model(self, model_type: str) -> bool:
        """Load a trained model from disk."""
        try:
            model_file = os.path.join(self.model_path, f'{model_type}.pkl')
            scaler_file = os.path.join(self.model_path, f'{model_type}_scaler.pkl')
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.models[model_type] = joblib.load(model_file)
                self.scalers[model_type] = joblib.load(scaler_file)
                return True
            return False
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")
            return False
    
    def evaluate_model(self, df: pd.DataFrame, true_labels: List[int], model_type: str = 'isolation_forest') -> Dict[str, Any]:
        """Evaluate model performance against true labels."""
        predictions = self.predict_anomalies(df, model_type)
        
        # Convert anomaly predictions to same format as true labels
        pred_labels = (~predictions['is_anomaly']).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],  # Normal class precision
            'recall': report['1']['recall'],        # Normal class recall
            'f1_score': report['1']['f1-score']     # Normal class F1
        }


class RuleBasedDetector:
    """Rule-based anomaly detection for specific transaction patterns."""
    
    def __init__(self):
        """Initialize rule-based detector."""
        self.rules = {
            'high_amount': {'threshold': 100000, 'severity': 'high'},
            'unusual_time': {'hours': [0, 1, 2, 3, 4, 5], 'severity': 'medium'},
            'rapid_succession': {'time_window': 300, 'max_count': 5, 'severity': 'high'},
            'round_amount': {'pattern': '00000', 'severity': 'low'}
        }
    
    def detect_high_amount(self, df: pd.DataFrame) -> pd.Series:
        """Detect transactions with unusually high amounts."""
        threshold = self.rules['high_amount']['threshold']
        return df['amount'] > threshold
    
    def detect_unusual_time(self, df: pd.DataFrame) -> pd.Series:
        """Detect transactions at unusual hours."""
        if 'timestamp' in df.columns:
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            unusual_hours = self.rules['unusual_time']['hours']
            return df_copy['timestamp'].dt.hour.isin(unusual_hours)
        return pd.Series([False] * len(df))
    
    def detect_round_amounts(self, df: pd.DataFrame) -> pd.Series:
        """Detect suspiciously round amounts."""
        pattern = self.rules['round_amount']['pattern']
        return df['amount'].astype(str).str.endswith(pattern)
    
    def apply_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all rule-based detections."""
        results = df.copy()
        
        results['high_amount_flag'] = self.detect_high_amount(df)
        results['unusual_time_flag'] = self.detect_unusual_time(df)
        results['round_amount_flag'] = self.detect_round_amounts(df)
        
        # Combine flags
        results['rule_based_anomaly'] = (
            results['high_amount_flag'] | 
            results['unusual_time_flag'] | 
            results['round_amount_flag']
        )
        
        return results
