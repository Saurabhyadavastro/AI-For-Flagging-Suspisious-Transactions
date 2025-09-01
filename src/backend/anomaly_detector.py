"""
Core AI Anomaly Detection Module for Indian Transaction Fraud Detection.

This module implements unsupervised learning algorithms to detect suspicious transactions
in Indian financial data. It uses advanced machine learning techniques including
Isolation Forest and One-Class SVM for anomaly detection without labeled data.

Key Features:
- Multiple unsupervised learning algorithms
- Comprehensive feature engineering for Indian financial data
- Geographic and temporal pattern analysis
- Performance optimization for large datasets
- Detailed anomaly scoring and explanation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, time
import warnings
from pathlib import Path
import json

# Machine Learning imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndianTransactionAnomalyDetector:
    """
    Advanced anomaly detection system for Indian financial transactions.
    
    This class implements multiple unsupervised learning algorithms to detect
    suspicious patterns in transaction data. It's specifically designed for
    the Indian financial ecosystem with features that capture local patterns.
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize the anomaly detection system.
        
        Args:
            contamination (float): Expected proportion of outliers in the data (default: 0.05)
            random_state (int): Random state for reproducible results (default: 42)
        """
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize models
        self.isolation_forest = None
        self.one_class_svm = None
        self.dbscan = None
        
        # Feature engineering components
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
        # Model performance metrics
        self.model_scores = {}
        
        logger.info(f"Initialized IndianTransactionAnomalyDetector with contamination={contamination}")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specifically for transaction anomaly detection.
        
        This method creates comprehensive features that capture patterns
        specific to the Indian financial ecosystem.
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Engineered features ready for ML models
        """
        logger.info("Starting feature engineering for Indian transaction data")
        
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # === TEMPORAL FEATURES ===
        if 'transaction_date' in features_df.columns:
            features_df['transaction_date'] = pd.to_datetime(features_df['transaction_date'])
            features_df['day_of_week'] = features_df['transaction_date'].dt.dayofweek
            features_df['day_of_month'] = features_df['transaction_date'].dt.day
            features_df['month'] = features_df['transaction_date'].dt.month
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
            features_df['is_month_end'] = (features_df['day_of_month'] >= 28).astype(int)
        
        if 'transaction_time' in features_df.columns:
            # Convert time to minutes since midnight for numerical analysis
            if features_df['transaction_time'].dtype == 'object':
                features_df['transaction_time'] = pd.to_datetime(features_df['transaction_time'], format='%H:%M:%S').dt.time
            
            features_df['hour'] = features_df['transaction_time'].apply(lambda x: x.hour if pd.notna(x) else 12)
            features_df['minute'] = features_df['transaction_time'].apply(lambda x: x.minute if pd.notna(x) else 0)
            features_df['minutes_since_midnight'] = features_df['hour'] * 60 + features_df['minute']
            
            # Time-based risk categories for Indian context
            features_df['is_night_transaction'] = ((features_df['hour'] >= 23) | (features_df['hour'] <= 5)).astype(int)
            features_df['is_business_hours'] = ((features_df['hour'] >= 9) & (features_df['hour'] <= 17)).astype(int)
            features_df['is_peak_hours'] = ((features_df['hour'] >= 10) & (features_df['hour'] <= 14)).astype(int)
        
        # === AMOUNT FEATURES ===
        if 'transaction_amount_inr' in features_df.columns:
            features_df['amount_log'] = np.log1p(features_df['transaction_amount_inr'])
            features_df['amount_sqrt'] = np.sqrt(features_df['transaction_amount_inr'])
            
            # Indian-specific amount thresholds
            features_df['is_high_value'] = (features_df['transaction_amount_inr'] > 200000).astype(int)  # RTGS threshold
            features_df['is_cash_limit'] = (features_df['transaction_amount_inr'] > 50000).astype(int)   # Cash reporting limit
            features_df['is_round_amount'] = (features_df['transaction_amount_inr'] % 1000 == 0).astype(int)
            features_df['is_suspicious_round'] = (features_df['transaction_amount_inr'] % 10000 == 0).astype(int)
            
            # Amount percentile features
            features_df['amount_percentile'] = features_df['transaction_amount_inr'].rank(pct=True)
        
        # === CATEGORICAL ENCODING ===
        categorical_features = ['payment_method', 'location', 'merchant_category']
        
        for feature in categorical_features:
            if feature in features_df.columns:
                # Label encoding for categorical features
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    features_df[f'{feature}_encoded'] = self.encoders[feature].fit_transform(features_df[feature].astype(str))
                else:
                    # Handle unseen categories
                    known_categories = set(self.encoders[feature].classes_)
                    features_df[feature] = features_df[feature].astype(str)
                    unknown_mask = ~features_df[feature].isin(known_categories)
                    
                    if unknown_mask.any():
                        # Assign unknown categories to a default value
                        features_df.loc[unknown_mask, feature] = self.encoders[feature].classes_[0]
                    
                    features_df[f'{feature}_encoded'] = self.encoders[feature].transform(features_df[feature])
                
                # One-hot encoding for high cardinality features
                if features_df[feature].nunique() <= 20:  # Only for low cardinality
                    dummies = pd.get_dummies(features_df[feature], prefix=feature, drop_first=True)
                    features_df = pd.concat([features_df, dummies], axis=1)
        
        # === PAYMENT METHOD SPECIFIC FEATURES ===
        if 'payment_method' in features_df.columns:
            # Risk scores based on Indian payment method characteristics
            payment_risk_scores = {
                'UPI': 0.1,      # Low risk, widely used
                'IMPS': 0.2,     # Medium-low risk
                'NEFT': 0.3,     # Medium risk
                'Net Banking': 0.2,  # Medium-low risk
                'Card': 0.4,     # Medium risk due to skimming
                'RTGS': 0.6,     # Higher risk for large amounts
                'Wallet': 0.3,   # Medium risk
                'Cash': 0.8      # Highest risk for large amounts
            }
            features_df['payment_risk_score'] = features_df['payment_method'].map(payment_risk_scores).fillna(0.5)
        
        # === LOCATION FEATURES ===
        if 'location' in features_df.columns:
            # Indian city tiers based on economic activity and fraud patterns
            tier1_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune']
            tier2_cities = ['Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore']
            
            features_df['is_tier1_city'] = features_df['location'].isin(tier1_cities).astype(int)
            features_df['is_tier2_city'] = features_df['location'].isin(tier2_cities).astype(int)
            features_df['is_tier3_city'] = (~features_df['location'].isin(tier1_cities + tier2_cities)).astype(int)
        
        # === CUSTOMER BEHAVIOR FEATURES ===
        if 'customer_id' in features_df.columns:
            # Customer transaction frequency
            customer_counts = features_df['customer_id'].value_counts()
            features_df['customer_transaction_count'] = features_df['customer_id'].map(customer_counts)
            features_df['is_frequent_customer'] = (features_df['customer_transaction_count'] > 5).astype(int)
            
            # Customer amount patterns
            customer_avg_amounts = features_df.groupby('customer_id')['transaction_amount_inr'].mean()
            features_df['customer_avg_amount'] = features_df['customer_id'].map(customer_avg_amounts)
            features_df['amount_deviation_from_avg'] = (
                features_df['transaction_amount_inr'] - features_df['customer_avg_amount']
            ) / features_df['customer_avg_amount']
        
        # === SEQUENCE AND VELOCITY FEATURES ===
        if 'transaction_date' in features_df.columns and 'customer_id' in features_df.columns:
            # Sort by customer and date
            features_df = features_df.sort_values(['customer_id', 'transaction_date'])
            
            # Time between transactions for the same customer
            features_df['time_since_last_transaction'] = (
                features_df.groupby('customer_id')['transaction_date']
                .diff().dt.total_seconds() / 3600  # Hours
            ).fillna(24)  # Default to 24 hours for first transaction
            
            features_df['is_rapid_transaction'] = (features_df['time_since_last_transaction'] < 1).astype(int)
        
        # === STATISTICAL FEATURES ===
        # Z-scores for amount relative to overall distribution
        if 'transaction_amount_inr' in features_df.columns:
            amount_mean = features_df['transaction_amount_inr'].mean()
            amount_std = features_df['transaction_amount_inr'].std()
            features_df['amount_zscore'] = (features_df['transaction_amount_inr'] - amount_mean) / amount_std
            features_df['is_amount_outlier'] = (np.abs(features_df['amount_zscore']) > 3).astype(int)
        
        # === SELECT NUMERICAL FEATURES FOR ML ===
        # Select only numerical features for ML models
        numerical_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove original categorical columns and keep only engineered features
        exclude_columns = ['transaction_date', 'transaction_time'] + categorical_features
        if 'is_suspicious' in numerical_features:
            exclude_columns.append('is_suspicious')  # Remove target if present
        
        for col in exclude_columns:
            if col in numerical_features:
                numerical_features.remove(col)
        
        final_features = features_df[numerical_features].fillna(0)
        
        self.feature_names = numerical_features
        logger.info(f"Feature engineering completed. Created {len(numerical_features)} features")
        
        return final_features
    
    def _fit_isolation_forest(self, X: pd.DataFrame) -> IsolationForest:
        """
        Train Isolation Forest model for anomaly detection.
        
        Args:
            X (pd.DataFrame): Preprocessed feature matrix
            
        Returns:
            IsolationForest: Trained Isolation Forest model
        """
        logger.info("Training Isolation Forest model...")
        
        # Hyperparameter tuning for Isolation Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_samples': ['auto', 0.8],
            'max_features': [0.8, 1.0],
            'contamination': [self.contamination]
        }
        
        base_model = IsolationForest(random_state=self.random_state)
        
        # Use a subset for hyperparameter tuning if dataset is large
        if len(X) > 10000:
            sample_size = min(5000, len(X))
            X_sample = X.sample(n=sample_size, random_state=self.random_state)
        else:
            X_sample = X
        
        # Custom scoring function for unsupervised learning
        def anomaly_score(estimator, X):
            scores = estimator.decision_function(X)
            return scores.mean()
        
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            scoring=anomaly_score,
            cv=3,
            n_jobs=-1
        )
        
        grid_search.fit(X_sample)
        best_model = grid_search.best_estimator_
        
        # Fit on full dataset
        best_model.fit(X)
        
        logger.info(f"Isolation Forest training completed. Best parameters: {grid_search.best_params_}")
        return best_model
    
    def _fit_one_class_svm(self, X: pd.DataFrame) -> OneClassSVM:
        """
        Train One-Class SVM model for anomaly detection.
        
        Args:
            X (pd.DataFrame): Preprocessed feature matrix
            
        Returns:
            OneClassSVM: Trained One-Class SVM model
        """
        logger.info("Training One-Class SVM model...")
        
        # Scale features for SVM
        if 'svm_scaler' not in self.scalers:
            self.scalers['svm_scaler'] = StandardScaler()
            X_scaled = self.scalers['svm_scaler'].fit_transform(X)
        else:
            X_scaled = self.scalers['svm_scaler'].transform(X)
        
        # Hyperparameter tuning for One-Class SVM
        param_grid = {
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'nu': [self.contamination, self.contamination * 1.5, self.contamination * 2]
        }
        
        base_model = OneClassSVM()
        
        # Use a subset for hyperparameter tuning if dataset is large
        if len(X_scaled) > 5000:
            sample_size = min(2000, len(X_scaled))
            indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled[indices]
        else:
            X_sample = X_scaled
        
        # Custom scoring function
        def svm_score(estimator, X):
            scores = estimator.decision_function(X)
            return scores.mean()
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring=svm_score,
            cv=3,
            n_jobs=-1
        )
        
        grid_search.fit(X_sample)
        best_model = grid_search.best_estimator_
        
        # Fit on full dataset
        best_model.fit(X_scaled)
        
        logger.info(f"One-Class SVM training completed. Best parameters: {grid_search.best_params_}")
        return best_model
    
    def _fit_dbscan_clustering(self, X: pd.DataFrame) -> DBSCAN:
        """
        Train DBSCAN clustering for additional anomaly detection perspective.
        
        Args:
            X (pd.DataFrame): Preprocessed feature matrix
            
        Returns:
            DBSCAN: Trained DBSCAN clustering model
        """
        logger.info("Training DBSCAN clustering model...")
        
        # Scale features for DBSCAN
        if 'dbscan_scaler' not in self.scalers:
            self.scalers['dbscan_scaler'] = StandardScaler()
            X_scaled = self.scalers['dbscan_scaler'].fit_transform(X)
        else:
            X_scaled = self.scalers['dbscan_scaler'].transform(X)
        
        # Use PCA for dimensionality reduction if too many features
        if X_scaled.shape[1] > 20:
            pca = PCA(n_components=min(20, X_scaled.shape[1]), random_state=self.random_state)
            X_scaled = pca.fit_transform(X_scaled)
        
        # Hyperparameter tuning for DBSCAN
        eps_range = np.arange(0.1, 1.0, 0.1)
        min_samples_range = [3, 5, 10, 15]
        
        best_score = -1
        best_params = {}
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:  # More than just noise
                    try:
                        score = silhouette_score(X_scaled, labels)
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
                    except:
                        continue
        
        # Train final model with best parameters
        if best_params:
            final_dbscan = DBSCAN(**best_params)
            final_dbscan.fit(X_scaled)
            logger.info(f"DBSCAN training completed. Best parameters: {best_params}")
        else:
            # Fallback parameters
            final_dbscan = DBSCAN(eps=0.5, min_samples=5)
            final_dbscan.fit(X_scaled)
            logger.info("DBSCAN training completed with default parameters")
        
        return final_dbscan
    
    def fit(self, df: pd.DataFrame, use_models: List[str] = None) -> Dict[str, Any]:
        """
        Train anomaly detection models on transaction data.
        
        Args:
            df (pd.DataFrame): Transaction data with required columns
            use_models (List[str], optional): Models to train ['isolation_forest', 'one_class_svm', 'dbscan']
                                            If None, trains all models
        
        Returns:
            Dict[str, Any]: Training results and model performance metrics
        """
        logger.info(f"Starting training on {len(df)} transactions")
        
        if use_models is None:
            use_models = ['isolation_forest', 'one_class_svm', 'dbscan']
        
        # Feature engineering
        X = self._prepare_features(df)
        
        results = {
            'training_samples': len(X),
            'features_created': len(X.columns),
            'feature_names': list(X.columns),
            'models_trained': [],
            'training_time': {}
        }
        
        # Train Isolation Forest
        if 'isolation_forest' in use_models:
            start_time = datetime.now()
            self.isolation_forest = self._fit_isolation_forest(X)
            training_time = (datetime.now() - start_time).total_seconds()
            
            results['models_trained'].append('isolation_forest')
            results['training_time']['isolation_forest'] = training_time
        
        # Train One-Class SVM
        if 'one_class_svm' in use_models:
            start_time = datetime.now()
            self.one_class_svm = self._fit_one_class_svm(X)
            training_time = (datetime.now() - start_time).total_seconds()
            
            results['models_trained'].append('one_class_svm')
            results['training_time']['one_class_svm'] = training_time
        
        # Train DBSCAN
        if 'dbscan' in use_models:
            start_time = datetime.now()
            self.dbscan = self._fit_dbscan_clustering(X)
            training_time = (datetime.now() - start_time).total_seconds()
            
            results['models_trained'].append('dbscan')
            results['training_time']['dbscan'] = training_time
        
        logger.info(f"Training completed. Models trained: {results['models_trained']}")
        return results
    
    def predict_anomalies(self, df: pd.DataFrame, ensemble_method: str = 'majority_vote') -> Dict[str, Any]:
        """
        Detect anomalies in transaction data using trained models.
        
        Args:
            df (pd.DataFrame): Transaction data to analyze
            ensemble_method (str): Method to combine predictions ('majority_vote', 'weighted_average', 'conservative')
        
        Returns:
            Dict[str, Any]: Comprehensive anomaly detection results
        """
        logger.info(f"Starting anomaly detection on {len(df)} transactions")
        
        # Feature engineering
        X = self._prepare_features(df)
        
        # Initialize results dictionary
        results = {
            'total_transactions': len(df),
            'total_suspicious_transactions': 0,
            'suspicious_hotspots': {},
            'suspicious_time_patterns': {},
            'anomaly_scores': {},
            'model_predictions': {},
            'flagged_transactions': pd.DataFrame(),
            'feature_importance': {},
            'confidence_metrics': {}
        }
        
        # Collect predictions from all trained models
        predictions = {}
        scores = {}
        
        # Isolation Forest predictions
        if self.isolation_forest is not None:
            if_predictions = self.isolation_forest.predict(X)
            if_scores = self.isolation_forest.decision_function(X)
            
            # Convert to binary (1 = normal, -1 = anomaly) -> (0 = normal, 1 = anomaly)
            predictions['isolation_forest'] = (if_predictions == -1).astype(int)
            scores['isolation_forest'] = -if_scores  # Invert so higher score = more anomalous
            
            results['model_predictions']['isolation_forest'] = predictions['isolation_forest'].tolist()
            results['anomaly_scores']['isolation_forest'] = scores['isolation_forest'].tolist()
        
        # One-Class SVM predictions
        if self.one_class_svm is not None:
            X_scaled = self.scalers['svm_scaler'].transform(X)
            svm_predictions = self.one_class_svm.predict(X_scaled)
            svm_scores = self.one_class_svm.decision_function(X_scaled)
            
            predictions['one_class_svm'] = (svm_predictions == -1).astype(int)
            scores['one_class_svm'] = -svm_scores  # Invert so higher score = more anomalous
            
            results['model_predictions']['one_class_svm'] = predictions['one_class_svm'].tolist()
            results['anomaly_scores']['one_class_svm'] = scores['one_class_svm'].tolist()
        
        # DBSCAN predictions (noise points are considered anomalies)
        if self.dbscan is not None:
            X_scaled = self.scalers['dbscan_scaler'].transform(X)
            
            # Use PCA if it was used during training
            if X_scaled.shape[1] > 20:
                pca = PCA(n_components=min(20, X_scaled.shape[1]), random_state=self.random_state)
                X_scaled = pca.fit_transform(X_scaled)
            
            dbscan_labels = self.dbscan.fit_predict(X_scaled)
            dbscan_predictions = (dbscan_labels == -1).astype(int)  # -1 = noise/anomaly
            
            predictions['dbscan'] = dbscan_predictions
            # For DBSCAN, use distance to nearest cluster as anomaly score
            # This is a simplified approach
            scores['dbscan'] = dbscan_predictions.astype(float)
            
            results['model_predictions']['dbscan'] = predictions['dbscan'].tolist()
            results['anomaly_scores']['dbscan'] = scores['dbscan'].tolist()
        
        # Ensemble prediction
        if predictions:
            prediction_matrix = np.column_stack(list(predictions.values()))
            score_matrix = np.column_stack(list(scores.values()))
            
            if ensemble_method == 'majority_vote':
                # Simple majority vote
                ensemble_predictions = (prediction_matrix.sum(axis=1) > len(predictions) / 2).astype(int)
                ensemble_scores = score_matrix.mean(axis=1)
            
            elif ensemble_method == 'weighted_average':
                # Weighted by model confidence
                weights = [0.4, 0.4, 0.2] if len(predictions) == 3 else [0.5, 0.5]  # IF, SVM, DBSCAN
                ensemble_scores = np.average(score_matrix, axis=1, weights=weights[:len(predictions)])
                ensemble_predictions = (ensemble_scores > np.percentile(ensemble_scores, 95)).astype(int)
            
            else:  # conservative
                # Flag as anomaly only if all models agree
                ensemble_predictions = (prediction_matrix.sum(axis=1) == len(predictions)).astype(int)
                ensemble_scores = score_matrix.min(axis=1)
            
            # Update results with ensemble predictions
            anomaly_indices = np.where(ensemble_predictions == 1)[0]
            results['total_suspicious_transactions'] = len(anomaly_indices)
            
            # Create flagged transactions dataframe
            flagged_df = df.iloc[anomaly_indices].copy()
            flagged_df['anomaly_score'] = ensemble_scores[anomaly_indices]
            flagged_df['prediction_confidence'] = prediction_matrix[anomaly_indices].mean(axis=1)
            
            # Add model-specific scores
            for model_name, model_scores in scores.items():
                flagged_df[f'{model_name}_score'] = model_scores[anomaly_indices]
            
            results['flagged_transactions'] = flagged_df
            
            # === ANALYSIS: SUSPICIOUS HOTSPOTS ===
            if len(flagged_df) > 0 and 'location' in flagged_df.columns:
                location_counts = flagged_df['location'].value_counts().to_dict()
                total_by_location = df['location'].value_counts().to_dict()
                
                # Calculate suspicion rates by location
                hotspots = {}
                for location, suspicious_count in location_counts.items():
                    total_count = total_by_location.get(location, 1)
                    suspicion_rate = (suspicious_count / total_count) * 100
                    hotspots[location] = {
                        'suspicious_transactions': suspicious_count,
                        'total_transactions': total_count,
                        'suspicion_rate_percentage': round(suspicion_rate, 2)
                    }
                
                # Sort by suspicion rate
                results['suspicious_hotspots'] = dict(
                    sorted(hotspots.items(), key=lambda x: x[1]['suspicion_rate_percentage'], reverse=True)
                )
            
            # === ANALYSIS: TIME PATTERNS ===
            if len(flagged_df) > 0:
                time_patterns = {}
                
                # Hour-based analysis
                if 'transaction_time' in flagged_df.columns:
                    # Extract hour from transaction_time
                    if flagged_df['transaction_time'].dtype == 'object':
                        flagged_df['hour'] = pd.to_datetime(flagged_df['transaction_time'], format='%H:%M:%S').dt.hour
                    else:
                        flagged_df['hour'] = flagged_df['transaction_time'].apply(lambda x: x.hour if pd.notna(x) else 12)
                    
                    hour_counts = flagged_df['hour'].value_counts().to_dict()
                    total_hour_counts = df['hour'] if 'hour' in df.columns else pd.to_datetime(df['transaction_time'], format='%H:%M:%S').dt.hour
                    total_hour_counts = total_hour_counts.value_counts().to_dict()
                    
                    hourly_patterns = {}
                    for hour, count in hour_counts.items():
                        total = total_hour_counts.get(hour, 1)
                        rate = (count / total) * 100
                        hourly_patterns[hour] = {
                            'suspicious_count': count,
                            'total_count': total,
                            'suspicion_rate_percentage': round(rate, 2)
                        }
                    
                    time_patterns['hourly_distribution'] = dict(
                        sorted(hourly_patterns.items(), key=lambda x: x[1]['suspicion_rate_percentage'], reverse=True)
                    )
                
                # Day of week analysis
                if 'transaction_date' in flagged_df.columns:
                    flagged_df['day_of_week'] = pd.to_datetime(flagged_df['transaction_date']).dt.day_name()
                    dow_counts = flagged_df['day_of_week'].value_counts().to_dict()
                    
                    df_temp = df.copy()
                    df_temp['day_of_week'] = pd.to_datetime(df_temp['transaction_date']).dt.day_name()
                    total_dow_counts = df_temp['day_of_week'].value_counts().to_dict()
                    
                    dow_patterns = {}
                    for dow, count in dow_counts.items():
                        total = total_dow_counts.get(dow, 1)
                        rate = (count / total) * 100
                        dow_patterns[dow] = {
                            'suspicious_count': count,
                            'total_count': total,
                            'suspicion_rate_percentage': round(rate, 2)
                        }
                    
                    time_patterns['day_of_week_distribution'] = dow_patterns
                
                results['suspicious_time_patterns'] = time_patterns
            
            # === FEATURE IMPORTANCE ANALYSIS ===
            if self.isolation_forest is not None and hasattr(self.isolation_forest, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.isolation_forest.feature_importances_))
                results['feature_importance'] = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
            
            # === CONFIDENCE METRICS ===
            if len(predictions) > 1:
                # Calculate agreement between models
                agreement_rate = (prediction_matrix.sum(axis=1) == len(predictions)).mean()
                disagreement_rate = ((prediction_matrix.max(axis=1) - prediction_matrix.min(axis=1)) > 0).mean()
                
                results['confidence_metrics'] = {
                    'model_agreement_rate': round(agreement_rate, 4),
                    'model_disagreement_rate': round(disagreement_rate, 4),
                    'ensemble_method_used': ensemble_method,
                    'models_used': list(predictions.keys()),
                    'average_anomaly_score': round(ensemble_scores.mean(), 4),
                    'anomaly_score_std': round(ensemble_scores.std(), 4)
                }
        
        logger.info(f"Anomaly detection completed. Found {results['total_suspicious_transactions']} suspicious transactions")
        return results
    
    def save_models(self, model_path: str) -> Dict[str, str]:
        """
        Save trained models to disk for later use.
        
        Args:
            model_path (str): Directory path to save models
        
        Returns:
            Dict[str, str]: Paths where models were saved
        """
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save models
        if self.isolation_forest is not None:
            if_path = model_dir / 'isolation_forest.joblib'
            joblib.dump(self.isolation_forest, if_path)
            saved_paths['isolation_forest'] = str(if_path)
        
        if self.one_class_svm is not None:
            svm_path = model_dir / 'one_class_svm.joblib'
            joblib.dump(self.one_class_svm, svm_path)
            saved_paths['one_class_svm'] = str(svm_path)
        
        if self.dbscan is not None:
            dbscan_path = model_dir / 'dbscan.joblib'
            joblib.dump(self.dbscan, dbscan_path)
            saved_paths['dbscan'] = str(dbscan_path)
        
        # Save scalers and encoders
        if self.scalers:
            scalers_path = model_dir / 'scalers.joblib'
            joblib.dump(self.scalers, scalers_path)
            saved_paths['scalers'] = str(scalers_path)
        
        if self.encoders:
            encoders_path = model_dir / 'encoders.joblib'
            joblib.dump(self.encoders, encoders_path)
            saved_paths['encoders'] = str(encoders_path)
        
        # Save feature names
        if self.feature_names:
            features_path = model_dir / 'feature_names.json'
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f)
            saved_paths['feature_names'] = str(features_path)
        
        logger.info(f"Models saved to {model_path}")
        return saved_paths
    
    def load_models(self, model_path: str) -> Dict[str, bool]:
        """
        Load previously trained models from disk.
        
        Args:
            model_path (str): Directory path containing saved models
        
        Returns:
            Dict[str, bool]: Status of model loading
        """
        model_dir = Path(model_path)
        load_status = {}
        
        # Load models
        if_path = model_dir / 'isolation_forest.joblib'
        if if_path.exists():
            self.isolation_forest = joblib.load(if_path)
            load_status['isolation_forest'] = True
        else:
            load_status['isolation_forest'] = False
        
        svm_path = model_dir / 'one_class_svm.joblib'
        if svm_path.exists():
            self.one_class_svm = joblib.load(svm_path)
            load_status['one_class_svm'] = True
        else:
            load_status['one_class_svm'] = False
        
        dbscan_path = model_dir / 'dbscan.joblib'
        if dbscan_path.exists():
            self.dbscan = joblib.load(dbscan_path)
            load_status['dbscan'] = True
        else:
            load_status['dbscan'] = False
        
        # Load scalers and encoders
        scalers_path = model_dir / 'scalers.joblib'
        if scalers_path.exists():
            self.scalers = joblib.load(scalers_path)
            load_status['scalers'] = True
        else:
            load_status['scalers'] = False
        
        encoders_path = model_dir / 'encoders.joblib'
        if encoders_path.exists():
            self.encoders = joblib.load(encoders_path)
            load_status['encoders'] = True
        else:
            load_status['encoders'] = False
        
        # Load feature names
        features_path = model_dir / 'feature_names.json'
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            load_status['feature_names'] = True
        else:
            load_status['feature_names'] = False
        
        logger.info(f"Models loaded from {model_path}: {load_status}")
        return load_status


# Convenience functions for easy usage
def detect_anomalies_from_dataframe(df: pd.DataFrame, 
                                   contamination: float = 0.05,
                                   models_to_use: List[str] = None,
                                   ensemble_method: str = 'majority_vote') -> Dict[str, Any]:
    """
    Convenience function to detect anomalies from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): Transaction data
        contamination (float): Expected proportion of anomalies
        models_to_use (List[str]): Models to use for detection
        ensemble_method (str): Method to combine model predictions
    
    Returns:
        Dict[str, Any]: Comprehensive anomaly detection results
    """
    detector = IndianTransactionAnomalyDetector(contamination=contamination)
    
    # Train models
    training_results = detector.fit(df, use_models=models_to_use)
    
    # Detect anomalies
    detection_results = detector.predict_anomalies(df, ensemble_method=ensemble_method)
    
    # Combine results
    combined_results = {
        'training_info': training_results,
        'detection_results': detection_results
    }
    
    return combined_results


def detect_anomalies_from_file(file_path: str,
                              contamination: float = 0.05,
                              models_to_use: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to detect anomalies from a CSV or Excel file.
    
    Args:
        file_path (str): Path to the data file
        contamination (float): Expected proportion of anomalies
        models_to_use (List[str]): Models to use for detection
    
    Returns:
        Dict[str, Any]: Comprehensive anomaly detection results
    """
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return detect_anomalies_from_dataframe(df, contamination, models_to_use)


# Example usage and testing function
def test_anomaly_detector():
    """
    Test function to demonstrate the suspicious transaction flagging capabilities.
    """
    # This would typically use the data generator from the previous module
    # For testing purposes, create some sample data
    
    import random
    from datetime import datetime, timedelta
    
    # Generate sample transaction data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(n_samples)],
        'customer_id': [f'CUST{random.randint(1000, 9999)}' for _ in range(n_samples)],
        'transaction_amount_inr': np.random.lognormal(8, 1.5, n_samples),
        'transaction_date': [datetime.now() - timedelta(days=random.randint(0, 30)) for _ in range(n_samples)],
        'transaction_time': [f'{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}' for _ in range(n_samples)],
        'payment_method': np.random.choice(['UPI', 'NEFT', 'IMPS', 'Card'], n_samples),
        'location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai'], n_samples),
        'merchant_category': np.random.choice(['Grocery', 'Restaurant', 'Fuel', 'Retail'], n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Run anomaly detection
    results = detect_anomalies_from_dataframe(df)
    
    print("Anomaly Detection Test Results:")
    print(f"Total transactions analyzed: {results['detection_results']['total_transactions']}")
    print(f"Suspicious transactions found: {results['detection_results']['total_suspicious_transactions']}")
    print(f"Models used: {results['training_info']['models_trained']}")
    
    return results


if __name__ == "__main__":
    # Run test if script is executed directly
    test_results = test_anomaly_detector()
