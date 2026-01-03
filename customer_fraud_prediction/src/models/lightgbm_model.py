"""
LightGBM model for fraud detection with class imbalance handling

LightGBM is often faster than XGBoost and can handle imbalance well.
Differences from XGBoost:
- Uses leaf-wise tree growth (vs level-wise in XGBoost)
- Faster training on large datasets
- Lower memory usage
"""

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.lightgbm


class LightGBMFraudDetector:
    """
    LightGBM-based fraud detection with imbalance handling
    
    Parameters:
    use_smote : bool, default=False
        If True, use SMOTE oversampling
        
    smote_strategy : float or str, default='auto'
        SMOTE sampling strategy
        
    random_state : int, default=42
        Random seed for reproducibility
        
    **lgb_params : dict
        Additional LightGBM parameters
    """
    
    def __init__(self, use_smote=False, smote_strategy='auto', random_state=42, **lgb_params):
        self.use_smote = use_smote
        self.smote_strategy = smote_strategy
        self.random_state = random_state
        self.lgb_params = lgb_params
        self.model = None
        self.smote = None
    
    def _calculate_scale_pos_weight(self, y):
        """Calculate scale_pos_weight for class imbalance"""
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        
        if n_positive == 0:
            raise ValueError("No positive samples in training data!")
        
        scale_pos_weight = n_negative / n_positive
        
        print(f"\n=== Class Imbalance Analysis ===")
        print(f"Negative samples: {n_negative:,}")
        print(f"Positive samples: {n_positive:,}")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        return scale_pos_weight
    
    def _apply_smote(self, X, y):
        """Apply SMOTE oversampling"""
        print(f"Original fraud count: {np.sum(y == 1):,}")
        
        self.smote = SMOTE(
            sampling_strategy=self.smote_strategy,
            random_state=self.random_state,
            k_neighbors=5
        )
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        print(f"Resampled fraud count: {np.sum(y_resampled == 1):,}")
        
        return X_resampled, y_resampled
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train LightGBM model
        """
        print("LightGBM Training with Imbalance Handling")
        
        # Apply SMOTE if requested
        if self.use_smote:
            X_train_processed, y_train_processed = self._apply_smote(X_train, y_train)
            scale_pos_weight = 1
        else:
            X_train_processed = X_train
            y_train_processed = y_train
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        
        # LightGBM parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': self.random_state,
            'verbose': -1  # Suppress warnings
        }
        
        # Override with user parameters
        default_params.update(self.lgb_params)
        
        for key, value in default_params.items():
            print(f"{key}: {value}")
        
        # Initialize model
        self.model = LGBMClassifier(**default_params)
        
        # Set up early stopping
        eval_set = [(X_train_processed, y_train_processed)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train
        print(f"Training LightGBM ...")
        self.model.fit(
            X_train_processed,
            y_train_processed,
            eval_set=eval_set if len(eval_set) > 1 else None,
            eval_metric='binary_logloss'
        )
        
        print(f"✓ Training complete!")
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.model is None:
            raise ValueError("Model not trained!")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained!")
        return dict(zip(
            self.model.feature_name_,
            self.model.feature_importances_
        ))
