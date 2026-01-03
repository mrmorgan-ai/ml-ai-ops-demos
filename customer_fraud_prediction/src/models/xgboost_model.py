"""
XGBoost model for fraud detection with class imbalance handling

This module implements XGBoost classifier with two approaches:
1. Class weights (scale_pos_weight parameter)
2. SMOTE oversampling
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

class XGBoostFraudDetector:
    """
    XGBoost-based fraud detection with imbalance handling
    
    Parameters:
    use_smote : bool, default=False
        If True, use SMOTE oversampling instead of class weights
        
    smote_strategy : float or str, default='auto'
        SMOTE sampling strategy
        - 'auto': Balance to 1:1 ratio
        - float: Specify ratio (e.g., 0.5 = 1:2 ratio)
        
    random_state : int, default=42
        Random seed for reproducibility
        
    **xgb_params : dict
        Additional XGBoost parameters to override defaults
        
    Methods:
    - Automatic class weight calculation
    - Optional SMOTE oversampling
    - MLflow integration for experiment tracking
    - Model signature inference for production deployment
    """    
    def __init__(self, 
                use_smote:bool = False,
                smote_strategy:str = 'auto',
                random_state:int = 42,
                **xgb_params):
        self.use_smote = use_smote
        self.smote_strategy = smote_strategy
        self.random_state = random_state
        self.xgb_params = xgb_params
        self.model = None
        self.smote = None

    def _calculate_scale_pos_weight(self, y):
        """
        Calculate scale_pos_weight for handling class imbalance
        
        Parameters:
        y : array-like
            Target labels (0 = legitimate, 1 = fraud)
        
        Returns:
        float : Calculated scale_pos_weight value
        """
        n_positive = np.sum(y==1)
        n_negative = np.sum(y==0)
        
        if n_positive == 0:
            raise ValueError("No positive value found in training data")
        
        scale_pos_weight = n_negative/n_positive
        
        print(f"Negative samples (legitimate): {n_negative:,}")
        print(f"Positive samples (fraud):      {n_positive:,}")
        print(f"Calculated scale_pos_weight:   {scale_pos_weight:.2f}")
        return scale_pos_weight

    def _apply_smote(self, X, y):
        """
        Apply SMOTE oversampling to balance the dataset
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training features
            
        y : array-like, shape (n_samples,)
            Training labels
        
        Returns:
        X_resampled, y_resampled : Resampled training data
        """
        print(f"Original shape: {X.shape}")
        print(f"Original fraud count: {np.sum(y == 1):,}")       

        # Initialize SMOTE
        self.smote = SMOTE(
            sampling_strategy=self.smote_strategy,
            random_state = self.random_state,
            k_neighbors = 5 # selected for synthetic sample generation
        )
        
        # Apply SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X,y)
        
        print(f"Resampled shape: {X_resampled.shape}")
        print(f"Resampled fraud count: {np.sum(y_resampled == 1):,}")
        print(f"New fraud rate: {np.mean(y_resampled):.4f}") # This is because 'auto' was selected for sampling_strategy
        return X_resampled, y_resampled

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model with imbalance handling
        
        Training process:
        1. Calculate class weights OR apply SMOTE
        2. Set up XGBoost parameters
        3. Train with early stopping (if validation set provided)
        4. Store trained model
        
        Parameters:
        X_train : array-like, shape (n_samples, n_features)
            Training features
            
        y_train : array-like, shape (n_samples,)
            Training labels
            
        X_val : array-like, optional
            Validation features for early stopping
            
        y_val : array-like, optional
            Validation labels for early stopping
        
        Returns:
        self : Returns self for method chaining
        """
        print("XGBoost Training with Imbalance Handling")        

        # Apply SMOTE if requested
        if self.use_smote:
            X_train_processed, y_train_processed = self._apply_smote(X_train, y_train)
            scale_pos_weight = 1 # Not needed with SMOTE, data balanced
        else:
            X_train_processed = X_train
            y_train_processed = y_train
            scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            
        # Setup XGBoost parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'tree_method': 'hist',  # Faster training
            'use_label_encoder': False
        }
        
        # Overide with provided paraters
        default_params.update(self.xgb_params)        

        print("XGBoost parameters")
        for key, value in default_params.items():
            print(f"{key}: {value}")
            
        # Initialize model
        self.model = XGBClassifier(**default_params)
        
        # Set up early stopping if provided
        eval_set = [(X_train_processed, y_train_processed)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        print("Training XGBoost ...")
        self.model.fit(
            X_train_processed,
            y_train_processed,
            eval_set = eval_set,
            verbose = False # True to see training progress
        )
        
        # Get best iteration (for early stopping)
        if hasattr(self.model, 'best_iteration'):
            print(f"Best iteration: {self.model.best_iteration}")
        
        print(f"✓ Training complete!")
        
        return self        

    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like
            Features to predict
        
        Returns:
        --------
        array : Predicted class labels (0 or 1)
        """    
        if self.model is None:
            raise ValueError("Model not exists")
        
        return self.model.predict(X)

    def get_feature_importance(self):
        """
        Get feature importance scores
        
        XGBoost provides multiple importance types:
        - 'weight': Number of times feature used in splits
        - 'gain': Average gain when feature is used
        - 'cover': Average coverage when feature is used
        
        Returns:
        dict : Feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not exists.")
        
        return self.model.get_booster().get_score(importance_type='gain')    
