"""
Main MLflow pipeline for Credit Card Fraud Detection
Complete training pipeline with multiple models and MLflow tracking

Steps:
1. Load preprocessed data
2. Train baseline models with different imbalance handling techniques
3. Evaluate and compare models
4. Log everything to MLflow for experiment tracking
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


import mlflow
import mlflow.data
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset

from src.pipelines.data_preprocessing import FraudDataLoader
from src.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    find_optimal_threshold
)
from src.models.lightgbm_model import LightGBMFraudDetector
from src.models.xgboost_model import XGBoostFraudDetector

def initialize_mlflow():
    """
    Initialize MLflow experiment
    """
    # Set tracking URI. Local for dev, cloud server in prod
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create or get experiment
    experiment_name = "Fraud_Detection_Production"
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "credit_card_fraud",
                "team":"risk_analytics",
                "mlflow_version":mlflow.__version__
            }
        )
        print(f"Created the experiment {experiment_name} with ID {experiment_id}")
        
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id  # type: ignore
        print(f"Using experiment {experiment} with ID {experiment_id}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_name
    
def log_data_loading_run():
    """
    Loading data and preprocessing process
    """
    # Initialize data loader
    data_loader = FraudDataLoader(
        data_path=r"data\01-raw\creditcard.csv",
        random_state=42
    )
    
    # Start a run and execute within its context
    with mlflow.start_run(run_name="01_data_loading_and_split") as run:
        print("MLFlow Run: data loading and temporal split")
        
        # Add tags for governance
        mlflow.set_tag("stage","data_preprocessing")
        mlflow.set_tag("data_source","kaggle_creditcard_fraud")
        mlflow.set_tag("split_strategy","temporal")
        
        # Load raw data
        df = data_loader.load_data()
        
        # Get and log data stats
        stats = data_loader.get_data_stats(df)
        
        # Log parameters
        mlflow.log_param("data_path", data_loader.data_path)
        mlflow.log_param("random_state", data_loader.random_state)
        mlflow.log_param("total_transactions", stats['total_transactions'])
        mlflow.log_param("features_count", stats['features_count'])        
            
        # Log metrics
        mlflow.log_metric("fraud_count", stats['fraud_count'])
        mlflow.log_metric("fraud_rate", stats['fraud_rate'])
        mlflow.log_metric("legitimate_count", stats['legitimate_count'])
        mlflow.log_metric("avg_transaction_amount", stats['avg_transaction_amount'])
        mlflow.log_metric("max_transaction_amount", stats['max_transaction_amount'])
        mlflow.log_metric("avg_fraud_amount", stats['avg_fraud_amount'])
        mlflow.log_metric("time_span_hours", stats['time_span_hours'])
        
        # Apply temporal script
        train_df,val_df,test_df = data_loader.temporal_train_test_split(
            df,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2
        )
        
        # Log split statistics
        mlflow.log_metric("train_samples", len(train_df))
        mlflow.log_metric("train_fraud_count", int(train_df['Class'].sum()))
        mlflow.log_metric("train_fraud_rate", float(train_df['Class'].mean()))
        
        mlflow.log_metric("val_samples", len(val_df))
        mlflow.log_metric("val_fraud_count", int(val_df['Class'].sum()))
        mlflow.log_metric("val_fraud_rate", float(val_df['Class'].mean()))
        
        mlflow.log_metric("test_samples", len(test_df))
        mlflow.log_metric("test_fraud_count", int(test_df['Class'].sum()))
        mlflow.log_metric("test_fraud_rate", float(test_df['Class'].mean()))
        
        # Save splits to disk
        split_paths = data_loader.save_splits(train_df, val_df, test_df)
        
        # Log dataset artifact for quick inspection purpose
        train_sample = train_df.head(100)
        train_sample.to_csv(r"data\processed\customer_fraud_detection\train_sample.csv", index=False)
        mlflow.log_artifact(r"data\processed\customer_fraud_detection\train_sample.csv", artifact_path="data_samples")

        # Log dataset metadata
        train_dataset = mlflow.data.from_pandas(  # type: ignore
            train_df,
            source=split_paths['train_path'],
            targets="Class",
            name="fraud_detection_train"
        )
        mlflow.log_input(train_dataset, context="training")
        
        val_dataset = mlflow.data.from_pandas( # type: ignore
            val_df,
            source=split_paths['val_path'],
            targets="Class",
            name="fraud_detection_val"
        )
        mlflow.log_input(val_dataset, context="validation")
        
        # Save and log data statistics as json artifact
        stats_extended = {
            **stats,
            'split_strategy': 'temporal',
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'split_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(r"data\processed\customer_fraud_detection\data_stats.json","w") as f:
            json.dump(stats_extended, f, indent=4)
            
        mlflow.log_artifact(r"data\processed\customer_fraud_detection\data_stats.json", artifact_path="metadata")
        print(f"MLFlow run completed\n")
        print(f"Experiment: {mlflow.get_experiment(run.info.experiment_id).name}")

        return run.info.run_id
    
def train_and_log_model(
    model_name, model, 
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_names,
    tags=None
    ):
    """
    Train a model and log everything to MLflow
    
    This is the core MLOps function that:
    1. Trains the model
    2. Evaluates on validation and test sets
    3. Creates visualizations
    4. Logs params, metrics, artifacts to MLflow
    5. Saves the model with signature for deployment
    
    Parameters:
    model_name : str
        Name for the MLflow run
        
    model : XGBoostFraudDetector or LightGBMFraudDetector
        Model instance (not yet trained)
        
    X_train, y_train : Training data
    X_val, y_val : Validation data
    X_test, y_test : Test data (held out for final evaluation)
    
    feature_names : list
        List of feature names
        
    tags : dict, optional
        Additional tags for MLflow run
    
    Returns:
    dict : Dictionary with run_id and key metrics
    """
    # Enable autologging based on model type
    if isinstance(model, XGBoostFraudDetector):
        mlflow.xgboost.autolog(log_models=False) # type: ignore
    elif isinstance(model, LightGBMFraudDetector):
        mlflow.lightgbm.autolog(log_models=False) # type: ignore
        
    # Start mlflow run
    with mlflow.start_run(run_name=model_name) as run:
        print(f"MLflow Run: {model_name}")
        
        # Step 1: Log tags for organizational management
        mlflow.set_tag("stage", "model_training")
        mlflow.set_tag("model_type", type(model).__name__) # __name__ returns the name in str
        mlflow.set_tag("sampling_method","SMOTE" if model.use_smote else "class_weights")
        
        # In case tags come as json file
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Step 2: Log parameters (hyperparameters and configurations)
        mlflow.log_param("use_smote", model.use_smote)
        mlflow.log_param("random_state",model.random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Step 3: train the model
        print("Training ...")
        model.train(X_train,y_train, X_val, y_val)
        
        # Step 4: evaluate on validation set
        print("Validating on validation set ...")
        y_val_pred = model.predict(X_val)
        ## Probability of fraud
        y_val_pred_proba = model.predict_proba(X_val)[:,1] # type: ignore 
        
        ## Calculate all metrics
        val_metrics = calculate_metrics(
            y_val,
            y_val_pred,
            y_val_pred_proba,
            fn_cost = 150,
            fp_cost = 5,
            tp_cost = 2,
            tn_cost = 0
        )
        
        ## Log validation metrics to mlflow
        mlflow.log_metrics({
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1_score": val_metrics['f1_score'],
            "val_accuracy": val_metrics['accuracy'],
            "val_roc_auc": val_metrics['roc_auc'],
            "val_pr_auc": val_metrics['pr_auc'],
            "val_matthews_corr": val_metrics['matthews_corr'],
            "val_total_cost": val_metrics['total_cost'],
            "val_cost_per_transaction": val_metrics['cost_per_transaction'],
            "val_savings": val_metrics['savings'],
            "val_savings_percentage": val_metrics['savings_percentage']            
        })
        
        ## Log confusion matrix components
        mlflow.log_metrics({
            "val_true_negatives": val_metrics['confusion_matrix']['tn'],
            "val_false_positives": val_metrics['confusion_matrix']['fp'],
            "val_false_negatives": val_metrics['confusion_matrix']['fn'],
            "val_true_positives": val_metrics['confusion_matrix']['tp']
        })
        
        # Step 5: create and log visualizations (artifacts)
        ## by artifacts, i mean files, photos, models, data, etc.
        os.makedirs("outputs", exist_ok=True)
        
        ## Consusion matrix
        cm_path = rf"outputs\{model_name}_confusion_matrix_val.png"
        plot_confusion_matrix(y_val, y_val_pred, save_path=cm_path)
        mlflow.log_artifact(cm_path, artifact_path='plots')
        
        ## Precision-recall curve
        pr_path = rf"outputs\{model_name}_pr_curve_val.png"
        plot_precision_recall_curve(y_val, y_val_pred_proba, save_path=pr_path)
        mlflow.log_artifact(pr_path, artifact_path='plots')
        
        ## Threshold optimization
        threshold_path = rf"outputs\{model_name}_threshold_optimization.png"
        threshold_results = find_optimal_threshold(
            y_val, y_val_pred_proba,
            fn_cost=150, fp_cost=5, tp_cost=2, tn_cost=0,
            save_path=threshold_path            
        )
        mlflow.log_artifact(threshold_path, artifact_path='plots')
        
        ## Log optimal threshold results
        mlflow.log_metrics({
            "optimal_threshold": threshold_results['optimal_threshold'],
            "optimal_threshold_cost": threshold_results['optimal_cost'],
            "default_threshold_cost": threshold_results['default_cost'],
            "threshold_savings": threshold_results['savings']
        })
        
        # Step 6: evaluate on test set for final evaluation
        print("Test set evaluation ...")
        ## Use optimal threshold from validation set
        y_test_pred_proba = model.predict_proba(X_test)[:,1] # type: ignore
        y_test_pred_optimal = (y_test_pred_proba >= threshold_results["optimal_threshold"]).astype(int)
        
        ## Calculate test metrics with optimal threshold
        test_metrics = calculate_metrics(
            y_test,
            y_test_pred_optimal,
            y_test_pred_proba,
            fn_cost=150, fp_cost=5, tp_cost=2, tn_cost=0
        )
        
        ## Log test metrics
        mlflow.log_metrics({
            "test_precision": test_metrics['precision'],
            "test_recall": test_metrics['recall'],
            "test_f1_score": test_metrics['f1_score'],
            "test_accuracy": test_metrics['accuracy'],
            "test_roc_auc": test_metrics['roc_auc'],
            "test_pr_auc": test_metrics['pr_auc'],
            "test_matthews_corr": test_metrics['matthews_corr'],
            "test_total_cost": test_metrics['total_cost'],
            "test_cost_per_transaction": test_metrics['cost_per_transaction']
        })

        # Test set visualizations
        cm_test_path = f'outputs/{model_name}_confusion_matrix_test.png'
        plot_confusion_matrix(y_test, y_test_pred_optimal, save_path=cm_test_path)
        mlflow.log_artifact(cm_test_path, artifact_path='plots')
        
        pr_test_path = f'outputs/{model_name}_pr_curve_test.png'
        plot_precision_recall_curve(y_test, y_test_pred_proba, save_path=pr_test_path)
        mlflow.log_artifact(pr_test_path, artifact_path='plots')
        
        # Step 7: LOG MODEL FOR DEPLOYMENT
        print("Saving model ...")
        
        ## Create model signature. This defines the expected input/output schema for the model
        signature = infer_signature(X_train, model.predict(X_train))
        
        ## Create input example for documentation purposes
        input_example = X_train[:5]
        
        ## Log model with signature
        if isinstance(model, XGBoostFraudDetector):
            mlflow.xgboost.log_model( # type: ignore
                model.model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=None
            )
        elif isinstance(model, LightGBMFraudDetector):
            mlflow.lightgbm.log_model( # type: ignore
                model.model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=None
            )
        
        print(f"Model saved with signature")
        
        # Step 8: log summary       
        summary = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'sampling_method': "SMOTE" if model.use_smote else "class_weights",
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'optimal_threshold': threshold_results['optimal_threshold'],
            'run_id': run.info.run_id,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = rf'outputs\{model_name}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        mlflow.log_artifact(summary_path, artifact_path='metadata')
        
      
        print(f"TRAINING COMPLETE: {model_name}")
     
        print(f"Run ID: {run.info.run_id}")
        print(f"\nKey Metrics (Validation Set):")
        print(f"  Precision:     {val_metrics['precision']:.4f}")
        print(f"  Recall:        {val_metrics['recall']:.4f}")
        print(f"  F1-Score:      {val_metrics['f1_score']:.4f}")
        print(f"  PR-AUC:        {val_metrics['pr_auc']:.4f}")
        print(f"  Business Cost: ${val_metrics['total_cost']:,.2f}")
        print(f"\nKey Metrics (Test Set):")
        print(f"  Precision:     {test_metrics['precision']:.4f}")
        print(f"  Recall:        {test_metrics['recall']:.4f}")
        print(f"  F1-Score:      {test_metrics['f1_score']:.4f}")
        print(f"  PR-AUC:        {test_metrics['pr_auc']:.4f}")
        print(f"  Business Cost: ${test_metrics['total_cost']:,.2f}")
        print(f"{'='*80}\n")
        
        return {
            'run_id': run.info.run_id,
            'val_pr_auc': val_metrics['pr_auc'],
            'test_pr_auc': test_metrics['pr_auc'],
            'val_cost': val_metrics['total_cost'],
            'test_cost': test_metrics['total_cost']
        }
    
def train_baseline_models():
    """
    Train all baseline models and log to MLflow

    Models:
    1. XGBoost with class weights
    2. XGBoost with SMOTE
    3. LightGBM with class weights

    Returns:
    dict : Results from all model runs
    """

    print("Baseline model training ...")

    # Load processed data
    print(" Loading Preprocessed Data")
    data_loader = FraudDataLoader(random_state=42)

    # Load splits
    train_df = pd.read_csv('data/processed/train_df.csv')
    val_df = pd.read_csv('data/processed/val_df.csv')
    test_df = pd.read_csv('data/processed/test_df.csv')

    print(f"Train set: {len(train_df):,} samples")
    print(f"Val set:   {len(val_df):,} samples")
    print(f"Test set:  {len(test_df):,} samples")

    # Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_features(
        train_df, val_df, test_df
    )

    feature_names = X_train.columns.tolist()

    print(f"Data loaded and preprocessed")
    print(f"\nFeatures: {len(feature_names)}")

    # Train models
    results = {}

    ## Model 1: xgboost with class weights as baseline
    print("Model 1: XGBoost with Class Weights")

    xgb_baseline = XGBoostFraudDetector(
        use_smote=False,
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    results['xgb_baseline'] = train_and_log_model(
        model_name="02_xgboost_baseline_class_weights",
        model=xgb_baseline,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        tags={
            "algorithm": "XGBoost",
            "imbalance_technique": "class_weights",
            "baseline": "true"
        }
    )

    # Model 2: xgboost with smote
    print("Model 2: XGBoost with SMOTE")

    xgb_smote = XGBoostFraudDetector(
        use_smote=True,
        smote_strategy='auto',
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    results['xgb_smote'] = train_and_log_model(
        model_name="03_xgboost_smote",
        model=xgb_smote,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        tags={
            "algorithm": "XGBoost",
            "imbalance_technique": "SMOTE",
            "baseline": "false"
        }
    )

    # Model 3: lightgbm with class weights
    print("Model 3: LightGBM with Class Weights")

    lgb_baseline = LightGBMFraudDetector(
        use_smote=False,
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    results['lgb_baseline'] = train_and_log_model(
        model_name="04_lightgbm_baseline_class_weights",
        model=lgb_baseline,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        tags={
            "algorithm": "LightGBM",
            "imbalance_technique": "class_weights",
            "baseline": "true"
        }
    )

    # Compare results
    print("Model Comparison Summary")

    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Val PR-AUC': [results[k]['val_pr_auc'] for k in results],
        'Test PR-AUC': [results[k]['test_pr_auc'] for k in results],
        'Val Cost': [results[k]['val_cost'] for k in results],
        'Test Cost': [results[k]['test_cost'] for k in results]
    })

    print("\n")
    print(comparison_df.to_string(index=False))

    # Find best model
    best_model = comparison_df.loc[comparison_df['Test PR-AUC'].idxmax()]

    print(f"Best Model (by Test PR-AUC): {best_model['Model']}")
    print(f"Test PR-AUC:   {best_model['Test PR-AUC']:.4f}")
    print(f"Test Cost:     ${best_model['Test Cost']:,.2f}")

    return results       
        
if __name__ == "__main__":
    # Step 1: initialize mlflow
    print("CREDIT CARD FRAUD DETECTION - MLFLOW PIPELINE")

    experiment_name = initialize_mlflow()
    print(f"Starting mlflow experiment: {experiment_name}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")
    
    # Step 2: Check if data loading run already exists
    print("Checking for existing data loading run")
    
    if not os.path.exists('data/processed/train_df.csv'):
        print("Data not found. Running data loading pipeline...")
        run_id = log_data_loading_run()
        print(f"\nData loading complete!..")
    else:
        print("Preprocessed data found. Skipping data loading.")

    # Step 3: Train baseline models
    results = train_baseline_models()
    print("Training complete!!!")
