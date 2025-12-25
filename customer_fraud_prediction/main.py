import json
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import pandas as pd
from src.pipelines.data_preprocessing import FraudDataLoader

def initialize_mlflow():
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
    First run: loading data and preprocessing process
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
if __name__ == "__main__":
    # Step 1
    experiment_name = initialize_mlflow()
    print(f"Starting mlflow experiment: {experiment_name}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")
    
    # Step 2
    run_id = log_data_loading_run()

    print(f"\nData loading complete!")
