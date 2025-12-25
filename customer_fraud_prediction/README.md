Experiment: Fraud_Detection_Production
- Run: 01_data_loading_and_split - phase 1
- Run: 02_xgboost_baseline_class_weights - phase 2
- Run: 03_xgboost_smote - phase 2
- Run: 04_lightgbm_baseline - phase 2
- Run: 05_lightgbm_threshold_optimized - phase 2

Each run logs:
- Parameters: All hyperparameters, sampling strategy
- Metrics: 10+ metrics including business cost
- Artifacts: 3+ plots, trained model
- Tags: model_type, sampling_method, stage
