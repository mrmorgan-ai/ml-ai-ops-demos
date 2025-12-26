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

## Evaluation Module
In production, you'll optimize threshold based on business requirements, not just F1-score.
### Concept 1: The Confusion Matrix
The confusion matrix is the source of truth for all classification metrics.

### Concept 2: Why Precision and Recall Trade Off
This is **critical** to understand for production ML:

High Threshold (e.g., prob > 0.9 to predict fraud):
Result: Very conservative flagging
- Precision: HIGH (95%+) - flags are very accurate
- Recall: LOW (30%) - miss most frauds
- Use case: When false alarms are very expensive

Low Threshold (e.g., prob > 0.1 to predict fraud):
Result: Aggressive flagging
- Precision: LOW (20%) - many false alarms
- Recall: HIGH (95%) - catch almost all frauds
- Use case: When missing fraud is very expensive

### Concept 3: PR-AUC vs ROC-AUC (CRITICAL for Imbalanced Data)
ROC-AUC (Receiver Operating Characteristic):
- Plots: True Positive Rate vs False Positive Rate
- Problem: Misleading for imbalanced data
- Why: With 99.83% negatives, a bad model can still get high ROC-AUC

PR-AUC (Precision-Recall):
- Plots: Precision vs Recall at different thresholds
- Better for imbalanced: Focuses on minority class performance
- Industry standard for fraud, medical diagnosis, anomaly detection
