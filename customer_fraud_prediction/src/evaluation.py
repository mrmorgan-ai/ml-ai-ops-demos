import numpy as np
import pandas as pd
from  sklearn.metrics import(
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_pred_proba,
                      fn_cost=150,fp_cost=5,tp_cost=2,tn_cost=0):
    """
    Calculate evaluation metrics for imbalanced classification.
    This function calculate both standard ML and business metrics for fraud detection
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0 = legitimate, 1 = fraud)
        
    y_pred : array-like, shape (n_samples,)
        Predicted binary labels (0 or 1)
        
    y_pred_proba : array-like, shape (n_samples,)
        Predicted probabilities for positive class (fraud)
        Used for ROC-AUC and PR-AUC calculations
        
    fn_cost : float, default=150
        Cost of False Negative (missed fraud)
        Represents average fraud loss per transaction
        
    fp_cost : float, default=5
        Cost of False Positive (false alarm)
        Represents investigation cost + customer friction
        
    tp_cost : float, default=2
        Cost of True Positive (correctly caught fraud)
        Represents investigation cost when catching real fraud
        
    tn_cost : float, default=0
        Cost of True Negative (correctly identified legitimate)
        Usually 0 as no action is taken
    
    Returns:
    dict : Dictionary containing all evaluation metrics
        Keys include: confusion_matrix, precision, recall, f1_score, accuracy,
                     roc_auc, pr_auc, matthews_corr, total_cost, cost_per_transaction
    """
    
    # ============================================================================
    # STEP 1: CONFUSION MATRIX CALCULATION
    # ============================================================================
    cm = confusion_matrix(y_true, y_pred)

    # Extract all values from confusion matrix
    # ravel() flattens 2D array to 1D: [[TN, FP], [FN, TP]] → [TN, FP, FN, TP]
    tn,fp,fn,tp = cm.ravel()

    # Handle properly data type for cm values. 
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    print(f"\nConfusion Matrix")
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")
    
     
    # ============================================================================
    # STEP 2: STANDARD CLASSIFICATION METRICS
    # ============================================================================
    # Precision
    precision = precision_score(y_true,y_pred,zero_division=0)
    
    # Recall
    recall = recall_score(y_true,y_pred,zero_division=0)
    
    # F1 Score
    f1 = f1_score(y_true,y_pred,zero_division=0)
    
    # Accuracy
    accuracy = accuracy_score(y_true,y_pred)
    
    print(f"\nStandard Metrics")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f} (misleading for imbalanced data)")
    
    # ============================================================================
    # STEP 3: IMBALANCED-DATA-SPECIFIC METRICS
    # ============================================================================
    # ROC-AUC 
    try:
        roc_auc = roc_auc_score(y_true,y_pred_proba)
    except ValueError:
        roc_auc = 0.0
        print("Warning: ROC-AUC undefined (Only one class in y_true)")
        
    # PR-AUC
    try:
        pr_auc = average_precision_score(y_true,y_pred_proba)
    except ValueError:
        pr_auc = 0.0
        print("Warning: PR-AUC undefined (Only one class in y_true)")    

    # Matthews Correlation
    mcc = matthews_corrcoef(y_true,y_pred)

    print(f"\nImbalanced-Specific Metrics")
    print(f"ROC-AUC:    {roc_auc:.4f} (can be misleading)")
    print(f"PR-AUC:     {pr_auc:.4f} (better for imbalanced)")
    print(f"Matthews Correlation: {mcc:.4f}")
    
    # ============================================================================
    # STEP 4: BUSINESS METRICS (COST-SENSITIVE LEARNING)
    # ============================================================================
    # Calculate total cost across all transactions
    total_cost = (
        (fn * fn_cost) +  # Missed fraud losses
        (fp * fp_cost) +  # False alarm costs
        (tp * tp_cost) +  # Investigation costs for caught fraud
        (tn * tn_cost)    # No cost for correct negatives
    )

    # Normalize by number of transactions
    cost_per_transaction = total_cost/len(y_true)
    
    # Calculate expected savings vs "caught no frauds" baseline
    total_frauds = int(np.sum(y_true))
    baseline_cost = total_frauds*fn_cost
    savings = baseline_cost - total_cost
    savings_percentage = (savings/baseline_cost)*100 if baseline_cost > 0 else 0
    
    print(f"\nBusiness Metrics")
    print(f"Total Cost:        ${total_cost:,.2f}")
    print(f"Cost per Txn:      ${cost_per_transaction:,.4f}")
    print(f"Baseline Cost:     ${baseline_cost:,.2f}")
    print(f"Savings:           ${savings:,.2f} ({savings_percentage:.1f}%)")
    print(f"\nCost Breakdown:")
    print(f"Missed Frauds (FN):     {fn:,} * ${fn_cost} = ${fn * fn_cost:,.2f}")
    print(f"False Alarms (FP):      {fp:,} * ${fp_cost} = ${fp * fp_cost:,.2f}")
    print(f"Caught Frauds (TP):     {tp:,} * ${tp_cost} = ${tp * tp_cost:,.2f}")

    # ============================================================================
    # STEP 5: RETURN ALL METRICS AS DICTIONARY
    # ============================================================================
    return {
        # Confusion Matrix Components
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        },
        
        # Standard Classification Metrics
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        
        # Imbalanced-Specific Metrics
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'matthews_corr': float(mcc),
        
        # Business Metrics
        'total_cost': float(total_cost),
        'cost_per_transaction': float(cost_per_transaction),
        'baseline_cost': float(baseline_cost),
        'savings': float(savings),
        'savings_percentage': float(savings_percentage),
        
        # Additional Context
        'total_samples': len(y_true),
        'total_frauds': total_frauds,
        'fraud_rate': float(np.mean(y_true))
    }
    
def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=False):
    """
    Create a Confusion Matrix Headmap for fraud detection
    This visualization is critical for imbalanced classification because it shows
    the exact breakdown of predictions. For fraud detection, we're especially
    interested in False Negatives (missed frauds) and False Positives (false alarms).
    
    Parameters:
    y_true : array-like
        True binary labels (0 = legitimate, 1 = fraud)
        
    y_pred : array-like
        Predicted binary labels
        
    save_path : str, optional
        Path to save the plot. If None, plot is displayed but not saved.
        In MLflow context, always provide a path to log as artifact.
        
    normalize : bool, default=False
        If True, normalize confusion matrix to show proportions
        Useful for imbalanced data to see percentages
    
    Returns:
    str : Path where plot was saved (for MLflow logging)
    
    Notes:
    For imbalanced data:
    - Raw counts: Shows actual numbers (TN will be huge, TP will be tiny)
    - Normalized: Shows percentages (better for understanding model behavior)
    
    We use both in the same plot: raw counts as main values, percentages in parentheses
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true,y_pred)
    
    # Calculate normalized version
    cm_normalized = cm.astype('float')/(cm.sum(axis=1)[:,np.newaxis])
    
    # Create figure
    plt.figure(figsize=(10,8))
    
    # Create annotations combining counts and percentages
    # Format: "COUNT\n(PERCENTAGE%)"
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_normalized[i, j] * 100
            annotations[i, j] = f'{count:,}\n({percentage:.2f}%)'
    
    # Create heatmap using seaborn
    # 'Blues' colormap: darker = more samples
    sns.heatmap(
        cm if not normalize else cm_normalized,
        annot=annotations,
        fmt='',  # Don't format (we already formatted in annotations)
        cmap='Blues',
        square=True,
        cbar=True,
        xticklabels=['Legitimate', 'Fraud'],
        yticklabels=['Legitimate', 'Fraud'],
        linewidths=2,
        linecolor='white'
    )
    
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix - Fraud Detection\n(Count and Percentage)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add interpretation box for stakeholders
    tn, fp, fn, tp = cm.ravel()
    interpretation = (
        f"Model Performance Summary:\n"
        f"Caught {tp:,} frauds (True Positives)\n"
        f"Missed {fn:,} frauds (False Negatives) ← Critical!\n"
        f"{fp:,} false alarms (False Positives)\n"
        f"{tn:,} correct legitimate txns"
    )
    
    plt.text(
        0.5, -0.25, interpretation,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return save_path
    
if __name__ == "__main__":
    print("Test calculation metrics") 
    
    np.random.seed(42)
    n_samples = 1000
    n_frauds = 5
    
    y_true = np.array([1]*n_frauds + [0]*(n_samples - n_frauds))
    
    # Simulate model prediction probability
    y_pred_proba = np.random.rand(n_samples)
    y_pred_proba[:n_frauds] += 0.6
    y_pred_proba = np.clip(y_pred_proba, 0, 1) # Keep values between 0 and 1: lower becomes 0, upper becomes 1

    # Apply threshold for prediction
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print(f"\nCalculated metrics")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"{subkey}: {subvalue}")

        print(f"{key}: {value}")
