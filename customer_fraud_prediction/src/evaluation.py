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
    
    # STEP 1: CONFUSION MATRIX CALCULATION
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
    
    # STEP 2: STANDARD CLASSIFICATION METRICS
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
    
    # STEP 3: IMBALANCED-DATA-SPECIFIC METRICS
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
    
    # STEP 4: BUSINESS METRICS (COST-SENSITIVE LEARNING)
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

    # STEP 5: RETURN ALL METRICS AS DICTIONARY
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

def plot_precision_recall_curve(y_true,y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curve - THE key visualization for imbalanced classification
    
    The PR curve shows the tradeoff between precision (accuracy of fraud flags) and
    recall (percentage of frauds caught) at different classification thresholds.
    
    For imbalanced fraud detection (0.17% fraud rate):
    - A random classifier would have PR-AUC ≈ 0.0017 (the fraud rate)
    - A perfect classifier would have PR-AUC = 1.0
    - Real models typically achieve PR-AUC between 0.3 and 0.8
    
    Parameters:
    y_true : array-like
        True binary labels (0 = legitimate, 1 = fraud)
        
    y_pred_proba : array-like
        Predicted probabilities for positive class (fraud)
        
    save_path : str, optional
        Path to save the plot
    
    Returns:
    str : Path where plot was saved
    
    Notes:
    The PR curve is MORE informative than ROC curve for imbalanced data because:
    - ROC uses False Positive RATE (FP / (FP + TN)) - misleading when TN is huge
    - PR uses actual False Positive count in precision calculation
    - PR focuses on minority class performance
    """
    # Calculate precision and recall at different thresholds
    precision,recall,thresholds = precision_recall_curve(y_true,y_pred_proba)
    
    # Calculate PR-AUC (Average precision)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Calculate basesline
    fraud_rate = np.mean(y_pred_proba)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve
    plt.plot(
        recall, precision,
        color='blue',
        lw=3,
        label=f'Model (PR-AUC = {pr_auc:.4f})'
    )
    
    # Plot baseline (random classifier)
    # For imbalanced data, random classifier PR-AUC ≈ fraud_rate
    plt.plot(
        [0, 1], [fraud_rate, fraud_rate],
        color='red',
        lw=2,
        linestyle='--',
        label=f'Random Classifier (Baseline = {fraud_rate:.4f})'
    )
    
    # Plot iso-F1 curves (curves of constant F1-score)
    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.3, linestyle=':')
    
    plt.annotate('Iso-F1 curves', xy=(0.9, 0.9), xytext=(0.7, 0.85),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Fraud Detection Rate)', fontsize=14, fontweight='bold')
    plt.ylabel('Precision (Accuracy of Flags)', fontsize=14, fontweight='bold')
    plt.title('Precision-Recall Curve - Fraud Detection\nHigher PR-AUC = Better Model', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add interpretation box
    interpretation = (
        f"Model Performance:\n"
        f"• PR-AUC: {pr_auc:.4f}\n"
        f"• Baseline: {fraud_rate:.4f} (random)\n"
        f"• Improvement: {(pr_auc/fraud_rate):.1f}x better than random\n\n"
        f"Interpretation:\n"
        f"• Higher curve = better model\n"
        f"• Top-right corner = ideal (100% precision & recall)\n"
        f"• Trade-off: High recall → more false alarms"
    )
    
    plt.text(
        0.02, 0.02, interpretation,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return save_path

def find_optimal_threshold(y_true, y_pred_proba, 
                          fn_cost=150, fp_cost=5, tp_cost=2, tn_cost=0,
                          save_path=None):
    """
    Find optimal classification threshold by minimizing business cost
    
    The default threshold of 0.5 is rarely optimal for imbalanced classification.
    This function finds the threshold that minimizes total business cost by:
    1. Testing 1000 different thresholds (0.001 to 0.999)
    2. Calculating business cost at each threshold
    3. Returning the threshold with minimum cost
    
    Parameters:
    y_true : array-like
        True binary labels
        
    y_pred_proba : array-like
        Predicted probabilities for positive class
        
    fn_cost, fp_cost, tp_cost, tn_cost : float
        Business costs for each outcome type
        
    save_path : str, optional
        Path to save the threshold optimization plot
    
    Returns:
    dict : Contains optimal_threshold, optimal_cost, cost_at_default (0.5)
    
    Notes:
    For fraud detection with extreme imbalance:
    - Default threshold (0.5) often catches <30% of frauds
    - Optimal threshold is typically much lower (0.05 to 0.25)
    - Lower threshold = more false alarms but fewer missed frauds
    - Goal: Minimize total cost, not maximize F1-score
    """
    
    # Test range of thresholds
    thresholds = np.linspace(0.001, 0.999, 1000)
    costs = []
    precisions = []
    recalls = []
    f1_scores = []
    
    print("Threshold Optimization")
    print("Testing 1000 thresholds to minimize business cost...")
    
    # Calculate metrics at each threshold
    for threshold in thresholds:
        # Apply threshold to get predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate business cost
        cost = (fn * fn_cost) + (fp * fp_cost) + (tp * tp_cost) + (tn * tn_cost)
        costs.append(cost)
        
        # Calculate metrics for plotting
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Find optimal threshold (minimum cost)
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]
    
    # Calculate cost at default threshold (0.5)
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    default_cost = costs[default_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_true, y_pred_optimal)
    tn, fp, fn, tp = cm_optimal.ravel()
    
    print(f"\nOptimization Complete!")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  Cost at Optimal:   ${optimal_cost:,.2f}")
    print(f"  Cost at Default (0.5): ${default_cost:,.2f}")
    print(f"  Savings: ${default_cost - optimal_cost:,.2f} ({((default_cost - optimal_cost)/default_cost)*100:.1f}%)")
    print(f"\n  At Optimal Threshold:")
    print(f"    Precision: {precisions[optimal_idx]:.4f}")
    print(f"    Recall:    {recalls[optimal_idx]:.4f}")
    print(f"    F1-Score:  {f1_scores[optimal_idx]:.4f}")
    print(f"    Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Create visualization
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cost vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(thresholds, costs, 'b-', linewidth=2, label='Total Cost')
        ax1.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal: {optimal_threshold:.3f}')
        ax1.axvline(0.5, color='orange', linestyle='--', linewidth=2, 
                   label='Default: 0.500')
        ax1.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Business Cost ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Threshold vs Business Cost', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Precision & Recall vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
        ax2.plot(thresholds, recalls, 'b-', linewidth=2, label='Recall')
        ax2.plot(thresholds, f1_scores, 'purple', linewidth=2, label='F1-Score')
        ax2.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
        ax2.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Plot 3: Confusion Matrix Components vs Threshold
        ax3 = axes[1, 0]
        
        # Recalculate for all components
        tps, fps, fns, tns = [], [], [], []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            tns.append(tn)
        
        ax3.plot(thresholds, tps, label='True Positives (Caught Frauds)', linewidth=2)
        ax3.plot(thresholds, fps, label='False Positives (False Alarms)', linewidth=2)
        ax3.plot(thresholds, fns, label='False Negatives (Missed Frauds)', linewidth=2)
        ax3.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax3.set_title('Confusion Matrix Components vs Threshold', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        THRESHOLD OPTIMIZATION SUMMARY
        
        Optimal Threshold: {optimal_threshold:.4f}
        Default Threshold: 0.5000
        
        COST COMPARISON:
        • Cost at Optimal:     ${optimal_cost:>12,.2f}
        • Cost at Default:     ${default_cost:>12,.2f}
        • Savings:             ${default_cost - optimal_cost:>12,.2f}
        • Improvement:         {((default_cost - optimal_cost)/default_cost)*100:>12.1f}%
        
        PERFORMANCE AT OPTIMAL THRESHOLD:
        • Precision:           {precisions[optimal_idx]:>12.4f}
        • Recall:              {recalls[optimal_idx]:>12.4f}
        • F1-Score:            {f1_scores[optimal_idx]:>12.4f}
        
        CONFUSION MATRIX AT OPTIMAL:
        • True Positives:      {tp:>12,}
        • False Positives:     {fp:>12,}
        • False Negatives:     {fn:>12,}
        • True Negatives:      {tn:>12,}
        
        RECOMMENDATION:
        Use threshold = {optimal_threshold:.4f} in production
        to minimize business cost.
        """
        
        ax4.text(0.1, 0.5, summary_text, 
                fontsize=11, 
                family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Threshold Optimization for Fraud Detection', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Threshold optimization plot saved to: {save_path}")
        plt.close()
    
    return {
        'optimal_threshold': float(optimal_threshold),
        'optimal_cost': float(optimal_cost),
        'default_cost': float(default_cost),
        'savings': float(default_cost - optimal_cost),
        'savings_percentage': float((default_cost - optimal_cost) / default_cost * 100),
        'precision_at_optimal': float(precisions[optimal_idx]),
        'recall_at_optimal': float(recalls[optimal_idx]),
        'f1_at_optimal': float(f1_scores[optimal_idx])
    }
        
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
