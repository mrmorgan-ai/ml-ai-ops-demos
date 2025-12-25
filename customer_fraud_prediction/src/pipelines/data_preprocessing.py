import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FraudDataLoader:
    def __init__(self, data_path, random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        df = pd.read_csv(self.data_path, header=0)
        print(f"Loaded {df.shape[0]} transactions")
        return df

    def get_data_stats(self, df) -> Dict:
        stats = {
            "total_transactions": len(df),
            "fraud_count": int(df["Class"].sum()),
            "fraud_rate": float(df["Class"].mean()),
            "legitimate_count": int((df["Class"] == 0).sum()),
            "avg_transaction_amount": float(df["Amount"].mean()),
            "max_transaction_amount": float(df["Amount"].max()),
            "min_transaction_amount": float(df["Amount"].min()),
            "time_span_hours": float((df["Time"].max() - df["Time"].min()) / 3600),
            "features_count": len(df.columns) - 1,  # Exclude target
        }

        # Fraud stats
        fraud_df = df[df["Class"] == 1]
        stats["avg_fraud_amount"] = (
            float(fraud_df["Amount"].mean()) if len(fraud_df) > 0 else 0
        )
        stats["median_fraud_amount"] = (
            float(fraud_df["Amount"].median()) if len(fraud_df) > 0 else 0
        )

        return stats

    def temporal_train_test_split(
        self,
        df,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
    ):
        """
        CRITICAL: Use temporal split, not random split for fraud detection

        Why: Fraudsters evolve tactics over time. Random split leaks future info.
        """
        df_sorted = df.sort_values("Time").reset_index(drop=True)

        n = len(df_sorted)
        # Last row in train set
        train_end = int(n * train_size)
        # Last row in validation set
        val_end = int(n * (train_size + val_size))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        print(
            f"Train: {len(train_df):,} transactions ({train_df['Class'].sum()} frauds, {train_df['Class'].mean():.4%} fraud rate)"
        )
        print(
            f"Val:   {len(val_df):,} transactions ({val_df['Class'].sum()} frauds, {val_df['Class'].mean():.4%} fraud rate)"
        )
        print(
            f"Test:  {len(test_df):,} transactions ({test_df['Class'].sum()} frauds, {test_df['Class'].mean():.4%} fraud rate)"
        )

        return train_df, val_df, test_df

    def prepare_features(self, train_df, val_df, test_df):
        """
        Prepare features and scale Amount (V1-V28 already scaled via PCA)

        Note: Only Amount needs scaling as V1-V28 are PCA components (already normalized according to source)
        Time is kept for drift detection but excluded from model features
        """
        feature_columns = [col for col in train_df if col not in ["Class", "Time"]]

        # Separate features and targets
        X_train = train_df[feature_columns].copy()
        y_train = train_df["Class"].copy() # noqa: F841

        # Separate features and targets
        X_val = val_df[feature_columns].copy()
        y_val = val_df["Class"].copy()  # noqa: F841

        # Separate features and targets
        X_test = test_df[feature_columns].copy()
        y_test = test_df["Class"].copy()  # noqa: F841

        # Scale Amount only because V1-V28 are PCA components according to source
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()

        # Fit scaler only on training data to avoid data leakage
        # This helps to maintain stats values the model learned during training phase
        X_train_scaled["Amount"] = self.scaler.fit_transform(X_train[["Amount"]])
        X_val_scaled["Amount"] = self.scaler.transform(X_val[["Amount"]])
        X_test_scaled["Amount"] = self.scaler.transform(X_test[["Amount"]])

    def save_splits(
        self,
        train_df,
        val_df,
        test_df,
        output_dir="data/processed/customer_fraud_detection",
    ):
        os.makedirs(output_dir, exist_ok=True)

        train_df.to_csv(f"{output_dir}/train_df.csv", index=False)
        val_df.to_csv(f"{output_dir}/val_df.csv", index=False)
        test_df.to_csv(f"{output_dir}/test_df.csv", index=False)

        print(f"\nSaved files to {output_dir}")

        return {
            "train_path": f"{output_dir}/train.csv",
            "val_path": f"{output_dir}/val.csv",
            "test_path": f"{output_dir}/test.csv",
        }
