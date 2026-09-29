"""
ml_pipeline/feature_selection.py

Automated Feature Selection for Credit Risk.
Uses Information Value (IV) to drop weak predictors and Pearson correlation 
to remove multicollinearity, updating config.yaml automatically.
"""

import logging
import yaml
import pandas as pd
import numpy as np

# SDK Import
from credit_toolbox.metrics import calculate_iv

logger = logging.getLogger(__name__)

def select_features(data_path: str, target_col: str, config_path: str) -> None:
    logger.info("Starting automated feature selection...")
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 1. Information Value (IV) Filtering
    iv_results = calculate_iv(X, y)
    
    # Bureau standards: IV < 0.02 is useless, IV > 0.5 is suspicious (leakage)
    selected_by_iv = {
        feat: iv for feat, iv in iv_results.items() 
        if 0.02 <= iv <= 0.50
    }
    logger.info(f"Features passing IV thresholds: {len(selected_by_iv)}")
    
    # 2. Multicollinearity Filter (Correlation Matrix)
    # Keep the feature with the higher IV when two features are highly correlated
    corr_matrix = X[list(selected_by_iv.keys())].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper_tri.columns:
        highly_correlated = upper_tri[col][upper_tri[col] > 0.75].index.tolist()
        for correlated_col in highly_correlated:
            if selected_by_iv[col] < selected_by_iv[correlated_col]:
                to_drop.add(col)
            else:
                to_drop.add(correlated_col)

    final_features = [f for f in selected_by_iv.keys() if f not in to_drop]
    logger.info(f"Final feature count after correlation pruning: {len(final_features)}")

    # 3. Update config.yaml dynamically
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["features"] = final_features
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    logger.info("config.yaml successfully updated with optimized feature set.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    select_features(
        data_path="data/processed/train_batch.parquet",
        target_col="default_flag",
        config_path="ml_pipeline/config.yaml"
    )