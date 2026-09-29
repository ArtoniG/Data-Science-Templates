"""
ml_pipeline/drift_monitor.py

Monitors Population Stability Index (PSI) to detect data drift.
Compares fresh production data against the baseline saved in pipeline_state.json.
"""

import logging
import json
import yaml
import pandas as pd
from pathlib import Path

# SDK Import
from credit_toolbox.metrics import calculate_psi

logger = logging.getLogger(__name__)

def monitor_drift(production_data_path: str, artifacts_dir: str, config_path: str):
    logger.info("Initiating data drift monitoring...")
    
    state_file = Path(artifacts_dir) / "pipeline_state.json"
    with open(state_file, "r") as f:
        state = json.load(f)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    prod_df = pd.read_parquet(production_data_path)
    
    # Extract baseline distributions from the WOE Encoder state
    # This is why exporting the state declaratively in Phase 4 was so powerful
    woe_state = state["steps"]["woe_encoder"]["binning_logic"]
    
    drift_alerts = []
    
    for feature, baseline_bins in woe_state.items():
        if feature not in prod_df.columns:
            logger.error(f"CRITICAL: Feature '{feature}' missing from production data!")
            continue
            
        # Calculate PSI using Phase 3 toolbox function
        prod_series = prod_df[feature]
        psi_value = calculate_psi(expected_bins=baseline_bins, actual_data=prod_series)
        
        if psi_value >= config["monitoring"]["psi_alert_threshold"]:
            drift_alerts.append((feature, psi_value))
            logger.warning(f"SEVERE DRIFT DETECTED: {feature} | PSI: {psi_value:.4f}")
        elif psi_value >= config["monitoring"]["psi_warning_threshold"]:
            logger.info(f"Moderate Drift: {feature} | PSI: {psi_value:.4f}")
            
    if drift_alerts:
        # In production, this would trigger PagerDuty or send a Slack/Teams alert
        logger.error(f"Model requires retraining. {len(drift_alerts)} features breached PSI thresholds.")
    else:
        logger.info("No significant data drift detected. Model remains stable.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor_drift(
        production_data_path="data/processed/production_batch_yesterday.parquet",
        artifacts_dir="ml_pipeline/artifacts",
        config_path="ml_pipeline/config.yaml"
    )