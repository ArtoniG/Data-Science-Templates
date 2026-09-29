"""
ml_pipeline/train.py

Production Training Entrypoint for Credit Risk Models.
This script consumes the custom `credit_toolbox` SDK to build, train, audit, 
and serialize the scoring pipeline. 

Crucially, it relies on Phase 6 Quality Gates to guarantee enterprise compliance 
before exporting the deterministic JSON state (zero pickle files).

Context: Built for batch execution in CI/CD (e.g., GitHub Actions -> Vertex AI).
Date: 2026-09-28 | Armenia, Quindio, Colombia
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# SDK Imports - Consuming the modules built in previous steps
from credit_toolbox.core.exceptions import QualityGateError, CreditToolboxError
from credit_toolbox.transformers.missing_imputer import MissingImputer
from credit_toolbox.transformers.outlier_capper import OutlierCapper
from credit_toolbox.transformers.woe_encoder import WOEEncoder
from credit_toolbox.governance.quality_gates import PipelineAuditor
from credit_toolbox.governance.model_card import ArtifactExporter

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration defining features, targets, and hyperparameters."""
    logger.info(f"Loading training configuration from {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the training dataset. 
    In production, this might read directly from BigQuery/dbt outputs.
    """
    logger.info(f"Loading training data from {data_path}")
    # Using parquet for type safety and compression in enterprise pipelines
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Constructs the strict Scikit-Learn pipeline using Phase 4 stateful transformers.
    """
    logger.info("Constructing credit risk pipeline topology...")
    
    features = config["features"]
    
    # Standard enterprise scorecard topology
    pipeline = Pipeline(steps=[
        ("imputer", MissingImputer(columns=features, strategy=config["imputation_strategy"])),
        ("capper", OutlierCapper(columns=features, lower_quantile=0.01, upper_quantile=0.99)),
        ("woe_encoder", WOEEncoder(columns=features)),
        ("classifier", LogisticRegression(
            penalty=config["logistic_regression"]["penalty"],
            C=config["logistic_regression"]["C"],
            class_weight="balanced",
            random_state=42,
            max_iter=500
        ))
    ])
    
    return pipeline


def main(config_path: str):
    """Main training orchestration."""
    try:
        # 1. Setup
        config = load_config(config_path)
        model_name = config["model_metadata"]["name"]
        artifacts_dir = Path(config["model_metadata"]["artifacts_dir"])
        
        X_train, y_train = load_data(
            data_path=config["data"]["train_path"], 
            target_col=config["data"]["target"]
        )

        # 2. Pipeline Construction & Training
        pipeline = build_pipeline(config)
        logger.info("Fitting pipeline on training data...")
        pipeline.fit(X_train, y_train)

        # 3. Phase 6 Governance: Quality Gates Enforcement
        logger.info("Executing Phase 6 Quality Gates...")
        auditor = PipelineAuditor(strict_mode=True)
        auditor.audit(pipeline, X_train, y_train)
        
        # 4. Serialization & Documentation
        logger.info("Quality Gates passed. Exporting declarative state and model card...")
        exporter = ArtifactExporter(
            pipeline=pipeline, 
            model_name=model_name, 
            output_dir=str(artifacts_dir)
        )
        exporter.export_all()
        
        logger.info(f"Training complete. Artifacts saved to {artifacts_dir}/")
        
    except QualityGateError as qe:
        logger.error(f"MODEL REJECTED BY GOVERNANCE: {qe}")
        sys.exit(1) # Fail the CI/CD pipeline immediately
    except CreditToolboxError as ce:
        logger.error(f"SDK Error encountered: {ce}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected training failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Credit Risk Model Training Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="ml_pipeline/config.yaml",
        help="Path to the model configuration YAML file."
    )
    
    args = parser.parse_args()
    main(args.config)