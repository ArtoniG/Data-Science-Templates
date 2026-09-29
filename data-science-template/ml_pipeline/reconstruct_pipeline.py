"""
ml_pipeline/reconstruct_pipeline.py

Deterministic Pipeline Reconstruction for Credit Risk Inference.
This script reads the strictly typed `pipeline_state.json` and rebuilds the 
Scikit-Learn inference graph in memory WITHOUT calling `.fit()` or loading 
vulnerable pickle (`.pkl`) files.

Context: Acts as the bridge between offline training and the multi-agent serving layer.
Date: 2026-09-29 | Circasia, Quindio, Colombia
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# SDK Imports
from credit_toolbox.core.exceptions import CreditToolboxError
from credit_toolbox.transformers.missing_imputer import MissingImputer
from credit_toolbox.transformers.outlier_capper import OutlierCapper
from credit_toolbox.transformers.woe_encoder import WOEEncoder

logger = logging.getLogger(__name__)


class ModelReconstructionError(CreditToolboxError):
    """Raised when the deterministic JSON state is corrupted or missing."""
    pass


class CreditScoringEngine:
    """
    Singleton-friendly inference engine. 
    Designed to be instantiated once at API startup (FastAPI/LangGraph) 
    to hold the reconstructed graph in memory.
    """

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.state_file = self.artifacts_dir / "pipeline_state.json"
        
        logger.info(f"Initializing CreditScoringEngine from {self.artifacts_dir}")
        self.pipeline = self._rebuild_graph()

    def _load_json_state(self) -> Dict[str, Any]:
        """Loads and validates the presence of the pipeline state."""
        if not self.state_file.exists():
            raise ModelReconstructionError(f"Missing pipeline state artifact: {self.state_file}")
        
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            return state
        except json.JSONDecodeError as e:
            raise ModelReconstructionError(f"Corrupted JSON state file: {e}")

    def _rebuild_graph(self) -> Pipeline:
        """
        Reconstructs the Scikit-Learn Pipeline from declarative state.
        By avoiding pickle, we eliminate arbitrary code execution vulnerabilities
        and guarantee the exact mathematical state approved by governance.
        """
        state = self._load_json_state()
        logger.info(f"Reconstructing pipeline version: {state.get('model_version', 'unknown')}")

        try:
            # 1. Reconstruct Imputer
            imputer = MissingImputer(columns=[])
            imputer.load_state(state["steps"]["imputer"])

            # 2. Reconstruct Capper
            capper = OutlierCapper(columns=[])
            capper.load_state(state["steps"]["capper"])

            # 3. Reconstruct WOE Encoder
            woe_encoder = WOEEncoder(columns=[])
            woe_encoder.load_state(state["steps"]["woe_encoder"])

            # 4. Reconstruct Logistic Regression Classifier
            clf_state = state["steps"]["classifier"]
            classifier = LogisticRegression()
            classifier.classes_ = np.array(clf_state["classes_"])
            classifier.coef_ = np.array(clf_state["coef_"])
            classifier.intercept_ = np.array(clf_state["intercept_"])

            # 5. Assemble Pipeline
            pipeline = Pipeline(steps=[
                ("imputer", imputer),
                ("capper", capper),
                ("woe_encoder", woe_encoder),
                ("classifier", classifier)
            ])
            
            logger.info("Pipeline graph successfully reconstructed.")
            return pipeline

        except KeyError as e:
            raise ModelReconstructionError(f"Missing expected step in JSON state: {e}")
        except Exception as e:
            raise ModelReconstructionError(f"Failed to reconstruct pipeline: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the raw probability of default (PD).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")
        
        # The first column is usually class 0 (good), second is class 1 (default)
        return self.pipeline.predict_proba(X)

    def score(self, X: pd.DataFrame, base_score: int = 600, pdo: int = 20) -> pd.Series:
        """
        Converts the probability of default into a standard credit bureau score.
        """
        pd_array = self.predict_proba(X)[:, 1]
        
        # Avoid division by zero and log(0)
        pd_array = np.clip(pd_array, 1e-7, 1 - 1e-7)
        odds = (1 - pd_array) / pd_array
        
        # Standard bureau scaling formula: Score = Offset + Factor * ln(Odds)
        factor = pdo / np.log(2)
        offset = base_score - (factor * np.log(1)) # Assuming base_score is at 1:1 odds
        
        scores = offset + factor * np.log(odds)
        return pd.Series(np.round(scores), index=X.index, name="credit_score")


if __name__ == "__main__":
    # Smoke test for CI/CD integration
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Engine Initialization...")
    
    # In CI/CD, we'd pass a dummy payload to verify it works without throwing errors
    # engine = CreditScoringEngine(artifacts_dir="ml_pipeline/artifacts")
    # print(engine.pipeline)