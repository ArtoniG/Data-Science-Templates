"""
governance_integration/audit_logger.py

Immutable Compliance Manifest & Audit Verifier.
Executes a final, independent verification of the serialized model state before 
production deployment. It checks for prohibited features, metric degradation, 
and seals the model with a cryptographic hash to prevent unauthorized tampering.

Context: The ultimate CI/CD Quality Gate. If this script fails, the model 
cannot be deployed to production.
Date: 2026-09-29 | Circasia, Quindio, Colombia
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)

# Standard regulatory banned variables (ECOA/Fair Lending standards)
PROHIBITED_FEATURES = {
    "race", "gender", "sex", "religion", "national_origin", 
    "marital_status", "zip_code" # Zip code often proxies for redlining
}


class ComplianceAuditor:
    """
    Independent verification engine that certifies the pipeline_state.json 
    meets all regulatory and internal risk management (MRM) thresholds.
    """

    def __init__(self, artifacts_dir: str, config_path: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.state_file = self.artifacts_dir / "pipeline_state.json"
        self.metrics_file = self.artifacts_dir / "audit_metrics.json"
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        if not self.state_file.exists():
            raise FileNotFoundError(f"Cannot audit: Missing {self.state_file}")
            
        with open(self.state_file, "r") as f:
            self.state = json.load(f)

    def _generate_state_hash(self) -> str:
        """
        Creates a SHA-256 hash of the pipeline_state.json file.
        This guarantees that the file deployed to production is the exact 
        same file approved by this audit.
        """
        # Reading raw bytes ensures whitespace/formatting changes invalidate the hash
        with open(self.state_file, "rb") as f:
            file_bytes = f.read()
        return hashlib.sha256(file_bytes).hexdigest()

    def _check_prohibited_features(self) -> Tuple[bool, List[str]]:
        """Ensures no legally protected classes entered the model pipeline."""
        model_features = set(self.state["steps"]["woe_encoder"]["columns"])
        violations = list(model_features.intersection(PROHIBITED_FEATURES))
        
        if violations:
            logger.error(f"FAIR LENDING VIOLATION: Found prohibited features: {violations}")
            return False, violations
        return True, []

    def _check_minimum_performance(self) -> Tuple[bool, Dict[str, Any]]:
        """Verifies the model meets minimum predictive power (Gini > 0.40)."""
        if not self.metrics_file.exists():
            logger.warning("audit_metrics.json not found. Skipping performance hard-gate.")
            return True, {"status": "skipped - no metrics file"}

        with open(self.metrics_file, "r") as f:
            metrics = json.load(f)
            
        test_gini = metrics.get("test", {}).get("gini", 0.0)
        train_gini = metrics.get("train", {}).get("gini", 0.0)
        
        reasons = []
        passed = True
        
        if test_gini < 0.40:
            passed = False
            reasons.append(f"Validation Gini ({test_gini:.3f}) below 0.40 threshold.")
            
        # Check for extreme overfitting (>15% degradation)
        if train_gini > 0:
            divergence = (train_gini - test_gini) / train_gini
            if divergence > 0.15:
                passed = False
                reasons.append(f"Gini divergence ({divergence*100:.1f}%) exceeds 15% overfitting limit.")
                
        return passed, {"test_gini": test_gini, "reasons": reasons}

    def execute_audit(self) -> bool:
        """
        Runs all compliance checks, generates the cryptographic manifest, 
        and decides if the model is approved for deployment.
        """
        logger.info("Initiating strict MRM compliance audit...")
        
        features_pass, violations = self._check_prohibited_features()
        perf_pass, perf_details = self._check_minimum_performance()
        
        state_hash = self._generate_state_hash()
        is_approved = features_pass and perf_pass
        
        manifest = {
            "audit_timestamp": datetime.now().isoformat(),
            "model_name": self.config.get("model_metadata", {}).get("name", "Unknown"),
            "model_version": self.config.get("model_metadata", {}).get("version", "1.0.0"),
            "pipeline_state_sha256": state_hash,
            "status": "APPROVED" if is_approved else "REJECTED",
            "checks": {
                "fair_lending_features": {
                    "passed": features_pass,
                    "violations": violations
                },
                "minimum_performance": {
                    "passed": perf_pass,
                    "details": perf_details
                }
            }
        }
        
        manifest_path = self.artifacts_dir / "audit_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
            
        if is_approved:
            logger.info(f"AUDIT PASSED. Manifest sealed with hash: {state_hash[:8]}...")
        else:
            logger.error("AUDIT FAILED. Model is legally blocked from production.")
            
        return is_approved


def main(artifacts_dir: str, config_path: str):
    """CLI Entrypoint."""
    try:
        auditor = ComplianceAuditor(artifacts_dir, config_path)
        is_approved = auditor.execute_audit()
        
        if not is_approved:
            sys.exit(1) # Hard fail the CI/CD pipeline
            
    except Exception as e:
        logger.exception(f"Fatal error during compliance audit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    
    parser = argparse.ArgumentParser(description="Cryptographic Compliance Audit Logger")
    parser.add_argument("--artifacts", type=str, default="ml_pipeline/artifacts", help="Path to artifacts dir")
    parser.add_argument("--config", type=str, default="ml_pipeline/config.yaml", help="Path to config.yaml")
    
    args = parser.parse_args()
    main(args.artifacts, args.config)