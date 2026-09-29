"""
governance_integration/extract_reason_codes.py

Adverse Action & Reason Code Extractor.
Consumes the deterministic pipeline state to generate regulatory-compliant 
reason codes for credit application rejections.

Context: Designed to be imported by the multi-agent API (serving layer) or run 
as a standalone batch job for internal audits.
Date: 2026-09-29
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

# 1. Import the reconstruction engine built in Phase 1
from ml_pipeline.reconstruct_pipeline import CreditScoringEngine

# 2. Import the regulatory logic from the Toolbox SDK
from credit_toolbox.governance.reason_codes import ReasonCodeExtractor
from credit_toolbox.core.exceptions import CreditToolboxError

logger = logging.getLogger(__name__)


class AdverseActionGenerator:
    """
    Wraps the reconstructed pipeline and the SDK's ReasonCodeExtractor to 
    provide a unified interface for downstream production systems.
    """

    def __init__(self, artifacts_dir: str, mapping_file: str = "governance_integration/code_mapping.json"):
        self.artifacts_dir = Path(artifacts_dir)
        
        # Initialize the deterministic engine (Zero Pickle)
        logger.info("Initializing CreditScoringEngine for Reason Code extraction...")
        self.engine = CreditScoringEngine(artifacts_dir=str(self.artifacts_dir))
        
        # Initialize SDK Extractor passing the reconstructed Scikit-Learn graph
        self.extractor = ReasonCodeExtractor(pipeline=self.engine.pipeline)
        
        # Load human-readable mappings (Internal Feature -> Regulatory Code)
        self.code_mappings = self._load_mappings(mapping_file)

    def _load_mappings(self, mapping_file: str) -> Dict[str, str]:
        """Loads the dictionary that maps model features to legal adverse action texts."""
        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            logger.warning(f"Mapping file {mapping_file} not found. Defaulting to raw feature names.")
            return {}
        with open(mapping_path, "r") as f:
            return json.load(f)

    def format_codes(self, raw_reasons: pd.DataFrame) -> pd.DataFrame:
        """Translates raw feature names into standardized regulatory codes."""
        formatted = raw_reasons.copy()
        for col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: self.code_mappings.get(x, x))
        return formatted

    def generate(self, X: pd.DataFrame, top_k: int = 4, return_scores: bool = True) -> pd.DataFrame:
        """
        Calculates the top K reason codes for a batch of applicants.
        
        Args:
            X: DataFrame of raw applicant features.
            top_k: Number of adverse codes to return per applicant.
            return_scores: If True, appends the actual credit score to the output.
            
        Returns:
            DataFrame containing Reason_1, Reason_2, ... Reason_K (and Credit_Score).
        """
        logger.info(f"Generating top {top_k} reason codes for {len(X)} applicants.")
        
        try:
            # The SDK extractor handles the WOE drop-one / coefficient math
            raw_reasons = self.extractor.extract(X, top_k=top_k)
            mapped_reasons = self.format_codes(raw_reasons)
            
            if return_scores:
                scores = self.engine.score(X)
                # Concat the calculated score with the mapped reason codes
                return pd.concat([scores, mapped_reasons], axis=1)
            
            return mapped_reasons
            
        except CreditToolboxError as ce:
            logger.error(f"SDK failed to extract reason codes: {ce}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during reason code generation: {e}")
            raise


def main(input_path: str, output_path: str, artifacts_dir: str):
    """CLI orchestrator for batch processing."""
    try:
        logger.info(f"Loading applicant batch from {input_path}")
        df_input = pd.read_parquet(input_path)
        
        # In a real scenario, you'd filter for declined applications only, 
        # but for auditing we generate them for the whole batch.
        generator = AdverseActionGenerator(artifacts_dir=artifacts_dir)
        results = generator.generate(X=df_input, top_k=4)
        
        # Save output for audit trail
        results.to_parquet(output_path)
        logger.info(f"Reason codes successfully exported to {output_path}")
        
    except Exception as e:
        logger.error("Batch reason code generation failed.")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    
    parser = argparse.ArgumentParser(description="Generate FCRA Reason Codes from Pipeline State")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output Parquet file")
    parser.add_argument("--artifacts", type=str, default="ml_pipeline/artifacts", help="Path to pipeline state")
    
    args = parser.parse_args()
    main(args.input, args.output, args.artifacts)