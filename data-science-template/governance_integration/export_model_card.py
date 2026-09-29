"""
governance_integration/export_model_card.py

Automated Regulatory Model Card Generator.
Reads the declarative `pipeline_state.json`, model configuration, and audit metrics
to dynamically render a compliance-ready Markdown document using Jinja2.

Context: Executed as the final step of the CI/CD training pipeline before 
model promotion to ensure Model Risk Management (MRM) documentation is immutable 
and tightly coupled to the exact model state.
Date: 2026-09-29 | Circasia, Quindio, Colombia
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """
    Orchestrates the aggregation of model state, configuration, and metrics
    to render the regulatory Model Card.
    """

    def __init__(
        self, 
        artifacts_dir: str, 
        config_path: str, 
        template_dir: str = "governance_integration/templates"
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.config_path = Path(config_path)
        self.template_dir = Path(template_dir)
        
        # Load Data Sources
        self.config = self._load_yaml(self.config_path)
        self.state = self._load_json(self.artifacts_dir / "pipeline_state.json")
        
        # Metrics are typically saved by the PipelineAuditor in Phase 6
        self.metrics = self._load_json(self.artifacts_dir / "audit_metrics.json", optional=True)
        
        # Initialize Jinja2 Environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(searchpath=self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_json(self, path: Path, optional: bool = False) -> Dict[str, Any]:
        if not path.exists():
            if optional:
                logger.warning(f"Optional file not found: {path}. Defaulting to empty metrics.")
                return {}
            raise FileNotFoundError(f"Required state file not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _build_feature_table_rows(self) -> str:
        """
        Extracts feature metadata from the WOE Encoder state to populate the IV/PSI table.
        """
        try:
            woe_state = self.state["steps"]["woe_encoder"]
            features = woe_state.get("columns", [])
            binning_logic = woe_state.get("binning_logic", {})
            
            rows = []
            for feat in features:
                # Extract number of bins as a proxy for PSI baseline complexity
                bins = len(binning_logic.get(feat, {}))
                
                # In a real scenario, IV would be stored in the WOE state or metrics JSON
                iv = self.metrics.get("feature_iv", {}).get(feat, "N/A")
                if isinstance(iv, float):
                    iv = f"{iv:.4f}"
                
                rows.append(f"| {feat} | Numeric/Categorical | {iv} | {bins} Bins | Yes |")
                
            return "\n".join(rows)
        except KeyError as e:
            logger.error(f"Failed to parse WOE state for feature table: {e}")
            return "| Error | Error | Error | Error | Error |"

    def _prepare_context(self) -> Dict[str, Any]:
        """Maps JSON/YAML data to the exact Jinja2 template variables."""
        
        # Safe extraction of nested config and metrics
        meta = self.config.get("model_metadata", {})
        data = self.config.get("data", {})
        log_reg = self.config.get("logistic_regression", {})
        train_metrics = self.metrics.get("train", {})
        test_metrics = self.metrics.get("test", {})
        
        # Compute divergence safely
        gini_train = train_metrics.get("gini", 0.0)
        gini_test = test_metrics.get("gini", 0.0)
        divergence = 0.0
        if gini_train > 0:
            divergence = ((gini_train - gini_test) / gini_train) * 100

        return {
            "model_name": meta.get("name", "Unknown_Model"),
            "model_version": meta.get("version", "1.0.0"),
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_owner": meta.get("owner", "Credit Analytics Team"),
            "deployment_status": "PENDING MRM APPROVAL",
            
            "model_description": meta.get("description", "Credit scoring model."),
            "imputation_strategy": self.config.get("imputation_strategy", "median"),
            "logistic_c": log_reg.get("C", 1.0),
            
            # These would ideally be in config.yaml, hardcoded here for safety mapping
            "base_score": 600,
            "odds_ratio": "50:1",
            "pdo": 20,
            
            "train_start_date": data.get("train_start", "N/A"),
            "train_end_date": data.get("train_end", "N/A"),
            "train_size": self.metrics.get("train_size", "N/A"),
            "test_size": self.metrics.get("test_size", "N/A"),
            
            "target_definition_bad": "90+ Days Past Due within 12 months",
            "target_definition_good": "Never 30+ DPD within 12 months",
            "target_definition_indeterminate": "1-89 DPD (Excluded)",
            
            "gini_train": f"{gini_train:.3f}",
            "gini_test": f"{gini_test:.3f}",
            "status_gini": "PASS" if gini_test >= 0.40 else "FAIL",
            
            "ks_train": f"{train_metrics.get('ks', 0.0):.3f}",
            "ks_test": f"{test_metrics.get('ks', 0.0):.3f}",
            "status_ks": "PASS" if test_metrics.get('ks', 0.0) >= 0.35 else "FAIL",
            
            "auc_train": f"{train_metrics.get('auc', 0.0):.3f}",
            "auc_test": f"{test_metrics.get('auc', 0.0):.3f}",
            "status_auc": "PASS" if test_metrics.get('auc', 0.0) >= 0.70 else "FAIL",
            
            "gini_divergence": f"{divergence:.1f}",
            "feature_metrics_table_rows": self._build_feature_table_rows(),
            
            "top_reason_1": "High revolving utilization",
            "top_reason_2": "Recent severe delinquency",
            "top_reason_3": "Low depth of credit history",
        }

    def generate_and_save(self, template_name: str = "model_card_template.md") -> str:
        """Renders the Markdown template and saves it to the artifacts directory."""
        logger.info(f"Generating Model Card using template: {template_name}")
        
        try:
            template = self.jinja_env.get_template(template_name)
        except TemplateNotFound:
            raise FileNotFoundError(f"Template '{template_name}' not found in {self.template_dir}")

        context = self._prepare_context()
        rendered_md = template.render(context)
        
        output_filename = f"MODEL_CARD_{context['model_name']}_v{context['model_version']}.md"
        output_path = self.artifacts_dir / output_filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_md)
            
        logger.info(f"Model Card successfully exported to {output_path}")
        return str(output_path)


def main(config_path: str, artifacts_dir: str, template_dir: str):
    """CLI Entrypoint for CI/CD Pipeline Execution."""
    try:
        generator = ModelCardGenerator(
            artifacts_dir=artifacts_dir,
            config_path=config_path,
            template_dir=template_dir
        )
        generator.generate_and_save()
    except Exception as e:
        logger.error(f"Failed to generate model card: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
    
    parser = argparse.ArgumentParser(description="Automated MRM Model Card Generator")
    parser.add_argument("--config", type=str, default="ml_pipeline/config.yaml", help="Path to config.yaml")
    parser.add_argument("--artifacts", type=str, default="ml_pipeline/artifacts", help="Path to artifacts dir")
    parser.add_argument("--templates", type=str, default="governance_integration/templates", help="Path to templates dir")
    
    args = parser.parse_args()
    main(args.config, args.artifacts, args.templates)