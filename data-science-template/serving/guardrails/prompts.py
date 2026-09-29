"""
serving_layer/guardrails/prompts.py

LLM Guardrails and Prompt Templates for the ExplanationAgent.
Uses Pydantic to strictly coerce and validate Large Language Model outputs, 
ensuring generated Adverse Action reasons are legally compliant, formatted 
correctly, and free of hallucinations or prohibited discriminatory language.

Date: 2026-09-29 | Circasia, Quindio, Colombia
"""

import json
from typing import List
from pydantic import BaseModel, Field, field_validator, ValidationError

# Regulatory definitions for Fair Lending constraints
PROHIBITED_LLM_TERMS = {
    "race", "gender", "sex", "religion", "national origin", 
    "marital", "zip code", "neighborhood", "age", "ethnicity"
}

# ---------------------------------------------------------------------------
# 1. Prompt Templates
# ---------------------------------------------------------------------------

EXPLANATION_SYSTEM_PROMPT = """
You are an automated regulatory compliance agent operating within a secure credit bureau environment.
Your sole function is to translate mathematical model penalizations (Weight of Evidence / Log-Odds) 
into plain-language Adverse Action codes for consumers.

RULES:
1. Output MUST be valid JSON matching the exact schema provided.
2. The explanation must be objective, financial, and directly related to the feature provided.
3. Keep the explanation under 150 characters.
4. NEVER mention or infer demographics, location, or personal attributes.
"""

EXPLANATION_USER_PROMPT = """
Convert the following model penalty into a human-readable reason.

Model Data:
- Feature: {feature_name}
- Input Value: {raw_value}
- Penalty Magnitude (Log-Odds): {penalty_weight}

Respond strictly in JSON:
{{
    "feature_name": "{feature_name}",
    "human_readable_reason": "..."
}}
"""

# ---------------------------------------------------------------------------
# 2. Pydantic Output Guardrails
# ---------------------------------------------------------------------------

class LLMAdverseAction(BaseModel):
    """
    Strict validation envelope for the LLM's raw text generation.
    If the LLM hallucinates outside these bounds, Pydantic catches it immediately.
    """
    feature_name: str
    human_readable_reason: str = Field(
        ..., 
        min_length=10, 
        max_length=150, 
        description="Plain text explanation of the penalty."
    )

    @field_validator("human_readable_reason")
    @classmethod
    def enforce_fair_lending_language(cls, text: str) -> str:
        """
        Hard-stops any LLM output that accidentally includes protected class keywords.
        """
        text_lower = text.lower()
        violations = [term for term in PROHIBITED_LLM_TERMS if term in text_lower]
        
        if violations:
            raise ValueError(f"FAIR LENDING VIOLATION: LLM generated prohibited terms: {violations}")
        
        return text

class LLMResponseGuard(BaseModel):
    """Parses batch outputs if the LLM processes multiple features at once."""
    reasons: List[LLMAdverseAction]

# ---------------------------------------------------------------------------
# 3. Guardrail Execution Engine
# ---------------------------------------------------------------------------

def parse_and_validate_llm_output(raw_llm_json_string: str) -> LLMAdverseAction:
    """
    Attempts to parse the raw string from the LLM into the Pydantic model.
    Provides a safe fallback if the LLM output is hopelessly malformed or illegal.
    """
    try:
        # Step 1: Force basic JSON parsing
        parsed_dict = json.loads(raw_llm_json_string)
        
        # Step 2: Push through Pydantic validators (Length limits & Fair Lending checks)
        validated_data = LLMAdverseAction(**parsed_dict)
        return validated_data
        
    except json.JSONDecodeError as e:
        # Fallback for malformed JSON hallucination
        return LLMAdverseAction(
            feature_name="unknown",
            human_readable_reason="Unable to generate specific reason due to system format error."
        )
    except ValidationError as e:
        # Fallback for Fair Lending violations or length issues caught by Pydantic
        error_msg = e.errors()[0].get('msg', 'Validation failed')
        return LLMAdverseAction(
            feature_name="compliance_override",
            human_readable_reason=f"Standardized fallback: Risk criteria not met."
        )

def format_explanation_prompt(feature_name: str, raw_value: float, penalty_weight: float) -> str:
    """Hydrates the template for the Orchestrator to pass to the LLM."""
    return EXPLANATION_USER_PROMPT.format(
        feature_name=feature_name,
        raw_value=raw_value,
        penalty_weight=round(penalty_weight, 4)
    )