"""
serving_layer/schemas.py

Pydantic Guardrails for the Multi-Agent Credit Engine.
These schemas act as the first line of defense (the Validation Agent), 
ensuring that only strictly typed, legally permissible, and logically bounded 
data enters the orchestration graph.

Date: 2026-09-29 | Circasia, Quindio, Colombia
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class CreditApplicationRequest(BaseModel):
    """
    Strict input schema for retail credit applications.
    Enforces absolute boundaries to prevent adversarial inputs or 
    out-of-distribution pipeline crashes.
    """
    application_id: str = Field(..., description="Unique UUID for the transaction")
    
    # Financial & Bureau Features
    revolving_utilization: float = Field(..., ge=0.0, le=5.0, description="Total balance / Total limit")
    age_of_oldest_tradeline_months: int = Field(..., ge=0, le=1200)
    number_of_open_credit_lines: int = Field(..., ge=0, le=200)
    number_of_derogatory_marks: int = Field(..., ge=0, le=50)
    recent_inquiries_6m: int = Field(..., ge=0, le=100)
    
    # Demographic / Income Features
    monthly_income: float = Field(..., ge=0.0, description="Verified monthly income in local currency")
    debt_to_income_ratio: float = Field(..., ge=0.0, le=10.0)

    @field_validator("monthly_income", "revolving_utilization", mode="before")
    @classmethod
    def prevent_nan_and_infinity(cls, value: float) -> float:
        """Pydantic guardrail to reject NaN or Inf values immediately."""
        import math
        if math.isnan(value) or math.isinf(value):
            raise ValueError("NaN or Infinity detected in strict financial fields.")
        return value


class AdverseActionCode(BaseModel):
    """Structured explanation for model penalization."""
    feature_name: str
    impact_magnitude: float
    human_readable_reason: str


class CreditDecisionResponse(BaseModel):
    """
    Standardized immutable output contract for the Serving API.
    """
    application_id: str
    decision: str = Field(..., pattern="^(APPROVED|REJECTED|MANUAL_REVIEW)$")
    credit_score: int = Field(..., ge=300, le=850)
    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    adverse_action_codes: Optional[List[AdverseActionCode]] = None
    execution_time_ms: float