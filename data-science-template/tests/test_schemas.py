"""
tests/test_schemas.py
"""

import math
import pytest
from pydantic import ValidationError
from serving_layer.schemas import CreditApplicationRequest, CreditDecisionResponse


def test_valid_credit_application_request():
    """Verifies that a well-formed payload passes validation."""
    payload = {
        "application_id": "123e4567-e89b-12d3-a456-426614174000",
        "revolving_utilization": 0.35,
        "age_of_oldest_tradeline_months": 72,
        "number_of_open_credit_lines": 5,
        "number_of_derogatory_marks": 0,
        "recent_inquiries_6m": 1,
        "monthly_income": 5500.0,
        "debt_to_income_ratio": 0.28,
    }
    req = CreditApplicationRequest(**payload)
    assert req.application_id == "123e4567-e89b-12d3-a456-426614174000"
    assert req.monthly_income == 5500.0


@pytest.mark.parametrize(
    "invalid_field, invalid_value",
    [
        ("revolving_utilization", -0.1),  # Out of lower bound (ge=0.0)
        ("revolving_utilization", 10.0),  # Out of upper bound (le=5.0)
        ("monthly_income", float("nan")),  # NaN injection
        ("monthly_income", float("inf")),  # Infinity injection
        ("debt_to_income_ratio", 15.0),  # Out of upper bound (le=10.0)
    ],
)
def test_invalid_input_guardrails(invalid_field, invalid_value):
    """Ensures input guardrails reject boundary violations and non-finite numbers."""
    base_payload = {
        "application_id": "app_test_001",
        "revolving_utilization": 0.2,
        "age_of_oldest_tradeline_months": 48,
        "number_of_open_credit_lines": 3,
        "number_of_derogatory_marks": 0,
        "recent_inquiries_6m": 0,
        "monthly_income": 4000.0,
        "debt_to_income_ratio": 0.3,
    }
    base_payload[invalid_field] = invalid_value

    with pytest.raises(ValidationError):
        CreditApplicationRequest(**base_payload)


def test_credit_decision_response_schema():
    """Validates output response schema structure and enum enforcement."""
    res = CreditDecisionResponse(
        application_id="app_test_001",
        decision="APPROVED",
        credit_score=720,
        probability_of_default=0.02,
        execution_time_ms=12.5,
    )
    assert res.decision == "APPROVED"
    assert res.credit_score == 720

    with pytest.raises(ValidationError):
        # Invalid decision pattern
        CreditDecisionResponse(
            application_id="app_test_001",
            decision="SOME_CUSTOM_DECISION",
            credit_score=720,
            probability_of_default=0.02,
            execution_time_ms=12.5,
        )