"""
tests/test_app.py
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from serving_layer.app import app, breaker
from serving_layer.circuit_breaker import CircuitState
from serving_layer.schemas import CreditDecisionResponse


@pytest.fixture
def client():
    """Provides a clean TestClient instance."""
    return TestClient(app)


@patch("serving_layer.app.orchestrator")
def test_score_application_endpoint_success(mock_orchestrator, client):
    """Tests successful scoring API execution (HTTP 200)."""
    # Mock orchestrator response
    mock_orchestrator.process_application.return_value = CreditDecisionResponse(
        application_id="app_123",
        decision="APPROVED",
        credit_score=750,
        probability_of_default=0.01,
        execution_time_ms=5.0,
    )

    payload = {
        "application_id": "app_123",
        "revolving_utilization": 0.15,
        "age_of_oldest_tradeline_months": 60,
        "number_of_open_credit_lines": 4,
        "number_of_derogatory_marks": 0,
        "recent_inquiries_6m": 0,
        "monthly_income": 6000.0,
        "debt_to_income_ratio": 0.20,
    }

    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "APPROVED"
    assert data["credit_score"] == 750
    assert "X-Process-Time-Ms" in response.headers


def test_score_application_endpoint_validation_error(client):
    """Tests that invalid JSON payloads return HTTP 422 Unprocessable Entity."""
    invalid_payload = {
        "application_id": "app_123",
        "revolving_utilization": -5.0,  # Invalid bound
    }
    response = client.post("/v1/score", json=invalid_payload)
    assert response.status_code == 422


def test_health_check_endpoint(client):
    """Tests liveness probe status when healthy vs degraded."""
    breaker.state = CircuitState.CLOSED
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    # Simulate tripped circuit
    breaker.state = CircuitState.OPEN
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    
    # Reset state
    breaker.state = CircuitState.CLOSED