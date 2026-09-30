"""
tests/test_circuit_breaker.py
"""

import time
import pytest
from serving_layer.circuit_breaker import CircuitBreaker, CircuitState


def fallback_handler(*args, **kwargs):
    return "FALLBACK_TRIGGERED"


def test_circuit_breaker_normal_execution():
    """Tests that normal execution keeps the circuit CLOSED."""
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=10)

    @breaker
    def successful_service():
        return "SUCCESS"

    assert successful_service() == "SUCCESS"
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_trips_to_open():
    """Tests that consecutive failures trip the circuit to OPEN and route to fallback."""
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout_sec=10,
        fallback_function=fallback_handler,
    )

    @breaker
    def failing_service():
        raise RuntimeError("Service failure")

    # Failure 1
    assert failing_service() == "FALLBACK_TRIGGERED"
    assert breaker.state == CircuitState.CLOSED

    # Failure 2 (reaches threshold)
    assert failing_service() == "FALLBACK_TRIGGERED"
    assert breaker.state == CircuitState.OPEN

    # Subsequent call while OPEN instantly triggers fallback without executing wrapped logic
    assert failing_service() == "FALLBACK_TRIGGERED"


def test_circuit_breaker_recovery_half_open():
    """Tests transition from OPEN to HALF_OPEN after timeout and reset to CLOSED upon success."""
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout_sec=1,  # Short timeout for testing
        fallback_function=fallback_handler,
    )

    should_fail = True

    @breaker
    def dynamic_service():
        if should_fail:
            raise RuntimeError("Temporary error")
        return "RECOVERED"

    # Trip the circuit
    dynamic_service()
    assert breaker.state == CircuitState.OPEN

    # Wait for recovery timeout to elapse
    time.sleep(1.1)

    # Next call should trigger HALF_OPEN check; success closes the circuit
    should_fail = False
    result = dynamic_service()
    assert result == "RECOVERED"
    assert breaker.state == CircuitState.CLOSED