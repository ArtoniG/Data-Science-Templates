"""
serving_layer/circuit_breaker.py

MLOps Circuit Breaker for High-Availability Serving.
Prevents cascading failures in the credit scoring API. If the model execution 
graph starts throwing anomalies (e.g., memory leaks, feature drift timeouts), 
the circuit trips to OPEN, instantly returning a safe fallback response 
(e.g., "MANUAL_REVIEW") rather than hanging the bureau's upstream systems.

Date: 2026-09-29 | Circasia, Quindio, Colombia
"""

import time
import logging
from enum import Enum
from functools import wraps
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "CLOSED"       # Normal operation
    OPEN = "OPEN"           # Failing, requests instantly rejected/fallback
    HALF_OPEN = "HALF_OPEN" # Testing recovery


class CircuitBreaker:
    def __init__(
        self, 
        failure_threshold: int = 5, 
        recovery_timeout_sec: int = 30,
        fallback_function: Optional[Callable] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_sec
        self.fallback_function = fallback_function
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def _trip_circuit(self):
        logger.critical(f"CIRCUIT TRIPPED: {self.failure_threshold} consecutive failures. Entering OPEN state.")
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()

    def _attempt_reset(self):
        self.state = CircuitState.HALF_OPEN
        logger.info("CIRCUIT HALF-OPEN: Testing recovery with next request.")

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self._attempt_reset()
                else:
                    if self.fallback_function:
                        logger.warning("Circuit OPEN. Routing to fallback function.")
                        return self.fallback_function(*args, **kwargs)
                    raise RuntimeError("Circuit is OPEN. Service unavailable.")

            try:
                result = func(*args, **kwargs)
                
                # If we were testing recovery and it succeeded, close the circuit
                if self.state == CircuitState.HALF_OPEN:
                    logger.info("CIRCUIT CLOSED: Recovery successful.")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    
                return result
                
            except Exception as e:
                self.failure_count += 1
                logger.error(f"Execution failure {self.failure_count}/{self.failure_threshold}: {e}")
                
                if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                    self._trip_circuit()
                    
                if self.fallback_function:
                    return self.fallback_function(*args, **kwargs)
                raise e

        return wrapper