"""
serving_layer/app.py

FastAPI Real-Time Serving Layer.
Mounts the Multi-Agent Credit Orchestrator to a high-concurrency ASGI server.
Protected by Pydantic schemas and the Circuit Breaker to ensure millisecond 
latency and zero downtime.

Date: 2026-09-29 | Circasia, Quindio, Colombia
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .schemas import CreditApplicationRequest, CreditDecisionResponse
from .agent_orchestrator import CreditOrchestrator
from .circuit_breaker import CircuitBreaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for the Orchestrator
orchestrator = None


def fallback_credit_decision(request: CreditApplicationRequest) -> CreditDecisionResponse:
    """Safe fallback when the primary ML graph is unreachable."""
    return CreditDecisionResponse(
        application_id=request.application_id,
        decision="MANUAL_REVIEW",
        credit_score=0,
        probability_of_default=1.0,
        adverse_action_codes=[{"feature_name": "system", "impact_magnitude": 0.0, "human_readable_reason": "System degraded. Routed to manual underwriting."}],
        execution_time_ms=0.0
    )


# Instantiate the Circuit Breaker
breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=30, fallback_function=fallback_credit_decision)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the pipeline state and instantiate the Orchestrator exactly once 
    during container boot, preventing disk I/O on individual requests.
    """
    global orchestrator
    logger.info("Booting ML API. Loading immutable pipeline state into memory...")
    try:
        # In a real environment, this path is passed via env vars
        orchestrator = CreditOrchestrator(state_file_path="/app/artifacts/pipeline_state.json")
        logger.info("Model state loaded. API is ready to receive traffic.")
        yield
    finally:
        logger.info("Shutting down API. Releasing resources.")
        orchestrator = None

app = FastAPI(title="Multi-Agent Credit Decision API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Injects true system latency into the response headers for Datadog/Prometheus tracking."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time-Ms"] = str(round(process_time * 1000, 2))
    return response


@app.post("/v1/score", response_model=CreditDecisionResponse)
@breaker
def score_application(payload: CreditApplicationRequest):
    """
    Primary ingestion endpoint.
    Payload is automatically validated by the Pydantic schema before entering the function.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Model orchestrator not initialized.")
    
    # The CircuitBreaker wrapper handles exceptions and fallback routing automatically
    return orchestrator.process_application(payload)


@app.get("/health")
def health_check():
    """Liveness probe for Kubernetes."""
    if breaker.state.value == "OPEN":
        return JSONResponse(status_code=503, content={"status": "degraded", "circuit_state": "OPEN"})
    return {"status": "healthy", "circuit_state": breaker.state.value}