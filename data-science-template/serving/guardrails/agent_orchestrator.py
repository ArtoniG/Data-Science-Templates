"""
serving_layer/agent_orchestrator.py

Multi-Agent Orchestrator for Real-Time Credit Scoring.
Coordinates a swarm of specialized agents to process an application:
1. ValidationAgent (Pydantic-enforced input parsing)
2. ScoringAgent (In-memory execution of pipeline_state.json)
3. ExplanationAgent (Dynamic Reason Code extraction)
4. PolicyAgent (Applies business rules: cutoffs, auto-rejects)

This architecture guarantees separation of concerns and extreme auditability.
"""

import time
import logging
from typing import Dict, Any
from .schemas import CreditApplicationRequest, CreditDecisionResponse, AdverseActionCode

# Assuming these agents will be implemented in subsequent steps
# from .agents import ScoringAgent, ExplanationAgent, PolicyAgent

logger = logging.getLogger(__name__)


class CreditOrchestrator:
    """
    The central intelligence that routes payloads through the agent graph.
    """
    def __init__(self, state_file_path: str):
        # In a real implementation, we instantiate the sub-agents here, passing 
        # the parsed state so it only loads into memory once during container boot.
        logger.info("Initializing Multi-Agent Credit Orchestrator...")
        
        # self.scoring_agent = ScoringAgent(state_file_path)
        # self.explanation_agent = ExplanationAgent(state_file_path)
        # self.policy_agent = PolicyAgent(config_path="policy_rules.yaml")
        
        self.state_file_path = state_file_path

    def process_application(self, request: CreditApplicationRequest) -> CreditDecisionResponse:
        """
        Executes the directed acyclic graph (DAG) of the credit decision process.
        """
        start_time = time.perf_counter()
        app_id = request.application_id
        logger.info(f"[{app_id}] Initiating multi-agent orchestration.")
        
        try:
            # Step 1: Feature Extraction (Dictionary conversion for the Scoring Agent)
            raw_features = request.model_dump(exclude={"application_id"})
            
            # Step 2: Scoring Agent (WOE Encoding -> Logistic Regression)
            # score, pd, woe_transformed_vector = self.scoring_agent.predict(raw_features)
            
            # --- MOCK EXECUTION FOR ARCHITECTURE DEMONSTRATION ---
            score = 685
            pd = 0.045
            woe_transformed_vector = {"revolving_utilization": -1.2, "recent_inquiries_6m": -0.8}
            # -----------------------------------------------------

            logger.debug(f"[{app_id}] ScoringAgent returned Score: {score}, PD: {pd:.4f}")

            # Step 3: Policy Agent (Business Rules, Cutoffs)
            # decision = self.policy_agent.evaluate(score, raw_features)
            decision = "REJECTED" if score < 660 else "APPROVED"

            # Step 4: Explanation Agent (Adverse Action / Reason Codes)
            # Only trigger explanation logic if the application was rejected or borderline.
            adverse_actions = None
            if decision in ["REJECTED", "MANUAL_REVIEW"]:
                # adverse_actions = self.explanation_agent.extract_reasons(woe_transformed_vector)
                
                # --- MOCK ADVERSE ACTION ---
                adverse_actions = [
                    AdverseActionCode(
                        feature_name="revolving_utilization",
                        impact_magnitude=-1.2,
                        human_readable_reason="Proportion of balances to credit limits is too high."
                    )
                ]
                # ---------------------------
                
            execution_time = (time.perf_counter() - start_time) * 1000

            # Step 5: Construct Final Output via Validation Guardrail
            response = CreditDecisionResponse(
                application_id=app_id,
                decision=decision,
                credit_score=score,
                probability_of_default=pd,
                adverse_action_codes=adverse_actions,
                execution_time_ms=execution_time
            )
            
            logger.info(f"[{app_id}] Orchestration complete. Decision: {decision} in {execution_time:.2f}ms")
            return response

        except Exception as e:
            logger.error(f"[{app_id}] Fatal error during orchestration: {str(e)}", exc_info=True)
            raise RuntimeError(f"Orchestration failed for application {app_id}") from e