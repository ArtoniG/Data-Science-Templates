"""
serving_layer/batch_scoring.py

Asynchronous High-Throughput Batch Scoring Engine.
Designed for portfolio-level recalibration (e.g., monthly risk reassessments).
Bypasses the HTTP layer (FastAPI) and interfaces directly with the 
Multi-Agent Orchestrator using multi-processing for CPU-bound parallelism.

Date: 2026-09-29 | Circasia, Quindio, Colombia
"""

import time
import logging
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from .schemas import CreditApplicationRequest, CreditDecisionResponse
from .agent_orchestrator import CreditOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Global orchestrator instance per worker process
_orchestrator = None


def worker_init(state_file_path: str):
    """
    Initializes the orchestrator inside each independent worker process.
    Crucial for avoiding lock contention and serialization overhead 
    when using ProcessPoolExecutor.
    """
    global _orchestrator
    _orchestrator = CreditOrchestrator(state_file_path=state_file_path)


def process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a localized chunk of the portfolio dataset through the 
    Pydantic guardrails and the Orchestrator DAG.
    """
    results = []
    
    # We iterate rows here, but for extreme scale, the ScoringAgent 
    # could be refactored to accept matrix inputs directly.
    for _, row in chunk_df.iterrows():
        try:
            # 1. Strict Input Validation (Even for historical batch data)
            request = CreditApplicationRequest(**row.to_dict())
            
            # 2. Execute the DAG
            response: CreditDecisionResponse = _orchestrator.process_application(request)
            
            # 3. Extract necessary outputs
            results.append({
                "application_id": response.application_id,
                "decision": response.decision,
                "credit_score": response.credit_score,
                "probability_of_default": response.probability_of_default,
                "execution_time_ms": response.execution_time_ms,
                "error": None
            })
        except Exception as e:
            # Trap errors so a single corrupted row doesn't kill a 10-million row job
            results.append({
                "application_id": row.get("application_id", "UNKNOWN_OR_MISSING"),
                "decision": "ERROR",
                "credit_score": None,
                "probability_of_default": None,
                "execution_time_ms": 0.0,
                "error": str(e)
            })
            
    return pd.DataFrame(results)


def run_batch_scoring(
    input_path: str, 
    output_path: str, 
    state_file_path: str, 
    workers: int = 4, 
    chunk_size: int = 10000
):
    """
    Main execution loop for parallel portfolio scoring.
    """
    start_time = time.perf_counter()
    logger.info(f"Starting batch scoring job. Input: {input_path}, Workers: {workers}")

    # For production bureau datasets, PyArrow is preferred, but Pandas handles Parquet well.
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        logger.critical(f"Failed to read input dataset: {e}")
        return

    total_rows = len(df)
    logger.info(f"Loaded {total_rows} records for batch scoring.")

    # Slice dataframe into discrete memory-friendly chunks
    chunks = [df[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
    output_dfs = []
    
    # Distribute the chunks across CPU cores
    with ProcessPoolExecutor(
        max_workers=workers, 
        initializer=worker_init, 
        initargs=(state_file_path,)
    ) as executor:
        
        future_to_chunk = {
            executor.submit(process_chunk, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        for future in as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result_df = future.result()
                output_dfs.append(result_df)
                logger.info(f"Completed chunk {chunk_index + 1}/{len(chunks)}")
            except Exception as exc:
                logger.error(f"Chunk {chunk_index + 1} generated a fatal exception: {exc}")

    if not output_dfs:
        logger.error("No data processed successfully. Exiting.")
        return

    # Recombine results and export
    final_df = pd.concat(output_dfs, ignore_index=True)
    final_df.to_parquet(output_path, index=False)
    
    elapsed = time.perf_counter() - start_time
    throughput = total_rows / elapsed if elapsed > 0 else 0
    
    logger.info(f"Batch scoring complete. Scored {total_rows} records in {elapsed:.2f} seconds.")
    logger.info(f"System Throughput: {throughput:.2f} records/sec. Output saved to {output_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Credit Portfolio Batch Recalibration")
    parser.add_argument("--input", required=True, help="Path to input Parquet file")
    parser.add_argument("--output", required=True, help="Path to output Parquet file")
    parser.add_argument("--state-file", default="/app/artifacts/pipeline_state.json", help="Path to immutable pipeline state")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU cores to allocate")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Rows per processing batch")
    
    args = parser.parse_args()
    run_batch_scoring(
        input_path=args.input, 
        output_path=args.output, 
        state_file_path=args.state_file, 
        workers=args.workers,
        chunk_size=args.chunk_size
    )