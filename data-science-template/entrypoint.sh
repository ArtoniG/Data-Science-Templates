#!/bin/bash
set -e

# Default to API if no argument is provided
PROCESS_TYPE=${1:-api}

case "$PROCESS_TYPE" in
  api)
    echo "Starting FastAPI server for real-time inference..."
    # Uvicorn is used here; adjust workers based on your target CPU allocation
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
    ;;
  
  batch)
    echo "Starting batch scoring job..."
    # Pass any trailing arguments to the batch script
    shift
    exec python -m scripts.batch_score "$@"
    ;;
    
  *)
    echo "Unknown process type: $PROCESS_TYPE"
    echo "Valid options: api, batch"
    exit 1
    ;;
esac