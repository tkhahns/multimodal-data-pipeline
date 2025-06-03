#!/bin/bash
# filepath: /Users/evidenceb/Desktop/multimodal-data-pipeline/run_pipeline.sh
# Unified wrapper script for the multimodal data pipeline

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is required but not found."
    echo "Please run ./setup_env.sh first."
    exit 1
fi

# Check if Python script exists
if [ ! -f "run_simple.py" ]; then
    echo "Error: run_simple.py script not found."
    exit 1
fi

# Make sure the script is executable
chmod +x run_simple.py

# Execute the script within Poetry's run command
echo "Running multimodal data pipeline with Poetry..."
poetry run python run_simple.py "$@"

# Check exit status
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Pipeline execution failed with exit code $exit_code."
    exit $exit_code
fi

echo "Pipeline execution completed successfully."
