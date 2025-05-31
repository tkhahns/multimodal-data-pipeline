#!/bin/bash
# filepath: /Users/evidenceb/Desktop/multimodal-data-pipeline/run_pipeline.sh
# Script to run the multimodal data pipeline

# Check if running in Poetry environment
if [ -z "$POETRY_ACTIVE" ]; then
    echo "This script should be run within the Poetry environment."
    echo "Please run one of the following commands first:"
    echo "  poetry env activate"
    echo "Or run the script directly with Poetry:"
    echo "  poetry run ./run_pipeline.sh"
    echo ""
    echo "Attempting to activate Poetry environment automatically..."
    
    # Try both activation methods
    if [ -d "$(poetry env info --path 2>/dev/null)" ]; then
        # For Poetry 1.x compatibility
        source "$(poetry env info --path)/bin/activate" 2>/dev/null || \
        # For Poetry 2.x
        eval "$(poetry env use python3.12 --path 2>/dev/null)" 2>/dev/null
        
        if [ -z "$POETRY_ACTIVE" ]; then
            echo "Failed to activate Poetry environment automatically."
            echo "Please run the commands shown above."
            exit 1
        fi
    else
        echo "Could not locate Poetry environment."
        echo "Please make sure you've run ./setup_env.sh first."
        exit 1
    fi
    
    echo "Successfully activated Poetry environment."
fi

# Check for ffmpeg (needed for audio extraction)
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it before running this script."
    echo "You can install it using: brew install ffmpeg"
    exit 1
fi

# Set up variables
DATA_DIR="$(pwd)/data"
OUTPUT_DIR="$(pwd)/output/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

# Create output directories
mkdir -p "${OUTPUT_DIR}"

# Default to all features unless specified
FEATURES="basic_audio,librosa_spectral,speech_emotion,whisperx,xlsr_speech"

# Parse command-line arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--features)
            FEATURES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run_pipeline.sh [options]"
            echo ""
            echo "Options:"
            echo "  -d, --data-dir DIR    Directory with video/audio files (default: ./data)"
            echo "  -o, --output-dir DIR  Output directory (default: ./output/YYYYMMDD_HHMMSS)"
            echo "  -f, --features LIST   Comma-separated features to extract (default: all)"
            echo "                        Available: basic_audio,librosa_spectral,speech_emotion,"
            echo "                                  speech_separation,whisperx,xlsr_speech,s2t_speech"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}" # Restore positional parameters

# Log configuration
echo "Starting multimodal pipeline at $(date)" | tee -a "${LOG_FILE}"
echo "Data directory: ${DATA_DIR}" | tee -a "${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Features to extract: ${FEATURES}" | tee -a "${LOG_FILE}"

# Check if data directory exists and contains files
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory ${DATA_DIR} does not exist." | tee -a "${LOG_FILE}"
    exit 1
fi

# Convert comma-separated features to Python list format
FEATURES_LIST=$(echo $FEATURES | sed 's/,/", "/g')
FEATURES_LIST="[\"${FEATURES_LIST}\"]"

# Run the pipeline
echo "Running pipeline..." | tee -a "${LOG_FILE}"
python -c "
import sys
from pathlib import Path
sys.path.append('${PWD}')

try:
    from src.pipeline import MultimodalPipeline
    
    # Initialize pipeline
    pipeline = MultimodalPipeline(
        output_dir='${OUTPUT_DIR}', 
        features=${FEATURES_LIST}, 
        device='cpu'
    )
    
    # Process data directory
    results = pipeline.process_directory('${DATA_DIR}', is_video=True)
    
    # Print summary
    print(f'Successfully processed {len(results)} files')
    for filename in results:
        print(f'- {filename}')
    
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee -a "${LOG_FILE}"

# Check if process was successful
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully at $(date)" | tee -a "${LOG_FILE}"
    echo "Results saved to ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
else
    echo "Pipeline failed at $(date)" | tee -a "${LOG_FILE}"
fi

echo "Log file saved to ${LOG_FILE}"