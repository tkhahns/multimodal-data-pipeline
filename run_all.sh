#!/bin/bash
# Unified wrapper script for the multimodal data pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run setup
run_setup() {
    local setup_type="$1"
    
    print_status "Running environment setup..."
    
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is required but not found."
        print_status "Please install Poetry first: https://python-poetry.org/docs/#installation"
        exit 1
    fi
    
    case "$setup_type" in
        "full")
            print_status "Running full setup (including optional packages)..."
            poetry install
            ;;
        "quick")
            print_status "Running quick setup (essential packages only)..."
            poetry install --only main
            ;;
        *)
            print_status "Running default setup..."
            poetry install
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        print_success "Setup completed successfully!"
    else
        print_error "Setup failed!"
        exit 1
    fi
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if Poetry is installed
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry is required but not found."
        print_status "Please run: ./run_all.sh --setup"
        return 1
    fi
    
    # Check if Python script exists
    if [ ! -f "run_pipeline.py" ]; then
        print_error "run_pipeline.py script not found."
        return 1
    fi
    
    # Check if poetry environment is set up
    if ! poetry env info &> /dev/null; then
        print_warning "Poetry environment not found."
        print_status "Please run: ./run_all.sh --setup"
        return 1
    fi
    
    # Check if multimodal pipeline can be imported
    if ! poetry run python -c "from src.pipeline import MultimodalPipeline; print('✅ Pipeline import successful')" 2>/dev/null; then
        print_error "Could not import MultimodalPipeline."
        print_status "Make sure dependencies are installed by running: ./run_all.sh --setup"
        return 1
    fi
    
    print_success "All dependencies check passed!"
    return 0
}

# Function to list available features
list_features() {
    print_status "Available Features:"
    
    echo ""
    echo "📢 Audio Features:"
    echo "  • basic_audio          - Basic audio properties (duration, sample rate, channels)"
    echo "  • librosa_spectral     - Advanced spectral features using librosa"
    echo "  • opensmile           - OpenSMILE feature extraction"
    echo "  • audiostretchy       - Audio stretching and time modification"
    echo ""
    echo "🎤 Speech & Emotion Features:"
    echo "  • speech_emotion      - Speech emotion recognition"
    echo "  • heinsen_sentiment   - Heinsen routing sentiment analysis"
    echo "  • meld_emotion        - MELD emotion recognition"
    echo "  • speech_separation   - Speech source separation"
    echo ""
    echo "📝 Transcription Features:"
    echo "  • whisperx_transcription - WhisperX transcription with speaker diarization"
    echo ""
    echo "📄 Text Analysis Features:"
    echo "  • deberta_text        - DeBERTa text analysis"
    echo "  • simcse_text         - SimCSE sentence embeddings"
    echo "  • albert_text         - ALBERT text analysis"
    echo "  • sbert_text          - Sentence-BERT embeddings"
    echo "  • use_text            - Universal Sentence Encoder"
    echo ""
    echo "👁️ Computer Vision Features:"
    echo "  • emotieffnet_vision  - EmotiEffNet facial emotion recognition"
    echo "  • mediapipe_pose_vision - Google MediaPipe pose estimation"
    echo "  • deep_hrnet_vision   - Deep High-Resolution pose estimation"
    echo "  • simple_baselines_vision - Simple Baselines pose estimation"
    echo "  • ganimation_vision   - GANimation facial movements"
    echo "  • arbex_vision        - ARBEx emotion extraction"
    echo "  • openpose_vision     - OpenPose keypoint detection"
    echo "  • instadm_vision      - Insta-DM dense motion estimation"
    echo "  • optical_flow_vision - Optical flow movement estimation"
    echo "  • crowdflow_vision    - CrowdFlow person trajectories"
    echo "  • videofinder_vision  - VideoFinder object/people location (requires Ollama)"
    echo "  • smoothnet_vision    - SmoothNet pose estimation"
    echo "  • lanegcn_vision      - LaneGCN autonomous driving"
    echo "  • pare_vision         - PARE 3D human body estimation"
    echo "  • vitpose_vision      - ViTPose estimation"
    echo "  • psa_vision          - PSA pose estimation"
    echo "  • rsn_vision          - RSN pose estimation"
    echo "  • me_graphau_vision   - ME-GraphAU micro-expression"
    echo "  • dan_vision          - DAN emotion recognition"
    echo ""
}

# Function to show help
show_help() {
    cat << EOF
Usage: ./run_all.sh [options]

Setup Options:
  --setup               Run full environment setup
  --setup-quick         Run quick setup (skip optional packages)  
  --check-deps          Check if dependencies are installed

Pipeline Options:
  -d, --data-dir DIR    Directory with video/audio files (default: ./data)
  -o, --output-dir DIR  Output directory (default: ./output/YYYYMMDD_HHMMSS)
  -f, --features LIST   Comma-separated features to extract (default: all)
  --list-features       List available features and exit
  --is-audio            Process files as audio instead of video
  --log-file FILE       Path to log file (default: <output_dir>/pipeline.log)
  -h, --help            Show this help message

Available Features:
  Audio: basic_audio, librosa_spectral, opensmile, audiostretchy
  Speech: speech_emotion, heinsen_sentiment, meld_emotion, speech_separation
    Text: whisperx_transcription, deberta_text, simcse_text, albert_text, sbert_text, use_text
  Vision: emotieffnet_vision, mediapipe_pose_vision, deep_hrnet_vision, simple_baselines_vision,
                    ganimation_vision, arbex_vision, openpose_vision, instadm_vision,
          optical_flow_vision, crowdflow_vision, videofinder_vision, smoothnet_vision,
          lanegcn_vision, pare_vision, vitpose_vision, psa_vision, rsn_vision,
          me_graphau_vision, dan_vision

Notes:
- videofinder_vision requires Ollama to be installed and running
- Py-Feat (pyfeat_vision) is excluded in this build

Examples:
  ./run_all.sh --setup                    # Set up the environment
  ./run_all.sh                           # Run with all features
  ./run_all.sh --check-deps               # Check dependencies
  ./run_all.sh --list-features            # Show detailed feature list
  ./run_all.sh --data-dir /path/to/videos # Process specific directory
    ./run_all.sh --features basic_audio,speech_emotion  # Extract specific features
    ./run_all.sh --features lanegcn_vision,videofinder_vision # Vision features only

EOF
}

# Parse command line arguments
SETUP_MODE=""
CHECK_DEPS=false
LIST_FEATURES=false
PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --setup)
            SETUP_MODE="full"
            shift
            ;;
        --setup-quick)
            SETUP_MODE="quick"
            shift
            ;;
        --check-deps)
            CHECK_DEPS=true
            shift
            ;;
        --list-features)
            LIST_FEATURES=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # Pass all other arguments to the pipeline
            PIPELINE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Main execution logic
main() {
    print_status "Multimodal Data Pipeline - Unified Runner"
    
    # Handle setup mode
    if [ -n "$SETUP_MODE" ]; then
        run_setup "$SETUP_MODE"
        exit 0
    fi
    
    # Handle dependency check
    if [ "$CHECK_DEPS" = true ]; then
        if check_dependencies; then
            exit 0
        else
            exit 1
        fi
    fi
    
    # Handle list features
    if [ "$LIST_FEATURES" = true ]; then
        list_features
        exit 0
    fi
    
    # Check dependencies before running pipeline
    if ! check_dependencies; then
        print_error "Dependencies check failed!"
        print_status "Run './run_all.sh --setup' to install dependencies."
        exit 1
    fi
    
    # Make sure the script is executable
    chmod +x run_pipeline.py 2>/dev/null || true
    
    # Execute the pipeline
    print_status "Running multimodal data pipeline..."
    print_status "Arguments: ${PIPELINE_ARGS[*]}"

    # Optional preflight: warn if videofinder_vision selected without Ollama
    for i in "${!PIPELINE_ARGS[@]}"; do
        if [[ "${PIPELINE_ARGS[$i]}" == "--features" ]]; then
            next_index=$((i+1))
            features_arg="${PIPELINE_ARGS[$next_index]}"
            if [[ "$features_arg" == *"videofinder_vision"* ]]; then
                if ! command -v ollama &> /dev/null; then
                    print_warning "videofinder_vision selected but 'ollama' not found. Please install/start Ollama."
                fi
            fi
        fi
    done
    
    poetry run python run_pipeline.py "${PIPELINE_ARGS[@]}"
    
    # Check exit status
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        print_error "Pipeline execution failed with exit code $exit_code."
        exit $exit_code
    fi
    
    print_success "Pipeline execution completed successfully!"
}

# Run main function
main
