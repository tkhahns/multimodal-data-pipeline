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
    
    if [ ! -f "setup_env.sh" ]; then
        print_error "setup_env.sh not found!"
        exit 1
    fi
    
    chmod +x setup_env.sh
    
    case "$setup_type" in
        "full")
            print_status "Running full setup (including optional packages)..."
            ./setup_env.sh
            ;;
        "quick")
            print_status "Running quick setup (essential packages only)..."
            ./setup_env.sh --quick
            ;;
        *)
            print_status "Running default setup..."
            ./setup_env.sh
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
    
    print_success "All dependencies check passed!"
    return 0
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
  -f, --features LIST   Comma-separated features to extract
                        Available: basic_audio,librosa_spectral,opensmile,
                                  speech_emotion,heinsen_sentiment,speech_separation,
                                  whisperx_transcription,deberta_text,simcse_text,
                                  albert_text,sbert_text,use_text
  --list-features       List available features and exit
  --is-audio            Process files as audio instead of video
  --log-file FILE       Path to log file (default: <output_dir>/pipeline.log)
  -h, --help            Show this help message

Examples:
  ./run_all.sh --setup                    # Set up the environment
  ./run_all.sh                           # Run with default settings
  ./run_all.sh --check-deps               # Check dependencies
  ./run_all.sh --data-dir /path/to/videos # Process specific directory
  ./run_all.sh --features basic_audio,speech_emotion  # Extract specific features

EOF
}

# Parse command line arguments
SETUP_MODE=""
CHECK_DEPS=false
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
    
    # Check dependencies before running pipeline
    if ! check_dependencies; then
        print_error "Dependencies check failed!"
        print_status "Run './run_all.sh --setup' to install dependencies."
        exit 1
    fi
    
    # Make sure the script is executable
    chmod +x run_pipeline.py
    
    # Execute the pipeline
    print_status "Running multimodal data pipeline..."
    print_status "Arguments: ${PIPELINE_ARGS[*]}"
    
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
