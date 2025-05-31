#!/bin/bash
# filepath: /Users/evidenceb/Desktop/multimodal-data-pipeline/setup_env.sh

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 is required but not found."
    echo "Please install Python 3.12 and try again."
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is required but not found."
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is required for audio extraction."
    echo "Please install ffmpeg using: brew install ffmpeg"
    echo "Then run this script again."
    exit 1
fi

# Create a Poetry environment with Python 3.12
echo "Creating Poetry environment with Python 3.12..."
poetry env use python3.12

# Install dependencies
echo "Installing dependencies..."
# Install core dependencies
poetry install

# Install core packages directly to ensure they're available
echo "Installing essential packages..."
poetry run pip install numpy pandas librosa opencv-python torch torchaudio ffmpeg-python soundfile
poetry run pip install pyarrow fastparquet # For parquet support

# Install optional dependency groups
echo "Installing optional dependency groups..."
poetry install --with common

# Create necessary directories
mkdir -p output
mkdir -p pretrained_models

# Make the run script executable
chmod +x run_pipeline.sh

echo ""
echo "========================================================"
echo "Setup completed!"
echo ""
echo "To run the pipeline, you have two options:"
echo ""
echo "Option 1: Activate the environment first, then run the script:"
echo "  poetry env activate"
echo "  ./run_pipeline.sh"
echo ""
echo "Option 2: Run with Poetry directly:"
echo "  poetry run ./run_pipeline.sh"
echo ""
echo "To see available options:"
echo "  ./run_pipeline.sh --help"
echo "========================================================"
