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
# Install core dependencies and optional dependency groups
poetry install --with common,speech,text

# Ensure all dependencies are properly installed
echo "Verifying installations..."
poetry run python -c "
import importlib
import sys

# Check for required packages
required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'librosa': 'librosa',
    'opencv-python': 'cv2',
    'torch': 'torch',
    'torchaudio': 'torchaudio',
    'ffmpeg-python': 'ffmpeg',
    'soundfile': 'soundfile',
    'pyarrow': 'pyarrow',
    'transformers': 'transformers'
}

missing = []
for package_name, import_name in required_packages.items():
    try:
        importlib.import_module(import_name)
        print(f'✓ {package_name} installed')
    except ImportError:
        missing.append(package_name)
        print(f'✗ {package_name} not found')

if missing:
    print(f'\nInstalling missing packages...')
    
    # Install each package separately to avoid errors
    for package in missing:
        print(f'Installing {package}...')
        try:
            import subprocess
            subprocess.check_call(['pip', 'install', package])
            print(f'  ✓ {package} installed successfully')
        except Exception as e:
            print(f'  ✗ Failed to install {package}: {e}')
"

# Create necessary directories
mkdir -p output
mkdir -p pretrained_models

# Make the run scripts executable
chmod +x run_simple.py
chmod +x run_pipeline.sh

echo ""
echo "========================================================"
echo "Setup completed!"
echo ""
echo "To run the pipeline, you have two options:"
echo ""
echo "Option 1: Use the unified run script (recommended):"
echo "  ./run_pipeline.sh"
echo ""
echo "Option 2: Run with Poetry directly:"
echo "  poetry run python run_simple.py"
echo ""
echo "To see available options:"
echo "  ./run_pipeline.sh --help"
echo ""
echo "To check if all dependencies are installed:"
echo "  ./run_pipeline.sh --check-dependencies"
echo "========================================================"
