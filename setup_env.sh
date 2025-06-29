#!/bin/bash
# Comprehensive setup script for multimodal data pipeline
# Installs all required libraries using Poetry

echo "========================================================"
echo "Multimodal Data Pipeline - Comprehensive Setup"
echo "========================================================"

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg is not installed or not in PATH."
    echo "FFmpeg is required for video/audio processing."
    echo ""
    echo "Please install FFmpeg:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  CentOS/RHEL:   sudo yum install ffmpeg"
    echo "  Fedora:        sudo dnf install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo "  WSL:           sudo apt install ffmpeg"
    echo ""
    echo "Continuing with setup, but video processing may fail..."
    echo ""
fi

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

# Initialize Poetry project if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    echo "Initializing Poetry project..."
    poetry init --no-interaction --name multimodal-data-pipeline --version 0.1.0 --description "Comprehensive multimodal data processing pipeline"
fi

# Create a Poetry environment with Python 3.12
echo "Creating Poetry environment with Python 3.12..."
# Configure Poetry to create .venv in project directory
poetry config virtualenvs.in-project true
poetry env use python3.12

echo "Installing core dependencies..."

# 1) Pin NumPy to <2.0 first for MediaPipe compatibility
echo "-> Pinning NumPy to 1.x (for TensorFlow & MediaPipe compatibility)..."
poetry add "numpy>=1.26,<2.0"

# Core Python packages
echo "-> Installing core Python packages..."
poetry add pandas matplotlib seaborn scikit-learn jupyter notebook ipython

# Computer Vision & Image Processing
echo "-> Installing computer vision libraries..."
poetry add opencv-python pillow

# Audio Processing Libraries
echo "-> Installing audio processing libraries..."
poetry add librosa soundfile pydub

# PyTorch ecosystem
echo "-> Installing PyTorch ecosystem..."
poetry add torch torchvision torchaudio transformers

# Speech Processing Libraries
echo "-> Installing speech processing libraries..."
poetry add speechbrain opensmile

# NLP and Text Processing
echo "-> Installing NLP libraries..."
poetry add sentence-transformers

# NLP and Text Processing
echo "-> Installing NLP libraries..."
poetry add sentence-transformers

# Hugging Face ecosystem
echo "-> Installing Hugging Face libraries..."
poetry add datasets accelerate

# Video Processing
echo "-> Installing video processing libraries..."
poetry add ffmpeg ffmpeg-python moviepy

# Additional utilities
echo "-> Installing additional utilities..."
poetry add tqdm requests pathlib2 python-dotenv

# GitHub-based repositories and specialized libraries
echo "-> Installing GitHub-based and specialized libraries..."

# Speech Emotion Recognition (alternative implementation)
echo "-> Installing speech emotion recognition dependencies..."
poetry add joblib scikit-learn

# WhisperX for speech transcription (handle NumPy compatibility issues)
echo "-> Installing WhisperX (handling NumPy compatibility)..."
# WhisperX has complex dependency conflicts with transformers and numpy
# Try specific compatible versions first
poetry add "whisperx==3.1.1" || {
    echo "WhisperX 3.1.1 failed, trying 3.1.0..."
    poetry add "whisperx==3.1.0" || {
        echo "WhisperX 3.1.0 failed, trying to install with relaxed constraints..."
        # Try installing with specific compatible transformers version
        poetry add "transformers>=4.21.0,<4.25.0" "whisperx>=3.1,<3.2" || {
            echo "WhisperX with compatible transformers failed, trying manual approach..."
            # Install core dependencies separately
            poetry add "faster-whisper>=0.9.0,<1.0.0" "pyannote.audio" || {
                echo "WhisperX installation completely failed - installing OpenAI Whisper as fallback..."
                poetry add openai-whisper || {
                    echo "All whisper installations failed - will skip speech transcription"
                    echo "You may need to install transcription libraries manually later"
                    echo "Alternatives:"
                    echo "  - poetry add openai-whisper"
                    echo "  - poetry add faster-whisper"
                    echo "  - Use SpeechBrain (already installed) for basic speech processing"
                }
            }
        }
    }
}

# AudioStretchy for audio manipulation (PyPI version - simpler and more stable)
echo "-> Installing AudioStretchy..."
poetry add "audiostretchy[all]" || {
    echo "PyPI installation failed, trying GitHub version..."
    poetry add git+https://github.com/twardoch/audiostretchy.git
}

# Heinsen Routing for sentiment analysis (GitHub repository)
echo "-> Installing Heinsen Routing..."
poetry add git+https://github.com/glassroom/heinsen_routing.git

# Advanced ML libraries
echo "-> Installing advanced ML libraries..."
poetry add tensorflow tensorflow-hub

# MediaPipe for pose estimation and face detection
echo "-> Installing MediaPipe..."
poetry add mediapipe

# Additional computer vision libraries
echo "-> Installing additional computer vision libraries..."
poetry add timm efficientnet-pytorch

# Data handling and storage
echo "-> Installing data handling libraries..."
poetry add h5py pyarrow

# Optional: Try to install some research repositories (may fail, but we'll continue)
echo "-> Attempting to install additional research libraries..."

# Try to install some additional libraries that might be available via pip
echo "-> Installing additional optional libraries..."
poetry add dlib || echo "dlib installation failed - this is optional"
poetry add face-recognition || echo "face-recognition installation failed - this is optional"

echo ""
echo "========================================================"
echo "Core Installation Complete!"
echo "========================================================"

# Verify critical installations
echo "Verifying critical installations..."
poetry run python -c "
import importlib
import sys

# Check for critical packages
critical_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas', 
    'librosa': 'librosa',
    'opencv-python': 'cv2',
    'torch': 'torch',
    'torchaudio': 'torchaudio',
    'transformers': 'transformers',
    'speechbrain': 'speechbrain',
    'sentence-transformers': 'sentence_transformers',
    'ffmpeg-python': 'ffmpeg',
    'mediapipe': 'mediapipe'
}

# Optional packages that may fail
optional_packages = {
    'whisperx': 'whisperx',
    'faster-whisper': 'faster_whisper',
    'openai-whisper': 'whisper',
    'audiostretchy': 'audiostretchy'
}

print('Critical Package Verification:')
print('=' * 40)
success_count = 0
total_count = len(critical_packages)

for package_name, import_name in critical_packages.items():
    try:
        importlib.import_module(import_name)
        print(f'✓ {package_name}')
        success_count += 1
    except ImportError as e:
        print(f'✗ {package_name} - {e}')

print(f'\\nCritical packages: {success_count}/{total_count} installed successfully')

print('\\nOptional Package Verification:')
print('=' * 40)
optional_success = 0
optional_total = len(optional_packages)

for package_name, import_name in optional_packages.items():
    try:
        importlib.import_module(import_name)
        print(f'✓ {package_name}')
        optional_success += 1
    except ImportError as e:
        print(f'- {package_name} (optional) - not available')

print(f'\\nOptional packages: {optional_success}/{optional_total} installed')

if success_count == total_count:
    print('\\n🎉 All critical packages installed successfully!')
    if optional_success < optional_total:
        print('⚠️  Some optional packages are missing but the pipeline will still work.')
        print('   You can install missing packages manually if needed.')
else:
    print('\\n⚠️  Some critical packages failed to install. The pipeline may not work correctly.')
    print('   Please check the errors above and install missing packages manually.')

# Additional guidance for common issues
if optional_success == 0:
    print('\\n💡 WhisperX Installation Tips:')
    print('   If you need WhisperX for transcription, try installing manually:')
    print('   1. poetry add openai-whisper (simpler alternative)')
    print('   2. Or try: poetry add whisperx==3.1.1 --with transformers==4.21.0')
    print('   3. For speech transcription, you can also use SpeechBrain which is already installed')
"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p output
mkdir -p output/audio
mkdir -p output/audio/separated
mkdir -p output/features
mkdir -p data

# Make the run scripts executable
chmod +x run_pipeline.py
chmod +x run_all.sh

echo ""
echo "========================================================"
echo "Setup completed!"
echo ""
echo "Installed libraries include:"
echo "  • Core: NumPy, Pandas, Matplotlib, Scikit-learn"
echo "  • Computer Vision: OpenCV, Pillow, MediaPipe"
echo "  • Audio: Librosa, SoundFile, PyAudio, SpeechBrain"
echo "  • Deep Learning: PyTorch, Transformers, Sentence-Transformers"
echo "  • Speech: WhisperX, OpenSMILE"
echo "  • Video: FFmpeg-Python, MoviePy"
echo "  • NLP: Hugging Face ecosystem, TensorFlow Hub"
echo ""
echo "To run the pipeline:"
echo "  poetry run python run_pipeline.py"
echo ""
echo "To see available options:"
echo "  poetry run python run_pipeline.py --help"
echo ""
echo "To check dependencies:"
echo "  poetry run python run_pipeline.py --check-dependencies"
echo ""
echo "To use the unified runner:"
echo "  ./run_all.sh --help"