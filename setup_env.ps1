# Comprehensive setup script for multimodal data pipeline
# Installs all required libraries using Poetry

param(
    [switch]$Quick
)

Write-Host "========================================================"
Write-Host "Multimodal Data Pipeline - Comprehensive Setup"
Write-Host "========================================================"

# Function to check if a command exists
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check if Python 3.12 is installed
if (-not (Test-Command "python") -and -not (Test-Command "python3.12")) {
    Write-Host "Python 3.12 is required but not found." -ForegroundColor Red
    Write-Host "Please install Python 3.12 and try again." -ForegroundColor Red
    exit 1
}

# Determine Python command
$pythonCmd = if (Test-Command "python3.12") { "python3.12" } else { "python" }

# Check if Poetry is installed
if (-not (Test-Command "poetry")) {
    Write-Host "Poetry is required but not found." -ForegroundColor Yellow
    Write-Host "Installing Poetry..." -ForegroundColor Yellow
    
    # Install Poetry using the official installer
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | & $pythonCmd -
    
    # Add Poetry to PATH for current session
    $poetryPath = "$env:APPDATA\Python\Scripts"
    if (Test-Path $poetryPath) {
        $env:PATH = "$poetryPath;$env:PATH"
    }
    
    # Check if Poetry is now available
    if (-not (Test-Command "poetry")) {
        Write-Host "Poetry installation completed, but 'poetry' command not found in PATH." -ForegroundColor Yellow
        Write-Host "Please restart your terminal or add Poetry to your PATH manually." -ForegroundColor Yellow
        Write-Host "Poetry is typically installed to: $env:APPDATA\Python\Scripts" -ForegroundColor Yellow
        exit 1
    }
}

# Initialize Poetry project if pyproject.toml doesn't exist
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "Initializing Poetry project..." -ForegroundColor Green
    poetry init --no-interaction --name multimodal-data-pipeline --version 0.1.0 --description "Comprehensive multimodal data processing pipeline"
}

# Create a Poetry environment with Python 3.12
Write-Host "Creating Poetry environment with Python 3.12..." -ForegroundColor Green
# Configure Poetry to create .venv in project directory
poetry config virtualenvs.in-project true
poetry env use $pythonCmd

Write-Host "Installing core dependencies..." -ForegroundColor Green

# 1) Pin NumPy to <2.0 first for MediaPipe compatibility
Write-Host "-> Pinning NumPy to 1.x (for TensorFlow & MediaPipe compatibility)..." -ForegroundColor Cyan
poetry add "numpy>=1.26,<2.0"

# Core Python packages
Write-Host "-> Installing core Python packages..." -ForegroundColor Cyan
poetry add pandas matplotlib seaborn scikit-learn jupyter notebook ipython

# Computer Vision & Image Processing
Write-Host "-> Installing computer vision libraries..." -ForegroundColor Cyan
poetry add opencv-python pillow

# Audio Processing Libraries
Write-Host "-> Installing audio processing libraries..." -ForegroundColor Cyan
poetry add librosa soundfile pydub

# PyTorch ecosystem
Write-Host "-> Installing PyTorch ecosystem..." -ForegroundColor Cyan
poetry add torch torchvision torchaudio transformers

# Speech Processing Libraries
Write-Host "-> Installing speech processing libraries..." -ForegroundColor Cyan
poetry add speechbrain opensmile

# NLP and Text Processing
Write-Host "-> Installing NLP libraries..." -ForegroundColor Cyan
poetry add sentence-transformers

# Hugging Face ecosystem
Write-Host "-> Installing Hugging Face libraries..." -ForegroundColor Cyan
poetry add datasets accelerate

# Video Processing
Write-Host "-> Installing video processing libraries..." -ForegroundColor Cyan
poetry add ffmpeg ffmpeg-python moviepy

# Additional utilities
Write-Host "-> Installing additional utilities..." -ForegroundColor Cyan
poetry add tqdm requests pathlib2 python-dotenv

# GitHub-based repositories and specialized libraries
Write-Host "-> Installing GitHub-based and specialized libraries..." -ForegroundColor Cyan

# Speech Emotion Recognition (alternative implementation)
Write-Host "-> Installing speech emotion recognition dependencies..." -ForegroundColor Cyan
poetry add joblib scikit-learn

# Speech transcription (using stable OpenAI Whisper instead of WhisperX due to dependency conflicts)
Write-Host "-> Installing OpenAI Whisper for speech transcription..." -ForegroundColor Cyan
try {
    poetry add openai-whisper
} catch {
    Write-Host "OpenAI Whisper installation failed, trying alternative..." -ForegroundColor Yellow
    # Fallback to a basic speech recognition library
    try {
        poetry add SpeechRecognition
    } catch {
        Write-Host "Speech transcription libraries failed - you may need to install manually" -ForegroundColor Yellow
    }
}

# Additional audio processing libraries for speech analysis
Write-Host "-> Installing additional speech processing libraries..." -ForegroundColor Cyan
try {
    poetry add pyannote.audio
} catch {
    Write-Host "pyannote.audio installation failed - this is optional for speaker diarization" -ForegroundColor Yellow
}

# AudioStretchy for audio manipulation (PyPI version - simpler and more stable)
Write-Host "-> Installing AudioStretchy..." -ForegroundColor Cyan
try {
    poetry add "audiostretchy[all]"
} catch {
    Write-Host "PyPI installation failed, trying GitHub version..." -ForegroundColor Yellow
    poetry add git+https://github.com/twardoch/audiostretchy.git
}

# Heinsen Routing for sentiment analysis (GitHub repository)
Write-Host "-> Installing Heinsen Routing..." -ForegroundColor Cyan
poetry add git+https://github.com/glassroom/heinsen_routing.git

# Advanced ML libraries
Write-Host "-> Installing advanced ML libraries..." -ForegroundColor Cyan
poetry add tensorflow tensorflow-hub

# MediaPipe for pose estimation and face detection
Write-Host "-> Installing MediaPipe..." -ForegroundColor Cyan
poetry add mediapipe

# Additional computer vision libraries
Write-Host "-> Installing additional computer vision libraries..." -ForegroundColor Cyan
poetry add timm efficientnet-pytorch

# Data handling and storage
Write-Host "-> Installing data handling libraries..." -ForegroundColor Cyan
poetry add h5py pyarrow

# Optional: Try to install some research repositories (may fail, but we'll continue)
Write-Host "-> Attempting to install additional research libraries..." -ForegroundColor Cyan

# Try to install some additional libraries that might be available via pip
Write-Host "-> Installing additional optional libraries..." -ForegroundColor Cyan
try {
    poetry add dlib
} catch {
    Write-Host "dlib installation failed - this is optional (requires CMake & Visual Studio C++)" -ForegroundColor Yellow
}

try {
    poetry add face-recognition
} catch {
    Write-Host "face-recognition installation failed - this is optional (requires CMake & Visual Studio C++)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================================"
Write-Host "Core Installation Complete!" -ForegroundColor Green
Write-Host "========================================================"

# Verify critical installations
Write-Host "Verifying critical installations..." -ForegroundColor Green

$verificationScript = @"
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
    'mediapipe': 'mediapipe',
    'openai-whisper': 'whisper'
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

print(f'\nSuccessfully installed: {success_count}/{total_count} critical packages')

if success_count == total_count:
    print('\n🎉 All critical packages installed successfully!')
else:
    print('\n⚠️  Some packages failed to install. You may need to install them manually.')
"@

poetry run python -c $verificationScript

# Create necessary directories
Write-Host ""
Write-Host "Creating project directories..." -ForegroundColor Green
$directories = @("output", "output/audio", "output/audio/separated", "output/features", "data")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host ""
Write-Host "========================================================"
Write-Host "Setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Installed libraries include:" -ForegroundColor White
Write-Host "  • Core: NumPy, Pandas, Matplotlib, Scikit-learn" -ForegroundColor White
Write-Host "  • Computer Vision: OpenCV, Pillow, MediaPipe" -ForegroundColor White
Write-Host "  • Audio: Librosa, SoundFile, PyAudio, SpeechBrain" -ForegroundColor White
Write-Host "  • Deep Learning: PyTorch, Transformers, Sentence-Transformers" -ForegroundColor White
Write-Host "  • Speech: OpenAI Whisper, OpenSMILE" -ForegroundColor White
Write-Host "  • Video: FFmpeg-Python, MoviePy" -ForegroundColor White
Write-Host "  • NLP: Hugging Face ecosystem, TensorFlow Hub" -ForegroundColor White
Write-Host ""
Write-Host "To run the pipeline:" -ForegroundColor Yellow
Write-Host "  poetry run python run_pipeline.py" -ForegroundColor White
Write-Host ""
Write-Host "To see available options:" -ForegroundColor Yellow
Write-Host "  poetry run python run_pipeline.py --help" -ForegroundColor White
Write-Host ""
Write-Host "To check dependencies:" -ForegroundColor Yellow
Write-Host "  poetry run python run_pipeline.py --check-dependencies" -ForegroundColor White
Write-Host ""
Write-Host "To use the unified runner:" -ForegroundColor Yellow
Write-Host "  .\run_all.ps1 -Help" -ForegroundColor White
Write-Host "========================================================"
