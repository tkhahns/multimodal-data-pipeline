# Unified wrapper script for the multimodal data pipeline
# PowerShell equivalent of run_all.sh

param(
    [switch]$Setup,
    [switch]$SetupQuick,
    [switch]$CheckDeps,
    [string]$DataDir,
    [string]$OutputDir,
    [string]$Features,
    [switch]$ListFeatures,
    [switch]$IsAudio,
    [string]$LogFile,
    [switch]$Help
)

# Error handling - stop on errors
$ErrorActionPreference = "Stop"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Show available features list
function Show-Features {
    Write-Host "`nAvailable Features:`" -ForegroundColor Blue
    Write-Host "`n📢 Audio Features:"; \
        Write-Host "  • basic_audio          - Basic audio properties (duration, sample rate, channels)"
        Write-Host "  • librosa_spectral     - Advanced spectral features using librosa"
        Write-Host "  • opensmile            - OpenSMILE feature extraction"
        Write-Host "  • audiostretchy        - Audio stretching and time modification"
    Write-Host "`n🎤 Speech & Emotion Features:"; \
        Write-Host "  • speech_emotion       - Speech emotion recognition"
        Write-Host "  • heinsen_sentiment    - Heinsen routing sentiment analysis"
        Write-Host "  • meld_emotion         - MELD emotion recognition"
        Write-Host "  • speech_separation    - Speech source separation"
    Write-Host "`n📝 Transcription Features:"; \
        Write-Host "  • whisperx_transcription - WhisperX transcription with speaker diarization"
    Write-Host "`n📄 Text Analysis Features:"; \
        Write-Host "  • deberta_text         - DeBERTa text analysis"
        Write-Host "  • simcse_text          - SimCSE sentence embeddings"
        Write-Host "  • albert_text          - ALBERT text analysis"
        Write-Host "  • sbert_text           - Sentence-BERT embeddings"
        Write-Host "  • use_text             - Universal Sentence Encoder"
    Write-Host "`n👁️ Computer Vision Features:"; \
        Write-Host "  • emotieffnet_vision   - EmotiEffNet facial emotion recognition"
        Write-Host "  • mediapipe_pose_vision - Google MediaPipe pose estimation"
        Write-Host "  • deep_hrnet_vision    - Deep High-Resolution pose estimation"
        Write-Host "  • simple_baselines_vision - Simple Baselines pose estimation"
        Write-Host "  • ganimation_vision    - GANimation facial movements"
        Write-Host "  • arbex_vision         - ARBEx emotion extraction"
        Write-Host "  • openpose_vision      - OpenPose keypoint detection"
        Write-Host "  • instadm_vision       - Insta-DM dense motion estimation"
        Write-Host "  • optical_flow_vision  - Optical flow movement estimation"
        Write-Host "  • crowdflow_vision     - CrowdFlow person trajectories"
        Write-Host "  • videofinder_vision   - VideoFinder object/people location (requires Ollama)"
        Write-Host "  • smoothnet_vision     - SmoothNet pose estimation"
        Write-Host "  • lanegcn_vision       - LaneGCN autonomous driving"
        Write-Host "  • pare_vision          - PARE 3D human body estimation"
        Write-Host "  • vitpose_vision       - ViTPose estimation"
        Write-Host "  • psa_vision           - PSA pose estimation"
        Write-Host "  • rsn_vision           - RSN pose estimation"
        Write-Host "  • me_graphau_vision    - ME-GraphAU micro-expression"
        Write-Host "  • dan_vision           - DAN emotion recognition"
    Write-Host ""
    Write-Host "Notes:" -ForegroundColor Blue
    Write-Host "- videofinder_vision requires Ollama to be installed and running"
    Write-Host "- Py-Feat (pyfeat_vision) is excluded in this build"
}

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

# Function to run setup
function Invoke-Setup {
    param([string]$SetupType)
    
    Write-Status "Running environment setup..."
    
    if (-not (Test-Path "poetry")) {
        Write-Error "poetry not found!"
        exit 1
    }
    
    switch ($SetupType) {
        "full" {
            Write-Status "Running full setup (including optional packages)..."
            try {
                poetry install
            }
            catch {
                Write-Error "Setup failed: $_"
                exit 1
            }
        }
        "quick" {
            Write-Status "Running quick setup (essential packages only)..."
            try {
                poetry install --only main
            }
            catch {
                Write-Error "Setup failed: $_"
                exit 1
            }
        }
        default {
            Write-Status "Running default setup..."
            try {
                poetry install
            }
            catch {
                Write-Error "Setup failed: $_"
                exit 1
            }
        }
    }
    
    Write-Success "Setup completed successfully!"
}

# Function to check dependencies
function Test-Dependencies {
    Write-Status "Checking dependencies..."
    
    # Check if Poetry is installed
    if (-not (Test-Command "poetry")) {
        Write-Error "Poetry is required but not found."
        Write-Status "Please run: .\run_all.ps1 -Setup"
        return $false
    }
    
    # Check if Python script exists
    if (-not (Test-Path "run_pipeline.py")) {
        Write-Error "run_pipeline.py script not found."
        return $false
    }
    
    # Check if poetry environment is set up
    try {
        poetry env info | Out-Null
    }
    catch {
        Write-Warning "Poetry environment not found."
        Write-Status "Please run: .\run_all.ps1 -Setup"
        return $false
    }
    
    Write-Success "All dependencies check passed!"
    return $true
}

# Function to show help
function Show-Help {
    Write-Host @"
Usage: .\run_all.ps1 [options]

Setup Options:
  -Setup               Run full environment setup
  -SetupQuick          Run quick setup (skip optional packages)  
  -CheckDeps           Check if dependencies are installed

Pipeline Options:
  -DataDir DIR         Directory with video/audio files (default: .\data)
  -OutputDir DIR       Output directory (default: .\output\YYYYMMDD_HHMMSS)
    -Features LIST       Comma-separated features to extract
  -ListFeatures        List available features and exit
  -IsAudio             Process files as audio instead of video
  -LogFile FILE        Path to log file (default: <output_dir>\pipeline.log)
  -Help                Show this help message

Notes:
    - videofinder_vision requires Ollama to be installed and running
    - Py-Feat (pyfeat_vision) is excluded in this build

Examples:
  .\run_all.ps1 -Setup                           # Set up the environment
  .\run_all.ps1                                  # Run with default settings
  .\run_all.ps1 -CheckDeps                       # Check dependencies
  .\run_all.ps1 -DataDir "C:\path\to\videos"     # Process specific directory
  .\run_all.ps1 -Features "basic_audio,speech_emotion"  # Extract specific features

"@
}

# Main execution logic
function Main {
    Write-Status "Multimodal Data Pipeline - Unified Runner"
    
    # Handle help
    if ($Help) {
        Show-Help
        return
    }
    
    # Handle setup mode
    if ($Setup) {
        Invoke-Setup "full"
        return
    }
    
    if ($SetupQuick) {
        Invoke-Setup "quick"
        return
    }
    
    # Handle dependency check
    if ($CheckDeps) {
        if (Test-Dependencies) {
            exit 0
        } else {
            exit 1
        }
    }

    # Handle list features
    if ($ListFeatures) {
        Show-Features
        return
    }
    
    # Check dependencies before running pipeline
    if (-not (Test-Dependencies)) {
        Write-Error "Dependencies check failed!"
        Write-Status "Run '.\run_all.ps1 -Setup' to install dependencies."
        exit 1
    }
    
    # Build arguments for the pipeline
    $pipelineArgs = @()
    
    if ($DataDir) {
        $pipelineArgs += @("--data-dir", $DataDir)
    }
    
    if ($OutputDir) {
        $pipelineArgs += @("--output-dir", $OutputDir)
    }
    
    if ($Features) {
        $pipelineArgs += @("--features", $Features)
    }
    
    if ($IsAudio) {
        $pipelineArgs += @("--is-audio")
    }
    
    if ($LogFile) {
        $pipelineArgs += @("--log-file", $LogFile)
    }
    
    # Execute the pipeline
    Write-Status "Running multimodal data pipeline..."
    Write-Status "Arguments: $($pipelineArgs -join ' ')"

    # Optional preflight: warn if videofinder_vision is selected without Ollama
    if ($Features -and ($Features -like "*videofinder_vision*")) {
        if (-not (Test-Command "ollama")) {
            Write-Warning "videofinder_vision selected but 'ollama' not found. Please install/start Ollama."
        }
    }
    
    try {
        poetry run python run_pipeline.py @pipelineArgs
        Write-Success "Pipeline execution completed successfully!"
    }
    catch {
        Write-Error "Pipeline execution failed: $_"
        exit $LASTEXITCODE
    }
}

# Run main function
Main
