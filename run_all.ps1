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
    
    if (-not (Test-Path "setup_env.ps1")) {
        Write-Error "setup_env.ps1 not found!"
        exit 1
    }
    
    switch ($SetupType) {
        "full" {
            Write-Status "Running full setup (including optional packages)..."
            try {
                & .\setup_env.ps1
            }
            catch {
                Write-Error "Setup failed: $_"
                exit 1
            }
        }
        "quick" {
            Write-Status "Running quick setup (essential packages only)..."
            try {
                & .\setup_env.ps1 -Quick
            }
            catch {
                Write-Error "Setup failed: $_"
                exit 1
            }
        }
        default {
            Write-Status "Running default setup..."
            try {
                & .\setup_env.ps1
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
  -Features LIST       Comma-separated features to extract                       Available: basic_audio,librosa_spectral,opensmile,
                                 speech_emotion,heinsen_sentiment,speech_separation,
                                 whisperx_transcription,deberta_text,simcse_text,
                                 albert_text,sbert_text,use_text
  -ListFeatures        List available features and exit
  -IsAudio             Process files as audio instead of video
  -LogFile FILE        Path to log file (default: <output_dir>\pipeline.log)
  -Help                Show this help message

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
    
    if ($ListFeatures) {
        $pipelineArgs += @("--list-features")
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
