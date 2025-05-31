# Manual Installation Guide

This document provides instructions for manually installing components that might require special handling.

## Core Dependencies

Most dependencies will be installed automatically by Poetry when running `./setup_env.sh`. However, some packages require additional steps.

## FFmpeg

FFmpeg is required for audio extraction from video files.

### macOS:
```bash
brew install ffmpeg
```

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

## Complex Dependencies

All dependencies including WhisperX are installed automatically by Poetry. However, if you encounter issues with WhisperX or other GitHub-based dependencies, you can try installing them manually:

```bash
# Make sure you are in the poetry environment
poetry shell

# Install WhisperX manually if needed
# poetry add git+https://github.com/m-bain/whisperx.git
```

## SpeechBrain Models

SpeechBrain will download models automatically, but if you want to predownload them:

```bash
# Create directory for pretrained models
mkdir -p pretrained_models

# SepFormer model
python -c "from speechbrain.pretrained import SepformerSeparation; model = SepformerSeparation.from_hparams(source='speechbrain/sepformer-libri3mix', savedir='pretrained_models/sepformer-libri3mix')"
```

## Other Models

Other models like XLSR will be downloaded automatically when first used. Make sure you have enough disk space (at least 10GB recommended).

## GPU Support (Optional)

For GPU acceleration:

1. Install CUDA and cuDNN according to your GPU and system requirements
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Update the `device` parameter to "cuda" when initializing the pipeline

## Additional Non-Python Dependencies

Some models might require additional system dependencies, especially for vision processing:

### macOS:
```bash
brew install cmake pkg-config
```

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential cmake pkg-config
```
