# Quick Start Guide

This guide provides a quick introduction to get you started with the Multimodal Data Pipeline.

## Setup

1. **Clone the repository and navigate to it**

2. **Install FFmpeg** (required for video/audio processing):

   **WSL/Linux:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

   **macOS:**
   ```bash
   brew install ffmpeg
   ```

   **Windows:**
   - Download from https://ffmpeg.org/download.html
   - Add to system PATH
   - Or use: `choco install ffmpeg`

3. **Run the setup script:**

   **Linux/macOS/WSL:**
   ```bash
   chmod +x run_all.sh
   ./run_all.sh --setup
   ```

   **Windows (Native PowerShell):**
   ```powershell
   .\run_all.ps1 -Setup
   ```

   **Important**: Use the correct script for your environment:
   - In WSL/Linux/macOS: Use `./run_all.sh`
   - In Windows PowerShell: Use `.\run_all.ps1`

3. **Set up HuggingFace authentication** (required for speaker diarization):
   
   a. **Create a HuggingFace account** at https://huggingface.co/join
   
   b. **Generate an access token** at https://huggingface.co/settings/tokens:
      - Click "New token"
      - Choose "Read" access 
      - Copy the generated token
   
   c. **Accept model licenses** (required):
      - Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and click "Agree"
      - Visit https://huggingface.co/pyannote/segmentation-3.0 and click "Agree"
     d. **Create authentication file**:
   
      **Linux/macOS:**
      ```bash
      echo "HF_TOKEN=your_huggingface_token_here" > .env
      ```
      
      **Windows (PowerShell):**
      ```powershell
      "HF_TOKEN=your_huggingface_token_here" | Out-File -FilePath .env -Encoding utf8
      ```
   
   **Important**: Replace `your_huggingface_token_here` with your actual token. Without this setup, speaker diarization features will not work.

4. **Make sure you have video files in the `data/` directory**

## Basic Usage

Run the pipeline with default settings:

**Linux/macOS/WSL:**
```bash
# Using the unified run script (recommended)
./run_all.sh

# See all available options
./run_all.sh --help

# List all available features
./run_all.sh --list-features

# Check if all dependencies are properly installed
./run_all.sh --check-deps
```

**Windows (Native PowerShell):**
```powershell
# Using the unified run script (recommended)
.\run_all.ps1

# See all available options
.\run_all.ps1 -Help

# List all available features
.\run_all.ps1 -ListFeatures

# Check if all dependencies are properly installed
.\run_all.ps1 -CheckDeps
```

This will:
- Process all video files in the `data/` directory
- Extract audio from each video
- Extract all enabled features
- Save results to `output/[timestamp]/`

## Customizing the Pipeline

### Selecting Specific Features

To extract only specific features:

**Linux/macOS:**
```bash
./run_all.sh --features basic_audio,speech_emotion
```

**Windows (PowerShell):**
```powershell
.\run_all.ps1 -Features "basic_audio,speech_emotion"
```

Available features:
- `basic_audio`: Volume and pitch (OpenCV)
- `librosa_spectral`: Spectral features (Librosa)
- `audiostretchy`: AudioStretchy time-stretching analysis
  - Analyzes audio for high-quality time-stretching parameters
  - Generates stretching configuration and predicted output characteristics
- `opensmile`: OpenSMILE Low-Level Descriptors and Functionals
- `speech_emotion`: Speech emotion recognition
- `heinsen_sentiment`: Heinsen routing sentiment analysis with capsule networks
- `speech_separation`: Speech source separation
- `whisperx_transcription`: WhisperX transcription with diarization
  - Uses OpenAI Whisper for speech-to-text
  - Uses pyannote.audio models for speaker identification
- `deberta_text`: DeBERTa benchmark performance analysis
  - Processes transcribed text from WhisperX or other sources
  - Computes performance metrics for SQuAD, MNLI, SST-2, QNLI, CoLA, RTE, MRPC, QQP, STS-B
- `simcse_text`: SimCSE contrastive learning analysis
  - Processes transcribed text from WhisperX or other sources
  - Computes STS benchmark correlations for sentence embedding quality
- `albert_text`: ALBERT language representation analysis
  - Processes transcribed text from WhisperX or other sources
  - Computes performance across 12 NLP benchmarks including GLUE tasks, SQuAD, and RACE
- `sbert_text`: Sentence-BERT dense vector representations and reranking
  - Processes transcribed text from WhisperX or other sources
  - Generates dense embeddings, correlation matrices, and reranker scores
- `use_text`: Universal Sentence Encoder for text classification and semantic analysis
  - Processes transcribed text from WhisperX or other sources
  - Generates fixed-length 512-dimensional embeddings and semantic similarity metrics
- `meld_emotion`: MELD emotion recognition for conversation analysis
  - Processes transcribed text with speaker information for multi-party dialogue emotion analysis
  - Analyzes conversation patterns, emotion distribution, and speaker interactions
- `pare_vision`: PARE 3D human body estimation and pose analysis
  - Processes video frames for comprehensive 3D human body estimation
  - Generates SMPL model parameters, 3D mesh vertices, and joint positions
- `vitpose_vision`: ViTPose Vision Transformer pose estimation
  - Processes video frames using Vision Transformers for pose estimation
  - Generates precision, recall, uncertainty, and overall performance metrics
- `psa_vision`: PSA keypoint heatmaps and segmentation masks
  - Processes video frames using Polarized Self-Attention mechanisms
  - Generates Average Precision and mean IoU metrics for keypoint and segmentation analysis

**Note**: When no specific features are specified, all available features are extracted by default.

### Processing a Specific Directory

**Linux/macOS:**
```bash
./run_all.sh --data-dir /path/to/your/videos
```

**Windows (PowerShell):**
```powershell
.\run_all.ps1 -DataDir "C:\path\to\your\videos"
```

### Custom Output Location

**Linux/macOS:**
```bash
./run_all.sh --output-dir /path/to/save/results
```

**Windows (PowerShell):**
```powershell
.\run_all.ps1 -OutputDir "C:\path\to\save\results"
```

## Using in Python Code

You can use the pipeline in two ways:

### Option 1: MultimodalFeatureExtractor (Recommended)

```python
from src.feature_extractor import MultimodalFeatureExtractor

# Initialize the extractor
extractor = MultimodalFeatureExtractor(
    features=['basic_audio', 'whisperx_transcription', 'deberta_text'],
    device='cpu'  # Use 'cuda' for GPU acceleration
)

# Process a video file (extracts audio, transcribes, and analyzes text)
features = extractor.extract_features('path/to/video.mp4')

# Process existing text data
text_data = {"transcript": "This is some text to analyze"}
features = extractor.extract_features(text_data)
```

### Option 2: Direct Pipeline Usage

```python
from src.pipeline import MultimodalPipeline

# Initialize the pipeline
pipeline = MultimodalPipeline(
    output_dir='my_results',
    features=['basic_audio', 'speech_emotion', 'deberta_text'],
    device='cpu'  # Use 'cuda' for GPU acceleration
)

# Process a single video
results = pipeline.process_video_file('path/to/video.mp4')

# Process a directory of videos
results = pipeline.process_directory('path/to/videos', is_video=True)
```

## Understanding the Output

The pipeline generates:

1. **Extracted audio files** in `[output_dir]/audio/`
2. **Feature files** in `[output_dir]/features/`:
   - JSON files with all features
   - Parquet files for tabular data
   - NPY files for large arrays
3. **Log file** in `[output_dir]/pipeline.log`

## Next Steps

- See the full README.md for detailed information
- Check doc/manual_installation.md for advanced installation options
- Explore src/example.py for programming examples

## Troubleshooting

### Script Execution Issues (WSL/Linux)

If you get the error `/bin/bash^M: bad interpreter` when running bash scripts:

```bash
# Fix line endings
dos2unix run_all.sh setup_env.sh

# Make executable
chmod +x run_all.sh setup_env.sh
```

This error occurs when files have Windows line endings instead of Unix line endings.

### Script Compatibility Issues

**Error**: `syntax error near unexpected token` when trying to run scripts

**Solution**: Use the correct script for your environment:
- **In WSL/Linux/macOS**: Use `./run_all.sh` (bash script)
- **In Windows PowerShell**: Use `.\run_all.ps1` (PowerShell script)

**Never**: Run `.ps1` files in bash or `.sh` files in PowerShell.

### FFmpeg Installation Issues

**Error**: `ffmpeg is not installed or not in PATH`

**Solution**: Install the FFmpeg system binary:

**WSL/Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to system PATH

**Verify:**
```bash
ffmpeg -version
```
