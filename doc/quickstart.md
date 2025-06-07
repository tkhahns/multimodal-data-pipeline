# Quick Start Guide

This guide provides a quick introduction to get you started with the Multimodal Data Pipeline.

## Setup

1. **Clone the repository and navigate to it**

2. **Run the setup script:**
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

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
      ```bash
      echo "HF_TOKEN=your_huggingface_token_here" > .env
      ```
   
   **Important**: Replace `your_huggingface_token_here` with your actual token. Without this setup, speaker diarization features will not work.

4. **Make sure you have video files in the `data/` directory**

## Basic Usage

Run the pipeline with default settings:
```bash
# Using the unified run script (recommended)
./run_pipeline.sh

# See all available options
./run_pipeline.sh --help

# List all available features
./run_pipeline.sh --list-features

# Check if all dependencies are properly installed
./run_pipeline.sh --check-dependencies
```

This will:
- Process all video files in the `data/` directory
- Extract audio from each video
- Extract all enabled features
- Save results to `output/[timestamp]/`

## Customizing the Pipeline

### Selecting Specific Features

To extract only specific features:
```bash
./run_pipeline.sh --features basic_audio,speech_emotion
```

Available features:
- `basic_audio`: Volume and pitch (OpenCV)
- `librosa_spectral`: Spectral features (Librosa)
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

**Note**: When no specific features are specified, all available features are extracted by default.

### Processing a Specific Directory

```bash
poetry run python run_simple.py --data-dir /path/to/your/videos
```

### Custom Output Location

```bash
poetry run python run_simple.py --output-dir /path/to/save/results
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
