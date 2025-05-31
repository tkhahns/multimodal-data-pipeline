# Quick Start Guide

This guide provides a quick introduction to get you started with the Multimodal Data Pipeline.

## Setup

1. **Clone the repository and navigate to it**

2. **Run the setup script:**
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

3. **Make sure you have video files in the `data/` directory**

## Basic Usage

Run the pipeline with default settings:
```bash
# Using the simple Python script with Poetry (recommended)
poetry run python run_simple.py

# See all available options
poetry run python run_simple.py --help

# List all available features
poetry run python run_simple.py --list-features
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
poetry run python run_simple.py --features basic_audio,speech_emotion
```

Available features:
- `basic_audio`: Volume and pitch (OpenCV)
- `librosa_spectral`: Spectral features (Librosa)
- `speech_emotion`: Speech emotion recognition
- `speech_separation`: Speech source separation
- `whisperx`: WhisperX transcription with diarization
- `xlsr_speech`: XLSR speech-to-text
- `s2t_speech`: S2T speech-to-text

### Processing a Specific Directory

```bash
poetry run python run_simple.py --data-dir /path/to/your/videos
```

### Custom Output Location

```bash
poetry run python run_simple.py --output-dir /path/to/save/results
```

## Using in Python Code

```python
from src.pipeline import MultimodalPipeline

# Initialize the pipeline
pipeline = MultimodalPipeline(
    output_dir='my_results',
    features=['basic_audio', 'speech_emotion'],
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
