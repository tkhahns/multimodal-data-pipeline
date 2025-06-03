# Multimodal Data Pipeline

A comprehensive toolkit for processing multimodal data across speech, vision, and text modalities. This pipeline extracts various audio features from video files, including basic audio characteristics, spectral features, speech emotion recognition, speaker separation, and speech-to-text transcription.

## Features

The pipeline currently supports the following audio feature extractors:

### Basic Audio Features (OpenCV)
- Audio volume (`oc_audvol`)
- Change in audio volume (`oc_audvol_diff`)
- Average audio pitch (`oc_audpit`)
- Change in audio pitch (`oc_audpit_diff`)

### Spectral Features (Librosa)
- Spectral centroid (`lbrs_spectral_centroid`)
- Spectral bandwidth (`lbrs_spectral_bandwidth`) 
- Spectral flatness (`lbrs_spectral_flatness`)
- Spectral rolloff (`lbrs_spectral_rolloff`)
- Zero crossing rate (`lbrs_zero_crossing_rate`)
- RMSE (`lbrs_rmse`)
- Tempo (`lbrs_tempo`)
- Single-value aggregations for each feature

### Speech Analysis
- Speech Emotion Recognition (`ser_*` emotion probabilities)
- Speech Separation (separated audio sources)
- Time-Accurate Speech Transcription with speaker diarization (WhisperX)
- Speech-to-Text transcription (XLSR and S2T models)

## Installation

### Prerequisites
- Python 3.12
- Poetry
- Git
- ffmpeg (required for audio extraction)

### Basic Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd multimodal-data-pipeline
   ```

2. Run the setup script to create the environment and install dependencies:
   ```
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

This will:
- Create a Poetry environment with Python 3.12
- Install all required dependencies
- Install optional dependencies like WhisperX (if possible)
- Set up necessary directories

## Usage

### Command Line

The easiest way to use the pipeline is through the unified run script:

```bash
# Using the shell wrapper (recommended)
./run_pipeline.sh

# Or using Poetry directly
poetry run python run_simple.py
```

This will process all video files in the `data/` directory and output results to `output/`.

#### Options

```
Usage: ./run_pipeline.sh [options]

Options:
  -d, --data-dir DIR    Directory with video/audio files (default: ./data)
  -o, --output-dir DIR  Output directory (default: ./output/YYYYMMDD_HHMMSS)
  -f, --features LIST   Comma-separated features to extract
                        Available: basic_audio,librosa_spectral,speech_emotion,
                                  speech_separation,whisperx,xlsr_speech,s2t_speech
  --list-features       List available features and exit
  --is-audio            Process files as audio instead of video
  --check-dependencies  Check if all required dependencies are installed
  --log-file FILE       Path to log file (default: <output_dir>/pipeline.log)
  -h, --help            Show this help message
```

#### Examples

Process all videos with all features:
```bash
./run_pipeline.sh
```

Process videos in a specific directory:
```bash
./run_pipeline.sh --data-dir /path/to/videos
```

Only extract basic audio and speech emotion features:
```bash
./run_pipeline.sh --features basic_audio,speech_emotion
```

Check if all dependencies are properly installed:
```bash
./run_pipeline.sh --check-dependencies
```

### Programmatic Usage

You can also use the pipeline programmatically in your Python code:

```python
from src.pipeline import MultimodalPipeline
from src.utils.audio_extraction import extract_audio_from_video

# Initialize the pipeline
pipeline = MultimodalPipeline(
    output_dir='output/my_results',
    features=['basic_audio', 'librosa_spectral', 'speech_emotion'],
    device='cpu'  # Use 'cuda' if you have a compatible GPU
)

# Process a video file
video_path = 'data/my_video.mp4'
results = pipeline.process_video_file(video_path)

# Or process an audio file directly
audio_path = 'data/my_audio.wav'
results = pipeline.process_audio_file(audio_path)

# Or process a whole directory
results = pipeline.process_directory('data/', is_video=True)
```

## Output Format

The pipeline generates the following outputs:

1. Extracted audio files (in `output/audio/`)
2. Feature JSONs with all computed features:
   - Individual JSON files per audio/video file with video name as the first key (in `output/features/`)
   - Complete JSON files with detailed feature information (in `output/features/`)
   - Consolidated JSON file with features from all files (`output/pipeline_features.json`)
3. Parquet files for tabular data (in `output/features/`)
4. Separate NPY files for large numpy arrays (in `output/features/`)

## Model Categories

- **Speech**: Speech emotion recognition, transcription, and audio feature extraction
- **Vision**: Pose estimation, facial expression analysis, and motion tracking (coming soon)
- **Text**: Sentence embeddings, contextual representations, and semantic analysis (coming soon)
- **Multimodal**: Combined audio-visual analysis and integration (coming soon)

## License

See the LICENSE file for details.
