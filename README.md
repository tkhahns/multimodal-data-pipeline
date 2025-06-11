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

### OpenSMILE Features
- Low-Level Descriptors (LLDs): Energy, spectral features, MFCCs, pitch, voice quality, LSFs
- Functional Statistics: Mean, std, percentiles, skewness, kurtosis, regression coefficients
- Uses ComParE 2016 feature set with `osm_*` prefix
- Extracts 700+ comprehensive audio features including time-series and statistical summaries

### AudioStretchy Analysis
- High-quality time-stretching analysis of WAV/MP3 files without changing pitch
- Features with `AS_*` prefix for time-stretching parameter analysis:
  - **Stretching Parameters**: Ratio, gap ratio, frequency bounds, buffer settings
  - **Detection Settings**: Fast detection, normal detection, double range options
  - **Audio Characteristics**: Sample rate, channels, frame counts, duration analysis
  - **Output Calculations**: Predicted output duration, frame counts, and ratios
- Utilizes AudioStretchy library for professional audio time-stretching analysis
- Provides comprehensive analysis without actually performing time-stretching
- Returns 16 single-value features for stretching configuration and audio properties

### Speech Analysis
- Speech Emotion Recognition (`ser_*` emotion probabilities)
- Speech Separation (separated audio sources)
- Time-Accurate Speech Transcription with speaker diarization (WhisperX)
  - Uses OpenAI Whisper models for transcription
  - Uses pyannote.audio models for speaker diarization:
    - `pyannote/speaker-diarization-3.1`
    - `pyannote/segmentation-3.0`

### Text Analysis (DeBERTa)
- Comprehensive benchmark performance metrics using DeBERTa model
- Features with `DEB_*` prefix for downstream task performance:
  - **SQuAD 1.1/2.0**: Reading comprehension (F1 and Exact Match scores)
  - **MNLI**: Natural Language Inference (matched/mismatched accuracy)
  - **SST-2**: Sentiment Classification (binary accuracy)
  - **QNLI**: Question Natural Language Inference (accuracy)
  - **CoLA**: Linguistic Acceptability (Matthews Correlation Coefficient)
  - **RTE**: Recognizing Textual Entailment (accuracy)
  - **MRPC**: Microsoft Research Paraphrase Corpus (accuracy and F1)
  - **QQP**: Quora Question Pairs (accuracy and F1)
  - **STS-B**: Semantic Textual Similarity (Pearson and Spearman correlations)
- Automatically processes transcribed text from WhisperX or other text sources
- Returns default performance metrics when no text is available

### Text Analysis (SimCSE)
- Contrastive learning framework for sentence embeddings
- Features with `CSE_*` prefix for STS benchmark performance:
  - **STS12-16**: Semantic Textual Similarity benchmarks 2012-2016
  - **STSBenchmark**: Main STS benchmark dataset
  - **SICKRelatedness**: Semantic relatedness evaluation
  - **Average**: Mean performance across all benchmarks
- Utilizes SimCSE (Simple Contrastive Learning of Sentence Embeddings) model
- Automatically processes transcribed text from WhisperX or other text sources
- Returns correlation scores indicating embedding quality

### Text Analysis (ALBERT)
- Language representation analysis using ALBERT (A Lite BERT)
- Features with `alb_*` prefix for comprehensive NLP benchmark performance:
  - **GLUE Tasks**: MNLI, QNLI, QQP, RTE, SST, MRPC, CoLA, STS
  - **SQuAD 1.1/2.0**: Reading comprehension (dev and test sets)
  - **RACE**: Reading comprehension for middle/high school levels
- Utilizes ALBERT's parameter-sharing architecture for efficient language understanding
- Automatically processes transcribed text from WhisperX or other text sources
- Returns single-value performance metrics across 12 benchmark tasks

### Text Analysis (Sentence-BERT)
- Dense vector representations and reranking capabilities
- Features with `BERT_*` prefix for embedding analysis and passage ranking:
  - **Dense Embeddings**: Correlational matrices for sentences and paragraphs
  - **Reranking Scores**: Cross-encoder scores for query-passage relevance
  - **Tensor Representations**: Flattened correlation matrices with shape metadata
- Utilizes Sentence-BERT (SBERT) with Siamese BERT-Networks architecture
- Automatically processes transcribed text from WhisperX or other text sources
- Returns embeddings, similarity matrices, and reranker scores for semantic analysis

### Text Analysis (Universal Sentence Encoder)
- Text classification, semantic similarity, and semantic clustering
- Features with `USE_*` prefix for embedding and semantic analysis:
  - **Fixed-Length Embeddings**: 512-dimensional vectors for any input text length
  - **Sentence Embeddings**: Individual embeddings for each sentence (USE_embed_sentence1, USE_embed_sentence2, etc.)
  - **Semantic Similarity**: Cosine similarity metrics between sentences
  - **Clustering Metrics**: Centroid distance, spread variance, and pairwise distances
- Utilizes Google's Universal Sentence Encoder from TensorFlow Hub
- Automatically processes transcribed text from WhisperX or other text sources
- Returns comprehensive embeddings and semantic analysis for classification and clustering tasks

## Installation

### Prerequisites
- Python 3.12
- Poetry
- Git

### HuggingFace Setup (Required)

This pipeline uses several HuggingFace models for speech processing. You'll need to:

1. **Create a HuggingFace account** at https://huggingface.co/join
2. **Generate an access token** at https://huggingface.co/settings/tokens
   - Click "New token"
   - Choose "Read" access (sufficient for most models)
   - Copy the generated token
3. **Accept model licenses** (required for some models):
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and click "Agree"
   - Visit https://huggingface.co/pyannote/segmentation-3.0 and click "Agree"
4. **Set up authentication** by creating a `.env` file:
   ```bash
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

**Note**: Without proper HuggingFace authentication, speaker diarization and some transcription features will not work.

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
   This will automatically install ffmpeg and all other required dependencies via Poetry.

3. Set up HuggingFace authentication (required for speaker diarization):
   ```bash
   # Create a .env file with your HuggingFace token
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```
   Get your token from https://huggingface.co/settings/tokens and make sure you've accepted the required model licenses (see Prerequisites section above).

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
                        Available: basic_audio,librosa_spectral,opensmile,
                                  speech_emotion,heinsen_sentiment,speech_separation,
                                  whisperx_transcription,deberta_text
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

Extract text analysis along with audio features:
```bash
./run_pipeline.sh --features basic_audio,whisperx_transcription,deberta_text
```

Check if all dependencies are properly installed:
```bash
./run_pipeline.sh --check-dependencies
```

### Programmatic Usage

You can use the pipeline programmatically in your Python code in two ways:

#### Option 1: MultimodalFeatureExtractor (Recommended)

The `MultimodalFeatureExtractor` provides a simple, unified interface for feature extraction:

```python
from src.feature_extractor import MultimodalFeatureExtractor

# Initialize the extractor
extractor = MultimodalFeatureExtractor(
    features=['basic_audio', 'librosa_spectral', 'deberta_text'],
    device='cpu',  # Use 'cuda' if you have a compatible GPU
    output_dir='output/my_results'
)

# Process a video file
video_path = 'data/my_video.mp4'
features = extractor.extract_features(video_path)

# Process an audio file
audio_path = 'data/my_audio.wav'
features = extractor.extract_features(audio_path)

# Process text directly
text_data = {"transcript": "This is some text to analyze"}
features = extractor.extract_features(text_data)

# Process existing feature dictionary (useful for adding text analysis to existing data)
existing_features = {"whisperx_transcript": "Transcribed speech text"}
enhanced_features = extractor.extract_features(existing_features)
```

#### Option 2: Direct Pipeline Usage

You can also use the pipeline directly for more control:

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

## Troubleshooting

### HuggingFace Authentication Issues

If you encounter errors related to model access:

1. **Verify your token is correct**: Check that your `.env` file contains the right token
2. **Accept model licenses**: Make sure you've clicked "Agree" on all required model pages
3. **Check token permissions**: Ensure your token has "Read" access
4. **Restart the pipeline**: After updating authentication, restart the pipeline completely

Common error messages and solutions:
- `401 Unauthorized`: Token is invalid or missing
- `403 Forbidden`: You haven't accepted the model license agreements
- `Repository not found`: Model name may have changed or requires special access

### Dependency Issues

If you encounter import errors:
```bash
# Check if all dependencies are installed
./run_pipeline.sh --check-dependencies

# Reinstall dependencies if needed
./setup_env.sh
```

## Model Categories

- **Speech**: Speech emotion recognition, transcription, and audio feature extraction
- **Text**: DeBERTa-based benchmark performance analysis with comprehensive NLP task metrics
- **Vision**: Pose estimation, facial expression analysis, and motion tracking (coming soon)
- **Multimodal**: Combined audio-visual analysis and integration (coming soon)

## License

See the LICENSE file for details.
