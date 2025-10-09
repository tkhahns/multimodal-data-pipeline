# Multimodal Data Pipeline
A compact toolkit to extract multimodal features (audio, speech, text, vision) from videos and audio files. Outputs JSON/Parquet in `output/` with clear feature prefixes.
```
multimodal-data-pipeline/
├── pyproject.toml          # Orchestrator project (installs the pipeline)
├── run_pipeline.py         # CLI entrypoint (adds packages/ to sys.path)
├── packages/
│   ├── core_pipeline/      # Runtime pipeline code and shared utilities
│   ├── audio_models/       # Audio + speech dependency sandbox
│   ├── cv_models/          # Vision dependency sandbox
│   └── nlp_models/         # Text/NLP dependency sandbox
├── tests/                  # Smoke tests and integration checks
├── run_all.sh              # WSL/Linux/macOS convenience wrapper
└── run_all.ps1             # Windows PowerShell convenience wrapper
```

- Each subfolder inside `packages/` has its own `pyproject.toml`, lock file and
  Poetry environment. Activate only the stacks you need while working on a
  modality to avoid dependency clashes (for example, TensorFlow vs. PyTorch).
- `packages/core_pipeline` now houses the orchestrator (`MultimodalPipeline`,
  `MultimodalFeatureExtractor`, CLI helpers, etc.).
- `packages/nlp_models` ships TensorFlow as an optional extra named
  `tensorflow-stack`; the root project enables it automatically so the
  Universal Sentence Encoder continues to work out-of-the-box.
- The root project lists the three subprojects as editable path dependencies so
  `poetry install` at the repository root provides an "everything" environment.
- You can still work inside a smaller environment by navigating into
  `packages/audio_models`, `packages/cv_models`, or `packages/nlp_models` and
  running `poetry install` there.

Common workflows

```bash
# Full environment (all modalities)
poetry install

# Only audio + speech stack
cd packages/audio_models
poetry install

# Only computer vision stack
cd packages/cv_models
poetry install

# Only NLP stack
cd packages/nlp_models
poetry install
```
The runtime code now imports directly from these packages; the legacy `src`
layout has been fully retired.

> **Note on recent dependency hardening**
>
> - GPU/CPU stacks now require `torch>=2.6` to satisfy CVE-2025-32434 safeguards. The
>   audio bundle also pins `torchaudio>=2.6` and the vision bundle expects
>   `torchvision>=0.21`.
> - Scientific routines share `scipy>=1.11,<1.13` across all modality packages.
> - Transformers-based text features expect `tf-keras>=2.15` to coexist with the
>   system `keras` package and avoid the Keras 3 incompatibility warning. Run
>   `poetry install` (root and inside each edited package) to refresh lock files
>   before executing the pipeline again.

## Py-Feat (Python 3.11 runner)

Py-Feat isn't compatible with Python 3.12, so the repo includes an isolated subproject to run it on Python 3.11 and feed results back to the main pipeline via a subprocess.

Location: `external/pyfeat_runner`

Setup (in WSL):

```bash
cd external/pyfeat_runner
poetry env use python3.11
poetry install
```

Usage (standalone):

```bash
poetry run python -m pyfeat_runner /absolute/path/to/video.mp4
```

The main pipeline will automatically use this runner if `py-feat` cannot import in the Python 3.12 environment. No further code changes are needed; just ensure the runner environment is set up once.


## Features (by prefix → model)

- Audio
  - oc_* → Basic audio volume/pitch (OpenCV)
  - lbrs_* → Spectral features (Librosa)
  - osm_* → openSMILE LLDs & functionals
  - AS_* → AudioStretchy analysis
- Speech
  - ser_* → Speech emotion recognition
  - WhX_* → WhisperX transcription + diarization
  - (paths for separated audio) → Speech separation
- Text
  - DEB_* → DeBERTa metrics
  - CSE_* → SimCSE STS metrics
  - alb_* → ALBERT benchmarks
  - BERT_* → Sentence-BERT embeddings/reranking
  - USE_* → Universal Sentence Encoder
  - MELD_* → Conversation emotion (MELD)
- Vision
  - PARE_* → PARE 3D body estimation
  - vit_* → ViTPose pose metrics
  - psa_* → Polarized Self-Attention
  - eln_* → EmotiEffNet (valence/arousal/AUs)
  - GMP_* → MediaPipe pose landmarks (33 body)
  - openPose_* → OpenPose 2D keypoints
  - ann_* → ME-GraphAU AUs
  - dan_* → DAN emotions
  - GAN_* → GANimation AUs
  - arbex_* → ARBEx emotions
  - indm_* → Insta-DM depth/motion
  - of_* → CrowdFlow optical flow/Crowd metrics
  - ViF_* → VideoFinder objects/people (requires Ollama)
  - net_* → SmoothNet temporal pose
  - GCN_* → LaneGCN motion forecasting
  - rsn_* → RSN keypoint localization
  - DHiR_* → Deep HRNet pose metrics
  - SBH_* → Simple Baselines pose metrics
- Facial (optional)
  - pf_* → Py-Feat facial analysis (requires Python 3.11 + numpy ~=1.23.x)

## Installation

Prerequisites
- Python 3.11, Poetry, Git
- FFmpeg (system binary)
- Ollama (required; included by default for VideoFinder)

Quick setup (WSL/Linux/macOS)
```bash
chmod +x run_all.sh
./run_all.sh
```

HuggingFace (required for diarization)
```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
# Accept licenses:
# https://huggingface.co/pyannote/speaker-diarization-3.1
# https://huggingface.co/pyannote/segmentation-3.0
```

Notes for token and permissions
- Create a Read token (or a Fine-grained token with “Models: read”). Write/Admin is not needed.
- You can also login instead of using .env:
  - `huggingface-cli login` (stores the token in your HF cache)
- If using .env, set both variables for widest compatibility:
  - `HF_TOKEN=...` and `HUGGINGFACE_HUB_TOKEN=...`
- If you see: “Could not download 'pyannote/speaker-diarization-3.1' … NoneType has no attribute 'to'”, it’s an auth/gating issue: ensure token is set and both model gates are accepted.

Windows (PowerShell)
```powershell
./run_all.ps1
"HF_TOKEN=your_huggingface_token_here" | Out-File -FilePath .env -Encoding utf8
```

FFmpeg tips
- WSL/Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: download from ffmpeg.org or `choco install ffmpeg`

WSL/Linux: system packages (for vision support)
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libopenblas-dev liblapack-dev \
  libx11-dev libgtk-3-dev libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
  python3-dev
```
Notes:
- Required: ffmpeg; GL/X11 libs (libgl1, libglib2.0-0, libsm6, libxrender1, libxext6) for OpenCV.
- Recommended: build-essential, cmake, pkg-config, python3-dev, BLAS (libopenblas-dev, liblapack-dev).
- Optional: libgtk-3-dev (only if you display image/video windows).

## Usage

CLI (recommended)
```bash
# Process all videos in ./data with all features
./run_all.sh

# See help and feature list
./run_all.sh --help
./run_all.sh --list-features

# Select features
./run_all.sh --features basic_audio,speech_emotion,vitpose_vision
```
PowerShell: use `./run_all.ps1` with equivalent flags.

Programmatic
```python
from core_pipeline import MultimodalFeatureExtractor

extractor = MultimodalFeatureExtractor(
    features=["basic_audio", "whisperx_transcription", "deberta_text"],
    device="cpu"
)
features = extractor.extract_features("data/sample.mp4")
```

## Output

- Audio files in `output/audio/`
- Per-file JSONs in `output/features/`
- Consolidated `output/pipeline_features.json`
- Large arrays optionally saved as `.npy`

## Options (short)

- `--data-dir`/`-d` input folder
- `--output-dir`/`-o` results folder
- `--features`/`-f` comma list (see `--list-features`)
- `--is-audio` to treat inputs as audio files
- `--check-deps` to verify dependencies

Available features (names)
- basic_audio, librosa_spectral, opensmile, audiostretchy
- speech_emotion, speech_separation, whisperx_transcription
- heinsen_sentiment, meld_emotion
- deberta_text, simcse_text, albert_text, sbert_text, use_text
- pare_vision, vitpose_vision, psa_vision, emotieffnet_vision, mediapipe_pose_vision,
  openpose_vision, me_graphau_vision, dan_vision, ganimation_vision,
  arbex_vision, instadm_vision, crowdflow_vision, deep_hrnet_vision, simple_baselines_vision,
  rsn_vision, optical_flow_vision, videofinder_vision, lanegcn_vision, smoothnet_vision

Note
- VideoFinder requires Ollama (included by default in setup)
- Py-Feat requires Python 3.11 with numpy ~=1.23.x

## Troubleshooting (essentials)

- HuggingFace auth: ensure `.env` has HF_TOKEN and licenses accepted
- FFmpeg: install system binary and ensure `ffmpeg -version` works
- Ollama: ensure it’s installed/running for `videofinder_vision`

## License

See LICENSE.