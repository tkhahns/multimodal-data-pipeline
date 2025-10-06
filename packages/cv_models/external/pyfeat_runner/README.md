# Py-Feat Runner Helper

This sidecar package lets the multimodal data pipeline run Py-Feat
(`feat.Detector`) in a Python 3.11 environment when the main runtime uses
Python 3.12 (where py-feat is currently unsupported).

## Usage

```bash
poetry install
poetry run python -m pyfeat_runner /path/to/video.mp4
```

The command prints a JSON payload containing the `pf_*` features. The main
pipeline consumes this JSON via the `PyFeatAnalyzer` subprocess fallback.
