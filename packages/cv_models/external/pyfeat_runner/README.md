# Py-Feat Runner Helper

This sidecar package is retained for manual diagnostics or environments that
still require a dedicated interpreter. The main pipeline now executes
Py-Feat (`feat.Detector`) directly inside its primary Python 3.12 runtime, so
this helper is optional.

## Usage

```bash
poetry install
poetry run python -m pyfeat_runner /path/to/video.mp4
```

The command prints a JSON payload containing the `pf_*` features. The main
pipeline consumes this JSON via the `PyFeatAnalyzer` subprocess fallback.
