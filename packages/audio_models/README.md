# Audio Models package

This sub-project isolates dependencies required by the audio and speech
feature extractors. Install it with Poetry to keep heavy libraries such as
`librosa`, `opensmile`, `torchaudio`, and `whisperx` out of the main
application environment until you explicitly opt in.

```bash
poetry install
```

Runtime implementations now live inside the `audio_models` package itself
(`audio_models.audio`, `audio_models.speech`, and `audio_models.utils`). This
sub-project remains an isolated dependency sandbox so you can iterate on audio
code without polluting other modality stacks.

## Dependency highlights

- `torch>=2.6` and `torchaudio>=2.6` patch the CVE-2025-32434 vulnerability hit
	when loading Hubert-based checkpoints. Re-run `poetry install` after pulling
	these changes so the upgraded wheels are built.
- `scipy` is constrained to `>=1.11,<1.13` to match the updated PyTorch builds
	and to restore compatibility with `scipy.stats.binom_test` used by legacy
	modules.
