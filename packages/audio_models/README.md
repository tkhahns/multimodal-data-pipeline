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
