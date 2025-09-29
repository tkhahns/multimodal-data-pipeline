# NLP Models package

This Poetry project isolates the transformer and TensorFlow stacks that power
text, sentiment, and dialogue analysis features. Use it whenever you need to
work on `nlp_models.text` or `nlp_models.emotion` modules without pulling those
heavy requirements into unrelated environments.

```bash
poetry install
```

Dependencies can diverge safely from the ones used in `audio_models` or
`cv_models`, letting you upgrade HuggingFace tooling without breaking vision
pipelines.
