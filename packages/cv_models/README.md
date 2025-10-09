# Computer Vision Models package

Install this sub-project to provision the heavyweight computer vision stack
required by pose estimation, facial analysis, and optical flow components.
Keeping it in its own Poetry environment prevents clashes with audio or NLP
packages that prefer different versions of PyTorch or OpenCV.

```bash
poetry install
```

All vision analyzers have been relocated into `cv_models.vision`, so this
package now contains both the dependency sandbox and the runtime modules.

## Updated binary wheels

- Vision features now depend on `torch>=2.6` and `torchvision>=0.21` to align
	with the audio stack and to satisfy upstream security mitigations.
- `scipy` is capped at `<1.13` because several analyzers still rely on
	soon-to-be-deprecated APIs such as `scipy.stats.bayes_mvs`.
- After syncing, run `poetry install` here (and optionally `poetry lock`) so
	the new versions are compiled before you launch the pipeline.
