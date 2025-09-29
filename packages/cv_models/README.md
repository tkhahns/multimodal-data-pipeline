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
