"""Speech emotion recognition module backed by a pretrained transformer."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class SpeechEmotionRecognizer:
    """Infer speech emotion probabilities using a pretrained transformer model.

    The previous placeholder implementation produced pseudo-random scores based on
    handcrafted heuristics. This version loads the `superb/hubert-large-superb-er`
    checkpoint (fine-tuned on the SUPERB emotion recognition benchmark) through the
    🤗 Transformers audio-classification pipeline and returns calibrated
    probabilities for every label exposed by the model.
    """

    def __init__(
        self,
        model_name: str = "superb/hubert-large-superb-er",
        *,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Create a recognizer around a pretrained Hugging Face checkpoint.

        Args:
            model_name: Hugging Face model id to load for audio emotion
                classification.
            device: Optional device spec ("cpu", "cuda", or CUDA index). When
                omitted the detector automatically picks GPU if available.
            cache_dir: Optional directory where the checkpoint should be cached.
        """

        pipeline_device = self._resolve_device(device)
        try:
            self._pipeline = pipeline(
                task="audio-classification",
                model=model_name,
                device=pipeline_device,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("Failed to load speech emotion model %s: %s", model_name, exc)
            self._pipeline = None
            self.emotions = ["angry", "happy", "sad", "neutral"]
            self._labels = [label.capitalize() for label in self.emotions]
            return

        config = self._pipeline.model.config
        id_to_label = getattr(config, "id2label", {})
        if not id_to_label:
            raise RuntimeError(
                "Loaded audio emotion model does not expose an id2label mapping."
            )

        # Preserve order of labels by their numeric ids.
        self._labels: List[str] = [id_to_label[i] for i in sorted(id_to_label)]
        self.emotions: List[str] = [label.lower() for label in self._labels]

    @staticmethod
    def _resolve_device(device: Optional[str]) -> int:
        """Translate a device string into a transformers pipeline device index."""

        if device is None:
            return 0 if torch.cuda.is_available() else -1

        if isinstance(device, str):
            spec = device.strip().lower()
            if spec == "cpu":
                return -1
            if spec.startswith("cuda"):
                if not torch.cuda.is_available():
                    return -1
                if ":" in spec:
                    _, idx = spec.split(":", maxsplit=1)
                    return int(idx)
                return 0

        # Fallback: assume the caller passed a numeric CUDA index already.
        try:
            return int(device)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return -1

    def predict(self, audio_path: str) -> Dict[str, float]:
        """Return per-emotion probabilities for the given audio file."""

        top_k = len(self._labels)
        if self._pipeline is None:
            uniform = 1.0 / len(self.emotions)
            return {f"ser_{label}": uniform for label in self.emotions}

        raw_predictions = self._pipeline(audio_path, top_k=top_k)

        # Normalise scores defensively to ensure a proper probability simplex.
        scores = np.array([entry["score"] for entry in raw_predictions], dtype=float)
        if not np.isclose(scores.sum(), 1.0):
            if scores.sum() == 0:
                scores = np.full_like(scores, 1.0 / len(scores))
            else:
                scores = scores / scores.sum()

        return {
            f"ser_{label.lower()}": float(score)
            for label, score in zip(self._labels, scores)
        }

