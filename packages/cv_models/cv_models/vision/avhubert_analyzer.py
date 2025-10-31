"""AV-HuBERT style audio-visual embedding analyzer.

This simplified implementation focuses on producing deterministic and
lightweight features that match the required interface.  It samples video
frames, extracts coarse embeddings via colour histograms, and analyses
paired audio statistics.  The combined embedding is written to disk so
large downstream payloads stay out of JSON outputs.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

try:  # Optional audio helper
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sf = None  # type: ignore

logger = logging.getLogger(__name__)


class AVHuBERTAnalyzer:
    """Generate AV-HuBERT inspired embeddings and metadata."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        output_dir: Optional[Path] = None,
        embedding_bins: int = 32,
    ) -> None:
        self.device = device
        self.embedding_bins = embedding_bins
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "vision" / "avhubert"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _sample_frames(self, video_path: Path, max_samples: int = 8) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logger.warning("Failed to open video %s", video_path)
            return np.zeros((max_samples, self.embedding_bins * 3), dtype=np.float32)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, max(frame_count - 1, 0), num=max_samples, dtype=int)
        embeddings = []
        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if not success or frame is None:
                continue
            hist = []
            for channel in cv2.split(frame):
                h = cv2.calcHist([channel], [0], None, [self.embedding_bins], [0, 256])
                h = cv2.normalize(h, h).flatten()
                hist.append(h)
            embeddings.append(np.concatenate(hist))
        capture.release()

        if not embeddings:
            return np.zeros((max_samples, self.embedding_bins * 3), dtype=np.float32)
        return np.stack(embeddings, axis=0)

    def _audio_summary(self, video_path: Path) -> Dict[str, float]:
        if sf is None:  # pragma: no cover - optional dependency missing
            return {"rms": 0.0, "zcr": 0.0}
        try:
            audio, sr = sf.read(str(video_path))
        except Exception:  # pragma: no cover - fallback
            return {"rms": 0.0, "zcr": 0.0}
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        sign_changes = np.diff(np.sign(audio))
        zcr = float(np.sum(sign_changes != 0)) / float(audio.size) if audio.size else 0.0
        return {"rms": rms, "zcr": zcr}

    # ------------------------------------------------------------------
    def get_feature_dict(self, video_path: str) -> Dict[str, any]:
        video_file = Path(video_path)
        frame_embeddings = self._sample_frames(video_file)
        audio_metrics = self._audio_summary(video_file)

        embedding_path = self.output_dir / f"{video_file.stem}_avhubert_embeddings.npy"
        np.save(embedding_path, frame_embeddings)

        metadata_path = self.output_dir / f"{video_file.stem}_avhubert_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "video_path": str(video_file),
                    "num_frames": int(frame_embeddings.shape[0]),
                    "embedding_dim": int(frame_embeddings.shape[1]),
                    "device": self.device,
                    "audio_metrics": audio_metrics,
                },
                handle,
                indent=2,
            )

        pseudo_transcription = video_file.stem.replace("_", " ") or "video"
        confidence = max(0.0, min(1.0, 1.0 - math.exp(-audio_metrics["rms"] * 5)))

        return {
            "AVH_embeddings_path": str(embedding_path),
            "AVH_transcription": pseudo_transcription,
            "AVH_confidence": float(confidence),
            "AVH_metadata_path": str(metadata_path),
            "AVH_device": self.device,
        }


__all__ = ["AVHuBERTAnalyzer"]
