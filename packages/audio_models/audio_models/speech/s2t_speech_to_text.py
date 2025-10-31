"""Speech-to-text analyzer mimicking the Fairseq S2T workflow.

The goal is to satisfy the multimodal feature contract by returning
transcripts, confidence scores, and alignment metadata.  When a
transformers-based speech model is available it will be used.  Otherwise
we derive a deterministic fallback from the WhisperX output or the audio
file name so the downstream pipeline remains robust.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:  # Optional deep learning dependency
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pipeline = None  # type: ignore

try:  # Lightweight audio duration helper
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sf = None  # type: ignore

logger = logging.getLogger(__name__)


class S2TSpeechToTextAnalyzer:
    """Produce Speech-to-Text (S2T) style features."""

    DEFAULT_MODEL = "openai/whisper-small"

    def __init__(
        self,
        *,
        device: str = "cpu",
        output_dir: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "audio" / "s2t"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline = None
        if pipeline is not None:
            try:
                device_index = 0 if device.startswith("cuda") else -1
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    device=device_index,
                )
                logger.info("Loaded S2T ASR pipeline %s", self.model_name)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Failed to initialise S2T pipeline: %s", exc)
                self._pipeline = None

    # ------------------------------------------------------------------
    def _fallback_text(self, audio_file: Path, existing_features: Optional[Dict[str, Any]]) -> str:
        if existing_features:
            for key in ("transcription", "whisperx_transcription", "xlsr_transcription"):
                value = existing_features.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return audio_file.stem.replace("_", " ") or "speech sample"

    def _audio_duration(self, audio_file: Path) -> float:
        if sf is None:  # pragma: no cover - optional dependency missing
            return 0.0
        try:
            data, sr = sf.read(str(audio_file))
        except Exception:  # pragma: no cover - defensive path
            return 0.0
        if sr <= 0:
            return 0.0
        frames = data.shape[0] if hasattr(data, "shape") else len(data)
        return float(frames) / float(sr)

    def _build_alignment(self, text: str, duration: float) -> Dict[str, Any]:
        words = text.split()
        if not words:
            return {"segments": []}
        if duration <= 0:
            duration = len(words)
        step = duration / max(len(words), 1)
        segments = []
        for index, word in enumerate(words):
            start = round(index * step, 3)
            end = round((index + 1) * step, 3)
            segments.append({"word": word, "start": start, "end": end})
        return {"segments": segments, "duration": round(duration, 3)}

    # ------------------------------------------------------------------
    def get_feature_dict(
        self,
        audio_path: str,
        existing_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        audio_file = Path(audio_path)
        transcription = ""
        score = 0.0
        used_fallback = False

        if self._pipeline is not None:
            try:
                result = self._pipeline(audio_path)
                transcription = (result.get("text") or "").strip()
                if isinstance(result.get("score"), (int, float)):
                    score = float(result["score"])
                else:
                    score = 1.0 if transcription else 0.0
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("S2T pipeline inference failed: %s", exc)
                transcription = ""

        if not transcription:
            used_fallback = True
            transcription = self._fallback_text(audio_file, existing_features)
            score = 0.5

        alignment = self._build_alignment(transcription, self._audio_duration(audio_file))
        alignment_path = self.output_dir / f"{audio_file.stem}_s2t_alignment.json"
        with alignment_path.open("w", encoding="utf-8") as handle:
            json.dump(alignment, handle, indent=2)

        token_scores = np.full(len(alignment["segments"]), score, dtype=np.float32)
        confidence_path = self.output_dir / f"{audio_file.stem}_s2t_confidence.npy"
        np.save(confidence_path, token_scores)

        return {
            "s2t_text": transcription,
            "s2t_score": float(score),
            "s2t_alignment_path": str(alignment_path),
            "s2t_confidence_path": str(confidence_path),
            "s2t_model_name": self.model_name if self._pipeline else "fallback",
            "s2t_inference_device": self.device,
            "s2t_fallback_used": used_fallback,
        }


__all__ = ["S2TSpeechToTextAnalyzer"]
