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
    """AV-HuBERT analyzer backed by the official model implementations.

    This module embeds and transcribes audiovisual speech using Facebook Research's
    AV-HuBERT checkpoints distributed via Hugging Face Transformers.  We ingest raw
    video, extract an audio track with FFmpeg, sample lip-aligned frames, and feed
    both modalities through ``AVHubertForCTC`` to obtain transcripts and hidden
    embeddings suitable for downstream tasks.
    """

    from __future__ import annotations

    import json
    import logging
    import os
    import subprocess
    import tempfile
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple

    import cv2
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image

    from cv_models.external.repo_manager import ensure_repo

    logger = logging.getLogger(__name__)


    try:  # Import once so initialization can fail fast with a clear message.
        from transformers import AVHubertForCTC, AutoProcessor  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "AVHuBERTAnalyzer requires the `transformers` package. Install it in the "
            "cv_models environment (e.g. `poetry add transformers[torch]`)."
        ) from exc


    @dataclass(frozen=True)
    class AVHubertConfig:
        """Runtime configuration for AV-HuBERT inference."""

        model_id: str
        sample_rate: int = 16_000
        target_fps: float = 25.0
        max_frames: int = 256
        frame_size: int = 96
        face_scale: float = 1.6


    class AVHuBERTAnalyzer:
        """Run AV-HuBERT on paired audio/video signals to produce embeddings."""

        MODEL_ENV = "AVHUBERT_MODEL_ID"
        DEFAULT_MODEL = "facebook/avhubert-large-30h-cv"

        def __init__(
            self,
            *,
            device: str = "cpu",
            output_dir: Optional[Path] = None,
            model_id: Optional[str] = None,
            sample_rate: int = 16_000,
            target_fps: float = 25.0,
            max_frames: int = 256,
            frame_size: int = 96,
            face_scale: float = 1.6,
        ) -> None:
            ensure_repo("av_hubert")  # Keep the official repo available for reference.

            resolved_model = model_id or os.getenv(self.MODEL_ENV) or self.DEFAULT_MODEL
            self.config = AVHubertConfig(
                model_id=resolved_model,
                sample_rate=sample_rate,
                target_fps=target_fps,
                max_frames=max_frames,
                frame_size=frame_size,
                face_scale=face_scale,
            )

            self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested for AV-HuBERT but CUDA is unavailable; defaulting to CPU.")

            logger.info("Loading AV-HuBERT model %s", self.config.model_id)
            self.processor = AutoProcessor.from_pretrained(self.config.model_id)
            self.model = AVHubertForCTC.from_pretrained(self.config.model_id)
            self.model.to(self.device)
            self.model.eval()

            output_root = Path(output_dir) if output_dir else Path.cwd() / "output"
            self.output_dir = output_root / "vision" / "avhubert"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Interfaces
        # ------------------------------------------------------------------
        def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_file}")

            logger.info("Running AV-HuBERT on %s", video_file)
            audio_waveform, sr = self._extract_audio(video_file)
            if audio_waveform.size == 0:
                raise RuntimeError("Failed to extract audio track for AV-HuBERT processing.")

            frame_batch = self._extract_frames(video_file)
            if frame_batch:
                processor_inputs = self._prepare_inputs(audio_waveform, sr, frame_batch)
            else:
                logger.warning("No frames extracted for AV-HuBERT; falling back to audio-only mode.")
                processor_inputs = self._prepare_inputs(audio_waveform, sr, None)

            logits, hidden_states = self._run_model(processor_inputs)
            transcription, confidence = self._decode(logits)

            embeddings_path = self._persist_embeddings(video_file.stem, hidden_states)
            metadata_path = self._persist_metadata(
                video_file,
                embeddings_path,
                transcription,
                confidence,
                len(frame_batch),
                audio_waveform.size,
            )

            return {
                "AVH_transcription": transcription,
                "AVH_confidence": confidence,
                "AVH_embeddings_path": str(embeddings_path),
                "AVH_metadata_path": str(metadata_path),
                "AVH_device": str(self.device),
            }

        # ------------------------------------------------------------------
        # Audio / Video preparation
        # ------------------------------------------------------------------
        def _extract_audio(self, video_path: Path) -> Tuple[np.ndarray, int]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.config.sample_rate),
                str(tmp_path),
            ]
            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError as exc:  # pragma: no cover - system dependency
                raise RuntimeError("ffmpeg is required for AV-HuBERT analysis but was not found in PATH.") from exc
            except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime failure
                raise RuntimeError(f"ffmpeg failed to extract audio from {video_path}") from exc

            try:
                audio, sr = sf.read(str(tmp_path), dtype="float32")
            finally:
                tmp_path.unlink(missing_ok=True)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr

        def _extract_frames(self, video_path: Path) -> List[Image.Image]:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                logger.warning("Unable to open video %s for frame extraction", video_path)
                return []

            fps = capture.get(cv2.CAP_PROP_FPS) or self.config.target_fps
            frame_interval = max(int(round(fps / self.config.target_fps)), 1)

            frames: List[Image.Image] = []
            frame_idx = 0
            while len(frames) < self.config.max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop = self._crop_face(rgb)
                resized = cv2.resize(crop, (self.config.frame_size, self.config.frame_size), interpolation=cv2.INTER_AREA)
                frames.append(Image.fromarray(resized))
                frame_idx += 1

            capture.release()
            return frames

        def _crop_face(self, frame: np.ndarray) -> np.ndarray:
            """Approximate mouth-centric crop using the largest detected face."""

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            if len(faces) == 0:
                return self._center_crop(frame)

            x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
            cx, cy = x + w / 2, y + h / 2
            size = max(w, h) * self.config.face_scale
            return self._crop_square(frame, cx, cy, size)

        def _center_crop(self, frame: np.ndarray) -> np.ndarray:
            height, width, _ = frame.shape
            side = min(height, width)
            return self._crop_square(frame, width / 2, height / 2, side)

        def _crop_square(self, frame: np.ndarray, cx: float, cy: float, size: float) -> np.ndarray:
            half = size / 2
            x1 = int(max(cx - half, 0))
            y1 = int(max(cy - half, 0))
            x2 = int(min(cx + half, frame.shape[1]))
            y2 = int(min(cy + half, frame.shape[0]))
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                return frame
            return cropped

        # ------------------------------------------------------------------
        # Model execution
        # ------------------------------------------------------------------
        def _prepare_inputs(
            self,
            audio: np.ndarray,
            sr: int,
            frames: Optional[List[Image.Image]],
        ) -> Dict[str, torch.Tensor]:
        audio_input = audio if sr == self.config.sample_rate else self._resample_audio(audio, sr)
            processor_kwargs: Dict[str, Any] = {
                "raw_speech": audio_input,
                "sampling_rate": self.config.sample_rate,
                "return_tensors": "pt",
                "padding": "longest",
            }
            if frames:
                processor_kwargs["video"] = frames

            inputs = self.processor(**processor_kwargs)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            return inputs

        def _resample_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
            try:
                import librosa  # type: ignore  # Local import avoids hard dependency during module import.
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Resampling audio for AV-HuBERT requires `librosa`. Install it via `poetry add librosa`."
                ) from exc

            resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
            return resampled.astype(np.float32)

        def _run_model(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            return logits, hidden_states

        def _decode(self, logits: torch.Tensor) -> Tuple[str, float]:
        probs = torch.softmax(logits, dim=-1)
        max_probs, pred_ids = torch.max(probs, dim=-1)
        transcription = self.processor.batch_decode(pred_ids.cpu(), skip_special_tokens=True)[0]
            confidence = float(max_probs.mean().cpu().item())
            return transcription.strip(), confidence

        # ------------------------------------------------------------------
        # Persistence
        # ------------------------------------------------------------------
        def _persist_embeddings(self, stem: str, hidden_states: torch.Tensor) -> Path:
            embedding = hidden_states.squeeze(0).cpu().numpy()
            path = self.output_dir / f"{stem}_avhubert_embeddings.npz"
            np.savez_compressed(path, embeddings=embedding)
            return path

        def _persist_metadata(
            self,
            video_path: Path,
            embeddings_path: Path,
            transcription: str,
            confidence: float,
            num_frames: int,
            audio_samples: int,
        ) -> Path:
            metadata = {
                "video_path": str(video_path),
                "model_id": self.config.model_id,
                "device": str(self.device),
                "num_frames": num_frames,
                "max_frames": self.config.max_frames,
                "embedding_path": str(embeddings_path),
                "transcription": transcription,
                "confidence": confidence,
                "audio_samples": audio_samples,
                "sample_rate": self.config.sample_rate,
            }
            path = self.output_dir / f"{video_path.stem}_avhubert_metadata.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            return path


    __all__ = ["AVHuBERTAnalyzer"]
