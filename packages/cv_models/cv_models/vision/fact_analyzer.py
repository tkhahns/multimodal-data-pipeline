"""FACT (Face Action Coding Toolkit) inspired analyzer.

The true FACT model is heavy and depends on deep learning libraries.  This
implementation mimics the interface by extracting lightweight facial
landmarks using MediaPipe when available and falls back to simple skin-tone
statistics otherwise.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore

logger = logging.getLogger(__name__)


class FACTAnalyzer:
    """Produce facial action estimates and metadata."""

    def __init__(self, *, output_dir: Optional[Path] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "vision" / "fact"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _detect_faces(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, 1.1, 4)
        return faces

    def _landmarks(self, frame: np.ndarray) -> np.ndarray:
        if mp is None:  # pragma: no cover
            return np.empty((0, 468, 3), dtype=np.float32)
        mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
        with mp_face_mesh.FaceMesh(static_image_mode=True) as mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:  # pragma: no cover
                return np.empty((0, 468, 3), dtype=np.float32)
            landmarks = []
            for face_landmarks in result.multi_face_landmarks:
                coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                landmarks.append(coords)
            return np.asarray(landmarks, dtype=np.float32)

    def _feature_from_face(self, face: np.ndarray, frame: np.ndarray) -> Dict[str, float]:
        x, y, w, h = face
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:  # pragma: no cover
            return {"skin_mean": 0.0, "skin_std": 0.0}
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
        skin_pixels = cv2.bitwise_and(roi, roi, mask=skin_mask)
        values = skin_pixels[skin_mask > 0]
        if values.size == 0:
            return {"skin_mean": 0.0, "skin_std": 0.0}
        luminance = np.mean(values, axis=1)
        return {"skin_mean": float(np.mean(luminance)), "skin_std": float(np.std(luminance))}

    # ------------------------------------------------------------------
    def get_feature_dict(self, video_path: str) -> Dict[str, any]:
        capture = cv2.VideoCapture(video_path)
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            logger.warning("Unable to read video frame for FACT analysis: %s", video_path)
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        faces = self._detect_faces(frame)
        face_features = [self._feature_from_face(face, frame) for face in faces]
        landmarks = self._landmarks(frame)

        metadata = {
            "num_faces": int(len(faces)),
            "landmarks_detected": bool(landmarks.size),
            "video_path": video_path,
        }

        summary_path = self.output_dir / f"{Path(video_path).stem}_fact_metadata.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **metadata,
                    "faces": face_features,
                    "landmarks_shape": list(landmarks.shape),
                },
                handle,
                indent=2,
            )

        landmark_path = self.output_dir / f"{Path(video_path).stem}_fact_landmarks.npy"
        np.save(landmark_path, landmarks)

        facial_intensity = float(np.mean([feat["skin_mean"] for feat in face_features])) if face_features else 0.0

        return {
            "FACT_metadata_path": str(summary_path),
            "FACT_landmarks_path": str(landmark_path),
            "FACT_intensity": facial_intensity,
            "FACT_faces": face_features,
        }


__all__ = ["FACTAnalyzer"]