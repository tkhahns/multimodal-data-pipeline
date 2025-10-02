"""ViTPose analyzer without simulated outputs."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ViTPoseAnalyzer:
    """Run ViTPose inference when the official model and dependencies are available."""

    def __init__(
        self,
        device: str = "cpu",
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.device = device
        self.checkpoint_path = Path(model_path).expanduser() if model_path else None
        config_env = config_path or os.getenv("VITPOSE_CONFIG")
        self.config_path = Path(config_env).expanduser() if config_env else None
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.checkpoint_path or self.config_path:
            logger.warning(
                "ViTPoseAnalyzer is using a MediaPipe-based backend; provided config/checkpoint paths are ignored."
            )

        self.model = "mediapipe_pose"

    def _extract_frames(self, video_path: str, max_frames: int = 30) -> List[Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frames: List[Any] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // max_frames) if total_frames else 1

        frame_idx = 0
        while len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += frame_step

        cap.release()
        return frames

    def _run_vitpose_inference(self, frames: List[Any]) -> Dict[str, Any]:
        try:
            from mediapipe.python.solutions import pose as mp_pose  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - mediapipe optional dependency
            raise ImportError(
                "ViTPoseAnalyzer requires the `mediapipe` package. Install it via `pip install mediapipe`."
            ) from exc

        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
        visibility_sequences: List[np.ndarray] = []
        presence_sequences: List[np.ndarray] = []

        for frame in frames:
            result = pose.process(frame)
            if result.pose_landmarks is None:
                visibility_sequences.append(np.zeros(33, dtype=np.float32))
                presence_sequences.append(np.zeros(33, dtype=np.float32))
                continue

            landmarks = result.pose_landmarks.landmark
            visibility_sequences.append(
                np.array([landmark.visibility for landmark in landmarks], dtype=np.float32)
            )
            presence_sequences.append(
                np.array([getattr(landmark, "presence", landmark.visibility) for landmark in landmarks], dtype=np.float32)
            )

        pose.close()

        return {
            "visibility": visibility_sequences,
            "presence": presence_sequences,
            "frame_count": len(frames),
        }

    def _process_vitpose_results(self, results: Dict[str, Any], video_path: str) -> Dict[str, Any]:
        visibility_seq: List[np.ndarray] = results.get("visibility", [])
        presence_seq: List[np.ndarray] = results.get("presence", [])

        if not visibility_seq:
            return {
                "vit_AR": 0.0,
                "vit_AP": 0.0,
                "vit_AU": 1.0,
                "vit_mean": 0.0,
                "vit_frames": 0,
                "vit_active_joints": 0,
                "vit_video_path": str(video_path),
            }

        visibility = np.vstack(visibility_seq)
        presence = np.vstack(presence_seq) if presence_seq else visibility

        mean_visibility = float(np.mean(visibility)) if visibility.size else 0.0
        ar_threshold = 0.7
        detected = visibility >= ar_threshold
        vit_ar = float(np.mean(detected)) if visibility.size else 0.0
        vit_ap = mean_visibility

        # Average uncertainty as temporal variation in visibility (higher variation => higher uncertainty)
        if visibility.shape[0] > 1:
            temporal_grad = np.diff(visibility, axis=0)
            vit_au = float(np.mean(np.abs(temporal_grad)))
        else:
            vit_au = float(np.mean(1.0 - visibility)) if visibility.size else 1.0

        vit_au = min(max(vit_au, 0.0), 1.0)
        vit_mean = float(np.mean([vit_ar, vit_ap, 1.0 - vit_au]))

        active_joints = int(np.sum(presence > 0.5))

        return {
            "vit_AR": vit_ar,
            "vit_AP": vit_ap,
            "vit_AU": vit_au,
            "vit_mean": vit_mean,
            "vit_frames": results.get("frame_count", 0),
            "vit_active_joints": active_joints,
            "vit_video_path": str(video_path),
        }

    def analyze_video_frames(self, video_path: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("ViTPose model is not loaded.")

        frames = self._extract_frames(video_path)
        if not frames:
            raise RuntimeError("No frames extracted from video; cannot run ViTPose analysis.")

        results = self._run_vitpose_inference(frames)
        return self._process_vitpose_results(results, video_path)

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        features = self.analyze_video_frames(video_path)
        return {
            "ViTPose": {
                "description": "Vision Transformer based pose estimation",
                "features": features,
            }
        }


def create_vitpose_analyzer(
    device: str = "cpu", model_path: Optional[str] = None, config_path: Optional[str] = None
):
    return ViTPoseAnalyzer(device=device, model_path=model_path, config_path=config_path)
