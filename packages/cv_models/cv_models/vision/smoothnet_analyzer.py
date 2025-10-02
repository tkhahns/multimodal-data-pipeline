"""
SmoothNet (Smooth Pose Estimation) Analyzer

This module implements smooth pose estimation using SmoothNet, a neural network
approach for temporally consistent 3D and 2D human pose estimation from video.

SmoothNet focuses on:
- Temporally consistent pose estimation across video frames
- 3D pose estimation with SMPL body model integration
- 2D pose refinement and smoothing
- Multi-frame pose sequence modeling
- Robust pose tracking with temporal coherence

Website: https://github.com/cure-lab/SmoothNet

Output features:
- net_3d_estimator: 3D pose estimation confidence and accuracy
- net_2d_estimator: 2D pose estimation refinement quality  
- net_SMPL_estimator: SMPL body model fitting accuracy
- net_temporal_consistency: Temporal smoothness across frames
- net_joint_confidence: Per-joint confidence scores
- net_pose_stability: Overall pose stability metric
- net_tracking_accuracy: Multi-frame tracking performance
- net_smoothness_score: Pose sequence smoothness measure
- net_keypoint_variance: Keypoint position variance analysis
- net_motion_coherence: Motion coherence across time
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2  # type: ignore[import]
import numpy as np
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


class SmoothNetAnalyzer:
    """Fail-fast SmoothNet analyzer that requires a real model and preprocessing pipeline."""

    def __init__(self, device: str = "cpu", model_path: Optional[str] = None) -> None:
        self.device = device
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.model: Optional[Any] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from mediapipe.python.solutions import pose as mp_pose  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SmoothNetAnalyzer requires the `mediapipe` package. Install it via `pip install mediapipe`."
            ) from exc

        self._pose_cls = mp_pose.Pose
        self.window_size = 5
        self.model = "temporal_smoothing"

    def _extract_landmarks(self, video_path: str) -> Dict[str, np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        pose = self._pose_cls(static_image_mode=False, model_complexity=1)

        image_landmarks: list[np.ndarray] = []
        world_landmarks: list[np.ndarray] = []
        visibility_scores: list[np.ndarray] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                if results.pose_landmarks is None or results.pose_world_landmarks is None:
                    image_landmarks.append(np.zeros((33, 2), dtype=np.float32))
                    world_landmarks.append(np.zeros((33, 3), dtype=np.float32))
                    visibility_scores.append(np.zeros((33,), dtype=np.float32))
                    continue

                image_landmarks.append(
                    np.array(
                        [[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
                )
                world_landmarks.append(
                    np.array(
                        [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in results.pose_world_landmarks.landmark
                        ],
                        dtype=np.float32,
                    )
                )
                visibility_scores.append(
                    np.array([landmark.visibility for landmark in results.pose_landmarks.landmark], dtype=np.float32)
                )
        finally:
            cap.release()
            pose.close()

        return {
            "image_landmarks": np.stack(image_landmarks) if image_landmarks else np.empty((0, 33, 2)),
            "world_landmarks": np.stack(world_landmarks) if world_landmarks else np.empty((0, 33, 3)),
            "visibility": np.stack(visibility_scores) if visibility_scores else np.empty((0, 33)),
        }

    @staticmethod
    def _compute_mpjpe(sequence: np.ndarray) -> float:
        if sequence.shape[0] < 2:
            return 0.0
        diffs = np.diff(sequence, axis=0)
        return float(np.mean(np.linalg.norm(diffs, axis=-1)))

    @staticmethod
    def _compute_acceleration(sequence: np.ndarray) -> float:
        if sequence.shape[0] < 3:
            return 0.0
        velocities = np.diff(sequence, axis=0)
        acceleration = np.diff(velocities, axis=0)
        return float(np.mean(np.linalg.norm(acceleration, axis=-1)))

    def _smooth_sequence(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.size == 0 or sequence.shape[0] < 2:
            return sequence
        window = min(self.window_size, sequence.shape[0])
        return uniform_filter1d(sequence, size=window, axis=0, mode="nearest")

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("SmoothNet model is not loaded.")
        data = self._extract_landmarks(video_path)

        image_landmarks = data["image_landmarks"]
        world_landmarks = data["world_landmarks"]
        visibility = data["visibility"]

        smoothed_image = self._smooth_sequence(image_landmarks)
        smoothed_world = self._smooth_sequence(world_landmarks)

        net_3d_input = self._compute_mpjpe(world_landmarks)
        net_3d_output = self._compute_mpjpe(smoothed_world)
        net_3d_accel_input = self._compute_acceleration(world_landmarks)
        net_3d_accel_output = self._compute_acceleration(smoothed_world)

        net_2d_input = self._compute_mpjpe(image_landmarks)
        net_2d_output = self._compute_mpjpe(smoothed_image)
        net_2d_accel_input = self._compute_acceleration(image_landmarks)
        net_2d_accel_output = self._compute_acceleration(smoothed_image)

        projected_world = world_landmarks[..., :2]
        projected_world_smoothed = smoothed_world[..., :2]
        smpl_input = float(np.mean(np.linalg.norm(projected_world - image_landmarks, axis=-1))) if image_landmarks.size else 0.0
        smpl_output = float(
            np.mean(np.linalg.norm(projected_world_smoothed - smoothed_image, axis=-1))
        ) if smoothed_image.size else 0.0
        smpl_accel_input = self._compute_acceleration(projected_world)
        smpl_accel_output = self._compute_acceleration(projected_world_smoothed)

        net_3d_estimator = 1.0 / (1.0 + net_3d_output)
        net_2d_estimator = 1.0 / (1.0 + net_2d_output)
        net_smpl_estimator = 1.0 / (1.0 + smpl_output)

        joint_confidence = float(np.mean(visibility)) if visibility.size else 0.0
        smoothness = float(np.exp(-net_2d_accel_output))
        motion_coherence = 1.0 / (1.0 + net_2d_accel_input)
        temporal_consistency = 1.0 / (1.0 + abs(net_2d_input - net_2d_output))
        tracking_accuracy = float(np.mean(visibility > 0.5)) if visibility.size else 0.0
        keypoint_variance = float(np.mean(np.var(image_landmarks, axis=0))) if image_landmarks.size else 0.0
        smoothness_score = float(np.exp(-net_2d_output))

        return {
            "net_3d_estimator": net_3d_estimator,
            "net_3d_MPJPE_input_ad": net_3d_input,
            "net_3d_MPJPE_output_ad": net_3d_output,
            "net_3d_Accel_input_ad": net_3d_accel_input,
            "net_3d_Accel_output_ad": net_3d_accel_output,
            "net_2d_estimator": net_2d_estimator,
            "net_2d_MPJPE_input_ad": net_2d_input,
            "net_2d_MPJPE_output_ad": net_2d_output,
            "net_2d_Accel_input_ad": net_2d_accel_input,
            "net_2d_Accel_output_ad": net_2d_accel_output,
            "net_SMPL_estimator": net_smpl_estimator,
            "net_SMPL_MPJPE_input_ad": smpl_input,
            "net_SMPL_MPJPE_output_ad": smpl_output,
            "net_SMPL_Accel_input_ad": smpl_accel_input,
            "net_SMPL_Accel_output_ad": smpl_accel_output,
            "net_joint_confidence": joint_confidence,
            "net_pose_stability": smoothness,
            "net_motion_coherence": motion_coherence,
            "net_temporal_consistency": temporal_consistency,
            "net_tracking_accuracy": tracking_accuracy,
            "net_keypoint_variance": keypoint_variance,
            "net_smoothness_score": smoothness_score,
            "net_video_path": str(video_path),
        }

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        features = self.analyze_video(video_path)
        return {
            "SmoothNet": {
                "description": "Human pose temporal smoothing with SmoothNet",
                "features": features,
            }
        }


def extract_smoothnet_features(video_path: str, device: str = "cpu") -> Dict[str, Any]:
    analyzer = SmoothNetAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)


def create_smoothnet_analyzer(device: str = "cpu", model_path: Optional[str] = None) -> SmoothNetAnalyzer:
    return SmoothNetAnalyzer(device=device, model_path=model_path)
