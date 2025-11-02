#!/usr/bin/env python3
"""
OpenPose analyzer for pose estimation and tracking.
Based on CMU's OpenPose: Real-time multi-person keypoint detection library.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

class OpenPoseAnalyzer:
    """
    OpenPose analyzer for real-time multi-person pose estimation and tracking.

    Rather than re-implementing OpenPose, this class invokes the official OpenPose binary
    (https://github.com/CMU-Perceptual-Computing-Lab/openpose) as an external process,
    parses the generated JSON keypoint files, and aggregates them into pipeline features.
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        confidence_threshold: float = 0.1,
        openpose_bin: Optional[str] = None,
        model_folder: Optional[str] = None,
        extra_flags: Optional[List[str]] = None,
        enable_face: bool = True,
        enable_hand: bool = False,
        render_pose: int = 2,
        keep_json: bool = False,
        output_root: Optional[Path] = None,
    ):
        """
        Initialize OpenPose analyzer.
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence threshold for keypoint detection
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.initialized = False
        self.openpose_bin = openpose_bin or os.environ.get("OPENPOSE_BIN")
        self.model_folder = model_folder or os.environ.get("OPENPOSE_MODEL_FOLDER")
        env_flags = os.environ.get("OPENPOSE_FLAGS")
        if env_flags and not extra_flags:
            extra_flags = [flag for flag in env_flags.split() if flag]
        self.extra_flags = extra_flags or []
        self.enable_face = enable_face
        self.enable_hand = enable_hand
        self.render_pose = render_pose
        self.keep_json = keep_json
        self.number_people_max = 10
        self.output_root = Path(output_root) if output_root else None
        
        # Body keypoint names (COCO format)
        self.keypoint_names = [
            'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
            'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee',
            'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye',
            'LEye', 'REar', 'LEar'
        ]

        self.openpose_gpu_mode = (os.environ.get("OPENPOSE_GPU_MODE") or "").upper()
        self.openpose_use_cuda = (os.environ.get("OPENPOSE_USE_CUDA") or "").upper()
        self.cpu_only_mode = (
            self.openpose_gpu_mode == "CPU_ONLY"
            or self.openpose_use_cuda in {"OFF", "0", "FALSE"}
        )
        if self.cpu_only_mode and isinstance(self.render_pose, int) and self.render_pose > 0:
            logger.info(
                "OpenPose configured for CPU-only; disabling rendered video output (render_pose=0)"
            )
            self.render_pose = 0
        
        # Initialize default metrics
        self.default_metrics = {
            'openPose_total_frames': 0,
            'openPose_pose_detected_frames': 0,
            'openPose_detection_rate': 0.0,
            'openPose_avg_keypoints_per_frame': 0.0,
            'openPose_avg_confidence': 0.0,
            'openPose_max_persons_detected': 0,
            'openPose_pose_video_path': "",
            'openPose_pose_gif_path': "",
            'openPose_SM_pic': "",
            'openPose_SM_preview_available': 0.0,
            'openPose_SM_preview_mean_intensity': 0.0,
            'openPose_SM_preview_contrast': 0.0,
        }
        
        # Add individual keypoint coordinates
        for keypoint in self.keypoint_names:
            self.default_metrics[f'openPose_{keypoint.lower()}_x'] = 0.0
            self.default_metrics[f'openPose_{keypoint.lower()}_y'] = 0.0
            self.default_metrics[f'openPose_{keypoint.lower()}_confidence'] = 0.0
        
        # Add pose angles and distances
        self.default_metrics.update({
            'openPose_left_arm_angle': 0.0,
            'openPose_right_arm_angle': 0.0,
            'openPose_left_leg_angle': 0.0,
            'openPose_right_leg_angle': 0.0,
            'openPose_torso_angle': 0.0,
            'openPose_shoulder_width': 0.0,
            'openPose_hip_width': 0.0,
            'openPose_body_height': 0.0
        })

    def _initialize_model(self) -> None:
        """Resolve OpenPose binary and ensure prerequisites are in place."""
        if self.initialized:
            return

        logger.info("Initializing OpenPose analyzer (external OpenPose binary)")

        if not self.openpose_bin:
            raise RuntimeError(
                "OpenPose binary path not provided. Set OPENPOSE_BIN environment variable "
                "or pass openpose_bin to OpenPoseAnalyzer."
            )

        candidate_path = Path(self.openpose_bin)
        if candidate_path.exists():
            resolved_bin = candidate_path
        else:
            resolved = shutil.which(self.openpose_bin)
            if not resolved:
                raise FileNotFoundError(f"Could not locate OpenPose executable '{self.openpose_bin}'.")
            resolved_bin = Path(resolved)

        if not os.access(resolved_bin, os.X_OK):
            raise PermissionError(f"OpenPose executable is not runnable: {resolved_bin}")

        if self.model_folder:
            model_folder_path = Path(self.model_folder)
            if not model_folder_path.exists():
                raise FileNotFoundError(
                    f"OpenPose model folder not found at {model_folder_path}. Set OPENPOSE_MODEL_FOLDER correctly."
                )
            self.model_folder = str(model_folder_path.resolve())

        self.openpose_bin = str(resolved_bin.resolve())
        self.initialized = True
        logger.info("OpenPose analyzer initialized with binary %s", self.openpose_bin)

    def _prepare_output_paths(self, video_path: Path) -> Tuple[Path, Path, Path, Path]:
        base_root = self.output_root if self.output_root else video_path.parent
        output_dir = base_root / "openpose_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        json_dir = output_dir / f"{video_path.stem}_json"
        if json_dir.exists() and not self.keep_json:
            shutil.rmtree(json_dir, ignore_errors=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        render_video_path = output_dir / f"{video_path.stem}_openpose.mp4"
        if render_video_path.exists():
            render_video_path.unlink()

        gif_path = output_dir / f"{video_path.stem}_openpose.gif"
        if gif_path.exists():
            gif_path.unlink()

        return output_dir, json_dir, render_video_path, gif_path

    def _build_openpose_command(self, video_path: Path, json_dir: Path, render_video_path: Path) -> List[str]:
        command: List[str] = [
            self.openpose_bin,
            "--video",
            str(video_path),
            "--display",
            "0",
            "--render_pose",
            str(self.render_pose),
            "--model_pose",
            "COCO",
            "--write_json",
            str(json_dir),
            "--number_people_max",
            str(self.number_people_max),
        ]

        if self.cpu_only_mode:
            command.extend(["--num_gpu", "0"])

        if self.render_pose and self.render_pose > 0:
            command.extend(["--write_video", str(render_video_path)])

        if self.model_folder:
            command.extend(["--model_folder", self.model_folder])

        if self.enable_face:
            command.append("--face")

        if self.enable_hand:
            command.append("--hand")

        if self.extra_flags:
            command.extend(self.extra_flags)

        return command

    def _run_openpose_cli(self, video_path: Path, json_dir: Path, render_video_path: Path) -> None:
        command = self._build_openpose_command(video_path, json_dir, render_video_path)
        logger.info("Running OpenPose binary: %s", " ".join(command))
        try:
            completed = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.stdout:
                logger.debug("OpenPose stdout: %s", completed.stdout.strip())
            if completed.stderr:
                logger.debug("OpenPose stderr: %s", completed.stderr.strip())
        except subprocess.CalledProcessError as exc:
            stderr_tail = exc.stderr[-1024:] if exc.stderr else str(exc)
            raise RuntimeError(f"OpenPose execution failed: {stderr_tail}") from exc

    @staticmethod
    def _extract_keypoints_from_flat(values: List[float]) -> Dict[str, Tuple[float, float, float]]:
        keypoints: Dict[str, Tuple[float, float, float]] = {}
        if not values:
            return keypoints

        expected = len(values) // 3
        # Ensure we only map the keypoints we have names for
        # (OpenPose COCO model has 18 keypoints)
        for idx, name in enumerate([
            'nose', 'neck', 'rshoulder', 'relbow', 'rwrist',
            'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee',
            'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye',
            'rear', 'lear'
        ]):
            base = idx * 3
            if base + 2 >= len(values):
                break
            x, y, c = values[base], values[base + 1], values[base + 2]
            keypoints[name] = (x, y, c)

        # If OpenPose returned more keypoints than expected (e.g., BODY_25), ignore extras
        return keypoints

    @staticmethod
    def _compute_bbox_from_keypoints(keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[int, int, int, int]:
        valid_points = [(x, y) for (x, y, c) in keypoints.values() if c > 0]
        if not valid_points:
            return (0, 0, 0, 0)
        xs, ys = zip(*valid_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

    def _poses_from_people(self, people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        poses: List[Dict[str, Any]] = []
        for person_id, person in enumerate(people):
            flat_keypoints = person.get('pose_keypoints_2d', [])
            if not flat_keypoints:
                continue
            keypoints = self._extract_keypoints_from_flat(flat_keypoints)
            if not keypoints:
                continue

            confidences = [kp[2] for kp in keypoints.values() if kp[2] > 0]
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            pose: Dict[str, Any] = {
                'person_id': person_id,
                'keypoints': keypoints,
                'confidence': avg_conf,
                'bbox': self._compute_bbox_from_keypoints(keypoints),
            }

            flat_keypoints_3d = person.get('pose_keypoints_3d', [])
            if flat_keypoints_3d:
                pose['keypoints_3d'] = flat_keypoints_3d

            poses.append(pose)
        return poses

    @staticmethod
    def _frame_index_from_json_name(filename: str) -> Optional[int]:
        try:
            parts = filename.split('_')
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
                if part.endswith('keypoints.json'):
                    digits = ''.join(ch for ch in part if ch.isdigit())
                    if digits:
                        return int(digits)
        except Exception:
            return None
        return None

    def _parse_frame_metrics(self, json_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Optional[int]]:
        frame_metrics: Dict[int, Dict[str, Any]] = {}
        best_frame_idx: Optional[int] = None
        best_confidence = -1.0

        json_files = sorted(json_dir.glob('*keypoints.json'))
        for order, json_file in enumerate(json_files):
            try:
                with json_file.open('r', encoding='utf-8') as fh:
                    data = json.load(fh)
            except Exception as exc:
                logger.warning("Failed to read OpenPose JSON %s: %s", json_file, exc)
                continue

            people = data.get('people', [])
            poses = self._poses_from_people(people)

            frame_idx = self._frame_index_from_json_name(json_file.stem)
            if frame_idx is None:
                frame_idx = order

            keypoints_detected = 0
            confidences: List[float] = []
            for pose in poses:
                keypoints_detected += sum(1 for _, (_, _, conf) in pose['keypoints'].items() if conf > self.confidence_threshold)
                confidences.append(pose['confidence'])

            avg_confidence = float(np.mean(confidences)) if confidences else 0.0

            frame_result: Dict[str, Any] = {
                'frame_idx': frame_idx,
                'poses_detected': len(poses),
                'keypoints_detected': keypoints_detected,
                'avg_confidence': avg_confidence,
            }

            if poses:
                best_pose = max(poses, key=lambda pose_item: pose_item['confidence'])
                pose_metrics = self._calculate_pose_metrics(poses)
                frame_result.update(pose_metrics)

                for name, (x, y, conf) in best_pose['keypoints'].items():
                    frame_result[f'{name}_x'] = x
                    frame_result[f'{name}_y'] = y
                    frame_result[f'{name}_confidence'] = conf

                if best_pose['confidence'] > best_confidence:
                    best_confidence = best_pose['confidence']
                    best_frame_idx = frame_idx

            frame_metrics[frame_idx] = frame_result

        return frame_metrics, best_frame_idx

    def _calculate_pose_metrics(self, poses: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate pose-related metrics from detected poses.
        
        Args:
            poses: List of detected poses
            
        Returns:
            Dictionary of pose metrics
        """
        if not poses:
            return {}
        
        # Use the most confident pose
        best_pose = max(poses, key=lambda p: p['confidence'])
        keypoints = best_pose['keypoints']
        
        metrics = {}
        
        # Calculate joint angles
        def angle_between_points(p1, p2, p3):
            """Calculate angle at p2 formed by p1-p2-p3"""
            try:
                a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                return angle
            except:
                return 0.0
        
        # Left arm angle (shoulder-elbow-wrist)
        if all(k in keypoints for k in ['lshoulder', 'lelbow', 'lwrist']):
            metrics['left_arm_angle'] = angle_between_points(
                keypoints['lshoulder'][:2], keypoints['lelbow'][:2], keypoints['lwrist'][:2]
            )
        
        # Right arm angle
        if all(k in keypoints for k in ['rshoulder', 'relbow', 'rwrist']):
            metrics['right_arm_angle'] = angle_between_points(
                keypoints['rshoulder'][:2], keypoints['relbow'][:2], keypoints['rwrist'][:2]
            )
        
        # Left leg angle (hip-knee-ankle)
        if all(k in keypoints for k in ['lhip', 'lknee', 'lankle']):
            metrics['left_leg_angle'] = angle_between_points(
                keypoints['lhip'][:2], keypoints['lknee'][:2], keypoints['lankle'][:2]
            )
        
        # Right leg angle
        if all(k in keypoints for k in ['rhip', 'rknee', 'rankle']):
            metrics['right_leg_angle'] = angle_between_points(
                keypoints['rhip'][:2], keypoints['rknee'][:2], keypoints['rankle'][:2]
            )
        
        # Body dimensions
        if 'rshoulder' in keypoints and 'lshoulder' in keypoints:
            metrics['shoulder_width'] = np.sqrt(
                (keypoints['rshoulder'][0] - keypoints['lshoulder'][0])**2 +
                (keypoints['rshoulder'][1] - keypoints['lshoulder'][1])**2
            )
        
        if 'rhip' in keypoints and 'lhip' in keypoints:
            metrics['hip_width'] = np.sqrt(
                (keypoints['rhip'][0] - keypoints['lhip'][0])**2 +
                (keypoints['rhip'][1] - keypoints['lhip'][1])**2
            )
        
        if 'nose' in keypoints and 'lankle' in keypoints:
            metrics['body_height'] = abs(keypoints['nose'][1] - keypoints['lankle'][1])
        
        # Torso angle (vertical alignment)
        if 'neck' in keypoints and 'rhip' in keypoints and 'lhip' in keypoints:
            hip_center_x = (keypoints['rhip'][0] + keypoints['lhip'][0]) / 2
            hip_center_y = (keypoints['rhip'][1] + keypoints['lhip'][1]) / 2
            torso_angle = np.arctan2(
                keypoints['neck'][0] - hip_center_x,
                keypoints['neck'][1] - hip_center_y
            ) * 180 / np.pi
            metrics['torso_angle'] = abs(torso_angle)
        
        return metrics

    @staticmethod
    def _frame_preview_stats(frame: Optional[np.ndarray]) -> Tuple[float, float, float]:
        if frame is None:
            return 0.0, 0.0, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray)) if gray.size else 0.0
        contrast = float(np.std(gray)) if gray.size else 0.0
        return 1.0, mean_intensity, contrast

    def _extract_preview_media(
        self,
        render_video_path: Path,
        gif_path: Path,
        best_frame_idx: Optional[int],
    ) -> Tuple[str, Tuple[float, float, float]]:
        if not render_video_path.exists():
            return "", (0.0, 0.0, 0.0)

        cap = cv2.VideoCapture(str(render_video_path))
        if not cap.isOpened():
            return "", (0.0, 0.0, 0.0)

        gif_frames: List[np.ndarray] = []
        preview_frame: Optional[np.ndarray] = None
        frame_order = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if best_frame_idx is not None and frame_order == best_frame_idx:
                    preview_frame = frame.copy()

                if frame_order % 5 == 0 and len(gif_frames) < 20:
                    resized = cv2.resize(frame, (320, 240))
                    gif_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

                frame_order += 1
        finally:
            cap.release()

        if preview_frame is None and gif_frames:
            preview_frame = cv2.cvtColor(gif_frames[0], cv2.COLOR_RGB2BGR)

        gif_path_str = ""
        if gif_frames:
            try:
                from PIL import Image

                pil_images = [Image.fromarray(frame) for frame in gif_frames]
                pil_images[0].save(
                    str(gif_path),
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=200,
                    loop=0,
                )
                gif_path_str = str(gif_path)
                logger.info("Created pose GIF: %s", gif_path_str)
            except ImportError:
                logger.warning("PIL not available - could not create GIF preview")

        preview_stats = self._frame_preview_stats(preview_frame)
        return gif_path_str, preview_stats

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Run OpenPose on a video and aggregate its outputs."""

        if not self.initialized:
            self._initialize_model()

        logger.info("Analyzing poses in video via OpenPose binary: %s", video_path)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        video_path_obj = Path(video_path)
        _, json_dir, render_video_path, gif_path = self._prepare_output_paths(video_path_obj)

        self._run_openpose_cli(video_path_obj, json_dir, render_video_path)

        frame_metrics_map, best_frame_idx = self._parse_frame_metrics(json_dir)
        if not frame_metrics_map:
            logger.warning("OpenPose produced no keypoint JSON files for %s", video_path)

        max_idx = max(frame_metrics_map.keys(), default=-1)
        if total_frames <= 0 or (max_idx >= 0 and max_idx + 1 > total_frames):
            total_frames = max_idx + 1 if max_idx >= 0 else 0

        frame_metrics: List[Dict[str, Any]] = []
        for frame_idx in range(total_frames):
            metrics = frame_metrics_map.get(frame_idx)
            if metrics is None:
                metrics = {
                    'frame_idx': frame_idx,
                    'poses_detected': 0,
                    'keypoints_detected': 0,
                    'avg_confidence': 0.0,
                }
            frame_metrics.append(metrics)

        results = self._aggregate_frame_results(frame_metrics, total_frames)
        results['openPose_pose_video_path'] = str(render_video_path) if render_video_path.exists() else ""

        gif_path_str, preview_stats = self._extract_preview_media(render_video_path, gif_path, best_frame_idx)
        results['openPose_pose_gif_path'] = gif_path_str
        results['openPose_SM_pic'] = ""
        results['openPose_SM_preview_available'] = preview_stats[0]
        results['openPose_SM_preview_mean_intensity'] = preview_stats[1]
        results['openPose_SM_preview_contrast'] = preview_stats[2]

        if not self.keep_json:
            shutil.rmtree(json_dir, ignore_errors=True)

        logger.info("OpenPose analysis completed for %s", video_path)

        return results

    def _aggregate_frame_results(self, frame_metrics: List[Dict], total_frames: int) -> Dict[str, Any]:
        """
        Aggregate frame-level results into video-level metrics.
        
        Args:
            frame_metrics: List of frame-level metrics
            total_frames: Total number of frames
            
        Returns:
            Aggregated results
        """
        if not frame_metrics:
            return self.default_metrics.copy()
        
        results = {}
        
        # Basic statistics
        frames_with_poses = sum(1 for fm in frame_metrics if fm['poses_detected'] > 0)
        results['openPose_total_frames'] = total_frames
        results['openPose_pose_detected_frames'] = frames_with_poses
        results['openPose_detection_rate'] = frames_with_poses / total_frames if total_frames > 0 else 0.0
        
        # Keypoint statistics
        all_keypoint_counts = [fm['keypoints_detected'] for fm in frame_metrics]
        results['openPose_avg_keypoints_per_frame'] = np.mean(all_keypoint_counts)
        
        # Confidence statistics
        all_confidences = [fm['avg_confidence'] for fm in frame_metrics if fm['avg_confidence'] > 0]
        results['openPose_avg_confidence'] = np.mean(all_confidences) if all_confidences else 0.0
        
        # Maximum persons detected
        results['openPose_max_persons_detected'] = max(fm['poses_detected'] for fm in frame_metrics)
        
        # Aggregate keypoint coordinates (average across frames where detected)
        for keypoint in self.keypoint_names:
            keypoint_lower = keypoint.lower()
            x_key = f'{keypoint_lower}_x'
            y_key = f'{keypoint_lower}_y'
            conf_key = f'{keypoint_lower}_confidence'

            x_values = [fm[x_key] for fm in frame_metrics
                        if x_key in fm and conf_key in fm and fm[conf_key] > self.confidence_threshold]
            y_values = [fm[y_key] for fm in frame_metrics
                        if y_key in fm and conf_key in fm and fm[conf_key] > self.confidence_threshold]
            conf_values = [fm[conf_key] for fm in frame_metrics if conf_key in fm]
            
            results[f'openPose_{keypoint_lower}_x'] = np.mean(x_values) if x_values else 0.0
            results[f'openPose_{keypoint_lower}_y'] = np.mean(y_values) if y_values else 0.0
            results[f'openPose_{keypoint_lower}_confidence'] = np.mean(conf_values) if conf_values else 0.0
        
        # Aggregate pose metrics
        pose_metric_names = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle', 
                           'torso_angle', 'shoulder_width', 'hip_width', 'body_height']
        
        for metric_name in pose_metric_names:
            values = [fm[metric_name] for fm in frame_metrics if metric_name in fm and fm[metric_name] > 0]
            results[f'openPose_{metric_name}'] = np.mean(values) if values else 0.0
        
        return results

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Get OpenPose features in the standard feature dictionary format.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with OpenPose features grouped by model
        """
        try:
            results = self.analyze_video(video_path)
            
            return {
                "Pose estimation and tracking": {
                    "description": "Real-time multi-person keypoint detection with pose estimation and tracking using OpenPose",
                    "features": results
                }
            }
            
        except Exception as e:
            logger.error(f"Error in OpenPose analysis: {e}")
            return {
                "Pose estimation and tracking": {
                    "description": "Real-time multi-person keypoint detection with pose estimation and tracking using OpenPose",
                    "features": self.default_metrics.copy()
                }
            }


def extract_openpose_features(video_path: str, device: str = 'cpu', confidence_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Extract OpenPose features from a video file.
    
    Args:
        video_path: Path to the video file
        device: Device to run on
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Dictionary containing OpenPose features
    """
    analyzer = OpenPoseAnalyzer(device=device, confidence_threshold=confidence_threshold)
    return analyzer.get_feature_dict(video_path)


if __name__ == "__main__":
    # Example usage
    video_path = "sample_video.mp4"
    features = extract_openpose_features(video_path, device='cpu')
    print("OpenPose Features:")
    for group_name, group_data in features.items():
        print(f"\n{group_name}:")
        print(f"  Description: {group_data['description']}")
        print(f"  Features extracted: {len(group_data['features'])}")
