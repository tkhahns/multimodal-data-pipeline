"""
Google MediaPipe Pose Estimation and Tracking
Based on: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from pathlib import Path
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)

class MediaPipePoseAnalyzer:
    """
    MediaPipe analyzer for pose estimation and tracking.
    
    This analyzer implements pose landmark detection with 33 landmarks,
    providing both normalized and world coordinates with visibility scores.
    """
    
    def __init__(self, device='cpu', min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe pose analyzer.
        
        Args:
            device: Device to run inference on (MediaPipe runs on CPU)
            min_detection_confidence: Minimum detection confidence threshold
            min_tracking_confidence: Minimum tracking confidence threshold
        """
        self.device = device
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.initialized = False
        
        # MediaPipe pose solution
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Pose landmark model
        self.pose = None
        
        # 33 pose landmarks according to MediaPipe
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        # Initialize default metrics for 33 landmarks
        self.default_metrics = {}
        
        # Normalized landmarks (x, y, z, visibility, presence)
        for i in range(1, 34):  # 1-33 landmarks
            self.default_metrics[f'GMP_land_x_{i}'] = 0.0
            self.default_metrics[f'GMP_land_y_{i}'] = 0.0
            self.default_metrics[f'GMP_land_z_{i}'] = 0.0
            self.default_metrics[f'GMP_land_visi_{i}'] = 0.0
            self.default_metrics[f'GMP_land_presence_{i}'] = 0.0
        
        # World landmarks (x, y, z, visibility, presence)
        for i in range(1, 34):  # 1-33 landmarks
            self.default_metrics[f'GMP_world_x_{i}'] = 0.0
            self.default_metrics[f'GMP_world_y_{i}'] = 0.0
            self.default_metrics[f'GMP_world_z_{i}'] = 0.0
            self.default_metrics[f'GMP_world_visi_{i}'] = 0.0
            self.default_metrics[f'GMP_world_presence_{i}'] = 0.0
        
        # Pose visualization image (base64 encoded)
        self.default_metrics['GMP_SM_pic'] = ""
        
    def _initialize_model(self):
        """Initialize the MediaPipe pose model."""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing MediaPipe Pose model...")
            
            # Initialize MediaPipe Pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,  # For video processing
                model_complexity=2,  # 0, 1, or 2 (higher = more accurate but slower)
                enable_segmentation=False,  # We don't need segmentation
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            self.initialized = True
            logger.info("MediaPipe Pose model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose model: {e}")
            raise
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """
        Process a single frame for pose landmarks.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Tuple of (landmarks_dict, annotated_frame)
        """
        if not self.initialized:
            self._initialize_model()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Initialize metrics with defaults
        metrics = self.default_metrics.copy()
        annotated_frame = None
        
        if results.pose_landmarks:
            # Extract normalized landmarks
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                idx = i + 1  # 1-based indexing
                metrics[f'GMP_land_x_{idx}'] = landmark.x
                metrics[f'GMP_land_y_{idx}'] = landmark.y
                metrics[f'GMP_land_z_{idx}'] = landmark.z
                metrics[f'GMP_land_visi_{idx}'] = landmark.visibility
                metrics[f'GMP_land_presence_{idx}'] = getattr(landmark, 'presence', 1.0)
            
            # Extract world landmarks if available
            if results.pose_world_landmarks:
                for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                    idx = i + 1  # 1-based indexing
                    metrics[f'GMP_world_x_{idx}'] = landmark.x
                    metrics[f'GMP_world_y_{idx}'] = landmark.y
                    metrics[f'GMP_world_z_{idx}'] = landmark.z
                    metrics[f'GMP_world_visi_{idx}'] = landmark.visibility
                    metrics[f'GMP_world_presence_{idx}'] = getattr(landmark, 'presence', 1.0)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return metrics, annotated_frame
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Base64 encoded string
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            logger.warning(f"Failed to encode image to base64: {e}")
            return ""
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze pose landmarks in a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing pose analysis results
        """
        if not self.initialized:
            self._initialize_model()
        
        logger.info(f"Analyzing pose landmarks in video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_metrics = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        best_annotated_frame = None
        max_landmarks_detected = 0
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                metrics, annotated_frame = self._process_frame(frame)
                
                # Count detected landmarks (non-zero visibility)
                landmarks_detected = sum(1 for i in range(1, 34) 
                                       if metrics.get(f'GMP_land_visi_{i}', 0) > 0.5)
                
                # Keep the frame with most landmarks for visualization
                if landmarks_detected > max_landmarks_detected:
                    max_landmarks_detected = landmarks_detected
                    best_annotated_frame = annotated_frame
                
                # Add frame index and timestamp
                metrics['frame_idx'] = frame_idx
                metrics['timestamp'] = frame_idx / fps if fps > 0 else frame_idx
                
                frame_metrics.append(metrics)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Completed pose analysis: {len(frame_metrics)} frames processed")
        
        # Aggregate results
        return self._aggregate_results(frame_metrics, best_annotated_frame, video_path)
    
    def _aggregate_results(self, frame_metrics: List[Dict[str, Any]], 
                          best_frame: Optional[np.ndarray], 
                          video_path: str) -> Dict[str, Any]:
        """
        Aggregate frame-level results into final metrics.
        
        Args:
            frame_metrics: List of per-frame metrics
            best_frame: Best annotated frame for visualization
            video_path: Path to the video file
            
        Returns:
            Aggregated pose analysis results
        """
        if not frame_metrics:
            result = self.default_metrics.copy()
            result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'landmarks_detected_frames': 0,
                'detection_rate': 0.0
            })
            return result
        
        # Calculate aggregated metrics (mean across all frames)
        aggregated = {}
        numeric_keys = [key for key in frame_metrics[0].keys() 
                       if key.startswith('GMP_') and key != 'GMP_SM_pic']
        
        for key in numeric_keys:
            values = [frame.get(key, 0.0) for frame in frame_metrics]
            aggregated[key] = float(np.mean(values))
        
        # Add visualization of best frame
        if best_frame is not None:
            aggregated['GMP_SM_pic'] = self._encode_image_to_base64(best_frame)
        else:
            aggregated['GMP_SM_pic'] = ""
        
        # Add summary statistics
        landmarks_detected_frames = sum(1 for frame in frame_metrics 
                                      if any(frame.get(f'GMP_land_visi_{i}', 0) > 0.5 
                                           for i in range(1, 34)))
        
        aggregated.update({
            'video_path': str(video_path),
            'total_frames': len(frame_metrics),
            'landmarks_detected_frames': landmarks_detected_frames,
            'detection_rate': landmarks_detected_frames / len(frame_metrics),
            'avg_landmarks_per_frame': float(np.mean([
                sum(1 for i in range(1, 34) if frame.get(f'GMP_land_visi_{i}', 0) > 0.5)
                for frame in frame_metrics
            ]))
        })
        
        return aggregated
    
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Get MediaPipe pose features for the pipeline.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with MediaPipe pose features
        """
        try:
            results = self.analyze_video(video_path)
            
            # Create feature group for the pipeline
            feature_dict = {
                "Pose estimation and tracking": {
                    "description": "Google MediaPipe pose landmark detection with 33 landmarks",
                    "features": results
                }
            }
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Error in MediaPipe pose analysis: {e}")
            
            # Return default values on error
            default_result = self.default_metrics.copy()
            default_result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'landmarks_detected_frames': 0,
                'detection_rate': 0.0,
                'avg_landmarks_per_frame': 0.0,
                'error': str(e)
            })
            
            feature_dict = {
                "Pose estimation and tracking": {
                    "description": "Google MediaPipe pose landmark detection with 33 landmarks",
                    "features": default_result
                }
            }
            
            return feature_dict

def extract_mediapipe_pose_features(video_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Extract MediaPipe pose features from a video file.
    
    Args:
        video_path: Path to the video file
        device: Device to run on (MediaPipe uses CPU)
        
    Returns:
        Dictionary containing pose features
    """
    analyzer = MediaPipePoseAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)
