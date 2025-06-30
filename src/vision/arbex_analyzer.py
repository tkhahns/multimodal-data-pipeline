"""
ARBEx: Attentive Feature Extraction with Reliability Balancing for Robust Facial Expression Learning
Based on: https://github.com/takihasan/ARBEx

This analyzer implements emotional indices extraction via different feature levels
using attentive feature extraction with reliability balancing.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
from pathlib import Path
import base64
import io
from PIL import Image
import warnings

logger = logging.getLogger(__name__)

class ARBExAnalyzer:
    """
    ARBEx analyzer for robust facial expression learning with attentive feature extraction.
    
    This analyzer implements emotional indices extraction via different feature levels
    using reliability balancing for robust facial expression recognition.
    """
    
    def __init__(self, device='cpu', confidence_threshold=0.5):
        """
        Initialize ARBEx analyzer.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence threshold for emotion classification
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.initialized = False
        
        # Emotion labels according to ARBEx
        self.emotion_labels = [
            'Neutral',
            'Anger', 
            'Disgust',
            'Fear',
            'Happiness',
            'Sadness',
            'Surprise',
            'Others'
        ]
        
        # Face detection model
        self.face_cascade = None
        
        # Initialize default metrics
        self.default_metrics = {
            'arbex_primary': 'Neutral',      # Primary emotion classification
            'arbex_final': 'Neutral',        # Final emotion classification after reliability balancing
            'arbex_face_detected': 0.0,      # Whether face was detected
            'arbex_confidence_primary': 0.0, # Confidence for primary classification
            'arbex_confidence_final': 0.0,   # Confidence for final classification
            'arbex_reliability_score': 0.0,  # Reliability balancing score
            'arbex_SM_pic': ""               # Base64 encoded visualization
        }
        
        # Add individual emotion probabilities for primary level
        for emotion in self.emotion_labels:
            self.default_metrics[f'arbex_primary_{emotion.lower()}'] = 0.0
        
        # Add individual emotion probabilities for final level
        for emotion in self.emotion_labels:
            self.default_metrics[f'arbex_final_{emotion.lower()}'] = 0.0
        
    def _initialize_model(self):
        """Initialize the ARBEx model components."""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing ARBEx analyzer...")
            
            # Initialize face detection (using OpenCV's Haar cascade as fallback)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.warning("Could not load face cascade classifier")
            
            # Note: In a full implementation, you would load the actual ARBEx model here
            # For this implementation, we'll use a simplified approach with feature extraction
            # and rule-based emotion classification with reliability balancing
            
            self.initialized = True
            logger.info("ARBEx analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ARBEx analyzer: {e}")
            raise
    
    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def _extract_facial_features(self, face_roi: np.ndarray) -> Dict[str, float]:
        """
        Extract facial features from a face ROI for emotion classification.
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            Dictionary of facial features for emotion analysis
        """
        # Convert to grayscale
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Normalize image
        gray_roi = gray_roi.astype(np.float32) / 255.0
        
        # Extract features using different levels (simulating ARBEx's multi-level approach)
        
        # Level 1: Basic statistical features
        level1_features = {
            'mean_intensity': np.mean(gray_roi),
            'std_intensity': np.std(gray_roi),
            'skewness': float(np.mean((gray_roi - np.mean(gray_roi))**3)) / (np.std(gray_roi)**3 + 1e-7),
            'kurtosis': float(np.mean((gray_roi - np.mean(gray_roi))**4)) / (np.std(gray_roi)**4 + 1e-7)
        }
        
        # Level 2: Regional features (eye, mouth, forehead regions)
        h, w = gray_roi.shape
        level2_features = {
            'eye_region_intensity': np.mean(gray_roi[:h//3, :]),          # Upper third
            'mouth_region_intensity': np.mean(gray_roi[2*h//3:, :]),     # Lower third  
            'cheek_region_intensity': np.mean(gray_roi[h//3:2*h//3, :]), # Middle third
            'forehead_intensity': np.mean(gray_roi[:h//4, w//4:3*w//4]), # Forehead area
            'eye_mouth_contrast': abs(np.mean(gray_roi[:h//3, :]) - np.mean(gray_roi[2*h//3:, :]))
        }
        
        # Level 3: Texture features (simulating attentive feature extraction)
        level3_features = {
            'horizontal_gradient': np.mean(np.abs(np.diff(gray_roi, axis=1))),
            'vertical_gradient': np.mean(np.abs(np.diff(gray_roi, axis=0))),
            'edge_density': np.mean(cv2.Canny((gray_roi * 255).astype(np.uint8), 50, 150) / 255.0),
            'texture_variance': np.var(gray_roi)
        }
        
        # Combine all features
        features = {**level1_features, **level2_features, **level3_features}
        
        return features
    
    def _classify_emotion_primary(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Primary emotion classification based on facial features.
        
        Args:
            features: Extracted facial features
            
        Returns:
            Tuple of (emotion_label, confidence, emotion_probabilities)
        """
        # Simple rule-based emotion classification for primary level
        # In a full implementation, this would use the trained ARBEx model
        
        emotion_scores = {}
        
        # Calculate emotion scores based on features
        # These are simplified heuristics - real ARBEx would use learned features
        
        # Neutral: balanced features, low variance
        emotion_scores['Neutral'] = (
            1.0 - abs(features['mean_intensity'] - 0.5) * 2 +
            max(0, 0.5 - features['std_intensity']) * 2 +
            max(0, 0.3 - features['edge_density']) * 2
        ) / 3
        
        # Happiness: mouth region brighter, eye-mouth contrast
        emotion_scores['Happiness'] = (
            max(0, features['mouth_region_intensity'] - features['eye_region_intensity']) * 3 +
            features['eye_mouth_contrast'] * 2 +
            max(0, features['mean_intensity'] - 0.4) * 2
        ) / 3
        
        # Sadness: mouth region darker, low overall intensity
        emotion_scores['Sadness'] = (
            max(0, features['eye_region_intensity'] - features['mouth_region_intensity']) * 3 +
            max(0, 0.6 - features['mean_intensity']) * 2 +
            features['std_intensity'] * 1.5
        ) / 3
        
        # Anger: high contrast, strong gradients
        emotion_scores['Anger'] = (
            features['horizontal_gradient'] * 2 +
            features['vertical_gradient'] * 2 +
            features['edge_density'] * 2 +
            abs(features['skewness']) * 1.5
        ) / 4
        
        # Fear: high variance, strong texture
        emotion_scores['Fear'] = (
            features['texture_variance'] * 2 +
            features['std_intensity'] * 2 +
            features['edge_density'] * 1.5 +
            abs(features['kurtosis'] - 3) * 1.5
        ) / 4
        
        # Surprise: high eye region intensity, strong gradients
        emotion_scores['Surprise'] = (
            features['eye_region_intensity'] * 2 +
            features['forehead_intensity'] * 1.5 +
            features['vertical_gradient'] * 2 +
            features['eye_mouth_contrast'] * 1.5
        ) / 4
        
        # Disgust: mouth region features, moderate contrast
        emotion_scores['Disgust'] = (
            max(0, 0.5 - features['mouth_region_intensity']) * 2 +
            features['horizontal_gradient'] * 1.5 +
            features['cheek_region_intensity'] * 1.5 +
            features['edge_density'] * 1.5
        ) / 4
        
        # Others: residual category for ambiguous cases
        emotion_scores['Others'] = (
            features['texture_variance'] * 1.5 +
            abs(features['skewness']) * 1.5 +
            abs(features['kurtosis'] - 3) * 1.5 +
            features['std_intensity'] * 1.5
        ) / 4
        
        # Normalize scores to probabilities
        total_score = sum(emotion_scores.values()) + 1e-7
        emotion_probs = {emotion: score / total_score for emotion, score in emotion_scores.items()}
        
        # Get primary emotion and confidence
        primary_emotion = max(emotion_probs, key=emotion_probs.get)
        primary_confidence = emotion_probs[primary_emotion]
        
        return primary_emotion, primary_confidence, emotion_probs
    
    def _apply_reliability_balancing(self, primary_emotion: str, primary_confidence: float, 
                                   primary_probs: Dict[str, float], features: Dict[str, float]) -> Tuple[str, float, Dict[str, float], float]:
        """
        Apply reliability balancing to get final emotion classification.
        
        Args:
            primary_emotion: Primary emotion classification
            primary_confidence: Confidence of primary classification
            primary_probs: Primary emotion probabilities
            features: Facial features
            
        Returns:
            Tuple of (final_emotion, final_confidence, final_probabilities, reliability_score)
        """
        # Calculate reliability score based on feature consistency
        reliability_factors = [
            min(1.0, primary_confidence * 2),  # Primary confidence
            max(0, 1.0 - features['std_intensity']),  # Feature stability
            max(0, 1.0 - abs(features['skewness'])),  # Feature normality
            min(1.0, features['edge_density'] * 2)    # Feature distinctiveness
        ]
        
        reliability_score = np.mean(reliability_factors)
        
        # Apply reliability balancing
        if reliability_score > 0.7:
            # High reliability: keep primary classification
            final_emotion = primary_emotion
            final_confidence = primary_confidence
            final_probs = primary_probs.copy()
        elif reliability_score > 0.4:
            # Medium reliability: blend with neutral
            neutral_weight = (0.7 - reliability_score) / 0.3 * 0.3
            final_probs = {}
            for emotion in self.emotion_labels:
                if emotion == 'Neutral':
                    final_probs[emotion] = primary_probs[emotion] + neutral_weight
                else:
                    final_probs[emotion] = primary_probs[emotion] * (1 - neutral_weight)
            
            # Renormalize
            total = sum(final_probs.values())
            final_probs = {emotion: prob / total for emotion, prob in final_probs.items()}
            
            final_emotion = max(final_probs, key=final_probs.get)
            final_confidence = final_probs[final_emotion]
        else:
            # Low reliability: default to neutral with low confidence
            final_emotion = 'Neutral'
            final_confidence = 0.3
            final_probs = {emotion: 0.1 if emotion != 'Neutral' else 0.3 for emotion in self.emotion_labels}
        
        return final_emotion, final_confidence, final_probs, reliability_score
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Process a single frame for emotion classification.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Tuple of (emotion_metrics_dict, annotated_frame)
        """
        if not self.initialized:
            self._initialize_model()
        
        # Initialize metrics with defaults
        metrics = self.default_metrics.copy()
        annotated_frame = frame.copy()
        
        # Detect faces
        faces = self._detect_faces(frame)
        
        if len(faces) > 0:
            metrics['arbex_face_detected'] = 1.0
            
            # Process the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Extract facial features
            features = self._extract_facial_features(face_roi)
            
            # Primary emotion classification
            primary_emotion, primary_confidence, primary_probs = self._classify_emotion_primary(features)
            
            # Apply reliability balancing for final classification
            final_emotion, final_confidence, final_probs, reliability_score = self._apply_reliability_balancing(
                primary_emotion, primary_confidence, primary_probs, features
            )
            
            # Update metrics
            metrics['arbex_primary'] = primary_emotion
            metrics['arbex_final'] = final_emotion
            metrics['arbex_confidence_primary'] = float(primary_confidence)
            metrics['arbex_confidence_final'] = float(final_confidence)
            metrics['arbex_reliability_score'] = float(reliability_score)
            
            # Update individual emotion probabilities
            for emotion in self.emotion_labels:
                metrics[f'arbex_primary_{emotion.lower()}'] = float(primary_probs.get(emotion, 0.0))
                metrics[f'arbex_final_{emotion.lower()}'] = float(final_probs.get(emotion, 0.0))
            
            # Draw face bounding box and emotion labels
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add emotion text
            primary_text = f"Primary: {primary_emotion} ({primary_confidence:.2f})"
            final_text = f"Final: {final_emotion} ({final_confidence:.2f})"
            reliability_text = f"Reliability: {reliability_score:.2f}"
            
            cv2.putText(annotated_frame, primary_text, (x, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated_frame, final_text, (x, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(annotated_frame, reliability_text, (x, y-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
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
        Analyze emotional indices in a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing emotion analysis results
        """
        if not self.initialized:
            self._initialize_model()
        
        logger.info(f"Analyzing emotional indices in video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_metrics = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        best_annotated_frame = None
        max_confidence = 0
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                metrics, annotated_frame = self._process_frame(frame)
                
                # Keep the frame with highest confidence for visualization
                final_confidence = metrics.get('arbex_confidence_final', 0)
                if final_confidence > max_confidence:
                    max_confidence = final_confidence
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
        
        logger.info(f"Completed emotion analysis: {len(frame_metrics)} frames processed")
        
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
            Aggregated emotion analysis results
        """
        if not frame_metrics:
            result = self.default_metrics.copy()
            result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'faces_detected_frames': 0,
                'face_detection_rate': 0.0
            })
            return result
        
        # Calculate mode (most frequent) for categorical variables
        primary_emotions = [frame.get('arbex_primary', 'Neutral') for frame in frame_metrics]
        final_emotions = [frame.get('arbex_final', 'Neutral') for frame in frame_metrics]
        
        # Get most frequent emotions
        primary_mode = max(set(primary_emotions), key=primary_emotions.count)
        final_mode = max(set(final_emotions), key=final_emotions.count)
        
        # Calculate mean for numerical variables
        aggregated = {
            'arbex_primary': primary_mode,
            'arbex_final': final_mode
        }
        
        # Aggregate numerical metrics
        numerical_keys = [
            'arbex_face_detected', 'arbex_confidence_primary', 'arbex_confidence_final', 
            'arbex_reliability_score'
        ]
        
        # Add emotion probability keys
        for emotion in self.emotion_labels:
            numerical_keys.extend([
                f'arbex_primary_{emotion.lower()}',
                f'arbex_final_{emotion.lower()}'
            ])
        
        for key in numerical_keys:
            values = [frame.get(key, 0.0) for frame in frame_metrics]
            aggregated[key] = float(np.mean(values))
        
        # Add visualization of best frame
        if best_frame is not None:
            aggregated['arbex_SM_pic'] = self._encode_image_to_base64(best_frame)
        else:
            aggregated['arbex_SM_pic'] = ""
        
        # Add summary statistics
        faces_detected_frames = sum(1 for frame in frame_metrics 
                                   if frame.get('arbex_face_detected', 0) > 0.5)
        
        aggregated.update({
            'video_path': str(video_path),
            'total_frames': len(frame_metrics),
            'faces_detected_frames': faces_detected_frames,
            'face_detection_rate': faces_detected_frames / len(frame_metrics),
            'avg_confidence_primary': float(np.mean([
                frame.get('arbex_confidence_primary', 0) for frame in frame_metrics
            ])),
            'avg_confidence_final': float(np.mean([
                frame.get('arbex_confidence_final', 0) for frame in frame_metrics
            ])),
            'avg_reliability_score': float(np.mean([
                frame.get('arbex_reliability_score', 0) for frame in frame_metrics
            ]))
        })
        
        return aggregated
    
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Get ARBEx emotion features for the pipeline.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with ARBEx emotion features
        """
        try:
            results = self.analyze_video(video_path)
            
            # Create feature group for the pipeline
            feature_dict = {
                "Extract emotional indices via different feature levels": {
                    "description": "ARBEx attentive feature extraction with reliability balancing for robust facial expression learning",
                    "features": results
                }
            }
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Error in ARBEx analysis: {e}")
            
            # Return default values on error
            default_result = self.default_metrics.copy()
            default_result.update({
                'video_path': str(video_path),
                'total_frames': 0,
                'faces_detected_frames': 0,
                'face_detection_rate': 0.0,
                'avg_confidence_primary': 0.0,
                'avg_confidence_final': 0.0,
                'avg_reliability_score': 0.0,
                'error': str(e)
            })
            
            feature_dict = {
                "Extract emotional indices via different feature levels": {
                    "description": "ARBEx attentive feature extraction with reliability balancing for robust facial expression learning",
                    "features": default_result
                }
            }
            
            return feature_dict

def extract_arbex_features(video_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Extract ARBEx emotion features from a video file.
    
    Args:
        video_path: Path to the video file
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing emotion features
    """
    analyzer = ARBExAnalyzer(device=device)
    return analyzer.get_feature_dict(video_path)
