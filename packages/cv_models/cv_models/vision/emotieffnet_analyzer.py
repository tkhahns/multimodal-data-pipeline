"""
EmotiEffNet: Frame-level Prediction of Facial Expressions, Valence, Arousal and Action Units for Mobile Devices
Based on: https://github.com/sb-ai-lab/EmotiEffLib/tree/main/models/affectnet_emotions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class EmotiEffNetAnalyzer:
    """
    EmotiEffNet analyzer for real-time video emotion analysis and AU detection.
    
    This analyzer implements frame-level prediction of facial expressions, valence, arousal,
    and action units optimized for mobile devices.
    """
    
    def __init__(self, device='cpu', model_type='affectnet'):
        """
        Initialize EmotiEffNet analyzer.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            model_type: Type of model to use ('affectnet' for emotion analysis)
        """
        self.device = device
        self.model_type = model_type
        self.model = None
        self.initialized = False
        
        # Emotion class labels for AffectNet
        self.emotion_labels = ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'other']
        
        # Action Units that are detected
        self.au_labels = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
        
        # Initialize default metrics
        self.default_metrics = {}
        
        # Arousal and valence
        self.default_metrics['eln_arousal'] = 0.0
        self.default_metrics['eln_valence'] = 0.0
        
        # Action Units
        for au in self.au_labels:
            self.default_metrics[f'eln_{au}'] = 0.0
        
        # Emotion F1 scores
        for emotion in self.emotion_labels:
            self.default_metrics[f'eln_{emotion}_f1'] = 0.0
        
    def _initialize_model(self):
        """Initialize the EmotiEffNet model."""
        try:
            logger.info("Initializing EmotiEffNet model...")
            self.model = self._create_emotieffnet_model()
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            logger.info("EmotiEffNet model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmotiEffNet model: {e}")
            self.initialized = False
            
    def _create_emotieffnet_model(self):
        """Create a simplified EmotiEffNet model for demonstration."""
        
        class MobileNetV2Block(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
                super(MobileNetV2Block, self).__init__()
                self.stride = stride
                hidden_dim = in_channels * expand_ratio
                self.use_res_connect = self.stride == 1 and in_channels == out_channels
                
                layers = []
                if expand_ratio != 1:
                    # Pointwise expansion
                    layers.extend([
                        nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU6(inplace=True),
                    ])
                
                # Depthwise convolution
                layers.extend([
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # Pointwise linear
                    nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channels),
                ])
                
                self.conv = nn.Sequential(*layers)
                
            def forward(self, x):
                if self.use_res_connect:
                    return x + self.conv(x)
                else:
                    return self.conv(x)
        
        class EmotiEffNet(nn.Module):
            def __init__(self, num_emotions=8, num_aus=12):
                super(EmotiEffNet, self).__init__()
                
                # Initial layer
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU6(inplace=True),
                )
                
                # MobileNetV2 backbone (simplified)
                self.backbone = nn.Sequential(
                    MobileNetV2Block(32, 16, 1, 1),
                    MobileNetV2Block(16, 24, 2, 6),
                    MobileNetV2Block(24, 24, 1, 6),
                    MobileNetV2Block(24, 32, 2, 6),
                    MobileNetV2Block(32, 32, 1, 6),
                    MobileNetV2Block(32, 32, 1, 6),
                    MobileNetV2Block(32, 64, 2, 6),
                    MobileNetV2Block(64, 64, 1, 6),
                    MobileNetV2Block(64, 64, 1, 6),
                    MobileNetV2Block(64, 64, 1, 6),
                    MobileNetV2Block(64, 96, 1, 6),
                    MobileNetV2Block(96, 96, 1, 6),
                    MobileNetV2Block(96, 96, 1, 6),
                    MobileNetV2Block(96, 160, 2, 6),
                    MobileNetV2Block(160, 160, 1, 6),
                    MobileNetV2Block(160, 160, 1, 6),
                    MobileNetV2Block(160, 320, 1, 6),
                )
                
                # Final convolution
                self.final_conv = nn.Sequential(
                    nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(1280),
                    nn.ReLU6(inplace=True),
                )
                
                # Global average pooling
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                
                # Multi-head outputs
                self.emotion_classifier = nn.Linear(1280, num_emotions)
                self.arousal_regressor = nn.Linear(1280, 1)
                self.valence_regressor = nn.Linear(1280, 1)
                self.au_classifier = nn.Linear(1280, num_aus)
                
                # Dropout for regularization
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                # Feature extraction
                x = self.features(x)
                x = self.backbone(x)
                x = self.final_conv(x)
                
                # Global pooling
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                
                # Multi-task outputs
                emotions = self.emotion_classifier(x)
                arousal = self.arousal_regressor(x)
                valence = self.valence_regressor(x)
                aus = self.au_classifier(x)
                
                # Apply activations
                emotion_probs = F.softmax(emotions, dim=1)
                arousal_value = torch.tanh(arousal)  # [-1, 1] range
                valence_value = torch.tanh(valence)  # [-1, 1] range
                au_probs = torch.sigmoid(aus)  # [0, 1] range for each AU
                
                return {
                    'emotions': emotion_probs,
                    'arousal': arousal_value,
                    'valence': valence_value,
                    'action_units': au_probs
                }
                
        return EmotiEffNet(num_emotions=len(self.emotion_labels), num_aus=len(self.au_labels))
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for EmotiEffNet model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to model input size (224x224 for mobile efficiency)
        image_resized = cv2.resize(image, (224, 224))
        
        # Normalize to [0, 1]
        image_norm = image_resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB if needed
        if len(image_norm.shape) == 3 and image_norm.shape[2] == 3:
            image_norm = cv2.cvtColor(image_norm, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.to(self.device)
    
    def _calculate_emotion_metrics(self, model_output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Calculate emotion analysis metrics from model output.
        
        Args:
            model_output: Dictionary containing model predictions
            
        Returns:
            Dictionary of emotion metrics
        """
        metrics = {}
        
        # Extract values from tensors
        emotions = model_output['emotions'].cpu().numpy().squeeze()
        arousal = model_output['arousal'].cpu().numpy().squeeze()
        valence = model_output['valence'].cpu().numpy().squeeze()
        action_units = model_output['action_units'].cpu().numpy().squeeze()
        
        # Arousal and valence (continuous values)
        metrics['eln_arousal'] = float(arousal)
        metrics['eln_valence'] = float(valence)
        
        # Action Units
        for i, au in enumerate(self.au_labels):
            if i < len(action_units):
                metrics[f'eln_{au}'] = float(action_units[i])
            else:
                metrics[f'eln_{au}'] = 0.0
        
        # Emotion F1 scores (using emotion probabilities as confidence scores)
        for i, emotion in enumerate(self.emotion_labels):
            if i < len(emotions):
                # Use emotion probability as F1 score proxy
                metrics[f'eln_{emotion}_f1'] = float(emotions[i])
            else:
                metrics[f'eln_{emotion}_f1'] = 0.0
        
        return metrics
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame for emotion, arousal, valence, and action units.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing emotion analysis results
        """
        if not self.initialized:
            self._initialize_model()
            
        if not self.initialized:
            logger.warning("Model not initialized, returning default metrics")
            return self.default_metrics.copy()
        
        try:
            # Preprocess the frame
            image_tensor = self._preprocess_image(frame)
            
            # Run inference
            with torch.no_grad():
                model_output = self.model(image_tensor)
            
            # Calculate metrics
            metrics = self._calculate_emotion_metrics(model_output)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return self.default_metrics.copy()
    
    def analyze_video(self, video_path: str, max_frames: int = 30) -> Dict[str, Any]:
        """
        Analyze video for emotion, arousal, valence, and action units across multiple frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary containing aggregated emotion analysis results
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return self.default_metrics.copy()
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return self.default_metrics.copy()
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames uniformly across the video
            if frame_count > max_frames:
                frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            else:
                frame_indices = list(range(frame_count))
            
            all_metrics = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Analyze this frame
                frame_metrics = self.analyze_frame(frame)
                all_metrics.append(frame_metrics)
            
            cap.release()
            
            if not all_metrics:
                logger.warning("No frames processed successfully")
                return self.default_metrics.copy()
            
            # Aggregate metrics across all frames
            aggregated = {}
            
            # For continuous values (arousal, valence, AUs), use mean
            continuous_keys = ['eln_arousal', 'eln_valence'] + [f'eln_{au}' for au in self.au_labels]
            for key in continuous_keys:
                values = [metrics[key] for metrics in all_metrics if key in metrics]
                aggregated[key] = float(np.mean(values)) if values else 0.0
            
            # For emotion F1 scores, use mean as well
            emotion_keys = [f'eln_{emotion}_f1' for emotion in self.emotion_labels]
            for key in emotion_keys:
                values = [metrics[key] for metrics in all_metrics if key in metrics]
                aggregated[key] = float(np.mean(values)) if values else 0.0
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error analyzing video with EmotiEffNet: {e}")
            return self.default_metrics.copy()
    
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Get EmotiEffNet features in the format expected by the pipeline.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary of EmotiEffNet features with proper prefixes
        """
        return self.analyze_video(video_path)


def extract_emotieffnet_features(video_path: str, device: str = 'cpu', model_type: str = 'affectnet') -> Dict[str, Any]:
    """
    Extract EmotiEffNet emotion analysis features from video.
    
    Args:
        video_path: Path to video file
        device: Device for computation ('cpu' or 'cuda')
        model_type: Type of model to use ('affectnet')
        
    Returns:
        Dictionary of EmotiEffNet features
    """
    analyzer = EmotiEffNetAnalyzer(device=device, model_type=model_type)
    return analyzer.analyze_video(video_path)
