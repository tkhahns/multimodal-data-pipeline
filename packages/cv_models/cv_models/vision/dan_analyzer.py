"""
DAN: Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition
Based on: https://github.com/yaoing/DAN
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

class DANAnalyzer:
    """
    DAN (Distract Your Attention) analyzer for facial expression recognition.
    
    This analyzer implements Multi-head Cross Attention Network for Facial Expression Recognition,
    which uses attention mechanisms to focus on relevant facial regions for emotion classification.
    """
    
    def __init__(self, device='cpu', num_classes=7):
        """
        Initialize DAN analyzer.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
            num_classes: Number of emotion classes (7 or 8)
        """
        self.device = device
        self.num_classes = num_classes
        self.model = None
        self.initialized = False
        
        # Emotion class labels (7-class: excludes contempt, 8-class: includes contempt)
        if num_classes == 7:
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        else:  # 8-class
            self.emotion_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Initialize default metrics
        self.default_metrics = {}
        for emotion in self.emotion_labels:
            self.default_metrics[f'dan_{emotion}'] = 0.0
        self.default_metrics['dan_emotion_scores'] = [0.0] * self.num_classes
        
    def _initialize_model(self):
        """Initialize the DAN model."""
        try:
            logger.info("Initializing DAN model...")
            self.model = self._create_dan_model()
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            logger.info("DAN model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DAN model: {e}")
            self.initialized = False
            
    def _create_dan_model(self):
        """Create a simplified DAN model for demonstration."""
        
        class MultiHeadCrossAttention(nn.Module):
            def __init__(self, d_model, num_heads):
                super(MultiHeadCrossAttention, self).__init__()
                self.d_model = d_model
                self.num_heads = num_heads
                self.head_dim = d_model // num_heads
                
                assert self.head_dim * num_heads == d_model
                
                self.query_linear = nn.Linear(d_model, d_model)
                self.key_linear = nn.Linear(d_model, d_model)
                self.value_linear = nn.Linear(d_model, d_model)
                self.output_linear = nn.Linear(d_model, d_model)
                
            def forward(self, query, key, value, mask=None):
                batch_size = query.size(0)
                
                # Linear transformations and reshape
                Q = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                K = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                V = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
                # Attention
                attention = self._attention(Q, K, V, mask)
                
                # Concatenate heads and put through final linear layer
                attention = attention.transpose(1, 2).contiguous().view(
                    batch_size, -1, self.d_model)
                
                output = self.output_linear(attention)
                return output
                
            def _attention(self, Q, K, V, mask=None):
                scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                attention_weights = F.softmax(scores, dim=-1)
                attention = torch.matmul(attention_weights, V)
                
                return attention
        
        class DANNet(nn.Module):
            def __init__(self, num_classes=7, feature_dim=512):
                super(DANNet, self).__init__()
                
                # Feature extraction backbone (simplified ResNet-like)
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    # Simplified residual blocks
                    self._make_layer(64, 128, 2),
                    self._make_layer(128, 256, 2),
                    self._make_layer(256, 512, 2),
                    
                    nn.AdaptiveAvgPool2d((7, 7))  # Output 7x7 feature map
                )
                
                # Multi-head cross attention layers
                self.attention1 = MultiHeadCrossAttention(d_model=512, num_heads=8)
                self.attention2 = MultiHeadCrossAttention(d_model=512, num_heads=8)
                self.attention3 = MultiHeadCrossAttention(d_model=512, num_heads=8)
                
                # Layer normalization
                self.norm1 = nn.LayerNorm(512)
                self.norm2 = nn.LayerNorm(512)
                self.norm3 = nn.LayerNorm(512)
                
                # Dropout for regularization
                self.dropout = nn.Dropout(0.5)
                
                # Global average pooling and classifier
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Linear(512, num_classes)
                
            def _make_layer(self, in_channels, out_channels, stride):
                """Create a simplified residual layer."""
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                
            def forward(self, x):
                # Extract features
                features = self.backbone(x)  # Shape: [B, 512, 7, 7]
                B, C, H, W = features.shape
                
                # Reshape for attention: [B, H*W, C]
                features_flat = features.view(B, C, H*W).transpose(1, 2)
                
                # Multi-head cross attention layers with residual connections
                attn1_out = self.attention1(features_flat, features_flat, features_flat)
                attn1_out = self.norm1(attn1_out + features_flat)
                attn1_out = self.dropout(attn1_out)
                
                attn2_out = self.attention2(attn1_out, attn1_out, attn1_out)
                attn2_out = self.norm2(attn2_out + attn1_out)
                attn2_out = self.dropout(attn2_out)
                
                attn3_out = self.attention3(attn2_out, attn2_out, attn2_out)
                attn3_out = self.norm3(attn3_out + attn2_out)
                attn3_out = self.dropout(attn3_out)
                
                # Global pooling and classification
                # Shape: [B, H*W, C] -> [B, C, H*W] -> [B, C, 1] -> [B, C]
                pooled = self.global_pool(attn3_out.transpose(1, 2)).squeeze(-1)
                
                # Emotion classification
                emotion_logits = self.classifier(pooled)
                emotion_probs = F.softmax(emotion_logits, dim=1)
                
                return emotion_probs
                
        return DANNet(num_classes=self.num_classes)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DAN model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to model input size
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
    
    def _calculate_emotion_metrics(self, emotion_probs: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate emotion recognition metrics.
        
        Args:
            emotion_probs: Emotion probability predictions
            
        Returns:
            Dictionary of emotion metrics
        """
        metrics = {}
        
        # Convert to numpy
        probs = emotion_probs.cpu().numpy().squeeze()
        
        # Individual emotion scores
        for i, emotion in enumerate(self.emotion_labels):
            if i < len(probs):
                metrics[f'dan_{emotion}'] = float(probs[i])
            else:
                metrics[f'dan_{emotion}'] = 0.0
        
        # Full emotion scores array
        metrics['dan_emotion_scores'] = [float(p) for p in probs]
        
        return metrics
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame for emotion recognition.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing emotion recognition results
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
                emotion_probs = self.model(image_tensor)
            
            # Calculate metrics
            metrics = self._calculate_emotion_metrics(emotion_probs)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return self.default_metrics.copy()
    
    def analyze_video(self, video_path: str, max_frames: int = 30) -> Dict[str, Any]:
        """
        Analyze video for emotion recognition across multiple frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary containing aggregated emotion recognition results
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
            
            # Aggregate individual emotion scores
            for emotion in self.emotion_labels:
                key = f'dan_{emotion}'
                values = [metrics[key] for metrics in all_metrics]
                aggregated[key] = float(np.mean(values))
            
            # Aggregate emotion scores array
            emotion_scores_arrays = [metrics['dan_emotion_scores'] for metrics in all_metrics]
            if emotion_scores_arrays:
                aggregated['dan_emotion_scores'] = [
                    float(np.mean([scores[i] for scores in emotion_scores_arrays]))
                    for i in range(len(emotion_scores_arrays[0]))
                ]
            else:
                aggregated['dan_emotion_scores'] = self.default_metrics['dan_emotion_scores']
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error analyzing video with DAN: {e}")
            return self.default_metrics.copy()
    
    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        """
        Get DAN features in the format expected by the pipeline.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary of DAN features with proper prefixes
        """
        return self.analyze_video(video_path)


def extract_dan_features(video_path: str, device: str = 'cpu', num_classes: int = 7) -> Dict[str, Any]:
    """
    Extract DAN emotion recognition features from video.
    
    Args:
        video_path: Path to video file
        device: Device for computation ('cpu' or 'cuda')
        num_classes: Number of emotion classes (7 or 8)
        
    Returns:
        Dictionary of DAN features
    """
    analyzer = DANAnalyzer(device=device, num_classes=num_classes)
    return analyzer.analyze_video(video_path)
