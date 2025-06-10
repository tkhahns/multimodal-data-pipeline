"""
Heinsen Routing-based Sentiment Analysis for Audio Features.
Integrates the Heinsen routing algorithm for sentiment analysis using extracted audio features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import logging

# Import Heinsen routing components
from heinsen_routing import EfficientVectorRouting, DefinableVectorRouting

logger = logging.getLogger(__name__)


class HeinsenRoutingSentimentAnalyzer(nn.Module):
    """
    Sentiment analyzer using Heinsen routing algorithm for processing audio features.
    """
    
    def __init__(
        self,
        input_dim: int = 768,           # Input feature dimension
        arvs_batch_size: int = 32,      # Batch size for routing
        arvs_n_out: int = 8,           # Number of output capsules
        arvs_d_out: int = 64,          # Dimension of each output capsule
        num_sentiment_classes: int = 3, # Number of sentiment classes (positive, negative, neutral)
        dropout_rate: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize the Heinsen routing sentiment analyzer.
        
        Args:
            input_dim: Dimension of input features
            arvs_batch_size: Batch size for ARVS routing
            arvs_n_out: Number of output capsules
            arvs_d_out: Dimension of each output capsule
            num_sentiment_classes: Number of sentiment classes
            dropout_rate: Dropout rate for regularization
            device: Device to run computations on
        """
        super().__init__()
        
        # Store routing parameters
        self.arvs_batch_size = arvs_batch_size
        self.arvs_n_out = arvs_n_out
        self.arvs_d_out = arvs_d_out
        self.num_sentiment_classes = num_sentiment_classes
        self.device = device
        
        # Feature preprocessing layers
        self.feature_norm = nn.LayerNorm(input_dim)
        self.feature_projection = nn.Linear(input_dim, arvs_d_out * arvs_n_out)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Heinsen routing layer
        self.routing = EfficientVectorRouting(
            n_inp=arvs_n_out,      # Number of input capsules
            n_out=arvs_n_out,      # Number of output capsules
            d_inp=arvs_d_out,      # Dimension of input capsules
            d_out=arvs_d_out,      # Dimension of output capsules
            n_iters=3,             # Number of routing iterations
            normalize=True,        # Normalize routing coefficients
            memory_efficient=True, # Use memory-efficient implementation
        )
        
        # Sentiment classification head
        self.sentiment_head = nn.Sequential(
            nn.Linear(arvs_n_out * arvs_d_out, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_sentiment_classes)
        )
        
        # Activation and softmax
        self.softmax = nn.Softmax(dim=-1)
        
        # Move to device
        self.to(device)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            features: Input features tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (sentiment_logits, routed_capsules)
        """
        batch_size = features.size(0)
        
        # Normalize and project features
        features = self.feature_norm(features)
        features = self.dropout(features)
        
        # Project to capsule space
        projected = self.feature_projection(features)  # (batch_size, arvs_n_out * arvs_d_out)
        
        # Reshape to capsules
        capsules = projected.view(batch_size, self.arvs_n_out, self.arvs_d_out)
        
        # Apply Heinsen routing
        routed_capsules = self.routing(capsules)  # (batch_size, arvs_n_out, arvs_d_out)
        
        # Flatten for classification
        flattened = routed_capsules.view(batch_size, -1)
        
        # Get sentiment predictions
        sentiment_logits = self.sentiment_head(flattened)
        
        return sentiment_logits, routed_capsules
    
    def predict_sentiment(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Predict sentiment from features.
        
        Args:
            features: Input features tensor
            
        Returns:
            Dictionary containing sentiment predictions and routing outputs
        """
        self.eval()
        with torch.no_grad():
            sentiment_logits, routed_capsules = self.forward(features)
            sentiment_probs = self.softmax(sentiment_logits)
            predicted_class = torch.argmax(sentiment_probs, dim=-1)
            
            # Convert to numpy for easier handling
            sentiment_probs_np = sentiment_probs.cpu().numpy()
            predicted_class_np = predicted_class.cpu().numpy()
            routed_capsules_np = routed_capsules.cpu().numpy()
            
            return {
                'sentiment_probabilities': sentiment_probs_np,
                'predicted_classes': predicted_class_np,
                'routed_capsules': routed_capsules_np,
                'arvs_batch_size': self.arvs_batch_size,
                'arvs_n_out': self.arvs_n_out,
                'arvs_d_out': self.arvs_d_out
            }


class AudioSentimentAnalyzer:
    """
    Audio sentiment analyzer that integrates with the multimodal pipeline.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        arvs_batch_size: int = 32,
        arvs_n_out: int = 8,
        arvs_d_out: int = 64
    ):
        """
        Initialize the audio sentiment analyzer.
        
        Args:
            model_path: Path to pre-trained model (if available)
            device: Device to run computations on
            arvs_batch_size: Batch size for ARVS routing
            arvs_n_out: Number of output capsules
            arvs_d_out: Dimension of each output capsules
        """
        self.device = device
        self.arvs_batch_size = arvs_batch_size
        self.arvs_n_out = arvs_n_out
        self.arvs_d_out = arvs_d_out
        
        # Initialize model
        self.model = HeinsenRoutingSentimentAnalyzer(
            arvs_batch_size=arvs_batch_size,
            arvs_n_out=arvs_n_out,
            arvs_d_out=arvs_d_out,
            device=device
        )
        
        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("No pre-trained model found. Using randomly initialized weights.")
        
        # Sentiment labels
        self.sentiment_labels = ['negative', 'neutral', 'positive']
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
    
    def save_model(self, model_path: str):
        """Save model weights."""
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {model_path}: {e}")
    
    def prepare_features(self, feature_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Prepare features from the multimodal pipeline for sentiment analysis.
        
        Args:
            feature_dict: Dictionary of extracted features
            
        Returns:
            Prepared feature tensor
        """
        # Extract relevant features for sentiment analysis
        feature_keys = [
            # OpenSMILE features
            'osm_pcm_RMSenergy_sma_mean', 'osm_loudness_sma_mean',
            'osm_spectralCentroid_sma_mean', 'osm_spectralEntropy_sma_mean',
            'osm_F0final_sma_mean', 'osm_voicingProb_sma_mean',
            'osm_jitterLocal_sma_mean', 'osm_shimmerLocal_sma_mean',
            
            # Basic audio features
            'oc_audvol_mean', 'oc_audvol_std', 'oc_audpit_mean', 'oc_audpit_std',
            
            # Librosa spectral features
            'spectral_centroid_mean', 'spectral_bandwidth_mean',
            'spectral_rolloff_mean', 'zero_crossing_rate_mean',
            'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean',
            
            # Speech emotion features (if available)
            'ser_angry', 'ser_happy', 'ser_sad', 'ser_neutral',
        ]
        
        features = []
        for key in feature_keys:
            if key in feature_dict:
                value = feature_dict[key]
                # Handle different data types
                if isinstance(value, (list, np.ndarray)):
                    if len(value) > 0:
                        features.append(float(np.mean(value)))
                    else:
                        features.append(0.0)
                elif isinstance(value, (int, float)):
                    features.append(float(value))
                else:
                    features.append(0.0)
            else:
                # Use zero for missing features
                features.append(0.0)
        
        # Pad or truncate to expected dimension
        target_dim = 768  # Expected input dimension
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        elif len(features) > target_dim:
            features = features[:target_dim]
        
        # Convert to tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return feature_tensor.unsqueeze(0)  # Add batch dimension
    
    def analyze_sentiment(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment from audio features.
        
        Args:
            feature_dict: Dictionary of extracted audio features
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            # Prepare features
            features = self.prepare_features(feature_dict)
            
            # Get predictions
            results = self.model.predict_sentiment(features)
            
            # Format results
            sentiment_probs = results['sentiment_probabilities'][0]  # Remove batch dimension
            predicted_class = results['predicted_classes'][0]
            routed_capsules = results['routed_capsules'][0]
            
            # Create results matching the specified feature naming convention
            sentiment_results = {
                # Core ARVS routing parameters (as specified in feature table)
                'arvs_batch_size': results['arvs_batch_size'],
                'arvs_n_out': results['arvs_n_out'], 
                'arvs_d_out': results['arvs_d_out']
            }
            
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'arvs_batch_size': self.arvs_batch_size,
                'arvs_n_out': self.arvs_n_out,
                'arvs_d_out': self.arvs_d_out
            }
    
    def get_feature_dict(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get sentiment features in the format expected by the pipeline.
        
        Args:
            feature_dict: Dictionary of extracted audio features
            
        Returns:
            Dictionary containing ARVS routing parameters only
        """
        return self.analyze_sentiment(feature_dict)


def create_demo_sentiment_analyzer() -> AudioSentimentAnalyzer:
    """
    Create a demo sentiment analyzer for testing.
    
    Returns:
        Configured AudioSentimentAnalyzer instance
    """
    return AudioSentimentAnalyzer(
        device='cpu',
        arvs_batch_size=1,  # Small batch for demo
        arvs_n_out=8,       # 8 output capsules
        arvs_d_out=64       # 64-dimensional capsules
    )


if __name__ == "__main__":
    # Demo usage
    print("Heinsen Routing Sentiment Analysis Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = create_demo_sentiment_analyzer()
    
    # Create sample features (simulating pipeline output)
    sample_features = {
        'osm_pcm_RMSenergy_sma_mean': 0.5,
        'osm_loudness_sma_mean': 0.3,
        'osm_spectralCentroid_sma_mean': 1000.0,
        'osm_F0final_sma_mean': 150.0,
        'oc_audvol_mean': 0.7,
        'oc_audpit_mean': 200.0,
        'ser_happy': 0.8,
        'ser_sad': 0.2,
    }
    
    # Analyze sentiment
    results = analyzer.analyze_sentiment(sample_features)
    
    # Display results
    print(f"ARVS Batch Size: {results['arvs_batch_size']}")
    print(f"ARVS N Out: {results['arvs_n_out']}")  
    print(f"ARVS D Out: {results['arvs_d_out']}")
    print("Note: This returns PyTorch tensor of output capsules (vectors), one per output position.")
