"""
Vision processing module for the multimodal data pipeline.

This module provides computer vision capabilities including:
- Human pose estimation and 3D body modeling
- Part attention regression
- Body part analysis
- 3D mesh generation
- Vision Transformer-based pose estimation
- Residual Steps Network for keypoint localization
- Polarized Self-Attention for keypoint heatmaps and segmentation masks
- Facial Action Unit recognition with AU relation graphs
- Emotional expression recognition with cross-attention networks
"""

from .pare_analyzer import PAREAnalyzer, create_pare_analyzer
from .vitpose_analyzer import ViTPoseAnalyzer, create_vitpose_analyzer
from .psa_analyzer import PSAAnalyzer, create_psa_analyzer
from .rsn_analyzer import RSNAnalyzer, extract_rsn_features
from .me_graphau_analyzer import MEGraphAUAnalyzer, extract_me_graphau_features
from .dan_analyzer import DANAnalyzer, extract_dan_features

__all__ = [
    'PAREAnalyzer',
    'create_pare_analyzer',
    'ViTPoseAnalyzer', 
    'create_vitpose_analyzer',
    'PSAAnalyzer',
    'create_psa_analyzer',
    'RSNAnalyzer',
    'extract_rsn_features',
    'MEGraphAUAnalyzer',
    'extract_me_graphau_features',
    'DANAnalyzer',
    'extract_dan_features'
]
