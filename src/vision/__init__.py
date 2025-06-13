"""
Vision processing module for the multimodal data pipeline.

This module provides computer vision capabilities including:
- Human pose estimation and 3D body modeling
- Part attention regression
- Body part analysis
- 3D mesh generation
"""

from .pare_analyzer import PAREAnalyzer, create_pare_analyzer

__all__ = [
    'PAREAnalyzer',
    'create_pare_analyzer'
]
