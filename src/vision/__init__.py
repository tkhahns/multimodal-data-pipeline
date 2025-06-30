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
- Real-time video emotion analysis and AU detection for mobile devices
- Google MediaPipe pose estimation and tracking with 33 landmarks
- Deep High-Resolution Network for high-precision pose estimation
- Simple Baselines for human pose estimation and tracking
- Py-Feat facial expression analysis with action units and emotions
- GANimation continuous manifold for anatomical facial movements
- ARBEx attentive feature extraction with reliability balancing for robust facial expression learning
- OpenPose real-time multi-person keypoint detection and pose estimation
- Insta-DM instant dense monocular depth estimation with motion analysis
- Optical Flow movement and estimation of motion with sparse and dense analysis
"""

from .pare_analyzer import PAREAnalyzer, create_pare_analyzer
from .vitpose_analyzer import ViTPoseAnalyzer, create_vitpose_analyzer
from .psa_analyzer import PSAAnalyzer, create_psa_analyzer
from .rsn_analyzer import RSNAnalyzer, extract_rsn_features
from .me_graphau_analyzer import MEGraphAUAnalyzer, extract_me_graphau_features
from .dan_analyzer import DANAnalyzer, extract_dan_features
from .emotieffnet_analyzer import EmotiEffNetAnalyzer, extract_emotieffnet_features
from .mediapipe_pose_analyzer import MediaPipePoseAnalyzer, extract_mediapipe_pose_features
from .deep_hrnet_analyzer import DeepHRNetAnalyzer, extract_deep_hrnet_features
from .simple_baselines_analyzer import SimpleBaselinesAnalyzer, extract_simple_baselines_features
from .pyfeat_analyzer import PyFeatAnalyzer, extract_pyfeat_features
from .ganimation_analyzer import GANimationAnalyzer, extract_ganimation_features
from .arbex_analyzer import ARBExAnalyzer, extract_arbex_features
from .openpose_analyzer import OpenPoseAnalyzer, extract_openpose_features
from .instadm_analyzer import InstaDMAnalyzer, extract_instadm_features
from .optical_flow_analyzer import OpticalFlowAnalyzer, extract_optical_flow_features
from .crowdflow_analyzer import CrowdFlowAnalyzer, extract_crowdflow_features

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
    'extract_dan_features',    'EmotiEffNetAnalyzer',
    'extract_emotieffnet_features',
    'MediaPipePoseAnalyzer',
    'extract_mediapipe_pose_features',
    'DeepHRNetAnalyzer',    'extract_deep_hrnet_features',
    'SimpleBaselinesAnalyzer',    'extract_simple_baselines_features',
    'PyFeatAnalyzer',    'extract_pyfeat_features',    'GANimationAnalyzer',
    'extract_ganimation_features',    'ARBExAnalyzer',
    'extract_arbex_features',    'OpenPoseAnalyzer',
    'extract_openpose_features',    'InstaDMAnalyzer',
    'extract_instadm_features',
    'OpticalFlowAnalyzer',
    'extract_optical_flow_features',
    'CrowdFlowAnalyzer',
    'extract_crowdflow_features'
]
