"""
Py-Feat Analyzer wrapper to produce pf_* features (AUs, emotions, face geometry).
"""
from typing import Dict, Any
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class PyFeatAnalyzer:
    def __init__(self, device: str = 'cpu'):
        try:
            from feat import Detector
        except Exception as e:
            raise ImportError("py-feat is not installed or failed to import. Install 'py-feat'.") from e
        self.device = device
        # Use default face model and AU/emotion heads
        self.detector = Detector()

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        try:
            # Run detection on video; Detector can accept paths
            result = self.detector.detect_video(video_path)
            # result is a pandas DataFrame-like object (FeatData)
            df = result.to_pandas() if hasattr(result, 'to_pandas') else result

            features: Dict[str, Any] = {}

            # Action Units: columns often like AU01_r, AU02_r, ...
            au_map = {
                'pf_au01': 'AU01_r','pf_au02': 'AU02_r','pf_au04': 'AU04_r','pf_au05': 'AU05_r','pf_au06': 'AU06_r',
                'pf_au07': 'AU07_r','pf_au09': 'AU09_r','pf_au10': 'AU10_r','pf_au11': 'AU11_r','pf_au12': 'AU12_r',
                'pf_au14': 'AU14_r','pf_au15': 'AU15_r','pf_au17': 'AU17_r','pf_au20': 'AU20_r','pf_au23': 'AU23_r',
                'pf_au24': 'AU24_r','pf_au25': 'AU25_r','pf_au26': 'AU26_r','pf_au28': 'AU28_r','pf_au43': 'AU43_r'
            }
            for out_key, col in au_map.items():
                if col in df.columns:
                    features[out_key] = float(np.nanmean(df[col].values))

            # Emotions (py-feat EmoNet columns e.g., anger, disgust, fear, happiness, sadness, surprise, neutral)
            emo_cols = ['anger','disgust','fear','happiness','sadness','surprise','neutral']
            for emo in emo_cols:
                if emo in df.columns:
                    features[f'pf_{emo}'] = float(np.nanmean(df[emo].values))

            # Face rect and score if available (e.g., facebox coords, confidence)
            for col_name, out_prefix in [
                ('face_x', 'pf_facerectx'), ('face_y', 'pf_facerecty'),
                ('face_w', 'pf_facerectwidth'), ('face_h', 'pf_facerectheight'), ('face_score', 'pf_facescore')
            ]:
                if col_name in df.columns:
                    features[out_prefix] = float(np.nanmean(df[col_name].values))

            # Head pose: pitch/roll/yaw if provided by model
            for ang in ['pitch','roll','yaw']:
                if ang in df.columns:
                    features[f'pf_{ang}'] = float(np.nanmean(df[ang].values))

            # 3D face landmarks summary if available (x,y,z means)
            for axis in ['x','y','z']:
                coln = f'landmark_{axis}'
                if coln in df.columns:
                    features[f'pf_{axis}'] = float(np.nanmean(df[coln].values))

            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": features
                }
            }
        except Exception as e:
            logger.error(f"Py-Feat analysis failed: {e}")
            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": {"pf_error": str(e)}
                }
            }
