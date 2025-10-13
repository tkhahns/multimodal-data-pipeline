"""
Py-Feat Analyzer wrapper to produce pf_* features (AUs, emotions, face geometry).
"""
from typing import Dict, Any, Optional
import numpy as np
import logging

from cv_models.utils.scipy_compat import ensure_legacy_stats

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = (
    "pf_au01",
    "pf_au02",
    "pf_au04",
    "pf_au05",
    "pf_au06",
    "pf_au07",
    "pf_au09",
    "pf_au10",
    "pf_au11",
    "pf_au12",
    "pf_au14",
    "pf_au15",
    "pf_au17",
    "pf_au20",
    "pf_au23",
    "pf_au24",
    "pf_au25",
    "pf_au26",
    "pf_au28",
    "pf_au43",
    "pf_anger",
    "pf_disgust",
    "pf_fear",
    "pf_happiness",
    "pf_sadness",
    "pf_surprise",
    "pf_neutral",
    "pf_facerectx",
    "pf_facerecty",
    "pf_facerectwidth",
    "pf_facerectheight",
    "pf_facescore",
    "pf_pitch",
    "pf_roll",
    "pf_yaw",
    "pf_x",
    "pf_y",
    "pf_z",
)


class PyFeatAnalyzer:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._init_error: Optional[str] = None
        ensure_legacy_stats()
        self.detector = None

        try:
            from feat import Detector  # type: ignore
            # Use default face model and AU/emotion heads
            self.detector = Detector()
        except Exception as e:
            message = f"Py-Feat detector init failed in main environment: {e}"
            logger.warning(message)
            self._init_error = str(e)
            self.detector = None

    def _postprocess_df(self, df) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        au_map = {
            'pf_au01': 'AU01_r','pf_au02': 'AU02_r','pf_au04': 'AU04_r','pf_au05': 'AU05_r','pf_au06': 'AU06_r',
            'pf_au07': 'AU07_r','pf_au09': 'AU09_r','pf_au10': 'AU10_r','pf_au11': 'AU11_r','pf_au12': 'AU12_r',
            'pf_au14': 'AU14_r','pf_au15': 'AU15_r','pf_au17': 'AU17_r','pf_au20': 'AU20_r','pf_au23': 'AU23_r',
            'pf_au24': 'AU24_r','pf_au25': 'AU25_r','pf_au26': 'AU26_r','pf_au28': 'AU28_r','pf_au43': 'AU43_r'
        }
        for out_key, col in au_map.items():
            if col in getattr(df, 'columns', []):
                features[out_key] = float(np.nanmean(df[col].values))

        emo_cols = ['anger','disgust','fear','happiness','sadness','surprise','neutral']
        for emo in emo_cols:
            if emo in getattr(df, 'columns', []):
                features[f'pf_{emo}'] = float(np.nanmean(df[emo].values))

        for col_name, out_prefix in [
            ('face_x', 'pf_facerectx'), ('face_y', 'pf_facerecty'),
            ('face_w', 'pf_facerectwidth'), ('face_h', 'pf_facerectheight'), ('face_score', 'pf_facescore')
        ]:
            if col_name in getattr(df, 'columns', []):
                features[out_prefix] = float(np.nanmean(df[col_name].values))

        for ang in ['pitch','roll','yaw']:
            if ang in getattr(df, 'columns', []):
                features[f'pf_{ang}'] = float(np.nanmean(df[ang].values))

        for axis in ['x','y','z']:
            coln = f'landmark_{axis}'
            if coln in getattr(df, 'columns', []):
                features[f'pf_{axis}'] = float(np.nanmean(df[coln].values))
        return features

    @staticmethod
    def _ensure_required_features(features: Dict[str, Any]) -> Dict[str, Any]:
        for key in REQUIRED_FEATURES:
            features.setdefault(key, 0.0)
        return features

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        if self.detector is None:
            message = self._init_error or "Py-Feat is not available in this environment."
            logger.error(f"Py-Feat detector unavailable: {message}")
            features = self._ensure_required_features({})
            features["pf_error"] = message
            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": features
                }
            }

        try:
            result = self.detector.detect_video(video_path)
            df = result.to_pandas() if hasattr(result, 'to_pandas') else result
            features = self._postprocess_df(df)
            features = self._ensure_required_features(features)
            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": features
                }
            }
        except Exception as e:
            logger.error(f"Py-Feat analysis failed: {e}")
            features = self._ensure_required_features({})
            features["pf_error"] = str(e)
            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": features
                }
            }
