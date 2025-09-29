"""
Py-Feat Analyzer wrapper to produce pf_* features (AUs, emotions, face geometry).
"""
from typing import Dict, Any
import numpy as np
import cv2
import logging
import json
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class PyFeatAnalyzer:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._use_subprocess = False
        # Project root = .../multimodal-data-pipeline
    # __file__ is .../packages/cv_models/cv_models/vision/pyfeat_analyzer.py
        self._runner_path = Path(__file__).resolve().parents[2] / 'external' / 'pyfeat_runner'
        try:
            from feat import Detector  # type: ignore
            # Use default face model and AU/emotion heads
            self.detector = Detector()
        except Exception as e:
            logger.warning(f"py-feat import failed in main env (likely Python 3.12). Will try subprocess runner. Error: {e}")
            self._use_subprocess = True
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

    def _run_subprocess(self, video_path: str) -> Dict[str, Any]:
        runner_dir = self._runner_path
        if not runner_dir.exists():
            return {"pf_error": f"pyfeat_runner not found at {runner_dir}"}

        # Try invoking via Poetry first, then python3.11, then current python
        cmds = []
        pyproject = runner_dir / 'pyproject.toml'
        if pyproject.exists():
            cmds.append(["poetry", "run", "python", "-m", "pyfeat_runner", video_path])
        cmds.append(["python3.11", "-m", "pyfeat_runner", video_path])
        cmds.append(["python", "-m", "pyfeat_runner", video_path])

        last_err = None
        for cmd in cmds:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(runner_dir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout:
                    data = json.loads(proc.stdout or '{}')
                    if 'features' in data:
                        return data['features']
                    if 'error' in data:
                        return {"pf_error": data['error']}
                    return data
                else:
                    last_err = f"code={proc.returncode} stderr={proc.stderr.strip()}"
            except FileNotFoundError as e:
                last_err = f"{e}"
                continue
            except Exception as e:
                last_err = f"{e}"
                continue
        return {"pf_error": f"runner failed: {last_err}"}

    def get_feature_dict(self, video_path: str) -> Dict[str, Any]:
        try:
            if not self._use_subprocess and self.detector is not None:
                # Run in-process
                result = self.detector.detect_video(video_path)
                df = result.to_pandas() if hasattr(result, 'to_pandas') else result
                features = self._postprocess_df(df)
            else:
                # Fallback to subprocess runner (Python 3.11 env)
                features = self._run_subprocess(video_path)

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
