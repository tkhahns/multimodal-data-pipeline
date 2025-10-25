"""
Py-Feat Analyzer wrapper to produce pf_* features (AUs, emotions, face geometry).
"""
from typing import Dict, Any, Optional, Tuple
import importlib.util
import logging
import sys

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

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
        self._ensure_lib2to3()
        self.detector = None

        try:
            # Patch sklearn tree node dtype compatibility
            self._patch_sklearn_compat()
            
            # Patch torch.nn.Module.load_state_dict to handle DataParallel mismatch
            import torch.nn as nn  # type: ignore[import]
            original_load_state_dict = nn.Module.load_state_dict
            
            def patched_load_state_dict(self, state_dict, strict=True):
                """Patched load_state_dict that handles DataParallel prefix mismatch."""
                if isinstance(state_dict, dict) and state_dict:
                    model_keys = set(self.state_dict().keys())
                    state_keys = set(state_dict.keys())
                    
                    # Check if we have a module. prefix mismatch
                    model_has_module = any(k.startswith('module.') for k in model_keys)
                    state_has_module = any(k.startswith('module.') for k in state_keys)
                    
                    if model_has_module and not state_has_module:
                        # Model expects module. prefix but state_dict doesn't have it
                        logger.info("Adding 'module.' prefix to state_dict keys")
                        state_dict = {'module.' + k: v for k, v in state_dict.items()}
                    elif not model_has_module and state_has_module:
                        # State_dict has module. prefix but model doesn't expect it
                        logger.info("Removing 'module.' prefix from state_dict keys")
                        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                
                return original_load_state_dict(self, state_dict, strict=strict)
            
            # Apply the monkey patch
            nn.Module.load_state_dict = patched_load_state_dict
            
            try:
                from feat import Detector  # type: ignore
                # Use default face model and AU/emotion heads
                self.detector = Detector()
            finally:
                # Restore original method
                nn.Module.load_state_dict = original_load_state_dict
                
                # Restore sklearn tree dtype if it was patched
                self._restore_sklearn_dtype()
                
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

        cleanup_dir: Optional[Path] = None
        try:
            try:
                feature_values = self._detect_features(video_path)
            except Exception as first_error:
                logger.warning("Py-Feat detection failed (%s); attempting sanitized copy", first_error)
                sanitized_path, cleanup_dir = self._sanitize_video(video_path)
                if sanitized_path is None:
                    raise

                try:
                    feature_values = self._detect_features(sanitized_path)
                except Exception as sanitized_error:
                    raise RuntimeError(f"Py-Feat sanitized detection failed: {sanitized_error}") from first_error

            return {
                "Facial Expression (Py-Feat)": {
                    "description": "Py-Feat: Python Facial Expression Analysis Toolbox",
                    "features": feature_values
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
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    @staticmethod
    def _patch_sklearn_compat():
        """Patch sklearn tree loading to handle old pickle format without missing_go_to_left field."""
        try:
            from sklearn.tree import _tree
            import numpy as np
            
            # Check if already patched - if so, just ensure it's still the old dtype
            if hasattr(_tree, '_patched_for_old_pickles'):
                # Already patched, ensure dtype is correct
                if not hasattr(_tree, '_original_NODE_DTYPE'):
                    # Restore original if lost
                    _tree._original_NODE_DTYPE = np.dtype([
                        ('left_child', '<i8'),
                        ('right_child', '<i8'),
                        ('feature', '<i8'),
                        ('threshold', '<f8'),
                        ('impurity', '<f8'),
                        ('n_node_samples', '<i8'),
                        ('weighted_n_node_samples', '<f8'),
                        ('missing_go_to_left', 'u1')
                    ])
                
                # Re-apply old dtype in case it was restored
                old_node_dtype = np.dtype([
                    ('left_child', '<i8'),
                    ('right_child', '<i8'),
                    ('feature', '<i8'),
                    ('threshold', '<f8'),
                    ('impurity', '<f8'),
                    ('n_node_samples', '<i8'),
                    ('weighted_n_node_samples', '<f8')
                ])
                _tree.NODE_DTYPE = old_node_dtype
                logger.debug("Re-applied sklearn tree dtype patch")
                return
            
            logger.info("Patching sklearn.tree._tree for old pickle compatibility")
            
            # Save original NODE_DTYPE
            _tree._original_NODE_DTYPE = _tree.NODE_DTYPE
            
            # Create a version without missing_go_to_left for backward compatibility
            # This matches the old format that py-feat models were saved with
            old_node_dtype = np.dtype([
                ('left_child', '<i8'),
                ('right_child', '<i8'),
                ('feature', '<i8'),
                ('threshold', '<f8'),
                ('impurity', '<f8'),
                ('n_node_samples', '<i8'),
                ('weighted_n_node_samples', '<f8')
            ])
            
            # Patch the module's dtype temporarily
            _tree.NODE_DTYPE = old_node_dtype
            
            # Mark as patched
            _tree._patched_for_old_pickles = True
            
            logger.info("sklearn tree dtype patched successfully")
            
        except Exception as e:
            logger.warning(f"Could not patch sklearn tree compatibility: {e}")
    
    @staticmethod
    def _restore_sklearn_dtype():
        """Restore original sklearn tree dtype after Py-Feat initialization."""
        try:
            from sklearn.tree import _tree
            
            if hasattr(_tree, '_original_NODE_DTYPE'):
                logger.info("Restoring original sklearn tree dtype")
                _tree.NODE_DTYPE = _tree._original_NODE_DTYPE
                delattr(_tree, '_original_NODE_DTYPE')
                if hasattr(_tree, '_patched_for_old_pickles'):
                    delattr(_tree, '_patched_for_old_pickles')
                    
        except Exception as e:
            logger.warning(f"Could not restore sklearn tree dtype: {e}")

    @staticmethod
    def _ensure_lib2to3() -> None:
        """Ensure lib2to3 (removed in Python 3.13) is available for Py-Feat imports."""
        spec = importlib.util.find_spec("lib2to3")
        if spec is not None:
            return

        from types import ModuleType

        shim = ModuleType("lib2to3")
        shim.__path__ = []  # type: ignore[attr-defined]

        pytree = ModuleType("lib2to3.pytree")

        def _convert(node, results=None):  # pragma: no cover - compatibility shim
            return node

        pytree.convert = _convert  # type: ignore[attr-defined]

        shim.pytree = pytree  # type: ignore[attr-defined]

        sys.modules.setdefault("lib2to3", shim)
        sys.modules.setdefault("lib2to3.pytree", pytree)

    def _detect_features(self, video_path: str) -> Dict[str, Any]:
        """Run Py-Feat detection with the necessary compatibility patches."""
        # Re-apply sklearn patch before video detection (models may load lazily)
        self._patch_sklearn_compat()
        try:
            result = self.detector.detect_video(video_path)
        finally:
            # Always restore regardless of success/failure
            self._restore_sklearn_dtype()

        df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        features = self._postprocess_df(df)
        return self._ensure_required_features(features)

    def _sanitize_video(self, video_path: str) -> Tuple[Optional[str], Optional[Path]]:
        """Create a sanitized temporary copy of the video tolerant to decode errors."""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            logger.warning("FFmpeg not found; cannot sanitize video for Py-Feat.")
            return None, None

        source_path = Path(video_path)
        if not source_path.exists():
            logger.warning("Video path does not exist: %s", video_path)
            return None, None

        temp_dir = Path(tempfile.mkdtemp(prefix="pyfeat_sanitize_"))
        sanitized_path = temp_dir / source_path.name

        cmd = [
            ffmpeg_path,
            "-y",
            "-err_detect",
            "ignore_err",
            "-i",
            str(source_path),
            "-c",
            "copy",
            str(sanitized_path),
        ]

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.stderr:
                logger.debug("FFmpeg sanitize stderr: %s", completed.stderr.decode(errors="ignore"))
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "FFmpeg failed to sanitize video for Py-Feat: %s",
                exc.stderr.decode(errors="ignore") if exc.stderr else exc,
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None
        except Exception as exc:
            logger.warning("Unexpected error while sanitizing video for Py-Feat: %s", exc)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None

        logger.info("Created sanitized video copy for Py-Feat at %s", sanitized_path)
        return str(sanitized_path), temp_dir
