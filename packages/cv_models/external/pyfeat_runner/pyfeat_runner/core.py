from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from .compat import ensure_legacy_stats

logger = logging.getLogger(__name__)


def _postprocess_df(df) -> Dict[str, Any]:
    features: Dict[str, Any] = {}

    au_map = {
        "pf_au01": "AU01_r",
        "pf_au02": "AU02_r",
        "pf_au04": "AU04_r",
        "pf_au05": "AU05_r",
        "pf_au06": "AU06_r",
        "pf_au07": "AU07_r",
        "pf_au09": "AU09_r",
        "pf_au10": "AU10_r",
        "pf_au11": "AU11_r",
        "pf_au12": "AU12_r",
        "pf_au14": "AU14_r",
        "pf_au15": "AU15_r",
        "pf_au17": "AU17_r",
        "pf_au20": "AU20_r",
        "pf_au23": "AU23_r",
        "pf_au24": "AU24_r",
        "pf_au25": "AU25_r",
        "pf_au26": "AU26_r",
        "pf_au28": "AU28_r",
        "pf_au43": "AU43_r",
    }
    columns = getattr(df, "columns", [])
    for out_key, col in au_map.items():
        if col in columns:
            values = df[col].values
            if len(values):
                features[out_key] = float(np.nanmean(values))

    emo_cols = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    for emo in emo_cols:
        if emo in columns:
            values = df[emo].values
            if len(values):
                features[f"pf_{emo}"] = float(np.nanmean(values))

    for col_name, out_prefix in [
        ("face_x", "pf_facerectx"),
        ("face_y", "pf_facerecty"),
        ("face_w", "pf_facerectwidth"),
        ("face_h", "pf_facerectheight"),
        ("face_score", "pf_facescore"),
    ]:
        if col_name in columns:
            values = df[col_name].values
            if len(values):
                features[out_prefix] = float(np.nanmean(values))

    for ang in ["pitch", "roll", "yaw"]:
        if ang in columns:
            values = df[ang].values
            if len(values):
                features[f"pf_{ang}"] = float(np.nanmean(values))

    for axis in ["x", "y", "z"]:
        coln = f"landmark_{axis}"
        if coln in columns:
            values = df[coln].values
            if len(values):
                features[f"pf_{axis}"] = float(np.nanmean(values))

    if not features:
        features["pf_warning"] = "No Py-Feat outputs were produced."
    return features


def generate_features(video_path: str) -> Dict[str, Any]:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    ensure_legacy_stats()
    try:
        from feat import Detector  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("py-feat is not installed in the runner environment") from exc

    detector = Detector()
    result = detector.detect_video(str(video))
    df = result.to_pandas() if hasattr(result, "to_pandas") else result
    features = _postprocess_df(df)
    payload: Dict[str, Any] = {
        "model": "py-feat",
        "video": str(video.resolve()),
        "features": features,
    }
    return payload


def run_cli(video_path: str) -> int:
    try:
        payload = generate_features(video_path)
        print(json.dumps(payload))
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.exception("Py-Feat runner failed")
        error_payload = {"error": str(exc)}
        print(json.dumps(error_payload))
        return 1
