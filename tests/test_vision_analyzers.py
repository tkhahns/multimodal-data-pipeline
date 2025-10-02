import math
from pathlib import Path

import cv2  # type: ignore[import]
import numpy as np

from cv_models.vision.vitpose_analyzer import ViTPoseAnalyzer
from cv_models.vision.psa_analyzer import PSAAnalyzer
from cv_models.vision.lanegcn_analyzer import LaneGCNAnalyzer
from cv_models.vision.smoothnet_analyzer import SmoothNetAnalyzer
from cv_models.vision.crowdflow_analyzer import CrowdFlowAnalyzer


def _create_synthetic_video(output_path: Path, frame_count: int = 40) -> str:
    width, height = 256, 256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))

    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = 40 + (idx * 4) % 160
        y = 80 + int(30 * np.sin(idx / 5))
        cv2.circle(frame, (x, y), 15, (255, 255, 255), -1)
        cv2.rectangle(frame, (60, 180), (60 + idx % 80, 200), (0, 255, 0), -1)
        writer.write(frame)

    writer.release()
    return str(output_path)


def _assert_numeric_features(feature_dict: dict[str, float], keys: list[str]) -> None:
    for key in keys:
        assert key in feature_dict, f"Expected feature '{key}' in feature map"
        value = feature_dict[key]
        assert value == value, f"Feature '{key}' must not be NaN"
        assert math.isfinite(value), f"Feature '{key}' must be finite"


def test_vision_analyzers_smoke(tmp_path: Path) -> None:
    video_path = _create_synthetic_video(tmp_path / "synthetic_motion.mp4")

    vitpose = ViTPoseAnalyzer()
    vit_features = vitpose.get_feature_dict(video_path)["ViTPose"]["features"]
    _assert_numeric_features(vit_features, ["vit_AR", "vit_AP", "vit_AU", "vit_mean"])

    psa = PSAAnalyzer()
    psa_features = psa.get_feature_dict(video_path)["PSA"]["features"]
    _assert_numeric_features(psa_features, ["psa_AP", "psa_val_mloU"])

    lanegcn = LaneGCNAnalyzer()
    lane_features = lanegcn.get_feature_dict(video_path)["LaneGCN"]["features"]
    _assert_numeric_features(
        lane_features,
        [
            "GCN_min_ade_k1",
            "GCN_min_fde_k1",
            "GCN_MR_k1",
            "GCN_min_ade_k6",
            "GCN_min_fde_k6",
            "GCN_MR_k6",
        ],
    )

    smoothnet = SmoothNetAnalyzer()
    smooth_features = smoothnet.get_feature_dict(video_path)["SmoothNet"]["features"]
    _assert_numeric_features(
        smooth_features,
        [
            "net_3d_estimator",
            "net_2d_estimator",
            "net_SMPL_estimator",
            "net_joint_confidence",
            "net_pose_stability",
            "net_motion_coherence",
        ],
    )

    crowdflow = CrowdFlowAnalyzer()
    crowd_features = crowdflow.get_feature_dict(video_path)["CrowdFlow"]["features"]
    _assert_numeric_features(
        crowd_features,
        [
            "of_fg_static_epe_st",
            "of_fg_dynamic_epe_st",
            "of_bg_static_epe_st",
            "of_avg_epe_st",
            "of_ta_IM01",
            "of_pt_IM01",
        ],
    )
