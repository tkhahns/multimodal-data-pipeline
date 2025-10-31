"""Utility analyzer that extracts representative frames from a video."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2

logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract evenly spaced frames and persist them on disk."""

    def __init__(self, *, output_dir: Optional[Path] = None, frame_count: int = 16) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir = self.output_dir / "vision" / "frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = frame_count

    def get_feature_dict(self, video_path: str) -> Dict[str, any]:
        video_file = Path(video_path)
        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            logger.warning("Could not open video for frame extraction: %s", video_path)
            return {"frame_paths": [], "frame_count": 0}

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.frame_count
        step = max(total_frames // max(self.frame_count, 1), 1)

        saved_paths = []
        for idx in range(0, total_frames, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            frame_path = self.output_dir / f"{video_file.stem}_frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(str(frame_path))
            if len(saved_paths) >= self.frame_count:
                break

        capture.release()

        return {
            "VFE_frame_paths": saved_paths,
            "VFE_frame_count": len(saved_paths),
            "VFE_video_path": str(video_file),
        }


__all__ = ["VideoFrameExtractor"]
