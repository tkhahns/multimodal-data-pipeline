"""
Utilities module.
"""

# Import main functions and classes for easier access
try:
    from .audio_extraction import extract_audio_from_video, extract_audio_from_videos
    from .file_utils import ensure_dir, clean_dir, save_json, load_json, find_files
except ImportError as e:
    import warnings
    warnings.warn(f"Some utility modules couldn't be imported: {e}")

__all__ = [
    'extract_audio_from_video',
    'extract_audio_from_videos',
    'ensure_dir',
    'clean_dir',
    'save_json',
    'load_json',
    'find_files'
]
