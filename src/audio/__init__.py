"""
Audio processing modules.
"""

# Import main classes for easier access
try:
    from .basic_features import AudioFeatureExtractor
    from .spectral_features import LibrosaFeatureExtractor
except ImportError as e:
    import warnings
    warnings.warn(f"Some audio modules couldn't be imported: {e}")

__all__ = [
    'AudioFeatureExtractor',
    'LibrosaFeatureExtractor'
]
