#!/usr/bin/env python3
"""
Test script for OpenSMILE feature extraction.
"""
import sys
from pathlib import Path
from src.audio.opensmile_features import OpenSMILEFeatureExtractor

def test_opensmile_features(audio_path: str):
    """Test OpenSMILE feature extraction."""
    print(f"Testing OpenSMILE feature extraction on: {audio_path}")
    
    # Initialize extractor
    extractor = OpenSMILEFeatureExtractor()
    
    # Extract features
    features = extractor.get_feature_dict(audio_path)
    
    print(f"\nExtracted {len(features)} features:")
    
    # Group features by type
    lld_features = {k: v for k, v in features.items() if k.startswith('osm_') and '_mean' not in k and '_std' not in k}
    functional_features = {k: v for k, v in features.items() if '_mean' in k or '_std' in k or '_percentile' in k}
    other_features = {k: v for k, v in features.items() if not k.startswith('osm_')}
    
    print(f"\nLow-Level Descriptors (LLD) features: {len(lld_features)}")
    for name, values in list(lld_features.items())[:5]:  # Show first 5
        if hasattr(values, '__len__') and len(values) > 0:
            print(f"  {name}: {len(values)} frames (mean: {sum(values)/len(values):.4f})")
        else:
            print(f"  {name}: {values}")
    if len(lld_features) > 5:
        print(f"  ... and {len(lld_features) - 5} more LLD features")
    
    print(f"\nFunctional features: {len(functional_features)}")
    for name, value in list(functional_features.items())[:5]:  # Show first 5
        print(f"  {name}: {value:.4f}")
    if len(functional_features) > 5:
        print(f"  ... and {len(functional_features) - 5} more functional features")
    
    if other_features:
        print(f"\nOther features: {len(other_features)}")
        for name, value in other_features.items():
            print(f"  {name}: {value}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if Path(audio_path).exists():
            test_opensmile_features(audio_path)
        else:
            print(f"Audio file not found: {audio_path}")
    else:
        # Test with default audio file
        default_audio = "output/opensmile_test/audio/MVI_0574.wav"
        if Path(default_audio).exists():
            test_opensmile_features(default_audio)
        else:
            print("Usage: python test_opensmile.py <audio_file>")
            print("No default audio file found.")
