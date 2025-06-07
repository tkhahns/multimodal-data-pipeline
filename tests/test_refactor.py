#!/usr/bin/env python3
"""
Test script to verify the refactored pipeline behavior.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import MultimodalPipeline

def test_default_behavior():
    """Test that default behavior extracts all features."""
    print("Testing default behavior (no features specified)...")
    
    # Create pipeline with no features specified (should default to all)
    pipeline = MultimodalPipeline(output_dir="test_output", features=None)
    
    expected_features = [
        "basic_audio",
        "librosa_spectral", 
        "opensmile",
        "speech_emotion",
        "speech_separation",
        "whisperx_transcription"
    ]
    
    print(f"Expected features: {expected_features}")
    print(f"Pipeline features: {pipeline.features}")
    
    # Check that all expected features are in the pipeline
    if set(pipeline.features) == set(expected_features):
        print("✓ Default behavior works correctly - all features are extracted")
        return True
    else:
        print("✗ Default behavior failed")
        missing = set(expected_features) - set(pipeline.features)
        extra = set(pipeline.features) - set(expected_features)
        if missing:
            print(f"  Missing features: {missing}")
        if extra:
            print(f"  Extra features: {extra}")
        return False

def test_specific_features():
    """Test that specific features can still be selected."""
    print("\nTesting specific feature selection...")
    
    # Create pipeline with specific features
    selected_features = ["basic_audio", "librosa_spectral"]
    pipeline = MultimodalPipeline(output_dir="test_output", features=selected_features)
    
    print(f"Selected features: {selected_features}")
    print(f"Pipeline features: {pipeline.features}")
    
    if pipeline.features == selected_features:
        print("✓ Specific feature selection works correctly")
        return True
    else:
        print("✗ Specific feature selection failed")
        return False

def test_no_comprehensive_feature():
    """Test that 'comprehensive' is no longer available as a feature."""
    print("\nTesting that 'comprehensive' feature is no longer available...")
    
    try:
        # Try to create pipeline with comprehensive feature
        pipeline = MultimodalPipeline(output_dir="test_output", features=["comprehensive"])
        
        # Try to get the comprehensive extractor (should fail)
        extractor = pipeline._get_extractor("comprehensive")
        if extractor is None:
            print("✓ 'comprehensive' feature correctly removed - extractor returns None")
            return True
        else:
            print("✗ 'comprehensive' feature still available as extractor")
            return False
            
    except Exception as e:
        print(f"✓ 'comprehensive' feature correctly removed - raises exception: {e}")
        return True

if __name__ == "__main__":
    print("Testing refactored pipeline behavior...\n")
    
    results = []
    results.append(test_default_behavior())
    results.append(test_specific_features())
    results.append(test_no_comprehensive_feature())
    
    print(f"\nTest Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Refactoring successful.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
