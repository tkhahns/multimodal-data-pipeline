#!/usr/bin/env python3
"""
Integration test for DAN (Distract Your Attention) emotional expression recognition.

This script tests the basic functionality of the DAN analyzer and its integration
with the multimodal pipeline.
"""

import sys
import traceback
from pathlib import Path
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

def test_dan_import():
    """Test DAN analyzer import."""
    print("Testing DAN analyzer import...")
    try:
        from src.vision.dan_analyzer import DANAnalyzer, extract_dan_features
        print("✅ DAN analyzer imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import DAN analyzer: {e}")
        traceback.print_exc()
        return False

def test_dan_initialization():
    """Test DAN analyzer initialization."""
    print("Testing DAN analyzer initialization...")
    try:
        from src.vision.dan_analyzer import DANAnalyzer
        
        # Test 7-class model
        analyzer_7 = DANAnalyzer(device="cpu", num_classes=7)
        print("✅ DAN analyzer (7-class) initialized successfully")
        
        # Test 8-class model
        analyzer_8 = DANAnalyzer(device="cpu", num_classes=8)
        print("✅ DAN analyzer (8-class) initialized successfully")
        
        # Check emotion labels
        print(f"7-class emotions: {analyzer_7.emotion_labels}")
        print(f"8-class emotions: {analyzer_8.emotion_labels}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize DAN analyzer: {e}")
        traceback.print_exc()
        return False

def test_dan_analyze_frame():
    """Test DAN frame analysis."""
    print("Testing DAN frame analysis...")
    try:
        from src.vision.dan_analyzer import DANAnalyzer
        
        analyzer = DANAnalyzer(device="cpu", num_classes=7)
        
        # Create a dummy frame
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Analyze the frame
        features = analyzer.analyze_frame(dummy_frame)
        
        # Check expected features
        expected_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        for emotion in expected_emotions:
            key = f'dan_{emotion}'
            if key not in features:
                print(f"❌ Missing expected feature: {key}")
                return False
            if not isinstance(features[key], (int, float)):
                print(f"❌ Invalid feature type for {key}: {type(features[key])}")
                return False
        
        # Check emotion scores array
        if 'dan_emotion_scores' not in features:
            print("❌ Missing dan_emotion_scores")
            return False
        
        emotion_scores = features['dan_emotion_scores']
        if not isinstance(emotion_scores, list) or len(emotion_scores) != 7:
            print(f"❌ Invalid dan_emotion_scores: {emotion_scores}")
            return False
        
        print("✅ DAN frame analysis completed successfully")
        print(f"Sample features: {dict(list(features.items())[:3])}")
        return True
        
    except Exception as e:
        print(f"❌ Failed DAN frame analysis: {e}")
        traceback.print_exc()
        return False

def test_dan_get_feature_dict():
    """Test DAN get_feature_dict method."""
    print("Testing DAN get_feature_dict method...")
    try:
        from src.vision.dan_analyzer import DANAnalyzer
        
        analyzer = DANAnalyzer(device="cpu", num_classes=7)
        
        # Test with a non-existent video (should return default metrics)
        features = analyzer.get_feature_dict("non_existent_video.mp4")
        
        # Should return default metrics without crashing
        if not features:
            print("❌ get_feature_dict returned empty features")
            return False
        
        # Check that all expected keys are present
        expected_keys = ['dan_angry', 'dan_disgust', 'dan_fear', 'dan_happy', 
                        'dan_neutral', 'dan_sad', 'dan_surprise', 'dan_emotion_scores']
        
        for key in expected_keys:
            if key not in features:
                print(f"❌ Missing expected key in get_feature_dict: {key}")
                return False
        
        print("✅ DAN get_feature_dict method works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Failed DAN get_feature_dict test: {e}")
        traceback.print_exc()
        return False

def test_dan_extract_function():
    """Test DAN extract function."""
    print("Testing DAN extract function...")
    try:
        from src.vision.dan_analyzer import extract_dan_features
        
        # Test with non-existent video (should return default metrics)
        features = extract_dan_features("non_existent_video.mp4", device="cpu", num_classes=7)
        
        if not features:
            print("❌ extract_dan_features returned empty features")
            return False
        
        # Check basic structure
        if 'dan_emotion_scores' not in features:
            print("❌ extract_dan_features missing dan_emotion_scores")
            return False
        
        print("✅ DAN extract function works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Failed DAN extract function test: {e}")
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test DAN integration with the pipeline."""
    print("Testing DAN pipeline integration...")
    try:
        from src.pipeline import MultimodalPipeline
        
        # Initialize pipeline with DAN feature
        pipeline = MultimodalPipeline(
            features=["dan_vision"],
            output_dir="test_output",
            device="cpu"
        )
        
        # Check that DAN extractor was created
        extractor = pipeline._get_extractor("dan_vision")
        if extractor is None:
            print("❌ DAN extractor not created in pipeline")
            return False
        
        print("✅ DAN pipeline integration works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Failed DAN pipeline integration test: {e}")
        traceback.print_exc()
        return False

def test_vision_module_exports():
    """Test that DAN is properly exported from vision module."""
    print("Testing vision module exports...")
    try:
        from src.vision import DANAnalyzer, extract_dan_features
        print("✅ DAN components properly exported from vision module")
        return True
    except Exception as e:
        print(f"❌ Failed to import DAN from vision module: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all DAN integration tests."""
    print("=" * 70)
    print("DAN (Distract Your Attention) Integration Tests")
    print("=" * 70)
    
    tests = [
        test_dan_import,
        test_vision_module_exports,
        test_dan_initialization,
        test_dan_analyze_frame,
        test_dan_get_feature_dict,
        test_dan_extract_function,
        test_pipeline_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("DAN Integration Test Results")
    print("=" * 70)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed! DAN integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
