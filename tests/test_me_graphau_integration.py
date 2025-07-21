"""
Integration test for ME-GraphAU facial action unit recognition.

This script tests the integration of ME-GraphAU into the multimodal pipeline.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_me_graphau_integration():
    """Test ME-GraphAU integration with the pipeline."""
    try:
        print("Testing ME-GraphAU integration...")
        
        # Test importing the analyzer
        from vision.me_graphau_analyzer import MEGraphAUAnalyzer, extract_me_graphau_features
        print("✓ ME-GraphAU analyzer import successful")
        
        # Test importing from vision module
        from vision import MEGraphAUAnalyzer as MEGraphAUFromModule
        print("✓ ME-GraphAU import from vision module successful")
        
        # Test pipeline import
        from pipeline import MultimodalPipeline
        print("✓ Pipeline import successful")
        
        # Test pipeline initialization with ME-GraphAU feature
        pipeline = MultimodalPipeline(
            features=["me_graphau_vision"],
            device="cpu"
        )
        print("✓ Pipeline initialization with me_graphau_vision successful")
        
        # Test extractor initialization
        extractor = pipeline._get_extractor("me_graphau_vision")
        print("✓ ME-GraphAU extractor initialization successful")
        
        # Test analyzer instantiation
        analyzer = MEGraphAUAnalyzer(device="cpu")
        print("✓ ME-GraphAU analyzer instantiation successful")
        
        # Check default metrics
        defaults = analyzer.default_metrics
        expected_features = [
            'ann_AU1_bp4d', 'ann_AU2_bp4d', 'ann_AU4_bp4d', 'ann_AU6_bp4d',
            'ann_AU7_bp4d', 'ann_AU10_bp4d', 'ann_AU12_bp4d', 'ann_AU14_bp4d',
            'ann_AU15_bp4d', 'ann_AU17_bp4d', 'ann_AU23_bp4d', 'ann_AU24_bp4d',
            'ann_avg_bp4d', 'ann_AU1_dis', 'ann_AU2_dis', 'ann_AU4_dis',
            'ann_AU6_dis', 'ann_AU9_dis', 'ann_AU12_dis', 'ann_AU25_dis',
            'ann_AU26_dis', 'ann_avg_dis'
        ]
        
        for feature in expected_features:
            if feature not in defaults:
                raise ValueError(f"Missing expected feature: {feature}")
        
        print(f"✓ All {len(expected_features)} expected ME-GraphAU features present")
        
        print("\\n🎉 All ME-GraphAU integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_me_graphau_integration()
    sys.exit(0 if success else 1)
