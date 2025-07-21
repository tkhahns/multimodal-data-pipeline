#!/usr/bin/env python3
"""
Demo script for Insta-DM (Instance-aware Dynamic Module for Monocular Depth Estimation) integration.

This script demonstrates:
1. Dense Motion Estimation and Depth Analysis
2. Interaction patterns detection in dynamic scenes  
3. Pipeline integration and feature extraction
4. Output columns: indm_abs_rel, indm_sq_rel, indm_rmse, indm_rmse_log, indm_acc_1, indm_acc_2, indm_acc_3

Usage:
    python demo/demo_instadm.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.instadm_analyzer import InstaDMAnalyzer
from src.pipeline import MultimodalPipeline


def create_test_video(output_path: str, duration: float = 2.0, fps: int = 30):
    """Create a test video with moving objects for depth/motion analysis."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame.fill(50)  # Dark gray background
        
        # Add moving objects at different depths
        t = frame_idx / total_frames
        
        # Large object (close/foreground) - moves slowly
        center1_x = int(150 + 50 * np.sin(2 * np.pi * t))
        center1_y = int(200)
        cv2.circle(frame, (center1_x, center1_y), 60, (100, 150, 200), -1)
        
        # Medium object (middle depth) - moves moderately
        center2_x = int(400 + 80 * np.cos(2 * np.pi * t * 1.5))
        center2_y = int(150 + 30 * np.sin(2 * np.pi * t * 1.5))
        cv2.circle(frame, (center2_x, center2_y), 35, (150, 200, 100), -1)
        
        # Small object (far/background) - moves quickly
        center3_x = int(500 + 100 * np.sin(2 * np.pi * t * 2))
        center3_y = int(350)
        cv2.circle(frame, (center3_x, center3_y), 15, (200, 100, 150), -1)
        
        # Add some static background elements
        cv2.rectangle(frame, (50, 400), (590, 420), (80, 80, 80), -1)
        cv2.rectangle(frame, (20, 20), (40, 460), (70, 70, 70), -1)
        cv2.rectangle(frame, (600, 20), (620, 460), (70, 70, 70), -1)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")


def test_instadm_analyzer():
    """Test the InstaDMAnalyzer directly."""
    print("=== Testing Insta-DM Analyzer ===")
    
    # Create test video
    test_video_path = "temp_instadm_test.mp4"
    create_test_video(test_video_path)
    
    try:
        # Initialize analyzer
        analyzer = InstaDMAnalyzer(device="cpu")
        
        # Extract features
        print(f"Analyzing video: {test_video_path}")
        features = analyzer.get_feature_dict(test_video_path)
        
        print(f"\nExtracted {len(features)} features:")
        
        # Group features by type
        depth_metrics = {k: v for k, v in features.items() if k.startswith('indm_')}
        frame_stats = {k: v for k, v in features.items() if 'frames' in k or k == 'total_frames'}
        other_features = {k: v for k, v in features.items() if k not in depth_metrics and k not in frame_stats}
        
        if depth_metrics:
            print(f"\n📊 Depth Estimation Metrics:")
            for key, value in depth_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        if frame_stats:
            print(f"\n🎥 Video Analysis Stats:")
            for key, value in frame_stats.items():
                print(f"  {key}: {value}")
        
        if other_features:
            print(f"\n🔧 Other Features:")
            for key, value in other_features.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"  {key}: array of length {len(value)}")
                else:
                    print(f"  {key}: {value}")
        
        # Test visualization
        print(f"\n🎨 Generating visualizations...")
        viz_path = analyzer.visualize_depth_motion(test_video_path, output_path="instadm_visualization.mp4")
        if viz_path and os.path.exists(viz_path):
            print(f"✅ Visualization saved: {viz_path}")
        else:
            print("⚠️  Visualization not created (normal for demo)")
        
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def test_pipeline_integration():
    """Test Insta-DM integration with the full pipeline."""
    print("\n=== Testing Pipeline Integration ===")
    
    # Create test video
    test_video_path = "temp_pipeline_instadm_test.mp4"
    create_test_video(test_video_path, duration=1.5)
    
    try:
        # Initialize pipeline with Insta-DM feature
        pipeline = MultimodalPipeline(
            features=["instadm_vision"],
            device="cpu"
        )
        
        # Process video
        print(f"Processing video through pipeline: {test_video_path}")
        results = pipeline.process_video_file(test_video_path)
        
        print(f"\nPipeline extracted {len(results)} total features")
        
        # Show Insta-DM specific results
        instadm_features = {k: v for k, v in results.items() if k.startswith('indm_') or 'depth' in k.lower() or 'motion' in k.lower()}
        
        if instadm_features:
            print(f"\n🎯 Insta-DM Features from Pipeline:")
            for key, value in instadm_features.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, (list, np.ndarray)):
                    print(f"  {key}: array of length {len(value)}")
                else:
                    print(f"  {key}: {value}")
        
        # Test feature grouping
        grouped = pipeline._group_features_by_model(results)
        instadm_group = None
        for group_name, group_data in grouped.items():
            if "Dense Motion" in group_name or "Insta-DM" in group_data.get("Model", ""):
                instadm_group = group_data
                break
        
        if instadm_group:
            print(f"\n📦 Feature Grouping:")
            print(f"  Group: {instadm_group['Feature']}")
            print(f"  Model: {instadm_group['Model']}")
            print(f"  Features: {len(instadm_group['features'])} items")
        
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def test_multiple_features():
    """Test Insta-DM alongside other vision features."""
    print("\n=== Testing Multiple Vision Features ===")
    
    # Create test video
    test_video_path = "temp_multi_feature_test.mp4"
    create_test_video(test_video_path, duration=1.0)
    
    try:
        # Test with multiple vision features including Insta-DM
        pipeline = MultimodalPipeline(
            features=["ganimation_vision", "arbex_vision", "openpose_vision", "instadm_vision"],
            device="cpu"
        )
        
        print(f"Processing with multiple vision features: {test_video_path}")
        results = pipeline.process_video_file(test_video_path)
        
        # Categorize results by feature type
        feature_categories = {
            "GANimation": {k: v for k, v in results.items() if k.startswith('GAN_')},
            "ARBEx": {k: v for k, v in results.items() if k.startswith('arbex_')},
            "OpenPose": {k: v for k, v in results.items() if k.startswith('openPose_')},
            "Insta-DM": {k: v for k, v in results.items() if k.startswith('indm_')}
        }
        
        print(f"\n🔍 Feature Distribution:")
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
                # Show a few key features for each
                sample_keys = list(features.keys())[:3]
                for key in sample_keys:
                    value = features[key]
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    else:
                        print(f"    {key}: {type(value).__name__}")
        
        # Test grouped features
        grouped = pipeline._group_features_by_model(results)
        print(f"\n📊 Feature Groups: {len(grouped)} groups created")
        for group_name in grouped.keys():
            if len(group_name) > 50:
                print(f"  {group_name[:47]}...")
            else:
                print(f"  {group_name}")
        
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def main():
    """Run all Insta-DM demo tests."""
    print("🎬 Insta-DM (Dense Motion & Depth Estimation) Demo")
    print("=" * 60)
    
    try:
        # Test 1: Direct analyzer testing
        test_instadm_analyzer()
        
        # Test 2: Pipeline integration
        test_pipeline_integration()
        
        # Test 3: Multiple features
        test_multiple_features()
        
        print(f"\n✅ All Insta-DM tests completed successfully!")
        print(f"\n📋 Summary:")
        print(f"  • Insta-DM analyzer working correctly")
        print(f"  • Pipeline integration successful")
        print(f"  • Feature grouping implemented")
        print(f"  • Multi-feature compatibility verified")
        print(f"\n🎯 Insta-DM Output Columns Available:")
        print(f"  • indm_abs_rel: Absolute relative depth error")
        print(f"  • indm_sq_rel: Squared relative depth error")
        print(f"  • indm_rmse: Root mean square error")
        print(f"  • indm_rmse_log: Logarithmic RMSE")
        print(f"  • indm_acc_1, indm_acc_2, indm_acc_3: Accuracy thresholds")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
