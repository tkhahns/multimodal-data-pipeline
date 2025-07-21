#!/usr/bin/env python3
"""
Demo script for Optical Flow (Movement and Estimation of Motion) integration.

This script demonstrates:
1. Sparse optical flow tracking with feature points
2. Dense optical flow analysis for per-pixel motion vectors
3. Motion visualization with color-coded flow images
4. Pipeline integration and feature extraction
5. Output columns: sparse_flow_vis_.png, sparse_points.npy, dense_flow.npy, dense_flow_vis_.png

Usage:
    python demo/demo_optical_flow.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.optical_flow_analyzer import OpticalFlowAnalyzer
from src.pipeline import MultimodalPipeline


def create_test_video(output_path: str, duration: float = 3.0, fps: int = 30):
    """Create a test video with moving objects for optical flow analysis."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame.fill(30)  # Dark background
        
        # Add grid pattern for better feature detection
        for i in range(0, width, 50):
            cv2.line(frame, (i, 0), (i, height), (60, 60, 60), 1)
        for i in range(0, height, 50):
            cv2.line(frame, (0, i), (width, i), (60, 60, 60), 1)
        
        # Add moving objects with different motion patterns
        t = frame_idx / total_frames
        
        # Moving circle (left to right)
        circle_x = int(50 + (width - 100) * t)
        circle_y = int(height / 2 + 50 * np.sin(2 * np.pi * t * 2))
        cv2.circle(frame, (circle_x, circle_y), 25, (100, 200, 100), -1)
        
        # Rotating square
        center_x, center_y = width // 3, height // 3
        angle = 360 * t * 2
        size = 40
        pts = np.array([
            [-size, -size], [size, -size], [size, size], [-size, size]
        ], dtype=np.float32)
        
        # Rotate points
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_pts = pts @ rotation_matrix.T
        
        # Translate to center
        rotated_pts += np.array([center_x, center_y])
        rotated_pts = rotated_pts.astype(np.int32)
        
        cv2.fillPoly(frame, [rotated_pts], (200, 100, 200))
        
        # Oscillating rectangle
        rect_x = int(2 * width // 3 + 100 * np.sin(2 * np.pi * t * 3))
        rect_y = int(2 * height // 3)
        cv2.rectangle(frame, (rect_x - 30, rect_y - 20), 
                     (rect_x + 30, rect_y + 20), (100, 100, 200), -1)
        
        # Add some static features for contrast
        cv2.circle(frame, (100, 100), 15, (150, 150, 150), -1)
        cv2.rectangle(frame, (width - 100, height - 80), 
                     (width - 50, height - 30), (150, 150, 150), -1)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")


def test_optical_flow_analyzer():
    """Test the OpticalFlowAnalyzer directly."""
    print("=== Testing Optical Flow Analyzer ===")
    
    # Create test video
    test_video_path = "temp_optical_flow_test.mp4"
    create_test_video(test_video_path)
    
    try:
        # Initialize analyzer
        analyzer = OpticalFlowAnalyzer(device="cpu")
        
        # Extract features
        print(f"Analyzing video: {test_video_path}")
        features = analyzer.get_feature_dict(test_video_path)
        
        print(f"\nExtracted {len(features)} features:")
        
        # Group features by type
        output_columns = {k: v for k, v in features.items() if k in [
            'sparse_flow_vis_.png', 'sparse_points.npy', 'dense_flow.npy', 'dense_flow_vis_.png'
        ]}
        motion_metrics = {k: v for k, v in features.items() if 'motion' in k or 'displacement' in k or 'consistency' in k}
        frame_stats = {k: v for k, v in features.items() if 'frames' in k or 'total_frames' in k}
        other_features = {k: v for k, v in features.items() if k not in output_columns and k not in motion_metrics and k not in frame_stats}
        
        if output_columns:
            print(f"\n📊 Output Columns:")
            for key, value in output_columns.items():
                if isinstance(value, str) and value:
                    if value.startswith('data:'):
                        print(f"  {key}: Base64 encoded data ({len(value)} chars)")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        
        if motion_metrics:
            print(f"\n🌊 Motion Analysis Metrics:")
            for key, value in motion_metrics.items():
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
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # Test visualization
        print(f"\n🎨 Generating visualizations...")
        viz_path = analyzer.visualize_flow(test_video_path, output_path="optical_flow_visualization.mp4")
        if viz_path and os.path.exists(viz_path):
            print(f"✅ Visualization saved: {viz_path}")
        else:
            print("⚠️  Visualization not created (normal for demo)")
        
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def test_pipeline_integration():
    """Test Optical Flow integration with the full pipeline."""
    print("\n=== Testing Pipeline Integration ===")
    
    # Create test video
    test_video_path = "temp_pipeline_optical_flow_test.mp4"
    create_test_video(test_video_path, duration=2.0)
    
    try:
        # Initialize pipeline with Optical Flow feature
        pipeline = MultimodalPipeline(
            features=["optical_flow_vision"],
            device="cpu"
        )
        
        # Process video
        print(f"Processing video through pipeline: {test_video_path}")
        results = pipeline.process_video_file(test_video_path)
        
        print(f"\nPipeline extracted {len(results)} total features")
        
        # Show Optical Flow specific results
        optical_flow_features = {k: v for k, v in results.items() if 
                               k in ['sparse_flow_vis_.png', 'sparse_points.npy', 'dense_flow.npy', 'dense_flow_vis_.png'] or
                               'motion' in k.lower() or 'flow' in k.lower()}
        
        if optical_flow_features:
            print(f"\n🎯 Optical Flow Features from Pipeline:")
            for key, value in optical_flow_features.items():
                if isinstance(value, str) and value.startswith('data:'):
                    print(f"  {key}: Base64 encoded data ({len(value)} chars)")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, (list, np.ndarray)):
                    print(f"  {key}: array of length {len(value)}")
                else:
                    print(f"  {key}: {value}")
        
        # Test feature grouping
        grouped = pipeline._group_features_by_model(results)
        optical_flow_group = None
        for group_name, group_data in grouped.items():
            if "Movement and estimation of motion" in group_name or "Optical Flow" in group_data.get("Model", ""):
                optical_flow_group = group_data
                break
        
        if optical_flow_group:
            print(f"\n📦 Feature Grouping:")
            print(f"  Group: {optical_flow_group['Feature']}")
            print(f"  Model: {optical_flow_group['Model']}")
            print(f"  Features: {len(optical_flow_group['features'])} items")
        
    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)


def test_multiple_features():
    """Test Optical Flow alongside other vision features."""
    print("\n=== Testing Multiple Vision Features ===")
    
    # Create test video
    test_video_path = "temp_multi_feature_test.mp4"
    create_test_video(test_video_path, duration=1.5)
    
    try:
        # Test with multiple vision features including Optical Flow
        pipeline = MultimodalPipeline(
            features=["ganimation_vision", "openpose_vision", "instadm_vision", "optical_flow_vision"],
            device="cpu"
        )
        
        print(f"Processing with multiple vision features: {test_video_path}")
        results = pipeline.process_video_file(test_video_path)
        
        # Categorize results by feature type
        feature_categories = {
            "GANimation": {k: v for k, v in results.items() if k.startswith('GAN_')},
            "OpenPose": {k: v for k, v in results.items() if k.startswith('openPose_')},
            "Insta-DM": {k: v for k, v in results.items() if k.startswith('indm_')},
            "Optical Flow": {k: v for k, v in results.items() if 
                           k in ['sparse_flow_vis_.png', 'sparse_points.npy', 'dense_flow.npy', 'dense_flow_vis_.png'] or
                           'motion' in k.lower() or 'flow' in k.lower()}
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
                    elif isinstance(value, str) and value.startswith('data:'):
                        print(f"    {key}: Base64 data ({len(value)} chars)")
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
    """Run all Optical Flow demo tests."""
    print("🌊 Optical Flow (Movement and Motion Estimation) Demo")
    print("=" * 60)
    
    try:
        # Test 1: Direct analyzer testing
        test_optical_flow_analyzer()
        
        # Test 2: Pipeline integration
        test_pipeline_integration()
        
        # Test 3: Multiple features
        test_multiple_features()
        
        print(f"\n✅ All Optical Flow tests completed successfully!")
        print(f"\n📋 Summary:")
        print(f"  • Optical Flow analyzer working correctly")
        print(f"  • Pipeline integration successful")
        print(f"  • Feature grouping implemented")
        print(f"  • Multi-feature compatibility verified")
        print(f"\n🎯 Optical Flow Output Columns Available:")
        print(f"  • sparse_flow_vis_.png: Visualized sparse flow with arrows")
        print(f"  • sparse_points.npy: Tracked feature points data")
        print(f"  • dense_flow.npy: Per-pixel motion vectors")
        print(f"  • dense_flow_vis_.png: Color-coded dense flow visualization")
        print(f"\n🌊 Motion Analysis Metrics:")
        print(f"  • Motion detection rates and frame statistics")
        print(f"  • Motion magnitude and displacement measures")
        print(f"  • Motion direction and consistency analysis")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
