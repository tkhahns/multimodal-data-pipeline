#!/usr/bin/env python3
"""
Demo script for Google MediaPipe pose estimation and tracking.

This script demonstrates how to use the MediaPipePoseAnalyzer to extract
pose landmarks from video files, providing 33 landmarks with both normalized
and world coordinates.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.mediapipe_pose_analyzer import MediaPipePoseAnalyzer
import json

def demo_mediapipe_pose_analysis():
    """Demonstrate MediaPipe pose analysis functionality."""
    
    print("=" * 60)
    print("Google MediaPipe Pose Estimation and Tracking Demo")
    print("=" * 60)
    
    # Initialize the analyzer
    print("\n1. Initializing MediaPipe Pose analyzer...")
    analyzer = MediaPipePoseAnalyzer(
        device='cpu',  # MediaPipe runs on CPU
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe Pose analyzer initialized successfully")
    
    # Check for demo video files
    demo_video_paths = [
        "data/demo_video.mp4",
        "data/sample.mp4", 
        "data/test_video.mp4",
        "../data/demo_video.mp4",
        "../data/sample.mp4"
    ]
    
    video_path = None
    for path in demo_video_paths:
        if Path(path).exists():
            video_path = path
            break
    
    if not video_path:
        print("\n⚠️  No demo video found in common locations.")
        print("Please place a video file in one of these locations:")
        for path in demo_video_paths:
            print(f"  - {path}")
        print("\nCreating a minimal demo with default values...")
        
        # Show default feature structure
        demo_features()
        return
    
    print(f"\n2. Found demo video: {video_path}")
    
    # Analyze the video
    print("\n3. Running MediaPipe pose analysis...")
    print("   This may take a moment depending on video length...")
    
    try:
        # Get features using the pipeline interface
        features = analyzer.get_feature_dict(video_path)
        
        print("✓ MediaPipe pose analysis completed successfully!")
        
        # Display results
        print("\n4. MediaPipe Pose Analysis Results:")
        print("-" * 40)
        
        pose_features = features.get("Pose estimation and tracking", {}).get("features", {})
        
        # Summary statistics
        total_frames = pose_features.get('total_frames', 0)
        detected_frames = pose_features.get('landmarks_detected_frames', 0)
        detection_rate = pose_features.get('detection_rate', 0.0)
        avg_landmarks = pose_features.get('avg_landmarks_per_frame', 0.0)
        
        print(f"Total frames processed: {total_frames}")
        print(f"Frames with pose detected: {detected_frames}")
        print(f"Detection rate: {detection_rate:.2%}")
        print(f"Average landmarks per frame: {avg_landmarks:.1f}")
        
        # Sample landmark values (first few landmarks)
        print(f"\nSample Normalized Landmarks (first 5):")
        for i in range(1, 6):
            x = pose_features.get(f'GMP_land_x_{i}', 0.0)
            y = pose_features.get(f'GMP_land_y_{i}', 0.0)
            z = pose_features.get(f'GMP_land_z_{i}', 0.0)
            visibility = pose_features.get(f'GMP_land_visi_{i}', 0.0)
            print(f"  Landmark {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}, visibility={visibility:.3f}")
        
        # World coordinates sample
        print(f"\nSample World Coordinates (first 5):")
        for i in range(1, 6):
            x = pose_features.get(f'GMP_world_x_{i}', 0.0)
            y = pose_features.get(f'GMP_world_y_{i}', 0.0)
            z = pose_features.get(f'GMP_world_z_{i}', 0.0)
            print(f"  World {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        
        # Check if pose visualization was generated
        pose_image = pose_features.get('GMP_SM_pic', '')
        if pose_image:
            print(f"\n✓ Pose visualization generated (base64 encoded image)")
            print(f"  Image size: {len(pose_image)} characters")
        else:
            print(f"\n⚠️  No pose visualization generated")
        
        # Feature count
        mediapipe_features = [k for k in pose_features.keys() if k.startswith('GMP_')]
        print(f"\nTotal MediaPipe features extracted: {len(mediapipe_features)}")
        
        # Save results to file
        output_file = "mediapipe_pose_demo_results.json"
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during MediaPipe analysis: {e}")
        print("\nThis might be due to:")
        print("  - Missing MediaPipe installation")
        print("  - Video format not supported")
        print("  - Insufficient system resources")
        
        # Show demo features anyway
        demo_features()

def demo_features():
    """Show the structure of MediaPipe pose features."""
    print("\n5. MediaPipe Pose Feature Structure:")
    print("-" * 40)
    
    print("Normalized Landmarks (33 landmarks):")
    print("  GMP_land_x_1 to GMP_land_x_33    - X coordinates (0.0 to 1.0)")
    print("  GMP_land_y_1 to GMP_land_y_33    - Y coordinates (0.0 to 1.0)")
    print("  GMP_land_z_1 to GMP_land_z_33    - Z coordinates (depth)")
    print("  GMP_land_visi_1 to GMP_land_visi_33 - Visibility scores (0.0 to 1.0)")
    print("  GMP_land_presence_1 to GMP_land_presence_33 - Presence scores")
    
    print("\nWorld Coordinates (33 landmarks):")
    print("  GMP_world_x_1 to GMP_world_x_33  - World X coordinates (meters)")
    print("  GMP_world_y_1 to GMP_world_y_33  - World Y coordinates (meters)")  
    print("  GMP_world_z_1 to GMP_world_z_33  - World Z coordinates (meters)")
    print("  GMP_world_visi_1 to GMP_world_visi_33 - Visibility scores")
    print("  GMP_world_presence_1 to GMP_world_presence_33 - Presence scores")
    
    print("\nVisualization:")
    print("  GMP_SM_pic                        - Base64 encoded pose visualization")
    
    print("\nStatistics:")
    print("  total_frames                      - Total frames processed")
    print("  landmarks_detected_frames         - Frames with pose detected")
    print("  detection_rate                    - Percentage of successful detections")
    print("  avg_landmarks_per_frame           - Average visible landmarks per frame")
    
    print(f"\nTotal features: ~330+ (33 landmarks × 10 attributes + visualization + stats)")

def show_landmark_mapping():
    """Show the mapping of MediaPipe landmark indices to body parts."""
    print("\n6. MediaPipe Pose Landmark Mapping:")
    print("-" * 40)
    
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    for i, name in enumerate(landmark_names, 1):
        print(f"  {i:2d}: {name}")

if __name__ == "__main__":
    print("Starting MediaPipe Pose Estimation Demo...")
    
    try:
        demo_mediapipe_pose_analysis()
        show_landmark_mapping()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
