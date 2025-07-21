#!/usr/bin/env python3
"""
Demo script for OpenPose real-time multi-person keypoint detection and pose estimation.
This demonstrates pose estimation and tracking with skeleton visualization.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from vision.openpose_analyzer import OpenPoseAnalyzer, extract_openpose_features

def main():
    print("OpenPose Real-time Multi-Person Pose Estimation Demo")
    print("=" * 55)
    
    # Initialize analyzer
    print("Initializing OpenPose analyzer...")
    analyzer = OpenPoseAnalyzer(device='cpu', confidence_threshold=0.1)
    
    # Demo with a test video (you can replace this with your own video)
    test_video = project_root / "test_data" / "sample_video.mp4"
    
    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        print("Please provide a video file to test with.")
        
        # Create a simple test video using OpenCV if available
        try:
            import cv2
            import numpy as np
            
            print("Creating a simple test video with moving people...")
            test_video.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simple video with moving human-like figures
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(test_video), fourcc, 30.0, (640, 480))
            
            for i in range(120):  # 4 seconds at 30 fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Create multiple human-like figures
                for person_id in range(2):
                    # Person positions
                    center_x = 160 + person_id * 320 + int(50 * np.sin(i * 0.1 + person_id * np.pi))
                    center_y = 240
                    
                    # Body scale
                    scale = 1.0 + 0.3 * np.sin(i * 0.05 + person_id * 0.5)
                    
                    # Head
                    head_y = int(center_y - 120 * scale)
                    cv2.circle(frame, (center_x, head_y), int(20 * scale), (255, 255, 255), 2)
                    
                    # Body lines (stick figure)
                    # Torso
                    torso_top = (center_x, int(center_y - 100 * scale))
                    torso_bottom = (center_x, int(center_y - 20 * scale))
                    cv2.line(frame, torso_top, torso_bottom, (255, 255, 255), 3)
                    
                    # Arms (moving)
                    arm_angle = np.sin(i * 0.2 + person_id) * 0.5
                    left_shoulder = (int(center_x - 30 * scale), int(center_y - 80 * scale))
                    right_shoulder = (int(center_x + 30 * scale), int(center_y - 80 * scale))
                    
                    left_elbow = (int(left_shoulder[0] - 40 * scale * np.cos(arm_angle)), 
                                 int(left_shoulder[1] + 40 * scale * np.sin(arm_angle)))
                    right_elbow = (int(right_shoulder[0] + 40 * scale * np.cos(arm_angle)), 
                                  int(right_shoulder[1] + 40 * scale * np.sin(arm_angle)))
                    
                    left_wrist = (int(left_elbow[0] - 30 * scale), int(left_elbow[1] + 30 * scale))
                    right_wrist = (int(right_elbow[0] + 30 * scale), int(right_elbow[1] + 30 * scale))
                    
                    # Draw arms
                    cv2.line(frame, torso_top, left_shoulder, (255, 255, 255), 2)
                    cv2.line(frame, torso_top, right_shoulder, (255, 255, 255), 2)
                    cv2.line(frame, left_shoulder, left_elbow, (255, 255, 255), 2)
                    cv2.line(frame, right_shoulder, right_elbow, (255, 255, 255), 2)
                    cv2.line(frame, left_elbow, left_wrist, (255, 255, 255), 2)
                    cv2.line(frame, right_elbow, right_wrist, (255, 255, 255), 2)
                    
                    # Legs (walking motion)
                    leg_angle = np.sin(i * 0.3 + person_id + np.pi) * 0.3
                    left_hip = (int(center_x - 15 * scale), int(center_y - 20 * scale))
                    right_hip = (int(center_x + 15 * scale), int(center_y - 20 * scale))
                    
                    left_knee = (int(left_hip[0] + 20 * scale * np.sin(leg_angle)), 
                                int(left_hip[1] + 50 * scale))
                    right_knee = (int(right_hip[0] - 20 * scale * np.sin(leg_angle)), 
                                 int(right_hip[1] + 50 * scale))
                    
                    left_ankle = (int(left_knee[0] + 10 * scale * np.sin(leg_angle)), 
                                 int(left_knee[1] + 50 * scale))
                    right_ankle = (int(right_knee[0] - 10 * scale * np.sin(leg_angle)), 
                                  int(right_knee[1] + 50 * scale))
                    
                    # Draw legs
                    cv2.line(frame, torso_bottom, left_hip, (255, 255, 255), 2)
                    cv2.line(frame, torso_bottom, right_hip, (255, 255, 255), 2)
                    cv2.line(frame, left_hip, left_knee, (255, 255, 255), 2)
                    cv2.line(frame, right_hip, right_knee, (255, 255, 255), 2)
                    cv2.line(frame, left_knee, left_ankle, (255, 255, 255), 2)
                    cv2.line(frame, right_knee, right_ankle, (255, 255, 255), 2)
                    
                    # Add keypoints
                    keypoints = [
                        (center_x, head_y - 10),  # nose
                        torso_top,  # neck
                        left_shoulder, right_shoulder,
                        left_elbow, right_elbow,
                        left_wrist, right_wrist,
                        left_hip, right_hip,
                        left_knee, right_knee,
                        left_ankle, right_ankle
                    ]
                    
                    for point in keypoints:
                        cv2.circle(frame, point, 3, (0, 255, 0), -1)
                
                out.write(frame)
            
            out.release()
            print(f"Created test video: {test_video}")
            
        except ImportError:
            print("OpenCV not available. Please provide your own test video.")
            return
    
    if test_video.exists():
        print(f"\nProcessing video: {test_video}")
        
        # Method 1: Using the analyzer directly
        print("\n1. Using OpenPoseAnalyzer directly:")
        results = analyzer.analyze_video(str(test_video))
        
        print(f"   Video processed: {results.get('openPose_total_frames', 0)} frames")
        print(f"   Poses detected in: {results.get('openPose_pose_detected_frames', 0)} frames")
        print(f"   Pose detection rate: {results.get('openPose_detection_rate', 0):.2%}")
        print(f"   Average keypoints per frame: {results.get('openPose_avg_keypoints_per_frame', 0):.2f}")
        print(f"   Average confidence: {results.get('openPose_avg_confidence', 0):.3f}")
        print(f"   Maximum persons detected: {results.get('openPose_max_persons_detected', 0)}")
        
        # Show some keypoint positions
        print("\n   Sample Keypoint Positions:")
        keypoints = ['nose', 'neck', 'rshoulder', 'lshoulder', 'rhip', 'lhip']
        for keypoint in keypoints:
            x = results.get(f'openPose_{keypoint}_x', 0.0)
            y = results.get(f'openPose_{keypoint}_y', 0.0)
            conf = results.get(f'openPose_{keypoint}_confidence', 0.0)
            if conf > 0.1:  # Only show confident detections
                print(f"   {keypoint}: ({x:.1f}, {y:.1f}) confidence: {conf:.3f}")
        
        # Show pose measurements
        print("\n   Pose Measurements:")
        measurements = ['shoulder_width', 'hip_width', 'body_height']
        for measurement in measurements:
            value = results.get(f'openPose_{measurement}', 0.0)
            if value > 0:
                print(f"   {measurement}: {value:.1f} pixels")
        
        # Show joint angles
        print("\n   Joint Angles:")
        angles = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle', 'torso_angle']
        for angle in angles:
            value = results.get(f'openPose_{angle}', 0.0)
            if value > 0:
                print(f"   {angle}: {value:.1f} degrees")
        
        # Method 2: Using the pipeline function
        print("\n2. Using extract_openpose_features function:")
        feature_dict = extract_openpose_features(str(test_video), device='cpu')
        
        for feature_group, data in feature_dict.items():
            print(f"   Feature Group: {feature_group}")
            print(f"   Description: {data['description']}")
            features = data['features']
            print(f"   Total features: {len(features)}")
            
            # Count detected keypoints
            keypoint_count = 0
            total_confidence = 0
            for key, value in features.items():
                if key.endswith('_confidence') and value > 0.1:
                    keypoint_count += 1
                    total_confidence += value
            
            if keypoint_count > 0:
                avg_confidence = total_confidence / keypoint_count
                print(f"   Confident keypoints detected: {keypoint_count}")
                print(f"   Average keypoint confidence: {avg_confidence:.3f}")
        
        # Method 3: Testing with pipeline integration
        print("\n3. Testing pipeline integration:")
        try:
            from src.pipeline import MultimodalPipeline
            
            pipeline = MultimodalPipeline(
                features=["openpose_vision"],
                device='cpu'
            )
            
            pipeline_results = pipeline.process_video_file(str(test_video))
            
            # Check if OpenPose features are present
            openpose_found = False
            for key in pipeline_results.keys():
                if "Pose estimation and tracking" in key:
                    openpose_found = True
                    print(f"   ✓ OpenPose features found in pipeline results")
                    break
            
            if not openpose_found:
                print("   ✗ OpenPose features not found in pipeline results")
            
        except Exception as e:
            print(f"   Pipeline integration test failed: {e}")
        
        # Show output files
        output_video = results.get('openPose_pose_video_path', '')
        output_gif = results.get('openPose_pose_gif_path', '')
        
        print(f"\n4. Output Files:")
        if output_video and Path(output_video).exists():
            print(f"   ✓ Annotated video: {output_video}")
        else:
            print(f"   ✗ Annotated video not created")
        
        if output_gif and Path(output_gif).exists():
            print(f"   ✓ Summary GIF: {output_gif}")
        else:
            print(f"   ✗ Summary GIF not created")
        
        # Save results to file
        output_file = project_root / "output" / "openpose_demo_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
    print("\nDemo completed!")
    print("\nOpenPose Features Overview:")
    print("- Real-time multi-person keypoint detection with 18 body keypoints")
    print("- Provides x,y coordinates and confidence scores for each keypoint")
    print("- Calculates joint angles (arms, legs, torso) and body measurements")
    print("- Outputs annotated video and GIF with pose skeleton overlay")
    print("- Supports multi-person detection and tracking across frames")
    print("- Returns 50+ features including keypoints, angles, measurements, and statistics")

if __name__ == "__main__":
    main()
