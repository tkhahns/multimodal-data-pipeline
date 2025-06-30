#!/usr/bin/env python3
"""
Demo script for SmoothNet (Temporally Consistent Pose Estimation) integration.

This script demonstrates:
1. Temporally consistent 3D pose estimation from video
2. 2D pose refinement and smoothing
3. SMPL body model fitting integration
4. Multi-frame pose sequence modeling
5. Pipeline integration and feature extraction
6. Output columns: net_3d_estimator, net_2d_estimator, net_SMPL_estimator, etc.

Usage:
    python demo/demo_smoothnet.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import time

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.smoothnet_analyzer import SmoothNetAnalyzer
from src.pipeline import MultimodalPipeline


def create_test_video_with_person(output_path: str, duration: float = 3.0, fps: int = 30):
    """Create a test video with a moving person for pose estimation analysis."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Create base frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            intensity = int(50 + (y / height) * 100)
            frame[y, :] = [intensity, intensity//2, intensity//3]
        
        # Add walking person simulation (stick figure)
        t = frame_idx / total_frames
        
        # Person center moves across screen
        person_x = int(100 + t * 400)
        person_y = int(height // 2)
        
        # Walking animation phase
        walk_phase = (frame_idx * 0.3) % (2 * np.pi)
        
        # Head
        cv2.circle(frame, (person_x, person_y - 60), 20, (255, 255, 255), -1)
        
        # Body (torso)
        cv2.line(frame, (person_x, person_y - 40), (person_x, person_y + 20), (255, 255, 255), 3)
        
        # Arms with swing motion
        arm_swing = np.sin(walk_phase) * 20
        left_arm_x = person_x - 15 + int(arm_swing)
        right_arm_x = person_x + 15 - int(arm_swing)
        
        # Left arm
        cv2.line(frame, (person_x, person_y - 30), (left_arm_x, person_y - 10), (255, 255, 255), 2)
        cv2.line(frame, (left_arm_x, person_y - 10), (left_arm_x - 10, person_y + 10), (255, 255, 255), 2)
        
        # Right arm
        cv2.line(frame, (person_x, person_y - 30), (right_arm_x, person_y - 10), (255, 255, 255), 2)
        cv2.line(frame, (right_arm_x, person_y - 10), (right_arm_x + 10, person_y + 10), (255, 255, 255), 2)
        
        # Legs with walking motion
        leg_swing = np.sin(walk_phase) * 25
        left_leg_x = person_x - 10 + int(leg_swing)
        right_leg_x = person_x + 10 - int(leg_swing)
        
        # Left leg
        cv2.line(frame, (person_x, person_y + 20), (left_leg_x, person_y + 50), (255, 255, 255), 3)
        cv2.line(frame, (left_leg_x, person_y + 50), (left_leg_x, person_y + 80), (255, 255, 255), 3)
        
        # Right leg
        cv2.line(frame, (person_x, person_y + 20), (right_leg_x, person_y + 50), (255, 255, 255), 3)
        cv2.line(frame, (right_leg_x, person_y + 50), (right_leg_x, person_y + 80), (255, 255, 255), 3)
        
        # Add some noise for realism
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")


def demonstrate_smoothnet_analyzer():
    """Demonstrate the SmoothNet pose estimation analyzer."""
    print("=" * 80)
    print("SmoothNet (Temporally Consistent Pose Estimation) Analyzer Demo")
    print("=" * 80)
    
    # Create test video
    test_video_path = "demo_smoothnet_test_video.mp4"
    print(f"Creating test video with walking person: {test_video_path}")
    create_test_video_with_person(test_video_path)
    
    try:
        # Initialize SmoothNet analyzer
        print("\n1. Initializing SmoothNet analyzer...")
        analyzer = SmoothNetAnalyzer(device='cpu')
        
        print(f"   - Device: {analyzer.device}")
        print(f"   - Model loaded: {analyzer.is_model_loaded}")
        print(f"   - Feature names: {analyzer.feature_names}")
        
        # Extract features from video
        print(f"\n2. Extracting SmoothNet pose estimation features from video...")
        start_time = time.time()
        
        features = analyzer.extract_features(test_video_path)
        
        extraction_time = time.time() - start_time
        
        print(f"   - Extraction completed in {extraction_time:.2f} seconds")
        print(f"   - Number of features extracted: {len(features)}")
        
        # Display feature results
        print("\n3. SmoothNet Feature Results:")
        print("-" * 50)
        
        # 3D pose estimation results
        print("3D Pose Estimation:")
        print(f"   net_3d_estimator: {features.get('net_3d_estimator', 'N/A'):.4f}")
        
        # 2D pose estimation results
        print("2D Pose Estimation:")
        print(f"   net_2d_estimator: {features.get('net_2d_estimator', 'N/A'):.4f}")
        
        # SMPL model fitting results
        print("SMPL Body Model Fitting:")
        print(f"   net_SMPL_estimator: {features.get('net_SMPL_estimator', 'N/A'):.4f}")
        
        # Temporal consistency metrics
        print("Temporal Consistency:")
        print(f"   net_temporal_consistency: {features.get('net_temporal_consistency', 'N/A'):.4f}")
        print(f"   net_smoothness_score: {features.get('net_smoothness_score', 'N/A'):.4f}")
        print(f"   net_motion_coherence: {features.get('net_motion_coherence', 'N/A'):.4f}")
        
        # Quality metrics
        print("Quality Metrics:")
        print(f"   net_joint_confidence: {features.get('net_joint_confidence', 'N/A'):.4f}")
        print(f"   net_pose_stability: {features.get('net_pose_stability', 'N/A'):.4f}")
        print(f"   net_tracking_accuracy: {features.get('net_tracking_accuracy', 'N/A'):.4f}")
        print(f"   net_keypoint_variance: {features.get('net_keypoint_variance', 'N/A'):.4f}")
        
        # Analyze results
        print("\n4. Analysis Results:")
        print("-" * 50)
        
        # Overall pose estimation quality
        avg_estimator_score = (
            features.get('net_3d_estimator', 0) + 
            features.get('net_2d_estimator', 0) + 
            features.get('net_SMPL_estimator', 0)
        ) / 3
        
        print(f"Average pose estimation quality: {avg_estimator_score:.4f}")
        
        if avg_estimator_score > 0.7:
            print("   Result: HIGH quality pose estimation detected")
        elif avg_estimator_score > 0.4:
            print("   Result: MEDIUM quality pose estimation detected")
        else:
            print("   Result: LOW quality pose estimation detected")
        
        # Temporal consistency analysis
        temporal_score = features.get('net_temporal_consistency', 0)
        if temporal_score > 0.7:
            print(f"Temporal consistency: HIGH ({temporal_score:.4f}) - Smooth pose sequence")
        elif temporal_score > 0.4:
            print(f"Temporal consistency: MEDIUM ({temporal_score:.4f}) - Moderately smooth")
        else:
            print(f"Temporal consistency: LOW ({temporal_score:.4f}) - Jerky motion detected")
        
        # Motion coherence analysis
        coherence_score = features.get('net_motion_coherence', 0)
        if coherence_score > 0.7:
            print(f"Motion coherence: HIGH ({coherence_score:.4f}) - Coherent motion patterns")
        elif coherence_score > 0.4:
            print(f"Motion coherence: MEDIUM ({coherence_score:.4f}) - Some inconsistencies")
        else:
            print(f"Motion coherence: LOW ({coherence_score:.4f}) - Incoherent motion")
        
        return features
        
    except Exception as e:
        print(f"Error during SmoothNet analysis: {e}")
        return {}
    
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
            print(f"\nCleaned up test video: {test_video_path}")


def demonstrate_pipeline_integration():
    """Demonstrate SmoothNet integration with the multimodal pipeline."""
    print("\n" + "=" * 80)
    print("SmoothNet Pipeline Integration Demo")
    print("=" * 80)
    
    # Create test video
    test_video_path = "demo_smoothnet_pipeline_test.mp4"
    print(f"Creating test video for pipeline: {test_video_path}")
    create_test_video_with_person(test_video_path, duration=2.0)
    
    try:
        # Initialize pipeline with only SmoothNet
        print("\n1. Initializing pipeline with SmoothNet vision features...")
        pipeline = MultimodalPipeline(
            features=["smoothnet_vision"],
            device='cpu'
        )
        
        print(f"   - Selected features: {pipeline.features}")
        
        # Process video through pipeline
        print("\n2. Processing video through pipeline...")
        start_time = time.time()
        
        results = pipeline.process_video_file(test_video_path)
        
        processing_time = time.time() - start_time
        
        print(f"   - Pipeline processing completed in {processing_time:.2f} seconds")
        
        # Display pipeline results
        print("\n3. Pipeline Results:")
        print("-" * 50)
        
        if results:
            for group_name, group_data in results.items():
                print(f"\nGroup: {group_name}")
                if isinstance(group_data, dict) and 'features' in group_data:
                    print(f"Model: {group_data.get('Model', 'Unknown')}")
                    print(f"Feature Count: {len(group_data['features'])}")
                    
                    # Show first few features as example
                    feature_items = list(group_data['features'].items())
                    for i, (feature_name, feature_value) in enumerate(feature_items[:5]):
                        if isinstance(feature_value, (int, float)):
                            print(f"   {feature_name}: {feature_value:.4f}")
                        else:
                            print(f"   {feature_name}: {type(feature_value).__name__}")
                    
                    if len(feature_items) > 5:
                        print(f"   ... and {len(feature_items) - 5} more features")
        else:
            print("No results returned from pipeline")
        
        return results
        
    except Exception as e:
        print(f"Error during pipeline integration: {e}")
        return {}
    
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
            print(f"\nCleaned up test video: {test_video_path}")


def main():
    """Main demo function."""
    print("SmoothNet (Temporally Consistent Pose Estimation) Demo")
    print("Website: https://github.com/cure-lab/SmoothNet")
    print()
    
    try:
        # Demo 1: SmoothNet analyzer standalone
        analyzer_results = demonstrate_smoothnet_analyzer()
        
        # Demo 2: Pipeline integration
        pipeline_results = demonstrate_pipeline_integration()
        
        # Summary
        print("\n" + "=" * 80)
        print("Demo Summary")
        print("=" * 80)
        
        if analyzer_results:
            print("✓ SmoothNet analyzer demonstration completed successfully")
            print(f"  - Extracted {len(analyzer_results)} pose estimation features")
            
            # Highlight key metrics
            key_metrics = [
                'net_3d_estimator', 'net_2d_estimator', 'net_SMPL_estimator',
                'net_temporal_consistency', 'net_smoothness_score', 'net_motion_coherence'
            ]
            
            print("  - Key metrics:")
            for metric in key_metrics:
                if metric in analyzer_results:
                    print(f"    {metric}: {analyzer_results[metric]:.4f}")
        else:
            print("✗ SmoothNet analyzer demonstration failed")
        
        if pipeline_results:
            print("✓ Pipeline integration demonstration completed successfully")
            print(f"  - Processed video through {len(pipeline_results)} feature groups")
        else:
            print("✗ Pipeline integration demonstration failed")
        
        print("\nSmoothNet Features:")
        print("- Temporally consistent 3D and 2D pose estimation")
        print("- SMPL body model fitting and integration")
        print("- Multi-frame pose sequence modeling")
        print("- Neural network-based pose smoothing")
        print("- Robust tracking with temporal coherence")
        print("- Motion coherence analysis across time")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
