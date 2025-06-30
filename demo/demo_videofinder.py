#!/usr/bin/env python3
"""
Demo script for VideoFinder (Object and People Localization) integration.

This script demonstrates:
1. Object detection and localization in video frames
2. People detection and tracking consistency
3. Match analysis between consecutive frames
4. Consistency scoring for detected objects/people
5. Pipeline integration and feature extraction
6. Output columns: ViF_consistency_1, ViF_match_1, ViF_consistency_2, ViF_match_2, etc.

Usage:
    python demo/demo_videofinder.py
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

from src.vision.videofinder_analyzer import VideoFinderAnalyzer
from src.pipeline import MultimodalPipeline


def create_test_video_with_objects(output_path: str, duration: float = 3.0, fps: int = 30):
    """Create a test video with moving objects and people for localization analysis."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Create base frame with textured background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add textured background
        for y in range(0, height, 40):
            for x in range(0, width, 40):
                color = (50 + (x + y) % 100, 30 + (x - y) % 80, 40 + (x * y) % 60)
                cv2.rectangle(frame, (x, y), (x + 40, y + 40), color, -1)
        
        t = frame_idx / total_frames
        
        # Add moving person (larger, person-like shape)
        person_x = int(100 + t * 300)
        person_y = int(height // 2 - 50)
        
        # Person silhouette
        # Head
        cv2.circle(frame, (person_x, person_y), 25, (200, 150, 100), -1)
        # Body
        cv2.rectangle(frame, (person_x - 20, person_y + 25), (person_x + 20, person_y + 100), (180, 130, 90), -1)
        # Arms
        cv2.rectangle(frame, (person_x - 35, person_y + 30), (person_x - 20, person_y + 80), (180, 130, 90), -1)
        cv2.rectangle(frame, (person_x + 20, person_y + 30), (person_x + 35, person_y + 80), (180, 130, 90), -1)
        # Legs
        cv2.rectangle(frame, (person_x - 15, person_y + 100), (person_x - 5, person_y + 150), (120, 80, 60), -1)
        cv2.rectangle(frame, (person_x + 5, person_y + 100), (person_x + 15, person_y + 150), (120, 80, 60), -1)
        
        # Add moving car (rectangular object)
        car_x = int(500 - t * 200)
        car_y = int(height // 2 + 50)
        
        # Car body
        cv2.rectangle(frame, (car_x - 40, car_y - 15), (car_x + 40, car_y + 15), (100, 100, 200), -1)
        # Car wheels
        cv2.circle(frame, (car_x - 25, car_y + 15), 8, (50, 50, 50), -1)
        cv2.circle(frame, (car_x + 25, car_y + 15), 8, (50, 50, 50), -1)
        # Car windows
        cv2.rectangle(frame, (car_x - 30, car_y - 10), (car_x + 30, car_y - 5), (150, 200, 255), -1)
        
        # Add stationary object (building/structure)
        building_x = 500
        building_y = 100
        cv2.rectangle(frame, (building_x, building_y), (building_x + 80, building_y + 120), (120, 120, 120), -1)
        # Building windows
        for i in range(3):
            for j in range(4):
                window_x = building_x + 10 + i * 20
                window_y = building_y + 20 + j * 20
                cv2.rectangle(frame, (window_x, window_y), (window_x + 10, window_y + 15), (255, 255, 200), -1)
        
        # Add dynamic object (ball bouncing)
        ball_phase = (frame_idx * 0.2) % (2 * np.pi)
        ball_x = int(200 + 50 * np.sin(ball_phase * 2))
        ball_y = int(150 + 30 * np.abs(np.sin(ball_phase)))
        cv2.circle(frame, (ball_x, ball_y), 20, (50, 255, 50), -1)
        
        # Add some noise for realism
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add frame information
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Objects: Person, Car, Building, Ball", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")


def demonstrate_videofinder_analyzer():
    """Demonstrate the VideoFinder object and people localization analyzer."""
    print("=" * 80)
    print("VideoFinder (Object and People Localization) Analyzer Demo")
    print("=" * 80)
    
    # Create test video
    test_video_path = "demo_videofinder_test_video.mp4"
    print(f"Creating test video with objects and people: {test_video_path}")
    create_test_video_with_objects(test_video_path)
    
    try:
        # Initialize VideoFinder analyzer
        print("\n1. Initializing VideoFinder analyzer...")
        analyzer = VideoFinderAnalyzer(device='cpu')
        
        print(f"   - Device: {analyzer.device}")
        print(f"   - Model loaded: {analyzer.is_model_loaded}")
        print(f"   - Confidence threshold: {analyzer.confidence_threshold}")
        print(f"   - Number of detection categories: {len(analyzer.detection_categories)}")
        
        # Extract features from video
        print(f"\n2. Extracting VideoFinder localization features from video...")
        start_time = time.time()
        
        features = analyzer.get_feature_dict(test_video_path)
        
        extraction_time = time.time() - start_time
        
        print(f"   - Extraction completed in {extraction_time:.2f} seconds")
        
        # Extract the actual features from the feature dict
        if "Locate the objects and people" in features:
            actual_features = features["Locate the objects and people"]["features"]
            print(f"   - Number of features extracted: {len(actual_features)}")
        else:
            actual_features = features
            print(f"   - Number of features extracted: {len(features)}")
        
        # Display feature results
        print("\n3. VideoFinder Feature Results:")
        print("-" * 50)
        
        # Object detection summary
        print("Object Detection Summary:")
        print(f"   Total frames: {actual_features.get('total_frames', 'N/A')}")
        print(f"   Objects detected frames: {actual_features.get('objects_detected_frames', 'N/A')}")
        print(f"   People detected frames: {actual_features.get('people_detected_frames', 'N/A')}")
        print(f"   Detection rate: {actual_features.get('detection_rate', 'N/A'):.4f}")
        
        # Average detections
        print("Average Detections:")
        print(f"   Avg objects per frame: {actual_features.get('avg_objects_per_frame', 'N/A'):.2f}")
        print(f"   Avg people per frame: {actual_features.get('avg_people_per_frame', 'N/A'):.2f}")
        print(f"   Total detected objects: {actual_features.get('total_detected_objects', 'N/A')}")
        print(f"   Total detected people: {actual_features.get('total_detected_people', 'N/A')}")
        
        # Consistency and match results for first few objects
        print("\nConsistency and Match Analysis (first 5 detections):")
        for i in range(1, 6):
            consistency_key = f"ViF_consistency_{i}"
            match_key = f"ViF_match_{i}"
            
            if consistency_key in actual_features and match_key in actual_features:
                consistency = actual_features[consistency_key]
                match = actual_features[match_key]
                print(f"   Detection {i}:")
                print(f"      Consistency: {consistency}")
                print(f"      Match: {match}")
        
        # Show sample of other ViF features
        vif_features = {k: v for k, v in actual_features.items() if k.startswith('ViF_')}
        if vif_features:
            print(f"\nTotal ViF features: {len(vif_features)}")
            print("Sample ViF features:")
            for i, (key, value) in enumerate(list(vif_features.items())[:10]):
                print(f"   {key}: {value}")
            if len(vif_features) > 10:
                print(f"   ... and {len(vif_features) - 10} more ViF features")
        
        # Analyze results
        print("\n4. Analysis Results:")
        print("-" * 50)
        
        # Detection quality analysis
        detection_rate = actual_features.get('detection_rate', 0)
        if detection_rate > 0.8:
            print(f"Detection quality: EXCELLENT ({detection_rate:.4f}) - High detection rate")
        elif detection_rate > 0.6:
            print(f"Detection quality: GOOD ({detection_rate:.4f}) - Moderate detection rate")
        elif detection_rate > 0.3:
            print(f"Detection quality: FAIR ({detection_rate:.4f}) - Low detection rate")
        else:
            print(f"Detection quality: POOR ({detection_rate:.4f}) - Very low detection rate")
        
        # Consistency analysis
        consistency_scores = []
        for i in range(1, 11):  # Check first 10 detections
            consistency_key = f"ViF_consistency_{i}"
            if consistency_key in actual_features:
                consistency_value = actual_features[consistency_key]
                if isinstance(consistency_value, str) and '/' in consistency_value:
                    try:
                        num, den = consistency_value.split('/')
                        score = float(num) / float(den)
                        consistency_scores.append(score)
                    except:
                        pass
        
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            print(f"Average consistency score: {avg_consistency:.4f}")
            if avg_consistency > 0.8:
                print("   Result: HIGH consistency - Stable object tracking")
            elif avg_consistency > 0.5:
                print("   Result: MEDIUM consistency - Moderate tracking stability")
            else:
                print("   Result: LOW consistency - Unstable tracking")
        
        # Match analysis
        match_count = 0
        total_matches = 0
        for i in range(1, 11):  # Check first 10 detections
            match_key = f"ViF_match_{i}"
            if match_key in actual_features:
                match_value = actual_features[match_key]
                total_matches += 1
                if isinstance(match_value, str) and match_value.lower() in ['yes', 'true']:
                    match_count += 1
        
        if total_matches > 0:
            match_rate = match_count / total_matches
            print(f"Match rate: {match_rate:.4f} ({match_count}/{total_matches})")
            if match_rate > 0.7:
                print("   Result: HIGH match rate - Good object correspondence")
            elif match_rate > 0.4:
                print("   Result: MEDIUM match rate - Some object correspondence")
            else:
                print("   Result: LOW match rate - Poor object correspondence")
        
        return actual_features
        
    except Exception as e:
        print(f"Error during VideoFinder analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
            print(f"\nCleaned up test video: {test_video_path}")


def demonstrate_pipeline_integration():
    """Demonstrate VideoFinder integration with the multimodal pipeline."""
    print("\n" + "=" * 80)
    print("VideoFinder Pipeline Integration Demo")
    print("=" * 80)
    
    # Create test video
    test_video_path = "demo_videofinder_pipeline_test.mp4"
    print(f"Creating test video for pipeline: {test_video_path}")
    create_test_video_with_objects(test_video_path, duration=2.0)
    
    try:
        # Initialize pipeline with only VideoFinder
        print("\n1. Initializing pipeline with VideoFinder vision features...")
        pipeline = MultimodalPipeline(
            features=["videofinder_vision"],
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
                    
                    # Show sample features
                    feature_items = list(group_data['features'].items())
                    
                    # Show summary features first
                    summary_features = ['total_frames', 'detection_rate', 'avg_objects_per_frame', 'avg_people_per_frame']
                    for feat_name in summary_features:
                        if feat_name in group_data['features']:
                            value = group_data['features'][feat_name]
                            if isinstance(value, (int, float)):
                                print(f"   {feat_name}: {value:.4f}")
                            else:
                                print(f"   {feat_name}: {value}")
                    
                    # Show sample ViF features
                    vif_features = [(k, v) for k, v in feature_items if k.startswith('ViF_')]
                    if vif_features:
                        print(f"   Sample ViF features ({len(vif_features)} total):")
                        for i, (feature_name, feature_value) in enumerate(vif_features[:5]):
                            print(f"      {feature_name}: {feature_value}")
                        if len(vif_features) > 5:
                            print(f"      ... and {len(vif_features) - 5} more ViF features")
        else:
            print("No results returned from pipeline")
        
        return results
        
    except Exception as e:
        print(f"Error during pipeline integration: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
            print(f"\nCleaned up test video: {test_video_path}")


def main():
    """Main demo function."""
    print("VideoFinder (Object and People Localization) Demo")
    print("Website: https://github.com/win4r/VideoFinder-Llama3.2-vision-Ollama")
    print()
    
    try:
        # Demo 1: VideoFinder analyzer standalone
        analyzer_results = demonstrate_videofinder_analyzer()
        
        # Demo 2: Pipeline integration
        pipeline_results = demonstrate_pipeline_integration()
        
        # Summary
        print("\n" + "=" * 80)
        print("Demo Summary")
        print("=" * 80)
        
        if analyzer_results:
            print("✓ VideoFinder analyzer demonstration completed successfully")
            print(f"  - Extracted features for object and people localization")
            
            # Highlight key metrics
            print("  - Key metrics:")
            print(f"    Detection rate: {analyzer_results.get('detection_rate', 'N/A'):.4f}")
            print(f"    Avg objects per frame: {analyzer_results.get('avg_objects_per_frame', 'N/A'):.2f}")
            print(f"    Avg people per frame: {analyzer_results.get('avg_people_per_frame', 'N/A'):.2f}")
            
            # Count ViF features
            vif_count = len([k for k in analyzer_results.keys() if k.startswith('ViF_')])
            print(f"    ViF features generated: {vif_count}")
        else:
            print("✗ VideoFinder analyzer demonstration failed")
        
        if pipeline_results:
            print("✓ Pipeline integration demonstration completed successfully")
            print(f"  - Processed video through {len(pipeline_results)} feature groups")
        else:
            print("✗ Pipeline integration demonstration failed")
        
        print("\nVideoFinder Features:")
        print("- Object detection and localization in video frames")
        print("- People detection and tracking consistency")
        print("- Match analysis between consecutive frames")
        print("- Consistency scoring for detected objects/people")
        print("- ViF_consistency_N and ViF_match_N feature columns")
        print("- Integration with Llama3.2-vision and Ollama")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
