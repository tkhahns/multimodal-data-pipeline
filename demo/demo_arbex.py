#!/usr/bin/env python3
"""
Demo script for ARBEx attentive feature extraction with reliability balancing for robust facial expression learning.
This demonstrates emotional indices extraction via different feature levels.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from vision.arbex_analyzer import ARBExAnalyzer, extract_arbex_features

def main():
    print("ARBEx Emotional Expression Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer
    print("Initializing ARBEx analyzer...")
    analyzer = ARBExAnalyzer(device='cpu', confidence_threshold=0.5)
    
    # Demo with a test video (you can replace this with your own video)
    test_video = project_root / "test_data" / "sample_video.mp4"
    
    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        print("Please provide a video file to test with.")
        
        # Create a simple test video using OpenCV if available
        try:
            import cv2
            import numpy as np
            
            print("Creating a simple test video...")
            test_video.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simple video with a moving face-like pattern and expressions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(test_video), fourcc, 30.0, (640, 480))
            
            for i in range(90):  # 3 seconds at 30 fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Create a simple face-like pattern
                center_x, center_y = 320, 240
                
                # Face outline (circle)
                cv2.circle(frame, (center_x, center_y), 100, (255, 255, 255), 2)
                
                # Eyes (vary with frame to simulate expressions)
                eye_y_offset = int(5 * np.sin(i * 0.1)) if i < 30 else 0  # Blinking in first second
                cv2.circle(frame, (center_x - 30, center_y - 20 + eye_y_offset), 8, (255, 255, 255), -1)
                cv2.circle(frame, (center_x + 30, center_y - 20 + eye_y_offset), 8, (255, 255, 255), -1)
                
                # Mouth (changes with frame to simulate different emotions)
                if i < 30:  # Neutral to happy
                    mouth_curve = int(5 * (i / 30))
                    cv2.ellipse(frame, (center_x, center_y + 30), (20, 10 + mouth_curve), 0, 0, 180, (255, 255, 255), 2)
                elif i < 60:  # Happy to surprised
                    mouth_size = 10 + int(10 * ((i - 30) / 30))
                    cv2.circle(frame, (center_x, center_y + 30), mouth_size, (255, 255, 255), 2)
                else:  # Surprised to sad
                    mouth_curve = 15 - int(15 * ((i - 60) / 30))
                    cv2.ellipse(frame, (center_x, center_y + 30), (20, mouth_curve), 0, 180, 360, (255, 255, 255), 2)
                
                # Eyebrows (vary for different expressions)
                if i >= 30 and i < 60:  # Surprise - raised eyebrows
                    cv2.line(frame, (center_x - 40, center_y - 40), (center_x - 20, center_y - 45), (255, 255, 255), 2)
                    cv2.line(frame, (center_x + 20, center_y - 45), (center_x + 40, center_y - 40), (255, 255, 255), 2)
                elif i >= 60:  # Sad - lowered eyebrows
                    cv2.line(frame, (center_x - 40, center_y - 35), (center_x - 20, center_y - 30), (255, 255, 255), 2)
                    cv2.line(frame, (center_x + 20, center_y - 30), (center_x + 40, center_y - 35), (255, 255, 255), 2)
                else:  # Normal eyebrows
                    cv2.line(frame, (center_x - 40, center_y - 35), (center_x - 20, center_y - 35), (255, 255, 255), 2)
                    cv2.line(frame, (center_x + 20, center_y - 35), (center_x + 40, center_y - 35), (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            print(f"Created test video with simulated expressions: {test_video}")
            
        except ImportError:
            print("OpenCV not available. Please provide your own test video.")
            return
    
    if test_video.exists():
        print(f"\nProcessing video: {test_video}")
        
        # Method 1: Using the analyzer directly
        print("\n1. Using ARBExAnalyzer directly:")
        results = analyzer.analyze_video(str(test_video))
        
        print(f"   Video processed: {results.get('total_frames', 0)} frames")
        print(f"   Faces detected in: {results.get('faces_detected_frames', 0)} frames")
        print(f"   Face detection rate: {results.get('face_detection_rate', 0):.2%}")
        print(f"   Primary emotion: {results.get('arbex_primary', 'Unknown')}")
        print(f"   Final emotion: {results.get('arbex_final', 'Unknown')}")
        print(f"   Average primary confidence: {results.get('avg_confidence_primary', 0):.3f}")
        print(f"   Average final confidence: {results.get('avg_confidence_final', 0):.3f}")
        print(f"   Average reliability score: {results.get('avg_reliability_score', 0):.3f}")
        
        # Show emotion probabilities
        print("\n   Primary Level Emotion Probabilities:")
        emotions = ['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'others']
        for emotion in emotions:
            key = f'arbex_primary_{emotion}'
            value = results.get(key, 0.0)
            print(f"   {emotion.title()}: {value:.3f}")
        
        print("\n   Final Level Emotion Probabilities:")
        for emotion in emotions:
            key = f'arbex_final_{emotion}'
            value = results.get(key, 0.0)
            print(f"   {emotion.title()}: {value:.3f}")
        
        # Method 2: Using the pipeline function
        print("\n2. Using extract_arbex_features function:")
        feature_dict = extract_arbex_features(str(test_video), device='cpu')
        
        for feature_group, data in feature_dict.items():
            print(f"   Feature Group: {feature_group}")
            print(f"   Description: {data['description']}")
            features = data['features']
            print(f"   Total features: {len(features)}")
            
            # Show key metrics
            key_metrics = ['arbex_primary', 'arbex_final', 'arbex_confidence_primary', 
                          'arbex_confidence_final', 'arbex_reliability_score']
            print("   Key Metrics:")
            for metric in key_metrics:
                if metric in features:
                    print(f"   {metric}: {features[metric]}")
        
        # Method 3: Testing with pipeline integration
        print("\n3. Testing pipeline integration:")
        try:
            from pipeline import MultimodalPipeline
            
            pipeline = MultimodalPipeline(
                features=["arbex_vision"],
                device='cpu'
            )
            
            pipeline_results = pipeline.process_video_file(str(test_video))
            
            # Check if ARBEx features are present
            arbex_found = False
            for key in pipeline_results.keys():
                if "Extract emotional indices via different feature levels" in key:
                    arbex_found = True
                    print(f"   ✓ ARBEx features found in pipeline results")
                    break
            
            if not arbex_found:
                print("   ✗ ARBEx features not found in pipeline results")
            
        except Exception as e:
            print(f"   Pipeline integration test failed: {e}")
        
        # Save results to file
        output_file = project_root / "output" / "arbex_demo_results.json"
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
    print("\nARBEx Features Overview:")
    print("- Primary emotion classification: Initial emotion recognition")
    print("- Final emotion classification: After reliability balancing")
    print("- Reliability balancing: Adjusts classifications based on feature consistency")
    print("- Supports 8 emotion categories: Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise, Others")
    print("- Multi-level feature extraction: Statistical, regional, and texture features")
    print("- Attentive feature extraction with confidence scoring")

if __name__ == "__main__":
    main()
