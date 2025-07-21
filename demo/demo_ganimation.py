#!/usr/bin/env python3
"""
Demo script for GANimation continuous manifold for anatomical facial movements.
This demonstrates Action Unit (AU) intensity estimation at discrete levels.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from vision.ganimation_analyzer import GANimationAnalyzer, extract_ganimation_features

def main():
    print("GANimation Action Unit Intensity Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer
    print("Initializing GANimation analyzer...")
    analyzer = GANimationAnalyzer(device='cpu', confidence_threshold=0.5)
    
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
            
            # Create a simple video with a moving face-like pattern
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(test_video), fourcc, 30.0, (640, 480))
            
            for i in range(90):  # 3 seconds at 30 fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Create a simple face-like pattern
                center_x, center_y = 320, 240
                
                # Face outline (circle)
                cv2.circle(frame, (center_x, center_y), 100, (255, 255, 255), 2)
                
                # Eyes
                cv2.circle(frame, (center_x - 30, center_y - 20), 10, (255, 255, 255), -1)
                cv2.circle(frame, (center_x + 30, center_y - 20), 10, (255, 255, 255), -1)
                
                # Mouth (changes with frame to simulate expression)
                mouth_y = center_y + 30 + int(10 * np.sin(i * 0.2))
                cv2.ellipse(frame, (center_x, mouth_y), (20, 10), 0, 0, 180, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            print(f"Created test video: {test_video}")
            
        except ImportError:
            print("OpenCV not available. Please provide your own test video.")
            return
    
    if test_video.exists():
        print(f"\nProcessing video: {test_video}")
        
        # Method 1: Using the analyzer directly
        print("\n1. Using GANimationAnalyzer directly:")
        results = analyzer.analyze_video(str(test_video))
        
        print(f"   Video processed: {results.get('total_frames', 0)} frames")
        print(f"   Faces detected in: {results.get('faces_detected_frames', 0)} frames")
        print(f"   Face detection rate: {results.get('face_detection_rate', 0):.2%}")
        print(f"   Average AU activations per frame: {results.get('avg_au_activations_per_frame', 0):.2f}")
        
        # Show some Action Unit intensities
        print("\n   Sample Action Unit Intensities:")
        for au in ['AU1', 'AU2', 'AU6', 'AU12', 'AU25']:
            for intensity in [0, 33, 66, 99]:
                key = f'GAN_{au}_{intensity}'
                value = results.get(key, 0.0)
                if value > 0.1:  # Only show significant activations
                    print(f"   {key}: {value:.3f}")
        
        # Method 2: Using the pipeline function
        print("\n2. Using extract_ganimation_features function:")
        feature_dict = extract_ganimation_features(str(test_video), device='cpu')
        
        for feature_group, data in feature_dict.items():
            print(f"   Feature Group: {feature_group}")
            print(f"   Description: {data['description']}")
            features = data['features']
            print(f"   Total features: {len(features)}")
            
            # Count AU activations by intensity level
            intensity_counts = {0: 0, 33: 0, 66: 0, 99: 0}
            for key, value in features.items():
                if key.startswith('GAN_AU') and value > 0.1:
                    for intensity in [0, 33, 66, 99]:
                        if key.endswith(f'_{intensity}'):
                            intensity_counts[intensity] += 1
                            break
            
            print("   AU activations by intensity level:")
            for intensity, count in intensity_counts.items():
                print(f"   Level {intensity}: {count} AUs")
          # Method 3: Testing with pipeline integration
        print("\n3. Testing pipeline integration:")
        try:
            from pipeline import MultimodalPipeline
            
            pipeline = MultimodalPipeline(
                features=["ganimation_vision"],
                device='cpu'
            )
            
            pipeline_results = pipeline.process_video_file(str(test_video))
            
            # Check if GANimation features are present
            ganimation_found = False
            for key in pipeline_results.keys():
                if "Continuous manifold for anatomical facial movements" in key:
                    ganimation_found = True
                    print(f"   ✓ GANimation features found in pipeline results")
                    break
            
            if not ganimation_found:
                print("   ✗ GANimation features not found in pipeline results")
            
        except Exception as e:
            print(f"   Pipeline integration test failed: {e}")
        
        # Save results to file
        output_file = project_root / "output" / "ganimation_demo_results.json"
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
    print("\nGANimation Features Overview:")
    print("- Estimates Action Unit (AU) intensities at 4 discrete levels: 0, 33, 66, 99")
    print("- Covers 17 facial Action Units from the Facial Action Coding System (FACS)")
    print("- Provides continuous manifold representation for anatomical facial movements")
    print("- Outputs include face detection statistics and AU activation summaries")

if __name__ == "__main__":
    main()
