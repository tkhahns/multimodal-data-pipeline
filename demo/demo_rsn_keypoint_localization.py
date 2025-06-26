"""
Demo script for RSN (Residual Steps Network) keypoint localization feature extraction.

This script demonstrates how to use the RSN analyzer for keypoint localization
and pose estimation from video files.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline import MultimodalPipeline


def main():
    """Main function to demonstrate RSN keypoint localization."""
    print("=" * 60)
    print("RSN Keypoint Localization Demo")
    print("=" * 60)
    
    # Setup pipeline with only RSN vision features
    output_dir = Path("output") / "rsn_demo"
    pipeline = MultimodalPipeline(
        output_dir=output_dir,
        features=["rsn_vision"],  # Only extract RSN features
        device="cpu"  # Use CPU for demo (change to "cuda" if GPU available)
    )
    
    print(f"Pipeline initialized with output directory: {output_dir}")
    print("Features to extract: RSN keypoint localization")
    print()
    
    # Example video files to process (replace with actual video paths)
    video_files = [
        "sample_video.mp4",  # Replace with actual video path
        "test_video.avi",    # Replace with actual video path
    ]
    
    # Filter to only existing files
    existing_files = [f for f in video_files if os.path.exists(f)]
    
    if not existing_files:
        print("No video files found. Please update the video_files list with actual video paths.")
        print("Example usage:")
        print("1. Place a video file (e.g., 'test_video.mp4') in the current directory")
        print("2. Update the video_files list above with the correct path")
        print("3. Run this script again")
        return
    
    # Process each video file
    for video_path in existing_files:
        print(f"Processing video: {video_path}")
        print("-" * 40)
        
        try:
            # Process the video file
            features = pipeline.process_video_file(video_path)
            
            # Display extracted RSN features
            print("RSN Keypoint Localization Features:")
            for feature_name, feature_value in features.items():
                if feature_name.startswith("rsn_"):
                    if isinstance(feature_value, (int, float)):
                        print(f"  {feature_name}: {feature_value:.4f}")
                    else:
                        print(f"  {feature_name}: {feature_value}")
            
            print(f"\\nFeatures saved to: {output_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            print("=" * 60)
    
    # Demonstrate feature grouping
    print("\\nFeature Grouping Example:")
    print("-" * 30)
    
    if existing_files:
        try:
            features = pipeline.process_video_file(existing_files[0])
            grouped_features = pipeline._group_features_by_model(features)
            
            if "Keypoint localization" in grouped_features:
                rsn_group = grouped_features["Keypoint localization"]
                print(f"Feature Category: {rsn_group['Feature']}")
                print(f"Model: {rsn_group['Model']}")
                print(f"Number of features: {len(rsn_group['features'])}")
                print("Feature details:")
                
                for feature_name, feature_value in rsn_group['features'].items():
                    if isinstance(feature_value, (int, float)):
                        print(f"  {feature_name}: {feature_value:.4f}")
                    else:
                        print(f"  {feature_name}: {feature_value}")
        except Exception as e:
            print(f"Error in feature grouping demonstration: {e}")
    
    print("\\nDemo completed!")
    print("\\nRSN Features Explained:")
    print("- rsn_gflops: Computational complexity in GFLOPS")
    print("- rsn_ap: Average Precision for keypoint detection")
    print("- rsn_ap50: AP at IoU=0.50")
    print("- rsn_ap75: AP at IoU=0.75")
    print("- rsn_apm: AP for medium objects")
    print("- rsn_apl: AP for large objects")
    print("- rsn_ar_head: Average Recall for head keypoints")
    print("- rsn_shoulder, rsn_elbow, rsn_wrist: Upper body keypoint accuracy")
    print("- rsn_hip, rsn_knee, rsn_ankle: Lower body keypoint accuracy")
    print("- rsn_mean: Overall mean performance metric")


if __name__ == "__main__":
    main()
