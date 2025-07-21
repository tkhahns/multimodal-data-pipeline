"""
Demo script for ME-GraphAU facial action unit (AU) recognition feature extraction.

This script demonstrates how to use the ME-GraphAU analyzer for facial action unit
recognition using AU relation graphs from video files.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline import MultimodalPipeline


def main():
    """Main function to demonstrate ME-GraphAU facial action unit recognition."""
    print("=" * 70)
    print("ME-GraphAU Facial Action Unit Recognition Demo")
    print("=" * 70)
    
    # Setup pipeline with only ME-GraphAU vision features
    output_dir = Path("output") / "me_graphau_demo"
    pipeline = MultimodalPipeline(
        output_dir=output_dir,
        features=["me_graphau_vision"],  # Only extract ME-GraphAU features
        device="cpu"  # Use CPU for demo (change to "cuda" if GPU available)
    )
    
    print(f"Pipeline initialized with output directory: {output_dir}")
    print("Features to extract: ME-GraphAU facial action unit recognition")
    print()
    
    # Example video files to process (replace with actual video paths)
    video_files = [
        "sample_face_video.mp4",  # Replace with actual video path
        "test_facial_video.avi",  # Replace with actual video path
    ]
    
    # Filter to only existing files
    existing_files = [f for f in video_files if os.path.exists(f)]
    
    if not existing_files:
        print("No video files found. Please update the video_files list with actual video paths.")
        print("Example usage:")
        print("1. Place a video file with facial expressions (e.g., 'face_video.mp4') in the current directory")
        print("2. Update the video_files list above with the correct path")
        print("3. Run this script again")
        return
    
    # Process each video file
    for video_path in existing_files:
        print(f"Processing video: {video_path}")
        print("-" * 50)
        
        try:
            # Process the video file
            features = pipeline.process_video_file(video_path)
            
            # Display extracted ME-GraphAU features
            print("ME-GraphAU Facial Action Unit Features:")
            print()
            
            # BP4D dataset features
            print("BP4D Dataset Action Units:")
            bp4d_features = []
            for feature_name, feature_value in features.items():
                if feature_name.startswith("ann_") and feature_name.endswith("_bp4d"):
                    bp4d_features.append((feature_name, feature_value))
            
            for feature_name, feature_value in sorted(bp4d_features):
                if isinstance(feature_value, (int, float)):
                    print(f"  {feature_name}: {feature_value:.4f}")
                else:
                    print(f"  {feature_name}: {feature_value}")
            
            print()
            
            # DISFA dataset features
            print("DISFA Dataset Action Units:")
            disfa_features = []
            for feature_name, feature_value in features.items():
                if feature_name.startswith("ann_") and feature_name.endswith("_dis"):
                    disfa_features.append((feature_name, feature_value))
            
            for feature_name, feature_value in sorted(disfa_features):
                if isinstance(feature_value, (int, float)):
                    print(f"  {feature_name}: {feature_value:.4f}")
                else:
                    print(f"  {feature_name}: {feature_value}")
            
            print(f"\\nFeatures saved to: {output_dir}")
            print("=" * 70)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            print("=" * 70)
    
    # Demonstrate feature grouping
    print("\\nFeature Grouping Example:")
    print("-" * 40)
    
    if existing_files:
        try:
            features = pipeline.process_video_file(existing_files[0])
            grouped_features = pipeline._group_features_by_model(features)
            
            if "Facial action, AU relation graph" in grouped_features:
                au_group = grouped_features["Facial action, AU relation graph"]
                print(f"Feature Category: {au_group['Feature']}")
                print(f"Model: {au_group['Model']}")
                print(f"Number of features: {len(au_group['features'])}")
                print("Feature details:")
                
                # Separate BP4D and DISFA features for better display
                bp4d_feats = {k: v for k, v in au_group['features'].items() if k.endswith('_bp4d')}
                disfa_feats = {k: v for k, v in au_group['features'].items() if k.endswith('_dis')}
                
                if bp4d_feats:
                    print("  BP4D Features:")
                    for feature_name, feature_value in sorted(bp4d_feats.items()):
                        if isinstance(feature_value, (int, float)):
                            print(f"    {feature_name}: {feature_value:.4f}")
                        else:
                            print(f"    {feature_name}: {feature_value}")
                
                if disfa_feats:
                    print("  DISFA Features:")
                    for feature_name, feature_value in sorted(disfa_feats.items()):
                        if isinstance(feature_value, (int, float)):
                            print(f"    {feature_name}: {feature_value:.4f}")
                        else:
                            print(f"    {feature_name}: {feature_value}")
        except Exception as e:
            print(f"Error in feature grouping demonstration: {e}")
    
    print("\\nDemo completed!")
    print("\\nME-GraphAU Features Explained:")
    print("BP4D Dataset (12 Action Units):")
    print("- ann_AU1_bp4d to ann_AU24_bp4d: Individual action unit predictions")
    print("- ann_avg_bp4d: Average accuracy across all BP4D action units")
    print("\\nDISFA Dataset (8 Action Units):")
    print("- ann_AU1_dis to ann_AU26_dis: Individual action unit predictions")
    print("- ann_avg_dis: Average accuracy across all DISFA action units")
    print("\\nAction Units correspond to specific facial movements:")
    print("- AU1: Inner brow raiser, AU2: Outer brow raiser")
    print("- AU4: Brow lowerer, AU6: Cheek raiser")
    print("- AU7: Lid tightener, AU9: Nose wrinkler")
    print("- AU10: Upper lip raiser, AU12: Lip corner puller")
    print("- AU14: Dimpler, AU15: Lip corner depressor")
    print("- AU17: Chin raiser, AU23: Lip tightener")
    print("- AU24: Lip pressor, AU25: Lips part")
    print("- AU26: Jaw drop")


if __name__ == "__main__":
    main()
