#!/usr/bin/env python3
"""
Deep HRNet Pose Estimation Demo

This script demonstrates the usage of Deep HRNet analyzer for high-resolution pose estimation.
It shows how to analyze video files for pose landmarks with body part accuracy metrics and
COCO-style Average Precision (AP) and Average Recall (AR) evaluation.

Usage:
    python demo_deep_hrnet.py [video_path] [--device cpu/cuda] [--confidence 0.3]

Example:
    python demo_deep_hrnet.py sample_video.mp4 --device cpu --confidence 0.5
"""

import sys
import os
import argparse
from pathlib import Path
import json
import numpy as np

# Add the src directory to the path to import the analyzer
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from vision.deep_hrnet_analyzer import DeepHRNetAnalyzer, extract_deep_hrnet_features
    print("✅ Successfully imported Deep HRNet analyzer")
except ImportError as e:
    print(f"❌ Failed to import Deep HRNet analyzer: {e}")
    print("Make sure you're running this from the demo directory and that the src directory exists")
    sys.exit(1)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def analyze_deep_hrnet_features(features_dict: dict):
    """Analyze and display Deep HRNet features in detail."""
    
    print_section("DEEP HRNET POSE ESTIMATION ANALYSIS")
    
    for group_name, group_data in features_dict.items():
        if "Pose estimation (high-resolution)" in group_name:
            print(f"\n🎯 Feature Group: {group_name}")
            print(f"📋 Description: {group_data.get('description', 'N/A')}")
            
            features = group_data.get('features', {})
            
            # Display body part accuracy metrics
            print_subsection("Body Part Accuracy Metrics")
            body_parts = ['Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle']
            for part in body_parts:
                key = f'DHiR_{part}'
                if key in features:
                    accuracy = features[key]
                    status = "🟢 Good" if accuracy > 0.7 else "🟡 Fair" if accuracy > 0.4 else "🔴 Low"
                    print(f"  {part:>8}: {accuracy:6.3f} {status}")
            
            # Display overall accuracy metrics
            print_subsection("Overall Accuracy Metrics")
            overall_keys = ['DHiR_Mean', 'DHiR_Meanat0.1']
            for key in overall_keys:
                if key in features:
                    value = features[key]
                    label = "Mean Accuracy" if key == 'DHiR_Mean' else "Mean (>0.1 confidence)"
                    status = "🟢 Excellent" if value > 0.8 else "🟡 Good" if value > 0.6 else "🔴 Needs improvement"
                    print(f"  {label:>20}: {value:6.3f} {status}")
            
            # Display Average Precision (AP) metrics
            print_subsection("Average Precision (AP) Metrics")
            ap_metrics = {
                'DHiR_AP': 'Overall AP',
                'DHiR_AP_5': 'AP @ IoU=0.5',
                'DHiR_AP_75': 'AP @ IoU=0.75',
                'DHiR_AP_M': 'AP Medium',
                'DHiR_AP_L': 'AP Large'
            }
            for key, label in ap_metrics.items():
                if key in features:
                    value = features[key]
                    status = "🟢 Excellent" if value > 0.8 else "🟡 Good" if value > 0.6 else "🔴 Fair" if value > 0.3 else "⚫ Low"
                    print(f"  {label:>15}: {value:6.3f} {status}")
            
            # Display Average Recall (AR) metrics
            print_subsection("Average Recall (AR) Metrics")
            ar_metrics = {
                'DHiR_AR': 'Overall AR',
                'DHiR_AR_5': 'AR @ IoU=0.5',
                'DHiR_AR_75': 'AR @ IoU=0.75',
                'DHiR_AR_M': 'AR Medium',
                'DHiR_AR_L': 'AR Large'
            }
            for key, label in ar_metrics.items():
                if key in features:
                    value = features[key]
                    status = "🟢 Excellent" if value > 0.8 else "🟡 Good" if value > 0.6 else "🔴 Fair" if value > 0.3 else "⚫ Low"
                    print(f"  {label:>15}: {value:6.3f} {status}")
            
            # Display processing statistics
            print_subsection("Processing Statistics")
            stats_keys = [
                ('total_frames', 'Total Frames'),
                ('pose_detected_frames', 'Pose Detected'),
                ('detection_rate', 'Detection Rate'),
                ('avg_keypoints_per_frame', 'Avg Keypoints/Frame')
            ]
            
            for key, label in stats_keys:
                if key in features:
                    value = features[key]
                    if 'rate' in key:
                        print(f"  {label:>20}: {value:6.1%}")
                    elif 'avg' in key:
                        print(f"  {label:>20}: {value:6.1f}")
                    else:
                        print(f"  {label:>20}: {value}")

def demonstrate_deep_hrnet_analyzer(video_path: str, device: str = 'cpu', confidence: float = 0.3):
    """Demonstrate Deep HRNet analyzer functionality."""
    
    print_section("DEEP HRNET ANALYZER DEMONSTRATION")
    print(f"📁 Video: {video_path}")
    print(f"💻 Device: {device}")
    print(f"🎯 Confidence Threshold: {confidence}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    try:
        # Method 1: Using the high-level function
        print_subsection("Method 1: Using High-Level Function")
        print("🔄 Extracting Deep HRNet features using extract_deep_hrnet_features()...")
        
        features_dict = extract_deep_hrnet_features(video_path, device=device)
        print("✅ Feature extraction completed!")
        
        # Analyze the results
        analyze_deep_hrnet_features(features_dict)
        
        # Method 2: Using the analyzer class directly
        print_subsection("Method 2: Using Analyzer Class Directly")
        print("🔄 Initializing Deep HRNet analyzer...")
        
        analyzer = DeepHRNetAnalyzer(device=device, confidence_threshold=confidence)
        print("✅ Analyzer initialized!")
        
        print("🔄 Analyzing video for pose estimation...")
        results = analyzer.analyze_video(video_path)
        print("✅ Analysis completed!")
        
        # Display key metrics from direct analysis
        print("\n📊 Key Metrics from Direct Analysis:")
        if results:
            print(f"  • Mean Body Part Accuracy: {results.get('DHiR_Mean', 0):.3f}")
            print(f"  • Overall AP: {results.get('DHiR_AP', 0):.3f}")
            print(f"  • Overall AR: {results.get('DHiR_AR', 0):.3f}")
            print(f"  • Detection Rate: {results.get('detection_rate', 0):.1%}")
            print(f"  • Frames Processed: {results.get('total_frames', 0)}")
        
        # Save results to JSON file
        output_file = f"deep_hrnet_results_{Path(video_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(features_dict, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Deep HRNet Pose Estimation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_deep_hrnet.py sample_video.mp4
  python demo_deep_hrnet.py sample_video.mp4 --device cuda --confidence 0.5
        """
    )
    
    parser.add_argument(
        'video_path',
        nargs='?',
        default='sample_video.mp4',
        help='Path to the video file to analyze (default: sample_video.mp4)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to run inference on (default: cpu)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Confidence threshold for keypoint detection (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Deep HRNet Pose Estimation Demo")
    print("📖 This demo shows high-resolution pose estimation with AP/AR metrics")
    
    # Run the demonstration
    demonstrate_deep_hrnet_analyzer(args.video_path, args.device, args.confidence)
    
    print("\n" + "="*60)
    print("🎉 Demo completed!")
    print("📚 Learn more about Deep HRNet:")
    print("   https://github.com/leoxiaobin/deep-high-resolution-net.pytorch")
    print("="*60)

if __name__ == "__main__":
    main()
