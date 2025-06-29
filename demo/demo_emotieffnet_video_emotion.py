#!/usr/bin/env python3
"""
Demo script for EmotiEffNet real-time video emotion analysis and AU detection.

This script demonstrates how to use the EmotiEffNet analyzer for:
- Frame-level prediction of facial expressions
- Valence and arousal estimation  
- Action Unit (AU) detection
- Real-time video emotion analysis

The EmotiEffNet model outputs:
- eln_arousal: Arousal level
- eln_valence: Valence level  
- eln_AU1, eln_AU2, eln_AU4, eln_AU6, eln_AU7, eln_AU10, eln_AU12, eln_AU15, eln_AU23, eln_AU24, eln_AU25, eln_AU26: Action Units
- eln_neutral_f1, eln_anger_f1, eln_disgust_f1, eln_fear_f1, eln_happiness_f1, eln_sadness_f1, eln_surprise_f1, eln_other_f1: Emotion F1 scores

Usage:
    python demo/demo_emotieffnet_video_emotion.py [video_file]
"""

import sys
import json
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.vision.emotieffnet_analyzer import EmotiEffNetAnalyzer

def demo_emotieffnet_analyzer(video_path: str = None):
    """
    Demonstrate EmotiEffNet analyzer functionality.
    
    Args:
        video_path: Path to video file (optional)
    """
    print("=" * 60)
    print("EmotiEffNet Real-time Video Emotion Analysis Demo")
    print("=" * 60)
    print()
    
    # Initialize the analyzer
    print("Initializing EmotiEffNet analyzer...")
    analyzer = EmotiEffNetAnalyzer(device='cpu')
    
    # Get default features
    print("\nDemonstrating default EmotiEffNet features:")
    default_features = analyzer.get_feature_dict()
    
    print("\nEmotiEffNet Output Features:")
    print("-" * 30)
    for key, value in default_features.items():
        if key.startswith('eln_'):
            print(f"{key}: {value}")
    
    # If video file provided, analyze it
    if video_path and Path(video_path).exists():
        print(f"\nAnalyzing video: {video_path}")
        print("-" * 40)
        
        try:
            video_features = analyzer.get_feature_dict(video_path)
            
            print("\nExtracted EmotiEffNet Features:")
            print("-" * 35)
            
            # Group features by category
            arousal_valence = {}
            action_units = {}
            emotions = {}
            
            for key, value in video_features.items():
                if key.startswith('eln_'):
                    if 'arousal' in key or 'valence' in key:
                        arousal_valence[key] = value
                    elif key.startswith('eln_AU'):
                        action_units[key] = value
                    elif '_f1' in key:
                        emotions[key] = value
            
            # Display arousal and valence
            if arousal_valence:
                print("\nArousal and Valence:")
                for key, value in arousal_valence.items():
                    print(f"  {key}: {value:.4f}")
            
            # Display action units
            if action_units:
                print("\nAction Units (AU):")
                for key, value in sorted(action_units.items()):
                    au_number = key.replace('eln_AU', '')
                    print(f"  AU{au_number}: {value:.4f}")
            
            # Display emotion F1 scores
            if emotions:
                print("\nEmotion F1 Scores:")
                for key, value in emotions.items():
                    emotion = key.replace('eln_', '').replace('_f1', '')
                    print(f"  {emotion.capitalize()}: {value:.4f}")
                    
        except Exception as e:
            print(f"Error analyzing video: {e}")
            print("This might be due to missing dependencies or video format issues.")
    
    else:
        if video_path:
            print(f"\nWarning: Video file not found: {video_path}")
        print("\nTo analyze a video file, provide the path as an argument:")
        print("python demo/demo_emotieffnet_video_emotion.py path/to/video.mp4")
    
    print("\n" + "=" * 60)
    print("EmotiEffNet Feature Summary:")
    print("=" * 60)
    print("✓ Arousal and valence estimation")
    print("✓ 12 Action Units (AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU15, AU23, AU24, AU25, AU26)")
    print("✓ 8 Emotion classifications (neutral, anger, disgust, fear, happiness, sadness, surprise, other)")
    print("✓ Frame-level analysis optimized for mobile devices")
    print("✓ Real-time emotion processing capability")
    print()

def main():
    """Main function to run the demo."""
    video_path = None
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    demo_emotieffnet_analyzer(video_path)

if __name__ == "__main__":
    main()
