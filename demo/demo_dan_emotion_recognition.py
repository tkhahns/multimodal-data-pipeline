"""
Demo script for DAN (Distract Your Attention) emotional expression recognition feature extraction.

This script demonstrates how to use the DAN analyzer for facial emotion classification
using multi-head cross attention networks from video files.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pipeline import MultimodalPipeline


def main():
    """Main function to demonstrate DAN emotional expression recognition."""
    print("=" * 70)
    print("DAN Emotional Expression Recognition Demo")
    print("=" * 70)
    
    # Setup pipeline with only DAN vision features
    output_dir = Path("output") / "dan_demo"
    pipeline = MultimodalPipeline(
        output_dir=output_dir,
        features=["dan_vision"],  # Only extract DAN features
        device="cpu"  # Use CPU for demo (change to "cuda" if GPU available)
    )
    
    print(f"Pipeline initialized with output directory: {output_dir}")
    print("Features to extract: DAN emotional expression recognition")
    print()
    
    # Example video files to process (replace with actual video paths)
    video_files = [
        "sample_emotion_video.mp4",  # Replace with actual video path
        "test_facial_emotion.avi",   # Replace with actual video path
    ]
    
    # Filter to only existing files
    existing_files = [f for f in video_files if os.path.exists(f)]
    
    if not existing_files:
        print("No video files found. Please update the video_files list with actual video paths.")
        print("Example usage:")
        print("1. Place a video file with facial expressions (e.g., 'emotion_video.mp4') in the current directory")
        print("2. Update the video_files list above with the correct path")
        print("3. Run the script again")
        print()
        
        # Create a sample test using the DAN analyzer directly
        print("Creating sample DAN analysis...")
        demonstrate_dan_features()
        return
    
    # Process each video file
    for video_file in existing_files:
        print(f"Processing video: {video_file}")
        print("-" * 50)
        
        try:
            # Extract features from the video
            features = pipeline.process_video_file(video_file)
            
            # Display DAN emotion features
            print("DAN Emotional Expression Features:")
            print_dan_features(features)
            
            # Save results to JSON
            output_file = output_dir / f"{Path(video_file).stem}_dan_features.json"
            with open(output_file, 'w') as f:
                json.dump(features, f, indent=2)
            
            print(f"Features saved to: {output_file}")
            print()
            
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            print()


def print_dan_features(features):
    """Print DAN emotional expression features in a formatted way."""
    dan_features = {k: v for k, v in features.items() if k.startswith('dan_')}
    
    if not dan_features:
        print("No DAN features found in the extracted features.")
        return
    
    print("Individual Emotion Scores:")
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for emotion in emotions:
        key = f'dan_{emotion}'
        if key in dan_features:
            score = dan_features[key]
            bar = '█' * int(score * 20)  # Visual bar representation
            print(f"  {emotion.capitalize():>8}: {score:.4f} {bar}")
    
    # Find dominant emotion
    emotion_scores = {emotion: dan_features.get(f'dan_{emotion}', 0) for emotion in emotions}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    dominant_score = emotion_scores[dominant_emotion]
    
    print(f"\nDominant Emotion: {dominant_emotion.capitalize()} ({dominant_score:.4f})")
    
    # Show full emotion scores array if available
    if 'dan_emotion_scores' in dan_features:
        scores_array = dan_features['dan_emotion_scores']
        print(f"\nFull Emotion Scores Array: {scores_array}")
    
    print(f"\nTotal DAN features extracted: {len(dan_features)}")


def demonstrate_dan_features():
    """Demonstrate DAN features using the analyzer directly."""
    try:
        from src.vision.dan_analyzer import DANAnalyzer
        import numpy as np
        
        print("Initializing DAN analyzer...")
        analyzer = DANAnalyzer(device="cpu", num_classes=7)
        
        # Create a dummy frame for demonstration
        dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        print("Analyzing sample frame...")
        features = analyzer.analyze_frame(dummy_frame)
        
        print("Sample DAN Features (using random frame):")
        print_dan_features(features)
        
        print("\nNote: This is a demonstration with random data.")
        print("For real emotion recognition, provide actual video files with facial expressions.")
        
    except Exception as e:
        print(f"Error during DAN demonstration: {e}")


def analyze_emotion_distribution(features):
    """Analyze the distribution of emotions in the extracted features."""
    dan_features = {k: v for k, v in features.items() if k.startswith('dan_') and k != 'dan_emotion_scores'}
    
    if not dan_features:
        return
    
    print("\nEmotion Analysis:")
    print("-" * 30)
    
    # Sort emotions by score
    sorted_emotions = sorted(dan_features.items(), key=lambda x: x[1], reverse=True)
    
    print("Emotion Ranking (highest to lowest):")
    for i, (emotion_key, score) in enumerate(sorted_emotions, 1):
        emotion_name = emotion_key.replace('dan_', '').capitalize()
        confidence = "High" if score > 0.7 else "Medium" if score > 0.3 else "Low"
        print(f"  {i}. {emotion_name:>8}: {score:.4f} ({confidence} confidence)")
    
    # Calculate emotion diversity (entropy-like measure)
    scores = list(dan_features.values())
    if scores:
        # Normalize scores to probabilities
        total = sum(scores)
        if total > 0:
            probs = [s / total for s in scores]
            diversity = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            print(f"\nEmotion Diversity Score: {diversity:.4f}")
            print("(Higher values indicate more mixed emotions)")


if __name__ == "__main__":
    main()
