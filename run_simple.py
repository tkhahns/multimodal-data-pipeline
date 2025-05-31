#!/usr/bin/env python3
# filepath: /Users/evidenceb/Desktop/multimodal-data-pipeline/run_simple.py
"""
Simple entry point for the multimodal data pipeline.
This script can be run directly with Poetry without activating a shell:
    poetry run python run_simple.py [options]
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Multimodal Data Pipeline")
    parser.add_argument(
        "-d", "--data-dir", default="data", help="Directory with video/audio files"
    )
    parser.add_argument(
        "-o", "--output-dir", help="Output directory (default: ./output/YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "-f", "--features", 
        help="Comma-separated features to extract (default: all supported features)"
    )
    parser.add_argument(
        "--list-features", action="store_true", help="List available features and exit"
    )
    parser.add_argument(
        "--is-audio", action="store_true", 
        help="Process files as audio instead of video"
    )
    args = parser.parse_args()
    
    # Try importing the pipeline module
    try:
        from src.pipeline import MultimodalPipeline
    except ImportError:
        print("Error: Could not import MultimodalPipeline.")
        print("Make sure dependencies are installed by running: ./setup_env.sh")
        sys.exit(1)
        
    # List features if requested
    if args.list_features:
        print("Available features:")
        print("  basic_audio      - Volume and pitch using OpenCV")
        print("  librosa_spectral - Spectral features using Librosa")
        print("  speech_emotion   - Speech emotion recognition")
        print("  speech_separation - Speech source separation")
        print("  whisperx         - WhisperX transcription with diarization")
        print("  xlsr_speech      - XLSR speech-to-text")
        print("  s2t_speech       - S2T speech-to-text")
        sys.exit(0)
    
    # Parse features
    features = None
    if args.features:
        features = args.features.split(",")
    
    # Set up output directory
    output_dir = args.output_dir
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", timestamp)
    
    # Initialize the pipeline
    pipeline = MultimodalPipeline(
        output_dir=output_dir,
        features=features,
        device="cpu"  # Use 'cuda' if GPU is available
    )
    
    # Check if the data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist.")
        sys.exit(1)
    
    # Process the data directory
    print(f"Processing directory: {data_dir}")
    try:
        results = pipeline.process_directory(data_dir, is_video=(not args.is_audio))
        
        # Print results summary
        print(f"\nSuccessfully processed {len(results)} files:")
        for filename in results:
            print(f"  - {filename}")
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        import traceback
        print(f"Error processing files: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
