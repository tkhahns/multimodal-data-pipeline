"""
Example script showing how to use the multimodal data pipeline.
"""
import os
import argparse
from pathlib import Path
from src.pipeline import MultimodalPipeline
from src.utils.file_utils import ensure_dir

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Example script for the multimodal data pipeline')
    parser.add_argument('--video', help='Path to a video file')
    parser.add_argument('--audio', help='Path to an audio file')
    parser.add_argument('--data-dir', help='Directory containing video files')
    parser.add_argument('--output-dir', default='output/example', help='Output directory')
    parser.add_argument('--features', default='basic_audio,librosa_spectral',
                       help='Comma-separated list of features to extract')
    args = parser.parse_args()
    
    # Convert features string to list
    features = args.features.split(',')
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Initialize pipeline
    pipeline = MultimodalPipeline(
        output_dir=output_dir,
        features=features,
        device='cpu'  # Use 'cuda' if GPU is available
    )
    
    # Process input based on provided arguments
    if args.video:
        print(f"Processing video file: {args.video}")
        results = pipeline.process_video_file(args.video)
        print(f"Results saved to {output_dir}")
        
    elif args.audio:
        print(f"Processing audio file: {args.audio}")
        results = pipeline.process_audio_file(args.audio)
        print(f"Results saved to {output_dir}")
        
    elif args.data_dir:
        print(f"Processing all videos in directory: {args.data_dir}")
        results = pipeline.process_directory(args.data_dir, is_video=True)
        print(f"Processed {len(results)} files. Results saved to {output_dir}")
        
    else:
        # Use default data directory
        default_data_dir = Path(__file__).parent.parent / 'data'
        if default_data_dir.exists() and any(default_data_dir.iterdir()):
            print(f"Processing all videos in default directory: {default_data_dir}")
            results = pipeline.process_directory(default_data_dir, is_video=True)
            print(f"Processed {len(results)} files. Results saved to {output_dir}")
        else:
            print("No input provided and default data directory is empty or doesn't exist.")
            print("Please specify --video, --audio, or --data-dir.")
            return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
