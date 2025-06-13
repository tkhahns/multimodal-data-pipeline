#!/usr/bin/env python3
# filepath: /Users/evidenceb/Desktop/multimodal-data-pipeline/run_simple.py
"""
Main entry point for the multimodal data pipeline.
It can be run directly with Poetry without activating a shell:
    poetry run python run_simple.py [options]
"""
import os
import sys
import argparse
import logging
import traceback
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
    parser.add_argument(
        "--log-file", 
        help="Path to log file (default: <output_dir>/pipeline.log)"
    )
    parser.add_argument(
        "--check-dependencies", action="store_true", 
        help="Check if all required dependencies are installed"
    )
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_dependencies:
        try:
            import importlib
            required_packages = {
                'numpy': 'numpy',
                'pandas': 'pandas',
                'librosa': 'librosa',
                'opencv-python': 'cv2',
                'torch': 'torch',
                'torchaudio': 'torchaudio',
                'ffmpeg-python': 'ffmpeg',
                'soundfile': 'soundfile',
                'pyarrow': 'pyarrow',
                'transformers': 'transformers'
            }
            missing = []
            print("Checking required dependencies:")
            
            for package_name, import_name in required_packages.items():
                try:
                    importlib.import_module(import_name)
                    print(f"  ✓ {package_name}")
                except ImportError:
                    missing.append(package_name)
                    print(f"  ✗ {package_name} - MISSING")
                    
            if missing:
                print("\nSome dependencies are missing. Please run: ./setup_env.sh")
                print("Or install the missing packages with: poetry add " + " ".join(missing))
                sys.exit(1)
            else:
                print("\nAll dependencies are installed correctly!")
                sys.exit(0)
        except Exception as e:
            print(f"Error checking dependencies: {e}")
            sys.exit(1)
    
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
        print("  opensmile        - OpenSMILE Low-Level Descriptors and Functionals")
        print("  audiostretchy    - AudioStretchy time-stretching analysis")
        print("  speech_emotion   - Speech emotion recognition")
        print("  heinsen_sentiment - Heinsen routing sentiment analysis")
        print("  meld_emotion     - MELD emotion recognition during social interactions")
        print("  speech_separation - Speech source separation")
        print("  whisperx_transcription - WhisperX transcription with diarization")
        print("  deberta_text     - DeBERTa text analysis with benchmark performance")
        print("  simcse_text      - SimCSE contrastive learning of sentence embeddings")
        print("  albert_text      - ALBERT language representation analysis")
        print("  sbert_text       - Sentence-BERT dense vector representations")
        print("  use_text         - Universal Sentence Encoder text analysis")
        print("")
        print("Note: When no features are specified, all features are extracted by default.")
        sys.exit(0)
    
    # Parse features
    features = None
    if args.features:
        features = args.features.split(",")
    
    # Set up output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = "output"  # Use a fixed output directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = args.log_file if args.log_file else os.path.join(output_dir, "pipeline.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Multimodal Pipeline started at {datetime.now()}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Features to extract: {features if features else 'all available'}")
    
    # Check if ffmpeg is available
    import shutil
    if not shutil.which("ffmpeg"):
        logging.error("Error: ffmpeg is not installed or not in PATH")
        logging.error("Please install ffmpeg before running this script")
        sys.exit(1)
    
    # Initialize the pipeline
    try:
        pipeline = MultimodalPipeline(
            output_dir=output_dir,
            features=features,
            device="cpu"  # Use 'cuda' if GPU is available
        )
        
        # Check for CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                logging.info("CUDA is available! GPU will be used if supported by models.")
        except (ImportError, AttributeError):
            logging.warning("Could not check CUDA availability. Using CPU.")
    
        # Check if the data directory exists
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logging.error(f"Error: Data directory {data_dir} does not exist.")
            sys.exit(1)
        
        # Process the data directory
        logging.info(f"Processing directory: {data_dir}")
        results = pipeline.process_directory(data_dir, is_video=(not args.is_audio))
        
        # Print results summary
        logging.info(f"\nSuccessfully processed {len(results)} files:")
        for filename in results:
            logging.info(f"  - {filename}")
        logging.info(f"\nResults saved to: {output_dir}")
        logging.info(f"Log file saved to: {log_file}")
        logging.info(f"Features JSON: {os.path.join(output_dir, 'pipeline_features.json')}")
        print(f"\nResults saved to: {output_dir}")
        print(f"Log file saved to: {log_file}")
        print(f"Features JSON: {os.path.join(output_dir, 'pipeline_features.json')}")
        
    except Exception as e:
        logging.error(f"Error processing files: {e}")
        logging.error(traceback.format_exc())
        print(f"Error processing files: {e}")
        print("Check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
