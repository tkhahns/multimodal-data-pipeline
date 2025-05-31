#!/bin/bash
# filepath: /Users/evidenceb/Desktop/multimodal-data-pipeline/run.sh
# Simple wrapper script to run the pipeline with Poetry

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is required but not found."
    echo "Please run ./setup_env.sh first."
    exit 1
fi

# Pass all arguments to the run_pipeline.sh script
echo "Running pipeline with Poetry..."

# Execute the script within Poetry's run command
poetry run python -c "
import sys
import os
from pathlib import Path
import subprocess

# Change to the script directory
script_dir = Path('${PWD}')
os.chdir(script_dir)

# Add the current directory to Python path
sys.path.append(str(script_dir))

try:
    # Run the pipeline
    from src.pipeline import MultimodalPipeline
    
    # Process command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Multimodal Data Pipeline')
    parser.add_argument('-d', '--data-dir', default='data', help='Directory with video/audio files')
    parser.add_argument('-o', '--output-dir', help='Output directory (default: ./output/YYYYMMDD_HHMMSS)')
    parser.add_argument('-f', '--features', help='Comma-separated features to extract')
    args, unknown = parser.parse_known_args()
    
    # Set up features list
    features = None
    if args.features:
        features = args.features.split(',')
    
    # Initialize pipeline
    pipeline = MultimodalPipeline(
        output_dir=args.output_dir, 
        features=features, 
        device='cpu'
    )
    
    # Process data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f'Error: Data directory {data_dir} does not exist.')
        sys.exit(1)
        
    print(f'Processing directory: {data_dir}')
    results = pipeline.process_directory(data_dir, is_video=True)
    
    # Print summary
    print(f'Successfully processed {len(results)} files')
    for filename in results:
        print(f'- {filename}')
        
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
    sys.exit(1)
" "$@"

# Check exit status
if [ $? -ne 0 ]; then
    echo "Pipeline execution failed."
    exit 1
else
    echo "Pipeline execution completed successfully."
fi
