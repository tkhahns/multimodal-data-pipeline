"""
Comprehensive feature extractor that combines all requested features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom feature extractors
from src.audio.advanced_features import extract_advanced_audio_features
from src.emotion.simple_emotion_recognition import get_emotion_features_dict
from src.speech.whisperx_transcriber import WhisperXTranscriber
from src.speech.speech_separator import SpeechSeparator
from src.utils.audio_extraction import extract_audio_from_video

logger = logging.getLogger(__name__)

class ComprehensiveFeatureExtractor:
    """
    Extract all requested features from audio files.
    """
    
    def __init__(self):
        # Load HuggingFace token from environment
        hf_token = os.getenv('HF_TOKEN')
        if hf_token and hf_token != 'your_token_here':
            print(f"Using HuggingFace token for speaker diarization")
        else:
            print("Warning: No HuggingFace token found. Speaker diarization may not work.")
            print("Please set HF_TOKEN in your .env file to enable diarization.")
            hf_token = None
            
        self.whisperx_transcriber = WhisperXTranscriber(hf_token=hf_token)
        self.speech_separator = SpeechSeparator()
    
    def extract_all_features(self, input_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Extract all requested features from an audio or video file.
        
        Args:
            input_path: Path to the audio or video file
            output_dir: Output directory for separated audio files
            
        Returns:
            Dict with all extracted features
        """
        try:
            features = {}
            
            # Check if input is a video file and extract audio if needed
            input_path = Path(input_path)
            video_extensions = ['.mp4', '.MP4', '.avi', '.mov', '.MOV', '.mkv']
            
            if input_path.suffix in video_extensions:
                print(f"Video file detected, extracting audio from: {input_path}")
                if output_dir:
                    audio_output_dir = Path(output_dir) / "audio"
                    os.makedirs(audio_output_dir, exist_ok=True)
                else:
                    audio_output_dir = input_path.parent / "output" / "audio"
                    os.makedirs(audio_output_dir, exist_ok=True)
                
                # Extract audio from video
                audio_path = extract_audio_from_video(
                    str(input_path), 
                    str(audio_output_dir), 
                    format="wav", 
                    sample_rate=16000
                )
                print(f"Audio extracted to: {audio_path}")
            else:
                # Input is already an audio file
                audio_path = str(input_path)
            
            print(f"Extracting comprehensive features from: {audio_path}")
            
            # 1. Extract advanced audio features (oc_audvol, oc_audpit, etc.)
            print("-> Extracting audio volume and pitch features...")
            audio_features = extract_advanced_audio_features(audio_path)
            features.update(audio_features)
            
            # 2. Extract emotion recognition features (ser_*)
            print("-> Extracting emotion recognition features...")
            emotion_features = get_emotion_features_dict(audio_path)
            features.update(emotion_features)
            
            # 3. Extract WhisperX transcription with speaker diarization
            print("-> Extracting WhisperX transcription and speaker features...")
            whisperx_features = self.whisperx_transcriber.get_feature_dict(audio_path)
            features.update(whisperx_features)
            
            # 4. Extract speech separation features
            if output_dir:
                print("-> Extracting speech separation features...")
                separation_features = self.speech_separator.get_feature_dict(audio_path, output_dir)
                features.update(separation_features)
            
            print(f"Successfully extracted {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive features from {input_path}: {e}")
            return {
                'error': str(e),
                'input_path': str(input_path)
            }

def extract_comprehensive_features(input_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to extract all comprehensive features.
    
    Args:
        input_path: Path to the audio or video file
        output_dir: Output directory for separated audio files
        
    Returns:
        Dict with all extracted features
    """
    extractor = ComprehensiveFeatureExtractor()
    return extractor.extract_all_features(input_path, output_dir)

if __name__ == "__main__":
    # Test the comprehensive feature extractor
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        features = extract_comprehensive_features(input_path, output_dir)
        
        print(f"\nComprehensive features for {input_path}:")
        print("=" * 60)
        
        # Group features by type for better readability
        audio_features = {k: v for k, v in features.items() if k.startswith('oc_aud')}
        emotion_features = {k: v for k, v in features.items() if k.startswith('ser_')}
        whisperx_features = {k: v for k, v in features.items() if k.startswith('WhX_')}
        separation_features = {k: v for k, v in features.items() if 'source' in k}
        other_features = {k: v for k, v in features.items() 
                         if not any(k.startswith(prefix) for prefix in ['oc_aud', 'ser_', 'WhX_']) 
                         and 'source' not in k}
        
        if audio_features:
            print("\nAudio Features:")
            for feature, value in audio_features.items():
                if isinstance(value, (list, tuple)) and len(value) > 10:
                    print(f"  {feature}: {len(value)} values (mean: {sum(value)/len(value):.4f})")
                else:
                    print(f"  {feature}: {value}")
        
        if emotion_features:
            print("\nEmotion Features:")
            for feature, value in emotion_features.items():
                print(f"  {feature}: {value:.4f}")
        
        if whisperx_features:
            print("\nWhisperX Speaker/Word Features:")
            for feature, value in list(whisperx_features.items())[:10]:  # Show first 10
                print(f"  {feature}: {str(value)[:100]}")
            if len(whisperx_features) > 10:
                print(f"  ... and {len(whisperx_features) - 10} more WhisperX features")
        
        if separation_features:
            print("\nSpeech Separation Features:")
            for feature, value in separation_features.items():
                print(f"  {feature}: {value}")
        
        if other_features:
            print("\nOther Features:")
            for feature, value in other_features.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {feature}: {value[:100]}...")
                else:
                    print(f"  {feature}: {value}")
                    
    else:
        print("Usage: python comprehensive_features.py <audio_or_video_file> [output_dir]")
