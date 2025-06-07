"""
Main pipeline for processing audio files with all available features.
"""
import os
import json
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union
from datetime import datetime

from src.utils.audio_extraction import extract_audio_from_video, extract_audio_from_videos
from src.audio.basic_features import AudioFeatureExtractor
from src.audio.spectral_features import LibrosaFeatureExtractor
from src.speech.emotion_recognition import SpeechEmotionRecognizer
from src.speech.speech_separator import SpeechSeparator
from src.speech.whisperx_transcriber import WhisperXTranscriber
from src.features.comprehensive_features import ComprehensiveFeatureExtractor

class MultimodalPipeline:
    """Main pipeline for processing multimodal data."""
    
    def __init__(
        self,
        output_dir: Union[str, Path] = None,
        features: List[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize the multimodal pipeline.
        
        Args:
            output_dir: Directory to save output files
            features: List of features to extract (if None, extract all)
            device: Device to run models on ("cpu" or "cuda")
        """
        self.device = device
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = Path("output")
        else:
            self.output_dir = Path(output_dir)
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "audio", exist_ok=True)
        os.makedirs(self.output_dir / "features", exist_ok=True)
        
        # Initialize feature list
        all_features = [
            "basic_audio",      # Volume and pitch from OpenCV
            "librosa_spectral", # Spectral features from librosa
            "speech_emotion",   # Speech emotion recognition
            "speech_separation", # Speech source separation
            "whisperx_transcription", # WhisperX transcription with diarization
            "comprehensive"    # All advanced features (oc_audvol, ser_*, WhX_highlight_diarize_*)
        ]
        
        self.features = features if features is not None else all_features
        
        # Initialize feature extractors (lazily loaded later)
        self.extractors = {}
    
    def _get_extractor(self, feature_name: str) -> Any:
        """
        Get or initialize a feature extractor.
        
        Args:
            feature_name: Name of the feature extractor
            
        Returns:
            Any: The feature extractor object
        """
        if feature_name not in self.extractors:
            if feature_name == "basic_audio":
                self.extractors[feature_name] = AudioFeatureExtractor()
            elif feature_name == "librosa_spectral":
                self.extractors[feature_name] = LibrosaFeatureExtractor()
            elif feature_name == "speech_emotion":
                self.extractors[feature_name] = SpeechEmotionRecognizer()
            elif feature_name == "speech_separation":
                self.extractors[feature_name] = SpeechSeparator(device=self.device)
            elif feature_name == "whisperx_transcription":
                self.extractors[feature_name] = WhisperXTranscriber(device=self.device)
            elif feature_name == "comprehensive":
                self.extractors[feature_name] = ComprehensiveFeatureExtractor()
                
        return self.extractors.get(feature_name)
    
    def extract_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract all enabled features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Dictionary with all extracted features
        """
        features = {}
        
        # Extract basic audio features
        if "basic_audio" in self.features:
            print(f"Extracting basic audio features from {audio_path}")
            extractor = self._get_extractor("basic_audio")
            basic_features = extractor.extract_all_features(audio_path)
            features.update(basic_features)
        
        # Extract librosa spectral features
        if "librosa_spectral" in self.features:
            print(f"Extracting librosa spectral features from {audio_path}")
            extractor = self._get_extractor("librosa_spectral")
            spectral_features = extractor.extract_all_features(audio_path)
            features.update(spectral_features)
        
        # Extract speech emotion features
        if "speech_emotion" in self.features:
            print(f"Extracting speech emotion features from {audio_path}")
            extractor = self._get_extractor("speech_emotion")
            emotion_features = extractor.predict(audio_path)
            features.update(emotion_features)
        
        # Extract speech separation features
        if "speech_separation" in self.features:
            print(f"Extracting speech separation features from {audio_path}")
            extractor = self._get_extractor("speech_separation")
            sep_out_dir = self.output_dir / "audio" / "separated"
            os.makedirs(sep_out_dir, exist_ok=True)
            separation_features = extractor.get_feature_dict(audio_path, sep_out_dir)
            # Don't add the raw audio to features dict, just paths
            for key, val in separation_features.items():
                if "_path" in key:
                    features[key] = val
        
        # Extract comprehensive features (oc_audvol, ser_*, WhX_highlight_diarize_*)
        if "comprehensive" in self.features:
            print(f"Extracting comprehensive features from {audio_path}")
            extractor = self._get_extractor("comprehensive")
            comp_out_dir = self.output_dir / "audio" / "separated"
            os.makedirs(comp_out_dir, exist_ok=True)
            comprehensive_features = extractor.extract_all_features(audio_path, comp_out_dir)
            features.update(comprehensive_features)
        
        # Extract WhisperX features (separate from comprehensive for backward compatibility)
        if "whisperx_transcription" in self.features:
            print(f"Extracting WhisperX transcription features from {audio_path}")
            extractor = self._get_extractor("whisperx_transcription")
            # Limit maximum number of speakers to 3
            whisperx_features = extractor.get_feature_dict(audio_path, max_speakers=3)
            features.update(whisperx_features)
        
        return features
    
    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Process a single audio file through the pipeline.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Extracted features
        """
        audio_path = Path(audio_path)
        print(f"Processing audio file: {audio_path}")
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Extract features
        features = self.extract_features(str(audio_path))
        
        # Save features as JSON and/or parquet
        base_name = audio_path.stem
        feature_file_json = self.output_dir / "features" / f"{base_name}.json"
        
        # Create a JSON structure with the file name as the first key
        json_features = {base_name: {}}
        
        # Add metadata
        try:
            import librosa
            audio_data, sr = librosa.load(str(audio_path), sr=None)
            json_features[base_name]["metadata"] = {
                "filename": audio_path.name,
                "file_size_bytes": os.path.getsize(audio_path),
                "duration_seconds": len(audio_data) / sr,
                "sample_rate": sr,
                "channels": audio_data.shape[1] if len(audio_data.shape) > 1 else 1,
                "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "features_included": self.features
            }
        except Exception as e:
            json_features[base_name]["metadata"] = {
                "filename": audio_path.name,
                "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "features_included": self.features,
                "metadata_error": str(e)
            }
        
        # Process features for JSON
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                # Convert all arrays to JSON-compatible data with statistics for large arrays
                if value.size > 1000:
                    # Include statistics for large arrays
                    json_features[base_name][key] = {
                        'mean': float(np.mean(value)),
                        'min': float(np.min(value)),
                        'max': float(np.max(value)),
                        'std': float(np.std(value)),
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'samples': [float(x) if isinstance(x, (np.number, np.float32, np.float64)) else x for x in value[:10].tolist()] if value.size > 10 else [float(x) if isinstance(x, (np.number, np.float32, np.float64)) else x for x in value.tolist()]
                    }
                else:
                    # For smaller arrays, convert to list and include directly
                    if value.dtype.kind in 'fc':  # float or complex
                        json_features[base_name][key] = [float(x) for x in value.tolist()]
                    elif value.dtype.kind in 'iu':  # integer
                        json_features[base_name][key] = [int(x) for x in value.tolist()]
                    else:
                        json_features[base_name][key] = value.tolist()
            elif isinstance(value, (np.number, np.float32, np.float64, np.int32, np.int64)):
                # Convert numpy scalar types to native Python types
                json_features[base_name][key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
            elif isinstance(value, (str, int, float, bool, list, dict)):
                # Other Python native types go directly to JSON
                json_features[base_name][key] = value
        
        # Save a single JSON file
        with open(feature_file_json, "w") as f:
            json.dump(json_features, f, indent=2)
            
        return features
    
    def process_video_file(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file by extracting audio and then processing it.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict[str, Any]: Extracted features
        """
        video_path = Path(video_path)
        print(f"Processing video file: {video_path}")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract audio from video
        audio_output_dir = self.output_dir / "audio"
        audio_path = extract_audio_from_video(
            video_path, 
            audio_output_dir, 
            format="wav", 
            sample_rate=16000
        )
        
        # Process the audio
        return self.process_audio_file(audio_path)
    
    def process_directory(self, directory: Union[str, Path], is_video: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Process all files in a directory.
        
        Args:
            directory: Path to the directory containing files
            is_video: Whether the files are videos (True) or audio (False)
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping filenames to their features
        """
        directory = Path(directory)
        print(f"Processing directory: {directory}")
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        # Find files
        extensions = [".mp4", ".MP4", ".avi", ".mov", ".MOV", ".mkv"] if is_video else [".wav", ".mp3", ".flac"]
        files = []
        for ext in extensions:
            files.extend(list(directory.glob(f"*{ext}")))
        
        if not files:
            raise FileNotFoundError(f"No {'video' if is_video else 'audio'} files found in {directory}")
        
        # Process each file
        results = {}
        for file_path in files:
            try:
                if is_video:
                    features = self.process_video_file(file_path)
                else:
                    features = self.process_audio_file(file_path)
                    
                results[file_path.name] = features
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Create a consolidated JSON file with features from all files
        try:
            consolidated_json = {}
            for filename, file_features in results.items():
                consolidated_json[filename] = {}
                
                # Add all serializable features
                for key, value in file_features.items():
                    if isinstance(value, np.ndarray):
                        if value.size > 1000:
                            # Include statistics for large arrays
                            consolidated_json[filename][key] = {
                                'mean': float(np.mean(value)),
                                'min': float(np.min(value)),
                                'max': float(np.max(value)),
                                'std': float(np.std(value)),
                                'shape': list(value.shape),
                                'dtype': str(value.dtype),
                                'samples': [float(x) if isinstance(x, (np.number, np.float32, np.float64)) else x for x in value[:5].tolist()] if value.size > 5 else [float(x) if isinstance(x, (np.number, np.float32, np.float64)) else x for x in value.tolist()]
                            }
                        else:
                            # Convert numpy values to Python native types
                            if value.dtype.kind in 'fc':  # float or complex
                                consolidated_json[filename][key] = [float(x) for x in value.tolist()]
                            elif value.dtype.kind in 'iu':  # integer
                                consolidated_json[filename][key] = [int(x) for x in value.tolist()]
                            else:
                                consolidated_json[filename][key] = value.tolist()
                    elif not callable(value):
                        # Handle other numpy types that might be scalars
                        if isinstance(value, (np.number, np.float32, np.float64, np.int32, np.int64)):
                            consolidated_json[filename][key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                        else:
                            consolidated_json[filename][key] = value
            
            # Save consolidated JSON
            with open(self.output_dir / "pipeline_features.json", "w") as f:
                json.dump(consolidated_json, f, indent=2)
                
            print(f"Consolidated features saved to {self.output_dir / 'pipeline_features.json'}")
                
        except Exception as e:
            print(f"Warning: Could not save consolidated JSON: {e}")
            traceback.print_exc()
        
        return results
