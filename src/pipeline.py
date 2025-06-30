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
from src.audio.opensmile_features import OpenSMILEFeatureExtractor
from src.speech.emotion_recognition import SpeechEmotionRecognizer
from src.emotion.heinsen_routing_sentiment import AudioSentimentAnalyzer
from src.speech.speech_separator import SpeechSeparator
from src.speech.whisperx_transcriber import WhisperXTranscriber
from src.text.deberta_analyzer import DeBERTaAnalyzer

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
            "opensmile",        # OpenSMILE Low-Level Descriptors and Functionals
            "audiostretchy",    # AudioStretchy high-quality time-stretching analysis
            "speech_emotion",   # Speech emotion recognition
            "heinsen_sentiment", # Heinsen routing sentiment analysis
            "meld_emotion",     # MELD emotion recognition during social interactions
            "speech_separation", # Speech source separation
            "whisperx_transcription", # WhisperX transcription with diarization
            "deberta_text",     # DeBERTa text analysis with benchmark performance metrics
            "simcse_text",      # SimCSE contrastive learning of sentence embeddings
            "albert_text",      # ALBERT language representation analysis
            "sbert_text",       # Sentence-BERT dense vector representations and reranking            "use_text",         # Universal Sentence Encoder for text classification and semantic analysis            "emotieffnet_vision", # EmotiEffNet real-time video emotion analysis and AU detection
            "mediapipe_pose_vision", # Google MediaPipe pose estimation and tracking with 33 landmarks            "deep_hrnet_vision", # Deep High-Resolution Network for high-precision pose estimation            "simple_baselines_vision", # Simple Baselines for human pose estimation and tracking            "pyfeat_vision",    # Py-Feat facial expression analysis with action units and emotions            "ganimation_vision", # GANimation continuous manifold for anatomical facial movements            "arbex_vision",     # ARBEx attentive feature extraction with reliability balancing for robust facial expression learning            "openpose_vision",  # OpenPose real-time multi-person keypoint detection and pose estimation            "instadm_vision",   # Insta-DM instant dense monocular depth estimation with motion analysis            "optical_flow_vision", # Optical Flow movement and estimation of motion with sparse and dense analysis
            "crowdflow_vision", # CrowdFlow optical flow fields, person trajectories, and tracking accuracy
            "videofinder_vision", # VideoFinder object and people localization with consistency and match metrics
            "smoothnet_vision", # SmoothNet temporally consistent 3D and 2D human pose estimation with neural smoothing
            "pare_vision",      # PARE 3D human body estimation and pose analysis
            "vitpose_vision",   # ViTPose Vision Transformer pose estimation
            "rsn_vision",       # RSN Residual Steps Network keypoint localization
            "me_graphau_vision", # ME-GraphAU facial action unit recognition
            "dan_vision",       # DAN emotional expression recognition
        ]
        # Default behavior: extract all features when none specified
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
            elif feature_name == "opensmile":
                self.extractors[feature_name] = OpenSMILEFeatureExtractor()
            elif feature_name == "audiostretchy":
                from src.audio.audiostretchy_features import AudioStretchyAnalyzer
                self.extractors[feature_name] = AudioStretchyAnalyzer()
            elif feature_name == "speech_emotion":
                self.extractors[feature_name] = SpeechEmotionRecognizer()
            elif feature_name == "heinsen_sentiment":
                self.extractors[feature_name] = AudioSentimentAnalyzer(device=self.device)
            elif feature_name == "meld_emotion":
                from src.emotion.meld_emotion_analyzer import MELDEmotionAnalyzer
                self.extractors[feature_name] = MELDEmotionAnalyzer()
            elif feature_name == "speech_separation":
                self.extractors[feature_name] = SpeechSeparator(device=self.device)            
            elif feature_name == "whisperx_transcription":
                from src.speech.whisperx_transcriber import WhisperXTranscriber
                self.extractors[feature_name] = WhisperXTranscriber(device=self.device)
            elif feature_name == "deberta_text":
                self.extractors[feature_name] = DeBERTaAnalyzer(device=self.device)
            elif feature_name == "simcse_text":
                from src.text.simcse_analyzer import SimCSEAnalyzer
                self.extractors[feature_name] = SimCSEAnalyzer(device=self.device)
            elif feature_name == "albert_text":
                from src.text.albert_analyzer import ALBERTAnalyzer
                self.extractors[feature_name] = ALBERTAnalyzer(device=self.device)
            elif feature_name == "sbert_text":
                from src.text.sbert_analyzer import SBERTAnalyzer
                self.extractors[feature_name] = SBERTAnalyzer(device=self.device)
            elif feature_name == "use_text":
                from src.text.use_analyzer import USEAnalyzer
                self.extractors[feature_name] = USEAnalyzer(device=self.device)
            elif feature_name == "emotieffnet_vision":
                from src.vision.emotieffnet_analyzer import EmotiEffNetAnalyzer
                self.extractors[feature_name] = EmotiEffNetAnalyzer(device=self.device)
            elif feature_name == "mediapipe_pose_vision":
                from src.vision.mediapipe_pose_analyzer import MediaPipePoseAnalyzer
                self.extractors[feature_name] = MediaPipePoseAnalyzer(device=self.device)
            elif feature_name == "deep_hrnet_vision":
                from src.vision.deep_hrnet_analyzer import DeepHRNetAnalyzer
                self.extractors[feature_name] = DeepHRNetAnalyzer(device=self.device)
            elif feature_name == "simple_baselines_vision":
                from src.vision.simple_baselines_analyzer import SimpleBaselinesAnalyzer
                self.extractors[feature_name] = SimpleBaselinesAnalyzer(device=self.device)
            elif feature_name == "pyfeat_vision":
                from src.vision.pyfeat_analyzer import PyFeatAnalyzer
                self.extractors[feature_name] = PyFeatAnalyzer(device=self.device)
            elif feature_name == "ganimation_vision":
                from src.vision.ganimation_analyzer import GANimationAnalyzer
                self.extractors[feature_name] = GANimationAnalyzer(device=self.device)
            elif feature_name == "arbex_vision":
                from src.vision.arbex_analyzer import ARBExAnalyzer
                self.extractors[feature_name] = ARBExAnalyzer(device=self.device)
            elif feature_name == "openpose_vision":
                from src.vision.openpose_analyzer import OpenPoseAnalyzer
                self.extractors[feature_name] = OpenPoseAnalyzer(device=self.device)
            elif feature_name == "instadm_vision":
                from src.vision.instadm_analyzer import InstaDMAnalyzer
                self.extractors[feature_name] = InstaDMAnalyzer(device=self.device)            
            elif feature_name == "optical_flow_vision":
                from src.vision.optical_flow_analyzer import OpticalFlowAnalyzer
                self.extractors[feature_name] = OpticalFlowAnalyzer(device=self.device)
            elif feature_name == "crowdflow_vision":
                from src.vision.crowdflow_analyzer import CrowdFlowAnalyzer
                self.extractors[feature_name] = CrowdFlowAnalyzer(device=self.device)
            elif feature_name == "videofinder_vision":
                from src.vision.videofinder_analyzer import VideoFinderAnalyzer
                self.extractors[feature_name] = VideoFinderAnalyzer(device=self.device)
            elif feature_name == "smoothnet_vision":
                from src.vision.smoothnet_analyzer import SmoothNetAnalyzer
                self.extractors[feature_name] = SmoothNetAnalyzer(device=self.device)
            elif feature_name == "pare_vision":
                from src.vision.pare_analyzer import PAREAnalyzer
                self.extractors[feature_name] = PAREAnalyzer(device=self.device)
            elif feature_name == "vitpose_vision":
                from src.vision.vitpose_analyzer import ViTPoseAnalyzer
                self.extractors[feature_name] = ViTPoseAnalyzer(device=self.device)
            elif feature_name == "rsn_vision":
                from src.vision.rsn_analyzer import RSNAnalyzer
                self.extractors[feature_name] = RSNAnalyzer(device=self.device)
            elif feature_name == "me_graphau_vision":
                from src.vision.me_graphau_analyzer import MEGraphAUAnalyzer
                self.extractors[feature_name] = MEGraphAUAnalyzer(device=self.device)
            elif feature_name == "dan_vision":
                from src.vision.dan_analyzer import DANAnalyzer
                self.extractors[feature_name] = DANAnalyzer(device=self.device)
            elif feature_name == "psa_vision":
                from src.vision.psa_analyzer import PSAAnalyzer
                self.extractors[feature_name] = PSAAnalyzer(device=self.device)
                
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
        
        # Extract OpenSMILE features
        if "opensmile" in self.features:
            print(f"Extracting OpenSMILE features from {audio_path}")
            extractor = self._get_extractor("opensmile")
            opensmile_features = extractor.get_feature_dict(audio_path)
            features.update(opensmile_features)
        
        # Extract AudioStretchy features
        if "audiostretchy" in self.features:
            print(f"Extracting AudioStretchy time-stretching features from {audio_path}")
            extractor = self._get_extractor("audiostretchy")
            audiostretchy_features = extractor.get_feature_dict(audio_path)
            features.update(audiostretchy_features)
        
        # Extract speech emotion features
        if "speech_emotion" in self.features:
            print(f"Extracting speech emotion features from {audio_path}")
            extractor = self._get_extractor("speech_emotion")
            emotion_features = extractor.predict(audio_path)
            features.update(emotion_features)
         # Extract Heinsen routing sentiment features
        if "heinsen_sentiment" in self.features:
            print(f"Extracting Heinsen routing sentiment features from {audio_path}")
            extractor = self._get_extractor("heinsen_sentiment")
            sentiment_features = extractor.get_feature_dict(features)
            features.update(sentiment_features)

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

        # Extract WhisperX features
        if "whisperx_transcription" in self.features:
            print(f"Extracting WhisperX transcription features from {audio_path}")
            extractor = self._get_extractor("whisperx_transcription")
            # Limit maximum number of speakers to 3
            whisperx_features = extractor.get_feature_dict(audio_path, max_speakers=3)
            features.update(whisperx_features)

        # Extract MELD emotion features (after WhisperX to access transcription)
        if "meld_emotion" in self.features:
            print(f"Extracting MELD emotion recognition features from {audio_path}")
            extractor = self._get_extractor("meld_emotion")
            # Pass the entire feature dictionary to MELD analyzer 
            # It will look for transcribed text from WhisperX or other sources
            meld_features = extractor.get_feature_dict(features)
            features.update(meld_features)

        # Extract DeBERTa text analysis features
        if "deberta_text" in self.features:
            print(f"Extracting DeBERTa text analysis features from {audio_path}")
            extractor = self._get_extractor("deberta_text")
            # Pass the entire feature dictionary to DeBERTa analyzer 
            # It will look for transcribed text from WhisperX or other sources
            deberta_features = extractor.get_feature_dict(features)
            features.update(deberta_features)

        # Extract SimCSE text analysis features
        if "simcse_text" in self.features:
            print(f"Extracting SimCSE text analysis features from {audio_path}")
            extractor = self._get_extractor("simcse_text")
            # Pass the entire feature dictionary to SimCSE analyzer 
            # It will look for transcribed text from WhisperX or other sources
            simcse_features = extractor.get_feature_dict(features)
            features.update(simcse_features)

        # Extract ALBERT text analysis features
        if "albert_text" in self.features:
            print(f"Extracting ALBERT text analysis features from {audio_path}")
            extractor = self._get_extractor("albert_text")
            # Pass the entire feature dictionary to ALBERT analyzer 
            # It will look for transcribed text from WhisperX or other sources
            albert_features = extractor.get_feature_dict(features)
            features.update(albert_features)

        # Extract Sentence-BERT text analysis features
        if "sbert_text" in self.features:
            print(f"Extracting Sentence-BERT text analysis features from {audio_path}")
            extractor = self._get_extractor("sbert_text")
            # Pass the entire feature dictionary to Sentence-BERT analyzer 
            # It will look for transcribed text from WhisperX or other sources
            sbert_features = extractor.get_feature_dict(features)
            features.update(sbert_features)

        # Extract Universal Sentence Encoder text analysis features
        if "use_text" in self.features:
            print(f"Extracting Universal Sentence Encoder text analysis features from {audio_path}")
            extractor = self._get_extractor("use_text")
            # Pass the entire feature dictionary to USE analyzer 
            # It will look for transcribed text from WhisperX or other sources
            use_features = extractor.get_feature_dict(features)
            features.update(use_features)

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
        
        # Group features by model categories
        grouped_features = self._group_features_by_model(features)
        
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
        
        # Add grouped features
        for group_name, group_data in grouped_features.items():
            json_features[base_name][group_name] = {
                "Feature": group_data["Feature"],
                "Model": group_data["Model"],
                "features": {}
            }
            
            # Process features for JSON
            for key, value in group_data["features"].items():
                if isinstance(value, np.ndarray):
                    # Convert all arrays to JSON-compatible data with statistics for large arrays
                    if value.size > 1000:
                        # Include statistics for large arrays
                        json_features[base_name][group_name]["features"][key] = {
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
                            json_features[base_name][group_name]["features"][key] = [float(x) for x in value.tolist()]
                        elif value.dtype.kind in 'iu':  # integer
                            json_features[base_name][group_name]["features"][key] = [int(x) for x in value.tolist()]
                        else:
                            json_features[base_name][group_name]["features"][key] = value.tolist()
                elif isinstance(value, (np.number, np.float32, np.float64, np.int32, np.int64)):
                    # Convert numpy scalar types to native Python types
                    json_features[base_name][group_name]["features"][key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                elif isinstance(value, (str, int, float, bool, list, dict)):
                    # Other Python native types go directly to JSON
                    json_features[base_name][group_name]["features"][key] = value
        
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
          # Process the audio to get audio-based features
        features = self.extract_features(str(audio_path))
          # Extract EmotiEffNet vision features (video-specific)
        if "emotieffnet_vision" in self.features:
            print(f"Extracting EmotiEffNet real-time video emotion analysis and AU detection from {video_path}")
            extractor = self._get_extractor("emotieffnet_vision")
            emotieffnet_features = extractor.get_feature_dict(str(video_path))
            features.update(emotieffnet_features)
          # Extract MediaPipe pose features (video-specific)
        if "mediapipe_pose_vision" in self.features:
            print(f"Extracting MediaPipe pose estimation and tracking from {video_path}")
            extractor = self._get_extractor("mediapipe_pose_vision")
            mediapipe_features = extractor.get_feature_dict(str(video_path))
            features.update(mediapipe_features)
          # Extract Deep HRNet pose features (video-specific)
        if "deep_hrnet_vision" in self.features:
            print(f"Extracting Deep HRNet high-resolution pose estimation from {video_path}")
            extractor = self._get_extractor("deep_hrnet_vision")
            deep_hrnet_features = extractor.get_feature_dict(str(video_path))
            features.update(deep_hrnet_features)
          # Extract Simple Baselines pose features (video-specific)
        if "simple_baselines_vision" in self.features:
            print(f"Extracting Simple Baselines pose estimation and tracking from {video_path}")
            extractor = self._get_extractor("simple_baselines_vision")
            simple_baselines_features = extractor.get_feature_dict(str(video_path))
            features.update(simple_baselines_features)
          # Extract Py-Feat facial analysis features (video-specific)        if "pyfeat_vision" in self.features:
            print(f"Extracting Py-Feat facial expression analysis from {video_path}")
            extractor = self._get_extractor("pyfeat_vision")
            pyfeat_features = extractor.get_feature_dict(str(video_path))
            features.update(pyfeat_features)
        
        # Extract GANimation facial movement features (video-specific)
        if "ganimation_vision" in self.features:
            print(f"Extracting GANimation continuous manifold for anatomical facial movements from {video_path}")
            extractor = self._get_extractor("ganimation_vision")
            ganimation_features = extractor.get_feature_dict(str(video_path))
            features.update(ganimation_features)
        
        # Extract ARBEx emotional expression features (video-specific)
        if "arbex_vision" in self.features:
            print(f"Extracting ARBEx attentive feature extraction with reliability balancing from {video_path}")
            extractor = self._get_extractor("arbex_vision")
            arbex_features = extractor.get_feature_dict(str(video_path))
            features.update(arbex_features)
        
        # Extract OpenPose pose estimation features (video-specific)
        if "openpose_vision" in self.features:
            print(f"Extracting OpenPose real-time multi-person keypoint detection and pose estimation from {video_path}")
            extractor = self._get_extractor("openpose_vision")
            openpose_features = extractor.get_feature_dict(str(video_path))
            features.update(openpose_features)
          # Extract Insta-DM dense motion estimation features (video-specific)
        if "instadm_vision" in self.features:
            print(f"Extracting Insta-DM dense motion estimation and depth analysis from {video_path}")
            extractor = self._get_extractor("instadm_vision")
            instadm_features = extractor.get_feature_dict(str(video_path))
            features.update(instadm_features)
          # Extract Optical Flow movement and motion estimation features (video-specific)
        if "optical_flow_vision" in self.features:
            print(f"Extracting Optical Flow movement and motion estimation from {video_path}")
            extractor = self._get_extractor("optical_flow_vision")
            optical_flow_features = extractor.get_feature_dict(str(video_path))
            features.update(optical_flow_features)
          # Extract CrowdFlow optical flow fields, person trajectories, and tracking accuracy features (video-specific)
        if "crowdflow_vision" in self.features:
            print(f"Extracting CrowdFlow optical flow fields and person trajectories from {video_path}")
            extractor = self._get_extractor("crowdflow_vision")
            crowdflow_features = extractor.get_feature_dict(str(video_path))
            features.update(crowdflow_features)
        
        # Extract VideoFinder object and people localization features (video-specific)
        if "videofinder_vision" in self.features:
            print(f"Extracting VideoFinder object and people localization from {video_path}")
            extractor = self._get_extractor("videofinder_vision")
            videofinder_features = extractor.get_feature_dict(str(video_path))
            features.update(videofinder_features)
        
        # Extract SmoothNet pose estimation features (video-specific)
        if "smoothnet_vision" in self.features:
            print(f"Extracting SmoothNet temporally consistent pose estimation from {video_path}")
            extractor = self._get_extractor("smoothnet_vision")
            smoothnet_features = extractor.get_feature_dict(str(video_path))
            features.update(smoothnet_features)
            features.update(crowdflow_features)
        
        # Extract PARE vision features (video-specific)
        if "pare_vision" in self.features:
            print(f"Extracting PARE vision features from {video_path}")
            extractor = self._get_extractor("pare_vision")
            pare_features = extractor.get_feature_dict(str(video_path))
            features.update(pare_features)
        
        # Extract ViTPose vision features (video-specific)
        if "vitpose_vision" in self.features:
            print(f"Extracting ViTPose vision features from {video_path}")
            extractor = self._get_extractor("vitpose_vision")
            vitpose_features = extractor.get_feature_dict(str(video_path))
            features.update(vitpose_features)
        
        # Extract RSN vision features (video-specific)
        if "rsn_vision" in self.features:
            print(f"Extracting RSN keypoint localization features from {video_path}")
            extractor = self._get_extractor("rsn_vision")
            rsn_features = extractor.get_feature_dict(str(video_path))
            features.update(rsn_features)
        
        # Extract ME-GraphAU vision features (video-specific)
        if "me_graphau_vision" in self.features:
            print(f"Extracting ME-GraphAU facial action unit features from {video_path}")
            extractor = self._get_extractor("me_graphau_vision")
            me_graphau_features = extractor.get_feature_dict(str(video_path))
            features.update(me_graphau_features)
        
        # Extract DAN vision features (video-specific)
        if "dan_vision" in self.features:
            print(f"Extracting DAN emotional expression features from {video_path}")
            extractor = self._get_extractor("dan_vision")
            dan_features = extractor.get_feature_dict(str(video_path))
            features.update(dan_features)
        
        # Extract PSA vision features (video-specific)
        if "psa_vision" in self.features:
            print(f"Extracting PSA vision features from {video_path}")
            extractor = self._get_extractor("psa_vision")
            psa_features = extractor.get_feature_dict(str(video_path))
            features.update(psa_features)
        
        return features
    
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
                # Group features by model/algorithm categories
                grouped_features = self._group_features_by_model(file_features)
                
                # Convert the grouped features to JSON-compatible format
                consolidated_json[filename] = {}
                
                for group_name, group_data in grouped_features.items():
                    consolidated_json[filename][group_name] = {
                        "Feature": group_data["Feature"],
                        "Model": group_data["Model"],
                        "features": {}
                    }
                    
                    # Process each feature in the group
                    for key, value in group_data["features"].items():
                        if isinstance(value, np.ndarray):
                            if value.size > 1000:
                                # Include statistics for large arrays
                                consolidated_json[filename][group_name]["features"][key] = {
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
                                    consolidated_json[filename][group_name]["features"][key] = [float(x) for x in value.tolist()]
                                elif value.dtype.kind in 'iu':  # integer
                                    consolidated_json[filename][group_name]["features"][key] = [int(x) for x in value.tolist()]
                                else:
                                    consolidated_json[filename][group_name]["features"][key] = value.tolist()
                        elif not callable(value):
                            # Handle other numpy types that might be scalars
                            if isinstance(value, (np.number, np.float32, np.float64, np.int32, np.int64)):
                                consolidated_json[filename][group_name]["features"][key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                            else:
                                consolidated_json[filename][group_name]["features"][key] = value
            
            # Save consolidated JSON
            with open(self.output_dir / "pipeline_features.json", "w") as f:
                json.dump(consolidated_json, f, indent=2)
                
            print(f"Consolidated features saved to {self.output_dir / 'pipeline_features.json'}")
                
        except Exception as e:
            print(f"Warning: Could not save consolidated JSON: {e}")
            traceback.print_exc()
        
        return results
    
    def _group_features_by_model(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Group features by their model/algorithm categories according to the specification.
        
        Args:
            features: Dictionary of all extracted features
            
        Returns:
            Dictionary grouped by "Feature" categories
        """
        # Define feature groupings according to the specification table
        feature_groups = {
            "Audio volume": {
                "exact_matches": ["oc_audvol"],
                "model_name": "OpenCV"
            },
            "Change in audio volume": {
                "exact_matches": ["oc_audvol_diff"],
                "model_name": "OpenCV"
            },
            "Average audio pitch": {
                "exact_matches": ["oc_audpit"],
                "model_name": "OpenCV"
            },
            "Change in audio pitch": {
                "exact_matches": ["oc_audpit_diff"],
                "model_name": "OpenCV"
            },
            "Speech emotion/emotional speech classification": {
                "prefixes": ["ser_"],
                "exact_matches": [],
                "model_name": "Speech Emotion Recognition"
            },
            "Time-Accurate Speech Transcription": {
                "prefixes": ["WhX_"],
                "exact_matches": ["transcription", "language", "num_segments"],
                "model_name": "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio"
            },
            "Spectral Features, Pitch, Rhythm": {
                "prefixes": ["lbrs_"],
                "exact_matches": [],
                "model_name": "Librosa"
            },
            "(1) High-quality time-stretching of WAV/MP3 files without changing their pitch; (2) Time-stretch silence separately": {
                "prefixes": ["AS_"],
                "exact_matches": [],
                "model_name": "AudioStretchy"
            },
            "Speech feature extraction": {
                "prefixes": ["osm_"],
                "exact_matches": ["sample_rate", "hop_length", "num_frames"],
                "model_name": "openSMILE"
            },
            "Sentiment Analysis": {
                "prefixes": ["arvs_"],
                "exact_matches": [],
                "model_name": "AnAlgorithm for Routing Vectors in Sequences"
            },
            "Emotion Recognition during Social Interactions": {
                "prefixes": ["MELD_"],
                "exact_matches": [],
                "model_name": "MELD (Multimodal Multi-Party Dataset for Emotion Recognition in Conversation)"
            },
            "Disentangled Attention Mechanism & Enhanced Mask Decoder": {
                "prefixes": ["DEB_"],
                "exact_matches": [],
                "model_name": "DEBERTA"
            },
            "Contrastive Learning of Sentence Embeddings": {
                "prefixes": ["CSE_"],
                "exact_matches": [],
                "model_name": "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
            },
            "Language representation": {
                "prefixes": ["alb_"],
                "exact_matches": [],
                "model_name": "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
            },
            "Dense Vector Representations and Reranking": {
                "prefixes": ["BERT_"],
                "exact_matches": [],                "model_name": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
            },
            "text classification + semantic similarity + semantic cluster": {
                "prefixes": ["USE_"],
                "exact_matches": [],
                "model_name": "Universal Sentence Encoder"
            },
            "Real time video emotion analysis and AU detection": {
                "prefixes": ["eln_"],
                "exact_matches": [],
                "model_name": "Frame-level Prediction of Facial Expressions, Valence, Arousal and Action Units for Mobile Devices"
            },
            "Pose estimation and tracking": {
                "prefixes": ["GMP_"],
                "exact_matches": ["total_frames", "landmarks_detected_frames", "detection_rate", "avg_landmarks_per_frame"],
                "model_name": "Google MediaPipe"
            },
            "Pose estimation (high-resolution)": {
                "prefixes": ["DHiR_"],
                "exact_matches": ["total_frames", "pose_detected_frames", "detection_rate", "avg_keypoints_per_frame"],
                "model_name": "Deep High-Resolution Representation Learning for Human Pose Estimation"
            },
            "Pose estimation and tracking (simple baselines)": {
                "prefixes": ["SBH_"],
                "exact_matches": ["total_frames", "pose_detected_frames", "detection_rate", "avg_keypoints_per_frame"],
                "model_name": "Simple Baselines for Human Pose Estimation and Tracking"
            },
            "Actional annotation, Emotion indices, Face location and angles": {
                "prefixes": ["pf_"],
                "exact_matches": ["total_frames", "faces_detected_frames", "face_detection_rate", "avg_face_size", "avg_face_confidence"],
                "model_name": "Py-Feat: Python Facial Expression Analysis Toolbox"
            },
            "Continuous manifold for anatomical facial movements": {
                "prefixes": ["GAN_"],
                "exact_matches": ["total_frames", "faces_detected_frames", "face_detection_rate", "max_au_activations", "avg_au_activations_per_frame"],
                "model_name": "GANimation: Anatomy-aware Facial Animation from a Single Image"
            },
            "Extract emotional indices via different feature levels": {
                "prefixes": ["arbex_"],
                "exact_matches": ["total_frames", "faces_detected_frames", "face_detection_rate", "avg_confidence_primary", "avg_confidence_final", "avg_reliability_score"],
                "model_name": "ARBEx: Attentive Feature Extraction with Reliability Balancing for Robust Facial Expression Learning"
            },
            "Real-time multi-person keypoint detection and pose estimation": {
                "prefixes": ["openPose_"],
                "exact_matches": ["total_frames", "pose_detected_frames", "detection_rate", "avg_keypoints_per_frame", "avg_confidence", "max_persons_detected"],
                "model_name": "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"
            },            "Dense Motion Estimation, Depth in dynamic scenes, interaction patterns": {
                "prefixes": ["indm_"],
                "exact_matches": ["indm_abs_rel", "indm_sq_rel", "indm_rmse", "indm_rmse_log", "indm_acc_1", "indm_acc_2", "indm_acc_3", "total_frames", "depth_estimated_frames", "motion_detected_frames"],
                "model_name": "Insta-DM: Instance-aware Dynamic Module for Monocular Depth Estimation"
            },            "Movement and estimation of motion": {
                "prefixes": ["optical_flow_", "flow_"],
                "exact_matches": ["sparse_flow_vis_.png", "sparse_points.npy", "dense_flow.npy", "dense_flow_vis_.png", "motion_detected_frames", "avg_motion_magnitude", "max_motion_magnitude", "total_displacement", "dominant_motion_direction", "motion_consistency"],
                "model_name": "Optical Flow"
            },            "Optical flow fields, Person trajectories, Tracking accuracy": {
                "prefixes": ["of_"],
                "exact_matches": ["of_fg_static_epe_st", "of_fg_static_r2_st", "of_bg_static_epe_st", "of_bg_static_r2_st", "of_fg_dynamic_epe_st", "of_fg_dynamic_r2_st", "of_bg_dynamic_epe_st", "of_bg_dynamic_r2_st", "of_fg_avg_epe_st", "of_fg_avg_r2_st", "of_bg_avg_epe_st", "of_bg_avg_r2_st", "of_avg_epe_st", "of_avg_r2_st", "of_time_length_st", "of_ta_IM01", "of_ta_IM01_Dyn", "of_ta_IM02", "of_ta_IM02_Dyn", "of_ta_IM03", "of_ta_IM03_Dyn", "of_ta_IM04", "of_ta_IM04_Dyn", "of_ta_IM05", "of_ta_IM05_Dyn", "of_ta_average", "of_pt_IM01", "of_pt_IM01_Dyn", "of_pt_IM02", "of_pt_IM02_Dyn", "of_pt_IM03", "of_pt_IM03_Dyn", "of_pt_IM04", "of_pt_IM04_Dyn", "of_pt_IM05", "of_pt_IM05_Dyn", "of_pt_average"],
                "model_name": "CrowdFlow: Optical Flow Dataset and Benchmark for Visual Crowd Analysis"
            },
            "Locate the objects and people": {
                "prefixes": ["ViF_"],
                "exact_matches": ["total_frames", "objects_detected_frames", "people_detected_frames", "total_detected_objects", "total_detected_people", "avg_objects_per_frame", "avg_people_per_frame", "detection_rate"],
                "model_name": "VideoFinder"
            },            "Pose estimation": {
                "prefixes": ["net_"],
                "exact_matches": ["net_3d_estimator", "net_3d_MPJPE_input_ad", "net_3d_MPJPE_output_ad", "net_3d_Accel_input_ad", "net_3d_Accel_output_ad", "net_2d_estimator", "net_2d_MPJPE_input_ad", "net_2d_MPJPE_output_ad", "net_2d_Accel_input_ad", "net_2d_Accel_output_ad", "net_SMPL_estimator", "net_SMPL_MPJPE_input_ad", "net_SMPL_MPJPE_output_ad", "net_SMPL_Accel_input_ad", "net_SMPL_Accel_output_ad"],
                "model_name": "SmoothNet"
            },
            "3D Human Body Estimation and Pose Analysis": {
                "prefixes": ["PARE_"],
                "exact_matches": [],
                "model_name": "PARE (Part Attention Regressor for 3D Human Body Estimation)"
            },
            "Pose estimation (Vision Transformer)": {
                "prefixes": ["vit_"],
                "exact_matches": [],
                "model_name": "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation"
            },
            "Keypoint localization": {
                "prefixes": ["rsn_"],
                "exact_matches": [],
                "model_name": "Residual Steps Network (RSN)"
            },
            "Facial action, AU relation graph": {
                "prefixes": ["ann_"],
                "exact_matches": [],
                "model_name": "Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition"
            },
            "Emotional expression indices": {
                "prefixes": ["dan_"],
                "exact_matches": ["dan_emotion_scores"],
                "model_name": "DAN: Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition"
            }
        }
        
        # Initialize grouped features
        grouped_features = {}
        ungrouped_features = {}
        
        # Group features by category
        for feature_name, feature_value in features.items():
            matched = False
            
            for group_name, group_config in feature_groups.items():
                # Check exact matches first (more specific)
                if feature_name in group_config.get("exact_matches", []):
                    if group_name not in grouped_features:
                        grouped_features[group_name] = {
                            "Feature": group_name,
                            "Model": group_config["model_name"],
                            "features": {}
                        }
                    grouped_features[group_name]["features"][feature_name] = feature_value
                    matched = True
                    break
                
                # Check prefix matches
                if any(feature_name.startswith(prefix) for prefix in group_config.get("prefixes", [])):
                    if group_name not in grouped_features:
                        grouped_features[group_name] = {
                            "Feature": group_name,
                            "Model": group_config["model_name"],
                            "features": {}
                        }
                    grouped_features[group_name]["features"][feature_name] = feature_value
                    matched = True
                    break
            
            # If no match found, add to ungrouped
            if not matched:
                ungrouped_features[feature_name] = feature_value
        
        # Add ungrouped features if any exist
        if ungrouped_features:
            grouped_features["Other"] = {
                "Feature": "Other",
                "Model": "Various",
                "features": ungrouped_features
            }
        
        return grouped_features
