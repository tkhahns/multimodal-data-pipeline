"""
WhisperX for time-accurate speech transcription and diarization.
"""
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Any, Optional, Tuple

class WhisperXTranscriber:
    """Transcribe and diarize speech using WhisperX."""
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "float32",
        language: str = "en",
        batch_size: int = 16,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the WhisperX transcriber.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run on ("cpu" or "cuda")
            compute_type: Type for computation ("float32", "float16", "int8")
            language: Language code for transcription
            batch_size: Batch size for processing
            hf_token: HuggingFace token for accessing gated models
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.batch_size = batch_size
        self.hf_token = hf_token
        
        # Models will be loaded on-demand
        self.model = None
        self.diarization_model = None
        
    def _load_models(self):
        """Load WhisperX and diarization models."""
        try:
            import whisperx
            
            # Load ASR model
            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            
            # Load diarization model
            self.diarization_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
        except ImportError:
            print("WhisperX not installed. Please install it using: pip install git+https://github.com/m-bain/whisperx.git")
            raise
    
    def transcribe(
        self, 
        audio_path: str,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> Dict[str, Any]:
        """
        Transcribe and diarize an audio file.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            Dict[str, Any]: Transcription results with diarization
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Load models if not already loaded
        if self.model is None or self.diarization_model is None:
            self._load_models()
            
        try:
            import whisperx
            
            # Transcribe with word-level timestamps
            transcribe_result = self.model.transcribe(
                audio_path,
                batch_size=self.batch_size
            )
            
            # Align the transcription
            align_model, align_metadata = whisperx.load_align_model(
                language_code=transcribe_result["language"],
                device=self.device
            )
            
            aligned_result = whisperx.align(
                transcribe_result["segments"],
                align_model,
                align_metadata,
                audio_path,
                self.device,
                return_char_alignments=False
            )
            
            # Perform diarization
            diarize_segments = self.diarization_model(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Assign speaker labels to the segments
            result = whisperx.assign_word_speakers(
                diarize_segments,
                aligned_result
            )
            
            return result
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            raise
    
    def get_feature_dict(
        self, 
        audio_path: str,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> Dict[str, Any]:
        """
        Get a dictionary of WhisperX features with highlighting for diarization.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            Dict[str, Any]: Dictionary with WhisperX highlighted features
        """
        try:
            result = self.transcribe(audio_path, min_speakers, max_speakers)
            
            # Process the result to extract the highlighted diarization features
            feature_dict = {}
            
            for segment in result.get("segments", []):
                speaker = segment.get("speaker", "unknown")
                
                for word_idx, word in enumerate(segment.get("words", [])):
                    text = word.get("text", "")
                    highlight = "adjusted" in word  # Flag if timestamp was adjusted during diarization
                    
                    # Create a key for this word
                    key = f"WhX_highlight_diarize__{speaker}_word_{word_idx+1}"
                    feature_dict[key] = {
                        "text": text,
                        "highlighted": highlight,
                        "start": word.get("start"),
                        "end": word.get("end")
                    }
            
            # Add the full transcription as well
            feature_dict["transcription"] = " ".join([
                word["text"] 
                for segment in result.get("segments", [])
                for word in segment.get("words", [])
            ])
            
            return feature_dict
            
        except Exception as e:
            print(f"Error extracting WhisperX features: {e}")
            raise
