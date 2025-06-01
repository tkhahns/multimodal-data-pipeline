"""
XLSR (Cross-Lingual Speech Representations) Speech-to-Text module.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

class XLSRSpeechToText:
    """Speech-to-Text using XLSR (Cross-Lingual Speech Representations)."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        device: str = "cpu",
        return_hidden_states: bool = False
    ):
        """
        Initialize the XLSR Speech-to-Text model.
        
        Args:
            model_name: Name of the XLSR model on HuggingFace
            device: Device to run the model on ("cpu" or "cuda")
            return_hidden_states: Whether to return hidden states from the model
        """
        self.model_name = model_name
        self.device = device
        self.return_hidden_states = return_hidden_states
        
        # Models will be loaded on-demand
        self.model = None
        self.processor = None
        
    def _load_models(self):
        """Load the XLSR model and processor."""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            print("Transformers not installed. Please install it using: pip install transformers")
            raise
    
    def transcribe(self, audio_path: str) -> Dict[str, Union[str, np.ndarray]]:
        """
        Transcribe an audio file using XLSR.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict[str, Union[str, np.ndarray]]: Transcription and optionally hidden states
        """
        # Load models if not already loaded
        if self.model is None or self.processor is None:
            self._load_models()
            
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            try:
                audio, sample_rate = sf.read(audio_path)
                if audio.ndim > 1:  # Convert to mono
                    audio = audio.mean(axis=1)
            except Exception:
                # Fallback to librosa if soundfile fails
                audio, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Resample if needed
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Process audio with XLSR
            inputs = self.processor(
                audio, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=self.return_hidden_states)
            
            # Get transcription
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            result = {"transcription": transcription}
            
            # Add hidden states if requested
            if self.return_hidden_states and outputs.hidden_states is not None:
                # Get the hidden states from the last layer
                last_hidden_states = outputs.hidden_states[-1].cpu().numpy()
                result["hidden_states"] = last_hidden_states
                
            return result
            
        except Exception as e:
            print(f"Error in XLSR transcription: {e}")
            raise
    
    def get_feature_dict(self, audio_path: str) -> Dict[str, Union[str, np.ndarray]]:
        """
        Get a dictionary of XLSR features.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict[str, Union[str, np.ndarray]]: Dictionary with XLSR features
        """
        return self.transcribe(audio_path)
