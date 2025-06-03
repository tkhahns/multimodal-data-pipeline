"""
Speech separation module using SepFormer.
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
import os

class SpeechSeparator:
    """Speech separator using SepFormer from SpeechBrain."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the speech separation model.
        
        Args:
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.device = device
        self.model = None
        self.sample_rate = 8000  # SepFormer default
        
    def _load_model(self):
        """Load the SepFormer model from SpeechBrain."""
        try:
            from speechbrain.pretrained import SepformerSeparation
            self.model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-libri3mix", 
                savedir="pretrained_models/sepformer-libri3mix",
                run_opts={"device": self.device}
            )
        except ImportError:
            print("SpeechBrain not installed. Please install it using: poetry add speechbrain")
            raise
    
    def separate(self, audio_path: str, output_dir: str = None) -> Tuple[List[np.ndarray], List[str]]:
        """
        Separate speakers from an audio file.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save separated audio files (if None, files are not saved)
            
        Returns:
            Tuple[List[np.ndarray], List[str]]: List of separated audio waveforms and paths to saved files
        """
        # Load model if not already loaded
        if self.model is None:
            self._load_model()
        
        # Ensure audio is in the correct format for the model
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Perform separation
        est_sources = self.model.separate_file(audio_path)
        separated_sources = [source.cpu().numpy() for source in est_sources]
        
        # Save separated sources if output_dir is provided
        output_paths = []
        if output_dir:
            output_dir = Path(output_dir)
            base_name = Path(audio_path).stem
            
            # Create a subfolder for this specific audio file
            audio_out_dir = output_dir / base_name
            os.makedirs(audio_out_dir, exist_ok=True)
            
            for i, source in enumerate(separated_sources):
                # Using standard naming convention from HuggingFace SepFormer: source{i}hat.wav
                output_path = str(audio_out_dir / f"source{i+1}hat.wav")
                torchaudio.save(
                    output_path,
                    torch.tensor(source),
                    self.sample_rate
                )
                output_paths.append(output_path)
                
        return separated_sources, output_paths
    
    def get_feature_dict(self, audio_path: str, output_dir: str = None) -> dict:
        """
        Get a dictionary of separated speech features.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save separated audio files
            
        Returns:
            dict: Dictionary with separated sources
        """
        separated_sources, source_paths = self.separate(audio_path, output_dir)
        
        result = {}
        for i, (source, path) in enumerate(zip(separated_sources, source_paths if output_dir else [])):
            source_key = f"source{i+1}hat"
            result[source_key] = source
            
            if output_dir:
                result[f"{source_key}_path"] = path
                
        return result
