"""
Simple emotion recognition implementation as a fallback
for when complex libraries fail to install.
"""

import numpy as np
import librosa
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SimpleEmotionRecognizer:
    """
    A simple emotion recognition implementation that uses basic audio features
    to estimate emotional states. This is a fallback when complex libraries fail.
    """
    
    def __init__(self):
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom']
        
    def extract_features(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, float]:
        """Extract basic audio features for emotion recognition"""
        try:
            # Basic spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Aggregate features
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                'zcr_mean': np.mean(zcr),
                'zcr_std': np.std(zcr),
                'energy': np.mean(audio ** 2),
                'pitch_mean': np.mean(spectral_centroids),
                'pitch_std': np.std(spectral_centroids)
            }
            
            # Add MFCC features
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
                
            # Add chroma features
            for i in range(12):
                features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting emotion features: {e}")
            return {}
    
    def predict_emotion(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, Any]:
        """
        Predict emotion using simple heuristics based on audio features.
        This is a basic implementation for demonstration purposes.
        """
        try:
            features = self.extract_features(audio, sr)
            
            if not features:
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'probabilities': {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
                }
            
            # Simple heuristic-based emotion detection
            energy = features.get('energy', 0)
            pitch_mean = features.get('pitch_mean', 0)
            pitch_std = features.get('pitch_std', 0)
            zcr_mean = features.get('zcr_mean', 0)
            
            # Normalize features (rough normalization)
            energy_norm = min(energy * 1000, 1.0)  # Normalize energy
            pitch_norm = min(pitch_mean / 2000, 1.0)  # Normalize pitch
            pitch_var_norm = min(pitch_std / 500, 1.0)  # Normalize pitch variation
            zcr_norm = min(zcr_mean * 10, 1.0)  # Normalize ZCR
            
            # Simple rules for emotion detection
            probabilities = {}
            
            # High energy + high pitch variation = excited/happy
            if energy_norm > 0.6 and pitch_var_norm > 0.5:
                probabilities['happy'] = 0.4
                probabilities['ps'] = 0.3  # ps might mean "positive surprise"
                probabilities['angry'] = 0.2
                probabilities['neutral'] = 0.1
            # High energy + low pitch variation = angry
            elif energy_norm > 0.6 and pitch_var_norm < 0.3:
                probabilities['angry'] = 0.5
                probabilities['disgust'] = 0.2
                probabilities['neutral'] = 0.2
                probabilities['fear'] = 0.1
            # Low energy + low pitch = sad/boredom
            elif energy_norm < 0.3 and pitch_norm < 0.4:
                probabilities['sad'] = 0.4
                probabilities['boredom'] = 0.3
                probabilities['calm'] = 0.2
                probabilities['neutral'] = 0.1
            # High ZCR = fear/anxiety
            elif zcr_norm > 0.7:
                probabilities['fear'] = 0.4
                probabilities['ps'] = 0.2
                probabilities['angry'] = 0.2
                probabilities['neutral'] = 0.2
            # Medium energy = calm/neutral
            elif 0.3 <= energy_norm <= 0.6:
                probabilities['calm'] = 0.4
                probabilities['neutral'] = 0.3
                probabilities['happy'] = 0.2
                probabilities['boredom'] = 0.1
            # Default to neutral
            else:
                probabilities['neutral'] = 0.4
                probabilities['calm'] = 0.2
                probabilities['happy'] = 0.15
                probabilities['sad'] = 0.15
                probabilities['boredom'] = 0.1
            
            # Fill in remaining emotions with small probabilities
            total_prob = sum(probabilities.values())
            remaining_prob = 1.0 - total_prob
            remaining_emotions = [e for e in self.emotions if e not in probabilities]
            
            if remaining_emotions and remaining_prob > 0:
                prob_per_emotion = remaining_prob / len(remaining_emotions)
                for emotion in remaining_emotions:
                    probabilities[emotion] = prob_per_emotion
            
            # Ensure all emotions have probabilities
            for emotion in self.emotions:
                if emotion not in probabilities:
                    probabilities[emotion] = 0.01
            
            # Normalize probabilities
            total = sum(probabilities.values())
            probabilities = {k: v/total for k, v in probabilities.items()}
            
            # Get dominant emotion
            dominant_emotion = max(probabilities.items(), key=lambda x: x[1])
            
            return {
                'emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'probabilities': probabilities,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'probabilities': {emotion: 1.0/len(self.emotions) for emotion in self.emotions},
                'error': str(e)
            }

def analyze_emotion(audio_path: str) -> Dict[str, Any]:
    """
    Analyze emotion from an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with emotion analysis results
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050)
        
        # Initialize recognizer
        recognizer = SimpleEmotionRecognizer()
        
        # Predict emotion
        result = recognizer.predict_emotion(audio, sr)
        result['audio_path'] = audio_path
        result['audio_duration'] = len(audio) / sr
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing emotion for {audio_path}: {e}")
        return {
            'audio_path': audio_path,
            'emotion': 'neutral',
            'confidence': 0.0,
            'probabilities': {'neutral': 1.0},
            'error': str(e)
        }

def get_emotion_features_dict(audio_path: str) -> Dict[str, float]:
    """
    Get emotion features in the specific format requested (ser_*).
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dict with emotion features prefixed with 'ser_'
    """
    try:
        result = analyze_emotion(audio_path)
        probabilities = result.get('probabilities', {})
        
        # Convert to ser_* format
        emotion_features = {}
        expected_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom']
        
        for emotion in expected_emotions:
            emotion_features[f'ser_{emotion}'] = probabilities.get(emotion, 0.0)
        
        return emotion_features
        
    except Exception as e:
        logger.error(f"Error getting emotion features for {audio_path}: {e}")
        # Return zero probabilities for all emotions
        return {f'ser_{emotion}': 0.0 for emotion in ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom']}

if __name__ == "__main__":
    # Test the emotion recognizer
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        result = analyze_emotion(audio_path)
        print(f"Emotion analysis for {audio_path}:")
        print(f"Predicted emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        print("Probabilities:")
        for emotion, prob in result['probabilities'].items():
            print(f"  {emotion}: {prob:.3f}")
    else:
        print("Usage: python simple_emotion_recognition.py <audio_file>")
