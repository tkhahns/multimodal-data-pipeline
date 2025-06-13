"""
MELD Emotion Recognition Analyzer

This module implements emotion recognition during social interactions using
patterns and features inspired by the MELD dataset (Multimodal Multi-Party 
Dataset for Emotion Recognition in Conversation).

MELD focuses on:
- Multi-party conversations
- Emotion recognition in dialogues
- Speaker-specific emotion patterns
- Emotion transitions and shifts
- Conversational context analysis
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

class MELDEmotionAnalyzer:
    """
    Analyzer for emotion recognition during social interactions based on MELD patterns.
    
    MELD (Multimodal Multi-Party Dataset for Emotion Recognition in Conversation)
    focuses on analyzing emotions in conversational contexts with multiple speakers.
    """
    
    def __init__(self):
        """Initialize the MELD emotion analyzer."""
        # MELD emotion categories
        self.emotion_categories = [
            'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        ]
        
        # Emotional keywords and patterns for each category
        self.emotion_keywords = {
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'damn', 'shit', 'annoyed', 'irritated'],
            'disgust': ['disgusting', 'gross', 'yuck', 'eww', 'disgusted', 'revolting', 'nasty'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'panic'],
            'joy': ['happy', 'excited', 'great', 'awesome', 'wonderful', 'amazing', 'fantastic', 'love'],
            'neutral': ['okay', 'fine', 'alright', 'sure', 'yes', 'no', 'maybe', 'well'],
            'sadness': ['sad', 'depressed', 'upset', 'crying', 'tears', 'heartbroken', 'disappointed'],
            'surprise': ['wow', 'really', 'omg', 'amazing', 'incredible', 'unbelievable', 'shocking']
        }
        
        # Emotional intensity modifiers
        self.intensity_modifiers = {
            'high': ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely'],
            'medium': ['quite', 'rather', 'pretty', 'fairly', 'somewhat'],
            'low': ['slightly', 'a bit', 'kind of', 'sort of', 'little']
        }
    
    def analyze_conversation_emotions(self, transcript: str, speaker_info: Dict = None) -> Dict[str, Any]:
        """
        Analyze emotions in conversational transcript using MELD-inspired features.
        
        Args:
            transcript: Full conversation transcript
            speaker_info: Optional speaker diarization information
            
        Returns:
            Dictionary containing MELD-style emotion analysis features
        """
        features = {}
        
        # Parse transcript into utterances and speakers
        utterances = self._parse_transcript(transcript, speaker_info)
        
        # Basic conversation statistics
        features.update(self._calculate_basic_stats(utterances, transcript))
        
        # Emotion detection and counting
        features.update(self._analyze_emotions(utterances))
        
        # Emotion shift analysis
        features.update(self._analyze_emotion_shifts(utterances))
        
        # Speaker-specific analysis
        features.update(self._analyze_speaker_patterns(utterances))
        
        # Dialogue structure analysis
        features.update(self._analyze_dialogue_structure(utterances))
        
        return features
    
    def _parse_transcript(self, transcript: str, speaker_info: Dict = None) -> List[Dict]:
        """
        Parse transcript into structured utterances with speaker information.
        
        Args:
            transcript: Raw transcript text
            speaker_info: Speaker diarization data
            
        Returns:
            List of utterance dictionaries with speaker and emotion info
        """
        utterances = []
        
        # Split transcript into sentences/utterances
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have speaker info from WhisperX, use it
        if speaker_info and isinstance(speaker_info, dict):
            # Extract speaker segments from WhisperX output
            speaker_segments = []
            for key, value in speaker_info.items():
                if 'SPEAKER_' in key and 'word_' in key:
                    # Extract speaker and word info
                    parts = key.split('_')
                    if len(parts) >= 3:
                        speaker_id = f"{parts[1]}_{parts[2]}"  # SPEAKER_01
                        speaker_segments.append({
                            'speaker': speaker_id,
                            'text': str(value),
                            'key': key
                        })
            
            # Group by speakers and create utterances
            current_speaker = None
            current_text = ""
            
            for segment in speaker_segments:
                if segment['speaker'] != current_speaker:
                    if current_text.strip():
                        utterances.append({
                            'speaker': current_speaker or 'SPEAKER_01',
                            'text': current_text.strip(),
                            'length': len(current_text.strip()),
                            'word_count': len(current_text.strip().split())
                        })
                    current_speaker = segment['speaker']
                    current_text = segment['text']
                else:
                    current_text += " " + segment['text']
            
            # Add final utterance
            if current_text.strip():
                utterances.append({
                    'speaker': current_speaker or 'SPEAKER_01',
                    'text': current_text.strip(),
                    'length': len(current_text.strip()),
                    'word_count': len(current_text.strip().split())
                })
        
        # Fallback: create utterances from sentences with estimated speakers
        if not utterances:
            for i, sentence in enumerate(sentences):
                # Simple speaker alternation assumption
                speaker_id = f"SPEAKER_{i % 2:02d}"
                utterances.append({
                    'speaker': speaker_id,
                    'text': sentence,
                    'length': len(sentence),
                    'word_count': len(sentence.split())
                })
        
        # Add emotion analysis to each utterance
        for utterance in utterances:
            utterance['emotions'] = self._detect_emotions_in_text(utterance['text'])
            utterance['dominant_emotion'] = self._get_dominant_emotion(utterance['emotions'])
            utterance['emotion_intensity'] = self._calculate_emotion_intensity(utterance['text'])
        
        return utterances
    
    def _detect_emotions_in_text(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in text using keyword matching and patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of emotion scores
        """
        text_lower = text.lower()
        word_tokens = re.findall(r'\b\w+\b', text_lower)
        
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_categories}
        
        # Count emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1.0
        
        # Apply intensity modifiers
        for intensity, modifiers in self.intensity_modifiers.items():
            multiplier = {'high': 2.0, 'medium': 1.5, 'low': 0.5}[intensity]
            for modifier in modifiers:
                if modifier in text_lower:
                    # Boost all detected emotions
                    for emotion in emotion_scores:
                        if emotion_scores[emotion] > 0:
                            emotion_scores[emotion] *= multiplier
                    break
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v / total_score for k, v in emotion_scores.items()}
        else:
            # Default to neutral if no emotions detected
            emotion_scores['neutral'] = 1.0
        
        return emotion_scores
    
    def _get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> str:
        """Get the dominant emotion from scores."""
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_emotion_intensity(self, text: str) -> float:
        """
        Calculate overall emotional intensity of text.
        
        Args:
            text: Input text
            
        Returns:
            Emotion intensity score (0-1)
        """
        text_lower = text.lower()
        
        # Count emotional words
        emotional_word_count = 0
        for keywords in self.emotion_keywords.values():
            for keyword in keywords:
                emotional_word_count += text_lower.count(keyword)
        
        # Count intensity modifiers
        intensity_count = 0
        for modifiers in self.intensity_modifiers.values():
            for modifier in modifiers:
                intensity_count += text_lower.count(modifier)
        
        # Count exclamation marks and caps
        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Calculate intensity score
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        intensity = (
            (emotional_word_count * 0.4) +
            (intensity_count * 0.3) +
            (exclamation_count * 0.2) +
            (caps_ratio * 0.1)
        ) / word_count
        
        return min(intensity, 1.0)
    
    def _calculate_basic_stats(self, utterances: List[Dict], transcript: str) -> Dict[str, Any]:
        """Calculate basic conversation statistics."""
        features = {}
        
        # Modality (text-based analysis)
        features['MELD_modality'] = 'text'
        
        # Unique words
        all_words = []
        for utterance in utterances:
            words = re.findall(r'\b\w+\b', utterance['text'].lower())
            all_words.extend(words)
        features['MELD_unique_words'] = len(set(all_words))
        
        # Utterance length statistics
        lengths = [utterance['length'] for utterance in utterances]
        features['MELD_avg_utterance_length'] = np.mean(lengths) if lengths else 0.0
        features['MELD_max_utterance_length'] = max(lengths) if lengths else 0
        
        # Number of utterances
        features['MELD_num_utterances'] = len(utterances)
        
        # Estimate utterance duration (simplified)
        # Assume ~150 words per minute speaking rate
        word_counts = [utterance['word_count'] for utterance in utterances]
        durations = [wc / 2.5 for wc in word_counts]  # ~2.5 words per second
        features['MELD_avg_utterance_duration'] = np.mean(durations) if durations else 0.0
        
        return features
    
    def _analyze_emotions(self, utterances: List[Dict]) -> Dict[str, Any]:
        """Analyze emotion distribution in the conversation."""
        features = {}
        
        # Count emotions
        emotion_counts = Counter()
        total_emotions = 0
        
        for utterance in utterances:
            dominant_emotion = utterance['dominant_emotion']
            emotion_counts[dominant_emotion] += 1
            total_emotions += 1
        
        # Set emotion counts for each MELD emotion category
        for emotion in self.emotion_categories:
            features[f'MELD_count_{emotion}'] = emotion_counts.get(emotion, 0)
        
        # Average emotions per dialogue (simplified as per utterance)
        features['MELD_avg_num_emotions_per_dialogue'] = total_emotions / max(len(utterances), 1)
        
        return features
    
    def _analyze_emotion_shifts(self, utterances: List[Dict]) -> Dict[str, Any]:
        """Analyze emotion transitions and shifts in the conversation."""
        features = {}
        
        # Count emotion shifts
        emotion_shifts = 0
        for i in range(1, len(utterances)):
            prev_emotion = utterances[i-1]['dominant_emotion']
            curr_emotion = utterances[i]['dominant_emotion']
            if prev_emotion != curr_emotion:
                emotion_shifts += 1
        
        features['MELD_num_emotion_shift'] = emotion_shifts
        
        return features
    
    def _analyze_speaker_patterns(self, utterances: List[Dict]) -> Dict[str, Any]:
        """Analyze speaker-specific patterns."""
        features = {}
        
        # Count unique speakers
        speakers = set(utterance['speaker'] for utterance in utterances)
        features['MELD_num_speakers'] = len(speakers)
        
        # Count dialogues (simplified as speaker changes)
        dialogue_count = 1
        current_speaker = utterances[0]['speaker'] if utterances else None
        
        for utterance in utterances[1:]:
            if utterance['speaker'] != current_speaker:
                dialogue_count += 1
                current_speaker = utterance['speaker']
        
        features['MELD_num_dialogues'] = dialogue_count
        
        return features
    
    def _analyze_dialogue_structure(self, utterances: List[Dict]) -> Dict[str, Any]:
        """Analyze the structure of the dialogue."""
        features = {}
        
        # This could be extended with more sophisticated dialogue analysis
        # For now, we include basic structural features
        
        # Calculate average emotions per dialogue segment
        if utterances:
            total_emotions = sum(1 for u in utterances if u['dominant_emotion'] != 'neutral')
            features['MELD_avg_num_emotions_per_dialogue'] = total_emotions / max(features.get('MELD_num_dialogues', 1), 1)
        else:
            features['MELD_avg_num_emotions_per_dialogue'] = 0.0
        
        return features
    
    def get_feature_dict(self, transcript_data: Any) -> Dict[str, Any]:
        """
        Main interface for extracting MELD emotion features.
        
        Args:
            transcript_data: Can be string transcript or dict with transcript and speaker info
            
        Returns:
            Dictionary of MELD emotion features
        """
        try:
            # Handle different input types
            if isinstance(transcript_data, str):
                transcript = transcript_data
                speaker_info = None
            elif isinstance(transcript_data, dict):
                transcript = transcript_data.get('transcription', transcript_data.get('text', ''))
                speaker_info = transcript_data
            else:
                transcript = str(transcript_data)
                speaker_info = None
            
            if not transcript or transcript.strip() == '':
                logger.warning("Empty transcript provided to MELD analyzer")
                return self._get_default_features()
            
            # Analyze emotions in conversation
            features = self.analyze_conversation_emotions(transcript, speaker_info)
            
            # Add metadata
            features.update({
                'MELD_analysis_timestamp': np.datetime64('now').astype(str),
                'MELD_text_length': len(transcript),
                'MELD_model_name': 'MELD_conversation_emotion_analyzer',
                'MELD_version': '1.0.0'
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error in MELD emotion analysis: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features when analysis fails."""
        features = {}
        
        # Set default values for all MELD features
        features['MELD_modality'] = 'text'
        features['MELD_unique_words'] = 0
        features['MELD_avg_utterance_length'] = 0.0
        features['MELD_max_utterance_length'] = 0
        features['MELD_avg_num_emotions_per_dialogue'] = 0.0
        features['MELD_num_dialogues'] = 0
        features['MELD_num_utterances'] = 0
        features['MELD_num_speakers'] = 0
        features['MELD_num_emotion_shift'] = 0
        features['MELD_avg_utterance_duration'] = 0.0
        
        # Emotion counts
        for emotion in self.emotion_categories:
            features[f'MELD_count_{emotion}'] = 0
        
        # Metadata
        features.update({
            'MELD_analysis_timestamp': np.datetime64('now').astype(str),
            'MELD_text_length': 0,
            'MELD_model_name': 'MELD_conversation_emotion_analyzer',
            'MELD_version': '1.0.0'
        })
        
        return features


def create_meld_analyzer():
    """Factory function to create MELD emotion analyzer."""
    return MELDEmotionAnalyzer()
