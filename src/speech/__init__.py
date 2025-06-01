"""
Speech processing modules.
"""

# Import main classes for easier access
try:
    from .emotion_recognition import SpeechEmotionRecognizer
    from .speech_separator import SpeechSeparator
    from .whisperx_transcriber import WhisperXTranscriber
    from .xlsr_speech_to_text import XLSRSpeechToText
    from .s2t_speech_to_text import S2TSpeechToText
except ImportError as e:
    import warnings
    warnings.warn(f"Some speech modules couldn't be imported: {e}")

__all__ = [
    'SpeechEmotionRecognizer',
    'SpeechSeparator',
    'WhisperXTranscriber',
    'XLSRSpeechToText',
    'S2TSpeechToText'
]
