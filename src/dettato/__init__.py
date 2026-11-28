"""Dettato: multi-instrument audio-to-MIDI transcription by KobiMusic.

    >>> import dettato
    >>> result = dettato.transcribe("song.mp3")
    >>> result.save_midi("song.mid")
"""
from .pipeline import Dettato, Track, Transcription, transcribe

__version__ = "2.0.1"
__all__ = ["Dettato", "Track", "Transcription", "transcribe", "__version__"]
