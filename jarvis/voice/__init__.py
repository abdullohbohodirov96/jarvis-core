"""
jarvis.voice — voice I/O subsystem for the JARVIS desktop AI assistant.

Public API
----------
Wake-word detection::

    from jarvis.voice import WakeWordDetector, get_wake_word_detector
    detector = get_wake_word_detector()
    await detector.start()
    await detector.wait_for_wake_word()

Speech-to-text::

    from jarvis.voice import SpeechToText, TranscriptionResult, get_stt
    stt = get_stt()
    result: TranscriptionResult = await stt.transcribe(audio_bytes)

Text-to-speech::

    from jarvis.voice import TextToSpeech, get_tts
    tts = get_tts()
    await tts.speak("Hello, I am JARVIS.")

Full pipeline::

    from jarvis.voice import VoicePipeline, VoiceResult, get_pipeline
    pipeline = get_pipeline(ai_agent=my_agent)
    result: VoiceResult = await pipeline.listen_and_respond()
"""

from jarvis.voice.wake_word import WakeWordDetector, get_wake_word_detector
from jarvis.voice.stt import SpeechToText, TranscriptionResult, get_stt
from jarvis.voice.tts import TextToSpeech, get_tts
from jarvis.voice.pipeline import (
    VoiceInput,
    VoiceResult,
    VoicePipeline,
    get_pipeline,
)

__all__ = [
    # Wake word
    "WakeWordDetector",
    "get_wake_word_detector",
    # STT
    "SpeechToText",
    "TranscriptionResult",
    "get_stt",
    # TTS
    "TextToSpeech",
    "get_tts",
    # Pipeline
    "VoiceInput",
    "VoiceResult",
    "VoicePipeline",
    "get_pipeline",
]
