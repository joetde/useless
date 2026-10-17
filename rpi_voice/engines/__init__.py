from .audio_devices import detect_usb_mic_device, detect_usb_speaker_device
from .config import (
    DEFAULT_LANGUAGE,
    DEFAULT_PIPER_VOICE,
    DEFAULT_SPEECH_SPEED,
    LANGUAGE_NAMES,
    PIPER_VOICE_MAP,
    RATE,
    RELEASE_TIMEOUT,
    TRIGGER_KEY,
    TRIGGER_KEY_NAME,
    TTS_OUTPUT_FILENAME,
    VOICES_DIR,
    WAVE_OUTPUT_FILENAME,
    get_language_name,
)
from .recorder import HoldKeyAlsaRecorder, RecorderEngine
from .stt import FasterWhisperSTT, STTEngine, TranscriptionResult, TranscriptionSegment
from .tts import PiperTTSEngine, TTSEngine

__all__ = [
    "detect_usb_mic_device",
    "detect_usb_speaker_device",
    "DEFAULT_LANGUAGE",
    "DEFAULT_PIPER_VOICE",
    "DEFAULT_SPEECH_SPEED",
    "LANGUAGE_NAMES",
    "PIPER_VOICE_MAP",
    "RATE",
    "RELEASE_TIMEOUT",
    "TRIGGER_KEY",
    "TRIGGER_KEY_NAME",
    "TTS_OUTPUT_FILENAME",
    "VOICES_DIR",
    "WAVE_OUTPUT_FILENAME",
    "get_language_name",
    "RecorderEngine",
    "HoldKeyAlsaRecorder",
    "STTEngine",
    "FasterWhisperSTT",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TTSEngine",
    "PiperTTSEngine",
]
