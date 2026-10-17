import wave
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional, Union
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

from .config import (
    TTS_OUTPUT_FILENAME,
    VOICES_DIR,
    DEFAULT_SPEECH_SPEED,
    PIPER_VOICE_MAP,
    DEFAULT_PIPER_VOICE,
    get_language_name,
)


class TTSEngine(ABC):
    """Abstract base class for Text-to-Speech engines."""

    @abstractmethod
    def speak(self, text: str, lang_code: str = "en") -> None:
        """Synthesize text to speech and play it out loud."""
        pass


class PiperTTSEngine(TTSEngine):
    """Lightweight neural Text-to-Speech manager using Piper ONNX models."""

    def __init__(
        self,
        speaker_device: str,
        voices_dir: Union[str, Path] = VOICES_DIR,
        length_scale: float = DEFAULT_SPEECH_SPEED,
        voice_map: Optional[Dict[str, str]] = None,
        default_voice: str = DEFAULT_PIPER_VOICE,
    ):
        self.speaker_device = speaker_device
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.length_scale = length_scale
        self.voice_map = voice_map if voice_map is not None else PIPER_VOICE_MAP
        self.default_voice = default_voice
        self.loaded_voices: Dict[str, PiperVoice] = {}

    def get_voice(self, lang_code: str) -> PiperVoice:
        """Load (and download if needed) the Piper voice model for a language code."""
        voice_name = self.voice_map.get(lang_code, self.default_voice)
        if voice_name in self.loaded_voices:
            return self.loaded_voices[voice_name]

        model_path = self.voices_dir / f"{voice_name}.onnx"
        if not model_path.exists():
            print(f"Downloading TTS voice '{voice_name}' for '{lang_code}'...")
            try:
                download_voice(voice_name, self.voices_dir)
            except Exception as e:
                print(
                    f"Warning: Could not download voice '{voice_name}' ({e}). "
                    f"Falling back to '{self.default_voice}'."
                )
                voice_name = self.default_voice
                model_path = self.voices_dir / f"{voice_name}.onnx"
                if not model_path.exists():
                    download_voice(voice_name, self.voices_dir)

        if voice_name not in self.loaded_voices:
            print(f"Loading TTS voice model '{voice_name}'...")
            self.loaded_voices[voice_name] = PiperVoice.load(model_path)

        return self.loaded_voices[voice_name]

    def preload_voices(
        self,
        forced_language: Optional[str] = None,
        default_cached_langs: Iterable[str] = ("en", "fr"),
    ) -> None:
        """Pre-load the forced language voice or existing cached voices for instant playback."""
        if forced_language:
            self.get_voice(forced_language)
            return

        for lang in default_cached_langs:
            voice_name = self.voice_map.get(lang)
            if voice_name and (self.voices_dir / f"{voice_name}.onnx").exists():
                self.get_voice(lang)

        if not self.loaded_voices:
            self.get_voice("en")

    def synthesize_to_file(
        self,
        text: str,
        lang_code: str = "en",
        output_wav: str = TTS_OUTPUT_FILENAME,
    ) -> Optional[str]:
        """Synthesize text to a WAV file without playing it."""
        clean_text = text.strip()
        if not clean_text:
            return None

        voice = self.get_voice(lang_code)
        syn_config = SynthesisConfig(length_scale=self.length_scale)

        with wave.open(output_wav, "wb") as wf:
            voice.synthesize_wav(clean_text, wf, syn_config=syn_config)

        return output_wav

    def speak(
        self,
        text: str,
        lang_code: str = "en",
        output_wav: str = TTS_OUTPUT_FILENAME,
    ) -> None:
        """Synthesize text to speech and play it through the configured speaker."""
        wav_path = self.synthesize_to_file(text, lang_code=lang_code, output_wav=output_wav)
        if not wav_path:
            return

        print(f"Speaking back ({get_language_name(lang_code)})...")
        subprocess.run(
            ["aplay", "-q", "-D", self.speaker_device, wav_path],
            check=False,
        )
