import os
import random
import threading
import wave
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional, TYPE_CHECKING, Union
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

if TYPE_CHECKING:
    from .led import StatusLED

from .config import (
    TTS_OUTPUT_FILENAME,
    VOICES_DIR,
    DEFAULT_SPEECH_SPEED,
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_W_SCALE,
    DEFAULT_PRELOAD_LANGUAGES,
    DEFAULT_VOICE_EFFECT,
    VOICE_EFFECTS,
    PIPER_VOICE_MAP,
    PIPER_SPEAKER_MAP,
    DEFAULT_PIPER_VOICE,
    get_language_name,
)


class TTSEngine(ABC):
    """Abstract base class for Text-to-Speech engines."""

    @abstractmethod
    def speak(
        self,
        text: str,
        lang_code: str = "en",
        output_wav: str = TTS_OUTPUT_FILENAME,
        restore_led_ready: bool = False,
    ) -> None:
        """Synthesize text to speech and play it out loud."""
        pass

    def interrupt_playback(self) -> None:
        """Interrupt any currently playing audio."""
        pass


class PiperTTSEngine(TTSEngine):
    """Lightweight neural Text-to-Speech manager using Piper ONNX models."""

    def __init__(
        self,
        speaker_device: str,
        voices_dir: Union[str, Path] = VOICES_DIR,
        length_scale: float = DEFAULT_SPEECH_SPEED,
        noise_scale: float = DEFAULT_NOISE_SCALE,
        noise_w_scale: float = DEFAULT_NOISE_W_SCALE,
        voice_effect: str = DEFAULT_VOICE_EFFECT,
        voice_map: Optional[Dict[str, str]] = None,
        speaker_map: Optional[Dict[str, int]] = None,
        default_voice: str = DEFAULT_PIPER_VOICE,
        status_led: Optional["StatusLED"] = None,
    ):
        self.speaker_device = speaker_device
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.voice_effect = voice_effect
        self.voice_map = dict(voice_map) if voice_map is not None else dict(PIPER_VOICE_MAP)
        self.speaker_map = dict(speaker_map) if speaker_map is not None else dict(PIPER_SPEAKER_MAP)
        self.default_voice = default_voice
        self.status_led = status_led
        self.loaded_voices: Dict[str, PiperVoice] = {}
        self._lock = threading.Lock()
        self._playback_proc: Optional[subprocess.Popen] = None

    def get_voice_name(self, lang_code: str) -> str:
        """Return the configured Piper voice model name for a language code."""
        return self.voice_map.get(lang_code, self.default_voice)

    def get_voice(self, lang_code: str) -> PiperVoice:
        """Load (and download if needed) the Piper voice model for a language code."""
        voice_name = self.get_voice_name(lang_code)
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
        default_cached_langs: Iterable[str] = DEFAULT_PRELOAD_LANGUAGES,
    ) -> None:
        """Pre-load the forced language voice or default languages (en, fr, zh) for instant playback."""
        if forced_language:
            self.get_voice(forced_language)
            return

        for lang in default_cached_langs:
            self.get_voice(lang)

        if not self.loaded_voices:
            self.get_voice("en")

    def _resolve_effect(self) -> str:
        """Return the concrete effect name to apply (picking one if set to 'random')."""
        if not self.voice_effect or self.voice_effect == "none":
            return "none"
        if self.voice_effect == "random":
            choices = [k for k in VOICE_EFFECTS.keys() if k != "none"]
            return random.choice(choices)
        return self.voice_effect if self.voice_effect in VOICE_EFFECTS else "none"

    def _apply_audio_effect(self, wav_path: str, sample_rate: int, effect_name: str) -> None:
        """Apply a goofy FFmpeg audio filter in-place to the synthesized WAV file."""
        filter_template = VOICE_EFFECTS.get(effect_name, "")
        if not filter_template:
            return

        audio_filter = filter_template.format(sr=sample_rate)
        temp_out = f"{wav_path}.fx.wav"
        res = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                wav_path,
                "-af",
                audio_filter,
                temp_out,
            ],
            check=False,
        )
        if res.returncode == 0 and os.path.exists(temp_out):
            os.replace(temp_out, wav_path)

    def synthesize_to_file(
        self,
        text: str,
        lang_code: str = "en",
        output_wav: str = TTS_OUTPUT_FILENAME,
    ) -> Optional[str]:
        """Synthesize text to a WAV file (with optional goofy audio effect) without playing it."""
        clean_text = text.strip()
        if not clean_text:
            return None

        voice_name = self.get_voice_name(lang_code)
        voice = self.get_voice(lang_code)
        speaker_id = self.speaker_map.get(voice_name)
        if voice.config.num_speakers <= 1:
            speaker_id = None

        syn_config = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )

        with wave.open(output_wav, "wb") as wf:
            voice.synthesize_wav(clean_text, wf, syn_config=syn_config)

        effect_name = self._resolve_effect()
        self._last_applied_effect = effect_name
        if effect_name != "none":
            self._apply_audio_effect(output_wav, voice.config.sample_rate, effect_name)

        return output_wav

    def interrupt_playback(self) -> None:
        """Stop any currently playing `aplay` process immediately."""
        proc = self._playback_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def speak(
        self,
        text: str,
        lang_code: str = "en",
        output_wav: str = TTS_OUTPUT_FILENAME,
        restore_led_ready: bool = False,
    ) -> None:
        """Synthesize text to speech and play it through the configured speaker."""
        with self._lock:
            was_ready = self.status_led is not None and self.status_led.state == "ready"
            wav_path = self.synthesize_to_file(text, lang_code=lang_code, output_wav=output_wav)
            if not wav_path:
                if self.status_led is not None and not (restore_led_ready or was_ready):
                    self.status_led.off()
                return

            effect_label = getattr(self, "_last_applied_effect", "none")
            voice_name = self.get_voice_name(lang_code)
            fx_info = f", effect: {effect_label}" if effect_label != "none" else ""
            print(f"Speaking back ({get_language_name(lang_code)} [{voice_name}{fx_info}])...")
            if self.status_led is not None:
                self.status_led.off()
            try:
                self._playback_proc = subprocess.Popen(
                    ["aplay", "-q", "-D", self.speaker_device, wav_path]
                )
                self._playback_proc.wait()
            finally:
                self._playback_proc = None
                if self.status_led is not None and (restore_led_ready or was_ready):
                    self.status_led.pulse_ready()
