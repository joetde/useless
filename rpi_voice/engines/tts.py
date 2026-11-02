import os
import random
import subprocess
import threading
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, TYPE_CHECKING, Union
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

if TYPE_CHECKING:
    from .led import StatusLED

from .config import (
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_W_SCALE,
    DEFAULT_PIPER_VOICE,
    DEFAULT_PRELOAD_LANGUAGES,
    DEFAULT_SPEECH_SPEED,
    DEFAULT_VOICE_EFFECT,
    PIPER_SPEAKER_MAP,
    PIPER_VOICE_MAP,
    TTS_OUTPUT_FILENAME,
    VOICE_EFFECTS,
    VOICES_DIR,
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
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Synthesize text to speech and play it out loud."""

    def interrupt_playback(self) -> None:
        """Interrupt any currently playing audio."""


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
        self.voice_map = dict(voice_map or PIPER_VOICE_MAP)
        self.speaker_map = dict(speaker_map or PIPER_SPEAKER_MAP)
        self.default_voice = default_voice
        self.status_led = status_led
        self.loaded_voices: Dict[str, PiperVoice] = {}
        self._lock = threading.Lock()
        self._playback_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._abort_event = threading.Event()
        self._last_applied_effect: str = "none"

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
        """Pre-load the forced language voice or default languages (ko, fr, zh) for instant playback."""
        if forced_language:
            self.get_voice(forced_language)
            return

        for lang in default_cached_langs:
            self.get_voice(lang)

        if not self.loaded_voices:
            self.get_voice("ko")

    def _resolve_effect(self) -> str:
        """Return the concrete effect name to apply (picking one if set to 'random')."""
        if not self.voice_effect or self.voice_effect == "none":
            return "none"
        if self.voice_effect == "random":
            choices = [k for k in VOICE_EFFECTS if k != "none"]
            return random.choice(choices)
        return self.voice_effect if self.voice_effect in VOICE_EFFECTS else "none"

    def _apply_audio_effect(self, wav_path: str, sample_rate: int, effect_name: str) -> None:
        """Apply a goofy FFmpeg audio filter in-place to the synthesized WAV file."""
        filter_template = VOICE_EFFECTS.get(effect_name, "")
        if not filter_template:
            return

        audio_filter = filter_template.format(sr=sample_rate)
        temp_out = f"{wav_path}.fx.wav"
        try:
            res = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", wav_path,
                    "-af", audio_filter,
                    temp_out,
                ],
                check=False,
                timeout=10,
            )
            if res.returncode == 0 and os.path.exists(temp_out):
                os.replace(temp_out, wav_path)
        finally:
            if os.path.exists(temp_out):
                try:
                    os.remove(temp_out)
                except OSError:
                    pass

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
        speaker_id = self.speaker_map.get(voice_name) if voice.config.num_speakers > 1 else None

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

    @staticmethod
    def _terminate_process(proc: Optional[subprocess.Popen]) -> None:
        """Terminate a subprocess cleanly if it is still running."""
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
            except OSError:
                pass

    def interrupt_playback(self) -> None:
        """Stop any ongoing synthesis and currently playing `aplay`/`ffmpeg` processes immediately."""
        self._abort_event.set()
        self._terminate_process(self._playback_proc)
        self._terminate_process(self._ffmpeg_proc)

    def speak(
        self,
        text: str,
        lang_code: str = "en",
        output_wav: str = TTS_OUTPUT_FILENAME,
        restore_led_ready: bool = False,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Synthesize text sentence-by-sentence and stream directly through FFmpeg -> `aplay`."""
        clean_text = text.strip()
        if not clean_text:
            return

        def _is_aborted() -> bool:
            return self._abort_event.is_set() or (should_abort is not None and should_abort())

        with self._lock:
            self._abort_event.clear()
            if _is_aborted():
                return
            was_ready = self.status_led is not None and self.status_led.state == "ready"

            voice_name = self.get_voice_name(lang_code)
            voice = self.get_voice(lang_code)
            sample_rate = voice.config.sample_rate
            speaker_id = self.speaker_map.get(voice_name) if voice.config.num_speakers > 1 else None

            syn_config = SynthesisConfig(
                speaker_id=speaker_id,
                length_scale=self.length_scale,
                noise_scale=self.noise_scale,
                noise_w_scale=self.noise_w_scale,
            )

            effect_name = self._resolve_effect()
            self._last_applied_effect = effect_name
            filter_template = VOICE_EFFECTS.get(effect_name, "") if effect_name != "none" else ""
            audio_filter = filter_template.format(sr=sample_rate) if filter_template else ""

            aplay_cmd = [
                "aplay",
                "-q",
                "-D",
                self.speaker_device,
                "-f",
                "S16_LE",
                "-r",
                str(sample_rate),
                "-c",
                "1",
                "-t",
                "raw",
            ]

            ffmpeg_proc: Optional[subprocess.Popen] = None
            aplay_proc: Optional[subprocess.Popen] = None
            started_playback = False

            try:
                for chunk in voice.synthesize(clean_text, syn_config=syn_config):
                    if _is_aborted():
                        self._abort_event.set()
                        break
                    pcm_bytes = chunk.audio_int16_bytes
                    if not pcm_bytes:
                        continue

                    if not started_playback:
                        started_playback = True
                        fx_info = f", effect: {effect_name}" if effect_name != "none" else ""
                        print(f"Speaking back ({get_language_name(lang_code)} [{voice_name}{fx_info}])...")
                        if self.status_led is not None:
                            self.status_led.off()

                        if audio_filter:
                            ffmpeg_proc = subprocess.Popen(
                                [
                                    "ffmpeg",
                                    "-loglevel",
                                    "error",
                                    "-probesize",
                                    "32",
                                    "-f",
                                    "s16le",
                                    "-ar",
                                    str(sample_rate),
                                    "-ac",
                                    "1",
                                    "-i",
                                    "pipe:0",
                                    "-af",
                                    audio_filter,
                                    "-f",
                                    "s16le",
                                    "-ar",
                                    str(sample_rate),
                                    "-ac",
                                    "1",
                                    "pipe:1",
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                            )
                            self._ffmpeg_proc = ffmpeg_proc
                            aplay_proc = subprocess.Popen(
                                aplay_cmd,
                                stdin=ffmpeg_proc.stdout,
                            )
                            self._playback_proc = aplay_proc
                            if ffmpeg_proc.stdout is not None:
                                ffmpeg_proc.stdout.close()
                        else:
                            aplay_proc = subprocess.Popen(
                                aplay_cmd,
                                stdin=subprocess.PIPE,
                            )
                            self._playback_proc = aplay_proc

                    target_stdin = ffmpeg_proc.stdin if ffmpeg_proc is not None else (
                        aplay_proc.stdin if aplay_proc is not None else None
                    )
                    if target_stdin is None:
                        break
                    try:
                        target_stdin.write(pcm_bytes)
                        target_stdin.flush()
                    except (BrokenPipeError, OSError):
                        break

                # Close input pipes so ffmpeg and aplay flush remaining buffered audio and exit cleanly
                if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
                    try:
                        ffmpeg_proc.stdin.close()
                    except OSError:
                        pass
                elif aplay_proc is not None and aplay_proc.stdin is not None:
                    try:
                        aplay_proc.stdin.close()
                    except OSError:
                        pass

                while not _is_aborted():
                    ffmpeg_done = ffmpeg_proc is None or ffmpeg_proc.poll() is not None
                    aplay_done = aplay_proc is None or aplay_proc.poll() is not None
                    if ffmpeg_done and aplay_done:
                        break
                    self._abort_event.wait(0.02)
                else:
                    self._abort_event.set()
            finally:
                if self._abort_event.is_set():
                    self._terminate_process(aplay_proc)
                    self._terminate_process(ffmpeg_proc)
                self._playback_proc = None
                self._ffmpeg_proc = None
                if self.status_led is not None and not self._abort_event.is_set():
                    if restore_led_ready or was_ready:
                        self.status_led.pulse_ready()
                    elif not started_playback:
                        self.status_led.off()
