import os
import re
import sys
import time
import wave
import select
import termios
import tty
import argparse
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice

WAVE_OUTPUT_FILENAME = "/tmp/recorded_audio.wav"
TTS_OUTPUT_FILENAME = "/tmp/tts_output.wav"
VOICES_DIR = Path(__file__).resolve().parent / "voices"
RATE = 16000
TRIGGER_KEY = " "        # Spacebar
TRIGGER_KEY_NAME = "SPACE"
RELEASE_TIMEOUT = 0.75   # Seconds without key repeat before treating as released
DEFAULT_LANGUAGE = None  # Set to e.g. "en" or "fr" to always force a language, or None for auto-detect
DEFAULT_SPEECH_SPEED = 1.1  # length_scale > 1.0 speaks slightly slower and clearer for kids

# Mapping common language codes to human-readable names
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish",
    "no": "Norwegian", "fi": "Finnish", "da": "Danish", "tr": "Turkish",
}

# Clear, natural Piper neural voices mapped by ISO language code
PIPER_VOICE_MAP = {
    "en": "en_US-lessac-medium",
    "fr": "fr_FR-siwis-medium",
    "es": "es_ES-sharvard-medium",
    "de": "de_DE-mls-medium",
    "it": "it_IT-paola-medium",
    "pt": "pt_BR-faber-medium",
    "nl": "nl_NL-alex-medium",
    "ru": "ru_RU-irina-medium",
    "zh": "zh_CN-huayan-medium",
    "uk": "uk_UA-ukrainian_tts-medium",
    "pl": "pl_PL-gosia-medium",
    "sv": "sv_SE-nst-medium",
    "no": "no_NO-talesyntese-medium",
    "fi": "fi_FI-harri-medium",
    "da": "da_DK-talesyntese-medium",
    "tr": "tr_TR-dfki-medium",
    "ar": "ar_JO-kareem-medium",
    "hi": "hi_IN-priyamvada-medium",
}
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"


def detect_usb_mic_device():
    """Detect ALSA card name or index for the USB microphone."""
    try:
        output = subprocess.check_output(["arecord", "-l"], text=True)
        # Look for USB card name like: card 2: Device [USB PnP Sound Device]
        match = re.search(r"card\s+(\d+):\s+([a-zA-Z0-9_-]+)\s+\[.*USB.*\]", output, re.IGNORECASE)
        if match:
            card_id = match.group(2)  # Use card name (e.g., 'Device')
            return f"plughw:CARD={card_id},DEV=0"
    except Exception:
        pass
    # Fallback to plughw:CARD=Device,DEV=0 or default
    return "plughw:CARD=Device,DEV=0"


def detect_usb_speaker_device():
    """Detect ALSA card name or index for the USB speaker."""
    try:
        output = subprocess.check_output(["aplay", "-l"], text=True)
        # Look for USB playback device like: card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
        match = re.search(r"card\s+(\d+):\s+([a-zA-Z0-9_-]+)\s+\[.*\].*USB", output, re.IGNORECASE)
        if match:
            card_id = match.group(2)  # Use card name (e.g., 'UACDemoV10')
            return f"plughw:CARD={card_id},DEV=0"
    except Exception:
        pass
    # Fallback to default ALSA playback device
    return "default"


class TTSEngine:
    """Lightweight neural Text-to-Speech manager using Piper ONNX models."""

    def __init__(self, speaker_device, voices_dir=VOICES_DIR, length_scale=DEFAULT_SPEECH_SPEED):
        self.speaker_device = speaker_device
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.length_scale = length_scale
        self.loaded_voices = {}

    def get_voice(self, lang_code):
        """Load (and download if needed) the Piper voice model for a language code."""
        voice_name = PIPER_VOICE_MAP.get(lang_code, DEFAULT_PIPER_VOICE)
        if voice_name in self.loaded_voices:
            return self.loaded_voices[voice_name]

        model_path = self.voices_dir / f"{voice_name}.onnx"
        if not model_path.exists():
            print(f"Downloading TTS voice '{voice_name}' for '{lang_code}'...")
            try:
                download_voice(voice_name, self.voices_dir)
            except Exception as e:
                print(f"Warning: Could not download voice '{voice_name}' ({e}). Falling back to '{DEFAULT_PIPER_VOICE}'.")
                voice_name = DEFAULT_PIPER_VOICE
                model_path = self.voices_dir / f"{voice_name}.onnx"
                if not model_path.exists():
                    download_voice(voice_name, self.voices_dir)

        if voice_name not in self.loaded_voices:
            print(f"Loading TTS voice model '{voice_name}'...")
            self.loaded_voices[voice_name] = PiperVoice.load(model_path)

        return self.loaded_voices[voice_name]

    def speak(self, text, lang_code="en", output_wav=TTS_OUTPUT_FILENAME):
        """Synthesize text to speech and play it through the USB speaker."""
        clean_text = text.strip()
        if not clean_text:
            return

        voice = self.get_voice(lang_code)
        syn_config = SynthesisConfig(length_scale=self.length_scale)

        with wave.open(output_wav, "wb") as wf:
            voice.synthesize_wav(clean_text, wf, syn_config=syn_config)

        print(f"Speaking back ({ LANGUAGE_NAMES.get(lang_code, lang_code.upper()) })...")
        subprocess.run(
            ["aplay", "-q", "-D", self.speaker_device, output_wav],
            check=False,
        )


def wait_and_record_while_held(device, output_file=WAVE_OUTPUT_FILENAME):
    """
    Waits for TRIGGER_KEY to be held down in the terminal, records audio while held,
    and stops when the key is released (or pressed again to stop).
    Returns True if audio was recorded, False to quit.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print(f"\n[Ready] Hold [{TRIGGER_KEY_NAME}] to record (release to stop), or press 'q' to quit...")

    try:
        tty.setcbreak(fd)

        # Wait for trigger key press
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                ch = sys.stdin.read(1)
                if ch in ("q", "Q", "\x03"):  # 'q' or Ctrl+C
                    return False
                if ch == TRIGGER_KEY:
                    break

        # Remove previous file so we can verify fresh recording
        if os.path.exists(output_file):
            os.remove(output_file)

        cmd = [
            "arecord",
            "-D", device,
            "-f", "S16_LE",
            "-r", str(RATE),
            "-c", "1",
            output_file
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        start_time = time.time()
        last_event_time = start_time

        while True:
            elapsed = time.time() - start_time
            print(f"\r  * [RECORDING] {elapsed:4.1f}s (Keep holding [{TRIGGER_KEY_NAME}]...) ", end="", flush=True)

            # Check if arecord exited prematurely (e.g. device error)
            if process.poll() is not None:
                break

            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                # Drain all buffered key repeats
                chars = ""
                while select.select([sys.stdin], [], [], 0)[0]:
                    chars += sys.stdin.read(1)
                if "\x03" in chars or "q" in chars or "Q" in chars:
                    process.terminate()
                    process.wait()
                    return False
                if TRIGGER_KEY in chars:
                    last_event_time = time.time()
            else:
                # No key repeat received within timeout -> key was released
                if time.time() - last_event_time > RELEASE_TIMEOUT:
                    break

        # Stop recording cleanly so WAV header is finalized
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

        total_time = time.time() - start_time
        print(f"\r  * [RECORDING] Stopped ({total_time:.1f}s). Processing audio...          ")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    if not os.path.exists(output_file) or os.path.getsize(output_file) <= 44:
        print("Warning: Recording failed or was too short.")
        return True

    print(f"* Audio saved to {output_file} ({os.path.getsize(output_file):,} bytes)")
    return True


def transcribe_and_detect(model, tts_engine=None, audio_path=WAVE_OUTPUT_FILENAME, language=DEFAULT_LANGUAGE):
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) <= 44:
        return

    if language:
        lang_name = LANGUAGE_NAMES.get(language, language.upper())
        print(f"Transcribing in forced language: {lang_name} ({language})...")
    else:
        print("Transcribing and analyzing language...")

    # vad_filter=True filters out background silence to prevent hallucinations
    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
    )
    segments = list(segments_gen)

    detected_lang = language or info.language or "en"

    print("\n" + "=" * 48)
    print(" RESULTS")
    print("=" * 48)
    if language:
        lang_full = LANGUAGE_NAMES.get(language, language.upper())
        print(f"Forced Language   : {lang_full} ({language})")
    elif info.all_language_probs:
        print("Top 3 Language Matches:")
        for rank, (code, prob) in enumerate(info.all_language_probs[:3], start=1):
            name = LANGUAGE_NAMES.get(code, code.upper())
            marker = " (used for transcription)" if rank == 1 else ""
            print(f"  {rank}. {name:<12} ({code}) - {prob * 100:5.1f}%{marker}")
    else:
        lang_code = info.language
        lang_full = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        lang_prob = info.language_probability * 100
        print(f"Detected Language : {lang_full} ({lang_code}) - {lang_prob:.1f}% confidence")
    print("-" * 48)

    if not segments:
        print("Transcription     : (No speech detected / silence)")
        print("=" * 48)
    else:
        print("Transcription:")
        full_text_parts = []
        for seg in segments:
            text_part = seg.text.strip()
            print(f"  [{seg.start:5.2f}s -> {seg.end:5.2f}s] {text_part}")
            if text_part:
                full_text_parts.append(text_part)
        print("=" * 48)

        full_text = " ".join(full_text_parts)
        if tts_engine and full_text:
            tts_engine.speak(full_text, lang_code=detected_lang)


def main():
    parser = argparse.ArgumentParser(description="Record audio while holding spacebar, transcribe with Whisper, and read back via Piper TTS.")
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Force transcription and TTS language using ISO code (e.g., 'en', 'fr', 'es'). Omit for auto-detection."
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech readback."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEECH_SPEED,
        help=f"TTS length scale (default: {DEFAULT_SPEECH_SPEED}; >1.0 is slower/clearer, <1.0 is faster)."
    )
    args = parser.parse_args()

    mic_device = detect_usb_mic_device()
    speaker_device = detect_usb_speaker_device()
    print(f"Target mic device    : {mic_device}")
    print(f"Target speaker device: {speaker_device}")
    if args.language:
        lang_name = LANGUAGE_NAMES.get(args.language, args.language.upper())
        print(f"Forced language      : {lang_name} ({args.language})")

    print("Loading Whisper model ('tiny') with INT8 precision...")
    # Pre-load once so recording starts & transcribes immediately on button press
    model = WhisperModel("tiny", compute_type="int8")

    tts_engine = None
    if not args.no_tts:
        tts_engine = TTSEngine(speaker_device=speaker_device, length_scale=args.speed)
        # Pre-load the forced language voice (or cached voices like en/fr) so playback is instant
        if args.language:
            tts_engine.get_voice(args.language)
        else:
            for lang in ("en", "fr"):
                voice_file = VOICES_DIR / f"{PIPER_VOICE_MAP[lang]}.onnx"
                if voice_file.exists():
                    tts_engine.get_voice(lang)
            if not tts_engine.loaded_voices:
                tts_engine.get_voice("en")

    try:
        while wait_and_record_while_held(mic_device):
            transcribe_and_detect(model, tts_engine=tts_engine, language=args.language)
    except KeyboardInterrupt:
        pass
    print("\nExiting.")


if __name__ == "__main__":
    main()

