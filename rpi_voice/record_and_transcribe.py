import os
import re
import sys
import time
import select
import termios
import tty
import argparse
import subprocess
from faster_whisper import WhisperModel

WAVE_OUTPUT_FILENAME = "/tmp/recorded_audio.wav"
RATE = 16000
TRIGGER_KEY = " "        # Spacebar
TRIGGER_KEY_NAME = "SPACE"
RELEASE_TIMEOUT = 0.75   # Seconds without key repeat before treating as released
DEFAULT_LANGUAGE = None  # Set to e.g. "en" or "fr" to always force a language, or None for auto-detect

# Mapping common language codes to human-readable names
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish",
    "no": "Norwegian", "fi": "Finnish", "da": "Danish", "tr": "Turkish",
}

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

def transcribe_and_detect(model, audio_path=WAVE_OUTPUT_FILENAME, language=DEFAULT_LANGUAGE):
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
    else:
        print("Transcription:")
        for seg in segments:
            print(f"  [{seg.start:5.2f}s -> {seg.end:5.2f}s] {seg.text.strip()}")
    print("=" * 48)

def main():
    parser = argparse.ArgumentParser(description="Record audio while holding spacebar and transcribe with Whisper.")
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Force transcription language using ISO code (e.g., 'en', 'fr', 'es'). Omit for auto-detection."
    )
    args = parser.parse_args()

    device = detect_usb_mic_device()
    print(f"Target audio device: {device}")
    if args.language:
        lang_name = LANGUAGE_NAMES.get(args.language, args.language.upper())
        print(f"Forced language: {lang_name} ({args.language})")

    print("Loading Whisper model ('tiny') with INT8 precision...")
    # Pre-load once so recording starts & transcribes immediately on button press
    model = WhisperModel("tiny", compute_type="int8")

    try:
        while wait_and_record_while_held(device):
            transcribe_and_detect(model, language=args.language)
    except KeyboardInterrupt:
        pass
    print("\nExiting.")

if __name__ == "__main__":
    main()
