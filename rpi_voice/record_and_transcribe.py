import argparse
from typing import Optional

from engines import (
    DEFAULT_LANGUAGE,
    DEFAULT_SPEECH_SPEED,
    FasterWhisperSTT,
    HoldKeyAlsaRecorder,
    PiperTTSEngine,
    RecorderEngine,
    STTEngine,
    TTSEngine,
    TranscriptionResult,
    detect_usb_mic_device,
    detect_usb_speaker_device,
    get_language_name,
)


def print_transcription_result(result: TranscriptionResult) -> None:
    """Format and display a TranscriptionResult in the terminal."""
    print("\n" + "=" * 48)
    print(" RESULTS")
    print("=" * 48)
    if result.forced_language:
        lang_full = get_language_name(result.language)
        print(f"Forced Language   : {lang_full} ({result.language})")
    elif result.top_languages:
        print("Top 3 Language Matches:")
        for rank, (code, prob) in enumerate(result.top_languages[:3], start=1):
            name = get_language_name(code)
            marker = " (used for transcription)" if rank == 1 else ""
            print(f"  {rank}. {name:<12} ({code}) - {prob * 100:5.1f}%{marker}")
    else:
        lang_full = get_language_name(result.language)
        lang_prob = result.language_probability * 100
        print(f"Detected Language : {lang_full} ({result.language}) - {lang_prob:.1f}% confidence")
    print("-" * 48)

    if not result.segments:
        print("Transcription     : (No speech detected / silence)")
        print("=" * 48)
    else:
        print("Transcription:")
        for seg in result.segments:
            print(f"  [{seg.start:5.2f}s -> {seg.end:5.2f}s] {seg.text}")
        print("=" * 48)


def run_pipeline(
    recorder: RecorderEngine,
    stt_engine: STTEngine,
    tts_engine: Optional[TTSEngine] = None,
    language: Optional[str] = DEFAULT_LANGUAGE,
) -> None:
    """Run the interactive record -> transcribe -> speak loop."""
    try:
        while True:
            audio_path = recorder.record()
            if audio_path is None:
                # User requested to quit
                break
            if not audio_path:
                # Recording attempt failed or was too short; continue loop
                continue

            result = stt_engine.transcribe(audio_path, language=language)
            if result is None:
                continue

            print_transcription_result(result)

            if tts_engine and result.has_speech:
                tts_engine.speak(result.text, lang_code=result.language)
    except KeyboardInterrupt:
        pass
    print("\nExiting.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record audio while holding spacebar, transcribe with STT engine, and read back via TTS engine."
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Force transcription and TTS language using ISO code (e.g., 'en', 'fr', 'es'). Omit for auto-detection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        help="Whisper model size (default: 'tiny', e.g. 'base', 'small').",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable text-to-speech readback.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEECH_SPEED,
        help=f"TTS length scale (default: {DEFAULT_SPEECH_SPEED}; >1.0 is slower/clearer, <1.0 is faster).",
    )
    args = parser.parse_args()

    mic_device = detect_usb_mic_device()
    speaker_device = detect_usb_speaker_device()
    print(f"Target mic device    : {mic_device}")
    print(f"Target speaker device: {speaker_device}")
    if args.language:
        lang_name = get_language_name(args.language)
        print(f"Forced language      : {lang_name} ({args.language})")

    recorder = HoldKeyAlsaRecorder(device=mic_device)
    stt_engine = FasterWhisperSTT(model_size=args.model, compute_type="int8")

    tts_engine: Optional[TTSEngine] = None
    if not args.no_tts:
        piper_tts = PiperTTSEngine(speaker_device=speaker_device, length_scale=args.speed)
        piper_tts.preload_voices(forced_language=args.language)
        tts_engine = piper_tts

    run_pipeline(
        recorder=recorder,
        stt_engine=stt_engine,
        tts_engine=tts_engine,
        language=args.language,
    )


if __name__ == "__main__":
    main()
