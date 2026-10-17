import argparse
from typing import Optional

from engines import (
    AVAILABLE_PIPER_VOICES,
    DEFAULT_ALLOWED_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_LLM_MODELS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_SPEECH_SPEED,
    DEFAULT_STT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VOICE_EFFECT,
    FasterWhisperSTT,
    HoldKeyAlsaRecorder,
    LLMEngine,
    PIPER_VOICE_MAP,
    PiperTTSEngine,
    RecorderEngine,
    STTEngine,
    TTSEngine,
    TranscriptionResult,
    VOICE_EFFECTS,
    create_llm_engine,
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
        print("Top Language Matches:")
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


def print_llm_response(response_text: str) -> None:
    """Format and display the LLM answer in the terminal."""
    print("LLM Response:")
    print(f"  {response_text}")
    print("=" * 48)


def run_pipeline(
    recorder: RecorderEngine,
    stt_engine: STTEngine,
    llm_engine: Optional[LLMEngine] = None,
    tts_engine: Optional[TTSEngine] = None,
    language: Optional[str] = DEFAULT_LANGUAGE,
) -> None:
    """Run the interactive record -> transcribe -> LLM -> speak loop."""
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

            if not result.has_speech:
                continue

            spoken_text = result.text
            if llm_engine is not None:
                llm_reply = llm_engine.generate(result.text, language=result.language)
                if not llm_reply:
                    print("Warning: No response received from LLM.")
                    continue
                print_llm_response(llm_reply)
                spoken_text = llm_reply

            if tts_engine:
                tts_engine.speak(spoken_text, lang_code=result.language)
    except KeyboardInterrupt:
        pass
    print("\nExiting.")


def main() -> None:
    effect_choices = list(VOICE_EFFECTS.keys()) + ["random"]
    zh_voice_choices = AVAILABLE_PIPER_VOICES["zh"]
    llm_provider_choices = list(DEFAULT_LLM_MODELS.keys())
    default_allowed_str = ",".join(DEFAULT_ALLOWED_LANGUAGES) if DEFAULT_ALLOWED_LANGUAGES else "all"

    parser = argparse.ArgumentParser(
        description="Record audio while holding spacebar, transcribe with STT, ask an educational LLM, and speak the answer via TTS."
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Force transcription and TTS language using ISO code (e.g., 'en', 'fr', 'zh', 'es'). Omit for auto-detection.",
    )
    parser.add_argument(
        "--allowed-languages",
        type=str,
        default=default_allowed_str,
        help=f"Comma-separated ISO codes to restrict language auto-detection (default: '{default_allowed_str}'). Pass 'all' for all 99 Whisper languages.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_STT_MODEL,
        help=f"Whisper STT model size (default: '{DEFAULT_STT_MODEL}', e.g. 'tiny', 'base', 'small').",
    )
    parser.add_argument(
        "--no-audio-preprocess",
        action="store_true",
        help="Disable FFmpeg voice bandpass filter and dynamic volume normalization before STT.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=llm_provider_choices,
        default=DEFAULT_LLM_PROVIDER,
        help=f"LLM provider to use (default: '{DEFAULT_LLM_PROVIDER}'). Use 'echo' to simply repeat back.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=f"LLM model name override (default for gemini: '{DEFAULT_LLM_MODELS['gemini']}').",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Optional custom base URL for OpenAI-compatible or Ollama LLM providers.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional LLM API key (defaults to GEMINI_API_KEY / OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override the system prompt controlling LLM behavior.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM query and simply repeat the transcribed text back.",
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
    parser.add_argument(
        "-e", "--effect",
        type=str,
        choices=effect_choices,
        default=DEFAULT_VOICE_EFFECT,
        help=f"Goofy voice effect to apply (default: '{DEFAULT_VOICE_EFFECT}'). Use 'random' to pick a surprise effect each time, or 'none' for raw voice.",
    )
    parser.add_argument(
        "--zh-voice",
        type=str,
        choices=zh_voice_choices,
        default=PIPER_VOICE_MAP["zh"],
        help=f"Chinese Piper voice model to use (default: '{PIPER_VOICE_MAP['zh']}').",
    )
    args = parser.parse_args()

    allowed_languages = None
    if args.allowed_languages and args.allowed_languages.strip().lower() != "all":
        allowed_languages = [
            code.strip().lower()
            for code in args.allowed_languages.split(",")
            if code.strip()
        ]

    mic_device = detect_usb_mic_device()
    speaker_device = detect_usb_speaker_device()
    print(f"Target mic device    : {mic_device}")
    print(f"Target speaker device: {speaker_device}")
    if args.language:
        lang_name = get_language_name(args.language)
        print(f"Forced language      : {lang_name} ({args.language})")
    elif allowed_languages:
        print(f"Allowed languages    : {', '.join(allowed_languages)}")
    if not args.no_tts:
        print(f"Goofy voice effect   : {args.effect}")

    llm_engine: Optional[LLMEngine] = None
    if not args.no_llm and args.llm_provider != "echo":
        resolved_model = args.llm_model or DEFAULT_LLM_MODELS.get(args.llm_provider, "")
        print(f"LLM provider / model : {args.llm_provider} ({resolved_model})")
        llm_engine = create_llm_engine(
            provider=args.llm_provider,
            model=args.llm_model,
            api_key=args.api_key,
            base_url=args.llm_base_url,
            system_prompt=args.system_prompt,
        )
    else:
        print("LLM provider / model : disabled (echo mode)")

    recorder = HoldKeyAlsaRecorder(device=mic_device)
    stt_engine = FasterWhisperSTT(
        model_size=args.model,
        compute_type="int8",
        allowed_languages=allowed_languages,
        preprocess_audio=not args.no_audio_preprocess,
    )

    tts_engine: Optional[TTSEngine] = None
    if not args.no_tts:
        custom_voice_map = dict(PIPER_VOICE_MAP)
        custom_voice_map["zh"] = args.zh_voice
        piper_tts = PiperTTSEngine(
            speaker_device=speaker_device,
            length_scale=args.speed,
            voice_effect=args.effect,
            voice_map=custom_voice_map,
        )
        piper_tts.preload_voices(forced_language=args.language)
        tts_engine = piper_tts

    run_pipeline(
        recorder=recorder,
        stt_engine=stt_engine,
        llm_engine=llm_engine,
        tts_engine=tts_engine,
        language=args.language,
    )


if __name__ == "__main__":
    main()

