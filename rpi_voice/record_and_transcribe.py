import argparse
import signal
from typing import Optional

from engines import (
    AVAILABLE_PIPER_VOICES,
    DEFAULT_ALLOWED_LANGUAGES,
    DEFAULT_CONTEXT_IDLE_TIMEOUT,
    DEFAULT_GPIO_PIN,
    DEFAULT_LANG_SWITCH_PIN_1,
    DEFAULT_LANG_SWITCH_PIN_2,
    DEFAULT_LANGUAGE,
    DEFAULT_LED_GPIO_PIN,
    DEFAULT_LLM_MODELS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MAX_CONTEXT_TURNS,
    DEFAULT_SPEECH_SPEED,
    DEFAULT_STT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VOICE_EFFECT,
    FasterWhisperSTT,
    GPIOSwitchAlsaRecorder,
    HoldKeyAlsaRecorder,
    LLMEngine,
    LanguageSwitch,
    PIPER_VOICE_MAP,
    PiperTTSEngine,
    RecorderEngine,
    SESSIONS_DIR,
    STTEngine,
    SessionLogger,
    StatusLED,
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
    lang_full = get_language_name(result.language)
    if result.forced_language:
        print(f"Forced Language   : {lang_full} ({result.language})")
    elif result.top_languages:
        print("Top Language Matches:")
        for rank, (code, prob) in enumerate(result.top_languages[:3], start=1):
            name = get_language_name(code)
            marker = " (used for transcription)" if rank == 1 else ""
            print(f"  {rank}. {name:<12} ({code}) - {prob * 100:5.1f}%{marker}")
    else:
        print(f"Detected Language : {lang_full} ({result.language}) - {result.language_probability * 100:.1f}% confidence")
    print("-" * 48)

    if not result.segments:
        print("Transcription     : (No speech detected / silence)")
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
    session_logger: Optional[SessionLogger] = None,
    status_led: Optional[StatusLED] = None,
    language: Optional[str] = DEFAULT_LANGUAGE,
    language_switch: Optional[LanguageSwitch] = None,
) -> None:
    """Run the interactive record -> transcribe -> LLM -> speak loop."""
    if tts_engine is not None and hasattr(recorder, "on_press"):
        recorder.on_press = tts_engine.interrupt_playback

    while True:
        audio_path = recorder.record()
        if audio_path is None:
            break
        if not audio_path:
            continue

        if status_led is not None:
            status_led.solid_thinking()

        context_history = (
            session_logger.get_context_for_next_turn() if session_logger is not None else []
        )
        last_topic_hint = context_history[-1].user_text if context_history else None

        if recorder.is_triggered():
            print("  * [Interrupted] Button pressed — restarting round...")
            continue

        active_language = (
            language_switch.current_language if language_switch is not None else language
        )
        result = stt_engine.transcribe(
            audio_path,
            language=active_language,
            context_hint=last_topic_hint,
        )
        if result is None:
            continue
        if recorder.is_triggered():
            print("  * [Interrupted] Button pressed — restarting round...")
            continue

        print_transcription_result(result)
        if not result.has_speech:
            continue

        response_language = (
            language_switch.current_language if language_switch is not None else result.language
        )
        spoken_text = result.text
        llm_reply: Optional[str] = None

        if llm_engine is not None:
            llm_reply = llm_engine.generate(
                result.text,
                language=response_language,
                history=context_history,
            )
            if recorder.is_triggered():
                print("  * [Interrupted] Button pressed — restarting round...")
                continue
            if not llm_reply:
                print("Warning: No response received from LLM.")
                if session_logger is not None:
                    session_logger.log_turn(
                        transcription=result,
                        llm_response=None,
                        context_turns_used=len(context_history),
                    )
                continue
            print_llm_response(llm_reply)
            spoken_text = llm_reply

        if session_logger is not None:
            session_logger.log_turn(
                transcription=result,
                llm_response=llm_reply,
                context_turns_used=len(context_history),
            )

        if recorder.is_triggered():
            print("  * [Interrupted] Button pressed — restarting round...")
            continue

        if tts_engine is not None:
            final_speak_lang = (
                language_switch.current_language
                if language_switch is not None
                else response_language
            )
            tts_engine.speak(
                spoken_text,
                lang_code=final_speak_lang,
                should_abort=recorder.is_triggered,
            )
            if recorder.is_triggered():
                print("  * [Interrupted] Button pressed during playback — starting new recording...")
        elif status_led is not None:
            status_led.off()


def parse_args() -> argparse.Namespace:
    effect_choices = [*VOICE_EFFECTS.keys(), "random"]
    default_allowed_str = (
        ",".join(DEFAULT_ALLOWED_LANGUAGES) if DEFAULT_ALLOWED_LANGUAGES else "all"
    )

    parser = argparse.ArgumentParser(
        description="Record audio via GPIO switch or spacebar, transcribe with STT, ask an educational LLM, and speak via TTS."
    )
    parser.add_argument(
        "--gpio-pin",
        type=int,
        default=DEFAULT_GPIO_PIN,
        help=f"BCM GPIO pin for the record switch connected to GND (default: {DEFAULT_GPIO_PIN}, physical pin 40).",
    )
    parser.add_argument(
        "--led-pin",
        type=int,
        default=DEFAULT_LED_GPIO_PIN,
        help=f"BCM GPIO pin for the status LED connected to GND (default: {DEFAULT_LED_GPIO_PIN}, physical pin 33).",
    )
    parser.add_argument("--no-led", action="store_true", help="Disable the GPIO status LED.")
    parser.add_argument(
        "--lang-pin-1",
        type=int,
        default=DEFAULT_LANG_SWITCH_PIN_1,
        help=f"BCM GPIO pin for language switch State I -> French (default: {DEFAULT_LANG_SWITCH_PIN_1}, physical pin 3).",
    )
    parser.add_argument(
        "--lang-pin-2",
        type=int,
        default=DEFAULT_LANG_SWITCH_PIN_2,
        help=f"BCM GPIO pin for language switch State II -> Chinese (default: {DEFAULT_LANG_SWITCH_PIN_2}, physical pin 5; both HIGH -> Korean).",
    )
    parser.add_argument(
        "--no-lang-switch",
        action="store_true",
        help="Disable the physical GPIO language selector switch.",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Use terminal spacebar hold trigger instead of the physical GPIO switch.",
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Force transcription and TTS language using ISO code (e.g., 'ko', 'fr', 'zh'). Omit for auto-detection.",
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
        choices=list(DEFAULT_LLM_MODELS.keys()),
        default=DEFAULT_LLM_PROVIDER,
        help=f"LLM provider to use (default: '{DEFAULT_LLM_PROVIDER}'). Use 'echo' to repeat back.",
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
        "--context-turns",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TURNS,
        help=f"Number of previous user/assistant exchanges to pass to the LLM (default: {DEFAULT_MAX_CONTEXT_TURNS}, 0 to disable).",
    )
    parser.add_argument(
        "--context-idle-timeout",
        type=float,
        default=DEFAULT_CONTEXT_IDLE_TIMEOUT,
        help=f"Seconds of inactivity before resetting LLM conversation context (default: {DEFAULT_CONTEXT_IDLE_TIMEOUT:.0f}s).",
    )
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default=str(SESSIONS_DIR),
        help=f"Directory where per-session audit folders are stored (default: '{SESSIONS_DIR}').",
    )
    parser.add_argument(
        "--no-session-log",
        action="store_true",
        help="Disable saving session transcripts and audit logs to disk.",
    )
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech readback.")
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEECH_SPEED,
        help=f"TTS length scale (default: {DEFAULT_SPEECH_SPEED}; >1.0 is slower, <1.0 is faster).",
    )
    parser.add_argument(
        "-e", "--effect",
        type=str,
        choices=effect_choices,
        default=DEFAULT_VOICE_EFFECT,
        help=f"Goofy voice effect to apply (default: '{DEFAULT_VOICE_EFFECT}'). Use 'random' or 'none'.",
    )
    parser.add_argument(
        "--zh-voice",
        type=str,
        choices=AVAILABLE_PIPER_VOICES["zh"],
        default=PIPER_VOICE_MAP["zh"],
        help=f"Chinese Piper voice model to use (default: '{PIPER_VOICE_MAP['zh']}').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    if args.keyboard:
        print("Recording trigger    : Keyboard (SPACE)")
    else:
        print(f"Recording trigger    : Physical switch on GPIO {args.gpio_pin} (active LOW to GND)")
    if args.no_led:
        print("Status LED           : disabled")
    else:
        print(f"Status LED           : GPIO {args.led_pin} (pulsing=ready, off=recording, solid=thinking)")
    if args.language:
        print(f"Forced language      : {get_language_name(args.language)} ({args.language})")
    elif allowed_languages:
        print(f"Allowed languages    : {', '.join(allowed_languages)}")
    if not args.no_tts:
        print(f"Goofy voice effect   : {args.effect}")

    llm_engine: Optional[LLMEngine] = None
    resolved_model = ""
    effective_provider = "echo" if args.no_llm else args.llm_provider
    if effective_provider != "echo":
        resolved_model = args.llm_model or DEFAULT_LLM_MODELS.get(args.llm_provider, "")
        print(f"LLM provider / model : {args.llm_provider} ({resolved_model})")
        print(
            f"Context continuation : last {max(0, args.context_turns)} turn(s) "
            f"(idle reset: {args.context_idle_timeout:.0f}s)"
        )
        llm_engine = create_llm_engine(
            provider=args.llm_provider,
            model=args.llm_model,
            api_key=args.api_key,
            base_url=args.llm_base_url,
            system_prompt=args.system_prompt,
        )
    else:
        print("LLM provider / model : disabled (echo mode)")

    session_logger = SessionLogger(
        sessions_dir=args.sessions_dir,
        max_context_turns=args.context_turns,
        context_idle_timeout=args.context_idle_timeout,
        session_metadata={
            "stt_model": args.model,
            "forced_language": args.language,
            "allowed_languages": allowed_languages,
            "llm_provider": effective_provider,
            "llm_model": resolved_model or effective_provider,
            "voice_effect": None if args.no_tts else args.effect,
        },
        enabled=not args.no_session_log,
    )
    if not args.no_session_log:
        print(f"Session audit folder : {session_logger.session_dir}")

    status_led: Optional[StatusLED] = None
    recorder: Optional[RecorderEngine] = None
    language_switch: Optional[LanguageSwitch] = None

    def _handle_termination(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_termination)
    signal.signal(signal.SIGTERM, _handle_termination)

    try:
        status_led = StatusLED(gpio_pin=None if args.no_led else args.led_pin)

        if not args.no_lang_switch and args.language is None:
            language_switch = LanguageSwitch(
                pin_1=args.lang_pin_1,
                pin_2=args.lang_pin_2,
            )
            print(
                f"Language switch      : GPIO {args.lang_pin_1} (I=fr), "
                f"GPIO {args.lang_pin_2} (II=zh), both HIGH=ko "
                f"[current: {language_switch.describe_state()}]"
            )
        elif args.no_lang_switch:
            print("Language switch      : disabled")

        if args.keyboard:
            recorder = HoldKeyAlsaRecorder(device=mic_device, status_led=status_led)
        else:
            recorder = GPIOSwitchAlsaRecorder(
                device=mic_device,
                gpio_pin=args.gpio_pin,
                status_led=status_led,
            )

        stt_engine = FasterWhisperSTT(
            model_size=args.model,
            compute_type="int8",
            allowed_languages=allowed_languages,
            preprocess_audio=not args.no_audio_preprocess,
        )

        tts_engine: Optional[TTSEngine] = None
        if not args.no_tts:
            custom_voice_map = {**PIPER_VOICE_MAP, "zh": args.zh_voice}
            piper_tts = PiperTTSEngine(
                speaker_device=speaker_device,
                length_scale=args.speed,
                voice_effect=args.effect,
                voice_map=custom_voice_map,
                status_led=status_led,
            )
            piper_tts.preload_voices(
                forced_language=None if language_switch is not None else args.language
            )
            tts_engine = piper_tts

        if language_switch is not None:
            language_switch.tts_engine = tts_engine
            language_switch.recorder = recorder
            language_switch.start_monitoring()

        run_pipeline(
            recorder=recorder,
            stt_engine=stt_engine,
            llm_engine=llm_engine,
            tts_engine=tts_engine,
            session_logger=session_logger,
            status_led=status_led,
            language=args.language,
            language_switch=language_switch,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")
    finally:
        if language_switch is not None:
            language_switch.close()
        if recorder is not None:
            recorder.close()
        if status_led is not None:
            status_led.close()
        session_logger.close()
    print("Exiting.")


if __name__ == "__main__":
    main()
