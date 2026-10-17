from pathlib import Path

WAVE_OUTPUT_FILENAME = "/tmp/recorded_audio.wav"
WAVE_PREPROCESSED_FILENAME = "/tmp/recorded_audio_norm.wav"
TTS_OUTPUT_FILENAME = "/tmp/tts_output.wav"
VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
RATE = 16000
TRIGGER_KEY = " "        # Spacebar
TRIGGER_KEY_NAME = "SPACE"
RELEASE_TIMEOUT = 0.75   # Seconds without key repeat before treating as released
DEFAULT_LANGUAGE = None  # Set to e.g. "en" or "fr" to always force a language, or None for auto-detect
DEFAULT_STT_MODEL = "base"  # Upgraded from 'tiny' for much better multilingual & child-speech accuracy
DEFAULT_STT_BEAM_SIZE = 3   # Balanced beam size (avoids over-searching into English loanwords while staying fast)
DEFAULT_ALLOWED_LANGUAGES = ("en", "fr", "zh")  # Restrict auto-detection to active languages (None for all 99)
DEFAULT_VAD_PARAMETERS = {
    "min_silence_duration_ms": 400,
    "speech_pad_ms": 400,
    "threshold": 0.35,
}
# Voice bandpass (80Hz-7.5kHz) + dynamic audio normalization to boost quiet mic levels without clipping
STT_AUDIO_FILTER = "highpass=f=80,lowpass=f=7500,dynaudnorm=f=150:g=15:p=0.9:m=10"
DEFAULT_SPEECH_SPEED = 1.0  # length_scale > 1.0 speaks slower, < 1.0 speaks faster
DEFAULT_NOISE_SCALE = 0.85  # Higher noise_scale (default 0.667) adds more lively/goofy pitch & tone variation
DEFAULT_NOISE_W_SCALE = 0.9  # Phoneme duration variability for more animated cadence
DEFAULT_PRELOAD_LANGUAGES = ("en", "fr", "zh")
DEFAULT_VOICE_EFFECT = "helium"  # Default goofy voice effect ('chipmunk', 'helium', 'monster', 'robot', 'alien', 'wobble', 'random', 'none')

# Goofy FFmpeg audio filter templates ({sr} is replaced with the WAV sample rate)
VOICE_EFFECTS = {
    "none": "",
    "chipmunk": "asetrate={sr}*1.4,aresample={sr},atempo=1/1.2",
    "helium": "asetrate={sr}*1.65,aresample={sr},atempo=1/1.35",
    "monster": "asetrate={sr}*0.65,aresample={sr},atempo=1.35,aecho=0.8:0.88:40:0.4",
    "robot": "acrusher=level_in=1:level_out=1:bits=6:mode=log:aa=1,aecho=0.8:0.88:12:0.5,flanger=delay=2:depth=4:speed=5",
    "alien": "vibrato=f=8:d=0.8,flanger=delay=5:depth=5:speed=2,asetrate={sr}*1.2,aresample={sr},atempo=0.9",
    "wobble": "vibrato=f=6:d=1.0,apulsator=hz=3",
}

# Mapping common language codes to human-readable names
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish",
    "no": "Norwegian", "fi": "Finnish", "da": "Danish", "tr": "Turkish",
}

# Expressive Piper neural voices mapped by ISO language code
PIPER_VOICE_MAP = {
    "en": "en_GB-semaine-medium",
    "fr": "fr_FR-tom-medium",
    "zh": "zh_CN-huayan-medium",
    "es": "es_ES-sharvard-medium",
    "de": "de_DE-mls-medium",
    "it": "it_IT-paola-medium",
    "pt": "pt_BR-faber-medium",
    "nl": "nl_NL-alex-medium",
    "ru": "ru_RU-irina-medium",
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
DEFAULT_PIPER_VOICE = "en_GB-semaine-medium"

# Available downloaded/supported voices per language (including multiple Chinese voices)
AVAILABLE_PIPER_VOICES = {
    "en": [
        "en_GB-semaine-medium",
        "en_US-lessac-medium",
    ],
    "fr": [
        "fr_FR-tom-medium",
        "fr_FR-siwis-medium",
    ],
    "zh": [
        "zh_CN-huayan-medium",
        "zh_CN-xiao_ya-medium",
        "zh_CN-chaowen-medium",
    ],
}

# Optional default speaker IDs for multi-speaker models (e.g., SEMAINE speaker 3 = 'poppy', bubbly & animated)
PIPER_SPEAKER_MAP = {
    "en_GB-semaine-medium": 3,
}

# Default LLM configuration
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_LLM_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
    "echo": "echo",
}
DEFAULT_LLM_MODEL = DEFAULT_LLM_MODELS[DEFAULT_LLM_PROVIDER]
DEFAULT_MAX_CONTEXT_TURNS = 5       # Number of recent user/assistant back-and-forth exchanges to pass to LLM
DEFAULT_CONTEXT_IDLE_TIMEOUT = 600.0  # Seconds of inactivity (10 mins) before resetting conversation context
SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"

DEFAULT_SYSTEM_PROMPT = (
    "You are an educational voice assistant for a young child (around 4 to 8 years old). "
    "Follow these rules strictly:\n"
    "1. Always reply in the exact same language as the user's question (for example English, French, or Chinese).\n"
    "2. Only provide factual, educational, and age-appropriate explanations (science, nature, math, history, language, everyday how-things-work).\n"
    "3. Do NOT engage in roleplay, pretend play, fictional storytelling, or persona adoption. If asked to roleplay or for non-educational topics, gently redirect to an educational fact.\n"
    "4. Keep answers short, warm, clear, and easy for a small child to understand (1 to 3 simple sentences).\n"
    "5. Output plain spoken text only, suitable for Text-to-Speech. Never use markdown, bullet points, asterisks, symbols, or emojis."
)


def get_language_name(lang_code: str) -> str:
    """Return human-readable language name for an ISO language code."""
    if not lang_code:
        return "Unknown"
    return LANGUAGE_NAMES.get(lang_code, lang_code.upper())

