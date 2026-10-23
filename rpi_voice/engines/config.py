from pathlib import Path

WAVE_OUTPUT_FILENAME = "/tmp/recorded_audio.wav"
WAVE_PREPROCESSED_FILENAME = "/tmp/recorded_audio_norm.wav"
TTS_OUTPUT_FILENAME = "/tmp/tts_output.wav"
TTS_LANG_SWITCH_FILENAME = "/tmp/tts_lang_switch.wav"

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"

RATE = 16000
TRIGGER_KEY = " "        # Spacebar
TRIGGER_KEY_NAME = "SPACE"
RELEASE_TIMEOUT = 0.75   # Seconds without key repeat before treating as released
DEFAULT_POST_RELEASE_DURATION = 0.25  # Extra seconds recorded after switch release so trailing syllables aren't clipped

# GPIO pin numbers (BCM numbering)
DEFAULT_GPIO_PIN = 21            # Physical pin 40, switch connected between GPIO 21 and GND
DEFAULT_GPIO_BOUNCE_TIME = 0.05  # Seconds of debounce time for physical switch
DEFAULT_LANG_SWITCH_PIN_1 = 2    # Physical pin 3: switch state I -> French ('fr')
DEFAULT_LANG_SWITCH_PIN_2 = 3    # Physical pin 5: switch state II -> Chinese ('zh')
DEFAULT_LANG_SWITCH_SETTLE_TIME = 0.20  # Seconds switch state must remain stable before triggering
DEFAULT_LED_GPIO_PIN = 13        # Physical pin 33, LED connected between GPIO 13 and GND
DEFAULT_LED_PULSE_FADE_IN = 1.0
DEFAULT_LED_PULSE_FADE_OUT = 1.0

# Speech-to-Text settings
DEFAULT_LANGUAGE = None  # Force ISO language code (e.g. "en", "fr") or None for auto-detect
DEFAULT_STT_MODEL = "base"
DEFAULT_STT_BEAM_SIZE = 5
DEFAULT_ALLOWED_LANGUAGES = ("en", "fr", "zh")  # Restrict auto-detection (None for all 99)
DEFAULT_VAD_PARAMETERS = {
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 500,
    "threshold": 0.25,
}
# Voice bandpass (85Hz-7.5kHz) + FFT noise reduction + speech normalizer for short child utterances
STT_AUDIO_FILTER = (
    "highpass=f=85,lowpass=f=7500,afftdn=nf=-25,speechnorm=e=6:r=0.0001:l=1"
)

# Domain-specific initial prompts per language to bias Whisper toward child questions and Simplified Chinese
STT_INITIAL_PROMPTS = {
    "fr": (
        "Bonjour, comment ça va ? Il est quelle heure ? On se couche à quelle heure ? "
        "C'est quoi les nuages, le soleil, la lune, les étoiles, la pluie, les avions, les animaux, les canards ? "
        "Pourquoi le ciel est bleu ? Comment les avions volent ?"
    ),
    "en": (
        "Hello, how are you? What time is it? What are clouds, the sun, the moon, stars, rain, airplanes, animals? "
        "Why is the sky blue? How do airplanes fly?"
    ),
    "zh": (
        "你好，现在几点了？这是什么？为什么天空是蓝色的？云、太阳、月亮、星星、飞机、动物是什么？飞机是怎么飞的？"
    ),
}

# Text-to-Speech settings
DEFAULT_SPEECH_SPEED = 1.0   # length_scale > 1.0 speaks slower, < 1.0 speaks faster
DEFAULT_NOISE_SCALE = 0.85   # Higher noise_scale (default 0.667) adds lively pitch & tone variation
DEFAULT_NOISE_W_SCALE = 0.9  # Phoneme duration variability for animated cadence
DEFAULT_PRELOAD_LANGUAGES = ("en", "fr", "zh")
DEFAULT_VOICE_EFFECT = "helium"

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

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish",
    "no": "Norwegian", "fi": "Finnish", "da": "Danish", "tr": "Turkish",
}

LANGUAGE_SWITCH_ANNOUNCEMENTS = {
    "en": "Now it's English!",
    "fr": "Maintenant, c'est le français !",
    "zh": "现在是中文啦！",
}

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

AVAILABLE_PIPER_VOICES = {
    "en": ["en_GB-semaine-medium", "en_US-lessac-medium"],
    "fr": ["fr_FR-tom-medium", "fr_FR-siwis-medium"],
    "zh": ["zh_CN-huayan-medium", "zh_CN-xiao_ya-medium", "zh_CN-chaowen-medium"],
}

# Default speaker IDs for multi-speaker models (SEMAINE speaker 3 = 'poppy')
PIPER_SPEAKER_MAP = {
    "en_GB-semaine-medium": 3,
}

# Default LLM configuration
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_LLM_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
    "echo": "echo",
}
DEFAULT_MAX_CONTEXT_TURNS = 5         # Recent user/assistant exchanges passed to LLM
DEFAULT_CONTEXT_IDLE_TIMEOUT = 600.0  # Seconds of inactivity (10 mins) before resetting context

DEFAULT_SYSTEM_PROMPT = (
    "You are an educational voice assistant for a young child (around 3 to 5 years old). "
    "Follow these rules strictly:\n"
    "1. Always reply in the exact same language as the user's question (for example English, French, or Chinese).\n"
    "2. The input comes from automatic Speech-to-Text of a young child's voice and may contain phonetic transcription errors "
    "(e.g. 'ennuiage' instead of 'nuage', 'calards' instead of 'canards', 'aimat' instead of 'volent', 'Keller' instead of 'quelle heure'). "
    "Infer the intended words from phonetic similarity and conversation context before answering.\n"
    "3. Only provide factual, educational, and age-appropriate explanations (science, nature, math, history, language, everyday how-things-work).\n"
    "4. Do NOT engage in roleplay, pretend play, fictional storytelling, or persona adoption. If asked to roleplay or for non-educational topics, gently redirect to an educational fact.\n"
    "5. Keep answers short, warm, clear, and easy for a small child to understand (1 to 3 simple sentences).\n"
    "6. Output plain spoken text only, suitable for Text-to-Speech. Never use markdown, bullet points, asterisks, symbols, or emojis.\n\n"
    "Examples:\n"
    "- If asked 'how are you', do not respond 'I am well, thank you', just redirect them to asking questions gently."
)


def get_language_name(lang_code: str) -> str:
    """Return human-readable language name for an ISO language code."""
    if not lang_code:
        return "Unknown"
    return LANGUAGE_NAMES.get(lang_code, lang_code.upper())
