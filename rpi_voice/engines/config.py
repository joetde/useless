from pathlib import Path

WAVE_OUTPUT_FILENAME = "/tmp/recorded_audio.wav"
TTS_OUTPUT_FILENAME = "/tmp/tts_output.wav"
VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
RATE = 16000
TRIGGER_KEY = " "        # Spacebar
TRIGGER_KEY_NAME = "SPACE"
RELEASE_TIMEOUT = 0.75   # Seconds without key repeat before treating as released
DEFAULT_LANGUAGE = None  # Set to e.g. "en" or "fr" to always force a language, or None for auto-detect
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


def get_language_name(lang_code: str) -> str:
    """Return human-readable language name for an ISO language code."""
    if not lang_code:
        return "Unknown"
    return LANGUAGE_NAMES.get(lang_code, lang_code.upper())
