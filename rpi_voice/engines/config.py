from pathlib import Path

WAVE_OUTPUT_FILENAME = "/tmp/recorded_audio.wav"
TTS_OUTPUT_FILENAME = "/tmp/tts_output.wav"
VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
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


def get_language_name(lang_code: str) -> str:
    """Return human-readable language name for an ISO language code."""
    if not lang_code:
        return "Unknown"
    return LANGUAGE_NAMES.get(lang_code, lang_code.upper())
