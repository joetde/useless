import re
import subprocess


def _detect_usb_card(command: str, pattern: str, fallback: str) -> str:
    try:
        output = subprocess.check_output([command, "-l"], text=True, stderr=subprocess.DEVNULL)
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return f"plughw:CARD={match.group(2)},DEV=0"
    except (subprocess.SubprocessError, OSError):
        pass
    return fallback


def detect_usb_mic_device() -> str:
    """Detect ALSA card name for the USB microphone."""
    return _detect_usb_card(
        "arecord",
        r"card\s+(\d+):\s+([a-zA-Z0-9_-]+)\s+\[.*USB.*\]",
        "plughw:CARD=Device,DEV=0",
    )


def detect_usb_speaker_device() -> str:
    """Detect ALSA card name for the USB speaker."""
    return _detect_usb_card(
        "aplay",
        r"card\s+(\d+):\s+([a-zA-Z0-9_-]+)\s+\[.*\].*USB",
        "default",
    )
