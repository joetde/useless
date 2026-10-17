import re
import subprocess


def detect_usb_mic_device() -> str:
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


def detect_usb_speaker_device() -> str:
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
