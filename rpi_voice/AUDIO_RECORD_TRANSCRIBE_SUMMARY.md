# Audio Recording & AI Speech-to-Text Setup Summary

## Overview
This setup records audio for 10 seconds from an attached USB microphone on a Raspberry Pi 5 and performs local, offline Speech-to-Text (STT) transcription with automatic language identification.

---

## 1. System & Hardware Context
- **Host Device**: Raspberry Pi 5 (`georgipi5`), `aarch64`
- **Operating System**: Debian GNU/Linux 13 (trixie)
- **Python Version**: Python 3.13.5
- **Audio Capture Device**: `USB PnP Sound Device`
  - ALSA hardware identifier: `plughw:CARD=Device,DEV=0`
  - PipeWire Source: `PCM2902 Audio Codec Analog Mono`

---

## 2. Dependencies & Libraries Installed

### System Packages (Debian `apt`)
- `alsa-utils` & `ffmpeg`: Command-line tools for low-level audio capture and conversion.
- `libportaudio2` & `python3-pyaudio`: PortAudio library and Python bindings for audio I/O.

### Python Public Libraries
- **`faster-whisper` (1.2.1)**:
  - Re-implementation of OpenAI's Whisper model using **CTranslate2** (C++ inference engine).
  - Uses **INT8 quantization**, which consumes only ~100–150 MB of RAM (compared to PyTorch Whisper which requires >1 GB RAM and CUDA/heavy wheel dependencies).
  - Highly optimized for ARM64 NEON instructions on Raspberry Pi 5.
  - Native multi-language detection and built-in Silero **Voice Activity Detection (VAD)**.

---

## 3. Script Implementation Details

**Script Path**: `/home/guillaume/record_and_transcribe.py`

### Key Features
1. **Dynamic USB Microphone Detection**:
   - Inspects `arecord -l` to locate the USB card identifier dynamically (`plughw:CARD=Device,DEV=0`).
   - Avoids failures caused by card index renumbering on reboots or USB re-plugs.
   - Includes automatic fallback to ALSA `default`.
2. **Audio Capture Specifications**:
   - Sample rate: **16,000 Hz (16 kHz)**
   - Format: **16-bit Signed Little Endian (`S16_LE`)**
   - Channels: **Mono (1 channel)**
   - Output location: `/tmp/recorded_audio.wav`
   - Native format matches Whisper's acoustic feature extractor without requiring audio resampling.
3. **Interactive Recording Experience**:
   - Non-blocking execution with a live terminal countdown timer (`10s remaining...`).
   - Clean handling for user interrupt (`Ctrl+C`).
4. **AI Transcription & Language Awareness**:
   - Model: Whisper `tiny` (multi-lingual version).
   - Parameters: `beam_size=5`, `vad_filter=True`.
   - **VAD Filter**: Eliminates silence/background noise to prevent Whisper hallucinations (e.g. repeating "I'm sorry").
   - **Language Output**: Reports language name (e.g., `English`, `French`, `Spanish`), ISO code (`en`, `fr`), and confidence percentage.

---

## 4. Usage Instructions

Run the script directly using Python 3:

```bash
python3 /home/guillaume/record_and_transcribe.py
```

### Expected Output Example
```text
Target audio device: plughw:CARD=Device,DEV=0
* Recording started (10s). Speak into your USB microphone...
  [Recording] Done! Processing audio...            
* Audio saved to /tmp/recorded_audio.wav (320,044 bytes)

Loading Whisper model ('tiny') with INT8 precision...
Transcribing and analyzing language...

================================================
 RESULTS
================================================
Detected Language : English (en) - 98.4% confidence
------------------------------------------------
Transcription:
  [ 0.00s ->  4.20s] Hello, this is a test recording on the Raspberry Pi.
================================================
```

---

## 5. Maintenance & Troubleshooting

- **Checking Microphone Hardware**:
  ```bash
  arecord -l
  ```
- **Checking Recording Levels / Volume**:
  ```bash
  alsamixer
  # Press F6 to select "USB PnP Sound Device" and adjust capture volume
  ```
- **Changing Model Size**:
  In `/home/guillaume/record_and_transcribe.py`, you can change `WhisperModel("tiny", ...)` to `"base"` or `"small"` for higher transcription accuracy if needed.
