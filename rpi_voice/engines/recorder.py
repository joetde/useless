import os
import sys
import time
import signal
import select
import termios
import tty
import subprocess
from abc import ABC, abstractmethod
from typing import Optional

from .config import (
    WAVE_OUTPUT_FILENAME,
    RATE,
    TRIGGER_KEY,
    TRIGGER_KEY_NAME,
    RELEASE_TIMEOUT,
    DEFAULT_GPIO_PIN,
    DEFAULT_GPIO_BOUNCE_TIME,
)


class RecorderEngine(ABC):
    """Abstract base class for audio recording engines."""

    @abstractmethod
    def record(self, output_file: str = WAVE_OUTPUT_FILENAME) -> Optional[str]:
        """
        Wait for trigger / record audio to `output_file`.
        Returns:
            - Path to recorded audio file (`str`) if a valid recording was captured,
            - `""` (empty string) if recording attempt failed/was too short but session should continue,
            - `None` if the user requested to quit.
        """
        pass

    def close(self) -> None:
        """Release any hardware resources held by the recorder."""
        pass


class GPIOSwitchAlsaRecorder(RecorderEngine):
    """
    Waits for a physical switch connected between a GPIO pin (default BCM 21, physical pin 40)
    and Ground to be closed, records audio via ALSA `arecord` while closed, and stops
    when the switch is opened.
    """

    def __init__(
        self,
        device: str,
        gpio_pin: int = DEFAULT_GPIO_PIN,
        rate: int = RATE,
        bounce_time: float = DEFAULT_GPIO_BOUNCE_TIME,
    ):
        from gpiozero import Button

        self.device = device
        self.gpio_pin = gpio_pin
        self.rate = rate
        self.bounce_time = bounce_time
        self.button = Button(gpio_pin, pull_up=True, bounce_time=bounce_time)

    def close(self) -> None:
        if getattr(self, "button", None) is not None:
            self.button.close()
            self.button = None

    def _check_quit_key(self, use_stdin: bool) -> bool:
        """Non-blocking check if 'q' or Ctrl+C was pressed on stdin."""
        if not use_stdin:
            return False
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            chars = ""
            while select.select([sys.stdin], [], [], 0)[0]:
                chars += sys.stdin.read(1)
            if "\x03" in chars or "q" in chars or "Q" in chars:
                return True
        return False

    def record(self, output_file: str = WAVE_OUTPUT_FILENAME) -> Optional[str]:
        use_stdin = sys.stdin.isatty()
        fd = None
        old_settings = None

        if use_stdin:
            try:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
            except (termios.error, ValueError, OSError):
                use_stdin = False

        print(
            f"\n[Ready] Close switch on GPIO {self.gpio_pin} to record (open to stop)"
            + (", or press 'q' to quit..." if use_stdin else "...")
        )

        try:
            if use_stdin and fd is not None:
                tty.setcbreak(fd)

            # Wait for switch to be closed (is_pressed is True when connected to GND)
            while not self.button.is_pressed:
                if self._check_quit_key(use_stdin):
                    return None
                time.sleep(0.02)

            # Remove previous file so we can verify fresh recording
            if os.path.exists(output_file):
                os.remove(output_file)

            cmd = [
                "arecord",
                "-D", self.device,
                "-f", "S16_LE",
                "-r", str(self.rate),
                "-c", "1",
                output_file,
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            start_time = time.time()

            while self.button.is_pressed:
                elapsed = time.time() - start_time
                print(
                    f"\r  * [RECORDING] {elapsed:4.1f}s (Keep switch on GPIO {self.gpio_pin} closed...) ",
                    end="",
                    flush=True,
                )

                # Check if arecord exited prematurely (e.g. device error)
                if process.poll() is not None:
                    break

                if self._check_quit_key(use_stdin):
                    process.send_signal(signal.SIGINT)
                    process.wait()
                    return None

                time.sleep(0.03)

            # Stop recording cleanly via SIGINT so ALSA flushes remaining PCM frames and updates WAV header
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()

            total_time = time.time() - start_time
            print(f"\r  * [RECORDING] Stopped ({total_time:.1f}s). Processing audio...          ")

        finally:
            if use_stdin and fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if not os.path.exists(output_file) or os.path.getsize(output_file) <= 44:
            print("Warning: Recording failed or was too short.")
            return ""

        print(f"* Audio saved to {output_file} ({os.path.getsize(output_file):,} bytes)")
        return output_file


class HoldKeyAlsaRecorder(RecorderEngine):
    """
    Waits for a trigger key to be held down in the terminal, records audio via ALSA `arecord`
    while held, and stops when the key is released.
    """

    def __init__(
        self,
        device: str,
        rate: int = RATE,
        trigger_key: str = TRIGGER_KEY,
        trigger_key_name: str = TRIGGER_KEY_NAME,
        release_timeout: float = RELEASE_TIMEOUT,
    ):
        self.device = device
        self.rate = rate
        self.trigger_key = trigger_key
        self.trigger_key_name = trigger_key_name
        self.release_timeout = release_timeout

    def record(self, output_file: str = WAVE_OUTPUT_FILENAME) -> Optional[str]:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        print(f"\n[Ready] Hold [{self.trigger_key_name}] to record (release to stop), or press 'q' to quit...")

        try:
            tty.setcbreak(fd)

            # Wait for trigger key press
            while True:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch in ("q", "Q", "\x03"):  # 'q' or Ctrl+C
                        return None
                    if ch == self.trigger_key:
                        break

            # Remove previous file so we can verify fresh recording
            if os.path.exists(output_file):
                os.remove(output_file)

            cmd = [
                "arecord",
                "-D", self.device,
                "-f", "S16_LE",
                "-r", str(self.rate),
                "-c", "1",
                output_file,
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            start_time = time.time()
            last_event_time = start_time

            while True:
                elapsed = time.time() - start_time
                print(
                    f"\r  * [RECORDING] {elapsed:4.1f}s (Keep holding [{self.trigger_key_name}]...) ",
                    end="",
                    flush=True,
                )

                # Check if arecord exited prematurely (e.g. device error)
                if process.poll() is not None:
                    break

                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    # Drain all buffered key repeats
                    chars = ""
                    while select.select([sys.stdin], [], [], 0)[0]:
                        chars += sys.stdin.read(1)
                    if "\x03" in chars or "q" in chars or "Q" in chars:
                        process.send_signal(signal.SIGINT)
                        process.wait()
                        return None
                    if self.trigger_key in chars:
                        last_event_time = time.time()
                else:
                    # No key repeat received within timeout -> key was released
                    if time.time() - last_event_time > self.release_timeout:
                        break

            # Stop recording cleanly via SIGINT so ALSA flushes remaining PCM frames and updates WAV header
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()

            total_time = time.time() - start_time
            print(f"\r  * [RECORDING] Stopped ({total_time:.1f}s). Processing audio...          ")

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if not os.path.exists(output_file) or os.path.getsize(output_file) <= 44:
            print("Warning: Recording failed or was too short.")
            return ""

        print(f"* Audio saved to {output_file} ({os.path.getsize(output_file):,} bytes)")
        return output_file
