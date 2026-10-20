import atexit
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .led import StatusLED

from .config import (
    DEFAULT_GPIO_BOUNCE_TIME,
    DEFAULT_GPIO_PIN,
    DEFAULT_POST_RELEASE_DURATION,
    RATE,
    RELEASE_TIMEOUT,
    TRIGGER_KEY,
    TRIGGER_KEY_NAME,
    WAVE_OUTPUT_FILENAME,
)
from .led import release_gpio_device


@contextmanager
def _cbreak_terminal(require_tty: bool = False) -> Iterator[bool]:
    """Put stdin into cbreak mode if attached to a TTY, restoring settings on exit."""
    if not sys.stdin.isatty():
        if require_tty:
            raise RuntimeError("Keyboard recording requires an interactive TTY on stdin.")
        yield False
        return

    fd: Optional[int] = None
    old_settings = None
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        yield True
    except (termios.error, ValueError, OSError):
        if require_tty:
            raise
        yield False
    finally:
        if fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_available_keys(timeout: float = 0.0) -> str:
    """Read all currently buffered characters from stdin up to `timeout`."""
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return ""
    chars = sys.stdin.read(1)
    while select.select([sys.stdin], [], [], 0)[0]:
        chars += sys.stdin.read(1)
    return chars


def _is_quit_key(chars: str) -> bool:
    return any(k in chars for k in ("q", "Q", "\x03"))


class RecorderEngine(ABC):
    """Abstract base class for audio recording engines."""

    def __init__(
        self,
        device: str,
        rate: int = RATE,
        status_led: Optional["StatusLED"] = None,
    ):
        self.device = device
        self.rate = rate
        self.status_led = status_led

    @abstractmethod
    def record(self, output_file: str = WAVE_OUTPUT_FILENAME) -> Optional[str]:
        """
        Wait for trigger / record audio to `output_file`.
        Returns:
            - Path to recorded audio file (`str`) if a valid recording was captured,
            - `""` (empty string) if recording attempt failed/was too short,
            - `None` if the user requested to quit.
        """

    def close(self) -> None:
        """Release any hardware resources held by the recorder."""

    def _start_arecord(self, output_file: str) -> subprocess.Popen:
        if os.path.exists(output_file):
            os.remove(output_file)
        cmd = [
            "arecord",
            "-D", self.device,
            "-f", "S16_LE",
            "-r", str(self.rate),
            "-c", "1",
            "--buffer-time=100000",
            "--period-time=25000",
            output_file,
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _stop_arecord(process: subprocess.Popen) -> None:
        """Stop `arecord` cleanly via SIGINT so ALSA flushes PCM frames and updates the WAV header."""
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    @staticmethod
    def _validate_recording(output_file: str) -> str:
        if not os.path.exists(output_file) or os.path.getsize(output_file) <= 44:
            print("Warning: Recording failed or was too short.")
            return ""
        print(f"* Audio saved to {output_file} ({os.path.getsize(output_file):,} bytes)")
        return output_file


class GPIOSwitchAlsaRecorder(RecorderEngine):
    """
    Waits for a physical switch connected between a GPIO pin (default BCM 21, physical pin 40)
    and Ground to be closed, records audio via ALSA `arecord` while closed, and stops
    when the switch is opened (after a brief `post_release_duration` tail so trailing words aren't clipped).
    """

    def __init__(
        self,
        device: str,
        gpio_pin: int = DEFAULT_GPIO_PIN,
        rate: int = RATE,
        bounce_time: float = DEFAULT_GPIO_BOUNCE_TIME,
        post_release_duration: float = DEFAULT_POST_RELEASE_DURATION,
        status_led: Optional["StatusLED"] = None,
    ):
        super().__init__(device=device, rate=rate, status_led=status_led)
        from gpiozero import Button

        self.gpio_pin = gpio_pin
        self.bounce_time = bounce_time
        self.post_release_duration = max(0.0, float(post_release_duration))
        self.button = Button(gpio_pin, pull_up=True, bounce_time=bounce_time)
        atexit.register(self.close)

    def close(self) -> None:
        if self.button is not None:
            release_gpio_device(self.button, self.gpio_pin, pull_down=True)
            self.button = None

    def record(self, output_file: str = WAVE_OUTPUT_FILENAME) -> Optional[str]:
        with _cbreak_terminal(require_tty=False) as has_tty:
            suffix = ", or press 'q' to quit..." if has_tty else "..."
            print(f"\n[Ready] Close switch on GPIO {self.gpio_pin} to record (open to stop){suffix}")

            if self.status_led is not None:
                self.status_led.pulse_ready()

            while not self.button.is_pressed:
                if has_tty and _is_quit_key(_read_available_keys(0.0)):
                    if self.status_led is not None:
                        self.status_led.off()
                    return None
                time.sleep(0.02)

            if self.status_led is not None:
                self.status_led.off()

            process = self._start_arecord(output_file)
            start_time = time.monotonic()

            try:
                while self.button.is_pressed and process.poll() is None:
                    elapsed = time.monotonic() - start_time
                    print(
                        f"\r  * [RECORDING] {elapsed:4.1f}s (Keep switch on GPIO {self.gpio_pin} closed...) ",
                        end="",
                        flush=True,
                    )
                    if has_tty and _is_quit_key(_read_available_keys(0.0)):
                        self._stop_arecord(process)
                        return None
                    time.sleep(0.02)

                # Capture a brief post-release tail so the final syllable isn't cut off
                if self.post_release_duration > 0 and process.poll() is None:
                    time.sleep(self.post_release_duration)
            finally:
                if self.status_led is not None:
                    self.status_led.solid_thinking()
                self._stop_arecord(process)

            total_time = time.monotonic() - start_time
            print(f"\r  * [RECORDING] Stopped ({total_time:.1f}s). Processing audio...          ")

        return self._validate_recording(output_file)


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
        status_led: Optional["StatusLED"] = None,
    ):
        super().__init__(device=device, rate=rate, status_led=status_led)
        self.trigger_key = trigger_key
        self.trigger_key_name = trigger_key_name
        self.release_timeout = release_timeout

    def record(self, output_file: str = WAVE_OUTPUT_FILENAME) -> Optional[str]:
        with _cbreak_terminal(require_tty=True):
            print(
                f"\n[Ready] Hold [{self.trigger_key_name}] to record (release to stop), or press 'q' to quit..."
            )

            if self.status_led is not None:
                self.status_led.pulse_ready()

            while True:
                chars = _read_available_keys(0.1)
                if _is_quit_key(chars):
                    if self.status_led is not None:
                        self.status_led.off()
                    return None
                if self.trigger_key in chars:
                    break

            if self.status_led is not None:
                self.status_led.off()

            process = self._start_arecord(output_file)
            start_time = time.monotonic()
            last_event_time = start_time

            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - start_time
                    print(
                        f"\r  * [RECORDING] {elapsed:4.1f}s (Keep holding [{self.trigger_key_name}]...) ",
                        end="",
                        flush=True,
                    )
                    chars = _read_available_keys(0.05)
                    if chars:
                        if _is_quit_key(chars):
                            self._stop_arecord(process)
                            return None
                        if self.trigger_key in chars:
                            last_event_time = time.monotonic()
                    elif (time.monotonic() - last_event_time) > self.release_timeout:
                        break
            finally:
                if self.status_led is not None:
                    self.status_led.solid_thinking()
                self._stop_arecord(process)

            total_time = time.monotonic() - start_time
            print(f"\r  * [RECORDING] Stopped ({total_time:.1f}s). Processing audio...          ")

        return self._validate_recording(output_file)
