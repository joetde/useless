import atexit
import threading
import time
from typing import Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tts import TTSEngine

from .config import (
    DEFAULT_GPIO_BOUNCE_TIME,
    DEFAULT_LANG_SWITCH_PIN_1,
    DEFAULT_LANG_SWITCH_PIN_2,
    DEFAULT_LANG_SWITCH_SETTLE_TIME,
    LANGUAGE_SWITCH_ANNOUNCEMENTS,
    TTS_LANG_SWITCH_FILENAME,
    get_language_name,
)
from .led import release_gpio_device


class LanguageSwitch:
    """
    Monitors a 3-terminal switch connected to two GPIO pins (with internal/external pull-ups)
    and a common Ground pin:
      - State I  (GPIO `pin_1` LOW, `pin_2` HIGH) -> Force French ('fr')
      - State II (GPIO `pin_2` LOW, `pin_1` HIGH) -> Force Chinese ('zh')
      - Both GPIOs HIGH (`pin_1` HIGH, `pin_2` HIGH) -> Force Korean ('ko')

    Runs a background monitor thread so the user can flip the switch at any time
    and hear a spoken confirmation ("Now switching to ...") in the newly selected language.
    """

    def __init__(
        self,
        pin_1: int = DEFAULT_LANG_SWITCH_PIN_1,
        pin_2: int = DEFAULT_LANG_SWITCH_PIN_2,
        lang_state_1: str = "fr",
        lang_state_2: str = "zh",
        lang_both_high: str = "ko",
        bounce_time: float = DEFAULT_GPIO_BOUNCE_TIME,
        settle_time: float = DEFAULT_LANG_SWITCH_SETTLE_TIME,
        tts_engine: Optional["TTSEngine"] = None,
        announcements: Optional[Dict[str, str]] = None,
        on_change: Optional[Callable[[str], None]] = None,
        announce_changes: bool = True,
    ):
        from gpiozero import Button

        self.pin_1 = pin_1
        self.pin_2 = pin_2
        self.lang_state_1 = lang_state_1
        self.lang_state_2 = lang_state_2
        self.lang_both_high = lang_both_high
        self.bounce_time = bounce_time
        self.settle_time = settle_time
        self.tts_engine = tts_engine
        self.announcements = dict(announcements or LANGUAGE_SWITCH_ANNOUNCEMENTS)
        self.on_change = on_change
        self.announce_changes = announce_changes

        self.button_1 = Button(pin_1, pull_up=True, bounce_time=bounce_time)
        self.button_2 = Button(pin_2, pull_up=True, bounce_time=bounce_time)

        self._current_language: str = self.read_language()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        atexit.register(self.close)

    @property
    def current_language(self) -> str:
        """Return the currently active forced language code ('fr', 'zh', or 'ko')."""
        return self._current_language

    def read_language(self) -> str:
        """Read the instantaneous switch GPIO states and map to a language code."""
        if self.button_1 is None or self.button_2 is None:
            return self._current_language

        # is_pressed is True when the GPIO pin is pulled LOW to GND
        state_1_low = bool(self.button_1.is_pressed)
        state_2_low = bool(self.button_2.is_pressed)

        if state_1_low and not state_2_low:
            return self.lang_state_1
        if state_2_low and not state_1_low:
            return self.lang_state_2
        return self.lang_both_high

    def describe_state(self, lang_code: Optional[str] = None) -> str:
        """Return a human-readable description of the switch state for `lang_code`."""
        code = lang_code or self._current_language
        lang_name = get_language_name(code)
        if code == self.lang_state_1:
            return f"State I (GPIO {self.pin_1} LOW) -> {lang_name} ({code})"
        if code == self.lang_state_2:
            return f"State II (GPIO {self.pin_2} LOW) -> {lang_name} ({code})"
        return f"Both HIGH (GPIO {self.pin_1} & {self.pin_2} HIGH) -> {lang_name} ({code})"

    def get_announcement_text(self, lang_code: str) -> str:
        """Return the spoken phrase in `lang_code` announcing the language switch."""
        return self.announcements.get(lang_code, f"Now switching to {get_language_name(lang_code)}.")

    def start_monitoring(self) -> None:
        """Start the background thread that watches for switch changes at any time."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._current_language = self.read_language()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="LanguageSwitchMonitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        pending_lang: Optional[str] = None
        pending_since: float = 0.0

        while not self._stop_event.is_set():
            raw_lang = self.read_language()
            now = time.monotonic()

            if raw_lang != self._current_language:
                if raw_lang != pending_lang:
                    pending_lang = raw_lang
                    pending_since = now
                elif (now - pending_since) >= self.settle_time:
                    self._current_language = raw_lang
                    pending_lang = None
                    self._handle_language_change(raw_lang)
            else:
                pending_lang = None

            self._stop_event.wait(0.03)

    def _handle_language_change(self, new_lang: str) -> None:
        phrase = self.get_announcement_text(new_lang)
        print(f'\n[Language Switch] {self.describe_state(new_lang)} — "{phrase}"')

        if self.on_change is not None:
            try:
                self.on_change(new_lang)
            except Exception as e:
                print(f"Warning: Language switch callback error: {e}")

        if self.announce_changes and self.tts_engine is not None:
            try:
                self.tts_engine.interrupt_playback()
                self.tts_engine.speak(
                    phrase,
                    lang_code=new_lang,
                    output_wav=TTS_LANG_SWITCH_FILENAME,
                    restore_led_ready=True,
                )
            except Exception as e:
                print(f"Warning: Could not speak language switch announcement ({e}).")

    def close(self) -> None:
        """Stop monitoring and release GPIO pins."""
        self._stop_event.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            if threading.current_thread() is not self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
        self._monitor_thread = None

        release_gpio_device(self.button_1, self.pin_1)
        self.button_1 = None
        release_gpio_device(self.button_2, self.pin_2)
        self.button_2 = None
