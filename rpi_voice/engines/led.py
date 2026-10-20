import atexit
from typing import Any, Optional

from .config import (
    DEFAULT_LED_GPIO_PIN,
    DEFAULT_LED_PULSE_FADE_IN,
    DEFAULT_LED_PULSE_FADE_OUT,
)


def release_gpio_device(device: Any, default_pin: Optional[int] = None, pull_down: bool = False) -> None:
    """Safely close a gpiozero device and release its underlying lgpio pin."""
    if device is None:
        return
    pin_obj = getattr(device, "pin", None)
    factory = getattr(pin_obj, "factory", None)
    chip_handle = getattr(factory, "_handle", None)
    pin_info = getattr(pin_obj, "info", None)
    pin_num = getattr(pin_obj, "_number", default_pin)

    try:
        device.close()
    except Exception:
        pass

    if factory is not None and pin_info is not None:
        getattr(factory, "pins", {}).pop(pin_info, None)

    if chip_handle is not None and pin_num is not None:
        try:
            import lgpio

            lgpio.gpio_free(chip_handle, pin_num)
            if pull_down:
                lgpio.gpio_claim_input(chip_handle, pin_num, lgpio.SET_PULL_DOWN)
                lgpio.gpio_free(chip_handle, pin_num)
        except Exception:
            pass


class StatusLED:
    """
    Controls a status LED connected between a GPIO pin (default BCM 13, physical pin 33)
    and Ground:
      - `pulse_ready()`: Gently pulses when the device is ready to take input.
      - `off()`: Turns off (while recording or speaking, or when exiting).
      - `solid_thinking()`: Turns solid ON while processing (STT / LLM / TTS synthesis).
    """

    def __init__(
        self,
        gpio_pin: Optional[int] = DEFAULT_LED_GPIO_PIN,
        fade_in_time: float = DEFAULT_LED_PULSE_FADE_IN,
        fade_out_time: float = DEFAULT_LED_PULSE_FADE_OUT,
    ):
        self.gpio_pin = gpio_pin
        self.fade_in_time = fade_in_time
        self.fade_out_time = fade_out_time
        self._led = None
        self._state = "off"

        if gpio_pin is not None:
            from gpiozero import PWMLED

            self._led = PWMLED(gpio_pin, active_high=True, initial_value=0.0)
            atexit.register(self.close)

    @property
    def state(self) -> str:
        """Current logical LED state ('off', 'ready', or 'thinking')."""
        return self._state

    def pulse_ready(self) -> None:
        """Pulse the LED smoothly to indicate the device is ready for input."""
        self._state = "ready"
        if self._led is not None:
            self._led.pulse(
                fade_in_time=self.fade_in_time,
                fade_out_time=self.fade_out_time,
                background=True,
            )

    def off(self) -> None:
        """Turn the LED completely off."""
        self._state = "off"
        if self._led is not None:
            self._led.off()

    def solid_thinking(self) -> None:
        """Turn the LED solid on while thinking/processing before speaker playback."""
        self._state = "thinking"
        if self._led is not None:
            self._led.on()

    def close(self) -> None:
        """Turn off the LED, set the pin to input with pull-down (0V), and release the GPIO line."""
        if self._led is not None:
            try:
                self._led.off()
                self._led.value = 0.0
            except Exception:
                pass
            release_gpio_device(self._led, self.gpio_pin, pull_down=True)
            self._led = None
