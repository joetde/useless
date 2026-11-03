import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .config import (
    DEFAULT_LLM_MODELS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_SYSTEM_PROMPT,
    get_language_name,
)


@dataclass
class ChatTurn:
    """Represents one completed user/assistant exchange for conversation context."""
    user_text: str
    assistant_text: str
    language: Optional[str] = None
    timestamp: Optional[str] = None


class LLMError(Exception):
    """Base exception raised when an LLM engine fails to produce a reply."""

    def __init__(self, message: str, kind: str = "service"):
        super().__init__(message)
        self.kind = kind


class LLMNetworkError(LLMError):
    """Raised when the LLM API cannot be reached due to network, DNS, or timeout issues."""

    def __init__(self, message: str):
        super().__init__(message, kind="network")


class LLMServiceError(LLMError):
    """Raised when the LLM API returns an HTTP error, invalid payload, or blocked/empty response."""

    def __init__(self, message: str):
        super().__init__(message, kind="service")


def clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting and extra whitespace so TTS reads naturally."""
    cleaned = re.sub(r"[*_`#~]+", "", text)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-•]\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class LLMEngine(ABC):
    """Abstract base class for Large Language Model engines."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.3,
        max_output_tokens: int = 250,
        timeout: float = 15.0,
    ):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.last_error: Optional[LLMError] = None
        self._session = requests.Session()

    def _build_system_instruction(self, language: Optional[str] = None) -> str:
        """Build the system instruction, including a language hint when available."""
        if not language:
            return self.system_prompt
        lang_name = get_language_name(language)
        return (
            f"{self.system_prompt}\n"
            f"Note: The spoken input was detected as {lang_name} ({language}). "
            f"Make sure your response is in {lang_name}."
        )

    @abstractmethod
    def generate(
        self,
        prompt: str,
        language: Optional[str] = None,
        history: Optional[Sequence[ChatTurn]] = None,
    ) -> Optional[str]:
        """Send a user prompt (with optional recent chat history) to the LLM and return plain text."""


class EchoLLMEngine(LLMEngine):
    """Passthrough engine that repeats the input text without calling an LLM."""

    def generate(
        self,
        prompt: str,
        language: Optional[str] = None,
        history: Optional[Sequence[ChatTurn]] = None,
    ) -> Optional[str]:
        return prompt.strip() or None


class GeminiLLMEngine(LLMEngine):
    """Lightweight Google Gemini LLM engine using the REST API."""

    API_URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODELS["gemini"],
        api_key: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.3,
        max_output_tokens: int = 250,
        timeout: float = 15.0,
    ):
        super().__init__(
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        self.model = model.removeprefix("models/")
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY in your environment or pass --api-key."
            )

    def generate(
        self,
        prompt: str,
        language: Optional[str] = None,
        history: Optional[Sequence[ChatTurn]] = None,
    ) -> Optional[str]:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            return None

        self.last_error = None
        ctx_count = len(history) if history else 0
        ctx_info = f" with {ctx_count} prior turn(s) of context" if ctx_count else ""
        print(f"Asking Gemini ('{self.model}'{ctx_info})...")

        contents = []
        for turn in history or ():
            if turn.user_text:
                contents.append({"role": "user", "parts": [{"text": turn.user_text}]})
            if turn.assistant_text:
                contents.append({"role": "model", "parts": [{"text": turn.assistant_text}]})
        contents.append({"role": "user", "parts": [{"text": clean_prompt}]})

        payload = {
            "system_instruction": {
                "parts": [{"text": self._build_system_instruction(language)}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        try:
            response = self._session.post(
                self.API_URL_TEMPLATE.format(model=self.model),
                headers={"x-goog-api-key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                msg = "Gemini returned no candidates."
                print(f"Warning: {msg}")
                self.last_error = LLMServiceError(msg)
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            raw_text = " ".join(part.get("text", "") for part in parts if "text" in part)
            cleaned = clean_text_for_speech(raw_text) or None
            if cleaned is None:
                msg = "Gemini returned an empty response."
                print(f"Warning: {msg}")
                self.last_error = LLMServiceError(msg)
            return cleaned
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Network error communicating with Gemini API: {e}")
            self.last_error = LLMNetworkError(f"Gemini network error: {e}")
            return None
        except (requests.RequestException, ValueError) as e:
            print(f"Error communicating with Gemini API: {e}")
            self.last_error = LLMServiceError(f"Gemini API error: {e}")
            return None


class OpenAICompatibleLLMEngine(LLMEngine):
    """
    Generic OpenAI-compatible Chat Completions LLM engine.
    Works with OpenAI, Ollama (http://localhost:11434/v1), OpenRouter, Groq, etc.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODELS["openai"],
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.3,
        max_output_tokens: int = 250,
        timeout: float = 15.0,
    ):
        super().__init__(
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""

    def generate(
        self,
        prompt: str,
        language: Optional[str] = None,
        history: Optional[Sequence[ChatTurn]] = None,
    ) -> Optional[str]:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            return None

        self.last_error = None
        ctx_count = len(history) if history else 0
        ctx_info = f", {ctx_count} prior turn(s) of context" if ctx_count else ""
        print(f"Asking LLM ('{self.model}' at {self.base_url}{ctx_info})...")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = [
            {"role": "system", "content": self._build_system_instruction(language)},
        ]
        for turn in history or ():
            if turn.user_text:
                messages.append({"role": "user", "content": turn.user_text})
            if turn.assistant_text:
                messages.append({"role": "assistant", "content": turn.assistant_text})
        messages.append({"role": "user", "content": clean_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                msg = "LLM returned no choices."
                print(f"Warning: {msg}")
                self.last_error = LLMServiceError(msg)
                return None

            raw_text = choices[0].get("message", {}).get("content", "")
            cleaned = clean_text_for_speech(raw_text) or None
            if cleaned is None:
                msg = "LLM returned an empty response."
                print(f"Warning: {msg}")
                self.last_error = LLMServiceError(msg)
            return cleaned
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Network error communicating with LLM API: {e}")
            self.last_error = LLMNetworkError(f"LLM network error: {e}")
            return None
        except (requests.RequestException, ValueError) as e:
            print(f"Error communicating with LLM API: {e}")
            self.last_error = LLMServiceError(f"LLM API error: {e}")
            return None


def create_llm_engine(
    provider: str = DEFAULT_LLM_PROVIDER,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.3,
    max_output_tokens: int = 250,
) -> LLMEngine:
    """Factory to instantiate an LLMEngine for the requested provider."""
    provider_key = provider.lower().strip()
    resolved_model = model or DEFAULT_LLM_MODELS.get(provider_key, DEFAULT_LLM_MODELS["gemini"])
    common_kwargs = {
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    if provider_key == "echo":
        return EchoLLMEngine(**common_kwargs)
    if provider_key == "gemini":
        return GeminiLLMEngine(model=resolved_model, api_key=api_key, **common_kwargs)
    if provider_key == "ollama":
        return OpenAICompatibleLLMEngine(
            model=resolved_model,
            api_key=api_key or "ollama",
            base_url=base_url or "http://localhost:11434/v1",
            **common_kwargs,
        )
    if provider_key in ("openai", "openai-compatible"):
        return OpenAICompatibleLLMEngine(
            model=resolved_model,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            **common_kwargs,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Choose from: gemini, openai, ollama, echo."
    )
