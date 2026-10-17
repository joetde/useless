import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from faster_whisper import WhisperModel

from .config import DEFAULT_LANGUAGE, get_language_name


@dataclass
class TranscriptionSegment:
    """Represents a single time-aligned speech segment."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Structured output returned by any STTEngine implementation."""
    text: str
    language: str
    language_probability: float = 1.0
    forced_language: bool = False
    top_languages: List[Tuple[str, float]] = field(default_factory=list)
    segments: List[TranscriptionSegment] = field(default_factory=list)

    @property
    def has_speech(self) -> bool:
        return bool(self.text.strip())


class STTEngine(ABC):
    """Abstract base class for Speech-to-Text engines."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = DEFAULT_LANGUAGE,
    ) -> Optional[TranscriptionResult]:
        """Transcribe an audio file and return a structured TranscriptionResult."""
        pass


class FasterWhisperSTT(STTEngine):
    """Speech-to-Text engine backed by CTranslate2 faster-whisper."""

    def __init__(
        self,
        model_size: str = "tiny",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        print(f"Loading Whisper model ('{model_size}') with {compute_type.upper()} precision...")
        self.model = WhisperModel(model_size, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = DEFAULT_LANGUAGE,
    ) -> Optional[TranscriptionResult]:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) <= 44:
            return None

        if language:
            lang_name = get_language_name(language)
            print(f"Transcribing in forced language: {lang_name} ({language})...")
        else:
            print("Transcribing and analyzing language...")

        # vad_filter=True filters out background silence to prevent hallucinations
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        raw_segments = list(segments_gen)

        detected_lang = language or info.language or "en"
        lang_prob = 1.0 if language else getattr(info, "language_probability", 0.0)
        top_langs: List[Tuple[str, float]] = []
        if not language and getattr(info, "all_language_probs", None):
            top_langs = list(info.all_language_probs[:3])

        segments: List[TranscriptionSegment] = []
        full_text_parts: List[str] = []
        for seg in raw_segments:
            text_part = seg.text.strip()
            segments.append(
                TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text_part,
                )
            )
            if text_part:
                full_text_parts.append(text_part)

        return TranscriptionResult(
            text=" ".join(full_text_parts),
            language=detected_lang,
            language_probability=lang_prob,
            forced_language=bool(language),
            top_languages=top_langs,
            segments=segments,
        )
