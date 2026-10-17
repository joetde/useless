import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import (
    TranscriptionOptions,
    get_suppressed_tokens,
    restore_speech_timestamps,
)
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

from .config import (
    DEFAULT_ALLOWED_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_STT_BEAM_SIZE,
    DEFAULT_STT_MODEL,
    DEFAULT_VAD_PARAMETERS,
    RATE,
    STT_AUDIO_FILTER,
    WAVE_PREPROCESSED_FILENAME,
    get_language_name,
)


def preprocess_audio_file(
    input_path: str,
    output_path: str = WAVE_PREPROCESSED_FILENAME,
    audio_filter: str = STT_AUDIO_FILTER,
    sample_rate: int = RATE,
) -> str:
    """
    Apply a voice bandpass filter (80Hz-7.5kHz) and dynamic audio normalization via FFmpeg
    to boost quiet microphone recordings and strip low-frequency rumble before STT.
    Falls back to `input_path` if FFmpeg fails.
    """
    if not audio_filter:
        return input_path
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-af",
                audio_filter,
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                output_path,
            ],
            check=True,
            timeout=5,
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 44:
            return output_path
    except Exception as e:
        print(f"Warning: Audio preprocessing failed ({e}); using raw recording.")
    return input_path


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
        model_size: str = DEFAULT_STT_MODEL,
        compute_type: str = "int8",
        beam_size: int = DEFAULT_STT_BEAM_SIZE,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, float]] = None,
        allowed_languages: Optional[Sequence[str]] = DEFAULT_ALLOWED_LANGUAGES,
        preprocess_audio: bool = True,
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_parameters = (
            dict(vad_parameters) if vad_parameters is not None else dict(DEFAULT_VAD_PARAMETERS)
        )
        self.allowed_languages = (
            tuple(lang.lower().strip() for lang in allowed_languages if lang.strip())
            if allowed_languages
            else None
        )
        self.preprocess_audio = preprocess_audio

        print(f"Loading Whisper model ('{model_size}') with {compute_type.upper()} precision...")
        self.model = WhisperModel(
            model_size,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = DEFAULT_LANGUAGE,
    ) -> Optional[TranscriptionResult]:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) <= 44:
            return None

        target_audio_path = (
            preprocess_audio_file(audio_path) if self.preprocess_audio else audio_path
        )

        if language:
            lang_name = get_language_name(language)
            print(f"Transcribing in forced language: {lang_name} ({language})...")
        elif self.allowed_languages:
            allowed_str = ", ".join(self.allowed_languages)
            print(f"Transcribing and detecting language (candidates: {allowed_str})...")
        else:
            print("Transcribing and analyzing language...")

        sampling_rate = self.model.feature_extractor.sampling_rate
        audio = decode_audio(target_audio_path, sampling_rate=sampling_rate)

        speech_chunks = None
        if self.vad_filter:
            vad_opts = VadOptions(**self.vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_opts)
            if not speech_chunks:
                # No voice activity detected -> skip Whisper encoder entirely
                fallback_lang = language or (self.allowed_languages[0] if self.allowed_languages else "en")
                return TranscriptionResult(
                    text="",
                    language=fallback_lang,
                    language_probability=0.0,
                    forced_language=bool(language),
                    top_languages=[],
                    segments=[],
                )
            audio_chunks, _ = collect_chunks(audio, speech_chunks)
            audio = (
                np.concatenate(audio_chunks, axis=0)
                if audio_chunks and len(audio_chunks[0]) > 0
                else np.array([], dtype=np.float32)
            )
            if audio.size == 0:
                fallback_lang = language or (self.allowed_languages[0] if self.allowed_languages else "en")
                return TranscriptionResult(
                    text="",
                    language=fallback_lang,
                    language_probability=0.0,
                    forced_language=bool(language),
                    top_languages=[],
                )

        features = self.model.feature_extractor(audio)
        first_segment = pad_or_trim(features[:, : self.model.feature_extractor.nb_max_frames])
        # Encode first 30s window once and reuse for both language identification and decoding
        encoder_output = self.model.encode(first_segment)

        top_langs: List[Tuple[str, float]] = []
        if language:
            detected_lang = language.lower().strip()
            lang_prob = 1.0
        elif not self.model.model.is_multilingual:
            detected_lang = "en"
            lang_prob = 1.0
        else:
            raw_lang_results = self.model.model.detect_language(encoder_output)[0]
            all_probs = [(token[2:-2], float(prob)) for token, prob in raw_lang_results]
            if self.allowed_languages:
                filtered_probs = [
                    (code, prob) for code, prob in all_probs if code in self.allowed_languages
                ]
                if filtered_probs:
                    all_probs = filtered_probs
            top_langs = all_probs[:3]
            detected_lang, lang_prob = all_probs[0] if all_probs else ("en", 0.0)

        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=detected_lang,
        )

        options = TranscriptionOptions(
            beam_size=self.beam_size,
            best_of=self.beam_size,
            patience=1.0,
            length_penalty=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            prompt_reset_on_temperature=0.5,
            temperatures=[0.0, 0.2, 0.4],
            initial_prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=False,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",
            multilingual=False,
            max_new_tokens=None,
            clip_timestamps="0",
            hallucination_silence_threshold=None,
            hotwords=None,
        )

        segments_gen = self.model.generate_segments(
            features,
            tokenizer,
            options,
            log_progress=False,
            encoder_output=encoder_output,
        )
        if speech_chunks:
            segments_gen = restore_speech_timestamps(segments_gen, speech_chunks, sampling_rate)

        raw_segments = list(segments_gen)

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
