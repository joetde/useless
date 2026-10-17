import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import (
    DEFAULT_CONTEXT_IDLE_TIMEOUT,
    DEFAULT_MAX_CONTEXT_TURNS,
    SESSIONS_DIR,
    get_language_name,
)
from .llm import ChatTurn
from .stt import TranscriptionResult


class SessionLogger:
    """
    Manages per-session audit folders and rolling conversation context.

    Creates a dedicated folder under `sessions/session_<YYYYMMDD_HHMMSS>/` containing:
      - `session.json`  : Session metadata (start/end time, models, settings, turn count)
      - `turns.jsonl`   : Structured line-by-line JSON record of every interaction
      - `transcript.md` : Human-readable conversation log for easy parental auditing
    """

    def __init__(
        self,
        sessions_dir: Union[str, Path] = SESSIONS_DIR,
        max_context_turns: int = DEFAULT_MAX_CONTEXT_TURNS,
        context_idle_timeout: float = DEFAULT_CONTEXT_IDLE_TIMEOUT,
        session_metadata: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.max_context_turns = max(0, int(max_context_turns))
        self.context_idle_timeout = float(context_idle_timeout)
        self.started_at = datetime.now().astimezone()
        self.session_id = f"session_{self.started_at.strftime('%Y%m%d_%H%M%S')}"
        self.sessions_root = Path(sessions_dir)
        self.session_dir = self.sessions_root / self.session_id

        self.metadata_file = self.session_dir / "session.json"
        self.jsonl_file = self.session_dir / "turns.jsonl"
        self.transcript_file = self.session_dir / "transcript.md"

        self.turn_count = 0
        self._active_context: List[ChatTurn] = []
        self._last_interaction_monotonic: Optional[float] = None
        self.metadata: Dict[str, Any] = {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": None,
            "turn_count": 0,
            "max_context_turns": self.max_context_turns,
            "context_idle_timeout_sec": self.context_idle_timeout,
        }
        if session_metadata:
            self.metadata.update(session_metadata)

        if self.enabled:
            self._init_files()

    def _init_files(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_metadata()

        header_lines = [
            f"# Voice Session Transcript (`{self.session_id}`)",
            "",
            f"- **Started**: {self.started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        ]
        if "llm_provider" in self.metadata:
            llm_info = self.metadata.get("llm_provider", "none")
            if self.metadata.get("llm_model"):
                llm_info += f" ({self.metadata['llm_model']})"
            header_lines.append(f"- **LLM**: {llm_info}")
        if "stt_model" in self.metadata:
            header_lines.append(f"- **STT Model**: {self.metadata['stt_model']}")
        if "voice_effect" in self.metadata and self.metadata["voice_effect"]:
            header_lines.append(f"- **Voice Effect**: {self.metadata['voice_effect']}")
        header_lines.extend(["", "---", ""])

        with open(self.transcript_file, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))

    def _write_metadata(self) -> None:
        if not self.enabled:
            return
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def get_context_for_next_turn(self) -> List[ChatTurn]:
        """
        Return the recent `ChatTurn` list (up to `max_context_turns`) to pass to the LLM.
        If `context_idle_timeout > 0` and the time elapsed since the last turn exceeds it,
        automatically clears the in-memory context window.
        """
        if self.max_context_turns <= 0:
            return []

        now_mono = time.monotonic()
        if (
            self.context_idle_timeout > 0
            and self._last_interaction_monotonic is not None
            and (now_mono - self._last_interaction_monotonic) > self.context_idle_timeout
            and self._active_context
        ):
            idle_mins = (now_mono - self._last_interaction_monotonic) / 60.0
            print(
                f"  [Context] Idle for {idle_mins:.1f} min (> {self.context_idle_timeout / 60.0:.1f} min); "
                "starting fresh conversation context."
            )
            self._active_context.clear()
            if self.enabled:
                with open(self.transcript_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"*--- Context reset after {idle_mins:.1f} minutes of inactivity ---*\n\n"
                    )

        return list(self._active_context[-self.max_context_turns :])

    def clear_context(self) -> None:
        """Manually clear the active conversation context."""
        self._active_context.clear()
        self._last_interaction_monotonic = None

    def log_turn(
        self,
        transcription: TranscriptionResult,
        llm_response: Optional[str],
        context_turns_used: int = 0,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Record a completed user/LLM turn in the rolling context window and write to audit files.
        """
        now = datetime.now().astimezone()
        now_iso = now.isoformat(timespec="seconds")
        now_time_str = now.strftime("%H:%M:%S")
        self.turn_count += 1
        self._last_interaction_monotonic = time.monotonic()

        user_text = transcription.text.strip()
        assistant_text = (llm_response or "").strip()

        # Update rolling context if we got both user speech and an assistant reply
        if user_text and assistant_text and self.max_context_turns > 0:
            self._active_context.append(
                ChatTurn(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    language=transcription.language,
                    timestamp=now_iso,
                )
            )
            if len(self._active_context) > self.max_context_turns:
                self._active_context = self._active_context[-self.max_context_turns :]

        if not self.enabled:
            return

        lang_name = get_language_name(transcription.language)
        turn_record: Dict[str, Any] = {
            "turn": self.turn_count,
            "timestamp": now_iso,
            "language": transcription.language,
            "language_name": lang_name,
            "language_probability": round(float(transcription.language_probability), 4),
            "forced_language": bool(transcription.forced_language),
            "user_transcript": user_text,
            "segments": [
                {
                    "start": round(float(seg.start), 2),
                    "end": round(float(seg.end), 2),
                    "text": seg.text,
                }
                for seg in transcription.segments
            ],
            "llm_response": assistant_text if llm_response is not None else None,
            "context_turns_used": context_turns_used,
            "llm_provider": llm_provider or self.metadata.get("llm_provider"),
            "llm_model": llm_model or self.metadata.get("llm_model"),
        }

        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn_record, ensure_ascii=False) + "\n")

        lang_badge = (
            f"{lang_name} ({transcription.language}, forced)"
            if transcription.forced_language
            else f"{lang_name} ({transcription.language}, {transcription.language_probability * 100:.1f}%)"
        )
        ctx_badge = f" | Context: {context_turns_used} prior turn(s)" if context_turns_used > 0 else ""

        md_lines = [
            f"### Turn {self.turn_count} — {now_time_str} [{lang_badge}{ctx_badge}]",
            "",
            f"**Child (Voice Transcript):**",
            f"> {user_text or '(No speech detected)'}",
            "",
        ]
        if transcription.segments:
            seg_lines = [
                f"- `[{seg.start:5.2f}s -> {seg.end:5.2f}s]` {seg.text}"
                for seg in transcription.segments
            ]
            md_lines.extend(["<details><summary>Timestamped Segments</summary>", ""])
            md_lines.extend(seg_lines)
            md_lines.extend(["", "</details>", ""])

        md_lines.extend(
            [
                f"**Assistant (LLM Response):**",
                f"> {assistant_text if llm_response is not None else '(LLM disabled / no response)'}",
                "",
                "---",
                "",
            ]
        )

        with open(self.transcript_file, "a", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        self.metadata["turn_count"] = self.turn_count
        self.metadata["last_turn_at"] = now_iso
        self._write_metadata()

    def close(self) -> None:
        """Finalize session metadata when the session ends."""
        if not self.enabled:
            return
        ended_at = datetime.now().astimezone()
        self.metadata["ended_at"] = ended_at.isoformat(timespec="seconds")
        self.metadata["turn_count"] = self.turn_count
        self._write_metadata()
