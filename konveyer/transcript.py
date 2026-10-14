"""Расшифровка речи: сегменты со словами и их таймингом. Чистые данные, без Whisper."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    word: str          # как у faster-whisper: обычно с пробелом впереди
    speaker: str = ""  # S1, S2… пусто — спикер неизвестен, один стиль субтитров


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    speaker: str = ""


@dataclass
class Transcript:
    language: str = ""
    segments: list[Segment] = field(default_factory=list)
    language_probability: float | None = None
    vad: bool | None = None          # True — сегменты уже прошли vad_filter
    reliable: bool = True            # False — речи нет или ей нельзя верить
    speech_reason: str = ""

    @property
    def words(self) -> list[Word]:
        out: list[Word] = []
        for s in self.segments:
            if s.words:
                out.extend(s.words)
            elif s.text.strip():
                out.append(Word(s.start, s.end, " " + s.text.strip(), s.speaker))
        return out

    @property
    def empty(self) -> bool:
        return not any(s.text.strip() for s in self.segments)

    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def words_between(self, start: float, end: float) -> list[Word]:
        """Слова, середина которых попадает в [start, end]."""
        return [w for w in self.words if start <= (w.start + w.end) / 2 <= end]

    def to_dict(self) -> dict:
        return {"language": self.language, "language_probability": self.language_probability,
                "vad": self.vad, "reliable": self.reliable, "speech_reason": self.speech_reason,
                "segments": [asdict(s) for s in self.segments]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Transcript":
        if not d:
            return cls()
        segs = []
        for s in d.get("segments", []):
            words = [Word(float(w["start"]), float(w["end"]), str(w["word"]),
                          str(w.get("speaker") or "")) for w in s.get("words", [])]
            segs.append(Segment(float(s["start"]), float(s["end"]), str(s.get("text", "")), words,
                                _opt_float(s.get("no_speech_prob")), _opt_float(s.get("avg_logprob")),
                                _opt_float(s.get("compression_ratio")), str(s.get("speaker") or "")))
        reliable = d.get("reliable", True)
        if not isinstance(reliable, bool):
            reliable = str(reliable).strip().lower() not in ("0", "false", "нет", "no")
        vad = d.get("vad", None)
        if not isinstance(vad, bool):
            vad = None
        return cls(str(d.get("language", "")), segs, _opt_float(d.get("language_probability")),
                   vad, reliable, str(d.get("speech_reason") or ""))


def _opt_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
