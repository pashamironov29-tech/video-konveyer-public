"""Надёжность расшифровки: есть ли в ролике речь, которой можно верить.

Whisper на музыке часто «слышит» чужой язык и повторы («ルルル», «thank you thank you»).
Одного сигнала мало: низкая вероятность языка сама по себе субтитры не выключает.
Сильный сигнал (повтор, зашкал сжатия) выключает сразу. Слабые копятся и срабатывают
пачкой — порог weak_to_reject. Все пороги лежат в [speech] в konveyer.toml.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from konveyer.config import SpeechCfg
from konveyer.transcript import Transcript

SKIP_CAPTIONS = "речи нет — субтитры пропущены"
NO_SPEECH_MATERIAL = (
    "(речи нет — субтитры пропущены. Не описывай содержание, сюжет, жанр и тему "
    "и не выдумывай. Опирайся только на название, автора и описание.)"
)

# Целиком такая расшифровка — типичный мусор Whisper на музыке и тишине.
# Одно «thank you» при уверенном языке само по себе субтитры не снимает: это слабый сигнал.
_FILLER = re.compile(
    r"(?ix)^(?:"
    r"[\W_]+|"
    r"thank(?:s|\s+you)(?:\s+for\s+watching)?|"
    r"thanks\s+for\s+watching|"
    r"please\s+subscribe(?:\s+to\s+my\s+channel)?|"
    r"subtitles?\s+by(?:\s+the\s+community)?|"
    r"you|yeah|okay|ok|hmm+|um+|"
    r"music|музыка|подписывайтесь|продолжение\s+следует|"
    r"字幕|ご視聴ありがとう(?:ございました)?"
    r")+$"
)
_WORDS = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SpeechVerdict:
    ok: bool
    reason: str
    strong: tuple[str, ...] = ()
    weak: tuple[str, ...] = ()


def _words(text: str) -> list[str]:
    return [w.casefold() for w in _WORDS.findall(text or "")]


def char_run_ratio(text: str) -> float:
    """Доля символов, которые входят в повтор одного знака 3+ подряд (пробелы не считаем)."""
    chars = [c.casefold() for c in (text or "") if not c.isspace()]
    if not chars:
        return 0.0
    in_run, i, n = 0, 0, len(chars)
    while i < n:
        j = i + 1
        while j < n and chars[j] == chars[i]:
            j += 1
        if j - i >= 3:
            in_run += j - i
        i = j
    return in_run / n


def top_word_ratio(text: str) -> tuple[float, int]:
    words = _words(text)
    if not words:
        return 0.0, 0
    top = Counter(words).most_common(1)[0][1]
    return top / len(words), len(words)


def unique_word_ratio(text: str) -> tuple[float, int]:
    words = _words(text)
    if not words:
        return 1.0, 0
    return len(set(words)) / len(words), len(words)


def _weighted(tr: Transcript, attr: str) -> float | None:
    num = den = 0.0
    for seg in tr.segments:
        value = getattr(seg, attr, None)
        if value is None:
            continue
        span = max(seg.end - seg.start, 0.0)
        if span <= 0:
            continue
        num += span * float(value)
        den += span
    if den <= 0:
        return None
    return num / den


def speech_coverage(tr: Transcript, duration: float) -> float:
    if duration <= 0:
        return 1.0
    total = 0.0
    for seg in tr.segments:
        if seg.text.strip():
            total += max(seg.end - seg.start, 0.0)
    return min(total / duration, 1.0)


def is_filler(text: str) -> bool:
    compact = " ".join((text or "").split())
    return bool(compact) and bool(_FILLER.match(compact))


def assess_speech(tr: Transcript, duration: float, cfg: SpeechCfg) -> SpeechVerdict:
    """ok=False — субтитры, перевод и текст расшифровки в посте лучше не использовать."""
    if tr.empty:
        return SpeechVerdict(False, "пустая расшифровка")

    text = tr.text()
    strong: list[str] = []
    weak: list[str] = []

    comp = _weighted(tr, "compression_ratio")
    if comp is not None and comp > cfg.max_compression_ratio:
        strong.append("сжатие как у повторов")

    visible = [c for c in text if not c.isspace()]
    if len(visible) >= 6 and char_run_ratio(text) >= cfg.max_char_run_ratio:
        strong.append("повтор символов")

    top, n_words = top_word_ratio(text)
    if n_words >= cfg.min_words_for_repeat and top >= cfg.max_top_word_ratio:
        strong.append("повтор слов")

    no_speech = _weighted(tr, "no_speech_prob")
    logprob = _weighted(tr, "avg_logprob")
    if (no_speech is not None and logprob is not None
            and no_speech >= cfg.combo_no_speech and logprob <= cfg.combo_logprob):
        strong.append("модель почти не слышит речь")

    if strong:
        return SpeechVerdict(False, ", ".join(strong), tuple(strong), ())

    if tr.language_probability is not None and tr.language_probability < cfg.min_language_prob:
        weak.append("низкая вероятность языка")
    if no_speech is not None and no_speech > cfg.max_no_speech_prob:
        weak.append("высокая вероятность тишины")
    if logprob is not None and logprob < cfg.min_avg_logprob:
        weak.append("низкая уверенность распознавания")
    if duration >= cfg.min_duration_for_ratio and speech_coverage(tr, duration) < cfg.min_speech_ratio:
        if tr.vad:
            weak.append("VAD почти не нашёл речь")
        else:
            weak.append("мало времени с речью")
    uniq, n_uniq = unique_word_ratio(text)
    if n_uniq >= cfg.min_words_for_unique and uniq < cfg.min_unique_word_ratio:
        weak.append("слишком мало разных слов")
    if is_filler(text):
        weak.append("типичная галлюцинация")

    if len(weak) >= cfg.weak_to_reject:
        return SpeechVerdict(False, ", ".join(weak), (), tuple(weak))
    return SpeechVerdict(True, "", (), tuple(weak))
