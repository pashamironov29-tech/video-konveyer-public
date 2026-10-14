"""Русская озвучка клипа: переведённые реплики → Edge (или Silero, или XTTS) → voice.wav.

По умолчанию режим voiceover: закадровый перевод. Голос читает перевод
крупными блоками — несколько предложений подряд, пока не сменится пауза.
В синтез уходит текст без запятых, тире и многоточий: точки и вопросы остаются,
субтитры на экране не трогаем. Начало и конец блока не обрезаются и не затухают,
следующий блок не накладывается на предыдущий. Если русский не влезает в окно
с запасом около 2–3 с, фразу ужимают и чуть ускоряют, а не копят отставание.

Режим fit — прежняя подгонка под окно: Edge ускоряет rate не выше edge_max_rate,
дальше пауза соседа, в крайнем случае с конца уходят целые слова.

В синтез уходит текст после prepare_for_speech (цифры словами, латиница по-русски,
«З» → «зэ», «ИИ» → «и-и»). Строка субтитра — та же фраза, что говорит голос.
Паузы между предложениями — тишина на нашей стороне: в Edge уходит только чистый текст.
Оригинал приглушается на voice.duck_db и не возвращается, пока предложение не кончилось.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from konveyer.captions import Cue
from konveyer.config import (EDGE_VOICES, Settings, VoiceCfg, combine_hz,
                             combine_rates, format_percent, parse_hz, parse_percent,
                             speaker_error)
from konveyer.saytext import prepare_for_speech
from konveyer.llm import chat_json, set_stage
from konveyer.log import Log
from konveyer.transcript import Transcript
from konveyer.util import KonveyerError, need_tool, read_json, run, write_json

MIX_RATE = 48000
EDGE_ATTEMPTS = 8          # сколько раз спросить Edge, прежде чем оставить фразу без озвучки
EDGE_RETRY_CAP = 8.0       # верх паузы между попытками, секунды
EDGE_FALLBACK_ATTEMPTS = 3 # попыток на каждый запасной голос того же пола
EDGE_PATIENT_SEC = 120.0   # столько секунд пауз ещё ждём основной голос, прежде чем сдаться
EDGE_PATIENT_CAP = 20.0    # пауза между такими терпеливыми попытками
EDGE_TIMEOUT_SEC = 20.0    # один запрос к Edge дольше — считаем неудачей (зависший сокет)
EDGE_BREAKER = 12          # столько неудач подряд (по всем потокам) — Edge «лёг»: пауза для всех
EDGE_REST_SEC = 30.0       # пауза-отдых всех потоков после срабатывания выключателя
EDGE_PIECE_WORDS = 12      # длинную фразу, которую Edge не отдал целиком, режем на такие куски
# Запасные мультиязычные голоса: за спикерами их не закрепляют, поэтому они всегда свободны.
# Andrew намеренно нет. ru-RU-DariyaNeural в Edge не существует.
EDGE_SPARE_MALE = ("fr-FR-RemyMultilingualNeural", "en-AU-WilliamMultilingualNeural")
EDGE_SPARE_FEMALE = ("fr-FR-VivienneMultilingualNeural",)
VOICE_BORROW = 0.8         # сколько секунд можно занять у соседней реплики (режим fit)
VOICE_GAP = 0.02           # зазор между двумя голосами в режиме fit, чтобы они не слиплись
VOICEOVER_GAP = 0.15       # пауза между блоками закадра, если оригинал сам не молчит дольше
VOICE_FADE_SEC = 0.045     # мягкий конец куска в режиме fit; в закадре фейда нет
BLOCK_PAUSE = 0.65         # пауза оригинала, после которой начинается новый блок озвучки
BLOCK_SPAN = 22.0          # дольше этого блок режем по концу предложения
BLOCK_CHARS = 420
BLOCK_HARD_SPAN = 36.0     # даже середину мысли не тянем бесконечно
VOICE_CPS = 17.0           # примерно столько символов в секунду читает Edge (замер: 15–17)
DRIFT_SEC = 2.0            # русский блок ушёл вперёд оригинала больше этого — новый блок со своего места
BLOCK_HARD_CHARS = 800
# Резать озвучку посреди предложения нельзя: Edge ставит точку, и после «эти» — тишина.
# Запасная граница — запятая/тире/точка с запятой, если с обеих сторон хотя бы столько символов
# и последнее слово не служебное. Двоеточие не граница: за ним обычно перечисление.
SOFT_CUT_CHARS = 40
LAST_RESORT_SPAN = 60.0    # дальше предложение всё же режем, но без точки и с паузой <= JOIN_GAP
LAST_RESORT_CHARS = 1200
JOIN_GAP = 0.4
JOIN_PAUSE = 0.25          # слышимая пауза на стыке кусков одного предложения
JOIN_LOUD = 0.01           # громче этого — уже речь, тише — тишина в начале или конце файла Edge


def _speech_bounds(audio: np.ndarray) -> tuple[int, int] | None:
    """Первый и последний сэмпл речи в файле Edge (у него тишина в начале и в конце)."""
    loud = np.flatnonzero(np.abs(np.asarray(audio, np.float32).reshape(-1)) > JOIN_LOUD)
    if loud.size == 0:
        return None
    return int(loud[0]), int(loud[-1]) + 1
NO_CUT_AFTER = frozenset("""
и а но да или либо что чтобы как когда если хотя чем будто который которая которое которые
которого которой которых которым которыми котором это эти этот эта этих этим этими этой этому
тот та те то того тех тем теми той такой такая такие такое так там тут здесь вот
в во на с со к ко по о об обо от до из у за для при без над под через про между перед после
около среди вместо ради сквозь не ни же ли бы уже ещё еще даже лишь только именно
его её ее их мой моя мои моё свой своя свои своё наш наша наши ваш ваша ваши один одна одно
одни два две три четыре пять шесть семь восемь девять десять сто тысяча тысячи тысяч сотня
сотни сотен десятки десятков миллион миллиона миллионов миллиард миллиарда миллиардов
несколько много мало очень самый самая самое самые более менее всё все весь вся каждый
""".split())


def soft_cut_ok(text: str, left_chars: int, right_chars: int) -> bool:
    """Можно ли в крайнем случае закончить кусок озвучки на этой строке посреди предложения."""
    if left_chars < SOFT_CUT_CHARS or right_chars < SOFT_CUT_CHARS:
        return False
    t = (text or "").rstrip()
    while t and t[-1] in _TRAIL:
        t = t[:-1].rstrip()
    if not t or t[-1] not in ",;—–":
        return False
    words = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", t)
    if not words:
        return False
    last = words[-1].casefold()
    return not (last in NO_CUT_AFTER or last.isdigit())
LAG_LIMIT = 2.5            # закадр Shorts не отстаёт сильнее
TAIL_LIMIT = 3.0           # и не растягивает конец сильнее
DEMUCS_MISSING = ("demucs не установлен — оставляю оригинальный звук с приглушением. "
                  "Поставь: py -m pip install demucs")
XTTS_WORKER = Path(__file__).with_name("xtts_worker.py")

SHORTEN_SYSTEM = (
    "Ты сокращаешь фразу для русской озвучки видео. Она не успевает прозвучать за своё время. "
    "Сохрани смысл, тон и вопрос/восклицание, убери лишнее, пиши разговорно. "
    "Не добавляй ничего от себя. Верни ТОЛЬКО JSON: {\"text\": \"короткая фраза\"}."
)


# ------------------------------------------------------------------ когда озвучивать

def voice_gate(settings: Settings, tr: Transcript, llm) -> str:
    """Причина пропустить озвучку всего задания или "" — можно озвучивать."""
    if not tr.reliable:
        why = f" ({tr.speech_reason})" if tr.speech_reason else ""
        return f"озвучка пропущена: речи нет{why}"
    if not tr.words:
        return "озвучка пропущена: в расшифровке нет слов"
    if tr.language == "ru":
        return "озвучка пропущена: ролик уже на русском"
    if llm is None:
        return "озвучка пропущена: нужен перевод на русский, а DEEPSEEK_API_KEY не задан"
    return ""


# ------------------------------------------------------------------ текст и SSML

_SENTENCE = re.compile(r"[^.!?…]+(?:[.!?…]+|$)")


def clean_text(text: str) -> str:
    """Убрать то, что синтезатор читает плохо: кавычки, скобки, тире, эмодзи."""
    t = text.replace("—", ", ").replace("–", ", ").replace("...", "…").replace("&", " и ")
    t = re.sub(r"[«»\"“”„()\[\]{}*_#|/\\<>]", " ", t)
    t = "".join(ch for ch in t if ch.isalnum() or ch.isspace() or ch in ".,!?…:;-'")
    t = re.sub(r"\s+([.,!?…:;])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip(" ,;:-")


def strip_speech_punct(text: str) -> str:
    """Убрать запятые, тире и многоточия. Точки, вопросы и восклицания оставить.

    Дефис внутри слова («по-русски», «джи-пи-ти») не трогаем: паузу даёт только тире.
    Субтитры этой функцией не проходят — только строка для синтеза.
    """
    t = (text or "").replace("…", " ").replace("...", " ")
    t = t.replace("—", " ").replace("–", " ").replace(",", " ")
    t = re.sub(r"(?<=\s)-(?=\s)", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([.!?])", r"\1", t)
    return t.strip(" ,;:-")


_SYLL = re.compile(r"[аеёиоуыэюяaeiouy]+", re.IGNORECASE)


def count_syllables(text: str) -> int:
    return max(len(_SYLL.findall(text or "")), 1)


def ssml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def build_ssml(text: str, pause_ms: int = 300, ellipsis_ms: int = 500) -> str:
    """Интонация по знакам: вопрос выше, восклицание выше и бодрее, пауза на точке.

    После последнего предложения паузы нет — окно реплики и так кончается.
    """
    sentences = [s.strip() for s in _SENTENCE.findall(clean_text(text)) if s.strip()]
    parts = []
    for i, s in enumerate(sentences):
        body = ssml_escape(s)
        if s.endswith("?"):
            parts.append(f'<prosody pitch="high">{body}</prosody>')
        elif s.endswith("!"):
            parts.append(f'<prosody rate="fast" pitch="high">{body}</prosody>')
        else:
            parts.append(body)
        if i < len(sentences) - 1:
            if s.endswith("…"):
                parts.append(f'<break time="{ellipsis_ms}ms"/>')
            elif s[-1] in ".?!":
                parts.append(f'<break time="{pause_ms}ms"/>')
    return "<speak>" + " ".join(parts) + "</speak>"


def phrase_kind(text: str, serious_chars: int = 80) -> str:
    """Тип фразы для интонации Edge: вопрос, восклицание, многоточие, длинное, обычная."""
    t = clean_text(text)
    question, exclaim = t.rfind("?"), t.rfind("!")
    if question >= 0 or exclaim >= 0:
        return "question" if question > exclaim else "exclamation"
    if "…" in t:
        return "ellipsis"
    if len(t) >= max(int(serious_chars), 1):
        return "serious"
    return "normal"


def edge_text_ok(text: str) -> bool:
    """В Edge нельзя отдавать разметку: он читает её вслух как обычные слова."""
    low = (text or "").casefold()
    return ("<" not in (text or "") and ">" not in (text or "")
            and "break" not in low and "time=" not in low
            and "\ue000" not in (text or "") and "\ue001" not in (text or ""))


def edge_plain(text: str) -> str:
    """Убрать теги и служебные куски. Подготовленная речь остаётся словами."""
    t = (text or "").replace("\ue000", " ").replace("\ue001", " ")
    t = re.sub(r"<[^>]*>", " ", t)
    t = t.replace("<", " ").replace(">", " ")
    t = re.sub(r"(?i)\btime\s*=", " ", t)
    t = re.sub(r"(?i)\bbreak\b", " ", t)
    return " ".join(t.split())


def edge_segments(text: str, pause_ms: int = 300, ellipsis_ms: int = 500) -> list[tuple[str, int]]:
    """Фразы для Edge и пауза после каждой, в миллисекундах. Последняя без паузы.

    Пауза не тег: её вставляет движок тишиной между отдельными запросами.
    Текст уже прошёл prepare_for_speech, сюда разметка не добавляется.
    """
    raw = edge_plain(text)
    sentences = [s.strip() for s in _SENTENCE.findall(raw) if s.strip()]
    if not sentences:
        return [(raw, 0)] if raw else []
    out: list[tuple[str, int]] = []
    for i, sentence in enumerate(sentences):
        pause = 0
        if i < len(sentences) - 1:
            if sentence.endswith("…"):
                pause = int(ellipsis_ms)
            elif sentence[-1] in ".?!":
                pause = int(pause_ms)
        out.append((sentence, pause))
    return out


def edge_markup(text: str, pause_ms: int = 300, ellipsis_ms: int = 500) -> str:
    """Чистый текст для Edge, без тегов. Паузы живут в edge_segments, не в строке."""
    return " ".join(part for part, _pause in edge_segments(text, pause_ms, ellipsis_ms))


def _expr_on(expr: dict) -> bool:
    value = expr.get("enabled", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "да")


@dataclass
class Prosody:
    rate: str
    pitch: str
    volume: str
    kind: str

    def key(self) -> str:
        return f"{self.rate}|{self.pitch}|{self.volume}|{self.kind}"


def expression_prosody(text: str, cfg: VoiceCfg, boost: float = 1.0,
                       base_rate: str | None = None) -> Prosody:
    """Итоговые rate/pitch/volume: базовые + интонация фразы + ускорение (boost).

    boost — во сколько раз быстрее уже измеренной фразы. 1.0 — только интонация.
    Ускорение перемножается с rate интонации, а не заменяет его.
    base_rate подменяет [voice] edge_rate: у целого ролика свой чуть более быстрый темп.
    """
    expr = cfg.expression or {}
    kind = phrase_kind(text, int(expr.get("serious_chars", 80)))
    if _expr_on(expr):
        rate_s = str(expr.get(f"{kind}_rate", "+0%"))
        pitch_s = str(expr.get(f"{kind}_pitch", "+0Hz"))
        volume = str(expr.get(f"{kind}_volume", "+0%"))
    else:
        rate_s, pitch_s, volume = "+0%", "+0Hz", "+0%"
    base = cfg.edge_rate if base_rate is None else base_rate
    return Prosody(combine_rates(base, rate_s, boost),
                   combine_hz(cfg.edge_pitch, pitch_s), volume, kind)


PITCH_CHARS = 80     # длиннее — тон голоса не сдвигаем (вопрос/серьёзно), голос не «меняется»
SLOW_FILL = 0.85     # русский короче своего окна больше чем на 15% — читаем медленнее
SLOW_MIN = 0.85      # но не медленнее 0,85 от обычного темпа


def block_prosody(text: str, cfg: VoiceCfg, boost: float, base_rate: str,
                  ceiling: float, slow: float = 1.0) -> Prosody:
    """Темп блока: не медленнее базового и не быстрее потолка.

    Длинная фраза в закадре не замедляется «серьёзной» интонацией — от этого
    как раз копилось отставание. Вопрос по-прежнему выше тоном.
    """
    pros = expression_prosody(text, cfg, 1.0, base_rate=base_rate)
    if len(text or "") > PITCH_CHARS:
        # Тон сдвигаем только у коротких реплик: длинный кусок с «+8 Гц» на вопросе в конце
        # звучит как другой человек. Интонацию вопроса Edge и так делает сам по «?».
        pros = Prosody(pros.rate, combine_hz(cfg.edge_pitch, "+0Hz"), pros.volume, pros.kind)
    combined = 1.0 + parse_percent(pros.rate) / 100.0
    floor = 1.0 + parse_percent(base_rate) / 100.0
    final = max(combined, floor) * max(float(boost), 1.0)
    final = min(final, max(float(ceiling), 1.0))
    final *= min(max(float(slow), SLOW_MIN), 1.0)
    return Prosody(format_percent(final), pros.pitch, pros.volume, pros.kind)


def capped_boost(cfg: VoiceCfg, text: str, wanted: float) -> float:
    """Ускорение так, чтобы итог с интонацией не превысил edge_max_rate.

    wanted — во сколько раз короче уже измеренная фраза. Базовый rate и интонация
    уже сидят в проценте Edge, поэтому запас считается от них, а не сверху.
    """
    wanted = max(float(wanted), 1.0)
    ceiling = max(float(cfg.edge_max_rate), 1.0)
    base = expression_prosody(text, cfg, 1.0)
    base_factor = 1.0 + parse_percent(base.rate) / 100.0
    if base_factor <= 1e-6:
        return 1.0
    room = ceiling / base_factor
    return min(wanted, max(room, 1.0))


def expression_fingerprint(cfg: VoiceCfg) -> str:
    blob = json.dumps(cfg.expression, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def cache_key(tag: str, text: str, extra: str = "") -> str:
    raw = f"{tag}|{extra}|{text}" if extra else f"{tag}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


# ------------------------------------------------------------------ подгонка по времени

@dataclass
class Phrase:
    start: float              # время в готовом клипе (после вырезки пауз)
    slot: float               # сколько секунд есть до следующей реплики
    text: str
    spoken: str = ""          # что реально прозвучало (после сокращения)
    tempo: float = 1.0
    duration: float = 0.0     # длина после ускорения и обрезки
    status: str = ""          # ok | tempo | short | cut | fail
    said: str = ""            # что ушло в синтез (цифры и латиница уже словами)
    kind: str = ""            # question | exclamation | serious | ellipsis | normal
    rate: str = ""            # итоговый rate Edge (с интонацией и ускорением)
    dropped: str = ""         # целые слова, которые не влезли даже после сдвига окна
    origin: float = 0.0       # начало реплики до сдвига окна
    cue_end: float = 0.0      # конец английской реплики, чтобы видеть паузу до следующей
    speaker: str = ""         # S1, S2… пусто — голос по умолчанию
    voice: str = ""           # полное имя Edge, если для этого спикера другой голос
    fallback: str = ""        # запасной голос, если основной так и не ответил


def phrases_from_cues(cues: list[Cue], total: float, gap: float = 0.05,
                      min_slot: float = 0.2) -> list[Phrase]:
    out = []
    for i, c in enumerate(cues):
        shown = " ".join((c.text or "").split())
        if not any(ch.isalpha() for ch in clean_text(shown)) or c.start >= total:
            continue
        nxt = cues[i + 1].start if i + 1 < len(cues) else total
        slot = max(min(nxt, total) - c.start - gap, 0.0)
        # В закадре короткое английское окно фразу не выкидывает: её всё равно читают целиком.
        if min_slot > 0 and slot <= min_slot:
            continue
        start = round(c.start, 3)
        out.append(Phrase(start, round(slot, 3), shown, origin=start,
                          cue_end=round(float(c.end), 3),
                          speaker=str(getattr(c, "speaker", "") or "")))
    return out


_TRAIL = "\"'«»“”„()[]{}"  # кавычки и скобки после знака конец предложения не прячут


def continues_sentence(text: str) -> bool:
    """Строка не закончила мысль: на конце нет точки, вопроса, восклицания или многоточия.

    Закрывающая кавычка или скобка после знака не мешает: «свободы.» — конец.
    """
    t = (text or "").rstrip()
    while t and t[-1] in _TRAIL:
        t = t[:-1].rstrip()
    return bool(t) and t[-1] not in ".!?…"


def sentence_groups(texts: list[str], speakers: list[str] | None = None) -> list[list[int]]:
    """Индексы соседних строк одного предложения. Граница — . ! ? … в конце строки.

    Разные спикеры в один запрос не склеиваются: у каждого свой голос.
    """
    speakers = speakers or [""] * len(texts)
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, text in enumerate(texts):
        if cur and speakers[i] != speakers[cur[0]]:
            groups.append(cur)
            cur = []
        cur.append(i)
        if not continues_sentence(text):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def duck_spans(phrases: list[Phrase], bridge_sentence: bool = False) -> list[tuple[float, float]]:
    """Интервалы приглушения оригинала.

    bridge_sentence: между кусками одной фразы дырки нет, английский не выныривает.
    """
    spans: list[tuple[float, float]] = []
    prev = ""
    for ph in phrases:
        if ph.status == "fail" or ph.duration <= 0:
            continue
        a, b = ph.start, round(ph.start + ph.duration, 3)
        if spans and bridge_sentence and continues_sentence(prev):
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
        prev = ph.text
    return spans


@dataclass(frozen=True)
class WordMark:
    """Граница слова в секундах от начала куска звука."""
    text: str
    start: float
    end: float


EDGE_TICKS_PER_SEC = 10_000_000  # edge-tts: Offset и Duration в тиках по 100 нс


def _word_key(text: str) -> str:
    t = (text or "").casefold().replace("ё", "е")
    return "".join(ch for ch in t if ch.isalnum() or ch == "-")


def _part_keys(parts: list[str]) -> list[list[str]] | None:
    grouped: list[list[str]] = []
    for part in parts:
        keys = [_word_key(w) for w in (part or "").split()]
        keys = [k for k in keys if k]
        if not keys:
            return None
        grouped.append(keys)
    return grouped


def marks_from_edge_events(events: list[dict]) -> list[WordMark]:
    """WordBoundary из edge-tts.stream: offset и duration в тиках по 100 нс."""
    out: list[WordMark] = []
    for ev in events:
        if ev.get("type") not in (None, "WordBoundary"):
            continue
        try:
            offset = float(ev.get("offset") or 0)
            duration = float(ev.get("duration") or 0)
        except (TypeError, ValueError):
            continue
        start = round(offset / EDGE_TICKS_PER_SEC, 6)
        end = round(start + duration / EDGE_TICKS_PER_SEC, 6)
        out.append(WordMark(str(ev.get("text") or ""), start, end))
    return out


def _proportional_cuts(n: int, parts: list[str]) -> list[int]:
    """Границы срезов по длине текста. Последний срез доходит до конца звука."""
    count = len(parts)
    if count == 0:
        return [0]
    if n <= 0:
        return [0] * (count + 1)
    weights = [max(len((part or "").strip()), 1) for part in parts]
    total = sum(weights)
    cuts = [0]
    acc = 0
    for weight in weights[:-1]:
        acc += weight
        cuts.append(int(round(n * acc / total)))
    cuts.append(n)
    for i in range(1, len(cuts) - 1):
        room = n - (len(cuts) - 1 - i)
        cuts[i] = min(max(cuts[i], cuts[i - 1] + 1), room)
    cuts[-1] = n
    return cuts


def _cuts_from_marks(n: int, rate: int, parts: list[str],
                     marks: list[WordMark] | None) -> list[int] | None:
    """Срезы по началам слов Edge. None — границы не легли на куски, нужен запасной путь."""
    grouped = _part_keys(parts)
    if not grouped or not marks or rate <= 0 or n <= 0:
        return None
    usable = [m for m in marks if _word_key(m.text)]
    flat = [key for keys in grouped for key in keys]
    if len(usable) != len(flat):
        return None
    if any(_word_key(m.text) != key for m, key in zip(usable, flat)):
        return None
    raw: list[int] = []
    idx = 0
    prev = 0
    for keys in grouped[:-1]:
        idx += len(keys)
        sample = int(round(usable[idx].start * rate))
        if sample <= prev or sample >= n:
            return None
        raw.append(sample)
        prev = sample
    return [0, *raw, n]


def split_audio_parts(samples: np.ndarray, rate: int, parts: list[str],
                      marks: list[WordMark] | None = None) -> list[np.ndarray]:
    """Разложить один звук предложения по строкам субтитров.

    Сначала границы слов (WordBoundary). Если их нет или они не совпали со словами
    кусков — пропорционально длине текста. Сумма кусков равна исходному звуку:
    внутри предложения дырки не появляется.
    """
    audio = np.asarray(samples, np.float32).reshape(-1)
    if not parts:
        return []
    if len(parts) == 1:
        return [audio]
    cuts = _cuts_from_marks(audio.size, rate, parts, marks)
    if cuts is None:
        cuts = _proportional_cuts(audio.size, parts)
    return [audio[a:b] for a, b in zip(cuts, cuts[1:])]


def fade_tail(samples: np.ndarray, rate: int, seconds: float = VOICE_FADE_SEC) -> np.ndarray:
    """Короткое затухание в конце куска. Число сэмплов то же, слова не отрезаются."""
    if samples.size == 0 or seconds <= 0 or rate <= 0:
        return samples
    n = int(rate * seconds)
    n = min(n, max(samples.size // 3, 0))
    if n < 2:
        return samples
    out = np.array(samples, dtype=np.float32, copy=True)
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def plan_clip_tail(source_duration: float, clip_end: float, clip_len: float,
                   voice_end: float, tightened: bool = False) -> tuple[float, float]:
    """Сколько секунд дочитать из исходника и сколько добрать мягким затемнением.

    Голос не режем. Хвост берётся из source.mp4 с момента, где клип кончился,
    даже если внутри клипа уже вырезаны паузы: tightened на это не влияет.
    Если исходника не хватает, остаток — не стоп-кадр, а плавное затемнение.
    """
    # tightened не отключает хвост: паузы вырезаны внутри, исходник продолжается с clip_end.
    _ = tightened
    need = voice_end - clip_len
    if need <= 0.02:
        return 0.0, 0.0
    left = max(0.0, source_duration - clip_end)
    take = min(need, left)
    return round(take, 3), round(max(0.0, need - take), 3)


def plan_fit(duration: float, slot: float, max_tempo: float) -> tuple[str, float]:
    """ok — влезает; tempo — влезет с ускорением; shorten — даже max_tempo мало."""
    if duration <= slot + 1e-6:
        return "ok", 1.0
    need = duration / max(slot, 1e-6)
    if need <= max_tempo + 1e-6:
        return "tempo", round(need, 3)
    return "shorten", max_tempo


def schedule_voice(origins: list[float], durations: list[float], total: float,
                   lag_sec: float, max_rate: float, gap: float = VOICE_GAP) -> list[dict]:
    """Куда положить уже озвученные фразы длинного ролика.

    Пауза оригинала (тишина до следующей реплики) съедает отставание. Ускорение —
    только если без него отставание к следующей реплике больше lag_sec, и не сильнее
    max_rate. Последняя фраза не ускоряется: хвост длиннее ролика остаётся как есть.
    Слова эта функция не режет. duration 0 — фраза без звука, курсор не двигает.
    """
    if len(origins) != len(durations):
        raise KonveyerError("озвучка: число фраз и длин не совпало")
    max_rate = max(float(max_rate), 1.0)
    lag_sec = max(float(lag_sec), 0.0)
    gap = max(float(gap), 0.0)
    cursor = 0.0
    out: list[dict] = []
    last_i = len(origins) - 1
    for i, (origin, natural) in enumerate(zip(origins, durations)):
        origin = max(float(origin), 0.0)
        natural = max(float(natural), 0.0)
        if natural <= 0.04:
            out.append({"start": round(max(origin, cursor), 3), "tempo": 1.0,
                        "duration": 0.0, "lag": 0.0, "status": "fail"})
            continue
        start = max(origin, cursor)
        lag = max(start - origin, 0.0)
        if i == last_i:
            tempo, dur = 1.0, natural
            end = start + dur
            if end > float(total) + 0.05:
                status = "tail"
            elif lag > 0.08:
                status = "pause"
            else:
                status = "ok"
        else:
            next_origin = max(float(origins[i + 1]), 0.0)
            allowance = max(next_origin + lag_sec - start, 0.05)
            if natural <= allowance + 0.04:
                tempo, dur = 1.0, natural
                spilled = natural > max(next_origin - start, 0.0) + 0.08
                status = "pause" if lag > 0.08 or spilled else "ok"
            else:
                need = natural / allowance
                tempo = min(need, max_rate)
                dur = natural / tempo
                status = "tempo" if dur <= allowance + 0.04 else "over"
        out.append({
            "start": round(start, 3),
            "tempo": round(tempo, 4),
            "duration": round(dur, 3),
            "lag": round(lag, 3),
            "status": status,
        })
        cursor = start + dur + gap
    return out


def max_phrase_ends(starts: list[float], durations: list[float], total: float,
                    borrow: float = VOICE_BORROW, gap: float = VOICE_GAP) -> list[float]:
    """Поздний допустимый конец каждой фразы, в секундах клипа.

    Своё окно кончается у начала следующей. Если следующая короче своего окна,
    у неё забираем этот запас, но не больше borrow: голос чуть заходит на
    следующую реплику оригинала, а сама следующая озвучка сдвигается позже.
    """
    slots: list[float] = []
    for i, start in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else total
        slots.append(max(nxt - start - gap, 0.0))
    extra = [0.0] * len(starts)
    for i in range(len(starts) - 2, -1, -1):
        slack = slots[i + 1] + extra[i + 1] - max(durations[i + 1], 0.0)
        extra[i] = min(max(borrow, 0.0), max(0.0, slack))
    return [starts[i] + slots[i] + extra[i] for i in range(len(starts))]


def fit_words(text: str, duration: float, limit: float) -> tuple[str, str]:
    """(оставить, отбросить). Пустой хвост — фраза и так влезает.

    Режет только между словами. Одно слово, которое длиннее окна, возвращается
    целиком: сэмплы посреди него не пилим. Если слов несколько и звук длиннее
    окна — с конца уходит хотя бы одно слово, чтобы пересинтез стал короче.
    """
    words = [w for w in (text or "").split() if w]
    if not words or duration <= limit + 0.04:
        return " ".join(words), ""
    weights = [max(sum(ch.isalnum() for ch in w), 1) for w in words]
    budget = limit / max(duration, 1e-6) * sum(weights) * 0.92
    acc = 0.0
    keep = 0
    for i, weight in enumerate(weights):
        if i > 0 and acc + weight > budget:
            break
        acc += weight
        keep = i + 1
    if keep >= len(words):
        keep = max(len(words) - 1, 1)
    if keep >= len(words):
        return " ".join(words), ""
    return " ".join(words[:keep]), " ".join(words[keep:])


def align_cues_to_voice(cues: list[Cue], phrases: list[Phrase] | None,
                        to_voice: bool = False) -> list[Cue]:
    """Субтитры едут за озвучкой: то же начало и тот же текст, что реально сказан.

    to_voice: конец строки — конец русского голоса, английское окно не растягивает показ.
    Окна, которые голос не трогал, остаются как были. Если фразы наезжают друг
    на друга после сдвига, предыдущая кончается там, где начинается следующая.
    """
    if not cues or not phrases:
        return cues
    used: set[int] = set()
    for cue in cues:
        match = None
        shown = " ".join((cue.text or "").split())
        for i, ph in enumerate(phrases):
            if i in used or ph.duration <= 0 or ph.status == "fail":
                continue
            same = " ".join((ph.text or "").split()) == shown
            if same and abs(ph.origin - cue.start) < 0.05:
                match = i
                break
        if match is None:
            continue
        used.add(match)
        ph = phrases[match]
        cue.start = ph.start
        voice_end = ph.start + max(ph.duration, 0.2)
        if to_voice or cue.end <= cue.start:
            cue.end = voice_end
        else:
            cue.end = max(cue.end, voice_end)
        if ph.dropped and ph.spoken:
            cue.text = ph.spoken
    ordered = sorted(cues, key=lambda c: (c.start, c.end))
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev.end > nxt.start:
            prev.end = nxt.start
        if prev.end < prev.start:
            prev.end = prev.start
    return cues


def target_chars(text: str, duration: float, slot: float, max_tempo: float) -> int:
    """Сколько символов успеет прозвучать, с запасом 10%."""
    ratio = slot * max_tempo / max(duration, 1e-6)
    return max(8, int(len(text) * ratio * 0.9))


def _rest_chars(phrases: list[Phrase], i: int) -> int:
    """Сколько символов осталось до конца предложения того же спикера, начиная с фразы i."""
    who = phrases[i].speaker or ""
    total = 0
    for ph in phrases[i:]:
        if (ph.speaker or "") != who:
            break
        total += len(ph.text)
        if not continues_sentence(ph.text):
            break
    return total


def mid_sentence_tail(said: str) -> str:
    """Кусок кончается посреди предложения: в синтез без точки, на конце запятая."""
    s = (said or "").rstrip()
    while s and s[-1] in ".:;—–-…,":
        s = s[:-1].rstrip()
    return s + "," if s else s


def voice_blocks(phrases: list[Phrase], long_pause: float = BLOCK_PAUSE,
                 max_span: float = BLOCK_SPAN, max_chars: int = BLOCK_CHARS) -> list[list[int]]:
    """Индексы фраз одного куска озвучки.

    Кусок тянется, пока тот же спикер не замолкает надолго. Смена спикера
    начинает новый кусок, даже если паузы нет: у каждого свой голос.
    На конце предложения кусок также кончается, если он уже длинный.
    Середину предложения не режем, пока не упрёмся в жёсткий потолок.
    """
    blocks: list[list[int]] = []
    cur: list[int] = []

    def chars_of(idxs: list[int], extra: str = "") -> int:
        total = sum(len(phrases[i].text) for i in idxs)
        return total + len(extra)

    for i, ph in enumerate(phrases):
        if not cur:
            cur = [i]
            continue
        prev = phrases[cur[-1]]
        pause = ph.origin - max(prev.cue_end, prev.origin)
        span = ph.origin - phrases[cur[0]].origin
        sentence_done = not continues_sentence(prev.text)
        grown = chars_of(cur, ph.text)
        other = (ph.speaker or "") != (phrases[cur[0]].speaker or "")
        # Русский короче английского — голос кончает раньше и оставляет дыру;
        # длиннее — копит отставание внутри блока. В обоих случаях новый блок встаёт
        # на своё место, а предыдущий ужимается под своё окно.
        ahead = span - chars_of(cur) / VOICE_CPS
        if other:
            blocks.append(cur)
            cur = [i]
            continue
        want = (pause >= long_pause or abs(ahead) > DRIFT_SEC
                or span >= max_span or grown > max_chars
                or span >= BLOCK_HARD_SPAN or grown > BLOCK_HARD_CHARS)
        # Режем только на конце предложения. Посреди — лишь на запасной границе;
        # её нет — пусть отстаёт или чуть ускорится, но фраза звучит целиком.
        # Каждое предложение — свой кусок со своего места в оригинале: русский не убегает
        # вперёд и не копит отставание. Посреди предложения — только запасная граница.
        if sentence_done or (want and (
                soft_cut_ok(prev.text, chars_of(cur), _rest_chars(phrases, i))
                or span >= LAST_RESORT_SPAN or grown > LAST_RESORT_CHARS)):
            blocks.append(cur)
            cur = [i]
            continue
        cur.append(i)
    if cur:
        blocks.append(cur)
    return blocks


def shrink_words(text: str, max_chars: int) -> str:
    """Короче целыми словами. Одно слово, которое само длиннее лимита, остаётся."""
    words = [w for w in (text or "").split() if w]
    max_chars = max(int(max_chars), 1)
    if not words or len(" ".join(words)) <= max_chars:
        return " ".join(words)
    out: list[str] = []
    for word in words:
        cand = " ".join(out + [word])
        if out and len(cand) > max_chars:
            break
        out.append(word)
    if not out:
        out = [words[0]]
    result = " ".join(out)
    end = (text or "").rstrip()
    if end.endswith("?") and not result.endswith("?"):
        result += "?"
    elif end.endswith("!") and not result.endswith("!"):
        result += "!"
    elif end.endswith(".") and not result.endswith("."):
        result += "."
    return result


COMPRESS_SYSTEM = (
    "Ты ужимаешь реплику для закадровой озвучки. Уложись не больше чем в {chars} символов "
    "и около {syll} слогов. Сохрани смысл, имена, числа и вопрос, если он был. "
    "Убери воду, повторы и вводные. Не добавляй фактов. "
    "Верни ТОЛЬКО JSON: {{\"text\": \"короче\"}}."
)


def compress_voice_text(llm, text: str, max_chars: int, log: Log) -> str:
    """Короче для голоса. Субтитры этой строкой не подменяются.

    Нет модели или ответ не короче — режем сами, с конца, целыми словами.
    """
    src = " ".join((text or "").split())
    max_chars = max(int(max_chars), 8)
    if len(src) <= max_chars:
        return src
    short = ""
    if llm is not None:
        syll = max(4, int(count_syllables(src) * max_chars / max(len(src), 1)))
        system = COMPRESS_SYSTEM.format(chars=max_chars, syll=syll)
        set_stage(llm, "voice")
        try:
            data = chat_json(llm, system, f"Реплика: {src}\nНе длиннее {max_chars} символов.",
                             temperature=0.2)
        except Exception as e:
            log.info(f"озвучка: DeepSeek не сжал реплику ({str(e)[:120]})")
            data = None
        if isinstance(data, dict):
            short = clean_text(str(data.get("text") or ""))
    if not short or len(short) >= len(src):
        return shrink_words(src, max_chars)
    if len(short) > max_chars:
        short = shrink_words(short, max_chars)
    return short


def shorten_phrase(llm, text: str, max_chars: int, log: Log) -> str:
    """Короткая версия фразы от DeepSeek или "" (не вышло — дальше режем по времени)."""
    set_stage(llm, "voice")
    try:
        data = chat_json(llm, SHORTEN_SYSTEM,
                         f"Фраза: {text}\nНе длиннее {max_chars} символов.", temperature=0.3)
    except Exception as e:
        log.info(f"озвучка: DeepSeek не сократил фразу ({str(e)[:120]})")
        return ""
    short = clean_text(str((data or {}).get("text", ""))) if isinstance(data, dict) else ""
    if not short or len(short) >= len(text):
        return ""
    return short


# ------------------------------------------------------------------ звук

def trim_silence_span(samples: np.ndarray, rate: int, threshold: float = 0.01,
                      margin: float = 0.03) -> tuple[np.ndarray, float]:
    """Обрезать тишину по краям. Второй результат — секунды, снятые с начала."""
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples[:0], 0.0
    pad = int(rate * margin) if rate > 0 else 0
    start = max(int(loud[0]) - pad, 0)
    end = min(int(loud[-1]) + pad + 1, int(samples.size))
    shift = start / float(rate) if rate else 0.0
    return samples[start:end], shift


def trim_silence(samples: np.ndarray, rate: int, threshold: float = 0.01,
                 margin: float = 0.03) -> np.ndarray:
    audio, _shift = trim_silence_span(samples, rate, threshold, margin)
    return audio


def trim_tail_only(samples: np.ndarray, rate: int, threshold: float = 0.01,
                   margin: float = 0.05) -> np.ndarray:
    """Снять длинную тишину в конце. Начало не трогать.

    Короткое первое слово («и», «я», «до», «по») у Edge часто тише порога 0.01.
    Обрезка тишины с головы съедала его целиком. Хвост тишины на смысл не влияет.
    """
    audio = np.asarray(samples, np.float32).reshape(-1)
    if audio.size == 0 or rate <= 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    pad = int(rate * margin)
    end = min(int(loud[-1]) + pad + 1, int(audio.size))
    return audio[:end]


def shift_marks(marks: list[WordMark], shift: float, duration: float) -> list[WordMark]:
    """Сдвинуть границы после обрезки тишины в начале. За пределами звука — выбросить."""
    out: list[WordMark] = []
    for mark in marks:
        start = mark.start - shift
        end = mark.end - shift
        if end <= 0 or start >= duration:
            continue
        start = max(0.0, start)
        end = min(duration, max(end, start))
        out.append(WordMark(mark.text, start, end))
    return out


def write_mark_file(path: Path, marks: list[WordMark]) -> None:
    write_json(path, {"words": [{"text": m.text, "start": round(m.start, 4),
                                 "end": round(m.end, 4)} for m in marks]})


def read_mark_file(path: Path) -> list[WordMark]:
    data = read_json(path, {}) or {}
    if not isinstance(data, dict):
        return []
    return _coerce_marks(data.get("words") or [])


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise KonveyerError(f"{path.name}: ожидался 16-битный wav")
    data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def atempo_chain(tempo: float) -> str:
    """atempo старых ffmpeg берёт только 0.5–2.0, поэтому длинное ускорение — цепочкой."""
    parts, t = [], tempo
    while t > 2.0:
        parts.append("atempo=2.0")
        t /= 2.0
    parts.append(f"atempo={t:.4f}")
    return ",".join(parts)


def to_mix(path: Path, tempo: float, workdir: Path) -> np.ndarray:
    """wav → моно float32 48 кГц, с ускорением через ffmpeg atempo."""
    workdir.mkdir(parents=True, exist_ok=True)
    raw = workdir / (path.stem + ".mix.f32")
    af = (atempo_chain(tempo) + ",") if tempo > 1.0001 else ""
    run([need_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-i", str(path), "-af", f"{af}aresample={MIX_RATE}", "-ac", "1", "-ar", str(MIX_RATE),
         "-f", "f32le", str(raw)], "ffmpeg (озвучка)", timeout=600)
    try:
        return np.fromfile(raw, dtype="<f4")
    finally:
        raw.unlink(missing_ok=True)


def decode_to_f32(path: Path, workdir: Path) -> np.ndarray:
    """mp3/wav → моно float32 48 кГц. Edge отдаёт mp3 24 кГц, дальше все движки одинаковые."""
    raw = workdir / (path.stem + ".f32")
    run([need_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-i", str(path), "-ac", "1", "-ar", str(MIX_RATE), "-f", "f32le", str(raw)],
        "ffmpeg (edge)", timeout=120)
    try:
        data = np.fromfile(raw, dtype="<f4")
    finally:
        raw.unlink(missing_ok=True)
    return data


def duck_expr(intervals: list[tuple[float, float]], duck_db: float, fade: float = 0.15) -> str:
    """Выражение громкости для volume=…:eval=frame: 1 вне речи, 10^(dB/20) под речью."""
    if not intervals or duck_db >= 0:
        return ""
    k = 10 ** (duck_db / 20)
    fade = max(fade, 0.01)
    merged: list[list[float]] = []
    for a, b in sorted(intervals):
        if merged and a - merged[-1][1] <= 2 * fade:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    envs = [f"clip(min((t-{a - fade:.3f})/{fade:g},({b + fade:.3f}-t)/{fade:g}),0,1)"
            for a, b in merged]
    env = envs[0]
    for e in envs[1:]:
        env = f"max({env},{e})"
    return f"1-{1 - k:.4f}*{env}"


# ------------------------------------------------------------------ движки

class SileroEngine:
    """Silero v4_ru из локального файла (models.silero.ai может быть недоступен)."""
    name = "silero"

    def __init__(self, cfg: VoiceCfg, log: Log, loader=None):
        self.cfg, self.log = cfg, log
        self._loader = loader
        self._model = None
        self.rate = cfg.sample_rate

    @property
    def tag(self) -> str:
        return f"silero:{self.cfg.speaker}:{self.rate}"

    def _load(self):
        if self._model is not None:
            return self._model
        if self._loader is not None:
            self._model = self._loader(self.cfg)
            return self._model
        path = Path(self.cfg.model_path)
        if not path.is_file():
            raise KonveyerError(f"нет модели Silero: {path}. Укажи путь в [voice] model_path "
                                "(с models.silero.ai не качаю)")
        try:
            import torch
            import torch.package
        except Exception as e:
            raise KonveyerError(f"для Silero нужен torch ({e})") from e
        model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(torch.device(device))
        self.log.info(f"Silero v4_ru загружен ({device}), голос {self.cfg.speaker}")
        self._model = model
        return model

    def synth(self, texts: list[str]) -> list[np.ndarray | None]:
        model = self._load()
        out: list[np.ndarray | None] = []
        for text in texts:
            try:
                audio = model.apply_tts(ssml_text=build_ssml(text), speaker=self.cfg.speaker,
                                        sample_rate=self.rate, put_accent=True, put_yo=True)
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                out.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            except Exception as e:
                self.log.warn(f"Silero не озвучил «{text[:40]}»: {str(e)[:120]}")
                out.append(None)
        return out


class XTTSEngine:
    """XTTS v2 (клонирование голоса) в отдельном python — подпроцессом."""
    name = "xtts"

    def __init__(self, cfg: VoiceCfg, log: Log, workdir: Path):
        self.cfg, self.log, self.workdir = cfg, log, workdir
        sample = Path(cfg.sample) if cfg.sample else None
        if not sample or not sample.is_file():
            raise KonveyerError("для XTTS нужен образец голоса: --voice-sample путь.wav "
                                "(чистая запись 10–30 с)")
        self.sample = sample.resolve()
        self.python = cfg.xtts_python or sys.executable
        if cfg.xtts_python and not Path(cfg.xtts_python).is_file():
            raise KonveyerError(f"нет python для XTTS: {cfg.xtts_python} ([voice] xtts_python)")
        self.rate = 24000

    @property
    def tag(self) -> str:
        digest = hashlib.sha1(self.sample.read_bytes()).hexdigest()[:12]
        return f"xtts:{digest}"

    def synth(self, texts: list[str]) -> list[np.ndarray | None]:
        self.workdir.mkdir(parents=True, exist_ok=True)
        items = [{"text": t, "out": str((self.workdir / f"xtts_{i:03d}.wav").resolve())}
                 for i, t in enumerate(texts)]
        req = self.workdir / "xtts_request.json"
        write_json(req, {"model": self.cfg.xtts_model, "sample": str(self.sample),
                         "language": "ru", "items": items})
        try:
            proc = subprocess.run([self.python, str(XTTS_WORKER), str(req)], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=self.cfg.xtts_timeout, env=dict(os.environ))
        except (OSError, subprocess.TimeoutExpired) as e:
            raise KonveyerError(f"XTTS не запустился: {e}") from e
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-600:]
            raise KonveyerError("XTTS упал. Проверь `py -m konveyer doctor` и [voice] xtts_python. "
                                f"Лицензию модели принимают так: setx COQUI_TOS_AGREED 1\n{tail}")
        out: list[np.ndarray | None] = []
        for it in items:
            p = Path(it["out"])
            if not p.is_file():
                out.append(None)
                continue
            out.append(to_mix(p, 1.0, self.workdir))
            p.unlink(missing_ok=True)
        self.rate = MIX_RATE
        return out


_EDGE_PCT = re.compile(r"^[+-]\d+%$")
_EDGE_HZ = re.compile(r"^[+-]\d+Hz$")


def _round_half_away(value: float) -> int:
    """Ближайшее целое, половинка — от нуля. Так +29.6% становится +30%, а не +29%."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def edge_service_rate(rate: str) -> str:
    """Процент для edge-tts: только целый, шаблон ^[+-]\\d+%$.

    Дробь вроде +14.1% сервис отвергает. Округление не зажимает значение вниз
    (+29.6% становится +30%) и не стирает интонацию: -5% остаётся -5%.
    """
    text = (rate or "").strip()
    if _EDGE_PCT.fullmatch(text):
        return text
    return f"{_round_half_away(parse_percent(text)):+d}%"


def edge_service_hz(pitch: str) -> str:
    """Герцы для edge-tts: только целые, шаблон ^[+-]\\d+Hz$."""
    text = (pitch or "").strip()
    if _EDGE_HZ.fullmatch(text):
        return text
    return f"{_round_half_away(parse_hz(text)):+d}Hz"


def edge_retry_pause(attempt: int) -> float:
    """Секунды перед следующей попыткой Edge. Растёт и не длиннее EDGE_RETRY_CAP.

    attempt — номер уже провалившейся попытки, с нуля. Короткий обрыв сети
    переживается ещё несколькими запросами, а не сменой голоса.
    """
    cap = min(EDGE_RETRY_CAP, 1.5 * (1.5 ** attempt))
    low = max(1.0, cap * 0.75)
    return random.uniform(low, cap)


def _edge_fail_reason(exc: BaseException) -> str:
    name = type(exc).__name__
    if name.endswith("NoAudioReceived"):
        name = "NoAudioReceived"
    detail = " ".join(str(exc).split())
    if not detail or detail == name:
        return name
    return f"{name}: {detail[:80]}"


def speech_table(cfg: VoiceCfg) -> dict:
    """Пользовательские замены для голоса: [voice] replacements и [voice.pronounce]."""
    return {**(cfg.replacements or {}), **(getattr(cfg, "pronounce", None) or {})}


def edge_fallback_voices(cfg: VoiceCfg, voice: str) -> list[str]:
    """Запасные голоса Edge того же пола, если основной так и не ответил."""
    from konveyer.config import _FEMALE_VOICES, _MALE_VOICES
    short = next((k for k, v in EDGE_VOICES.items() if v == voice), "")
    male = list(dict.fromkeys([*(cfg.male_voices or []), *_MALE_VOICES]))
    female = list(dict.fromkeys([*(cfg.female_voices or []), *_FEMALE_VOICES]))
    pool = female if short in female else male if short in male else list(EDGE_VOICES)
    spare = (EDGE_SPARE_FEMALE if short in female else EDGE_SPARE_MALE if short in male
             else EDGE_SPARE_MALE + EDGE_SPARE_FEMALE)
    names = [*spare, *(EDGE_VOICES[n] for n in pool if n in EDGE_VOICES)]
    return [v for v in dict.fromkeys(names) if v != voice]


def edge_pieces(text: str, max_words: int = EDGE_PIECE_WORDS) -> list[str]:
    """Длинная фраза кусками: по знакам препинания, потом не длиннее max_words слов."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…:;,—])\s+", text or "") if p.strip()]
    out: list[str] = []
    for part in parts:
        words = part.split()
        while len(words) > max_words:
            out.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if words:
            out.append(" ".join(words))
    merged: list[str] = []
    for piece in out:
        if merged and len(piece.split()) < 3:
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    return merged


def call_with_timeout(fn, timeout: float):
    """Вызвать fn в отдельном потоке и ждать не дольше timeout секунд.

    Зависший вызов остаётся в фоне (поток-демон), а мы получаем TimeoutError
    и идём дальше — рендер не висит на одном сокете."""
    if not timeout or timeout <= 0:
        return fn()
    box: dict = {}

    def work():
        try:
            box["value"] = fn()
        except BaseException as e:      # noqa: BLE001 — отдадим вызывающему как есть
            box["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"Edge не ответил за {timeout:.0f} с")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def eta_text(done: int, total: int, elapsed: float) -> str:
    """«, осталось ~M мин» по скорости уже сделанного; пусто, пока мерить рано."""
    if done <= 0 or total <= 0 or done >= total or elapsed < 1:
        return ""
    left = elapsed / done * (total - done)
    return f", осталось ~{max(1, math.ceil(left / 60))} мин"


class _NoAudioReceived(Exception):
    """Пустой файл от Edge. Имя заканчивается на NoAudioReceived — в журнале так и пишем."""


def _coerce_marks(raw) -> list[WordMark]:
    """Границы из теста: WordMark или dict со start/end в секундах."""
    out: list[WordMark] = []
    for item in raw or []:
        if isinstance(item, WordMark):
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        try:
            start = float(item.get("start", item.get("offset", 0)) or 0)
        except (TypeError, ValueError):
            start = 0.0
        if item.get("end") is not None:
            try:
                end = float(item["end"])
            except (TypeError, ValueError):
                end = start
        else:
            try:
                end = start + float(item.get("duration") or 0)
            except (TypeError, ValueError):
                end = start
        if start > 1000:
            start /= EDGE_TICKS_PER_SEC
            end /= EDGE_TICKS_PER_SEC
        out.append(WordMark(text, start, end))
    return out


def _unpack_synth(result) -> tuple[np.ndarray | None, list[WordMark]]:
    """Ответ синтеза: массив или (массив, границы слов)."""
    marks_raw = None
    audio = result
    if isinstance(result, tuple) and len(result) == 2:
        audio, marks_raw = result
    if audio is None:
        return None, []
    arr = np.asarray(audio, np.float32).reshape(-1)
    if arr.size == 0:
        return None, []
    return arr, _coerce_marks(marks_raw)


class EdgeEngine:
    """Edge TTS (edge-tts): mp3 по сети, на диск кладём уже моно float32 48 кГц.

    Голос многоязычный читает русский текст как есть (после saytext). Если сервис
    молчит (NoAudioReceived или любая другая ошибка, часто из-за VPN) — ещё попытки
    именно Edge. Звука так и нет — фраза без озвучки, одна строка в журнале.
    На Silero эта фраза не переключается. Остальные фразы не отменяются.
    """
    name = "edge"
    rate_fit = True          # ускорение своим rate, atempo только если rate не попал в окно

    def __init__(self, cfg: VoiceCfg, log: Log, workdir: Path, synth_fn=None, sleeper=None,
                 clock=None):
        err = speaker_error("edge", cfg.speaker)
        if err:
            raise KonveyerError(err)
        self.cfg, self.log, self.workdir = cfg, log, workdir
        self.voice = EDGE_VOICES[cfg.speaker]
        self.rate = MIX_RATE
        self._synth_fn = synth_fn
        self._sleep = sleeper or time.sleep
        self._clock = clock or time.monotonic
        self._tl = threading.local()   # fell_back свой у каждого потока озвучки
        self._lock = threading.Lock()
        self.timeout = float(getattr(cfg, "edge_timeout", EDGE_TIMEOUT_SEC) or 0)
        self.breaker = int(getattr(cfg, "edge_breaker", EDGE_BREAKER) or 0)
        self.rest_sec = float(getattr(cfg, "edge_rest_sec", EDGE_REST_SEC) or 0)
        self._fails = 0
        self.tripped = False           # выключатель сработал: без долгих ожиданий, пока Edge не ответит
        self._rest_until = 0.0
        budget = float(getattr(cfg, "edge_budget_min", 0) or 0)
        self.deadline = self._clock() + budget * 60 if budget > 0 else None
        self.budget_min = budget
        self._budget_said = False
        self.fell_back = ""   # каким запасным голосом озвучен последний запрос
        # False — запасной голос не брать: кусок откладывается и переспрашивается в конце прохода
        self.allow_fallback = True

    @property
    def fell_back(self) -> str:
        return getattr(self._tl, "fell_back", "")

    @fell_back.setter
    def fell_back(self, value: str) -> None:
        self._tl.fell_back = value

    @property
    def quick(self) -> bool:
        """Предзапрос в потоке: пара попыток и всё — без терпеливых ожиданий и запасных."""
        return getattr(self._tl, "quick", False)

    @quick.setter
    def quick(self, value: bool) -> None:
        self._tl.quick = bool(value)

    # ---------- выключатель, отдых и общий лимит времени
    def over_budget(self) -> bool:
        if self.deadline is None or self._clock() < self.deadline:
            return False
        with self._lock:
            if not self._budget_said:
                self._budget_said = True
                self.log.warn(f"Edge: лимит времени озвучки ({self.budget_min:g} мин) исчерпан — "
                              f"дальше без ожиданий: не ответил основной голос — запасной")
        return True

    def _rest(self) -> None:
        with self._lock:
            wait = self._rest_until - self._clock()
        if wait > 0:
            self._sleep(wait)

    def _note_ok(self) -> None:
        with self._lock:
            self._fails = 0
            self.tripped = False

    def _note_fail(self, reason: str) -> None:
        with self._lock:
            self._fails += 1
            if not self.breaker or self._fails < self.breaker:
                return
            self._fails = 0
            self.tripped = True
            self._rest_until = self._clock() + self.rest_sec
        self.log.warn(f"Edge: {self.breaker} неудач подряд ({reason}) — отдых {self.rest_sec:.0f} с "
                      f"для всех потоков; без долгих ожиданий: фразы дробятся или откладываются")

    @property
    def tag(self) -> str:
        return (f"edge:{self.voice}:{self.cfg.edge_rate}:{self.cfg.edge_pitch}:"
                f"{expression_fingerprint(self.cfg)}")

    def _segments_of(self, text: str) -> list[tuple[str, int]]:
        expr = self.cfg.expression or {}
        return edge_segments(text, int(expr.get("pause_ms", 300)), int(expr.get("ellipsis_ms", 500)))

    def _ensure_package(self) -> None:
        if self._synth_fn is not None:
            return
        if importlib.util.find_spec("edge_tts") is None:
            raise KonveyerError("для Edge нужен пакет edge-tts: py -m pip install edge-tts")

    def synth(self, texts: list[str], prosody: list[Prosody] | None = None) -> list[np.ndarray | None]:
        return self.synth_with_voices(texts, prosody, None)

    def synth_with_voices(self, texts: list[str], prosody: list[Prosody] | None,
                          voices: list[str] | None) -> list[np.ndarray | None]:
        self._ensure_package()
        out: list[np.ndarray | None] = []
        for i, text in enumerate(texts):
            pros = prosody[i] if prosody else expression_prosody(text, self.cfg, 1.0)
            voice = voices[i] if voices and i < len(voices) and voices[i] else self.voice
            try:
                out.append(self._one(text, pros, voice))
            except Exception as e:
                self.log.warn(f"Edge не озвучил «{text[:40]}»: {str(e)[:120]}")
                out.append(None)
        return out

    def _one(self, text: str, pros: Prosody, voice: str | None = None) -> np.ndarray | None:
        parts = self._segments_of(text)
        if not parts:
            return None
        chunks: list[np.ndarray] = []
        for sentence, pause_ms in parts:
            audio = self._speak_sentence(sentence, pros, voice)
            if audio is None:
                continue
            chunks.append(np.asarray(audio, np.float32).reshape(-1))
            if pause_ms > 0:
                silence = int(self.rate * int(pause_ms) / 1000)
                if silence > 0:
                    chunks.append(np.zeros(silence, np.float32))
        if not chunks:
            return None
        return np.concatenate(chunks)

    def synth_sentence(self, text: str, pros: Prosody, voice: str = ""
                       ) -> tuple[np.ndarray | None, list[WordMark]]:
        """Одно предложение — один запрос Edge, вместе с границами слов.

        Текст не режется на куски: интонация целая. Границ нет — вызывающий
        разложит звук пропорционально длине строк.
        """
        self._ensure_package()
        return self._speak_sentence_marked(edge_plain(text), pros, voice or self.voice)

    def _speak_sentence(self, sentence: str, pros: Prosody, voice: str | None = None
                        ) -> np.ndarray | None:
        audio, _marks = self._speak_sentence_marked(sentence, pros, voice)
        return audio

    def _speak_sentence_marked(self, sentence: str, pros: Prosody, voice: str | None = None
                               ) -> tuple[np.ndarray | None, list[WordMark]]:
        plain = edge_plain(sentence)
        self.fell_back = ""
        if not plain:
            return None, []
        main = voice or self.voice
        # 1) основной голос целиком; 2) им же по кускам; 3) запасные голоса того же пола.
        # Фразу без звука не оставляем, пока хоть какой-то голос отвечает.
        late = self.over_budget()
        quick = self.quick
        tries = 1 if late else EDGE_FALLBACK_ATTEMPTS if quick else EDGE_ATTEMPTS
        audio, marks, reason = self._try_voice(plain, pros, main, tries)
        if audio is not None:
            return audio, marks
        if quick:
            return None, []          # предзапрос: дальше разберётся раскладка по порядку
        # тот же голос, но терпеливо: длинные паузы, всего до EDGE_PATIENT_SEC.
        # Не ждём, если сработал выключатель (Edge лёг для всех) или вышел лимит времени.
        waited = 0.0
        while waited < EDGE_PATIENT_SEC and not self.tripped and not self.over_budget():
            pause = min(EDGE_PATIENT_CAP, EDGE_PATIENT_SEC - waited)
            self._sleep(pause)
            waited += pause
            audio, marks, why = self._try_voice(plain, pros, main, 1)
            if audio is not None:
                self.log.warn(f"Edge: {main} ответил после паузы {waited:.0f} с: «{plain[:40]}»")
                return audio, marks
            reason = why or reason
        pieces = edge_pieces(plain)
        if len(pieces) > 1 and not self.over_budget():
            audio, marks, why = self._speak_pieces(pieces, pros, main)
            if audio is not None:
                self.log.warn(f"Edge: {main} не отдал фразу целиком ({reason}), "
                              f"озвучено им же по {len(pieces)} кускам: «{plain[:40]}»")
                return audio, marks
            reason = why or reason
        if not self.allow_fallback:
            self.log.warn(f"Edge: {main} не ответил ({reason}), кусок отложен до конца прохода: "
                          f"«{plain[:40]}»")
            return None, []
        for alt in edge_fallback_voices(self.cfg, main):
            audio, marks, why = self._try_voice(plain, pros, alt,
                                                1 if self.over_budget() else EDGE_FALLBACK_ATTEMPTS)
            if audio is not None:
                self.fell_back = alt
                self.log.warn(f"Edge: {main} не ответил ({reason}), "
                              f"«{plain[:40]}» озвучено запасным {alt}")
                return audio, marks
            reason = why or reason
        self.log.warn(f"Edge не озвучил «{plain[:40]}» ({reason}), фраза без озвучки")
        return None, []

    def _try_voice(self, plain: str, pros: Prosody, who: str, tries: int):
        reason = "пустой звук"
        for attempt in range(tries):
            self._rest()
            try:
                audio, marks = self._synthesize(plain, pros, who)
            except Exception as e:
                reason = _edge_fail_reason(e)
            else:
                if audio is not None and len(audio):
                    self._note_ok()
                    return np.asarray(audio, np.float32).reshape(-1), marks, ""
                reason = "пустой звук"
            self._note_fail(reason)
            if self.tripped or (tries > 1 and self.over_budget()):
                break                    # Edge лёг или время вышло — дальше дробить/откладывать
            if attempt + 1 < tries:
                self._sleep(edge_retry_pause(attempt))
        return None, [], reason

    def _speak_pieces(self, pieces: list[str], pros: Prosody, who: str):
        chunks: list[np.ndarray] = []
        marks: list[WordMark] = []
        at = 0.0
        for piece in pieces:
            audio, got, why = self._try_voice(piece, pros, who, EDGE_FALLBACK_ATTEMPTS)
            if audio is None:
                return None, [], why
            marks.extend(WordMark(m.text, m.start + at, m.end + at) for m in got)
            chunks.append(audio)
            at += audio.size / float(self.rate)
        return np.concatenate(chunks), marks, ""

    def _synthesize(self, markup: str, pros: Prosody, voice: str | None = None
                    ) -> tuple[np.ndarray | None, list[WordMark]]:
        if not edge_text_ok(markup):
            raise KonveyerError("в Edge ушёл текст с разметкой")
        rate = edge_service_rate(pros.rate)
        pitch = edge_service_hz(pros.pitch)
        volume = edge_service_rate(pros.volume)
        chosen = voice or self.voice
        if self._synth_fn is not None:
            fn = self._synth_fn
            return _unpack_synth(call_with_timeout(
                lambda: fn(markup, chosen, rate, pitch, volume), self.timeout))
        return self._from_service(markup, rate, pitch, volume, chosen)

    def _from_service(self, markup: str, rate: str, pitch: str, volume: str, voice: str | None = None
                      ) -> tuple[np.ndarray, list[WordMark]]:
        import edge_tts

        self.workdir.mkdir(parents=True, exist_ok=True)
        dest = self.workdir / f"edge_{uuid.uuid4().hex}.mp3"

        async def _save() -> list[WordMark]:
            # boundary=WordBoundary: без него сервис отдаёт только границы предложений.
            comm = edge_tts.Communicate(markup, voice or self.voice, rate=rate, pitch=pitch,
                                        volume=volume, boundary="WordBoundary")
            audio = bytearray()
            events: list[dict] = []
            async for message in comm.stream():
                kind = message.get("type")
                if kind == "audio":
                    audio.extend(message.get("data") or b"")
                elif kind == "WordBoundary":
                    events.append(message)
            if not audio:
                raise _NoAudioReceived("пустой файл")
            dest.write_bytes(bytes(audio))
            return marks_from_edge_events(events)

        try:
            if self.timeout > 0:
                async def _limited() -> list[WordMark]:
                    try:
                        return await asyncio.wait_for(_save(), self.timeout)
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"Edge не ответил за {self.timeout:.0f} с") from None
                marks = asyncio.run(_limited())
            else:
                marks = asyncio.run(_save())
            if not dest.is_file() or dest.stat().st_size == 0:
                raise _NoAudioReceived("пустой файл")
            return decode_to_f32(dest, self.workdir), marks
        finally:
            dest.unlink(missing_ok=True)


def make_engine(settings: Settings, log: Log, workdir: Path):
    err = speaker_error(settings.voice.engine, settings.voice.speaker)
    if err:
        raise KonveyerError(err)
    if settings.voice.engine == "xtts":
        return XTTSEngine(settings.voice, log, workdir)
    if settings.voice.engine == "edge":
        return EdgeEngine(settings.voice, log, workdir)
    return SileroEngine(settings.voice, log)


# ------------------------------------------------------------------ demucs

def demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def separate_music(src: Path, start: float, dur: float, folder: Path, cfg: VoiceCfg,
                   log: Log) -> Path | None:
    """Звук клипа без голоса (музыка и шумы) или None — тогда остаётся приглушённый оригинал."""
    if not demucs_available():
        log.warn(DEMUCS_MISSING)
        return None
    music = folder / "music.wav"
    stamp_file = folder / "music.json"
    stamp = {"start": round(start, 3), "dur": round(dur, 3), "model": cfg.demucs_model}
    if music.is_file() and read_json(stamp_file, {}) == stamp:
        return music
    work = folder / "demucs"
    work.mkdir(parents=True, exist_ok=True)
    seg = work / "segment.wav"
    try:
        run([need_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src.resolve()), "-vn",
             "-ac", "2", "-ar", "44100", str(seg)], "ffmpeg (звук для demucs)", timeout=600)
        run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", cfg.demucs_model,
             "-o", str(work), str(seg)], "demucs", timeout=3600)
    except KonveyerError as e:
        log.warn(f"demucs не отделил голос ({str(e)[:200]}) — оставляю приглушённый оригинал")
        return None
    found = work / cfg.demucs_model / seg.stem / "no_vocals.wav"
    if not found.is_file():
        log.warn("demucs не создал no_vocals.wav — оставляю приглушённый оригинал")
        return None
    os.replace(found, music)
    write_json(stamp_file, stamp)
    log.info("demucs: голос убран, остались музыка и шумы")
    return music


# ------------------------------------------------------------------ озвучка клипа

@dataclass
class VoiceTrack:
    path: Path
    intervals: list[tuple[float, float]]
    note: str
    phrases: list[Phrase]
    end: float = 0.0          # конец русской дорожки, секунды клипа


class VoiceOver:
    """Одна на задание: движок грузится один раз, озвученные фразы кешируются в voice_cache/."""

    def __init__(self, job_dir: Path, settings: Settings, log: Log, llm, engine=None, hear=None):
        self.dir, self.s, self.cfg, self.log, self.llm = job_dir, settings, settings.voice, log, llm
        self._engine = engine
        self._hear = hear
        self.broken = ""
        self.cache_dir = job_dir / "voice_cache"
        self._short_file = job_dir / "voice_short.json"
        self._short: dict[str, str] = read_json(self._short_file, {}) or {}

    def engine(self):
        if self._engine is None and not self.broken:
            try:
                self._engine = make_engine(self.s, self.log, self.cache_dir)
            except KonveyerError as e:
                self.broken = str(e)
                self.log.warn(f"озвучка выключена: {e}")
        return self._engine

    # ---------- синтез с кешем
    def _key(self, engine, text: str, extra: str = "") -> str:
        return cache_key(engine.tag, text, extra)

    def _who(self, engine, phrases: list[Phrase]) -> str:
        if engine.name not in ("silero", "edge"):
            return "XTTS"
        names = [self.cfg.speaker]
        for ph in phrases:
            if not ph.voice:
                continue
            short = next((k for k, v in EDGE_VOICES.items() if v == ph.voice), "")
            if short and short not in names:
                names.append(short)
        return "+".join(names) if len(names) > 1 else names[0]

    def _assign_voices(self, phrases: list[Phrase]) -> None:
        from konveyer.speakers import edge_voice_for, load_cast
        cast = load_cast(self.dir)
        # Реплика без метки берёт спикера соседа, а не голос по умолчанию.
        last = ""
        for ph in phrases:
            if ph.speaker:
                last = ph.speaker
            elif last:
                ph.speaker = last
        last = ""
        for ph in reversed(phrases):
            if ph.speaker:
                last = ph.speaker
            elif last:
                ph.speaker = last
        for ph in phrases:
            ph.voice = edge_voice_for(self.s, ph.speaker, cast)

    def _synth(self, engine, texts: list[str],
               prosodies: list[Prosody] | None = None,
               voices: list[str] | None = None) -> list[Path | None]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        voices = list(voices or [])
        paths: list[Path | None] = []
        for i, text in enumerate(texts):
            extra = prosodies[i].key() if prosodies else ""
            voice = voices[i] if i < len(voices) else ""
            if voice:
                extra = f"{extra}|voice={voice}" if extra else f"voice={voice}"
            paths.append(self.cache_dir / f"{self._key(engine, text, extra)}.wav")
        todo = [i for i, p in enumerate(paths) if p is None or not p.is_file()]
        if todo:
            chunk = [texts[i] for i in todo]
            pros = [prosodies[i] for i in todo] if prosodies else None
            voice_chunk = [voices[i] if i < len(voices) else "" for i in todo]
            if any(voice_chunk) and hasattr(engine, "synth_with_voices"):
                made = engine.synth_with_voices(chunk, pros, voice_chunk)
            elif pros:
                made = engine.synth(chunk, pros)
            else:
                made = engine.synth(chunk)
            for i, audio in zip(todo, made):
                if audio is None or not len(audio):
                    paths[i] = None
                    continue
                write_wav(paths[i], trim_tail_only(np.asarray(audio, np.float32), engine.rate),
                          engine.rate)
        return [p if p is not None and p.is_file() else None for p in paths]

    @staticmethod
    def _length(path: Path) -> float:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())

    def _speak(self, text: str) -> str:
        """Цифры, латиница и одиночные буквы — русские слова. Субтитры не меняются."""
        return prepare_for_speech(clean_text(text), speech_table(self.cfg))

    def _for_tts(self, text: str) -> str:
        """Строка в синтез. Запятые и тире выкидываем, точки и вопросы оставляем."""
        said = self._speak(text)
        if self.cfg.strip_commas:
            said = strip_speech_punct(said)
        return said

    def _shorten(self, text: str, max_chars: int) -> str:
        key = f"{max_chars}|{text}"
        if key not in self._short:
            self._short[key] = shorten_phrase(self.llm, text, max_chars, self.log)
            write_json(self._short_file, self._short)
        return self._short[key]

    def _boost_edge(self, engine, ph: Phrase, path: Path, needed: float) -> tuple[Path, str]:
        """Пересинтез Edge с rate до edge_max_rate. atempo — только если rate не попал в окно.

        Возвращает (файл, "ok"|"shorten"). "shorten" значит: даже после rate фраза длиннее,
        чем позволяет max_tempo. Ускорение уже внутри файла, ph.tempo тогда 1.0.
        """
        ratio = min(max(needed, 1.0), self.cfg.edge_max_rate)
        boosted = expression_prosody(ph.said, self.cfg, ratio)
        new = self._synth(engine, [ph.said], [boosted], [ph.voice])[0]
        if new is None:
            ph.status = "tempo"
            ph.tempo = round(min(needed, self.cfg.max_tempo), 3)
            return path, "ok"
        dur = self._length(new)
        ph.duration = dur
        ph.rate = boosted.rate
        ph.kind = boosted.kind
        if dur <= ph.slot + 1e-6:
            ph.status, ph.tempo = "tempo", 1.0
            return new, "ok"
        remain = dur / max(ph.slot, 1e-6)
        if remain <= self.cfg.max_tempo + 1e-6:
            ph.status, ph.tempo = "tempo", round(remain, 3)
            return new, "ok"
        ph.status, ph.tempo = "shorten", 1.0
        return new, "shorten"

    def _resynth_kept(self, engine, ph: Phrase, text: str, use_rate: bool):
        """Короче на целое слово. None — синтез не вышел, звук не трогаем."""
        said = self._speak(text)
        if use_rate:
            base = expression_prosody(ph.text or text, self.cfg, 1.0)
            pros = Prosody(ph.rate or base.rate, base.pitch, base.volume, ph.kind or base.kind)
            path = self._synth(engine, [said], [pros], [ph.voice])[0]
        else:
            path = self._synth(engine, [said], None, [ph.voice])[0]
        return path, said

    def _fit_words_audio(self, engine, ph: Phrase, samples: np.ndarray, limit_sec: float,
                         folder: Path, use_rate: bool) -> np.ndarray:
        """Укоротить звук пересинтезом целых слов. Буфер сэмплов не режется."""
        limit_n = int(max(limit_sec, 0.0) * MIX_RATE)
        slack = int(0.04 * MIX_RATE)
        guard = 0
        while samples.size > limit_n + slack and guard < 8:
            kept, dropped = fit_words(ph.spoken or ph.text, samples.size / MIX_RATE, limit_sec)
            if not dropped:
                break
            path, said = self._resynth_kept(engine, ph, kept, use_rate)
            if path is None:
                break
            new = to_mix(path, ph.tempo if ph.tempo > 0 else 1.0, folder)
            if new.size >= samples.size:
                break
            ph.spoken = kept
            ph.said = said
            ph.dropped = (ph.dropped + " " + dropped).strip()
            samples = new
            guard += 1
        return samples

    def _voiceover(self, folder: Path, phrases: list[Phrase], files: list[Path | None],
                   engine, total: float, clip_id) -> VoiceTrack | None:
        """Закадр: естественный темп, фразы подряд, без обрезки и без ускорения."""
        for ph, f in zip(phrases, files):
            if f is None:
                ph.status = "fail"
                ph.spoken = ""
                continue
            ph.spoken = ph.text
            ph.tempo = 1.0
            ph.dropped = ""
            ph.duration = self._length(f)
            ph.status = "ok"
        n = int(round(max(total, 0.0) * MIX_RATE))
        track = np.zeros(max(n, 1), dtype=np.float32)
        gain = 10 ** (self.cfg.gain_db / 20)
        cursor = 0.0
        for ph, f in zip(phrases, files):
            if f is None or ph.status == "fail":
                continue
            samples = to_mix(f, 1.0, folder)
            if samples.size == 0:
                ph.status = "fail"
                continue
            start = max(ph.origin, cursor, 0.0)
            ph.status = "lag" if start > ph.origin + 0.03 else "ok"
            ph.start = round(start, 3)
            a = int(round(ph.start * MIX_RATE))
            end_at = a + int(samples.size)
            if end_at > track.size:
                track = np.concatenate([track, np.zeros(end_at - track.size, np.float32)])
            track[a:end_at] += samples * gain
            ph.duration = round(samples.size / MIX_RATE, 3)
            cursor = ph.start + ph.duration + VOICEOVER_GAP
        return self._emit_voiceover(folder, track, phrases, engine, total, clip_id)

    def _cached_sentence(self, engine, text: str, pros: Prosody, voice: str = ""
                         ) -> tuple[Path | None, list[WordMark]]:
        """Один wav на всё предложение и границы слов рядом. Повторно Edge не спрашиваем."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        extra = pros.key() + (f"|voice={voice}" if voice else "")
        path = self.cache_dir / f"{self._key(engine, text, extra)}.wav"
        meta = path.with_suffix(".words.json")
        if path.is_file():
            return path, read_mark_file(meta) if meta.is_file() else []
        if voice:
            audio, marks = engine.synth_sentence(text, pros, voice)
        else:
            audio, marks = engine.synth_sentence(text, pros)
        if audio is None or len(audio) == 0:
            return None, []
        trimmed = trim_tail_only(np.asarray(audio, np.float32), engine.rate)
        if trimmed.size == 0:
            return None, []
        marks = shift_marks(marks, 0.0, trimmed.size / float(engine.rate))
        if getattr(engine, "fell_back", ""):
            # Запасной голос в кэш основного не кладём: в следующий раз снова спросим основной.
            path = path.with_name(path.stem + ".fallback.wav")
            meta = path.with_suffix(".words.json")
        write_wav(path, trimmed, engine.rate)
        write_mark_file(meta, marks)
        return path, marks

    def _prefetch(self, engine, blocks: list[list[int]], phrases: list[Phrase], base_rate: str,
                  max_rate: float, clip_id) -> None:
        """Первый вариант каждого блока — заранее, в несколько потоков, прямо в кэш.

        Раскладка по времени дальше идёт по порядку и берёт готовое из кэша; что не
        озвучилось здесь, она переспросит сама (и отложит или отдаст запасному голосу)."""
        workers = int(getattr(self.cfg, "edge_workers", 1) or 1)
        if workers <= 1 or engine.name != "edge" or not hasattr(engine, "synth_sentence"):
            return
        jobs = []
        for idxs in blocks:
            parts = [phrases[i] for i in idxs]
            text = " ".join(p.said for p in parts if p.said).strip()
            if not text:
                continue
            voice = parts[0].voice or ""
            pros = block_prosody(text, self.cfg, 1.0, base_rate, max_rate)
            extra = pros.key() + (f"|voice={voice}" if voice else "")
            if (self.cache_dir / f"{self._key(engine, text, extra)}.wav").is_file():
                continue
            jobs.append((text, pros, voice))
        if len(jobs) < 2:
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        total = len(jobs)
        self.log.info(f"клип {clip_id}: озвучка в {min(workers, total)} потока, "
                      f"запросов к Edge: {total} (готовых в кэше: {len(blocks) - total})")
        t0 = time.monotonic()
        step = max(1, total // 10)
        done = failed = 0
        with ThreadPoolExecutor(max_workers=min(workers, total)) as pool:
            futs = [pool.submit(self._prefetch_one, engine, t, p, v) for t, p, v in jobs]
            for fut in as_completed(futs):
                done += 1
                try:
                    path, _marks = fut.result()
                    failed += path is None
                except Exception as e:      # noqa: BLE001 — переспросит раскладка
                    failed += 1
                    self.log.warn(f"озвучка: запрос не удался ({str(e)[:100]})")
                if done % step == 0 or done == total:
                    self.log.info(f"озвучка: фразы {done} из {total} готовы"
                                  + eta_text(done, total, time.monotonic() - t0))
        if failed:
            self.log.info(f"озвучка: без ответа {failed} из {total} — переспрошу по порядку")

    def _prefetch_one(self, engine, text: str, pros: Prosody, voice: str):
        if hasattr(engine, "quick"):
            engine.quick = True
        try:
            return self._cached_sentence(engine, text, pros, voice)
        finally:
            if hasattr(engine, "quick"):
                engine.quick = False

    def _paste(self, track: np.ndarray, start_sample: int, audio: np.ndarray,
               gain: float) -> tuple[np.ndarray, int]:
        """Положить блок целиком. Чужой звук не суммируем и не затираем: место уже свободно."""
        chunk = np.asarray(audio, np.float32).reshape(-1)
        if chunk.size == 0:
            return track, start_sample
        end = start_sample + int(chunk.size)
        if end > track.size:
            track = np.concatenate([track, np.zeros(end - track.size, np.float32)])
        track[start_sample:end] = chunk * np.float32(gain)
        return track, end

    def _block_audio(self, engine, text: str, pros: Prosody, folder: Path, voice: str = ""
                     ) -> tuple[np.ndarray | None, list[WordMark]]:
        """Один запрос на весь блок, голосом этого спикера. Начало файла не обрезаем."""
        if not text:
            return None, []
        if engine.name == "edge" and hasattr(engine, "synth_sentence"):
            path, marks = self._cached_sentence(engine, text, pros, voice)
            if path is None:
                return None, []
            return to_mix(path, 1.0, folder), marks
        voices = [voice] if voice else None
        path = self._synth(engine, [text], [pros], voices)[0]
        if path is None:
            return None, []
        return to_mix(path, 1.0, folder), []

    def _respeed(self, engine, parts: list[Phrase], need: float, folder: Path,
                 base_rate: str, max_rate: float
                 ) -> tuple[np.ndarray | None, Prosody, list[WordMark]]:
        """Ускорить уже измеренный блок, не выше потолка. Слов не режем."""
        text = " ".join(p.said for p in parts if p.said).strip()
        voice = parts[0].voice if parts else ""
        need = max(float(need), 1.0)
        if getattr(engine, "rate_fit", False):
            pros = block_prosody(text, self.cfg, need, base_rate, max_rate)
            audio, marks = self._block_audio(engine, text, pros, folder, voice)
            return audio, pros, marks
        pros = block_prosody(text, self.cfg, 1.0, base_rate, max_rate)
        audio, _marks = self._block_audio(engine, text, pros, folder, voice)
        if audio is None or audio.size == 0:
            return None, pros, []
        tempo = min(need, max(float(max_rate), 1.0))
        if tempo <= 1.001:
            return audio, pros, []
        tmp = folder / "_tempo_src.wav"
        write_wav(tmp, audio, MIX_RATE)
        try:
            sped = to_mix(tmp, tempo, folder)
        finally:
            tmp.unlink(missing_ok=True)
        return sped, pros, []

    def _shrink_parts(self, parts: list[Phrase], ratio: float) -> bool:
        """Ужать реплики блока под окно. Текст субтитров (ph.text) не меняется."""
        changed = False
        ratio = min(max(float(ratio), 0.2), 0.97)
        for part in parts:
            src = " ".join((part.spoken or part.text).split())
            budget = max(8, int(len(src) * ratio * 0.90))
            if len(src) <= budget:
                continue
            short = compress_voice_text(self.llm, src, budget, self.log)
            if not short or len(short) >= len(src):
                continue
            part.spoken = short
            part.said = self._for_tts(short)
            changed = True
        return changed

    def _fit_block(self, engine, parts: list[Phrase], allowance: float, folder: Path,
                   base_rate: str, max_rate: float, slot: float | None = None):
        """Синтез блока. Сначала темп, если хватает потолка; иначе ужать текст и повторить."""
        def joined() -> str:
            return " ".join(p.said for p in parts if p.said).strip()

        text = joined()
        voice = parts[0].voice if parts else ""
        pros = block_prosody(text, self.cfg, 1.0, base_rate, max_rate)
        audio, marks = self._block_audio(engine, text, pros, folder, voice)
        if audio is None or audio.size == 0:
            return None, pros, [], False, False
        dur0 = audio.size / MIX_RATE
        if (slot and getattr(engine, "rate_fit", False) and dur0 > 0.3
                and dur0 < slot * SLOW_FILL):
            # Русский заметно короче оригинала: чуть медленнее, чтобы не оставлять дыру
            spros = block_prosody(text, self.cfg, 1.0, base_rate, max_rate,
                                  slow=dur0 / (slot * 0.95))
            if spros.rate != pros.rate:
                slow_audio, slow_marks = self._block_audio(engine, text, spros, folder, voice)
                if (slow_audio is not None and slow_audio.size
                        and slow_audio.size / MIX_RATE <= allowance + 0.06):
                    return slow_audio, spros, slow_marks, False, False
        compressed = False
        sped = False
        for _attempt in range(3):
            dur = audio.size / MIX_RATE
            if dur <= allowance + 0.06:
                return audio, pros, marks, compressed, sped
            need = dur / max(allowance, 0.25)
            if need <= max_rate + 1e-3:
                faster, fpros, fmarks = self._respeed(
                    engine, parts, need, folder, base_rate, max_rate)
                if faster is not None and faster.size < audio.size:
                    audio, pros, marks = faster, fpros, fmarks
                    sped = True
                    if audio.size / MIX_RATE <= allowance + 0.06:
                        return audio, pros, marks, compressed, sped
            if not self._shrink_parts(parts, allowance / max(dur, 0.25)):
                break
            compressed = True
            sped = False
            text = joined()
            pros = block_prosody(text, self.cfg, 1.0, base_rate, max_rate)
            audio, marks = self._block_audio(engine, text, pros, folder, voice)
            if audio is None or audio.size == 0:
                return None, pros, [], compressed, sped
        dur = audio.size / MIX_RATE
        if dur > allowance + 0.06:
            need = min(dur / max(allowance, 0.25), max_rate)
            if need > 1.01:
                faster, fpros, fmarks = self._respeed(
                    engine, parts, need, folder, base_rate, max_rate)
                if faster is not None and 0 < faster.size < audio.size:
                    audio, pros, marks = faster, fpros, fmarks
                    sped = True
        return audio, pros, marks, compressed, sped

    def _render_blocks(self, folder: Path, phrases: list[Phrase], engine, total: float,
                       clip_id, *, lag_sec: float, tail_sec: float, max_rate: float,
                       base_rate: str, gap: float, mode: str, check: bool,
                       progress_every: int = 1) -> VoiceTrack | None:
        """Блоки подряд: без фейда, без наложений, отставание не копится."""
        folder.mkdir(parents=True, exist_ok=True)
        from konveyer.saytext import latin_left
        left = list(dict.fromkeys(w for p in phrases
                                  for w in latin_left(clean_text(p.text), speech_table(self.cfg))))
        if left:
            self.log.info(f"латиница в озвучке: {', '.join(left[:40])}"
                          + (f" и ещё {len(left) - 40}" if len(left) > 40 else "")
                          + " — дополнить [voice.pronounce] или pronunciations.toml")
        blocks = voice_blocks(phrases)
        if not blocks:
            self.log.info(f"клип {clip_id}: озвучивать нечего — реплик нет")
            return None
        joined = sum(1 for block in blocks if len(block) > 1)
        if joined:
            self.log.info(f"клип {clip_id}: предложений одним запросом: {joined}")
        self.log.info(f"озвучка: {len(phrases)} фраз, блоков {len(blocks)}")
        n = int(round(max(float(total), 0.1) * MIX_RATE))
        track = np.zeros(max(n, 1), dtype=np.float32)
        gain = 10 ** (self.cfg.gain_db / 20)
        cursor = 0
        speech_end: int | None = None
        later: list[tuple[list[int], float, float]] = []   # не ответил основной голос — в конце
        if hasattr(engine, "allow_fallback"):
            engine.allow_fallback = False
        gap_n = int(round(max(float(gap), 0.0) * MIX_RATE))
        max_rate = max(float(max_rate), 1.0)
        # Текст каждого блока известен заранее — готовим его до синтеза
        for bi, idxs in enumerate(blocks):
            parts = [phrases[i] for i in idxs]
            for part in parts:
                if not part.said:
                    part.said = self._for_tts(part.text)
                if not part.spoken:
                    part.spoken = part.text
                part.dropped = ""
                part.tempo = 1.0
            nxt0 = None if bi + 1 == len(blocks) else phrases[blocks[bi + 1][0]]
            if (nxt0 is not None and continues_sentence(parts[-1].text)
                    and (nxt0.speaker or "") == (parts[-1].speaker or "")):
                # рез посреди предложения неизбежен: интонация без точки
                parts[-1].said = mid_sentence_tail(parts[-1].said)
        self._prefetch(engine, blocks, phrases, base_rate, max_rate, clip_id)
        t0 = time.monotonic()
        for bi, idxs in enumerate(blocks):
            parts = [phrases[i] for i in idxs]
            every = max(int(progress_every), 1)
            if idxs[0] % every == 0 or bi + 1 == len(blocks):
                self.log.info(f"озвучка: фразы {idxs[0] + 1}–{idxs[-1] + 1} из {len(phrases)}"
                              + eta_text(idxs[0], len(phrases), time.monotonic() - t0))
            origin = max(parts[0].origin, 0.0)
            is_last = bi + 1 == len(blocks)
            prev_part = phrases[blocks[bi - 1][-1]] if bi else None
            joining = (prev_part is not None and prev_part.duration > 0
                       and continues_sentence(prev_part.text)
                       and (prev_part.speaker or "") == (parts[0].speaker or ""))
            if joining:
                # продолжение того же предложения — не дольше JOIN_GAP после предыдущего куска
                origin = min(origin, prev_part.start + prev_part.duration + JOIN_GAP)
            if is_last:
                limit = float(total) + max(float(tail_sec), 0.0)
            else:
                next_origin = phrases[blocks[bi + 1][0]].origin
                limit = next_origin + max(float(lag_sec), 0.0) - max(float(gap), 0.0)
            start = max(origin, cursor / MIX_RATE)
            allowance = max(limit - start, 0.35)
            slot = (None if is_last or mode != "full"
                    else max(phrases[blocks[bi + 1][0]].origin - start, 0.0))
            audio, pros, marks, compressed, sped = self._fit_block(
                engine, parts, allowance, folder, base_rate, max_rate, slot=slot)
            if audio is None or audio.size == 0:
                for part in parts:
                    part.status = "fail"
                    part.spoken = ""
                    part.duration = 0.0
                speech_end = None
                later.append((idxs, start, allowance))
                continue
            a = max(int(round(start * MIX_RATE)), cursor)
            bounds = _speech_bounds(audio)
            if joining and speech_end is not None and bounds is not None:
                # Тишину в начале файла кладём поверх хвоста прошлого куска: на стыке
                # одного предложения слышно не больше JOIN_PAUSE, а не две тишины Edge подряд.
                want = speech_end + int(JOIN_PAUSE * MIX_RATE) - bounds[0]
                a = max(min(a, want), 0, speech_end + int(0.08 * MIX_RATE) - bounds[0])
                end = a + int(audio.size)
                if end > track.size:
                    track = np.concatenate([track, np.zeros(end - track.size, np.float32)])
                track[a:end] += np.asarray(audio, np.float32).reshape(-1) * np.float32(gain)
            else:
                track, end = self._paste(track, a, audio, gain)
            speech_end = a + bounds[1] if bounds is not None else None
            saids = [p.said or p.text for p in parts]
            cuts = _cuts_from_marks(audio.size, MIX_RATE, saids, marks)
            if cuts is None:
                cuts = _proportional_cuts(audio.size, saids)
            over = audio.size / MIX_RATE > allowance + 0.08
            base = a / MIX_RATE
            voice_end = (a + int(audio.size)) / MIX_RATE
            for i, part in enumerate(parts):
                dur = max((cuts[i + 1] - cuts[i]) / MIX_RATE, 0.0)
                part.start = round(base, 3)
                part.duration = round(dur, 3)
                if pros is not None:
                    part.kind, part.rate = pros.kind, pros.rate
                if over:
                    part.status = "over"
                elif is_last and voice_end > float(total) + 0.05:
                    part.status = "tail"
                elif compressed:
                    part.status = "short"
                elif sped:
                    part.status = "tempo"
                elif i == 0 and part.start > part.origin + 0.08:
                    # Отставание — только у начала блока. Вторая строка того же
                    # запроса начинается позже, потому что перед ней слова, не лаг.
                    part.status = "lag"
                else:
                    part.status = "ok"
                base += dur
            if over:
                sample = (parts[0].spoken or parts[0].text)[:48]
                self.log.info(f"клип {clip_id}: темп упёрся в потолок, слова не резал: {sample}")
            cursor = end + gap_n
        if hasattr(engine, "allow_fallback"):
            engine.allow_fallback = True
        if later:
            track = self._retry_later(engine, later, phrases, track, folder, base_rate,
                                      max_rate, gain, clip_id)
        spoken = [p for p in phrases if p.duration > 0 and p.status != "fail"]
        if not spoken:
            self.log.warn(f"клип {clip_id}: ни одна фраза не озвучилась")
            return None
        out = folder / "voice.wav"
        write_wav(out, track, MIX_RATE)
        if check or mode == "voiceover":
            return self._emit_voiceover(folder, track, phrases, engine, total, clip_id)
        intervals = duck_spans(phrases, bridge_sentence=True)
        write_json(folder / "voice.json", {"engine": engine.tag, "mode": mode,
                                           "phrases": [asdict(p) for p in phrases]})
        voice_end = max(p.start + p.duration for p in spoken)
        tail = max(0.0, voice_end - float(total))
        if tail > 0.05:
            self.log.info(f"клип {clip_id}: озвучка длиннее ролика на {tail:.1f} с")
        counts = {k: sum(p.status == k for p in phrases)
                  for k in ("tempo", "over", "short", "pause", "tail", "lag", "fail")}
        if counts["fail"]:
            lost = [p for p in phrases if p.status == "fail"]
            self.log.warn(f"ОШИБКА ОЗВУЧКИ: без звука {counts['fail']} из {len(phrases)} фраз, "
                          f"с {lost[0].origin:.1f} с: «{lost[0].text[:40]}»")
        who = (self._who(engine, phrases) if engine.name == "edge"
               else self.cfg.speaker if engine.name == "silero" else "XTTS")
        note = f"{who}, фраз: {len(spoken)}"
        extra = [f"ускорено {counts['tempo']}" if counts["tempo"] else "",
                 f"сжато {counts['short']}" if counts["short"] else "",
                 f"упор в темп {counts['over']}" if counts["over"] else "",
                 f"отставание {counts['lag']}" if counts["lag"] else "",
                 f"хвост {counts['tail']}" if counts["tail"] else "",
                 f"не озвучено {counts['fail']}" if counts["fail"] else ""]
        extra = [e for e in extra if e]
        if extra:
            note += " (" + ", ".join(extra) + ")"
        return VoiceTrack(out, intervals, note, phrases, end=round(voice_end, 3))

    def _retry_later(self, engine, later, phrases: list[Phrase], track: np.ndarray, folder: Path,
                     base_rate: str, max_rate: float, gain: float, clip_id) -> np.ndarray:
        """Отложенные куски: ещё раз основным голосом, и только потом — запасным."""
        self.log.info(f"клип {clip_id}: повтор отложенных кусков: {len(later)}")
        for idxs, start, allowance in later:
            parts = [phrases[i] for i in idxs]
            engine.fell_back = ""
            audio, pros, marks, _c, _s = self._fit_block(
                engine, parts, allowance, folder, base_rate, max_rate)
            if audio is None or audio.size == 0:
                continue
            fb = getattr(engine, "fell_back", "")
            a = int(round(start * MIX_RATE))
            end = a + int(audio.size)
            if end > track.size:
                track = np.concatenate([track, np.zeros(end - track.size, np.float32)])
            track[a:end] += np.asarray(audio, np.float32).reshape(-1) * np.float32(gain)
            saids = [p.said or p.text for p in parts]
            cuts = _cuts_from_marks(audio.size, MIX_RATE, saids, marks)
            if cuts is None:
                cuts = _proportional_cuts(audio.size, saids)
            base = a / MIX_RATE
            for i, part in enumerate(parts):
                dur = max((cuts[i + 1] - cuts[i]) / MIX_RATE, 0.0)
                part.start, part.duration = round(base, 3), round(dur, 3)
                part.spoken = part.text
                part.status = "ok"
                part.fallback = fb
                if pros is not None:
                    part.kind, part.rate = pros.kind, pros.rate
                base += dur
            if fb:
                self.log.warn(f"клип {clip_id}: ЗАПАСНОЙ ГОЛОС {fb} на {start:.1f} с: "
                              f"«{(parts[0].text or '')[:40]}»")
        return track

    def _emit_voiceover(self, folder: Path, track: np.ndarray, phrases: list[Phrase],
                        engine, total: float, clip_id) -> VoiceTrack | None:
        intervals = duck_spans(phrases, bridge_sentence=True)
        spoken = [p for p in phrases if p.duration > 0 and p.status != "fail"]
        if not spoken:
            self.log.warn(f"клип {clip_id}: ни одна фраза не озвучилась")
            return None
        out = folder / "voice.wav"
        write_wav(out, track, MIX_RATE)
        write_json(folder / "voice.json", {"engine": engine.tag, "mode": "voiceover",
                                           "phrases": [asdict(p) for p in phrases]})
        voice_end = max(ph.start + ph.duration for ph in spoken)
        lag = sum(p.status == "lag" for p in spoken)
        who = self._who(engine, phrases)
        note = f"{who}, закадр, фраз: {len(spoken)}"
        extra = []
        if lag:
            extra.append(f"отставание {lag}")
        if voice_end > total + 0.02:
            extra.append(f"дольше на {voice_end - total:.1f} с")
        fails = sum(p.status == "fail" for p in phrases)
        if fails:
            extra.append(f"не озвучено {fails}")
        if extra:
            note += " (" + ", ".join(extra) + ")"
        if lag:
            self.log.info(f"клип {clip_id}: закадр, отставание фраз: {lag}")
        if self.cfg.voice_check:
            try:
                from konveyer.voicecheck import review_voice
                review_voice(folder, phrases, self.s, self.log, self.llm, clip_id, hear=self._hear)
            except Exception as e:
                self.log.warn(f"клип {clip_id}: проверка озвучки пропущена ({str(e)[:160]})")
        return VoiceTrack(out, intervals, note, phrases, end=round(voice_end, 3))

    # ---------- клип
    def make(self, folder: Path, cues: list[Cue], total: float, clip_id) -> VoiceTrack | None:
        if self.broken:
            return None
        engine = self.engine()
        if engine is None:
            return None
        voiceover = self.cfg.voice_mode == "voiceover"
        phrases = phrases_from_cues(cues, total, min_slot=0.0 if voiceover else 0.2)
        self._assign_voices(phrases)
        if not phrases:
            self.log.info(f"клип {clip_id}: озвучивать нечего — реплик нет")
            return None
        use_rate = bool(getattr(engine, "rate_fit", False))
        prosodies = None
        for ph in phrases:
            # ph.text — строка субтитра. В синтез уходит _for_tts: без запятых, если включено.
            ph.said = self._for_tts(ph.text) if voiceover else self._speak(ph.text)
            if voiceover:
                ph.spoken = ph.text
        if voiceover and engine.name == "edge":
            try:
                return self._render_blocks(
                    folder, phrases, engine, total, clip_id,
                    lag_sec=LAG_LIMIT, tail_sec=TAIL_LIMIT,
                    max_rate=float(self.cfg.edge_max_rate),
                    base_rate=self.cfg.edge_rate, gap=VOICEOVER_GAP,
                    mode="voiceover", check=True)
            except KonveyerError as e:
                self.broken = str(e)
                self.log.warn(f"озвучка выключена: {e}")
                return None
        if use_rate:
            prosodies = [expression_prosody(p.said, self.cfg, 1.0) for p in phrases]
            for ph, pros in zip(phrases, prosodies):
                ph.kind, ph.rate = pros.kind, pros.rate
        try:
            files = self._synth(engine, [p.said for p in phrases], prosodies, [p.voice for p in phrases])
        except KonveyerError as e:
            self.broken = str(e)
            self.log.warn(f"озвучка выключена: {e}")
            return None
        if voiceover:
            return self._voiceover(folder, phrases, files, engine, total, clip_id)
        max_tempo = self.cfg.max_tempo

        # первый проход: что не влезает даже с ускорением — сократить
        retry: list[int] = []
        for i, (ph, f) in enumerate(zip(phrases, files)):
            if f is None:
                ph.status = "fail"
                continue
            ph.spoken, ph.duration = ph.text, self._length(f)
            action, tempo = plan_fit(ph.duration, ph.slot, max_tempo)
            ph.status, ph.tempo = action, tempo
            if use_rate and action == "tempo":
                f, action = self._boost_edge(engine, ph, f, tempo)
                files[i] = f
            # Edge текст не переписывает: не влезло после rate — окно и целые слова при сборке.
            will_shorten = (not use_rate and action == "shorten" and self.cfg.shorten
                            and self.llm is not None)
            if will_shorten:
                short = self._shorten(ph.text, target_chars(ph.text, ph.duration, ph.slot, max_tempo))
                if short:
                    ph.spoken = short
                    ph.said = self._speak(short)
                    retry.append(i)
                    continue
            if use_rate and action == "shorten":
                # сокращать нечем — rate до потолка, дальше окно и граница слова
                f, action = self._boost_edge(engine, ph, files[i], self.cfg.edge_max_rate)
                files[i] = f
                if action == "shorten":
                    ph.status, ph.tempo = "shorten", 1.0
        if retry:
            if use_rate:
                pros = [expression_prosody(phrases[i].said, self.cfg, 1.0) for i in retry]
                again = self._synth(engine, [phrases[i].said for i in retry], pros,
                                    [phrases[i].voice for i in retry])
            else:
                pros = None
                again = self._synth(engine, [phrases[i].said for i in retry], None,
                                    [phrases[i].voice for i in retry])
            for n, (i, f) in enumerate(zip(retry, again)):
                ph = phrases[i]
                if f is None:
                    ph.spoken = ph.text
                    ph.said = self._speak(ph.text)
                    continue
                files[i] = f
                ph.duration = self._length(f)
                if pros:
                    ph.kind, ph.rate = pros[n].kind, pros[n].rate
                action, tempo = plan_fit(ph.duration, ph.slot, max_tempo)
                if use_rate and action in ("tempo", "shorten"):
                    ratio = tempo if action == "tempo" else self.cfg.edge_max_rate
                    f2, leftover = self._boost_edge(engine, ph, f, ratio)
                    files[i] = f2
                    if leftover == "shorten":
                        ph.status, ph.tempo = "shorten", 1.0
                    else:
                        ph.status = "short"
                else:
                    ph.status = "short" if action != "shorten" else "shorten"
                    ph.tempo = tempo

        # сборка: окно можно растянуть за счёт паузы, звук посреди слова не режем
        for ph in phrases:
            ph.origin = ph.start
        durations = []
        for ph, f in zip(phrases, files):
            if f is None or ph.status == "fail":
                durations.append(0.0)
            else:
                durations.append(ph.duration / max(ph.tempo, 1e-6))
        ends = max_phrase_ends([p.origin for p in phrases], durations, total)
        n = int(round(total * MIX_RATE))
        track = np.zeros(max(n, 1), dtype=np.float32)
        intervals: list[tuple[float, float]] = []
        gain = 10 ** (self.cfg.gain_db / 20)
        cursor = 0.0
        for ph, f, end in zip(phrases, files, ends):
            if f is None or ph.status == "fail":
                continue
            samples = to_mix(f, ph.tempo, folder)
            start = ph.origin
            room = end - start
            if (samples.size > int((room + 0.04) * MIX_RATE) and cursor + VOICE_GAP < ph.origin):
                need = samples.size / MIX_RATE - max(room, 0.0)
                early = min(VOICE_BORROW, ph.origin - (cursor + VOICE_GAP), max(need, 0.0))
                start = max(0.0, ph.origin - early)
            if cursor > start:
                start = cursor
            limit_sec = max(0.05, min(end, total) - start)
            samples = self._fit_words_audio(engine, ph, samples, limit_sec, folder, use_rate)
            if samples is None or samples.size == 0:
                continue
            samples = fade_tail(samples, MIX_RATE)
            if ph.dropped:
                ph.status = "cut"
            elif ph.status == "shorten":
                ph.status = "tempo"
            ph.start = round(start, 3)
            a = int(round(ph.start * MIX_RATE))
            end_at = a + int(samples.size)
            if end_at > track.size:
                track = np.concatenate([track, np.zeros(end_at - track.size, np.float32)])
            if a >= track.size or end_at <= a:
                continue
            track[a:end_at] += samples * gain
            ph.duration = round(samples.size / MIX_RATE, 3)
            intervals.append((ph.start, round(ph.start + ph.duration, 3)))
            cursor = ph.start + ph.duration + VOICE_GAP
            if ph.start + ph.duration > total + 0.05:
                self.log.info(f"клип {clip_id}: озвучка длиннее клипа, "
                              f"хвост может упереться в конец: {(ph.spoken or ph.text)[:40]}")
        if not intervals:
            self.log.warn(f"клип {clip_id}: ни одна фраза не озвучилась")
            return None
        out = folder / "voice.wav"
        write_wav(out, track, MIX_RATE)
        write_json(folder / "voice.json", {"engine": engine.tag, "mode": self.cfg.voice_mode,
                                           "phrases": [asdict(p) for p in phrases]})
        for ph in phrases:
            if ph.dropped:
                self.log.info(f"клип {clip_id}: обрезка по границе слова, не влезло: {ph.dropped}")
        moved = sum(1 for p in phrases if p.duration > 0 and abs(p.start - p.origin) > 0.03)
        if moved:
            self.log.info(f"клип {clip_id}: сдвинуто фраз: {moved}")
        counts = {k: sum(p.status == k for p in phrases) for k in ("tempo", "short", "cut", "fail")}
        who = self._who(engine, phrases)
        note = f"{who}, фраз: {len(intervals)}"
        extra = [f"ускорено {counts['tempo']}" if counts["tempo"] else "",
                 f"сокращено {counts['short']}" if counts["short"] else "",
                 f"обрезано {counts['cut']}" if counts["cut"] else "",
                 f"сдвинуто {moved}" if moved else "",
                 f"не озвучено {counts['fail']}" if counts["fail"] else ""]
        extra = [e for e in extra if e]
        if extra:
            note += " (" + ", ".join(extra) + ")"
        if self.cfg.voice_check:
            try:
                from konveyer.voicecheck import review_voice
                review_voice(folder, phrases, self.s, self.log, self.llm, clip_id, hear=self._hear)
            except Exception as e:
                self.log.warn(f"клип {clip_id}: проверка озвучки пропущена ({str(e)[:160]})")
        voice_end = max((ph.start + ph.duration for ph in phrases
                         if ph.duration > 0 and ph.status != "fail"), default=0.0)
        return VoiceTrack(out, intervals, note, phrases, end=round(voice_end, 3))

    def make_timed(self, folder: Path, cues: list[Cue], total: float, clip_id, *,
                   lag_sec: float, max_rate: float, progress_every: int = 20) -> VoiceTrack | None:
        """Озвучка длинного ролика блоками: без обрезки слов и без накопления отставания.

        Пауза оригинала догоняет лаг. Если русский всё ещё длиннее окна, текст для
        голоса ужимается, субтитры остаются полными. Ускорение не выше max_rate.
        Хвост ролика не длиннее full.tail_sec. Проверку озвучки сюда не зовём.
        """
        if self.broken:
            return None
        engine = self.engine()
        if engine is None:
            return None
        phrases = phrases_from_cues(cues, max(total, 0.1) + 30.0)
        self._assign_voices(phrases)
        if not phrases:
            self.log.info(f"клип {clip_id}: озвучивать нечего — реплик нет")
            return None
        for ph in phrases:
            ph.said = self._for_tts(ph.text)
            ph.spoken = ph.text
            ph.origin = ph.start
        cap = min(max(float(max_rate), 1.0), float(self.cfg.edge_max_rate))
        try:
            return self._render_blocks(
                folder, phrases, engine, total, clip_id,
                lag_sec=float(lag_sec), tail_sec=float(self.s.full.tail_sec),
                max_rate=cap, base_rate=self.s.full.voice_rate, gap=0.04,
                mode="full", check=False, progress_every=progress_every)
        except KonveyerError as e:
            self.broken = str(e)
            self.log.warn(f"озвучка выключена: {e}")
            return None
