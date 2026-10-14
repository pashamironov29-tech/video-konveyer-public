"""Шаг 4: лучшие моменты для клипов.

1) Ролик короткий (X, Instagram) — берём целиком.
2) Есть речь и ключ DeepSeek — модель читает расшифровку с таймкодами и выбирает
   законченные по смыслу куски + пишет тексты на экран.
3) Иначе — эвристика: громкость + плотность речи + склейки.
Потом границы подгоняются к словам (не режем посреди слова) или к склейкам.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy as np

from konveyer.config import MomentsCfg, Settings
from konveyer.jobs import Job
from konveyer.llm import LLM, chat_json, set_stage
from konveyer.log import Log
from konveyer.textfit import split_text_lines
from konveyer.transcript import Transcript, Word
from konveyer.util import KonveyerError, write_json

TITLE_LINE = 18
OUTRO_SHARE = 0.08


@dataclass
class Moment:
    start: float
    end: float
    score: float = 5.0
    reason: str = ""
    kicker: str = ""
    title: str = ""
    bottom: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


# ------------------------------------------------------------------ время и расшифровка

def parse_time(value) -> float:
    """12.5 | "12.5" | "1:23" | "1:02:03.5" -> секунды."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", ".")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)", s)
    if not m:
        raise ValueError(f"не понимаю время: {value!r}")
    h, mnt, sec = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + float(sec)


def compact_transcript(tr: Transcript, max_chars: int) -> str:
    """Строки «[12.3-15.8] текст». Длинное склеивает в блоки, чтобы влезть в max_chars."""
    segs = [(s.start, s.end, s.text.strip()) for s in tr.segments if s.text.strip()]
    if not segs:
        return ""

    def render(blocks):
        return "\n".join(f"[{a:.1f}-{b:.1f}] {t}" for a, b, t in blocks)

    text = render(segs)
    window = 10.0
    blocks = segs
    while len(text) > max_chars and window <= 600:
        merged = []
        for a, b, t in segs:
            if merged and b - merged[-1][0] <= window:
                pa, _, pt = merged[-1]
                merged[-1] = (pa, b, f"{pt} {t}")
            else:
                merged.append((a, b, t))
        blocks = merged
        text = render(blocks)
        window *= 2
    if len(text) > max_chars:
        per = max(40, max_chars // max(len(blocks), 1) - 20)
        text = render([(a, b, t[:per]) for a, b, t in blocks])[:max_chars]
    return text


# ------------------------------------------------------------------ ИИ

_MOMENTS_HEAD = (
    "Ты монтажёр вертикальных роликов для ВК-группы «{group}». Аудитория взрослая, "
    "юмор — сухая ирония. Из расшифровки ролика ты выбираешь самые сильные куски для "
    "коротких клипов{titles_clause}.\n"
    "Правила выбора:\n"
    "- кусок законченный по смыслу: понятен без остального ролика, начинается с начала "
    "фразы и заканчивается на конце фразы или панчлайне;\n"
    "- длина от {min_sec:.0f} до {max_sec:.0f} секунд;\n"
    "- приоритет: неожиданное, смешное, конкретные факты и демонстрации, эмоции; "
    "вступления, рекламу и прощания не брать;\n"
    "- куски не пересекаются;\n"
    "- времена бери строго из таймкодов расшифровки, в секундах.\n"
)
_MOMENTS_TITLES = (
    "Тексты на экран (по-русски, без восклицательных знаков, без кликбейта-вранья, "
    "только то, что есть в ролике):\n"
    "- kicker: мелкая строка сверху, до 40 символов (кто/что, например имя автора или тема);\n"
    "- title: белый заголовок, 2 строки через «|», каждая до 18 символов;\n"
    "- bottom: жёлтая подпись снизу, 1–2 строки через «|», до 18 символов — суть или панчлайн.\n"
    'Ответ — только JSON: {"moments": [{"start": 0.0, "end": 0.0, "score": 1-10, '
    '"reason": "почему этот кусок", "kicker": "", "title": "", "bottom": ""}]}, '
    "отсортированный по score, лучший первым."
)
_MOMENTS_NO_TITLES = (
    "Тексты поверх кадра не пиши.\n"
    'Ответ — только JSON: {"moments": [{"start": 0.0, "end": 0.0, "score": 1-10, '
    '"reason": "почему этот кусок"}]}, отсортированный по score, лучший первым.'
)

TITLES_SYSTEM = (
    "Ты пишешь тексты на экран для вертикальных роликов ВК-группы «{group}». "
    "По-русски, сухая ирония, без восклицательных знаков, только факты из ролика. "
    "Для каждого ролика: kicker — мелкая строка сверху до 40 символов; title — белый "
    "заголовок, 2 строки через «|», до 18 символов в строке; bottom — жёлтая подпись, "
    "1–2 строки через «|», до 18 символов. "
    "Ответ — только JSON: {\"items\": [{\"id\": 1, \"kicker\": \"\", \"title\": \"\", \"bottom\": \"\"}]}."
)


def meta_block(meta: dict) -> str:
    parts = []
    for key, name in (("title", "Название"), ("uploader", "Автор"), ("site", "Площадка")):
        if meta.get(key):
            parts.append(f"{name}: {meta[key]}")
    if meta.get("description"):
        parts.append("Описание: " + meta["description"][:1200])
    return "\n".join(parts)


def moments_prompt(tr: Transcript, meta: dict, cfg: MomentsCfg, group: str,
                   max_chars: int, duration: float, with_titles: bool = True) -> tuple[str, str]:
    head = _MOMENTS_HEAD.format(group=group, min_sec=cfg.min_sec, max_sec=cfg.max_sec,
                                titles_clause=" и пишешь к ним короткие тексты на экран по-русски"
                                if with_titles else "")
    system = head + (_MOMENTS_TITLES if with_titles else _MOMENTS_NO_TITLES)
    user = (f"{meta_block(meta)}\nДлительность: {duration:.1f} с\n"
            f"Нужно клипов: до {cfg.count}\n\nРасшифровка (язык: {tr.language or '?'}):\n"
            f"{compact_transcript(tr, max_chars)}")
    return system, user


def parse_moments(data, duration: float) -> list[Moment]:
    items = data.get("moments", data.get("clips", [])) if isinstance(data, dict) else data
    out: list[Moment] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        try:
            start, end = parse_time(it.get("start")), parse_time(it.get("end"))
        except (ValueError, TypeError):
            continue
        start, end = max(0.0, start), min(duration, end)
        if end - start < 1.0:
            continue
        try:
            score = float(it.get("score", 5))
        except (TypeError, ValueError):
            score = 5.0
        out.append(Moment(start, end, max(0.0, min(10.0, score)),
                          str(it.get("reason", "")).strip(), str(it.get("kicker", "")).strip(),
                          str(it.get("title", "")).strip(), str(it.get("bottom", "")).strip()))
    return out


def llm_moments(llm: LLM, tr: Transcript, meta: dict, settings: Settings,
                duration: float) -> list[Moment]:
    set_stage(llm, "moments")
    system, user = moments_prompt(tr, meta, settings.moments, settings.post.group,
                                  settings.llm.max_transcript_chars, duration,
                                  with_titles=settings.render.titles)
    return parse_moments(chat_json(llm, system, user, temperature=0.7), duration)


def llm_titles(llm: LLM, moments: list[Moment], meta: dict, tr: Transcript, group: str) -> None:
    """Дописать тексты на экран тем моментам, где их нет (целый ролик, эвристика)."""
    need = [(i, m) for i, m in enumerate(moments, 1) if not m.title]
    if not need:
        return
    set_stage(llm, "moments")
    items = []
    for i, m in need:
        words = "" if not tr.reliable else " ".join(w.word.strip() for w in tr.words_between(m.start, m.end))
        items.append({"id": i, "start": round(m.start, 1), "end": round(m.end, 1),
                      "speech": words[:1500] or "(без речи)"})
    note = ""
    if not tr.reliable:
        note = "Речи в ролике нет. Не выдумывай тему, жанр и сюжет — только автор и название.\n\n"
    user = f"{meta_block(meta)}\n\n{note}Ролики:\n{json.dumps(items, ensure_ascii=False)}"
    data = chat_json(llm, TITLES_SYSTEM.replace("{group}", group), user)
    for it in (data.get("items", []) if isinstance(data, dict) else []):
        try:
            idx = int(it.get("id")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(moments) and not moments[idx].title:
            m = moments[idx]
            m.kicker = str(it.get("kicker", m.kicker)).strip()
            m.title = str(it.get("title", "")).strip()
            m.bottom = str(it.get("bottom", "")).strip()


# ------------------------------------------------------------------ без ИИ

def whole_moment(duration: float) -> Moment:
    return Moment(0.0, duration, 10.0, "ролик короткий — берём целиком")


def heuristic_moments(analysis: dict, tr: Transcript, duration: float,
                      cfg: MomentsCfg) -> list[Moment]:
    """Окна, где громче обычного, много речи и есть склейки."""
    hop = float(analysis.get("hop", 0.5))
    bins = max(int(np.ceil(duration / hop)), 1)

    def arr(values, fill):
        a = np.full(bins, fill, dtype=float)
        v = np.asarray(values[:bins], dtype=float)
        a[: v.size] = v
        return a

    loud = arr(analysis.get("loudness") or [], -60.0)
    speech = arr(analysis.get("speech") or [], 0.0)
    scenes = np.zeros(bins)
    for t in analysis.get("scenes") or []:
        i = int(t / hop)
        if 0 <= i < bins:
            scenes[i] = 1.0

    active = loud > -55
    if active.any():
        med, std = np.median(loud[active]), np.std(loud[active]) + 1e-6
    else:
        med, std = -60.0, 1.0
    z = np.clip((loud - med) / std, -3, 3)
    sp = speech / (speech.max() + 1e-6)
    per_bin = 0.6 * np.clip(z, 0, None) + 0.8 * sp + 0.3 * scenes - 0.5 * (~active)
    if duration > 180:                       # в конце длинных роликов титры и «подписывайтесь»
        tail = int(bins * (1 - OUTRO_SHARE))
        per_bin[tail:] -= 1.0

    target = min(max((cfg.min_sec + cfg.max_sec) / 2, cfg.min_sec), cfg.max_sec, duration)
    win = max(int(target / hop), 1)
    if win >= bins:
        return [Moment(0.0, duration, 5.0, "весь ролик — короче окна")]
    kernel = np.ones(win) / win
    window_score = np.convolve(per_bin, kernel, mode="valid")
    peaks = np.array([z[i:i + win].max() for i in range(window_score.size)])
    total = window_score + 0.25 * np.clip(peaks, 0, None)

    picked: list[int] = []
    for i in np.argsort(-total):
        if all(abs(int(i) - j) >= win for j in picked):
            picked.append(int(i))
        if len(picked) >= cfg.count:
            break
    span = float(total.max() - total.min()) or 1.0
    out = []
    for i in picked:
        score = round(3 + 7 * float(total[i] - total.min()) / span, 1)
        reasons = []
        if peaks[i] > 1:
            reasons.append("громкий пик")
        if sp[i:i + win].mean() > 0.4:
            reasons.append("много речи")
        if scenes[i:i + win].sum() >= 2:
            reasons.append("динамичный монтаж")
        out.append(Moment(i * hop, min(duration, (i + win) * hop), score,
                          "эвристика: " + (", ".join(reasons) or "самый живой участок")))
    return out


# ------------------------------------------------------------------ подгонка

def fit_moment(m: Moment, words: list[Word], scenes: list[float], duration: float,
               cfg: MomentsCfg) -> Moment:
    start, end = max(0.0, m.start), min(duration, m.end)
    if duration <= cfg.min_sec:
        return Moment(0.0, duration, m.score, m.reason, m.kicker, m.title, m.bottom)
    if end - start > cfg.max_sec:
        end = start + cfg.max_sec
    if end - start < cfg.min_sec:
        grow = cfg.min_sec - (end - start)
        end = min(duration, end + grow)
        start = max(0.0, end - cfg.min_sec)

    if words:
        near = [w for w in words if abs(w.start - start) <= 1.5]
        if near:
            w = min(near, key=lambda x: abs(x.start - start))
            prev_end = max((x.end for x in words if x.end <= w.start + 1e-6), default=0.0)
            start = max(w.start - 0.15, prev_end, 0.0)
        near = [w for w in words if abs(w.end - end) <= 1.5]
        if near:
            w = min(near, key=lambda x: abs(x.end - end))
            next_start = min((x.start for x in words if x.start >= w.end - 1e-6), default=duration)
            end = min(w.end + 0.3, next_start, duration)
    elif scenes:
        cand = [s for s in scenes if abs(s - start) <= 1.0]
        if cand:
            start = min(cand, key=lambda s: abs(s - start))
        cand = [s for s in scenes if abs(s - end) <= 1.0 and s > start + cfg.min_sec * 0.8]
        if cand:
            end = min(cand, key=lambda s: abs(s - end)) - 0.04

    if end - start > cfg.max_sec + 0.5:
        end = start + cfg.max_sec
    if end - start < cfg.min_sec * 0.8:
        end = min(duration, start + cfg.min_sec)
    return Moment(round(start, 3), round(end, 3), m.score, m.reason, m.kicker, m.title, m.bottom)


def remove_overlaps(moments: list[Moment], max_ratio: float = 0.25) -> list[Moment]:
    kept: list[Moment] = []
    for m in sorted(moments, key=lambda x: -x.score):
        ok = True
        for k in kept:
            inter = min(m.end, k.end) - max(m.start, k.start)
            if inter > max_ratio * min(m.duration, k.duration):
                ok = False
                break
        if ok:
            kept.append(m)
    return kept


def short_title(title: str, limit: int = 2 * TITLE_LINE) -> str:
    """Название ролика -> короткий заголовок: часть до « - », « | », «:»; обрезка по словам."""
    title = re.sub(r"https?://\S+|#\S+", "", title or "")
    title = re.sub(r"\s+", " ", title).strip()
    head = re.split(r"\s[-–—|•]\s|:\s|\s\|\s?", title)[0].strip()
    if len(head) >= 8:
        title = head
    if len(title) <= limit:
        return title
    cut = title[:limit + 1].rsplit(" ", 1)[0].rstrip(" ,.-–—")
    return cut or title[:limit]


def default_texts(m: Moment, meta: dict) -> None:
    """Тексты без ИИ: автор сверху, название ролика заголовком."""
    if not m.kicker:
        m.kicker = meta.get("uploader") or meta.get("site") or ""
    if not m.title:
        m.title = "|".join(split_text_lines(short_title(meta.get("title") or ""), 2, TITLE_LINE))


# ------------------------------------------------------------------ шаг целиком

def pick_moments(job: Job, settings: Settings, log: Log, llm: LLM | None,
                 tr: Transcript, analysis: dict) -> list[Moment]:
    meta = job.data["meta"]
    duration = float(meta["duration"])
    cfg = settings.moments
    count = int(job.data.get("options", {}).get("count") or cfg.count)
    cfg = MomentsCfg(**{**asdict(cfg), "count": count})

    usable = tr if tr.reliable else Transcript(tr.language)
    if not tr.reliable and not tr.empty:
        log.info("речи нет — моменты и заголовки без расшифровки")

    if duration <= cfg.whole_if_shorter:
        log.info(f"ролик {duration:.0f} с — делаю один клип целиком")
        moments = [whole_moment(duration)]
    elif llm and not usable.empty:
        log.info("прошу DeepSeek выбрать лучшие куски")
        try:
            moments = llm_moments(llm, usable, meta, settings, duration)
            if not moments:
                raise KonveyerError("модель не вернула ни одного подходящего куска")
        except KonveyerError as e:
            log.warn(f"{e}; подбираю без ИИ")
            moments = heuristic_moments(analysis, usable, duration, cfg)
    else:
        log.info("подбираю моменты без ИИ (громкость, речь, склейки)")
        moments = heuristic_moments(analysis, usable, duration, cfg)

    words = usable.words
    scenes = analysis.get("scenes") or []
    if not (len(moments) == 1 and moments[0].start == 0 and moments[0].end == duration):
        moments = [fit_moment(m, words, scenes, duration, cfg) for m in moments]
    moments = remove_overlaps(moments)[:cfg.count]

    if settings.render.titles:
        if llm and any(not m.title for m in moments):
            try:
                llm_titles(llm, moments, meta, usable, settings.post.group)
            except KonveyerError as e:
                log.warn(f"тексты на экран через ИИ не получились: {e}")
        for m in moments:
            default_texts(m, meta)
    else:
        for m in moments:
            m.kicker = m.title = m.bottom = ""
        log.info("надписи выключены — у DeepSeek их не прошу")

    clips = []
    for i, m in enumerate(moments, 1):
        clips.append({"id": i, **asdict(m), "file": "", "cover": ""})
        log.info(f"клип {i}: {m.start:.1f}–{m.end:.1f} с ({m.duration:.0f} с), "
                 f"оценка {m.score:g} — {m.reason[:80]}")
    write_json(job.moments_file, [asdict(m) for m in moments])
    job.data["clips"] = clips
    job.save()
    return moments
