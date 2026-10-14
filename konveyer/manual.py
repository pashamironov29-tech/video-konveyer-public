"""Shorts по ручным таймкодам: `clip <задание или ссылка> 2:10-2:50 [5:00-5:40 ...]`.

Моменты не выбираются — режем ровно заданные куски, границы подгоняем к ближайшему
концу предложения (±1,5 с). Скачанное, расшифровку, анализ и переводы берём из готового
задания, если оно есть: новое задание получает свою папку, старое не трогаем.
"""
from __future__ import annotations

import copy
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from konveyer.config import Settings
from konveyer.jobs import Job, find_job, label_from_source, list_jobs
from konveyer.log import Log
from konveyer.moments import Moment, default_texts, llm_titles, parse_time
from konveyer.transcript import Transcript, Word
from konveyer.util import KonveyerError, write_json

SNAP_SEC = 1.5          # насколько можно сдвинуть границу к концу предложения
LEAD_SEC = 0.15         # запас перед первым словом
TAIL_SEC = 0.3          # запас после последнего слова
END_RE = re.compile(r"[.!?…]+[\"»”'’)\]]*$")
# опции целого видео и автовыбора моментов в ручные Shorts не переносим
DROP_OPTIONS = {"full", "full_from", "full_to", "force_send", "count", "min", "max", "manual"}
REUSE_FILES = ("transcript.json", "analysis.json", "translations.json")


def parse_spans(items: list[str]) -> list[tuple[float, float]]:
    """["2:10-2:50", "300-340"] -> [(130, 170), (300, 340)]."""
    spans = []
    for raw in items:
        text = str(raw).strip().replace("–", "-").replace("—", "-")
        m = re.fullmatch(r"\s*([0-9:.,]+)\s*-\s*([0-9:.,]+)\s*", text)
        if not m:
            raise KonveyerError(f"не понимаю кусок «{raw}»: нужно начало-конец, например 2:10-2:50")
        try:
            a, b = parse_time(m.group(1)), parse_time(m.group(2))
        except ValueError as e:
            raise KonveyerError(str(e)) from e
        if b <= a:
            raise KonveyerError(f"кусок «{raw}»: конец должен быть позже начала")
        spans.append((a, b))
    if not spans:
        raise KonveyerError("нужен хотя бы один кусок, например 2:10-2:50")
    return spans


def is_spans(items: list[str]) -> bool:
    return bool(items) and all(re.search(r"\d\s*[-–—]\s*\d", str(x)) for x in items)


def _ends(w: Word) -> bool:
    return bool(END_RE.search((w.word or "").strip()))


def snap_start(t: float, words: list[Word], window: float = SNAP_SEC) -> float:
    """Начало — к ближайшему началу предложения, иначе к началу слова (слово не режем)."""
    if not words:
        return max(t, 0.0)
    firsts = [i for i, w in enumerate(words)
              if (i == 0 or _ends(words[i - 1])) and abs(w.start - t) <= window]
    if not firsts:
        firsts = [i for i, w in enumerate(words) if abs(w.start - t) <= window]
    if not firsts:
        return max(t, 0.0)
    i = min(firsts, key=lambda k: abs(words[k].start - t))
    prev_end = words[i - 1].end if i > 0 else 0.0
    return round(max(words[i].start - LEAD_SEC, prev_end, 0.0), 3)


def snap_end(t: float, words: list[Word], duration: float, window: float = SNAP_SEC) -> float:
    """Конец — к ближайшему концу предложения, иначе к концу слова."""
    if not words:
        return min(t, duration)
    lasts = [i for i, w in enumerate(words) if _ends(w) and abs(w.end - t) <= window]
    if not lasts:
        lasts = [i for i, w in enumerate(words) if abs(w.end - t) <= window]
    if not lasts:
        return min(t, duration)
    i = min(lasts, key=lambda k: abs(words[k].end - t))
    next_start = words[i + 1].start if i + 1 < len(words) else duration
    return round(min(words[i].end + TAIL_SEC, max(next_start, words[i].end), duration), 3)


def manual_moments(job: Job, settings: Settings, log: Log, llm, tr: Transcript,
                   spans: list[tuple[float, float]]) -> list[Moment]:
    meta = job.data.get("meta") or {}
    duration = float(meta.get("duration") or 0) or max(b for _a, b in spans)
    words = tr.words if tr.reliable else []
    moments = []
    for a, b in spans:
        a, b = max(0.0, a), min(duration, b)
        if b - a < 1.0:
            raise KonveyerError(f"кусок {fmt(a)}–{fmt(b)} за пределами ролика ({fmt(duration)})")
        start, end = snap_start(a, words), snap_end(b, words, duration)
        if end - start < 1.0:
            start, end = a, b
        moments.append(Moment(start, end, 10.0, f"ручной кусок {fmt(a)}–{fmt(b)}"))
    if settings.render.titles:
        if llm is not None:
            try:
                llm_titles(llm, moments, meta, tr if tr.reliable else Transcript(tr.language),
                           settings.post.group)
            except KonveyerError as e:
                log.warn(f"тексты на экран через ИИ не получились: {e}")
        for m in moments:
            default_texts(m, meta)
    else:
        for m in moments:
            m.kicker = m.title = m.bottom = ""
    clips = []
    for i, ((a, b), m) in enumerate(zip(spans, moments), 1):
        clips.append({"id": i, **asdict(m), "file": "", "cover": ""})
        log.info(f"клип {i}: задано {fmt(a)}–{fmt(b)}, режу {m.start:.1f}–{m.end:.1f} с "
                 f"({m.duration:.0f} с, границы по концам предложений)")
    write_json(job.moments_file, [asdict(m) for m in moments])
    job.data["clips"] = clips
    job.save()
    return moments


def fmt(sec: float) -> str:
    sec = max(float(sec), 0.0)
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def _is_source(ref: str) -> bool:
    return "://" in ref or Path(ref).is_file()


def find_base(settings: Settings, ref: str) -> Job | None:
    """Готовое задание: по id/папке, а для ссылки — последнее задание с тем же источником."""
    if not _is_source(ref):
        return find_job(settings, ref)
    for job in list_jobs(settings):
        video = job.source_video
        if (job.data.get("source") == ref and job.done("ingest") and job.done("transcribe")
                and video is not None and video.is_file()):
            return job
    return None


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def reuse_into(job: Job, base: Job, log: Log) -> list[str]:
    """Перенести скачанное и расшифровку из base в новое задание. Вернуть готовые шаги."""
    done = []
    video = base.source_video
    if base.done("ingest") and video is not None and video.is_file():
        job.data["meta"] = copy.deepcopy(base.data.get("meta") or {})
        _link(video, job.dir / video.name)
        done.append("ingest")
        for name in REUSE_FILES:
            src = base.dir / name
            if src.is_file():
                shutil.copy2(src, job.dir / name)
        if base.done("transcribe") and (job.dir / "transcript.json").is_file():
            done.append("transcribe")
            if base.done("analyze") and (job.dir / "analysis.json").is_file():
                done.append("analyze")
    for step in done:
        job.mark(step, "done", reused=base.id)
    if done:
        log.info(f"беру из задания {base.id}: {', '.join(done)}")
    return done


def manual_job(settings: Settings, ref: str, spans: list[tuple[float, float]],
               options: dict | None = None, label: str = "") -> tuple[Job, Job | None]:
    base = find_base(settings, ref)
    source = str(base.data.get("source") or ref) if base else ref
    opts = {k: v for k, v in ((base.data.get("options") or {}) if base else {}).items()
            if k not in DROP_OPTIONS}
    opts.update(options or {})
    opts["manual"] = [[round(a, 3), round(b, 3)] for a, b in spans]
    job = Job.create(settings, source, opts, label or (label_from_source(source) + "-clip"))
    log = Log(job.log_file)
    log.info(f"задание: {job.id} (Shorts по ручным таймкодам)")
    if base is not None:
        reuse_into(job, base, log)
    return job, base


def run_manual(settings: Settings, ref: str, items: list[str], options: dict | None = None,
               label: str = "", send: bool = False, force_send: bool = False) -> Job:
    """Новое задание с ручными кусками; шаг «моменты» берёт их из options.manual."""
    from konveyer.pipeline import Pipeline

    spans = parse_spans(items)
    job, _base = manual_job(settings, ref, spans, options, label)
    Pipeline(job, settings).run(send=send, force_send=force_send)
    return job


def spans_of(job: Job) -> list[tuple[float, float]]:
    raw = (job.data.get("options") or {}).get("manual") or []
    return [(float(a), float(b)) for a, b in raw]
