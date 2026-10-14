"""Целое видео для ВК: горизонтальный 16:9, без нарезки на Shorts.

Исходное разрешение, не выше 1080p. Картинка целиком: без зума и без размытого фона.
Субтитры снизу на весь ролик, перевод кусками. Озвучка — блоками, с догоном
отставания. Если файл больше порога Telegram, рядом лежит сжатая копия.
Файл лежит в jobs/<задание>/full/.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from konveyer.captions import caption_ass, make_cues
from konveyer.config import Settings
from konveyer.hardware import Encoder, detect_encoder, encoder_args
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.moments import compact_transcript
from konveyer.render import clip_words
from konveyer.speech import NO_SPEECH_MATERIAL, SKIP_CAPTIONS
from konveyer.transcript import Transcript
from konveyer.translate import translate_texts
from konveyer.util import (KonveyerError, ffmpeg_filter_path, ffprobe, fmt_time, mask,
                           need_tool, read_json, run, write_json)
from konveyer.voice import VoiceOver, align_cues_to_voice, duck_expr, separate_music, voice_gate


def is_full(job: Job) -> bool:
    return bool((job.data.get("options") or {}).get("full"))


def span_seconds(duration: float, start, end) -> tuple[float, float]:
    """Отрезок ролика. Пустые границы — весь ролик. Конец дальше ролика обрезается."""
    duration = float(duration)
    a = 0.0 if start in (None, "") else float(start)
    b = duration if end in (None, "") else float(end)
    if a < 0:
        a = 0.0
    if b > duration:
        b = duration
    if a >= duration or b - a < 0.4:
        raise KonveyerError("отрезок пустой: проверь --from и --to")
    return round(a, 3), round(b, 3)


def plan_span(job: Job, settings: Settings, log: Log, tr: Transcript) -> dict:
    """Один клип на весь ролик или на --from/--to. Моменты Shorts не выбираются."""
    del settings, tr
    meta = job.data["meta"]
    duration = float(meta["duration"])
    opts = job.data.get("options") or {}
    start, end = span_seconds(duration, opts.get("full_from"), opts.get("full_to"))
    if opts.get("full_to") not in (None, "") and float(opts["full_to"]) > duration + 0.05:
        log.info(f"конец отрезка дальше ролика, беру до {fmt_time(duration)}")
    log.info(f"целое видео: {fmt_time(start)}–{fmt_time(end)} "
             f"({fmt_time(end - start)}), без нарезки")
    clip = {
        "id": 1, "start": start, "end": end, "score": 0, "reason": "целое видео",
        "kicker": "", "title": "", "bottom": "", "file": "", "cover": "",
        "full": True, "layout": "wide",
    }
    job.data["clips"] = [clip]
    write_json(job.moments_file, [{"start": start, "end": end, "reason": "целое видео"}])
    job.save()
    return clip


# ------------------------------------------------------------------ кадр 16:9

def _even(x: float) -> int:
    v = int(round(x))
    v -= v % 2
    return max(v, 2)


@dataclass(frozen=True)
class WideFrame:
    width: int
    height: int
    content_w: int
    content_h: int


def full_canvas(src_w: int, src_h: int, max_height: int = 1080) -> WideFrame:
    """Кадр 16:9. Картинка вписывается целиком, не увеличивается и не выше max_height.

    Уже 16:9 и не крупнее потолка — размер исходника. Иначе чёрные поля, без зума
    и без размытия.
    """
    src_w = max(int(src_w), 2)
    src_h = max(int(src_h), 2)
    max_h = _even(max_height)
    max_w = _even(max_h * 16 / 9)
    scale = min(1.0, max_w / src_w, max_h / src_h)
    content_w = _even(src_w * scale)
    content_h = _even(src_h * scale)
    aspect = content_w / max(content_h, 1)
    target = 16 / 9
    if abs(aspect - target) < 0.02:
        width, height = content_w, content_h
    elif aspect < target:
        height = content_h
        width = _even(height * 16 / 9)
        if width < content_w:
            width = content_w
            height = _even(width * 9 / 16)
    else:
        width = content_w
        height = _even(width * 9 / 16)
        if height < content_h:
            height = content_h
            width = _even(height * 16 / 9)
    if width > max_w:
        width = max_w
    if height > max_h:
        height = max_h
    return WideFrame(width, height, min(content_w, width), min(content_h, height))


CAPTION_BOX = "&H60000000"   # подложка под субтитры: чёрный, ~38% прозрачности


def caption_px(height: int, settings: Settings) -> tuple[int, int]:
    """Размер шрифта и отступ снизу. На 1080p это 70 px (6,5% кадра) и 65 px от края."""
    cfg = settings.full
    size = cfg.caption_size or max(22, min(96, int(round(height * 0.065))))
    margin = cfg.caption_margin or max(24, int(round(height * 0.06)))
    return size, margin


def output_fps(meta: dict, fallback: int) -> int:
    raw = float(meta.get("fps") or 0)
    if 15 <= raw <= 60:
        return max(int(round(raw)), 1)
    return max(int(fallback), 1)


# ------------------------------------------------------------------ перевод кусками

def chunk_batches(texts: list[str], batch: int, max_chars: int) -> list[list[str]]:
    """Реплики пачками: не больше batch штук и не больше max_chars символов."""
    batch = max(int(batch), 1)
    max_chars = max(int(max_chars), 1)
    groups: list[list[str]] = []
    cur: list[str] = []
    chars = 0
    for text in texts:
        n = len(text)
        if cur and (len(cur) >= batch or chars + n > max_chars):
            groups.append(cur)
            cur, chars = [], 0
        cur.append(text)
        chars += n
    if cur:
        groups.append(cur)
    return groups


def translate_long(texts: list[str], llm, log: Log, subs_dir: str, batch: int,
                   max_chars: int) -> list[str]:
    """Перевод всего ролика несколькими запросами, чтобы не упереться в лимит."""
    if not texts:
        return []
    groups = chunk_batches(texts, batch, max_chars)
    out: list[str] = []
    for i, group in enumerate(groups, 1):
        log.info(f"перевод: кусок {i} из {len(groups)}, реплик {len(group)}, "
                 f"символов {sum(len(t) for t in group)}")
        out.extend(translate_texts(group, llm, log, subs_dir))
    return out


def translate_cached(texts: list[str], job: Job, llm, log: Log, settings: Settings) -> list[str]:
    cache_file = job.dir / "translations.json"
    cache: dict = read_json(cache_file, {}) or {}
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        if llm is None:
            for t in missing:
                cache[t] = t
        else:
            cfg = settings.full
            done = translate_long(missing, llm, log, settings.paths.subs_dir,
                                  cfg.translate_batch, cfg.translate_chars)
            cache.update(zip(missing, done))
        write_json(cache_file, cache)
    return [cache.get(t, t) for t in texts]


# ------------------------------------------------------------------ граф и кодирование

def build_wide_graph(*, width: int, height: int, content_w: int, content_h: int, fps: int,
                     ass_file: str | None, has_audio: bool, audio_in: str, duck: str,
                     voice_in: str, out_len: float, tail: float, fade: float) -> tuple[str, bool]:
    """Вписать картинку в 16:9 с чёрными полями. Хвост — последний кадр и фейд."""
    graph = (f"[0:v]fps={fps},setsar=1,"
             f"scale={content_w}:{content_h}:flags=lanczos,"
             f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    if tail > 0.05:
        graph += f",tpad=stop_mode=clone:stop_duration={tail:.3f}"
    if fade > 0.02 and tail > 0.05:
        start = max(out_len - fade, 0.0)
        graph += f",fade=t=out:st={start:.3f}:d={fade:.3f}"
    if ass_file:
        graph += f",ass='{ffmpeg_filter_path(ass_file)}'"
    graph += "[v]"
    if has_audio:
        chain = f"[{audio_in}]aresample=48000,apad=whole_dur={out_len:.3f}"
        if duck:
            chain += f",volume='{duck}':eval=frame"
        if voice_in:
            graph += (f";{chain}[a0];[{voice_in}]aresample=48000,apad=whole_dur={out_len:.3f}[vo];"
                      "[a0][vo]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0")
            if fade > 0.02 and tail > 0.05:
                start = max(out_len - fade, 0.0)
                graph += f",afade=t=out:st={start:.3f}:d={fade:.3f}"
            graph += "[a]"
        else:
            graph += f";{chain}[a]"
    elif voice_in:
        graph += f";[{voice_in}]aresample=48000,apad=whole_dur={out_len:.3f}[a]"
        has_audio = True
    return graph, has_audio


def run_watched(cmd: list[str], what: str, log: Log, cwd: Path, interval: float,
                timeout: float | None = None) -> None:
    """Запуск с сердцебиением в журнал: на длинном кодировании видно, что работа идёт."""
    log.info(f"{what}: старт")
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
        )
    except OSError as e:
        raise KonveyerError(f"{what}: не удалось запустить {cmd[0]} ({e})") from e
    err: list[str] = []

    def drain() -> None:
        if proc.stderr is not None:
            err.append(proc.stderr.read())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    t0 = time.monotonic()
    step = max(float(interval), 0.05)
    while True:
        try:
            proc.wait(timeout=step)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            log.info(f"{what}: ещё идёт, прошло {fmt_time(elapsed)}")
            if timeout is not None and elapsed > timeout:
                proc.kill()
                reader.join(timeout=2)
                raise KonveyerError(f"{what}: превышено время ожидания")
    reader.join(timeout=5)
    if proc.returncode != 0:
        tail = mask("".join(err).strip()[-1500:])
        raise KonveyerError(f"{what}: код выхода {proc.returncode}" + (f"\n{tail}" if tail else ""))


def _encode(src: Path, start: float, dur: float, graph: str, has_a: bool, out: Path,
            folder: Path, enc: Encoder, settings: Settings, log: Log, fps: int,
            extra: list[str]) -> Encoder:
    def cmd(encoder: Encoder) -> list[str]:
        line = [need_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src.resolve())]
        for name in extra:
            line += ["-i", name]
        line += ["-filter_complex", graph, "-map", "[v]"]
        if has_a:
            line += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        line += [*encoder.args, "-r", str(fps), "-movflags", "+faststart", out.name]
        return line

    label = f"кодирование целого видео ({enc.title}, {fmt_time(dur)})"
    try:
        run_watched(cmd(enc), label, log, folder, settings.full.progress_sec, timeout=6 * 3600)
        return enc
    except KonveyerError as e:
        if not enc.hardware:
            raise
        log.warn(f"{enc.title} не справился, кодирую на процессоре ({str(e)[:200]})")
        r = settings.render
        enc = Encoder("libx264", encoder_args("libx264", r.crf, r.preset))
        run_watched(cmd(enc), f"кодирование целого видео ({enc.title}, {fmt_time(dur)})",
                    log, folder, settings.full.progress_sec, timeout=6 * 3600)
        return enc


# ------------------------------------------------------------------ рендер

def _cue_list(words, settings: Settings):
    cfg = settings.full
    return make_cues(words, max_chars=cfg.max_chars, max_words=14, max_dur=4.5, gap=0.7)


def render_full(job: Job, settings: Settings, log: Log, tr: Transcript, llm,
                encoder: Encoder | None = None, voice_engine=None) -> None:
    clips = job.data.get("clips") or []
    if not clips:
        raise KonveyerError("нет отрезка целого видео — сначала шаг «лучшие моменты»")
    clip = clips[0]
    meta = job.data["meta"]
    src = job.source_video
    if src is None or not Path(src).is_file():
        raise KonveyerError("нет исходного видео")
    start = float(clip["start"])
    end = min(float(clip["end"]), float(meta["duration"]))
    if end <= start:
        raise KonveyerError("отрезок целого видео пустой")
    dur = end - start
    frame = full_canvas(int(meta["width"]), int(meta["height"]), settings.full.max_height)
    folder = job.dir / "full"
    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"кадр {frame.width}×{frame.height}, картинка {frame.content_w}×{frame.content_h}, "
             f"без зума и без размытия")
    words = clip_words(tr, start, end)
    cues = _cue_list(words, settings)
    cap_note = "нет"
    r = settings.render
    want = r.captions and (tr.reliable or settings.speech.force)
    if r.captions and not tr.reliable and not settings.speech.force:
        log.info(SKIP_CAPTIONS)
        cap_note = SKIP_CAPTIONS
        cues = []
    elif r.captions and not tr.reliable and settings.speech.force:
        log.info("речь ненадёжна — субтитры включены принудительно")
    if cues and tr.language and tr.language != "ru" and r.translate:
        if llm is not None:
            translated = translate_cached([c.text for c in cues], job, llm, log, settings)
            for cue, text in zip(cues, translated):
                cue.text, cue.words = text, []
            cap_note = "перевод на русский"
        else:
            log.warn("нет ключа DeepSeek — субтитры останутся на языке оригинала")
            cap_note = f"оригинал ({tr.language})"
    elif cues and tr.language == "ru":
        cap_note = "русские"
    elif cues:
        cap_note = f"оригинал ({tr.language or '?'})"

    tail = 0.0
    voice_inputs: list[str] = []
    voice_in = ""
    audio_in = "0:a"
    duck = ""
    clip.pop("voice", None)
    for old in ("voice.wav", "voice.json", "voice_check.json", "voice_sample.wav"):
        (folder / old).unlink(missing_ok=True)
    if settings.voice.enabled:
        why = voice_gate(settings, tr, llm)
        if why:
            log.info(why)
        elif not cues:
            log.info("озвучка пропущена: нет реплик")
        else:
            log.info("озвучка целого ролика, предложения целиком")
            voice = VoiceOver(job.dir, settings, log, llm, engine=voice_engine)
            track = voice.make_timed(
                folder, cues, dur, clip["id"], lag_sec=settings.full.lag_sec,
                max_rate=settings.voice.edge_max_rate, progress_every=settings.full.voice_batch)
            if track is not None:
                cues = align_cues_to_voice(cues, track.phrases)
                voice_inputs.append(track.path.name)
                voice_in = "1:a"
                duck = duck_expr(track.intervals, settings.voice.duck_db, settings.voice.duck_fade)
                voice_end = max(b for _, b in track.intervals)
                if voice_end > dur + 0.05:
                    tail = voice_end - dur
                    log.info(f"конец ролика продлеваю на {tail:.1f} с: последний кадр и фейд")
                note = track.note
                if settings.voice.separate and meta.get("has_audio"):
                    music = separate_music(src, start, dur, folder, settings.voice, log)
                    if music is not None:
                        voice_inputs.append(music.name)
                        audio_in = "2:a"
                        note += ", голос оригинала убран"
                clip["voice"] = note
                log.info(f"озвучка — {note}")
                if settings.voice.voice_check:
                    _check_voice(folder, track.phrases, settings, log, llm, clip["id"], voice._hear)

    ass_name = None
    if want and cues and not (settings.voice.enabled and clip.get("voice") and not settings.voice.subs):
        size, margin = caption_px(frame.height, settings)
        who = list(dict.fromkeys(c.speaker for c in cues if c.speaker))
        if who and settings.subs.speakers and cap_note not in ("нет", ""):
            cap_note += ", спикеры: " + ", ".join(who)
        ass = caption_ass(cues, settings, frame.width, frame.height, karaoke=False,
                          align=2, margin_v=margin, max_line=settings.full.max_line,
                          style="normal", size=size, outline=max(2, int(round(size * 0.06))),
                          max_lines=2, box=CAPTION_BOX if settings.full.caption_box else None)
        (folder / "captions.ass").write_text(ass, encoding="utf-8")
        ass_name = "captions.ass"
        if cap_note in ("нет", ""):
            cap_note = "есть"
        log.info(f"субтитры снизу, кегль {size}, отступ {margin}")
    elif settings.voice.enabled and clip.get("voice") and not settings.voice.subs:
        cap_note = "убраны (озвучка)"
    elif not r.captions:
        cap_note = "выключены"

    fade = settings.full.fade_sec if tail > 0.05 else 0.0
    if fade > tail > 0:
        fade = tail
    out_len = dur + tail
    fps = output_fps(meta, r.fps)
    graph, has_a = build_wide_graph(
        width=frame.width, height=frame.height, content_w=frame.content_w,
        content_h=frame.content_h, fps=fps, ass_file=ass_name,
        has_audio=bool(meta.get("has_audio")), audio_in=audio_in, duck=duck,
        voice_in=voice_in, out_len=out_len, tail=tail, fade=fade)
    (folder / "filter.txt").write_text(graph, encoding="utf-8")
    enc = encoder or detect_encoder(r.encoder, r.crf, r.preset)
    log.info(f"кодировщик: {enc.title}")
    part = folder / "video.part.mp4"
    enc = _encode(src, start, dur, graph, has_a, part, folder, enc, settings, log, fps, voice_inputs)
    out = folder / "video.mp4"
    os.replace(part, out)
    clip.update({
        "file": "full/video.mp4", "cover": "", "layout": "wide", "full": True,
        "width": frame.width, "height": frame.height, "encoder": enc.name,
        "out_duration": round(out_len, 2), "captions": cap_note, "zoom": 1.0,
        "tail": round(tail, 2),
    })
    job.save()
    log.info(f"файл: {out} ({out.stat().st_size / 2**20:.1f} МБ)")


def _check_voice(folder, phrases, settings, log, llm, clip_id, hear) -> None:
    try:
        from konveyer.voicecheck import review_voice_sample
        review_voice_sample(
            folder, phrases, settings, log, llm, clip_id,
            samples=settings.full.voice_check_samples,
            window=settings.full.voice_check_window, hear=hear)
    except Exception as e:
        log.warn(f"клип {clip_id}: проверка озвучки пропущена ({str(e)[:160]})")


# ------------------------------------------------------------------ пост

FULL_POST_SYSTEM = (
    "Ты пишешь пост ВК к длинному горизонтальному видео, не к Shorts. "
    "Группа «{group}». Язык русский, на «ты», ирония сухая, без восклицательных знаков "
    "и без детского тона. Это описание всего ролика: о чём он, кто автор (имя без ссылки), "
    "что в нём важного. Без пакета YouTube Shorts и без хештегов #shorts. "
    "Ссылок не ставить. Факты — ТОЛЬКО из расшифровки: никаких цифр, сумм, названий компаний, "
    "имён, событий и последствий, которых в ней прямо нет; не обобщай и не додумывай. "
    "Сомневаешься — не пиши. Если речи нет — не выдумывай сюжет.\n"
    "Отдельно дай таймкоды основных тем: от {topics} штук, время от начала ЭТОГО ролика "
    "(0:00 — начало отрезка). Короткий заголовок темы, 3–7 слов, без эмодзи.\n"
    "{rules}\n"
    "Ответ — только JSON: {{\"post\": \"...\", \"hashtags\": [\"#...\"], "
    "\"topics\": [{{\"time\": \"0:00\", \"title\": \"...\"}}]}}."
)


def parse_topics(data, limit: int, length: float = 0.0) -> list[dict]:
    """Таймкоды от модели: по возрастанию, без повторов, не дальше длины ролика."""
    from konveyer.moments import parse_time

    raw = data if isinstance(data, list) else []
    found: list[tuple[float, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or "").split())
        if not title:
            continue
        try:
            sec = parse_time(item.get("time", 0))
        except (TypeError, ValueError):
            continue
        if sec < 0 or (length > 0 and sec > length):
            continue
        found.append((float(sec), title[:80]))
    out: list[dict] = []
    seen = set()
    for sec, title in sorted(found, key=lambda x: x[0]):
        stamp = fmt_time(sec)
        if stamp in seen:
            continue
        seen.add(stamp)
        out.append({"time": stamp, "title": title})
        if len(out) >= limit:
            break
    return out


def topics_from_transcript(tr: Transcript, start: float, end: float, count: int) -> list[dict]:
    """Запасные таймкоды без модели: равномерно по репликам отрезка."""
    segs = [s for s in tr.segments if s.text.strip() and s.end > start and s.start < end]
    if not segs:
        return [{"time": "0:00", "title": "Начало"}]
    count = max(int(count), 1)
    if len(segs) <= count:
        picked = segs
    else:
        step = (len(segs) - 1) / max(count - 1, 1)
        picked = [segs[int(round(i * step))] for i in range(count)]
    out = []
    seen = set()
    for seg in picked:
        rel = max(seg.start - start, 0.0)
        stamp = fmt_time(rel)
        if stamp in seen:
            continue
        seen.add(stamp)
        words = seg.text.split()
        title = " ".join(words[:8]).strip(" .")
        if title:
            out.append({"time": stamp, "title": title[:80]})
    return out or [{"time": "0:00", "title": "Начало"}]


def _decorated(text: str, topics: list[dict], settings: Settings, meta: dict, data,
               ai: bool) -> dict:
    """Пост целого видео с эмодзи у абзацев и хештегами (см. postfmt)."""
    from konveyer.postfmt import decorate_post, model_tags

    body, tags = decorate_post(text, settings.post, model_tags(data), meta)
    return {"post": body, "topics": topics, "ai": ai, "hashtags": tags,
            "clock": bool(settings.post.emoji)}


def render_full_md(post: dict) -> str:
    from konveyer.postfmt import CLOCK

    head = f"{CLOCK} Таймкоды:" if post.get("clock") else "Таймкоды:"
    lines = [str(post.get("post") or "").strip(), "", head, ""]
    for topic in post.get("topics") or []:
        lines.append(f"{topic.get('time', '0:00')} — {topic.get('title', '')}".rstrip())
    if post.get("hashtags"):
        lines += ["", " ".join(post["hashtags"])]
    text = "\n".join(lines).strip() + "\n"
    if not post.get("ai", True):
        text += "\n> Черновик без ИИ: DeepSeek не был доступен. Допиши темы руками.\n"
    return text


def _slice_transcript(tr: Transcript, start: float, end: float) -> Transcript:
    from konveyer.transcript import Segment, Word

    segs = []
    for s in tr.segments:
        if s.end <= start or s.start >= end or not s.text.strip():
            continue
        words = []
        for w in s.words:
            if w.end <= start or w.start >= end:
                continue
            words.append(Word(round(max(w.start, start) - start, 3),
                              round(min(w.end, end) - start, 3), w.word, w.speaker))
        segs.append(Segment(round(max(s.start, start) - start, 3),
                            round(min(s.end, end) - start, 3), s.text.strip(), words))
    return Transcript(tr.language, segs, reliable=tr.reliable, speech_reason=tr.speech_reason)


def make_full_post(job: Job, settings: Settings, log: Log, llm, tr: Transcript,
                   avoid: str = "") -> dict:
    """Пост ВК к длинному видео: описание и таймкоды тем. Пакета Shorts нет.

    avoid — что проверка нашла выдуманным в прошлом варианте: это убрать.
    """
    from konveyer.llm import chat_json, set_stage
    from konveyer.moments import meta_block
    from konveyer.post import load_style

    meta = job.data.get("meta") or {}
    clip = (job.data.get("clips") or [{}])[0]
    start, end = float(clip.get("start") or 0), float(clip.get("end") or meta.get("duration") or 0)
    sliced = _slice_transcript(tr, start, end)
    topics_n = settings.full.topics
    post = None
    if llm is not None:
        log.info("пишу пост длинного видео через DeepSeek")
        set_stage(llm, "post")
        from konveyer.postfmt import PROMPT_RULES
        system = FULL_POST_SYSTEM.format(group=settings.post.group, topics=topics_n,
                                         rules=PROMPT_RULES.format(base=settings.post.hashtags))
        style = load_style(settings.post.style_files)
        if style:
            system += "\n\nПравила группы (тон, без структуры Shorts):\n" + style
        speech = (compact_transcript(sliced, settings.llm.max_transcript_chars)
                  if sliced.reliable and not sliced.empty else NO_SPEECH_MATERIAL)
        user = (f"{meta_block(meta)}\nДлина этого ролика: {fmt_time(max(end - start, 0))}\n"
                f"Язык речи: {tr.language or 'нет речи'}\n\nРасшифровка:\n{speech or '-'}")
        if avoid:
            user += ("\n\nПрошлый вариант поста проверка забраковала: " + avoid +
                     "\nУбери всё это. Пиши ТОЛЬКО то, что прямо сказано в расшифровке, "
                     "без цифр и названий, которых в ней нет.")
        try:
            data = chat_json(llm, system, user)
            if isinstance(data, dict) and str(data.get("post") or "").strip():
                topics = parse_topics(data.get("topics"), topics_n, max(end - start, 0.0))
                if not topics:
                    log.info("модель не дала таймкоды — соберу по репликам")
                    topics = topics_from_transcript(tr, start, end, topics_n)
                post = _decorated(str(data["post"]), topics, settings, meta, data, ai=True)
        except KonveyerError as e:
            log.warn(f"пост через ИИ не получился: {e}")
    if post is None:
        log.info("делаю черновик поста длинного видео")
        who = meta.get("uploader") or "автор"
        title = meta.get("title") or "ролик"
        topics = topics_from_transcript(tr, start, end, topics_n)
        text = (f"🤖 {title}\n\n{who} — ролик целиком, {fmt_time(max(end - start, 0))}.\n\n"
                "О чём видео:\n🔹 ...\n\nА ты досмотрел до конца? Пиши в комменты.")
        post = _decorated(text, topics, settings, meta, None, ai=False)
    write_json(job.post_json, post)
    job.post_md.write_text(render_full_md(post), encoding="utf-8")
    log.info(f"пост: {job.post_md.name}, таймкодов {len(post['topics'])}")
    return post


TOPIC_SYSTEM = (
    "Дай короткие осмысленные названия тем для таймкодов видео. На входе — пронумерованные куски "
    "русского перевода речи. Для каждого куска одна тема 3–6 слов по-русски, только о том, что "
    "в этом куске сказано, без выдумок, без кавычек и точки в конце. "
    "Ответ — только JSON: {\"titles\": [\"...\", ...]} в том же порядке и того же числа."
)


def topic_titles(llm, chunks: list[str], log: Log) -> list[str]:
    """Темы таймкодов от DeepSeek по кускам русского текста. Не вышло — пустой список."""
    from konveyer.llm import chat_json, set_stage

    if llm is None or not chunks:
        return []
    set_stage(llm, "post")
    user = "\n\n".join(f"{i + 1}. {c[:1500]}" for i, c in enumerate(chunks))
    try:
        data = chat_json(llm, TOPIC_SYSTEM, user, temperature=0)
    except Exception as e:
        log.info(f"темы таймкодов не получились ({str(e)[:120]})")
        return []
    titles = data.get("titles") if isinstance(data, dict) else None
    if not isinstance(titles, list) or len(titles) != len(chunks):
        return []
    return [" ".join(str(t).split()).strip(" .«»\"")[:80] for t in titles]


def simple_full_post(job: Job, settings: Settings, log: Log, tr: Transcript, llm=None) -> dict:
    """Пост без спорных фактов: заголовок, автор, длина и таймкоды тем."""
    meta = job.data.get("meta") or {}
    clip = (job.data.get("clips") or [{}])[0]
    start, end = float(clip.get("start") or 0), float(clip.get("end") or meta.get("duration") or 0)
    import re as _re
    topics = topics_from_transcript(tr, start, end, settings.full.topics)
    # таймкоды — по-русски: заголовок темы из русской озвучки/субтитров рядом с этим временем
    voice = read_json(job.dir / "full" / "voice.json", {})
    ru = [(float(p.get("origin") or 0.0), str(p.get("text") or ""))
          for p in (voice.get("phrases") or [])] if isinstance(voice, dict) else []
    if ru:
        from konveyer.moments import parse_time
        times = [parse_time(t["time"]) + start for t in topics]
        chunks = []
        for k, at in enumerate(times):
            stop = times[k + 1] if k + 1 < len(times) else float("inf")
            chunks.append(" ".join(txt for o, txt in ru if at - 0.5 <= o < stop - 0.5))
        titles = topic_titles(llm, chunks, log)
        for k, topic in enumerate(topics):
            if titles and titles[k]:
                topic["title"] = titles[k]
            elif chunks[k].strip():
                topic["title"] = " ".join(chunks[k].split()[:8]).strip(" .,:;—")[:80]
    who = meta.get("uploader") or "автор"
    title = str(meta.get("title") or "")
    head = title if len(_re.findall("[А-Яа-яЁё]", title)) > len(title) / 3 else ""
    text = ((head + "\n\n") if head else "") + \
        f"Видео целиком: {who}, {fmt_time(max(end - start, 0))}. Темы — по таймкодам ниже."
    post = _decorated(text, topics, settings, meta, None, ai=True)
    post["simple"] = True
    write_json(job.post_json, post)
    job.post_md.write_text(render_full_md(post), encoding="utf-8")
    log.info("пост упрощён: заголовок и таймкоды тем, без спорных фактов")
    return post


def repair_full_post(job: Job, settings: Settings, log: Log, llm, result, tries: int = 2):
    """Проверка нашла выдумки в посте: переписать по расшифровке (до tries раз), иначе простой пост.

    Возвращает итог последней проверки. Видео это не держит.
    """
    from konveyer.pipeline import load_transcript
    from konveyer.quality import assess_publish

    if result.ok or result.source != "deepseek" or llm is None:
        return result
    tr = load_transcript(job)
    for n in range(1, tries + 1):
        log.info(f"пост: переписываю по расшифровке ({n}/{tries}), убрать: {result.reason[:200]}")
        make_full_post(job, settings, log, llm, tr, avoid=result.reason)
        result = assess_publish(job, settings, log, llm)
        if result.ok:
            return result
        if result.source != "deepseek":
            return result
    simple_full_post(job, settings, log, tr, llm)
    return assess_publish(job, settings, log, llm)


# ------------------------------------------------------------------ Telegram

def full_limit_bytes(settings: Settings) -> int:
    """Порог файла в чат. Выше — сжатая копия, а если она не влезает, только путь."""
    mb = min(float(settings.full.telegram_max_mb), float(settings.telegram.max_mb))
    return max(int(mb * 1024 * 1024), 1)


def full_limit_mb(settings: Settings) -> float:
    return full_limit_bytes(settings) / (1024 * 1024)


def full_path_text(path: Path, post: str, settings: Settings) -> str:
    mb = path.stat().st_size / 2**20
    limit = full_limit_mb(settings)
    lines = [
        f"Видео {mb:.0f} МБ — больше порога {limit:g} МБ. "
        f"В Telegram файл не кладу и не сжимаю.",
        f"Файл: {path.resolve()}",
    ]
    if post.strip():
        lines += ["", post.strip()]
    return "\n".join(lines)


def full_compressed_text(original: Path, small: Path, post: str) -> str:
    """Сообщение к сжатой копии: в чат ушло video_tg, оригинал — путём."""
    lines = [
        f"В чат ушла сжатая копия 720p ({small.stat().st_size / 2**20:.0f} МБ).",
        f"Оригинал ({original.stat().st_size / 2**20:.0f} МБ): {original.resolve()}",
    ]
    if post.strip():
        lines += ["", post.strip()]
    return "\n".join(lines)


def telegram_target_bytes(settings: Settings) -> int:
    """Сколько байт метить. Не выше 94% порога, чтобы копия точно влезла в чат."""
    limit = full_limit_bytes(settings)
    wanted = int(float(settings.full.telegram_target_mb) * 1024 * 1024)
    return max(min(wanted, int(limit * 0.94)), 1)


def video_kbps_for(duration: float, target_bytes: int, audio_kbps: int = 128,
                   min_kbps: int = 300) -> int | None:
    """Битрейт картинки под целевой размер. None — даже минимум не влезает."""
    if duration <= 0 or target_bytes <= 0:
        return None
    video = int(target_bytes * 8 / duration / 1000) - int(audio_kbps)
    if video < int(min_kbps):
        return None
    return video


def tg_scale_filter(height: int) -> str:
    """720p, если исходник выше. Ниже — только чётные стороны, без увеличения.

    В фильтре нет запятой: ffmpeg режет -vf по запятым.
    """
    if int(height) > 720:
        return "scale=-2:720:flags=lanczos"
    return "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def tg_encode_commands(src: Path, dest: Path, *, height: int, kbps: int, has_audio: bool,
                       nvenc: bool, passlog: Path, ffmpeg: str | None = None) -> list[list[str]]:
    """Команды сжатия. NVENC — один проход, libx264 — два."""
    ff = ffmpeg or need_tool("ffmpeg")
    scale = tg_scale_filter(height)
    rate = ["-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k",
            "-pix_fmt", "yuv420p"]
    if nvenc:
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
               "-map", "0:v:0", "-vf", scale, "-c:v", "h264_nvenc", "-preset", "p5",
               "-rc", "vbr", *rate, "-multipass", "fullres"]
        if has_audio:
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", str(dest)]
        return [cmd]
    first = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
             "-map", "0:v:0", "-vf", scale, "-c:v", "libx264", "-preset", "medium",
             *rate, "-pass", "1", "-passlogfile", str(passlog), "-an", "-f", "null", os.devnull]
    second = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
              "-map", "0:v:0", "-vf", scale, "-c:v", "libx264", "-preset", "medium",
              *rate, "-pass", "2", "-passlogfile", str(passlog)]
    if has_audio:
        second += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"]
    else:
        second += ["-an"]
    second += ["-movflags", "+faststart", str(dest)]
    return [first, second]


def _drop_passlogs(folder: Path) -> None:
    for path in folder.glob("tgpass*"):
        path.unlink(missing_ok=True)


def compress_full_video(src: Path, settings: Settings, log: Log, *,
                        run=None, probe=None, nvenc=None) -> Path | None:
    """video_tg.mp4 рядом с оригиналом. None — в чат уйдёт только путь.

    NVENC, если он есть. Не вышло — libx264 в два прохода. Копия больше порога
    удаляется: мыло в чат не кладём.
    """
    probe = probe or ffprobe
    run = run or run_encode
    try:
        info = probe(src)
    except (KonveyerError, OSError) as e:
        log.warn(f"сжатие для Telegram не началось: {str(e)[:160]}")
        return None
    duration = float(info.get("duration") or 0)
    height = int(info.get("height") or 0)
    has_audio = bool(info.get("has_audio"))
    kbps = video_kbps_for(duration, telegram_target_bytes(settings), 128,
                          int(settings.full.min_video_kbps))
    if kbps is None:
        log.info("даже на минимальном битрейте ролик не влезает — не сжимаю")
        return None
    if nvenc is None:
        nvenc = detect_encoder("h264_nvenc").name == "h264_nvenc"
    dest = src.with_name("video_tg.mp4")
    passlog = src.with_name("tgpass")

    def once(use_nvenc: bool) -> None:
        for cmd in tg_encode_commands(src, dest, height=height, kbps=kbps, has_audio=has_audio,
                                      nvenc=use_nvenc, passlog=passlog):
            run(cmd, "сжатие для Telegram")

    try:
        try:
            once(bool(nvenc))
        except KonveyerError:
            if not nvenc:
                raise
            log.info("NVENC не сжал копию, пробую процессор")
            dest.unlink(missing_ok=True)
            once(False)
    except KonveyerError as e:
        log.warn(f"сжатие для Telegram не вышло: {str(e)[:200]}")
        dest.unlink(missing_ok=True)
        return None
    finally:
        _drop_passlogs(src.parent)
    limit = full_limit_bytes(settings)
    if not dest.is_file() or dest.stat().st_size > limit:
        if dest.is_file():
            dest.unlink()
        log.info("сжатая копия всё ещё больше порога — не сжимаю")
        return None
    log.info(f"сжатая копия: {dest.name}, {dest.stat().st_size / 2**20:.1f} МБ, {kbps} кбит/с")
    return dest


def run_encode(cmd: list[str], what: str) -> None:
    run(cmd, what, timeout=None)
