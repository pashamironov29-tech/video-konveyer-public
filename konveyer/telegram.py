"""Отправка через ваш Telegram-бот. Только отправка: входящие сообщения
конвейер не принимает. Токен и chat_id — из TG_BOT_TOKEN / TG_CHAT_ID."""
from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

from konveyer.clipdesc import fits_caption, read_description
from konveyer.config import Settings
from konveyer.envkeys import get_secret, mask
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.util import KonveyerError, ffprobe, need_tool, run

MSG_LIMIT = 4000
LEDGER = "delivered"     # job.json: что уже ушло в Telegram — второй раз не шлём без --resend
LOCK_NAME = "send.lock"
LOCK_STALE_SEC = 3600    # замок старше часа — остался от упавшего процесса
CAPTION_LIMIT = 1024


def split_message(text: str, limit: int = MSG_LIMIT) -> list[str]:
    """Режет длинный текст по абзацам/строкам, чтобы влезть в лимит Telegram."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    parts, cur = [], ""
    for para in text.split("\n"):
        while len(para) > limit:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(para[:limit])
            para = para[limit:]
        cand = f"{cur}\n{para}" if cur else para
        if len(cand) > limit:
            parts.append(cur)
            cur = para
        else:
            cur = cand
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def shrink_video(path: Path, target_bytes: int, duration: float) -> Path | None:
    """Пережать под лимит. None — если влезает только в мыле."""
    audio_kbps = 128
    for factor in (0.92, 0.78):
        video_kbps = int(target_bytes * 8 / max(duration, 1) / 1000 * factor) - audio_kbps
        if video_kbps < 400:
            return None
        dest = path.with_name(path.stem + "_tg.mp4")
        run([need_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-c:v", "libx264", "-preset", "medium", "-b:v", f"{video_kbps}k",
             "-maxrate", f"{video_kbps}k", "-bufsize", f"{video_kbps * 2}k", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-movflags", "+faststart", str(dest)],
            "ffmpeg (пережатие для Telegram)")
        if dest.stat().st_size <= target_bytes:
            return dest
    return None


class Telegram:
    def __init__(self, token: str, chat_id: str, session=None, max_mb: float = 50.0):
        if not token or not chat_id:
            raise KonveyerError("нужны переменные окружения TG_BOT_TOKEN и TG_CHAT_ID "
                                "(setx, затем новое окно терминала)")
        if session is None:
            import requests
            session = requests.Session()
        self.token, self.chat_id, self.session = token, chat_id, session
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.sent_ids: list[int] = []     # message_id всех удачных отправок, по порядку

    def _hide(self, text: str) -> str:
        return mask(str(text)).replace(self.token, "***")

    def call(self, method: str, data: dict, files: dict | None = None, timeout: int = 600) -> dict:
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        try:
            resp = self.session.post(url, data={"chat_id": self.chat_id, **data}, files=files,
                                     timeout=timeout)
        except Exception as e:
            raise KonveyerError(f"Telegram: сеть не отвечает ({self._hide(e)})") from None
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if not body.get("ok"):
            desc = body.get("description") or getattr(resp, "text", "")[:300]
            raise KonveyerError(f"Telegram ответил ошибкой {getattr(resp, 'status_code', '?')}: "
                                f"{self._hide(desc)}")
        result = body.get("result")
        if isinstance(result, dict) and result.get("message_id") is not None:
            self.sent_ids.append(result["message_id"])
        return body

    def send_message(self, text: str) -> int:
        parts = split_message(text)
        for p in parts:
            self.call("sendMessage", {"text": p, "disable_web_page_preview": "true"}, timeout=60)
        return len(parts)

    def send_video(self, path: Path, caption: str = "", log: Log | None = None,
                   thumb: Path | None = None) -> None:
        info = ffprobe(path)
        if path.stat().st_size > self.max_bytes:
            if log:
                log.info(f"{path.name}: {path.stat().st_size / 2**20:.1f} МБ — пережимаю под лимит бота")
            small = shrink_video(path, int(self.max_bytes * 0.94), info["duration"])
            if not small:
                raise KonveyerError(f"{path.name} не влезает в {self.max_bytes // 2**20} МБ даже после пережатия")
            path = small
        handles = []
        try:
            video = open(path, "rb")
            handles.append(video)
            files = {"video": (path.name, video, "video/mp4")}
            # Превью ролика — до 320 px и 200 КБ. Полная обложка всё равно уходит фото.
            if thumb is not None and thumb.is_file() and thumb.stat().st_size <= 200 * 1024:
                pic = open(thumb, "rb")
                handles.append(pic)
                files["thumbnail"] = (thumb.name, pic, "image/jpeg")
            self.call("sendVideo", {"caption": caption[:CAPTION_LIMIT], "supports_streaming": "true",
                                    "width": info["width"], "height": info["height"],
                                    "duration": int(info["duration"])}, files=files)
        finally:
            for handle in handles:
                handle.close()

    def send_photo(self, path: Path, caption: str = "") -> None:
        with open(path, "rb") as f:
            self.call("sendPhoto", {"caption": caption[:CAPTION_LIMIT]},
                      files={"photo": (path.name, f, "image/jpeg")}, timeout=120)


def make_telegram(settings: Settings, session=None) -> Telegram:
    return Telegram(get_secret("TG_BOT_TOKEN"), get_secret("TG_CHAT_ID"), session,
                    settings.telegram.max_mb)


# ------------------------------------------------------------------ учёт отправленного

def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def file_sig(path: Path) -> str:
    """Подпись файла: тот же файл второй раз не шлём, новый рендер — уже другой файл."""
    st = Path(path).stat()
    return f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}"


class Ledger:
    """Отметки в job.json: что ушло, message_id, хэш поста / подпись файла, время."""

    def __init__(self, job: Job, log: Log, tg: "Telegram", resend: bool = False):
        self.job, self.log, self.tg, self.resend = job, log, tg, resend

    @property
    def marks(self) -> dict:
        return self.job.data.setdefault(LEDGER, {})

    def sent(self, key: str, sig: str = "", what: str = "") -> bool:
        """True — уже отправлено, слать не надо (в журнал — «уже отправлено»)."""
        mark = self.marks.get(key)
        if not mark or self.resend:
            return False
        if sig and mark.get("sig") and mark["sig"] != sig and key != "post":
            return False      # другой файл (новый рендер) — это не повтор
        note = " (текст другой)" if key == "post" and sig and mark.get("sig") != sig else ""
        self.log.info(f"уже отправлено: {what or key} в {mark.get('at', '?')}{note} — "
                      f"не шлю повторно (переотправка: --resend)")
        return True

    def record(self, key: str, before: int, sig: str = "", **extra) -> None:
        ids = list(self.tg.sent_ids[before:]) if hasattr(self.tg, "sent_ids") else []
        self.marks[key] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "message_ids": ids,
                           **({"sig": sig} if sig else {}), **extra}
        self.job.save()

    def mark_pos(self) -> int:
        return len(getattr(self.tg, "sent_ids", []))


@contextmanager
def send_lock(job: Job):
    """Замок на время отправки: два процесса не шлют одно задание одновременно."""
    path = job.dir / LOCK_NAME
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                continue
            if age < LOCK_STALE_SEC:
                raise KonveyerError(f"отправка задания {job.id} уже идёт (замок {LOCK_NAME}); "
                                    f"если процесс упал — удали файл") from None
            path.unlink(missing_ok=True)
    else:
        raise KonveyerError(f"не удалось взять замок отправки {path}")
    try:
        os.write(fd, f"{os.getpid()} {time.strftime('%H:%M:%S')}".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def _who_line(meta: dict) -> str:
    """«Источник: @ник» (без платформы), иначе «Автор: имя»."""
    from konveyer.postfmt import source_line

    who = meta.get("uploader") or ""
    return source_line(meta) or (f"Автор: {who}" if who else "")


def full_caption(meta: dict) -> str:
    from konveyer.postfmt import scrub_platforms

    title = " ".join(scrub_platforms(str(meta.get("title") or "Видео")).split()) or "Видео"
    lines = [title, _who_line(meta)]
    return "\n".join(l for l in lines if l)[:CAPTION_LIMIT]


def clip_caption(clip: dict, meta: dict) -> str:
    title = " ".join(clip.get("title", "").replace("|", " ").split())
    bottom = " ".join(clip.get("bottom", "").replace("|", " ").split())
    lines = [f"Клип {clip['id']}: {title}".strip(), bottom, _who_line(meta)]
    return "\n".join(l for l in lines if l)


def send_job(job: Job, settings: Settings, log: Log, clips: bool = True, post: bool = True,
             only: list[int] | None = None, tg: Telegram | None = None, *,
             force: bool = False, llm=None, check: bool = True, resend: bool = False) -> dict:
    tg = tg or make_telegram(settings)
    with send_lock(job):
        sent = _send_job(job, settings, log, clips, post, only, tg, force=force, llm=llm,
                         check=check, resend=resend)
    from konveyer.cleanup import after_send

    after_send(job, settings, log)       # чистит, только если всё ушло
    return sent


def _send_job(job: Job, settings: Settings, log: Log, clips: bool, post: bool,
              only: list[int] | None, tg: Telegram, *, force: bool, llm, check: bool,
              resend: bool) -> dict:
    ledger = Ledger(job, log, tg, resend)
    if (post and not clips and not resend and LEDGER in job.data and "post" in ledger.marks
            and job.post_md.exists()):
        # только пост, и он уже ушёл — не гоняем проверку и не переписываем пост зря
        ledger.sent("post", text_hash(job.post_md.read_text(encoding="utf-8")), "пост")
        return job.data.get("sent") or {"clips": [], "post": True}
    if check and (clips or post):
        from konveyer.llm import make_llm
        from konveyer.quality import assess_publish, review_notice

        checker = llm if llm is not None else make_llm(settings, log, job=job, warn_missing=False)
        result = assess_publish(job, settings, log, checker)
        book = getattr(checker, "book", None)
        full = bool((job.data.get("options") or {}).get("full"))
        if not result.ok and full:
            # целое видео проверка не держит: пост переписываем, видео и обложка уходят всегда
            from konveyer.fullvideo import repair_full_post
            result = repair_full_post(job, settings, log, checker, result)
            if not result.ok:
                log.warn(f"проверка не пройдена ({result.reason}), целое видео всё равно отправляю")
                job.data["post_note"] = f"Пост: проверка не пройдена — {result.reason[:200]}"
            else:
                job.data.pop("post_note", None)
            force = True
        if not result.ok and not force:
            if book is not None:
                book.flush(final=True)
            notice = review_notice(job, result.reason)
            if not ledger.sent("review", text_hash(notice), "сообщение «на проверку»"):
                pos = ledger.mark_pos()
                tg.send_message(notice)
                ledger.record("review", pos, text_hash(notice))
            log.info(notice)
            job.data["review"] = {"ok": False, "reason": result.reason, "source": result.source}
            job.data["sent"] = {"clips": [], "post": False, "review": True}
            job.set_state("review", "send", result.reason)
            return job.data["sent"]
        if not result.ok and force:
            log.warn(f"проверка не пройдена ({result.reason}), отправляю принудительно")
        if book is not None and llm is None:
            # свой учёт, конвейер его не увидит — пишем итог здесь
            book.flush(final=True)
    if (job.data.get("options") or {}).get("full"):
        sent = send_full(job, settings, log, tg, clips=clips, post=post, ledger=ledger)
        if clips:
            from konveyer.voicecheck import voice_notes
            notes = voice_notes(job, settings)
            if job.data.get("post_note"):
                notes = (notes + "\n" if notes else "") + str(job.data["post_note"])
            if notes and not ledger.sent("guard", text_hash(notes), "сообщение сторожа"):
                pos = ledger.mark_pos()
                tg.send_message(notes)
                ledger.record("guard", pos, text_hash(notes))
                log.info("сообщение сторожа отправлено")
                sent["guard"] = True
                job.save()
        return sent
    sent = {"clips": [], "post": False}
    if clips:
        for clip in job.data.get("clips", []):
            if only and clip["id"] not in only:
                continue
            if not clip.get("file"):
                log.warn(f"клип {clip['id']} ещё не отрендерен — пропускаю")
                continue
            path = job.dir / clip["file"]
            thumb = _send_covers(tg, job, clip, log, ledger)
            key = f"clip:{clip['id']}"
            desc = read_description(job, clip) if settings.post.clip_desc else ""
            as_caption = bool(desc) and fits_caption(desc)
            if not ledger.sent(key, file_sig(path), f"клип {clip['id']}"):
                log.info(f"отправляю клип {clip['id']} ({path.stat().st_size / 2**20:.1f} МБ)"
                         + (", описание — подписью" if as_caption else ""))
                pos = ledger.mark_pos()
                caption = desc if as_caption else clip_caption(clip, job.data["meta"])
                tg.send_video(path, caption, log, thumb=thumb)
                ledger.record(key, pos, file_sig(path),
                              **({"desc": text_hash(desc)} if as_caption else {}))
                if desc and not as_caption:
                    # длиннее подписи — отдельным сообщением сразу после видео
                    pos = ledger.mark_pos()
                    tg.send_message(desc)
                    ledger.record(f"desc:{clip['id']}", pos, text_hash(desc))
                    log.info(f"описание клипа {clip['id']} отправлено сообщением после видео")
            elif desc and not as_caption and f"desc:{clip['id']}" not in ledger.marks:
                log.info(f"клип {clip['id']} уже ушёл раньше — описание отдельно не шлю "
                         f"(вместе с видео: --resend)")
            cover = job.dir / str(clip.get("cover") or "")
            if (thumb is None and settings.telegram.send_cover and clip.get("cover")
                    and not ledger.sent(f"cover:{clip['id']}", file_sig(cover), f"обложка {clip['id']}")):
                pos = ledger.mark_pos()
                tg.send_photo(cover, f"Обложка клипа {clip['id']}")
                ledger.record(f"cover:{clip['id']}", pos, file_sig(cover))
            sent["clips"].append(clip["id"])
    has_clips = any(c.get("file") for c in job.data.get("clips") or [])
    if post and clips and has_clips and not settings.telegram.shorts_post:
        log.info("общий пост после Shorts выключен ([telegram] shorts_post = false, "
                 "разово: send --shorts-post) — у каждого клипа своё описание")
        post = False
    if post and job.post_md.exists():
        text = job.post_md.read_text(encoding="utf-8")
        if ledger.sent("post", text_hash(text), "пост"):
            sent["post"] = True
        else:
            pos = ledger.mark_pos()
            n = tg.send_message(text)
            ledger.record("post", pos, text_hash(text))
            log.info(f"пост отправлен ({n} сообщ.)")
            sent["post"] = True
    job.data["sent"] = sent
    job.set_state("done")
    return sent


def _send_covers(tg: Telegram, job: Job, clip: dict | None, log: Log,
                 ledger: Ledger | None = None) -> Path | None:
    """Полные обложки фотографией перед видео. Возвращает файл для превью 320 px."""
    if not clip:
        return None
    from konveyer.cover import cover_files, telegram_thumb

    files = cover_files(job, clip)
    if not files:
        return None
    full = bool((job.data.get("options") or {}).get("full"))
    who = "видео" if full or clip.get("full") else f"клипа {clip.get('id')}"
    tag = "full" if full else str(clip.get("id"))
    for label, path in files:
        key = f"cover:{tag}:{label}"
        if ledger is not None and ledger.sent(key, file_sig(path), f"обложка {who} ({label})"):
            continue
        log.info(f"отправляю обложку {who} ({label})")
        pos = ledger.mark_pos() if ledger is not None else 0
        tg.send_photo(path, f"Обложка {who}: {label}")
        if ledger is not None:
            ledger.record(key, pos, file_sig(path))
    return telegram_thumb(files[0][1])


def full_video_path(job: Job) -> Path | None:
    for clip in job.data.get("clips") or []:
        rel = str(clip.get("file") or "")
        if rel:
            path = job.dir / rel
            if path.is_file():
                return path
    candidate = job.dir / "full" / "video.mp4"
    return candidate if candidate.is_file() else None


def send_full(job: Job, settings: Settings, log: Log, tg: Telegram, *,
              clips: bool = True, post: bool = True, ledger: Ledger | None = None) -> dict:
    """Целое видео: влезает — файлом. Больше порога — сжатая копия и путь оригинала.

    Если копия не собирается или всё равно больше порога, в чат уходит только путь.
    Что уже ушло (отметки в job.json), второй раз не шлётся.
    """
    from konveyer.fullvideo import (compress_full_video, full_compressed_text,
                                    full_limit_bytes, full_path_text)

    ledger = ledger or Ledger(job, log, tg)
    sent: dict = {"clips": [], "post": False, "full": True}
    path = full_video_path(job)
    post_text = job.post_md.read_text(encoding="utf-8") if post and job.post_md.exists() else ""
    if post_text and ledger.sent("post", text_hash(post_text), "пост"):
        sent["post"] = True
        post_text = ""
    clip = next((c for c in (job.data.get("clips") or []) if c.get("file")), None)
    thumb_src = None
    if clips:
        if clip:
            thumb_src = _send_covers(tg, job, clip, log, ledger)
        if path is None:
            log.warn("целого видео ещё нет — файл не отправляю")
        elif ledger.sent("video", file_sig(path), "целое видео"):
            sent["clips"] = [1]
        else:
            limit = full_limit_bytes(settings)
            size = path.stat().st_size
            small = None
            pos = ledger.mark_pos()
            extra: dict = {}
            if size <= limit:
                log.info(f"отправляю целое видео ({size / 2**20:.1f} МБ)")
                tg.send_video(path, full_caption(job.data.get("meta") or {}), log, thumb=thumb_src)
                sent["clips"] = [1]
            else:
                if settings.full.telegram_compress:
                    small = compress_full_video(path, settings, log)
                if small is not None and small.is_file() and small.stat().st_size <= limit:
                    log.info(f"отправляю сжатую копию ({small.stat().st_size / 2**20:.1f} МБ), "
                             f"оригинал {size / 2**20:.1f} МБ остаётся на диске")
                    tg.send_video(small, full_caption(job.data.get("meta") or {}), log, thumb=thumb_src)
                    tg.send_message(full_compressed_text(path, small, post_text))
                    sent["clips"] = [1]
                    sent["path"] = str(path.resolve())
                    try:
                        rel = small.relative_to(job.dir).as_posix()
                    except ValueError:
                        rel = small.name
                    sent["compressed"] = rel
                    extra["compressed"] = rel
                else:
                    note = full_path_text(path, post_text, settings)
                    log.info(f"видео {size / 2**20:.1f} МБ больше порога "
                             f"{limit / 2**20:.0f} МБ — в чат только путь")
                    tg.send_message(note)
                    sent["path"] = str(path.resolve())
                    extra["path_only"] = True
            ledger.record("video", pos, file_sig(path), **extra)
            if post_text and (sent.get("compressed") or extra.get("path_only")):
                # пост ушёл в одном сообщении с путём к оригиналу
                ledger.record("post", ledger.mark_pos(), text_hash(post_text), with_video=True)
                sent["post"] = True
                post_text = ""
    if post_text:
        pos = ledger.mark_pos()
        n = tg.send_message(post_text)
        ledger.record("post", pos, text_hash(post_text))
        log.info(f"пост отправлен ({n} сообщ.)")
        sent["post"] = True
    job.data["sent"] = sent
    job.set_state("done")
    return sent
