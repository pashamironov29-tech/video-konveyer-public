"""Шаг 1: скачать ролик (yt-dlp) или взять локальный файл, собрать метаданные."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from konveyer.config import Settings
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.util import KonveyerError, ffprobe, is_url, need_tool, read_json, run

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")


def ytdlp_format(max_height: int) -> str:
    """Лучшее качество до max_height, по возможности сразу mp4+m4a (без перекодирования)."""
    h = int(max_height)
    return (f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/"
            f"bv*[height<={h}]+ba/"
            f"b[height<={h}]/bv*+ba/b")


def ytdlp_command(url: str, folder: Path, settings: Settings) -> list[str]:
    cmd = [need_tool("yt-dlp"), "--no-playlist", "--no-progress",
           "-f", ytdlp_format(settings.ingest.max_height),
           "--merge-output-format", "mp4", "--write-info-json",
           "-P", str(folder), "-o", "source.%(ext)s"]
    if settings.ingest.cookies_from_browser:
        cmd += ["--cookies-from-browser", settings.ingest.cookies_from_browser]
    cookies = (settings.ingest.cookies_file or "").strip()
    if cookies:
        path = Path(cookies) if Path(cookies).is_absolute() else settings.root / cookies
        if path.is_file():
            cmd += ["--cookies", str(path)]
    return cmd + [url]


def meta_from_info(info: dict) -> dict:
    """Из .info.json yt-dlp — то, что нужно для поста: автор, название, описание."""
    uploader = (info.get("uploader") or info.get("channel") or info.get("creator")
                or info.get("uploader_id") or "")
    return {
        "title": (info.get("title") or "").strip(),
        "uploader": str(uploader).strip(),
        "uploader_id": str(info.get("uploader_id") or info.get("channel_id") or ""),
        # ник для «Источник: @ник»: у X — uploader_id, у Instagram — channel (uploader_id там число)
        "handle": next((str(v).lstrip("@") for v in (info.get("uploader_id"), info.get("channel"))
                        if v and not str(v).lstrip("@").isdigit() and " " not in str(v)), ""),
        "description": (info.get("description") or "").strip()[:4000],
        "url": info.get("webpage_url") or info.get("original_url") or "",
        "site": info.get("extractor_key") or info.get("extractor") or "",
        "upload_date": info.get("upload_date") or "",
        "tags": list(info.get("tags") or [])[:20],
    }


def _find_downloaded(folder: Path) -> Path:
    files = [p for p in folder.glob("source.*") if p.suffix.lower() in VIDEO_EXT]
    if not files:
        raise KonveyerError("yt-dlp отработал, но видеофайла нет. Возможно, по ссылке только картинка.")
    return max(files, key=lambda p: p.stat().st_size)


def download(url: str, job: Job, settings: Settings, log: Log) -> tuple[Path, dict]:
    log.info("качаю через yt-dlp (лучшее качество до %dp)" % settings.ingest.max_height)
    try:
        run(ytdlp_command(url, job.dir, settings), "yt-dlp (скачивание)", timeout=3600)
    except KonveyerError as e:
        hint = ""
        text = str(e).lower()
        if "login" in text or "cookies" in text or "private" in text or "rate" in text:
            hint = ("\nПодсказка: для Instagram и X часто нужен вход. Укажи в konveyer.toml "
                    "[ingest] cookies_from_browser = \"firefox\" (или chrome) "
                    "либо cookies_file = \"cookies/youtube_cookies.txt\" и повтори.")
        raise KonveyerError(f"{e}{hint}") from None
    video = _find_downloaded(job.dir)
    info_file = next(iter(job.dir.glob("source*.info.json")), None)
    info = read_json(info_file, {}) if info_file else {}
    return video, meta_from_info(info)


def take_local(path: str, job: Job, link: bool = False) -> tuple[Path, dict]:
    src = Path(path.strip('"')).expanduser()
    if not src.is_file():
        raise KonveyerError(f"файл не найден: {src}")
    dst = job.dir / f"source{src.suffix.lower() or '.mp4'}"
    linked = False
    if link:
        try:
            os.link(src, dst)             # тот же файл на диске: мгновенно, но правка тронет оригинал
            linked = True
        except OSError:
            pass
    if not linked:
        shutil.copy2(src, dst)
    return dst, {"title": src.stem, "uploader": "", "description": "", "url": "",
                 "site": "local", "upload_date": "", "tags": []}


def ingest(job: Job, settings: Settings, log: Log) -> None:
    source = job.data["source"]
    if is_url(source):
        video, meta = download(source, job, settings, log)
    else:
        log.info("беру локальный файл")
        video, meta = take_local(source, job, settings.ingest.link_local)
    try:
        probe = ffprobe(video)
    except KonveyerError:
        # не видео — не держим чужой файл в папке задания (её отдаёт веб-интерфейс)
        video.unlink(missing_ok=True)
        raise
    if probe["duration"] <= 0:
        raise KonveyerError("не удалось узнать длительность видео — файл битый?")
    meta.update(probe, file=video.name)
    job.data["meta"] = meta
    job.save()
    who = f", автор: {meta['uploader']}" if meta.get("uploader") else ""
    log.info(f"{meta['width']}x{meta['height']}, {probe['duration']:.1f} с, "
             f"звук: {'есть' if probe['has_audio'] else 'нет'}{who}")
    if meta.get("title"):
        log.info(f"название: {meta['title'][:100]}")
