"""Задание = папка jobs/<id>/ со всеми файлами и состоянием в job.json.

Каждый шаг пишет свой файл; повторный запуск пропускает готовые шаги,
поэтому можно поправить заголовок и перерендерить один клип, не скачивая заново.
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from konveyer.config import Settings
from konveyer.util import KonveyerError, read_json, slugify, write_json

STEPS = ("ingest", "transcribe", "analyze", "moments", "render", "cover", "post")
STEP_NAMES = {
    "ingest": "скачивание", "transcribe": "распознавание", "analyze": "анализ",
    "moments": "лучшие моменты", "render": "рендер", "cover": "обложка",
    "post": "пост", "send": "отправка",
}

_lock = threading.RLock()

_HOSTS = {"youtube.com": "youtube", "youtu.be": "youtube", "x.com": "x", "twitter.com": "x",
          "instagram.com": "insta", "tiktok.com": "tiktok", "vk.com": "vk", "vkvideo.ru": "vk"}


def label_from_source(source: str) -> str:
    """Короткая метка для имени папки: «youtube-abc123», «x-1790...», «rolik»."""
    s = source.strip()
    if re.match(r"^https?://", s, re.I):
        parsed = urlparse(s)
        host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
        site = next((v for k, v in _HOSTS.items() if host.endswith(k)), host.split(".")[0])
        vid = parse_qs(parsed.query).get("v", [""])[0]
        if not vid:
            parts = [p for p in parsed.path.split("/") if p]
            vid = parts[-1] if parts else ""
        return f"{site}-{vid}" if vid else site
    # Path на Linux не режет обратный слэш, а задания приходят и с Windows-путей.
    name = s.replace("\\", "/").rstrip("/").split("/")[-1]
    return Path(name).stem


class Job:
    def __init__(self, folder: Path, data: dict):
        self.dir = folder
        self.data = data

    # ---------- создание и загрузка
    @classmethod
    def create(cls, settings: Settings, source: str, options: dict | None = None,
               label: str = "") -> "Job":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}_{slugify(label or label_from_source(source), 32)}"
        folder = settings.jobs_dir / base
        n = 2
        while folder.exists():
            folder = settings.jobs_dir / f"{base}-{n}"
            n += 1
        folder.mkdir(parents=True)
        job = cls(folder, {
            "id": folder.name, "source": source, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "options": dict(options or {}), "state": "new", "current": "", "error": "",
            "meta": {}, "steps": {}, "clips": [], "sent": {},
        })
        job.save()
        return job

    @classmethod
    def load(cls, folder: Path) -> "Job":
        data = read_json(folder / "job.json")
        if not data:
            raise KonveyerError(f"в папке нет job.json: {folder}")
        return cls(folder, data)

    def reload(self) -> None:
        self.data = read_json(self.file, self.data)

    def save(self) -> None:
        with _lock:
            write_json(self.file, self.data)

    # ---------- пути
    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def file(self) -> Path:
        return self.dir / "job.json"

    @property
    def log_file(self) -> Path:
        return self.dir / "job.log"

    @property
    def source_video(self) -> Path | None:
        name = self.data.get("meta", {}).get("file")
        return self.dir / name if name else None

    transcript_file = property(lambda self: self.dir / "transcript.json")
    analysis_file = property(lambda self: self.dir / "analysis.json")
    moments_file = property(lambda self: self.dir / "moments.json")
    post_json = property(lambda self: self.dir / "post.json")
    post_md = property(lambda self: self.dir / "post.md")

    def clip_dir(self, clip_id: int) -> Path:
        return self.dir / "clips" / f"clip_{clip_id:02d}"

    # ---------- шаги
    def done(self, step: str) -> bool:
        return self.data["steps"].get(step, {}).get("status") == "done"

    def mark(self, step: str, status: str, **info) -> None:
        self.data["steps"][step] = {"status": status, "at": time.strftime("%H:%M:%S"), **info}
        self.save()

    def invalidate_from(self, step: str) -> None:
        """Сбросить шаг и все следующие (например, после правки моментов)."""
        idx = STEPS.index(step)
        for name in STEPS[idx:]:
            self.data["steps"].pop(name, None)
        self.save()

    def set_state(self, state: str, current: str = "", error: str = "") -> None:
        self.data.update(state=state, current=current, error=error)
        self.save()

    def clip(self, clip_id: int) -> dict:
        for c in self.data["clips"]:
            if c["id"] == clip_id:
                return c
        raise KonveyerError(f"в задании {self.id} нет клипа №{clip_id}")

    def summary(self) -> dict:
        m = self.data.get("meta", {})
        return {"id": self.id, "created": self.data.get("created"), "state": self.data.get("state"),
                "current": self.data.get("current"), "error": self.data.get("error"),
                "title": m.get("title") or self.data.get("source"), "uploader": m.get("uploader", ""),
                "clips": len(self.data.get("clips", []))}


def list_jobs(settings: Settings) -> list[Job]:
    root = settings.jobs_dir
    if not root.exists():
        return []
    jobs = []
    for folder in sorted(root.iterdir(), reverse=True):
        if (folder / "job.json").exists():
            try:
                jobs.append(Job.load(folder))
            except Exception:
                continue
    return jobs


def find_job(settings: Settings, ref: str) -> Job:
    """ref: полный id, начало id, «last» или путь к папке задания."""
    p = Path(ref)
    if (p / "job.json").exists():
        return Job.load(p)
    jobs = list_jobs(settings)
    if not jobs:
        raise KonveyerError("заданий пока нет. Начни с: python -m konveyer run <ссылка>")
    if ref in ("last", "последнее", ""):
        return jobs[0]
    exact = [j for j in jobs if j.id == ref]
    if exact:
        return exact[0]
    found = [j for j in jobs if j.id.startswith(ref) or ref in j.id]
    if len(found) == 1:
        return found[0]
    if not found:
        raise KonveyerError(f"задание «{ref}» не найдено. Список: python -m konveyer jobs")
    raise KonveyerError(f"под «{ref}» подходит несколько заданий: " + ", ".join(j.id for j in found[:5]))
