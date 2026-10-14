"""Автоочистка папок заданий.

1. Сразу после УСПЕШНОЙ отправки в Telegram из папки задания удаляются исходники
   (скачанное видео) и промежуточные файлы: аудио, озвучка, кэш озвучки, временные
   рендеры, кэш кадров. Остаются готовые ролики, обложки, посты и описания, job.json.
   Если отправка не прошла (или ушло не всё), ничего не удаляется.
2. При каждом запуске конвейера задания старше [cleanup] keep_days удаляются целиком,
   но только те, что уже успешно ушли в Telegram.

`konveyer cleanup --dry-run` показывает, что и сколько будет удалено, ничего не трогая.
Переменная окружения KONVEYER_CLEANUP=0 выключает автоочистку (так делают тесты).
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

LEDGER = "delivered"            # то же имя, что в telegram.py
LOCK_NAME = "send.lock"

KEEP_ROOT = {"job.json", "job.log", "post.md", "post.json", "transcript.json",
             "translations.json", "analysis.json", "moments.json"}
OUTPUT_DIRS = ("clips", "full")          # здесь лежат готовые ролики, обложки, описания
FINAL_VIDEOS = {"clip.mp4", "video.mp4", "video_tg.mp4"}
KEEP_EXT = {".md", ".jpg", ".jpeg", ".png", ".webp", ".json", ".ass", ".srt"}


@dataclass
class Plan:
    job: Path
    paths: list[Path] = field(default_factory=list)
    size: int = 0


def disabled_by_env() -> bool:
    return os.environ.get("KONVEYER_CLEANUP", "").strip() == "0"


def path_size(p: Path) -> int:
    if p.is_symlink() or p.is_file():
        try:
            return p.lstat().st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def _keep_output(p: Path) -> bool:
    """Файл внутри clips/clip_NN/ или full/: оставить?"""
    if p.is_dir():
        return False                       # кадры, чанки и прочие подпапки — временные
    name = p.name.lower()
    if name.endswith(".mp4"):
        return name in FINAL_VIDEOS
    return p.suffix.lower() in KEEP_EXT


def _protected(job_dir: Path, data: dict | None) -> set[Path]:
    """Файлы, на которые ссылается job.json как на готовый результат, — не трогаем никогда."""
    keep: set[Path] = set()
    for c in (data or {}).get("clips") or []:
        for k in ("file", "cover", "description"):
            if c.get(k):
                keep.add((job_dir / str(c[k])).resolve())
    return keep


def intermediates(job_dir: Path, data: dict | None = None) -> list[Path]:
    """Что удалить после отправки: всё, кроме готовых роликов, обложек, текстов и job.json."""
    keep = _protected(job_dir, data)
    return [p for p in _candidates(job_dir)
            if p.resolve() not in keep and not any(_inside(p, k) for k in keep)]


def _candidates(job_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(job_dir.iterdir()):
        if p.name in OUTPUT_DIRS and p.is_dir():
            if p.name == "full":
                out += [q for q in sorted(p.iterdir()) if not _keep_output(q)]
            else:
                for clip in sorted(p.iterdir()):
                    if clip.is_dir():
                        out += [q for q in sorted(clip.iterdir()) if not _keep_output(q)]
                    else:
                        out.append(clip)
        elif p.is_dir() or p.name not in KEEP_ROOT:
            if p.name == LOCK_NAME:
                continue
            out.append(p)
    return out


def delivered_all(data: dict) -> bool:
    """Ушло ли в Telegram всё готовое: целое видео или каждый отрендеренный клип."""
    marks = data.get(LEDGER) or {}
    if (data.get("sent") or {}).get("review"):
        return False
    if (data.get("options") or {}).get("full"):
        return "video" in marks
    rendered = [c for c in data.get("clips") or [] if c.get("file")]
    return bool(rendered) and all(f"clip:{c['id']}" in marks for c in rendered)


def _inside(root: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _remove(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)


def plan_job(job_dir: Path, data: dict | None = None) -> Plan:
    plan = Plan(job_dir, intermediates(job_dir, data))
    plan.size = sum(path_size(p) for p in plan.paths)
    return plan


def clean_job(job, log=None, dry_run: bool = False) -> Plan:
    """Удалить исходники и промежуточные файлы одного задания (только если всё ушло)."""
    plan = Plan(job.dir)
    if not delivered_all(job.data) or (job.dir / LOCK_NAME).exists():
        return plan
    plan = plan_job(job.dir, job.data)
    if dry_run or not plan.paths:
        return plan
    for p in plan.paths:
        if _inside(job.dir, p):
            _remove(p)
    job.data["cleanup"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "removed": len(plan.paths), "freed_mb": round(plan.size / 2**20, 1)}
    job.save()
    if log is not None:
        log.info(f"очистка: удалено {len(plan.paths)} исходников и временных файлов, "
                 f"освобождено {plan.size / 2**20:.1f} МБ")
    return plan


def after_send(job, settings, log) -> Plan | None:
    """Хук после send_job: чистим, только если очистка включена и всё ушло."""
    if not settings.cleanup.enabled or disabled_by_env():
        return None
    job.reload()
    if not delivered_all(job.data):
        log.info("очистка: отправлено не всё — файлы задания не трогаю")
        return None
    try:
        return clean_job(job, log)
    except OSError as e:
        log.warn(f"очистка не удалась: {e}")
        return None


def job_created(job_dir: Path, data: dict) -> float:
    stamp = str(data.get("created") or "")
    try:
        return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        try:
            return (job_dir / "job.json").stat().st_mtime
        except OSError:
            return time.time()


@dataclass
class OldJob:
    dir: Path
    age_days: float
    size: int
    sent: bool


def old_jobs(settings, now: float | None = None) -> list[OldJob]:
    """Задания старше keep_days (с отметкой, ушли ли они в Telegram)."""
    from konveyer.util import read_json

    keep = float(settings.cleanup.keep_days)
    root = settings.jobs_dir
    if keep <= 0 or not root.is_dir():
        return []
    now = time.time() if now is None else now
    found: list[OldJob] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "job.json").is_file():
            continue
        data = read_json(d / "job.json") or {}
        age = (now - job_created(d, data)) / 86400
        if age < keep:
            continue
        found.append(OldJob(d, age, path_size(d), delivered_all(data)))
    return found


def remove_old(settings, log=None, dry_run: bool = False, now: float | None = None) -> list[OldJob]:
    """Удалить целиком старые задания, которые уже ушли. Возвращает удалённые (или кандидатов)."""
    root = settings.jobs_dir
    done: list[OldJob] = []
    for old in old_jobs(settings, now):
        if not old.sent or (old.dir / LOCK_NAME).exists() or not _inside(root, old.dir):
            continue
        if not dry_run:
            try:
                shutil.rmtree(old.dir)
            except OSError as e:
                if log is not None:
                    log.warn(f"не удалось удалить {old.dir.name}: {e}")
                continue
        done.append(old)
    if done and log is not None and not dry_run:
        mb = sum(o.size for o in done) / 2**20
        log.info(f"очистка: удалено старых заданий {len(done)} (старше "
                 f"{settings.cleanup.keep_days:g} дн.), освобождено {mb:.1f} МБ")
    return done


def auto_cleanup(settings, log=None) -> list[OldJob]:
    """Вызывается при каждом запуске конвейера."""
    if not settings.cleanup.enabled or disabled_by_env():
        return []
    try:
        return remove_old(settings, log)
    except Exception as e:  # очистка не должна ронять конвейер
        if log is not None:
            log.warn(f"автоочистка не удалась: {e}")
        return []


def report(settings, dry_run: bool = True, job_dirs: list[Path] | None = None) -> str:
    """Текст для `konveyer cleanup`: что удалится (или удалено) и сколько места."""
    from konveyer.jobs import Job

    lines: list[str] = []
    total = 0
    dirs = job_dirs if job_dirs is not None else sorted(
        d for d in settings.jobs_dir.iterdir() if (d / "job.json").is_file()) \
        if settings.jobs_dir.is_dir() else []
    word = "будет удалено" if dry_run else "удалено"
    old_sent = {o.dir for o in old_jobs(settings) if o.sent} if job_dirs is None else set()
    lines.append("Исходники и временные файлы отправленных заданий:")
    skipped: list[str] = []
    for d in dirs:
        if d in old_sent:
            continue                       # уйдёт целиком ниже
        job = Job.load(d)
        if not delivered_all(job.data):
            skipped.append(d.name)
            continue
        plan = clean_job(job, None, dry_run=dry_run)
        if not plan.paths:
            continue
        total += plan.size
        names = ", ".join(p.relative_to(d).as_posix() + ("/" if p.is_dir() else "")
                          for p in plan.paths[:8])
        more = f" и ещё {len(plan.paths) - 8}" if len(plan.paths) > 8 else ""
        lines.append(f"  {d.name}: {plan.size / 2**20:.1f} МБ — {names}{more}")
    if skipped:
        lines.append(f"  не отправлены (не трогаю): {', '.join(skipped)}")
    if job_dirs is None:
        old = remove_old(settings, None, dry_run=dry_run)
        keep = settings.cleanup.keep_days
        lines.append(f"Задания старше {keep:g} дн. (целиком):")
        if not old:
            lines.append("  нет")
        for o in old:
            total += o.size
            lines.append(f"  {o.dir.name}: {o.age_days:.1f} дн., {o.size / 2**20:.1f} МБ")
        unsent = [o for o in old_jobs(settings) if not o.sent]
        if unsent:
            lines.append("  старые, но не отправленные (не трогаю): "
                         + ", ".join(o.dir.name for o in unsent))
    lines.append(f"Итого {word}: {total / 2**20:.1f} МБ")
    return "\n".join(lines)
