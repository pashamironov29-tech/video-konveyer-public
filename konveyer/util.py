"""Общие мелочи: запуск программ, ffprobe, время, имена файлов, JSON."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from konveyer.envkeys import mask


class KonveyerError(Exception):
    """Ошибка, которую показываем пользователю как есть, без трейсбэка."""


def need_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise KonveyerError(f"не найден {name}. Поставь его и проверь, что он есть в PATH.")
    return path


def run(cmd: list[str], what: str, cwd: Path | None = None, capture: bool = True,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Запустить программу; при ошибке — KonveyerError с хвостом stderr."""
    try:
        proc = subprocess.run(
            [str(c) for c in cmd], cwd=cwd, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None, timeout=timeout,
        )
    except OSError as e:
        raise KonveyerError(f"{what}: не удалось запустить {cmd[0]} ({e})") from e
    except subprocess.TimeoutExpired as e:
        raise KonveyerError(f"{what}: превышено время ожидания") from e
    if proc.returncode != 0:
        tail = mask((proc.stderr or "").strip()[-1500:]) if capture else ""
        raise KonveyerError(f"{what}: код выхода {proc.returncode}" + (f"\n{tail}" if tail else ""))
    return proc


def ffprobe(path: Path) -> dict:
    """Размер кадра (с учётом поворота), длительность, fps, есть ли звук."""
    out = run([need_tool("ffprobe"), "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", str(path)], "ffprobe").stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        raise KonveyerError(f"в файле нет видеодорожки: {path}")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    w, h = int(v["width"]), int(v["height"])
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    if "rotate" in (v.get("tags") or {}):
        rotation = int(float(v["tags"]["rotate"]))
    if abs(rotation) % 180 == 90:
        w, h = h, w
    fps = 30.0
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1"
    try:
        num, den = rate.split("/")
        if float(den):
            fps = float(num) / float(den) or 30.0
    except (ValueError, ZeroDivisionError):
        pass
    duration = float(data.get("format", {}).get("duration") or v.get("duration") or 0)
    return {"width": w, "height": h, "duration": duration, "fps": round(fps, 3),
            "has_audio": a is not None, "audio_codec": a.get("codec_name") if a else None}


def fmt_time(seconds: float) -> str:
    """75.4 -> '1:15', 3725 -> '1:02:05'."""
    s = int(max(seconds, 0))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


_TRANSLIT = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p",
     "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"],
))


def slugify(text: str, max_len: int = 40) -> str:
    """Короткое безопасное имя для папки: латиница, цифры, дефисы."""
    text = "".join(_TRANSLIT.get(ch, ch) for ch in (text or "").lower())
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:max_len].rstrip("-")) or "video"


FILTER_INLINE_MAX = 6000      # длиннее — граф из файла: на Windows командная строка до ~32 КБ
_script_flag: str | None = None


def ffmpeg_script_flag() -> str:
    """Как передать граф файлом: ffmpeg 7+ — «-/filter_complex», старые — «-filter_complex_script»."""
    global _script_flag
    if _script_flag is None:
        try:
            out = subprocess.run([need_tool("ffmpeg"), "-hide_banner", "-version"],
                                 capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.TimeoutExpired, KonveyerError):
            out = ""
        m = re.search(r"ffmpeg version n?(\d+)\.", out)
        # ночные сборки («N-12345-…») номера версии не пишут, а они новее 7.0
        old = bool(m) and int(m.group(1)) < 7
        _script_flag = "-filter_complex_script" if old else "-/filter_complex"
    return _script_flag


def filter_complex_args(graph: str, script: str = "filter.txt") -> list[str]:
    """Аргументы для графа. Короткий — строкой, длинный — из файла script
    (его пишет вызывающий в рабочую папку ffmpeg)."""
    if len(graph) <= FILTER_INLINE_MAX:
        return ["-filter_complex", graph]
    return [ffmpeg_script_flag(), script]


def ffmpeg_filter_path(path: Path | str) -> str:
    """Путь для аргумента фильтра ffmpeg внутри одинарных кавычек ('...'): прямые слэши,
    экранированное двоеточие, апостроф закрывает кавычку и экранируется на обоих уровнях разбора."""
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"'\\\''")


def write_json(path: Path, data) -> None:
    """Атомарная запись JSON: сначала во временный файл, потом замена."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows: замена не проходит, пока другой поток/процесс читает файл
                if attempt == 19:
                    raise
                time.sleep(0.05)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default=None):
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except PermissionError:
            if attempt == 19:            # Windows: файл как раз заменяет write_json
                raise
            time.sleep(0.05)


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", (s or "").strip(), re.I))
