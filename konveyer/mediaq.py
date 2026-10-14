"""Кадры, субтитры, звук и техника клипа перед Telegram.

Четыре проверки лёгкие и бесплатные, включены всегда. Оценка DeepSeek и взгляд
Ollama выключены, пока не передали --score или --vision-check.
Ollama не запущена — одна строка в журнале, конвейер не падает.
"""
from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import numpy as np

from konveyer.config import Settings
from konveyer.facetrack import FaceFinder
from konveyer.jobs import Job
from konveyer.llm import chat_json, set_stage
from konveyer.log import Log
from konveyer.util import ffprobe, need_tool, run

OLLAMA_DOWN = "Ollama не запущена — проверка кадров ИИ пропущена"

_POS = re.compile(r"\\pos\(([-\d.]+)\s*,\s*([-\d.]+)\)")
_FS = re.compile(r"\\fs(\d+)")
_AN = re.compile(r"\\an(\d)")
_TAG = re.compile(r"\{[^}]*\}")
_PLAY_X = re.compile(r"PlayResX:\s*(\d+)")
_PLAY_Y = re.compile(r"PlayResY:\s*(\d+)")
_MEAN = re.compile(r"mean_volume:\s*(-?[\d.]+)\s*dB")
_MAX = re.compile(r"max_volume:\s*(-?[\d.]+)\s*dB")
_SIL = re.compile(r"silence_start:\s*(-?[\d.]+)|silence_end:\s*(-?[\d.]+)")

SCORE_SYSTEM = (
    "Ты оцениваешь, зайдёт ли короткий ролик в ленте. "
    "Смотри только на текст клипа. Верни JSON "
    "{\"score\": целое от 1 до 10, \"reason\": \"одна короткая фраза по-русски\"}. "
    "10 — точно залетит, 1 — скучно или непонятно. Не выдумывай то, чего нет в тексте."
)

VISION_PROMPT = (
    "Посмотри кадры короткого вертикального ролика. "
    "Если чёрный экран, сильное размытие, лицо обрезано краем кадра или субтитры вылезают — "
    "ответь JSON {\"ok\": false, \"reason\": \"коротко по-русски, что не так\"}. "
    "Если кадр нормальный — {\"ok\": true, \"reason\": \"\"}."
)


def sharpness(frame: np.ndarray) -> float:
    """Средний перепад соседних пикселей. Почти ноль — кадр смыт."""
    gray = frame.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    if gray.size < 16:
        return 0.0
    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    return float(gx + gy)


def is_black(frame: np.ndarray) -> bool:
    return float(frame.mean()) < 10 and float(frame.max()) < 28


def face_cut(box: tuple[float, float, float, float]) -> bool:
    """Крупное лицо упёрлось в край кадра."""
    x, y, w, h = box
    if w < 0.08 or h < 0.08:
        return False
    if x < 0.012 or y < 0.012 or x + w > 0.988 or y + h > 0.988:
        return True
    return False


def frame_issues(frames: list[np.ndarray],
                 boxes: list[list[tuple[float, float, float, float]]] | None = None) -> tuple[list[str], list]:
    """Замечания по кадрам и лица с центрального кадра (для субтитров)."""
    issues: list[str] = []
    if not frames:
        return ["кадры: не удалось прочитать клип"], []
    blacks = sum(1 for f in frames if is_black(f))
    if blacks >= max(2, int(len(frames) * 0.6)):
        issues.append("кадры: чёрный экран")
    scores = [sharpness(f) for f in frames]
    if scores and sorted(scores)[len(scores) // 2] < 1.0:
        issues.append("кадры: сильное размытие")
    faces: list[tuple[float, float, float, float]] = []
    per = boxes if boxes is not None else [[] for _ in frames]
    cut = False
    for group in per:
        for box in group:
            if face_cut(box):
                cut = True
        faces.extend(group)
    if cut:
        issues.append("кадры: лицо обрезано краем")
    mid = per[len(per) // 2] if per else []
    return issues, mid or faces[:3]


def _style_fields(ass: str) -> dict:
    found = {"size": 74, "align": 2, "margin_v": 400, "margin_l": 60}
    for line in ass.splitlines():
        if not line.startswith("Style:"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 22:
            continue
        try:
            found["size"] = int(float(parts[2]))
            found["align"] = int(float(parts[18]))
            found["margin_l"] = int(float(parts[19]))
            found["margin_v"] = int(float(parts[21]))
        except ValueError:
            pass
        break
    return found


def _event_box(text: str, style: dict, width: int, height: int) -> tuple[float, float, float, float] | None:
    """left, top, w, h текста в пикселях PlayRes."""
    tags = "".join(m.group(0) for m in re.finditer(r"\{[^}]*\}", text))
    fs_m = _FS.search(tags)
    an_m = _AN.search(tags)
    pos = _POS.search(tags)
    fs = int(fs_m.group(1)) if fs_m else style["size"]
    align = int(an_m.group(1)) if an_m else style["align"]
    plain = _TAG.sub("", text).replace("\\N", "\n").replace("\\n", "\n")
    lines = [ln for ln in plain.split("\n") if ln.strip()] or [""]
    box_w = max(len(ln) for ln in lines) * fs * 0.56
    box_h = len(lines) * fs * 1.15
    if box_w < 1:
        return None
    if pos:
        x, y = float(pos.group(1)), float(pos.group(2))
    else:
        ml, mv = style["margin_l"], style["margin_v"]
        if align in (1, 4, 7):
            x = ml
        elif align in (3, 6, 9):
            x = width - ml
        else:
            x = width / 2
        if align in (7, 8, 9):
            y = mv
        elif align in (4, 5, 6):
            y = height / 2
        else:
            y = height - mv
    if align in (1, 4, 7):
        left = x
    elif align in (3, 6, 9):
        left = x - box_w
    else:
        left = x - box_w / 2
    if align in (7, 8, 9):
        top = y
    elif align in (4, 5, 6):
        top = y - box_h / 2
    else:
        top = y - box_h
    return left, top, box_w, box_h


def _intersects(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    face = max(bw * bh, 1.0)
    return inter > 0.35 * face


def subtitle_issues(ass: str, faces: list[tuple[float, float, float, float]] | None = None,
                    width: int = 1080, height: int = 1920) -> list[str]:
    if not ass or "Dialogue:" not in ass:
        return []
    mx = _PLAY_X.search(ass)
    my = _PLAY_Y.search(ass)
    width = int(mx.group(1)) if mx else width
    height = int(my.group(1)) if my else height
    style = _style_fields(ass)
    issues: list[str] = []
    overflow = False
    cover = False
    face_px = [(x * width, y * height, w * width, h * height) for x, y, w, h in (faces or [])]
    for line in ass.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        box = _event_box(parts[9], style, width, height)
        if box is None:
            continue
        left, top, bw, bh = box
        if left < -12 or top < -12 or left + bw > width + 12 or top + bh > height + 12:
            overflow = True
        for face in face_px:
            if _intersects(box, face):
                cover = True
    if overflow:
        issues.append("субтитры: вылезают за край")
    if cover:
        issues.append("субтитры: перекрывают лицо")
    return issues


def parse_volume(stderr: str) -> dict:
    mean = _MEAN.search(stderr or "")
    mx = _MAX.search(stderr or "")
    return {"mean_db": float(mean.group(1)) if mean else None,
            "max_db": float(mx.group(1)) if mx else None}


def parse_silence(stderr: str) -> list[tuple[float, float | None]]:
    out: list[tuple[float, float | None]] = []
    start = None
    for m in _SIL.finditer(stderr or ""):
        if m.group(1) is not None:
            start = float(m.group(1))
        elif m.group(2) is not None and start is not None:
            out.append((start, float(m.group(2))))
            start = None
    if start is not None:
        out.append((start, None))
    return out


def audio_issues(stats: dict, duration: float, *, silence_sec: float = 1.0,
                 min_mean_db: float = -36.0, clip_db: float = -0.5,
                 has_audio: bool = True) -> list[str]:
    if not has_audio:
        return ["звук: нет дорожки"]
    issues = []
    mean, mx = stats.get("mean_db"), stats.get("max_db")
    if mean is not None and mean <= min_mean_db:
        issues.append("звук: слишком тихо")
    if mx is not None and mx >= clip_db:
        issues.append("звук: клиппинг")
    for a, b in stats.get("silence") or []:
        end = duration if b is None else b
        length = end - a
        if length < silence_sec:
            continue
        if a <= 0.08:
            issues.append("звук: длинная тишина в начале")
        if b is None or (duration and b >= duration - 0.2):
            issues.append("звук: длинная тишина в конце")
    audio_dur = stats.get("audio_duration") or 0
    if audio_dur and duration and abs(audio_dur - duration) > 0.5:
        issues.append("звук: длина не совпадает с видео")
    # одна и та же фраза не повторяется
    return list(dict.fromkeys(issues))


def tech_issues(info: dict, *, width: int, height: int, shorts_max: float,
                shorts_pack: bool, readable: bool) -> list[str]:
    issues = []
    if not readable:
        issues.append("техника: файл не читается целиком")
        return issues
    w, h = int(info.get("width") or 0), int(info.get("height") or 0)
    if (w, h) != (width, height):
        issues.append(f"техника: разрешение {w}×{h}, нужно {width}×{height}")
    rate = int(info.get("bit_rate") or 0)
    floor = 800_000 if width >= 1080 and height >= 1920 else 250_000
    if rate and rate < floor:
        issues.append("техника: низкий битрейт")
    dur = float(info.get("duration") or 0)
    if shorts_pack and dur > shorts_max + 0.4:
        issues.append(f"техника: длиннее {shorts_max:.0f} с для Shorts")
    return issues


def grab_frames(path: Path, n: int = 4) -> list[np.ndarray]:
    """3–5 кадров по ролику. n=4 — середина диапазона."""
    info = ffprobe(path)
    dur = max(float(info["duration"] or 0), 0.3)
    times = [dur * (i + 1) / (n + 1) for i in range(n)]
    src_w, src_h = max(int(info["width"]), 2), max(int(info["height"]), 2)
    w = 640 if src_w >= 640 else src_w - src_w % 2
    h = max(int(round(src_h * w / src_w / 2)) * 2, 2)
    frames = []
    size = w * h * 3
    ff = need_tool("ffmpeg")
    for t in times:
        proc = subprocess.run(
            [ff, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
             "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout
        if len(raw) >= size:
            frames.append(np.frombuffer(raw[:size], np.uint8).reshape(h, w, 3).copy())
    return frames


def _faces(frames: list[np.ndarray]) -> list[list[tuple[float, float, float, float]]]:
    finder = FaceFinder()
    try:
        if not finder.available:
            return [[] for _ in frames]
        return [finder.boxes(f) for f in frames]
    finally:
        finder.close()


def measure_audio(path: Path) -> tuple[dict, bool]:
    """(статистика, файл прочитан целиком). Ошибка ffmpeg — не бросаем наружу."""
    ff = need_tool("ffmpeg")
    try:
        proc = subprocess.run(
            [ff, "-nostdin", "-hide_banner", "-v", "info", "-i", str(path),
             "-af", "volumedetect,silencedetect=noise=-40dB:d=0.8", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}, False
    err = proc.stderr or ""
    stats = parse_volume(err)
    stats["silence"] = parse_silence(err)
    readable = proc.returncode == 0 and stats["mean_db"] is not None
    if proc.returncode != 0 and stats["mean_db"] is None:
        return stats, False
    return stats, proc.returncode == 0 or stats["mean_db"] is not None


def _probe(path: Path) -> dict:
    info = ffprobe(path)
    # битрейт и длина звука — отдельно, чтобы не ломать старый ffprobe для остальных
    try:
        proc = run([need_tool("ffprobe"), "-v", "error", "-print_format", "json",
                    "-show_streams", "-show_format", str(path)], "ffprobe", timeout=60)
        import json
        data = json.loads(proc.stdout)
    except Exception:
        data = {}
    fmt = data.get("format") or {}
    try:
        info["bit_rate"] = int(float(fmt.get("bit_rate") or 0))
    except (TypeError, ValueError):
        info["bit_rate"] = 0
    audio = next((s for s in data.get("streams") or [] if s.get("codec_type") == "audio"), None)
    try:
        info["audio_duration"] = float((audio or {}).get("duration") or 0)
    except (TypeError, ValueError):
        info["audio_duration"] = 0.0
    if not info["audio_duration"] and info.get("has_audio"):
        info["audio_duration"] = float(info.get("duration") or 0)
    return info


def inspect_clip(path: Path, settings: Settings, log: Log, ass: str = "", *,
                 width: int | None = None, height: int | None = None,
                 shorts_pack: bool | None = None) -> list[str]:
    """Все бесплатные проверки одного файла. Сбой инструмента — строка в журнале, не падение.

    width/height и shorts_pack — для целого видео: свой кадр 16:9 и без проверки «длиннее 60 с».
    """
    q = settings.quality
    expect_w = settings.render.width if width is None else int(width)
    expect_h = settings.render.height if height is None else int(height)
    pack = settings.post.shorts_pack if shorts_pack is None else bool(shorts_pack)
    try:
        info = _probe(path)
    except Exception as e:
        log.info(f"{path.name}: файл не читается ({str(e)[:160]})")
        return ["техника: файл не читается целиком"]
    try:
        frames = grab_frames(path, 4)
        boxes = _faces(frames) if frames else []
    except Exception as e:
        log.info(f"{path.name}: кадры не прочитались ({str(e)[:160]})")
        frames, boxes = [], []
    frame_msg, faces = frame_issues(frames, boxes)
    try:
        stats, readable = measure_audio(path)
    except Exception as e:
        log.info(f"{path.name}: звук не измерен ({str(e)[:160]})")
        stats, readable = {}, True
    if stats:
        stats["audio_duration"] = info.get("audio_duration") or 0
    issues = list(frame_msg)
    issues.extend(subtitle_issues(ass, faces, expect_w, expect_h))
    if stats:
        issues.extend(audio_issues(stats, float(info.get("duration") or 0),
                                   silence_sec=q.silence_sec, min_mean_db=q.min_mean_db,
                                   clip_db=q.clip_db, has_audio=bool(info.get("has_audio"))))
    elif not info.get("has_audio"):
        issues.append("звук: нет дорожки")
    issues.extend(tech_issues(info, width=expect_w, height=expect_h,
                              shorts_max=q.shorts_max_sec, shorts_pack=pack,
                              readable=readable if stats or not info.get("has_audio") else readable))
    return issues


def score_issues(job: Job, settings: Settings, log: Log, llm) -> list[str]:
    if not settings.quality.score:
        return []
    if llm is None:
        log.info("оценка момента пропущена: нет ключа DeepSeek")
        return []
    issues = []
    for clip in job.data.get("clips") or []:
        bits = " ".join(str(clip.get(k) or "") for k in ("kicker", "title", "bottom", "reason"))
        ass = job.clip_dir(int(clip["id"])) / "captions.ass"
        if ass.is_file():
            from konveyer.quality import ass_plain
            bits += "\n" + ass_plain(ass)
        bits = " ".join(bits.split())
        if len(bits) < 8:
            continue
        set_stage(llm, "score")
        try:
            data = chat_json(llm, SCORE_SYSTEM, f"Текст клипа {clip['id']}:\n{bits[:1500]}",
                             temperature=0)
        except Exception as e:
            log.info(f"клип {clip['id']}: оценка не получилась ({str(e)[:160]})")
            continue
        if not isinstance(data, dict):
            log.info(f"клип {clip['id']}: оценка не получилась (нет числа)")
            continue
        try:
            score = int(round(float(data.get("score"))))
        except (TypeError, ValueError):
            log.info(f"клип {clip['id']}: оценка не получилась (нет числа)")
            continue
        score = min(max(score, 1), 10)
        if score < settings.quality.min_score:
            why = str(data.get("reason") or "").strip()
            tail = f" — {why}" if why else ""
            issues.append(f"клип {clip['id']}: оценка {score} из 10, ниже порога "
                          f"{settings.quality.min_score}{tail}")
    return issues


def _b64_jpeg(frame: np.ndarray) -> str:
    """Кадр в base64. Без cv2 пишем PPM и просим ffmpeg, но для Ollama хватает PNG через ничего.
    Здесь сырой PNG минимальным кодером не собираем: шлём JPEG, если есть cv2, иначе пропускаем кадр.
    """
    cv2 = None
    try:
        import cv2 as _cv
        cv2 = _cv
    except Exception:
        cv2 = None
    if cv2 is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            return base64.b64encode(buf.tobytes()).decode("ascii")
    # запасной путь: несжатый PNG вручную не нужен — соберём через ffmpeg в пайпе
    h, w = frame.shape[:2]
    ff = need_tool("ffmpeg")
    proc = subprocess.run(
        [ff, "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        input=np.ascontiguousarray(frame).tobytes(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if proc.returncode == 0 and proc.stdout:
        return base64.b64encode(proc.stdout).decode("ascii")
    return ""


_ollama_warned = False


def vision_issues(frames: list[np.ndarray], settings: Settings, log: Log,
                  poster=None) -> list[str]:
    """Один запрос в Ollama. Нет сервера — одна строка и пустой список, без отказа клипу."""
    global _ollama_warned
    if not settings.quality.vision or not frames:
        return []
    images = []
    for frame in frames[:4]:
        pic = _b64_jpeg(frame)
        if pic:
            images.append(pic)
    if not images:
        log.info("проверка кадров ИИ: кадр не собрался")
        return []
    url = settings.quality.ollama_url.rstrip("/") + "/api/chat"
    payload = {"model": settings.quality.vision_model, "stream": False,
               "messages": [{"role": "user", "content": VISION_PROMPT, "images": images}]}
    try:
        if poster is not None:
            resp = poster(url, payload)
        else:
            import requests
            resp = requests.post(url, json=payload, timeout=settings.quality.vision_timeout)
    except Exception:
        if not _ollama_warned:
            log.info(OLLAMA_DOWN)
            _ollama_warned = True
        return []
    status = getattr(resp, "status_code", 0)
    if status != 200:
        if not _ollama_warned:
            log.info(OLLAMA_DOWN if status in (0, 404, 502, 503) else
                     f"Ollama ответила {status} — проверка кадров ИИ пропущена")
            _ollama_warned = True
        return []
    try:
        body = resp.json()
        text = ((body.get("message") or {}).get("content") or body.get("response") or "")
    except Exception:
        log.info("проверка кадров ИИ: Ollama ответила не текстом")
        return []
    from konveyer.llm import extract_json
    try:
        data = extract_json(text)
    except Exception:
        log.info("проверка кадров ИИ: в ответе Ollama нет JSON")
        return []
    if isinstance(data, dict) and data.get("ok") is False:
        why = str(data.get("reason") or "кадр не понравился").strip()
        return [f"кадры ИИ: {why}"]
    return []


def reset_ollama_warning() -> None:
    global _ollama_warned
    _ollama_warned = False


def media_issues(job: Job, settings: Settings, log: Log, llm) -> list[str]:
    """Бесплатные проверки по каждому клипу, затем оценка и Ollama, если их включили."""
    from konveyer.voicecheck import gate_issues

    issues: list[str] = []
    issues.extend(gate_issues(job, settings))
    vision_frames: list[np.ndarray] = []
    saw_file = False
    for clip in job.data.get("clips") or []:
        rel = str(clip.get("file") or "")
        if not rel:
            continue
        path = job.dir / rel
        if not path.is_file():
            issues.append(f"клип {clip['id']}: файл не найден")
            continue
        saw_file = True
        ass_path = path.parent / "captions.ass"
        ass = ass_path.read_text(encoding="utf-8") if ass_path.is_file() else ""
        full = bool(clip.get("full") or (job.data.get("options") or {}).get("full"))
        try:
            if full:
                found = inspect_clip(
                    path, settings, log, ass,
                    width=int(clip.get("width") or 0) or None,
                    height=int(clip.get("height") or 0) or None,
                    shorts_pack=False)
            else:
                found = inspect_clip(path, settings, log, ass)
        except Exception as e:
            log.info(f"клип {clip['id']}: проверка не удалась ({str(e)[:160]})")
            continue
        issues.extend(f"клип {clip['id']}: {item}" if not item.startswith("клип") else item
                      for item in found)
        if settings.quality.vision and len(vision_frames) < 4:
            try:
                vision_frames.extend(grab_frames(path, 2)[:2])
            except Exception:
                pass
    issues.extend(score_issues(job, settings, log, llm))
    if settings.quality.vision:
        if not saw_file:
            pass
        else:
            try:
                issues.extend(vision_issues(vision_frames[:4], settings, log))
            except Exception as e:
                log.info(f"проверка кадров ИИ не удалась ({str(e)[:160]})")
    return issues


def ollama_status(settings: Settings) -> tuple[bool, str]:
    """doctor: жива ли Ollama и есть ли модель из конфига. Сеть только на 127.0.0.1."""
    url = settings.quality.ollama_url.rstrip("/") + "/api/tags"
    try:
        import requests
        resp = requests.get(url, timeout=1.5)
    except Exception:
        return False, "не запущена — проверка кадров ИИ недоступна"
    if resp.status_code != 200:
        return False, f"ответ {resp.status_code}"
    try:
        names = [str(m.get("name") or "") for m in (resp.json().get("models") or [])]
    except Exception:
        return True, "отвечает"
    want = settings.quality.vision_model
    have = any(n == want or n.startswith(want + ":") or n.split(":")[0] == want for n in names)
    if have:
        return True, f"есть, модель {want}"
    shown = ", ".join(n.split(":")[0] for n in names[:4] if n) or "моделей нет"
    return True, f"отвечает, модели {want} нет (есть: {shown})"
