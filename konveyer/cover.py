"""Обложка для ВК: кадр из видео или фон от нейросети, сверху заголовок Russo One.

Широкая (целое видео) — 1280×720. Вертикальная (Shorts) — 1080×1920.
Режим both кладёт обе: cover_frame.jpg и cover_ai.jpg. Если нейросеть не ответила,
остаётся кадр из ролика. JPEG меньше 2 МБ.
"""
from __future__ import annotations

import re
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import numpy as np

from konveyer.config import Settings
from konveyer.facetrack import FaceFinder
from konveyer.layout import font_candidates
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.mediaq import face_cut, is_black
from konveyer.moments import meta_block
from konveyer.util import KonveyerError, need_tool

WIDE = (1280, 720)
TALL = (1080, 1920)
JPEG_MAX = 2 * 1024 * 1024
THUMB_MAX = 320
THUMB_BYTES = 200 * 1024
FONT_FILE = "RussoOne-Regular.ttf"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/russoone/RussoOne-Regular.ttf"
SAFE_PROMPT = ", no people, no faces, no text, no watermark, high contrast"

TITLE_SYSTEM = (
    "Ты пишешь короткий заголовок обложки видео для группы «{group}». "
    "Только русский. От 2 до {max_words} слов. Можно вопрос или цифру. "
    "Без кликбейт-лжи: не добавляй факты, которых нет в названии и расшифровке. "
    "Без кавычек, хештегов и точки в конце. "
    "Примеры стиля: «ИИ ВЗЛОМАЛ САМ СЕБЯ?», «АЛЬТМАН ПРОТИВ МАСКА». "
    "Ответ — JSON {{\"title\": \"...\"}}."
)

PROMPT_SYSTEM = (
    "Write one English image prompt for a video thumbnail background. "
    "No real people, no celebrity likeness, no photorealistic faces, no letters, no watermark. "
    "Bold simple shapes, high contrast, cinematic light, empty area for a title. "
    "JSON {\"prompt\": \"...\"}."
)

_CLICK = re.compile(r"\b(шок|срочно|сенсация|жесть)\b|не\s+поверишь", re.I)
_WORD = r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё'’\-]{1,}"
_PERSON = rf"({_WORD}(?:\s+{_WORD}){{0,2}})"
_BAD_NAME = re.compile(
    r"\b(news|tv|channel|official|media|podcast|show|clips|канал|новости|тв|медиа|подкаст)\b",
    re.I)
_SPEAKER_PATTERNS = (
    rf"(?:[Ии]нтервью|[Рр]азговор|[Бб]еседа|[Пп]одкаст|[Ээ]фир|[Ii]nterview|[Cc]onversation)"
    rf"\s+(?:с|with)\s+{_PERSON}",
    rf"^{_PERSON}\s*[:|—–\-]\s+\S",
    rf"^{_PERSON}\s+(?:on|about|vs\.?|[Пп]ротив)\s+\S",
)


class ImageProvider(Protocol):
    def generate(self, prompt: str, width: int, height: int) -> bytes: ...


def bundled_font() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "fonts" / FONT_FILE


def resolve_font(settings: Settings, log: Log | None = None, fetch=None) -> Path:
    """Свой файл из конфига, иначе Russo One из репозитория, иначе скачать один раз."""
    custom = (settings.cover.font or "").strip()
    if custom:
        path = Path(custom)
        if not path.is_absolute():
            path = settings.root / path
        if path.is_file():
            return path
        if log:
            log.warn(f"шрифт обложки не найден ({path}), беру Russo One")
    bundled = bundled_font()
    if bundled.is_file():
        return bundled
    bundled.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = fetch(FONT_URL) if fetch else _download_font(FONT_URL)
        if not data or len(data) < 1000:
            raise RuntimeError("пустой файл шрифта")
        bundled.write_bytes(data)
        if log:
            log.info(f"скачал шрифт обложки: {bundled.name}")
        return bundled
    except Exception as e:
        for cand in font_candidates():
            if cand.is_file():
                if log:
                    log.warn(f"Russo One нет ({e}), рисую {Path(cand).name}")
                return Path(cand)
        raise KonveyerError(f"нет шрифта для обложки: {e}") from e


def _download_font(url: str) -> bytes:
    import requests
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def _pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise KonveyerError("для обложки нужен пакет pillow: py -m pip install pillow") from e
    return Image, ImageDraw, ImageFont


# ------------------------------------------------------------------ текст

def clean_title(raw: str, max_words: int) -> str:
    """2–5 слов, без кавычек и кликбейт-крика. Лишние слова отрезаются."""
    text = str(raw or "").replace("|", " ").replace("\n", " ")
    for ch in "«»\"“”":
        text = text.replace(ch, "")
    text = re.sub(r"[#@]", "", text)
    text = " ".join(text.split()).strip(" .")
    if not text or _CLICK.search(text):
        return ""
    words = text.split()
    limit = max(int(max_words), 1)
    asked = " ".join(words)
    if len(words) > limit:
        words = words[:limit]
        if "?" in asked and "?" not in " ".join(words):
            words[-1] = words[-1].rstrip(".!") + "?"
    return " ".join(words).strip()


def display_title(title: str) -> str:
    return " ".join(title.upper().split())


def _clean_name(name: str) -> str:
    name = " ".join(str(name or "").replace(":", " ").split()).strip(" -—–|")
    if not name or len(name) > 32 or len(name.split()) < 2 or _BAD_NAME.search(name):
        return ""
    return name


def guess_speaker(meta: dict | None) -> str:
    """Имя с плашки: из названия или описания, иначе автор, если это похоже на имя."""
    meta = meta or {}
    blobs = [str(meta.get("title") or ""), str(meta.get("description") or "")[:500]]
    for text in blobs:
        line = text.strip()
        if not line:
            continue
        for pat in _SPEAKER_PATTERNS:
            found = re.search(pat, line)
            if found:
                name = _clean_name(found.group(1))
                if name:
                    return name
    up = _clean_name(str(meta.get("uploader") or ""))
    parts = up.split()
    if len(parts) == 2 and all(p[:1].isupper() for p in parts):
        return up
    return ""


def wrap_words(text: str, measure, max_width: float) -> list[str]:
    """Перенос по словам. Слишком длинное слово остаётся на своей строке — шрифт уменьшится."""
    words = " ".join((text or "").split()).split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        cand = f"{cur} {word}"
        if measure(cand) <= max_width:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def fit_title(text: str, font_path: Path, max_width: int, max_height: int,
              max_size: int, min_size: int = 28, max_lines: int = 4) -> tuple[list[str], int]:
    """Самый крупный размер, при котором все слова влезают в прямоугольник."""
    _, _, ImageFont = _pil()
    text = " ".join((text or "").split())
    if not text:
        return [], min_size
    max_width = max(int(max_width), 8)
    max_height = max(int(max_height), 8)
    lo, hi = int(min_size), max(int(min_size), int(max_size))
    best_lines, best_size = [text], lo
    while lo <= hi:
        size = (lo + hi) // 2
        font = ImageFont.truetype(str(font_path), size)
        lines = wrap_words(text, font.getlength, max_width)
        too_wide = any(font.getlength(line) > max_width + 0.5 for line in lines)
        line_h = max(int(round(size * 1.12)), 1)
        fits = (not too_wide) and len(lines) <= max_lines and line_h * len(lines) <= max_height
        if fits:
            best_lines, best_size = lines, size
            lo = size + 1
        else:
            hi = size - 1
    font = ImageFont.truetype(str(font_path), best_size)
    best_lines = wrap_words(text, font.getlength, max_width)
    return best_lines, best_size


def fallback_title(meta: dict, speech: str, clip_title: str, max_words: int) -> str:
    for raw in (clip_title, (meta or {}).get("title") or "", speech):
        title = clean_title(str(raw or ""), max_words)
        if len(title.split()) >= 2:
            return title
    return clean_title("Смотри ролик", max_words) or "Смотри ролик"


def invent_title(llm, settings: Settings, meta: dict, speech: str, clip_title: str,
                 log: Log, override: str | None = None) -> str:
    limit = settings.cover.max_words
    if override:
        return clean_title(override, limit) or fallback_title(meta, speech, clip_title, limit)
    if llm is not None:
        from konveyer.llm import chat_json, set_stage
        set_stage(llm, "cover")
        system = TITLE_SYSTEM.format(group=settings.post.group, max_words=limit)
        user = (f"{meta_block(meta) or '(нет названия)'}\n\n"
                f"Заголовок клипа: {clip_title or '-'}\n\nРечь:\n{speech or '-'}")
        try:
            data = chat_json(llm, system, user, temperature=0.7)
            raw = data.get("title") if isinstance(data, dict) else ""
            title = clean_title(str(raw or ""), limit)
            if title:
                return title
        except Exception as e:
            log.warn(f"заголовок обложки через DeepSeek не вышел ({str(e)[:160]})")
    return fallback_title(meta, speech, clip_title, limit)


def invent_prompt(llm, meta: dict, speech: str, title: str, log: Log) -> str:
    prompt = ""
    if llm is not None:
        from konveyer.llm import chat_json, set_stage
        set_stage(llm, "cover")
        user = (f"{meta_block(meta) or '(нет названия)'}\nЗаголовок обложки: {title}\n\n"
                f"Речь:\n{(speech or '-')[:1500]}")
        try:
            data = chat_json(llm, PROMPT_SYSTEM, user, temperature=0.7)
            prompt = " ".join(str((data or {}).get("prompt") or "").split()) if isinstance(data, dict) else ""
        except Exception as e:
            log.warn(f"промпт картинки через DeepSeek не вышел ({str(e)[:160]})")
    if len(prompt) < 8:
        prompt = "Abstract cinematic background about the video topic, bold colors, simple shapes"
    return (prompt[:480] + SAFE_PROMPT).strip()


def _speech(tr, start: float, end: float, limit: int = 1800) -> str:
    if tr is None or getattr(tr, "empty", True) or not getattr(tr, "reliable", True):
        return ""
    parts = []
    for seg in tr.segments:
        if seg.end <= start or seg.start >= end or not seg.text.strip():
            continue
        parts.append(seg.text.strip())
    return " ".join(parts)[:limit]


# ------------------------------------------------------------------ кадр

def laplacian_variance(frame: np.ndarray) -> float:
    """Дисперсия лапласиана. Мало — кадр размыт."""
    gray = frame.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        center = gray[1:-1, 1:-1]
        lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center)
        return float(lap.var())


def _eye_patches(frame: np.ndarray, box: tuple[float, float, float, float]) -> list[np.ndarray]:
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int((x + bw) * w), int((y + bh) * h)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(max(x1, x0 + 1), w), min(max(y1, y0 + 1), h)
    face = frame[y0:y1, x0:x1]
    fh, fw = face.shape[:2]
    if fh < 12 or fw < 12:
        return []
    band = face[int(fh * 0.22):int(fh * 0.50), :]
    if band.size < 16:
        return []
    left = band[:, int(fw * 0.12):int(fw * 0.42)]
    right = band[:, int(fw * 0.58):int(fw * 0.88)]
    return [p for p in (left, right) if p.size >= 16]


def eye_open_score(frame: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Выше, если в полосе глаз есть тёмные зрачки и резкие края, а не гладкая кожа."""
    patches = _eye_patches(frame, box)
    if not patches:
        return 0.0
    sharp = 0.0
    dark = 0.0
    for patch in patches:
        gray = patch.mean(axis=2) if patch.ndim == 3 else patch
        sharp += laplacian_variance(patch)
        dark += float((gray < 70).mean())
    sharp_n = min(sharp / (sharp + 50.0), 1.0)
    dark_n = min(dark / len(patches) * 3.0, 1.0)
    return float(0.5 * sharp_n + 0.5 * dark_n)


def score_frame(frame: np.ndarray, boxes: list[tuple[float, float, float, float]] | None) -> float:
    """Крупное резкое лицо с открытыми глазами обходит размытый кадр и кадр без лица."""
    if frame is None or getattr(frame, "size", 0) == 0 or is_black(frame):
        return -1.0
    sharp = laplacian_variance(frame)
    boxes = boxes or []
    if not boxes:
        return sharp * 0.02
    box = max(boxes, key=lambda b: max(b[2], 0.0) * max(b[3], 0.0))
    area = max(box[2], 0.0) * max(box[3], 0.0)
    if area < 0.012:
        return sharp * 0.02
    eyes = eye_open_score(frame, box)
    cut = 0.35 if face_cut(box) else 1.0
    return float(area * (1.0 + sharp) * (0.15 + eyes) * cut)


def pick_best(frames: list[np.ndarray],
              boxes: list[list[tuple[float, float, float, float]]] | None = None) -> int:
    if not frames:
        raise KonveyerError("нет кадров для обложки")
    groups = boxes if boxes is not None else [[] for _ in frames]
    if len(groups) != len(frames):
        raise KonveyerError("список лиц не совпал с числом кадров")
    scores = [score_frame(frame, group) for frame, group in zip(frames, groups)]
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def sample_times(start: float, end: float, count: int) -> list[float]:
    start, end = float(start), float(end)
    count = max(int(count), 1)
    span = end - start
    if span <= 0.3:
        return [round(start, 3)]
    margin = min(0.35, span * 0.08)
    a, b = start + margin, end - margin
    if b <= a:
        return [round(start + span / 2, 3)]
    if count == 1:
        return [round((a + b) / 2, 3)]
    return [round(a + (b - a) * i / (count - 1), 3) for i in range(count)]


def grab_frames(video: Path, start: float, end: float, count: int,
                src_w: int, src_h: int, target_w: int = 1280) -> list[tuple[float, np.ndarray]]:
    """Несколько кадров по таймкодам. Ищем по ним, а не по всему ролику."""
    times = sample_times(start, end, count)
    src_w = max(int(src_w), 2)
    src_h = max(int(src_h), 2)
    width = min(src_w, max(int(target_w), 640))
    width -= width % 2
    height = int(round(src_h * width / src_w / 2)) * 2
    height = max(height, 2)
    size = width * height * 3
    out: list[tuple[float, np.ndarray]] = []
    for stamp in times:
        cmd = [need_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
               "-ss", f"{stamp:.3f}", "-i", str(video), "-frames:v", "1",
               "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw = proc.stdout or b""
        if len(raw) < size:
            continue
        frame = np.frombuffer(raw[:size], np.uint8).reshape(height, width, 3).copy()
        out.append((stamp, frame))
    return out


def _window(src_w: int, src_h: int, tw: int, th: int, want_h: float) -> tuple[float, float]:
    aspect = tw / th
    if src_w / max(src_h, 1) >= aspect:
        max_h = float(src_h)
    else:
        max_h = src_w / aspect
    max_h = max(max_h, 2.0)
    window_h = min(max(want_h, max_h * 0.42), max_h)
    return window_h * aspect, window_h


def crop_cover(frame: np.ndarray, box: tuple[float, float, float, float] | None,
               tw: int, th: int) -> tuple[np.ndarray, tuple[float, float, float, float] | None, str]:
    """Кадр нужного размера. На широкой обложке лицо сбоку, на вертикальной — ниже текста."""
    Image, _, _ = _pil()
    src_h, src_w = frame.shape[:2]
    vertical = th > tw
    if box is None:
        win_w, win_h = _window(src_w, src_h, tw, th, float(src_h))
        left = max((src_w - win_w) / 2, 0)
        top = max((src_h - win_h) / 2, 0)
        image = _cut(frame, left, top, win_w, win_h, tw, th, Image)
        return image, None, "top" if vertical else "left"

    x, y, bw, bh = box
    x = min(max(float(x), 0.0), 0.98)
    y = min(max(float(y), 0.0), 0.98)
    bw = min(max(float(bw), 0.02), 1.0 - x)
    bh = min(max(float(bh), 0.02), 1.0 - y)
    fx, fy, fw, fh = x * src_w, y * src_h, bw * src_w, bh * src_h
    face_cx, face_cy = fx + fw / 2, fy + fh / 2
    if vertical:
        want = fh / 0.34
        target_cx, target_cy = 0.50 * tw, 0.64 * th
        side = "top"
    else:
        want = fh / 0.68
        if face_cx < src_w * 0.5:
            target_cx, side = 0.30 * tw, "right"
        else:
            target_cx, side = 0.70 * tw, "left"
        target_cy = 0.48 * th
    win_w, win_h = _window(src_w, src_h, tw, th, want)
    left = face_cx - (target_cx / tw) * win_w
    top = face_cy - (target_cy / th) * win_h
    left = min(max(left, 0.0), max(src_w - win_w, 0.0))
    top = min(max(top, 0.0), max(src_h - win_h, 0.0))
    image = _cut(frame, left, top, win_w, win_h, tw, th, Image)
    out_box = ((fx - left) / win_w, (fy - top) / win_h, fw / win_w, fh / win_h)
    out_box = tuple(float(min(max(v, 0.0), 1.5)) for v in out_box)
    # Сторона текста — по тому, где лицо оказалось после обрезки.
    cx = out_box[0] + out_box[2] / 2
    if not vertical:
        side = "right" if cx < 0.5 else "left"
    return image, out_box, side


def _cut(frame, left, top, win_w, win_h, tw, th, Image) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    x0 = int(round(min(max(left, 0.0), src_w - 2)))
    y0 = int(round(min(max(top, 0.0), src_h - 2)))
    x1 = int(round(min(max(left + win_w, x0 + 2), src_w)))
    y1 = int(round(min(max(top + win_h, y0 + 2), src_h)))
    img = Image.fromarray(np.ascontiguousarray(frame[:, :, :3]), "RGB")
    cropped = img.crop((x0, y0, x1, y1))
    return np.asarray(cropped.resize((int(tw), int(th)), Image.Resampling.LANCZOS))


def punch(img: np.ndarray) -> np.ndarray:
    """Чуть больше контраста и насыщенности, чтобы кадр не выглядел серым."""
    f = img.astype(np.float32)
    mean = f.mean(axis=(0, 1), keepdims=True)
    f = (f - mean) * 1.18 + mean
    gray = f.mean(axis=2, keepdims=True)
    f = gray + (f - gray) * 1.32
    return np.clip(f, 0, 255).astype(np.uint8)


def shade(img: np.ndarray, side: str) -> np.ndarray:
    """Затемнение с той стороны, где будет текст."""
    f = img.astype(np.float32)
    h, w = f.shape[:2]
    if side == "right":
        ramp = np.linspace(0.0, 0.72, w, dtype=np.float32)[None, :]
    elif side == "top":
        ramp = np.linspace(0.70, 0.05, h, dtype=np.float32)[:, None]
    else:
        ramp = np.linspace(0.72, 0.0, w, dtype=np.float32)[None, :]
    f *= (1.0 - ramp[..., None] * 0.82)
    return np.clip(f, 0, 255).astype(np.uint8)


def text_region(width: int, height: int,
                face: tuple[float, float, float, float] | None,
                side: str) -> tuple[int, int, int, int, str]:
    """Прямоугольник под заголовок. На вертикали — над лицом, на широкой — сбоку от него."""
    margin = max(int(width * 0.045), 16)
    if side == "top" or height > width:
        top_h = int(height * 0.32)
        if face is not None:
            face_top = int(face[1] * height)
            top_h = min(top_h, max(face_top - margin, int(height * 0.12)))
        return margin, margin, width - 2 * margin, max(top_h, int(height * 0.12)), "center"
    col_w = int(width * 0.44)
    if face is not None:
        fx = face[0] * width
        fw = face[2] * width
        if side == "right":
            x = int(min(max(fx + fw + margin * 0.4, width * 0.50), width - col_w - margin))
        else:
            x = margin
            col_w = int(min(col_w, max(fx - margin * 1.4, width * 0.28)))
    else:
        x = margin
    x = int(min(max(x, margin), width - margin - 40))
    col_w = int(min(col_w, width - x - margin))
    align = "right" if side == "right" else "left"
    return x, margin, max(col_w, 40), height - 2 * margin, align


def _stroke(size: int) -> int:
    return max(4, min(16, int(size) // 8))


def compose_cover(frame: np.ndarray, title: str, speaker: str, font_path: Path, *,
                  text_color: str = "#FFFFFF", accent_color: str = "#FFE14A",
                  stroke_color: str = "#000000",
                  face: tuple[float, float, float, float] | None = None,
                  side: str = "left") -> object:
    """Картинка обложки. Текст не вылезает за прямоугольник и не садится на лицо."""
    Image, ImageDraw, ImageFont = _pil()
    height, width = frame.shape[:2]
    base = shade(punch(frame), side if side in ("left", "right", "top") else "left")
    img = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(img)
    shown = display_title(title)
    x, y, box_w, box_h, align = text_region(width, height, face, side)
    max_size = 92 if height > width else 78
    lines, size = fit_title(shown, font_path, box_w, box_h, max_size, min_size=26, max_lines=4)
    if lines:
        font = ImageFont.truetype(str(font_path), size)
        stroke = _stroke(size)
        line_h = int(round(size * 1.12))
        cursor = y
        for line in lines:
            if cursor + line_h > y + box_h:
                break
            line_w = font.getlength(line)
            if align == "right":
                lx = x + box_w - line_w
            elif align == "center":
                lx = x + (box_w - line_w) / 2
            else:
                lx = x
            draw.text((lx, cursor), line, font=font, fill=text_color,
                      stroke_width=stroke, stroke_fill=stroke_color)
            cursor += line_h
        _badge(draw, ImageFont, font_path, speaker, x, cursor + 8, box_w, align,
               accent_color, y + box_h)
    return img


def _badge(draw, ImageFont, font_path, speaker, x, y, box_w, align, accent, limit) -> None:
    name = " ".join(str(speaker or "").split())
    if not name:
        return
    size = 28
    font = ImageFont.truetype(str(font_path), size)
    while size > 16 and font.getlength(name) > box_w - 16:
        size -= 2
        font = ImageFont.truetype(str(font_path), size)
    if y + size + 14 > limit:
        return
    pad_x, pad_y = 10, 4
    text_w = font.getlength(name)
    bw, bh = text_w + pad_x * 2, size + pad_y * 2
    if align == "right":
        bx = x + box_w - bw
    elif align == "center":
        bx = x + (box_w - bw) / 2
    else:
        bx = x
    rect = [bx, y, bx + bw, y + bh]
    try:
        draw.rounded_rectangle(rect, radius=8, fill=accent)
    except Exception:
        draw.rectangle(rect, fill=accent)
    draw.text((bx + pad_x, y + pad_y - 1), name, font=font, fill="#111111")


def save_jpeg(img, path: Path, max_bytes: int = JPEG_MAX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = img.convert("RGB")
    for quality in (90, 82, 74, 64, 54, 44):
        buf = BytesIO()
        image.save(buf, "JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            path.write_bytes(buf.getvalue())
            return
    path.write_bytes(buf.getvalue())


def telegram_thumb(src: Path) -> Path | None:
    """JPEG до 320 px и 200 КБ — лимит превью у sendVideo. Полная обложка уходит отдельно."""
    if src is None or not Path(src).is_file():
        return None
    dest = Path(src).with_name(Path(src).stem + "_tg.jpg")
    try:
        Image, _, _ = _pil()
        img = Image.open(src).convert("RGB")
        img.thumbnail((THUMB_MAX, THUMB_MAX), Image.Resampling.LANCZOS)
        for quality in (85, 70, 55, 40):
            buf = BytesIO()
            img.save(buf, "JPEG", quality=quality, optimize=True)
            if buf.tell() <= THUMB_BYTES:
                dest.write_bytes(buf.getvalue())
                return dest
    except Exception:
        return None
    return None


# ------------------------------------------------------------------ нейросеть

def pollinations_url(base: str, prompt: str, width: int, height: int) -> str:
    text = " ".join((prompt or "").split()) or "abstract background"
    root = base.rstrip("/")
    return f"{root}/{quote(text, safe='')}?width={int(width)}&height={int(height)}&nologo=true"


class PollinationsProvider:
    """Бесплатная картинка с image.pollinations.ai. Сессия подменяется в тестах."""

    def __init__(self, base_url: str = "https://image.pollinations.ai/prompt",
                 timeout: float = 40.0, attempts: int = 3, session=None, sleep=None):
        self.base_url = base_url
        self.timeout = float(timeout)
        self.attempts = max(int(attempts), 1)
        self._session = session
        self._sleep = sleep or time.sleep

    def _http(self):
        if self._session is not None:
            return self._session
        import requests
        self._session = requests.Session()
        return self._session

    def generate(self, prompt: str, width: int, height: int) -> bytes:
        url = pollinations_url(self.base_url, prompt, width, height)
        last: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                resp = self._http().get(url, timeout=self.timeout)
                code = int(getattr(resp, "status_code", 200))
                data = getattr(resp, "content", b"") or b""
                if code != 200 or len(data) < 500:
                    raise RuntimeError(f"ответ {code}, байт {len(data)}")
                return data
            except Exception as e:
                last = e
                if attempt < self.attempts:
                    self._sleep(min(2 ** (attempt - 1), 8))
        raise KonveyerError(f"картинка не сгенерировалась: {last}")


def make_image_provider(settings: Settings, session=None, sleep=None) -> ImageProvider:
    name = (settings.cover.provider or "").strip()
    if name == "pollinations":
        return PollinationsProvider(settings.cover.pollinations_url, settings.cover.ai_timeout,
                                    settings.cover.ai_attempts, session=session, sleep=sleep)
    raise KonveyerError(f"неизвестный провайдер обложки: {name or '(пусто)'}")


def decode_image(data: bytes) -> np.ndarray:
    Image, _, _ = _pil()
    try:
        img = Image.open(BytesIO(data))
        img.load()
        return np.asarray(img.convert("RGB"))
    except Exception as e:
        raise RuntimeError(f"это не картинка ({e})") from e


def fit_canvas(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    return crop_cover(frame, None, tw, th)[0]


# ------------------------------------------------------------------ задание

def cover_size(full: bool) -> tuple[int, int]:
    return WIDE if full else TALL


def cover_rels(clip: dict, full: bool) -> dict[str, str]:
    if full:
        return {"frame": "full/cover_frame.jpg", "ai": "full/cover_ai.jpg"}
    n = int(clip["id"])
    stem = f"clip_{n:02d}_cover"
    return {"frame": f"clips/clip_{n:02d}/{stem}_frame.jpg",
            "ai": f"clips/clip_{n:02d}/{stem}_ai.jpg"}


def cover_files(job: Job, clip: dict) -> list[tuple[str, Path]]:
    labels = {"frame": "кадр", "ai": "нейросеть"}
    found = []
    for kind, label in labels.items():
        rel = str((clip.get("covers") or {}).get(kind) or "")
        if not rel:
            continue
        path = job.dir / rel
        if path.is_file():
            found.append((label, path))
    return found


def _video_span(job: Job, clip: dict) -> tuple[Path, float, float]:
    start, end = float(clip.get("start") or 0), float(clip.get("end") or 0)
    src = job.source_video
    if src is not None and src.is_file() and end - start >= 0.3:
        return src, start, end
    rel = str(clip.get("file") or "")
    if rel and (job.dir / rel).is_file():
        return job.dir / rel, 0.0, max(end - start, 0.4)
    raise KonveyerError("нет видео, из которого взять кадр обложки")


def _only(only) -> set[int] | None:
    if only is None:
        return None
    if isinstance(only, int):
        return {only}
    return {int(x) for x in only}


def make_covers(job: Job, settings: Settings, log: Log, llm=None, tr=None, *,
                title: str | None = None, only=None, grabber=None,
                provider: ImageProvider | None = None, finder=None) -> None:
    """Обложки для каждого клипа. --full — 1280×720, Shorts — 1080×1920."""
    if not settings.cover.enabled:
        log.info("обложки выключены")
        return
    clips = list(job.data.get("clips") or [])
    wanted = _only(only)
    if wanted is not None:
        clips = [c for c in clips if int(c.get("id", -1)) in wanted]
    if not clips:
        raise KonveyerError("нет клипов для обложки — сначала рендер")
    font = resolve_font(settings, log)
    mode = settings.cover.mode
    full = bool((job.data.get("options") or {}).get("full"))
    own_finder = finder is None
    finder = finder if finder is not None else FaceFinder()
    if provider is None and mode in ("ai", "both"):
        provider = make_image_provider(settings)
    meta = job.data.get("meta") or {}
    speaker = guess_speaker(meta)
    try:
        for clip in clips:
            _one_cover(job, settings, log, llm, tr, clip, font, mode, full, speaker,
                       title, grabber, provider, finder, meta)
    finally:
        if own_finder:
            finder.close()
    job.save()


def _one_cover(job, settings, log, llm, tr, clip, font, mode, full, speaker, title,
               grabber, provider, finder, meta) -> None:
    clip_full = full or bool(clip.get("full"))
    tw, th = cover_size(clip_full)
    rels = cover_rels(clip, clip_full)
    video, start, end = _video_span(job, clip)
    meta_w = int(meta.get("width") or 0)
    meta_h = int(meta.get("height") or 0)
    if grabber is not None:
        grabbed = grabber(video, start, end, settings.cover.candidates)
    else:
        grabbed = grab_frames(video, start, end, settings.cover.candidates,
                              meta_w or 1280, meta_h or 720, tw)
    if not grabbed:
        raise KonveyerError(f"не прочитал кадры для обложки клипа {clip.get('id')}")
    frames = [frame for _, frame in grabbed]
    boxes = []
    for frame in frames:
        try:
            found = list(finder.boxes(frame)) if finder is not None and finder.available else []
        except Exception:
            found = []
        boxes.append(found)
    best = pick_best(frames, boxes)
    stamp, chosen = grabbed[best]
    face_boxes = boxes[best]
    face = max(face_boxes, key=lambda b: b[2] * b[3]) if face_boxes else None
    cropped, face_out, side = crop_cover(chosen, face, tw, th)
    speech = _speech(tr, start, end)
    heading = invent_title(llm, settings, meta, speech, str(clip.get("title") or ""), log, title)
    if speaker:
        log.info(f"обложка клипа {clip.get('id')}: спикер «{speaker}»")
    produced: dict[str, str] = {}
    fallback = False
    want_frame = mode in ("frame", "both")
    want_ai = mode in ("ai", "both")
    if want_frame:
        path = job.dir / rels["frame"]
        save_jpeg(_paint(cropped, heading, speaker, font, settings, face_out, side), path,
                  int(settings.cover.jpeg_max_mb * 1024 * 1024))
        produced["frame"] = rels["frame"]
        log.info(f"обложка клипа {clip.get('id')}: кадр {tw}×{th}, «{display_title(heading)}», "
                 f"{path.name}")
    if want_ai:
        painted = _ai_background(provider, llm, meta, speech, heading, tw, th, log, finder)
        if painted is None:
            fallback = True
            if "frame" not in produced:
                path = job.dir / rels["frame"]
                save_jpeg(_paint(cropped, heading, speaker, font, settings, face_out, side), path,
                          int(settings.cover.jpeg_max_mb * 1024 * 1024))
                produced["frame"] = rels["frame"]
            log.info(f"обложка клипа {clip.get('id')}: нейросеть не ответила, оставляю кадр")
        else:
            ai_face, ai_side = _avoid_face(painted, finder)
            path = job.dir / rels["ai"]
            save_jpeg(_paint(painted, heading, speaker, font, settings, ai_face, ai_side), path,
                      int(settings.cover.jpeg_max_mb * 1024 * 1024))
            produced["ai"] = rels["ai"]
            log.info(f"обложка клипа {clip.get('id')}: нейросеть {tw}×{th}, {path.name}")
    for kind, rel in rels.items():
        stale = job.dir / rel
        if kind not in produced and stale.is_file():
            stale.unlink()
    clip["covers"] = {"title": heading, "speaker": speaker, "time": round(stamp, 2),
                      "frame": produced.get("frame", ""), "ai": produced.get("ai", ""),
                      "fallback": fallback}


def _paint(image, heading, speaker, font, settings, face, side):
    return compose_cover(image, heading, speaker, font, text_color=settings.cover.text_color,
                         accent_color=settings.cover.accent_color,
                         stroke_color=settings.cover.stroke_color, face=face, side=side)


def _ai_background(provider, llm, meta, speech, heading, tw, th, log, finder):
    if provider is None:
        return None
    prompt = invent_prompt(llm, meta, speech, heading, log)
    try:
        data = provider.generate(prompt, tw, th)
        return fit_canvas(decode_image(data), tw, th)
    except Exception as e:
        log.warn(f"генерация обложки не удалась ({str(e)[:180]})")
        return None


def _avoid_face(image, finder):
    boxes = []
    if finder is not None and getattr(finder, "available", False):
        try:
            boxes = list(finder.boxes(image))
        except Exception:
            boxes = []
    if not boxes:
        return None, "top" if image.shape[0] > image.shape[1] else "left"
    face = max(boxes, key=lambda b: b[2] * b[3])
    side = "top" if image.shape[0] > image.shape[1] else ("right" if face[0] + face[2] / 2 < 0.5 else "left")
    return face, side
