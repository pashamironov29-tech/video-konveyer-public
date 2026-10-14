"""Режим smart: вертикальное окно из горизонтального кадра следует за лицом.

Кадры читаются через ffmpeg (OpenCV не открывает пути с кириллицей на Windows),
лица ищет MediaPipe, запасной вариант — каскад Хаара из OpenCV. В OpenCV 5 файла
каскада в cv2/data может не быть: тогда не падаем, рендер уходит в раскладку blur.
Окно двигается только на склейках (внутри сцены стоит на месте) — так нет дёрганья.
"""
from __future__ import annotations

import os
import subprocess
from statistics import median

import numpy as np

from konveyer.util import need_tool

BLUR_FALLBACK = "детектора лиц нет (нет MediaPipe и каскада OpenCV) — раскладка blur"

SAMPLE_W = 320


def read_frames(video, start: float, duration: float, src_w: int, src_h: int,
                fps: float = 2.0) -> tuple[list[np.ndarray], float]:
    w = SAMPLE_W
    h = int(round(src_h * w / src_w / 2)) * 2
    cmd = [need_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video),
           "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    raw = proc.stdout
    frames = [np.frombuffer(raw[i:i + size], np.uint8).reshape(h, w, 3)
              for i in range(0, len(raw) - size + 1, size)]
    return frames, 1.0 / fps


def _try_mediapipe():
    try:
        import mediapipe as mp
        return mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
    except Exception:
        return None


def _try_haar():
    """Каскад Хаара, только если xml реально лежит на диске. Иначе None, без исключения."""
    try:
        import cv2
    except Exception:
        return None
    try:
        data = getattr(cv2, "data", None)
        base = getattr(data, "haarcascades", "") if data is not None else ""
        if not base:
            return None
        path = os.path.join(str(base), "haarcascade_frontalface_default.xml")
        if not os.path.isfile(path):
            return None
        clf = cv2.CascadeClassifier(path)
        if clf is None or clf.empty():
            return None
        return clf
    except Exception:
        return None


class FaceFinder:
    """Центр самого крупного лица по x (0..1) или None."""

    def __init__(self):
        self._mp = _try_mediapipe()
        self._haar = None if self._mp is not None else _try_haar()

    @property
    def available(self) -> bool:
        return self._mp is not None or self._haar is not None

    def boxes(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Прямоугольники лиц: xmin, ymin, ширина, высота, всё от 0 до 1."""
        try:
            if self._mp is not None:
                res = self._mp.process(frame)
                out = []
                for d in res.detections or []:
                    b = d.location_data.relative_bounding_box
                    out.append((float(b.xmin), float(b.ymin), float(b.width), float(b.height)))
                return out
            if self._haar is not None:
                import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
                h, w = frame.shape[:2]
                return [(float(x) / w, float(y) / h, float(fw) / w, float(fh) / h)
                        for x, y, fw, fh in faces]
        except Exception:
            return []
        return []

    def center_x(self, frame: np.ndarray) -> float | None:
        boxes = self.boxes(frame)
        if not boxes:
            return None
        x, _, w, h = max(boxes, key=lambda b: b[2] * b[3])
        return float(min(max(x + w / 2, 0.0), 1.0))

    def close(self):
        if self._mp is not None:
            try:
                self._mp.close()
            except Exception:
                pass


def crop_track(samples: list[tuple[float, float | None]], scenes: list[float], duration: float,
               src_w: int, crop_w: int) -> list[tuple[float, int]]:
    """[(начало сцены, x окна)]: на сцену — медиана положения лица, без лиц — как у соседей."""
    bounds = [0.0] + sorted(s for s in scenes if 0.3 < s < duration - 0.3) + [duration]
    max_x = max(src_w - crop_w, 0)
    center = max_x // 2
    track: list[tuple[float, int]] = []
    prev = None
    for a, b in zip(bounds, bounds[1:]):
        xs = [cx for t, cx in samples if a <= t < b and cx is not None]
        if xs:
            x = int(round(median(xs) * src_w - crop_w / 2))
        else:
            x = prev if prev is not None else None
        track.append((a, x))
        if x is not None:
            prev = x
    first = next((x for _, x in track if x is not None), center)
    out: list[tuple[float, int]] = []
    for a, x in track:
        x = first if x is None else x
        x = min(max(x, 0), max_x)
        x -= x % 2
        if out and abs(out[-1][1] - x) < src_w * 0.03:
            continue
        out.append((round(a, 3), x))
    return out or [(0.0, center - center % 2)]


def crop_x_expr(track: list[tuple[float, int]]) -> str:
    """Кусочно-постоянное x(t) для фильтра crop."""
    if len(track) == 1:
        return str(track[0][1])
    expr = str(track[-1][1])
    for (_, x), (t_next, _) in reversed(list(zip(track, track[1:]))):
        expr = f"if(lt(t,{t_next:.3f}),{x},{expr})"
    return expr


def fraction_track(samples: list[tuple[float, float | None]], scenes: list[float],
                   duration: float) -> list[tuple[float, float]]:
    """[(начало сцены, доля кадра 0..1)]. Как crop_track, но без пикселей: так зум не дёргается."""
    bounds = [0.0] + sorted(s for s in scenes if 0.3 < s < duration - 0.3) + [duration]
    track: list[tuple[float, float | None]] = []
    prev = None
    for a, b in zip(bounds, bounds[1:]):
        xs = [cx for t, cx in samples if a <= t < b and cx is not None]
        x = float(median(xs)) if xs else prev
        track.append((a, x))
        if x is not None:
            prev = x
    first = next((x for _, x in track if x is not None), 0.5)
    out: list[tuple[float, float]] = []
    for a, x in track:
        x = first if x is None else min(max(float(x), 0.0), 1.0)
        if out and abs(out[-1][1] - x) < 0.03:
            continue
        out.append((round(a, 3), round(x, 4)))
    return out or [(0.0, 0.5)]


def smooth_frac_expr(track: list[tuple[float, float]], fade: float = 0.45) -> str:
    """Доля центра: на склейке плавный переход, внутри сцены стоит."""
    if not track:
        return "0.5"
    if len(track) == 1:
        return f"{track[0][1]:.4f}"
    expr = f"{track[-1][1]:.4f}"
    for i in range(len(track) - 2, -1, -1):
        t0, x0 = track[i]
        t1, x1 = track[i + 1]
        span = max(t1 - t0, 0.0)
        f = min(fade, span)
        if f < 0.05 or abs(x0 - x1) < 1e-4:
            piece = f"{x0:.4f}"
        else:
            start = t1 - f
            piece = (f"if(lt(t,{start:.3f}),{x0:.4f},"
                     f"{x0:.4f}+({x1 - x0:.4f})*(t-{start:.3f})/{f:.3f})")
        expr = f"if(lt(t,{t1:.3f}),{piece},{expr})"
    return expr


def _cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None


def subject_x(frame: np.ndarray, prev: np.ndarray | None) -> tuple[float | None, np.ndarray, str]:
    """Центр движения или заметности по x (0..1). Лицо сюда не входит — его смотрит FaceFinder."""
    cv2 = _cv2()
    if cv2 is not None:
        try:
            return _subject_cv(frame, prev, cv2)
        except Exception:
            pass
    return _subject_np(frame, prev)


def _subject_np(frame: np.ndarray, prev: np.ndarray | None):
    gray = frame.astype(np.float32).mean(axis=2) if frame.ndim == 3 else frame.astype(np.float32)
    kind = "center"
    cx = None
    if prev is not None and prev.shape == gray.shape:
        diff = np.abs(gray - prev)
        col = (diff > 18).sum(axis=0).astype(np.float64)
        total = float(col.sum())
        if total >= gray.size * 0.008:
            xs = np.arange(col.size)
            cx = float((col * xs).sum() / total / col.size)
            kind = "motion"
    if cx is None:
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        energy = gx + gy
        h, w = energy.shape
        edge = max(h // 12, 1)
        energy[:edge, :] = 0
        energy[-edge:, :] = 0
        col = energy.sum(axis=0)
        total = float(col.sum())
        if total >= 1:
            xs = np.arange(w)
            cx = float((col * xs).sum() / total / w)
            kind = "object"
    return cx, gray, kind


def _subject_cv(frame, prev, cv2):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    kind, cx = "center", None
    if prev is not None and getattr(prev, "shape", None) == gray.shape:
        diff = cv2.absdiff(gray, prev)
        _, th = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        col = (th > 0).sum(axis=0).astype(np.float64)
        total = float(col.sum())
        if total >= th.size * 0.008:
            xs = np.arange(col.size)
            cx = float((col * xs).sum() / total / col.size)
            kind = "motion"
    if cx is None:
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        h, w = lap.shape
        edge = max(h // 12, 1)
        lap[:edge, :] = 0
        lap[-edge:, :] = 0
        col = lap.sum(axis=0)
        total = float(col.sum())
        if total >= 1:
            xs = np.arange(w)
            cx = float((col * xs).sum() / total / w)
            kind = "object"
    return cx, gray, kind


def subject_frac_expr(video, start: float, end: float, scenes_rel: list[float], src_w: int,
                      src_h: int, finder: FaceFinder | None = None) -> tuple[str, str]:
    """(выражение доли центра, «лицо» | «объект» | «центр»). Окно зума держится на этом."""
    duration = max(end - start, 0.1)
    try:
        frames, step = read_frames(video, start, duration, src_w, src_h, fps=1.0)
    except Exception:
        return "0.5", "центр"
    if not frames:
        return "0.5", "центр"
    if len(frames) > 8:
        idx = []
        for i in range(8):
            j = round(i * (len(frames) - 1) / 7)
            if not idx or idx[-1] != j:
                idx.append(j)
        frames = [frames[i] for i in idx]
        times = [i * step for i in idx]
    else:
        times = [i * step for i in range(len(frames))]
    own = finder is None
    finder = finder or FaceFinder()
    samples: list[tuple[float, float | None]] = []
    kinds: list[str] = []
    prev = None
    try:
        for t, frame in zip(times, frames):
            face = finder.center_x(frame) if finder.available else None
            if face is not None:
                samples.append((t, face))
                kinds.append("face")
                prev = None
                continue
            cx, prev, kind = subject_x(frame, prev)
            samples.append((t, cx))
            kinds.append(kind)
    finally:
        if own:
            finder.close()
    if any(k == "face" for k in kinds):
        samples = [(t, cx if k == "face" else None) for (t, cx), k in zip(samples, kinds)]
        note = "лицо"
    elif any(k in ("motion", "object") for k in kinds):
        note = "объект"
    else:
        note = "центр"
    return smooth_frac_expr(fraction_track(samples, scenes_rel, duration)), note


def face_crop_expr(video, start: float, end: float, scenes_rel: list[float], src_w: int,
                   src_h: int, crop_w: int, finder: FaceFinder | None = None) -> tuple[str, int]:
    """(выражение x, сколько кадров с лицом)."""
    frames, step = read_frames(video, start, end - start, src_w, src_h)
    own = finder is None
    finder = finder or FaceFinder()
    try:
        if not finder.available:
            return "0", 0
        samples = [(i * step, finder.center_x(f)) for i, f in enumerate(frames)]
    finally:
        if own:
            finder.close()
    found = sum(1 for _, c in samples if c is not None)
    track = crop_track(samples, scenes_rel, end - start, src_w, crop_w)
    return crop_x_expr(track), found
