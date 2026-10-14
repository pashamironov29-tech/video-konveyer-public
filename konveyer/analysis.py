"""Шаг 3: сигналы для поиска моментов без ИИ и для подгонки границ.

- громкость по окнам (смех, крики, музыка на пике);
- смены сцен (резать по склейке, а не посреди кадра) — идея PySceneDetect,
  но через фильтр ffmpeg select=scene, без лишних зависимостей;
- плотность речи (слов в секунду).
"""
from __future__ import annotations

import re
import subprocess

import numpy as np

from konveyer.config import Settings
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.transcribe import SAMPLE_RATE, load_audio
from konveyer.transcript import Transcript
from konveyer.util import need_tool, write_json

HOP = 0.5   # шаг окна, секунды

_PTS = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


def loudness_db(audio: np.ndarray, sr: int = SAMPLE_RATE, hop: float = HOP) -> list[float]:
    """RMS-громкость в дБ (0 — максимум) по окнам hop секунд."""
    n = max(int(sr * hop), 1)
    if audio.size == 0:
        return []
    usable = audio[: (audio.size // n) * n]
    if usable.size == 0:
        usable = np.pad(audio, (0, n - audio.size))
    frames = usable.reshape(-1, n).astype(np.float64)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    return [round(float(x), 2) for x in db]


def parse_scene_times(stderr: str) -> list[float]:
    """Времена склеек из вывода фильтра showinfo."""
    times = sorted({round(float(m.group(1)), 3) for m in _PTS.finditer(stderr)})
    return [t for t in times if t > 0.2]


def detect_scenes(video, threshold: float = 0.32) -> list[float]:
    cmd = [need_tool("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(video), "-an",
           "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace", timeout=3600)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_scene_times(proc.stderr or "")


def speech_density(transcript: Transcript, duration: float, hop: float = HOP) -> list[float]:
    """Слов в секунду по окнам hop."""
    bins = max(int(np.ceil(duration / hop)), 1)
    counts = np.zeros(bins)
    for w in transcript.words:
        i = min(int(((w.start + w.end) / 2) / hop), bins - 1)
        if i >= 0:
            counts[i] += 1
    return [round(float(c / hop), 2) for c in counts]


def analyze_job(job: Job, settings: Settings, log: Log, transcript: Transcript) -> dict:
    meta = job.data["meta"]
    duration = meta["duration"]
    loud: list[float] = []
    if meta.get("has_audio"):
        loud = loudness_db(load_audio(job.source_video))
    log.info("ищу смены сцен")
    scenes = detect_scenes(job.source_video)
    speech_src = transcript if transcript.reliable else Transcript()
    if not transcript.reliable and not transcript.empty:
        log.info("речь ненадёжна — плотность речи в анализе не учитываю")
    data = {"hop": HOP, "duration": duration, "loudness": loud, "scenes": scenes,
            "speech": speech_density(speech_src, duration)}
    write_json(job.analysis_file, data)
    log.info(f"склеек: {len(scenes)}, окон громкости: {len(loud)}")
    return data
