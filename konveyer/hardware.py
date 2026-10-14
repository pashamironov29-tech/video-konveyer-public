"""Что умеет железо: видеокодировщик ffmpeg и CUDA для Whisper.

Кодировщик проверяется пробным кодированием крошечного ролика: в сборке ffmpeg
h264_nvenc может быть, а видеокарты NVIDIA — нет (как на ноуте с Intel Iris Xe).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable

ENCODER_ORDER = ("h264_nvenc", "h264_qsv", "h264_amf", "libx264")
ENCODER_NAMES = {"h264_nvenc": "NVIDIA NVENC", "h264_qsv": "Intel Quick Sync",
                 "h264_amf": "AMD AMF", "libx264": "процессор (libx264)"}


@dataclass
class Encoder:
    name: str
    args: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return ENCODER_NAMES.get(self.name, self.name)

    @property
    def hardware(self) -> bool:
        return self.name != "libx264"


def encoder_args(name: str, crf: int = 20, preset: str = "medium") -> list[str]:
    """Аргументы качества: примерно одинаковое качество у всех кодировщиков."""
    if name == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", str(crf + 1), "-b:v", "0", "-pix_fmt", "yuv420p"]
    if name == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", str(crf + 2),
                "-pix_fmt", "nv12"]
    if name == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp",
                "-qp_i", str(crf), "-qp_p", str(crf + 2), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]


Runner = Callable[[list[str]], int]


def _default_runner(cmd: list[str]) -> int:
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=30).returncode
    except (OSError, subprocess.TimeoutExpired):
        return 1


_cache: dict[tuple, Encoder] = {}


def detect_encoder(preference: str = "auto", crf: int = 20, preset: str = "medium",
                   ffmpeg: str = "ffmpeg", runner: Runner | None = None) -> Encoder:
    """Первый работающий кодировщик. preference — имя или auto."""
    key = (preference, crf, preset, ffmpeg, runner is None)
    if runner is None and key in _cache:
        return _cache[key]
    runner = runner or _default_runner
    order = ENCODER_ORDER if preference == "auto" else (preference, "libx264")
    chosen = Encoder("libx264", encoder_args("libx264", crf, preset))
    for name in order:
        if name == "libx264":
            break
        args = encoder_args(name, crf, preset)
        test = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc2=size=320x240:rate=30:duration=0.3", *args, "-f", "null", "-"]
        if runner(test) == 0:
            chosen = Encoder(name, args)
            break
    if runner is _default_runner:
        _cache[key] = chosen
    return chosen


def cuda_devices() -> int:
    """Сколько CUDA-устройств видит CTranslate2 (движок faster-whisper)."""
    try:
        import torch  # noqa: F401  на Windows подтягивает cuBLAS/cuDNN из torch\lib
    except Exception:
        pass
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0
