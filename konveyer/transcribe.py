"""Шаг 2: распознавание речи faster-whisper с таймингом каждого слова.

CUDA есть (ПК с видеокартой NVIDIA) — large-v3 / float16. Нет (ноут) — small / int8 на CPU.
Если CUDA есть, но падает (например, нет ядер под Blackwell) — сами доделываем на CPU.
"""
from __future__ import annotations

import subprocess
import time
from typing import Protocol

import numpy as np

from konveyer.config import Settings
from konveyer.hardware import cuda_devices
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.speech import SKIP_CAPTIONS, assess_speech
from konveyer.transcript import Segment, Transcript, Word
from konveyer.util import KonveyerError, fmt_time, need_tool, write_json

SAMPLE_RATE = 16000


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def load_audio(path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Звук моно float32 через ffmpeg (встроенный декодер faster-whisper ломается на новых PyAV)."""
    try:
        proc = subprocess.run(
            [need_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        raise KonveyerError(f"не удалось запустить ffmpeg для звука ({e})") from e
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[-600:]
        raise KonveyerError(f"ffmpeg не смог достать звук:\n{err}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def pick_device(settings: Settings) -> tuple[str, str, str]:
    w = settings.whisper
    if w.device == "cpu":
        return "cpu", w.model_cpu, w.compute_cpu
    if w.device == "cuda" or cuda_devices() > 0:
        return "cuda", w.model_gpu, w.compute_gpu
    return "cpu", w.model_cpu, w.compute_cpu


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, duration: float, log: Log) -> Transcript: ...


class WhisperTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _run(self, audio, device, model_name, compute, duration, log) -> Transcript:
        from faster_whisper import WhisperModel

        log.info(f"модель {model_name} на {device.upper()} ({compute}); первый запуск скачает её")
        model = WhisperModel(model_name, device=device, compute_type=compute)
        w = self.settings.whisper
        segments, meta = model.transcribe(
            audio, language=w.language or None, vad_filter=w.vad,
            word_timestamps=True, beam_size=w.beam_size)
        prob = float(getattr(meta, "language_probability", 0) or 0)
        log.info(f"язык речи: {meta.language} ({prob:.0%})")
        out: list[Segment] = []
        last = 0.0
        for seg in segments:
            words = [Word(round(x.start, 3), round(x.end, 3), x.word) for x in (seg.words or [])]
            out.append(Segment(round(seg.start, 3), round(seg.end, 3), (seg.text or "").strip(), words,
                               _num(getattr(seg, "no_speech_prob", None)),
                               _num(getattr(seg, "avg_logprob", None)),
                               _num(getattr(seg, "compression_ratio", None))))
            if time.monotonic() - last > 5:
                last = time.monotonic()
                log.info(f"распознано {fmt_time(seg.end)} из {fmt_time(duration)}")
        return Transcript(meta.language or "", out, language_probability=_num(prob), vad=bool(w.vad))

    def transcribe(self, audio, duration, log) -> Transcript:
        device, model_name, compute = pick_device(self.settings)
        if device == "cuda":
            try:
                return self._run(audio, device, model_name, compute, duration, log)
            except Exception as e:
                log.warn(f"CUDA не сработала ({e}); перехожу на CPU")
            w = self.settings.whisper
            device, model_name, compute = "cpu", w.model_cpu, w.compute_cpu
        else:
            log.info("CUDA не найдена — распознаю на процессоре (медленнее, модель меньше)")
        try:
            return self._run(audio, device, model_name, compute, duration, log)
        except Exception as e:
            raise KonveyerError(f"faster-whisper не смог распознать речь: {e}") from e


def judge_transcript(job: Job, tr: Transcript, settings: Settings, log: Log,
                     empty_reason: str = "") -> Transcript:
    """Пометить, можно ли верить расшифровке, и записать это в задание."""
    if empty_reason or tr.empty:
        tr.reliable = False
        tr.speech_reason = empty_reason or tr.speech_reason or "пустая расшифровка"
    else:
        verdict = assess_speech(tr, float(job.data["meta"].get("duration") or 0), settings.speech)
        tr.reliable = verdict.ok
        tr.speech_reason = verdict.reason
    meta = job.data["meta"]
    meta["language"] = tr.language
    meta["language_probability"] = tr.language_probability
    meta["speech_ok"] = tr.reliable
    meta["speech_reason"] = tr.speech_reason
    if not tr.reliable and settings.render.captions and not settings.speech.force:
        meta["speech_note"] = SKIP_CAPTIONS
        tail = f" ({tr.speech_reason})" if tr.speech_reason else ""
        log.info(SKIP_CAPTIONS + tail)
    elif not tr.reliable and settings.speech.force and settings.render.captions:
        meta["speech_note"] = ""
        log.info(f"речь ненадёжна ({tr.speech_reason}) — субтитры включены принудительно")
    elif tr.empty or not tr.reliable:
        meta["speech_note"] = ""
        log.info("речи нет")
    else:
        meta["speech_note"] = ""
        log.info(f"слов: {len(tr.words)}")
    write_json(job.transcript_file, tr.to_dict())
    job.save()
    return tr


def transcribe_job(job: Job, settings: Settings, log: Log,
                   transcriber: Transcriber | None = None, diarizer=None) -> Transcript:
    meta = job.data["meta"]
    empty_reason = ""
    audio = None
    if not meta.get("has_audio"):
        log.info("звука нет — распознавать нечего")
        tr = Transcript()
        empty_reason = "звука нет"
    else:
        audio = load_audio(job.source_video)
        if audio.size == 0 or float(np.abs(audio).max(initial=0.0)) < 1e-4:
            log.info("звуковая дорожка пустая или тишина")
            tr = Transcript()
            empty_reason = "тишина"
            audio = None
        else:
            tr = (transcriber or WhisperTranscriber(settings)).transcribe(audio, meta["duration"], log)
    tr = judge_transcript(job, tr, settings, log, empty_reason)
    if audio is not None and settings.subs.speakers and not tr.empty:
        from konveyer.speakers import attach_speakers
        attach_speakers(job, settings, log, tr, audio, SAMPLE_RATE, diarizer=diarizer)
    return tr
