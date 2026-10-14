"""Конвейер целиком: шаги по порядку, готовые пропускаются, ошибка — с названием шага."""
from __future__ import annotations

import copy
import time

from konveyer.analysis import analyze_job
from konveyer.config import Settings, apply
from konveyer.ingest import ingest
from konveyer.jobs import STEP_NAMES, STEPS, Job
from konveyer.llm import LLM, make_llm
from konveyer.log import Log
from konveyer.moments import pick_moments
from konveyer.post import make_post
from konveyer.render import render_job
from konveyer.transcribe import Transcriber, transcribe_job
from konveyer.transcript import Transcript
from konveyer.util import KonveyerError, read_json

# ключ в job.options -> настройка
OPTION_MAP = {
    "count": "moments.count", "min": "moments.min_sec", "max": "moments.max_sec",
    "llm": "moments.use_llm", "layout": "render.layout", "tighten": "render.tighten",
    "captions": "render.captions", "titles": "render.titles",
    "translate": "render.translate", "karaoke": "render.karaoke",
    "device": "whisper.device", "force_captions": "speech.force",
    "voice": "voice.enabled", "voice_speaker": "voice.speaker", "voice_engine": "voice.engine",
    "voice_sample": "voice.sample", "voice_subs": "voice.subs", "separate": "voice.separate",
    "zoom": "render.zoom", "subs_style": "subs.style", "speakers": "subs.speakers",
    "per_speaker": "voice.per_speaker",
    "score": "quality.score", "vision_check": "quality.vision",
    "voice_check": "voice.voice_check",
    "voice_mode": "voice.voice_mode",
    "cover": "cover.enabled",
    "cover_mode": "cover.mode",
    "zoom_center": "render.zoom_center",
}

_UNSET = object()


def settings_for_job(settings: Settings, job: Job) -> Settings:
    s = copy.deepcopy(settings)
    for key, value in (job.data.get("options") or {}).items():
        if key == "voice_flat":
            if value:
                s.voice.expression["enabled"] = False
            continue
        if key in OPTION_MAP and value is not None and value != "":
            section, _, name = OPTION_MAP[key].partition(".")
            apply(s, section, name, value)
    return s


def load_transcript(job: Job) -> Transcript:
    return Transcript.from_dict(read_json(job.transcript_file, None))


class Pipeline:
    def __init__(self, job: Job, settings: Settings, log: Log | None = None,
                 llm=_UNSET, transcriber: Transcriber | None = None, encoder=None):
        self.job = job
        self.s = settings_for_job(settings, job)
        self.log = log or Log(job.log_file)
        self._llm = llm
        self.transcriber = transcriber
        self.encoder = encoder

    @property
    def llm(self) -> LLM | None:
        if self._llm is _UNSET:
            self._llm = make_llm(self.s, self.log, self.s.moments.use_llm, job=self.job)
        return self._llm

    def _flush_usage(self, final: bool = False) -> None:
        llm = None if self._llm is _UNSET else self._llm
        book = getattr(llm, "book", None)
        if book is not None:
            book.flush(final=final)

    def rerender_full(self) -> None:
        from konveyer.fullvideo import render_full
        render_full(self.job, self.s, self.log, load_transcript(self.job), self.llm,
                    encoder=self.encoder)

    def _step(self, name: str) -> None:
        job, s, log = self.job, self.s, self.log
        if name == "ingest":
            ingest(job, s, log)
        elif name == "transcribe":
            transcribe_job(job, s, log, self.transcriber)
        elif name == "analyze":
            analyze_job(job, s, log, load_transcript(job))
        elif name == "moments":
            if (job.data.get("options") or {}).get("manual"):
                from konveyer.manual import manual_moments, spans_of
                manual_moments(job, s, log, self.llm, load_transcript(job), spans_of(job))
            elif (job.data.get("options") or {}).get("full"):
                from konveyer.fullvideo import plan_span
                plan_span(job, s, log, load_transcript(job))
            else:
                pick_moments(job, s, log, self.llm, load_transcript(job), read_json(job.analysis_file, {}))
        elif name == "render":
            if (job.data.get("options") or {}).get("full"):
                job.data.pop("guard", None)
                self.rerender_full()
                if s.voice.enabled and (job.dir / "full" / "voice.json").is_file():
                    from konveyer.guard import guard_full
                    guard_full(job, s, log)       # только отчёт, ничего не переделывает
            else:
                render_job(job, s, log, load_transcript(job), read_json(job.analysis_file, {}),
                           self.llm, encoder=self.encoder)
        elif name == "cover":
            from konveyer.cover import make_covers
            make_covers(job, s, log, self.llm, load_transcript(job))
        elif name == "post":
            if (job.data.get("options") or {}).get("full"):
                from konveyer.fullvideo import make_full_post
                make_full_post(job, s, log, self.llm, load_transcript(job))
            else:
                make_post(job, s, log, self.llm, load_transcript(job))
        else:
            raise KonveyerError(f"неизвестный шаг: {name}")

    def run(self, steps=STEPS, force: tuple = (), send: bool = False,
            force_send: bool = False, resend: bool = False) -> Job:
        job, log = self.job, self.log
        force_send = force_send or bool((job.data.get("options") or {}).get("force_send"))
        current = ""
        try:
            if (job.data.get("options") or {}).get("full"):
                log.info("режим целого видео: горизонтальный 16:9, без нарезки на Shorts")
            for name in steps:
                current = name
                if job.done(name) and name not in force:
                    log.info(f"{STEP_NAMES[name]}: уже готово, пропускаю")
                    continue
                job.set_state("running", name)
                log.step(STEP_NAMES[name].capitalize())
                t0 = time.monotonic()
                self._step(name)
                sec = round(time.monotonic() - t0, 1)
                job.mark(name, "done", sec=sec)
                log.info(f"{STEP_NAMES[name]}: готово за {sec} с")
            if send:
                current = "send"
                job.set_state("running", "send")
                log.step("Отправка в Telegram")
                from konveyer.telegram import send_job
                send_job(job, self.s, log, llm=self.llm, force=force_send, resend=resend)
                if job.data.get("state") == "review":
                    self._flush_usage(final=True)
                    log.step("На проверку")
                    return job
            job.set_state("done")
            self._flush_usage(final=True)
            log.step("Готово")
            for c in job.data.get("clips", []):
                if c.get("file"):
                    log.info(f"клип {c['id']}: {job.dir / c['file']}")
                for kind in ("frame", "ai"):
                    rel = str((c.get("covers") or {}).get(kind) or "")
                    if rel:
                        log.info(f"обложка {c['id']} ({kind}): {job.dir / rel}")
            if job.post_md.exists():
                log.info(f"пост: {job.post_md}")
            return job
        except KonveyerError as e:
            if current in STEPS:
                job.mark(current, "failed", error=str(e))
            job.set_state("failed", current, str(e))
            log.info(f"ОШИБКА на шаге «{STEP_NAMES.get(current, current)}»: {e}")
            self._flush_usage(final=True)
            raise
        except Exception as e:
            msg = f"неожиданная ошибка {type(e).__name__}: {e}"
            if current in STEPS:
                job.mark(current, "failed", error=msg)
            job.set_state("failed", current, msg)
            log.info(f"ОШИБКА на шаге «{STEP_NAMES.get(current, current)}»: {msg}")
            self._flush_usage(final=True)
            raise

    def rerender(self, clip_ids: list[int]) -> Job:
        """Перерендерить отдельные клипы (после правки заголовка или границ)."""
        job, log = self.job, self.log
        job.set_state("running", "render")
        try:
            log.step(f"Перерендер клипов {', '.join(map(str, clip_ids))}")
            if (job.data.get("options") or {}).get("full"):
                from konveyer.fullvideo import render_full
                render_full(job, self.s, log, load_transcript(job), self.llm, encoder=self.encoder)
            else:
                render_job(job, self.s, log, load_transcript(job), read_json(job.analysis_file, {}),
                           self.llm, only=clip_ids, encoder=self.encoder)
            if self.s.cover.enabled:
                from konveyer.cover import make_covers
                make_covers(job, self.s, log, self.llm, load_transcript(job), only=clip_ids)
            job.set_state("done")
        except Exception as e:
            job.set_state("failed", "render", str(e))
            raise
        return job
