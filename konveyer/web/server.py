"""Локальный веб-интерфейс: http://127.0.0.1:8765

Только стандартная библиотека. Задания выполняются по одному в фоновом потоке.
POST-запросы требуют заголовок X-Konveyer: 1 — так чужая страница в браузере
не сможет дёрнуть API (для такого заголовка браузер требует CORS, а его нет).
"""
from __future__ import annotations

import json
import math
import mimetypes
import queue
import re
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from konveyer.config import SUBS_STYLES, VOICE_SPEAKERS, Settings
from konveyer.envkeys import SECRET_NAMES, has_secret, mask
from konveyer.jobs import STEPS, Job, list_jobs
from konveyer.log import Log, tail
from konveyer.util import KonveyerError
from konveyer.i18n import get_lang, tr

STATIC = Path(__file__).resolve().parent / "static"
JOB_ID = re.compile(r"^[\w.\-]+$")


def num_param(data: dict, key: str, lo: float, hi: float, integer: bool = False):
    """Число из запроса: без nan/inf, в диапазоне [lo, hi], целое — только целое. Пусто — None."""
    raw = data.get(key)
    if raw in (None, ""):
        return None
    if isinstance(raw, bool):
        raise ValueError(f"{key}: ожидалось число")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: ожидалось число") from None
    if not math.isfinite(value):
        raise ValueError(f"{key}: ожидалось конечное число")
    if integer:
        if not value.is_integer():
            raise ValueError(f"{key}: нужно целое число")
        value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{key}: от {lo:g} до {hi:g}")
    return value


Task = tuple[str, str, dict]          # (id задания, вид, данные)


def default_runner(settings: Settings, job: Job, kind: str, data: dict) -> None:
    from konveyer.pipeline import Pipeline, settings_for_job

    if kind == "run":
        Pipeline(job, settings).run(send=bool(data.get("tg")))
    elif kind == "redo":
        job.invalidate_from(data.get("from", "moments"))
        Pipeline(job, settings).run(send=bool(data.get("tg")))
    elif kind == "rerender":
        Pipeline(job, settings).rerender([int(x) for x in data["clips"]])
    elif kind == "post":
        job.data["steps"].pop("post", None)
        job.save()
        Pipeline(job, settings).run(steps=("post",))
    elif kind == "send":
        from konveyer.telegram import send_job
        job.set_state("running", "send")
        send_job(job, settings_for_job(settings, job), Log(job.log_file, echo=False),
                 clips=bool(data.get("clips", True)), post=bool(data.get("post", True)),
                 only=data.get("only") or None, force=bool(data.get("force")))
        job.reload()
        if job.data.get("state") != "review":
            job.set_state("done")
    else:
        raise ValueError(f"неизвестная задача {kind}")


class App:
    def __init__(self, settings: Settings,
                 runner: Callable[[Settings, Job, str, dict], None] | None = None):
        self.settings = settings
        self.runner = runner or default_runner
        self.tasks: queue.Queue[Task | None] = queue.Queue()
        self.active: str = ""
        self.pending: set[str] = set()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="konveyer-worker")
        self._worker.start()

    # ---------- очередь
    def submit(self, job: Job, kind: str, data: dict | None = None) -> None:
        job.set_state("queued", kind)
        self.pending.add(job.id)
        self.tasks.put((job.id, kind, data or {}))

    def _loop(self) -> None:
        while True:
            item = self.tasks.get()
            if item is None:
                return
            job_id, kind, data = item
            self.active = job_id
            try:
                job = self.job(job_id)
                if job:
                    self.runner(self.settings, job, kind, data)
            except Exception as e:
                job = self.job(job_id)
                if job:
                    job.reload()
                    if job.data.get("state") != "failed":
                        job.set_state("failed", kind, mask(str(e)))
                    with open(job.log_file, "a", encoding="utf-8") as f:
                        f.write(mask(traceback.format_exc()) + "\n")
            finally:
                self.active = ""
                self.pending.discard(job_id)
                self.tasks.task_done()

    def stop(self) -> None:
        self.tasks.put(None)

    # ---------- задания
    def job(self, job_id: str) -> Job | None:
        if not JOB_ID.match(job_id or ""):
            return None
        folder = self.settings.jobs_dir / job_id
        if not (folder / "job.json").exists():
            return None
        return Job.load(folder)


def _full_opts(data: dict) -> dict:
    """Галочка «целое видео» и необязательный отрезок. Пусто — обычные Shorts."""
    from konveyer.moments import parse_time

    start = data.get("full_from")
    end = data.get("full_to")
    has_time = start not in (None, "") or end not in (None, "")
    if not data.get("full") and not has_time:
        return {}
    opts: dict = {"full": True}
    try:
        if start not in (None, ""):
            opts["full_from"] = round(parse_time(start), 3)
        if end not in (None, ""):
            opts["full_to"] = round(parse_time(end), 3)
    except ValueError as e:
        raise ValueError(str(e)) from e
    if "full_from" in opts and "full_to" in opts and opts["full_to"] <= opts["full_from"]:
        raise ValueError("конец отрезка должен быть позже начала")
    return opts


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "konveyer"

        def log_message(self, *args):          # тихо, без строки на каждый запрос
            pass

        # ---------- ответы
        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _error(self, status: int, msg: str) -> None:
            lang = (self.headers.get("X-Konveyer-Lang") or "").strip().lower()
            self._json({"error": tr(msg, lang if lang in ("ru", "en") else None)}, status)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                raise ValueError("слишком большой запрос")
            raw = self.rfile.read(n) if n else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("ожидался JSON-объект")
            return data

        def _file(self, path: Path) -> None:
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range", "")
            m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start, end = max(0, size - int(m.group(2))), size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                    return
                status, length = 206, end - start + 1
                headers = {"Content-Range": f"bytes {start}-{end}/{size}"}
            else:
                start, length, status, headers = 0, size, 200, {}
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                        return
                    left -= len(chunk)

        # ---------- GET
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path in ("/", "/index.html"):
                    return self._file(STATIC / "index.html")
                if path.startswith("/static/"):
                    f = (STATIC / path[len("/static/"):]).resolve()
                    if STATIC.resolve() in f.parents and f.is_file():
                        return self._file(f)
                    return self._error(404, "нет такого файла")
                if path == "/api/status":
                    return self._json({
                        "active": app.active, "queued": app.tasks.qsize(),
                        "pending": sorted(app.pending),
                        "secrets": {n: has_secret(n) for n in SECRET_NAMES},
                        "defaults": {"zoom": app.settings.render.zoom,
                                     "subs_style": app.settings.subs.style},
                        "jobs_dir": str(app.settings.jobs_dir),
                        "lang": get_lang()})
                if path == "/api/jobs":
                    return self._json([j.summary() for j in list_jobs(app.settings)])
                m = re.match(r"^/api/jobs/([^/]+)$", path)
                if m:
                    job = app.job(m.group(1))
                    if not job:
                        return self._error(404, "задание не найдено")
                    post = job.post_md.read_text(encoding="utf-8") if job.post_md.exists() else ""
                    return self._json({"job": job.data, "log": mask(tail(job.log_file, 120)),
                                       "post": post, "steps": list(STEPS)})
                m = re.match(r"^/files/([^/]+)/(.+)$", path)
                if m:
                    job = app.job(m.group(1))
                    if not job:
                        return self._error(404, "задание не найдено")
                    f = (job.dir / m.group(2)).resolve()
                    if job.dir.resolve() not in f.parents or not f.is_file():
                        return self._error(404, "нет такого файла")
                    return self._file(f)
                return self._error(404, "не найдено")
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                return
            except Exception as e:
                return self._error(500, mask(str(e)))

        # ---------- POST
        def do_POST(self):
            if self.headers.get("X-Konveyer") != "1":
                return self._error(403, "нет заголовка X-Konveyer")
            path = unquote(urlparse(self.path).path)
            try:
                data = self._body()
                if path == "/api/jobs":
                    source = str(data.get("source", "")).strip()
                    if not source:
                        return self._error(400, "нужна ссылка или путь к файлу")
                    opts = {}
                    for key, lo, hi, integer in (("count", 1, 10, True), ("min", 3, 600, False),
                                                 ("max", 5, 600, False)):
                        value = num_param(data, key, lo, hi, integer)
                        if value is not None:
                            opts[key] = value
                    if "min" in opts and "max" in opts and opts["min"] > opts["max"]:
                        return self._error(400, "мин. длина клипа больше макс.")
                    if data.get("layout") in ("auto", "blur", "full", "smart"):
                        opts["layout"] = data["layout"]
                    z = num_param(data, "zoom", 1.0, 2.0)
                    if z is not None:
                        opts["zoom"] = round(z, 2)
                    if data.get("subs_style") in SUBS_STYLES:
                        opts["subs_style"] = data["subs_style"]
                    if data.get("no_speakers"):
                        opts["speakers"] = False
                    if data.get("one_voice"):
                        opts["per_speaker"] = False
                    if data.get("score"):
                        opts["score"] = True
                    if data.get("vision_check"):
                        opts["vision_check"] = True
                    for key in ("tighten",):
                        if data.get(key):
                            opts[key] = True
                    for key, opt in (("no_llm", "llm"), ("no_captions", "captions"),
                                     ("no_titles", "titles"), ("no_translate", "translate")):
                        if data.get(key):
                            opts[opt] = False
                    if data.get("force_captions"):
                        opts["force_captions"] = True
                    if data.get("cover_mode") in ("frame", "ai", "both"):
                        opts["cover_mode"] = data["cover_mode"]
                    if data.get("no_cover"):
                        opts["cover"] = False
                    if data.get("voice"):
                        opts["voice"] = True
                        if data.get("voice_speaker") in VOICE_SPEAKERS:
                            opts["voice_speaker"] = data["voice_speaker"]
                        if data.get("voice_no_subs"):
                            opts["voice_subs"] = False
                        if data.get("separate"):
                            opts["separate"] = True
                    full_opts = _full_opts(data)
                    if full_opts:
                        opts = {k: v for k, v in opts.items()
                                if k not in ("count", "min", "max", "layout", "zoom", "tighten")}
                        opts.update(full_opts)
                    job = Job.create(app.settings, source, opts)
                    app.submit(job, "run", {"tg": bool(data.get("tg"))})
                    return self._json({"id": job.id}, 201)

                m = re.match(r"^/api/jobs/([^/]+)/(\w+)(?:/(\d+))?$", path)
                if not m:
                    return self._error(404, "не найдено")
                job = app.job(m.group(1))
                if not job:
                    return self._error(404, "задание не найдено")
                action, num = m.group(2), m.group(3)
                if job.id in app.pending and action != "post_text":
                    return self._error(409, "задание уже в работе — подожди")
                if action == "clips" and num:
                    clip = job.clip(int(num))
                    for key in ("title", "kicker", "bottom"):
                        if key in data:
                            clip[key] = str(data[key])
                    for key in ("start", "end"):
                        value = num_param(data, key, 0.0, 86400.0)
                        if value is not None:
                            clip[key] = value
                    if data.get("layout") in ("auto", "blur", "full", "smart"):
                        clip["layout_req"] = "" if data["layout"] == "auto" else data["layout"]
                    z = num_param(data, "zoom", 1.0, 2.0)
                    if z is not None:
                        clip["zoom"] = round(z, 2)
                    if clip["end"] <= clip["start"]:
                        return self._error(400, "конец клипа должен быть позже начала")
                    job.save()
                    app.submit(job, "rerender", {"clips": [clip["id"]]})
                    return self._json({"ok": True})
                if action == "post_text":
                    job.post_md.write_text(str(data.get("text", "")), encoding="utf-8")
                    return self._json({"ok": True})
                if action == "post":
                    app.submit(job, "post")
                    return self._json({"ok": True})
                if action == "redo":
                    step = data.get("from", "moments")
                    if step not in STEPS:
                        return self._error(400, "неизвестный шаг")
                    app.submit(job, "redo", {"from": step})
                    return self._json({"ok": True})
                if action == "send":
                    if not (has_secret("TG_BOT_TOKEN") and has_secret("TG_CHAT_ID")):
                        return self._error(400, "не заданы TG_BOT_TOKEN и TG_CHAT_ID")
                    payload = {"clips": data.get("clips", True),
                               "post": data.get("post", True),
                               "only": data.get("only") or []}
                    if data.get("force"):
                        payload["force"] = True
                    app.submit(job, "send", payload)
                    return self._json({"ok": True})
                return self._error(404, "неизвестное действие")
            except (KonveyerError, ValueError, KeyError, json.JSONDecodeError) as e:
                return self._error(400, mask(str(e)))
            except Exception as e:
                return self._error(500, mask(str(e)))

    return Handler


def make_server(settings: Settings, app: App | None = None, port: int | None = None):
    app = app or App(settings)
    server = ThreadingHTTPServer((settings.web.host, settings.web.port if port is None else port),
                                 make_handler(app))
    server.daemon_threads = True
    return server, app


def serve(settings: Settings, open_browser: bool = True) -> None:
    server, app = make_server(settings)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(tr(f"Веб-интерфейс: {url}  (остановить — Ctrl+C)"), flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        server.server_close()
