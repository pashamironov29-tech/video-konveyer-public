"""Общие заглушки: без сети, без Whisper, без настоящего DeepSeek и Telegram."""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

# Хосты, в которые тест попытался выйти. Фикстура clean_env их обрывает.
BLOCKED_HOSTS: list[str] = []


def _local_host(host: str) -> bool:
    text = (host or "").strip().strip("[]").lower()
    return text in {"", "localhost", "127.0.0.1", "::1"} or text.startswith("127.")


def _install_network_guard(monkeypatch) -> None:
    """Любой внешний адрес обрывается сразу. 127.0.0.1 для веб-тестов остаётся."""
    real_dns = socket.getaddrinfo
    real_connect = socket.socket.connect

    def guarded_dns(host, port, *args, **kwargs):
        name = "" if host is None else str(host)
        if not _local_host(name):
            BLOCKED_HOSTS.append(name)
            raise OSError(f"тест не ходит в сеть ({name})")
        return real_dns(host, port, *args, **kwargs)

    def guarded_connect(self, address):
        host = ""
        if isinstance(address, tuple) and address:
            host = str(address[0])
        if host and not _local_host(host):
            BLOCKED_HOSTS.append(host)
            raise OSError(f"тест не ходит в сеть ({host})")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_dns)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from konveyer.config import load_settings  # noqa: E402
from konveyer.envkeys import SECRET_NAMES  # noqa: E402
from konveyer.hardware import Encoder, encoder_args  # noqa: E402
from konveyer.log import Log  # noqa: E402
from konveyer.transcript import Segment, Transcript, Word  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="нет ffmpeg в PATH")

FAKE_TOKEN = "123456:FAKE-test-token-not-real"
FAKE_KEY = "sk-fake-deepseek-key-for-tests"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Настоящие ключи из окружения в тесты не попадают. Наружу тесты не ходят."""
    BLOCKED_HOSTS.clear()
    for name in SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("KONVEYER_CONFIG", raising=False)
    monkeypatch.setenv("KONVEYER_CLEANUP", "0")   # автоочистка в тестах выключена
    _install_network_guard(monkeypatch)
    from konveyer import cli
    monkeypatch.setattr(
        cli, "edge_tts_status",
        lambda timeout=5.0: (False, "не установлен: py -m pip install edge-tts"))


@pytest.fixture
def settings(tmp_path):
    s = load_settings(tmp_path / "нет.toml", root=tmp_path)
    s.paths.subs_dir = ""
    s.post.style_files = []
    # По умолчанию проверка включена. Здесь её гасим, чтобы юнит-тесты не звали Whisper.
    s.voice.voice_check = False
    # Обложка по умолчанию both: нейросеть в тестах не ждём, сеть и так закрыта.
    s.cover.ai_attempts = 1
    s.cover.ai_timeout = 1.0
    s.cover.candidates = 4
    return s


@pytest.fixture
def quiet_log():
    return Log(None, echo=False)


@pytest.fixture
def fast_encoder():
    return Encoder("libx264", encoder_args("libx264", 30, "ultrafast"))


# ------------------------------------------------------------------ синтетическое видео

def make_video(path: Path, seconds: float = 20, size: str = "640x360", audio: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [path.name]
    subprocess.run(cmd, check=True, cwd=path.parent)
    return path


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("media") / "видео тест"


@pytest.fixture(scope="session")
def wide_video(video_dir):
    if not HAS_FFMPEG:
        pytest.skip("нет ffmpeg")
    return make_video(video_dir / "ролик 16x9.mp4", 20)


@pytest.fixture(scope="session")
def vertical_video(video_dir):
    if not HAS_FFMPEG:
        pytest.skip("нет ffmpeg")
    return make_video(video_dir / "вертикаль.mp4", 6, "360x640")


@pytest.fixture(scope="session")
def silent_video(video_dir):
    if not HAS_FFMPEG:
        pytest.skip("нет ffmpeg")
    return make_video(video_dir / "без звука.mp4", 4, audio=False)


# ------------------------------------------------------------------ поддельная речь

def english_transcript(duration: float = 20.0, pause: tuple[float, float] = (12.0, 14.0)) -> Transcript:
    """Слово каждые 0.5 с, пауза в середине, точка каждое 6-е слово."""
    segs: list[Segment] = []
    words: list[Word] = []
    t, i = 0.2, 0
    while t + 0.4 < duration:
        if not pause[0] <= t < pause[1]:
            i += 1
            words.append(Word(round(t, 3), round(t + 0.4, 3), f" word{i}" + ("." if i % 6 == 0 else "")))
            if i % 6 == 0:
                segs.append(Segment(words[0].start, words[-1].end,
                                    "".join(w.word for w in words).strip(), list(words)))
                words = []
        t += 0.5
    if words:
        segs.append(Segment(words[0].start, words[-1].end, "".join(w.word for w in words).strip(), words))
    return Transcript("en", segs)


class FakeTranscriber:
    def __init__(self, transcript: Transcript | None = None):
        self.transcript = transcript or english_transcript()
        self.calls = 0

    def transcribe(self, audio, duration, log):
        self.calls += 1
        return self.transcript


@pytest.fixture
def fake_transcriber():
    return FakeTranscriber()


# ------------------------------------------------------------------ поддельный DeepSeek

class FakeLLM:
    """Отвечает по смыслу системного промпта, как это делал бы DeepSeek."""

    def __init__(self, broken: bool = False, moments: list | None = None):
        self.broken = broken
        self.moments = moments
        self.calls: list[tuple[str, str]] = []

    def chat(self, system, user, json_mode=True, temperature=None):
        self.calls.append((system, user))
        if self.broken:
            return "извини, не могу"
        if "монтажёр" in system:
            return json.dumps({"moments": self.moments if self.moments is not None else [
                {"start": 2.0, "end": 9.0, "score": 9, "reason": "сильный кусок",
                 "kicker": "Тестовый автор", "title": "Робот|учится", "bottom": "Без паники"},
                {"start": 14.5, "end": 19.5, "score": 7, "reason": "второй кусок",
                 "kicker": "", "title": "", "bottom": ""},
            ]}, ensure_ascii=False)
        if "тексты на экран" in system:
            items = json.loads(user.split("Ролики:\n", 1)[1])
            return json.dumps({"items": [{"id": it["id"], "kicker": "Кикер",
                                          "title": f"Заголовок|{it['id']}", "bottom": "Подпись"}
                                         for it in items]}, ensure_ascii=False)
        if "ужимаешь реплику" in system:
            limit = 40
            if "Не длиннее " in user:
                head = user.split("Не длиннее ", 1)[1].split()[0]
                if head.isdigit():
                    limit = int(head)
            src = user.split("Реплика:", 1)[1].split("\n", 1)[0].strip() if "Реплика:" in user else ""
            out: list[str] = []
            for word in src.split():
                cand = " ".join([*out, word])
                if out and len(cand) > limit:
                    break
                out.append(word)
            return json.dumps({"text": " ".join(out) or "Коротко."}, ensure_ascii=False)
        if "сокращаешь фразу" in system:
            return json.dumps({"text": "Коротко."}, ensure_ascii=False)
        if "проверяешь тексты перед публикацией" in system:
            return json.dumps({"ok": True, "reason": "норм"}, ensure_ascii=False)
        if "сравниваешь русскую озвучку" in system:
            return json.dumps({"score": 9, "phrases": []}, ensure_ascii=False)
        if "длинному видео" in system or "длинному горизонтальному" in system:
            return json.dumps({
                "post": "🤖 Целый ролик\n\nТестовый автор разобрал тему.",
                "topics": [{"time": "0:00", "title": "Начало"},
                           {"time": "0:10", "title": "Середина"}],
            }, ensure_ascii=False)
        if "заголовок обложки" in system:
            return json.dumps({"title": "ИИ взломал сам себя?"}, ensure_ascii=False)
        if "thumbnail background" in system:
            return json.dumps({"prompt": "abstract neon circuit, no people, no text"}, ensure_ascii=False)
        if "описание к одному короткому" in system:
            return json.dumps({"text": "🐝 Агенты передают задачу друг другу.\n\n"
                                       "Тестовый автор рассказал об этом.\n\nСтрашно или круто?",
                               "hashtags": ["#Anthropic", "#ИИ"]}, ensure_ascii=False)
        if "ведёшь ВК-группу" in system:
            return "```json\n" + json.dumps({
                "post": "🤖 Тестовый пост\n\nТестовый автор показал робота.",
                "shorts_title": "Робот учится #ии",
                "shorts_description": "Коротко о роботе\n#ии",
                "comment": "А ты бы доверил роботу?"}, ensure_ascii=False) + "\n```"
        if "переводишь субтитры" in system or "субтитр" in system.lower():
            batch = json.loads(user)
            return json.dumps({k: f"РУ {v}" for k, v in batch.items()}, ensure_ascii=False)
        return "{}"


@pytest.fixture
def fake_llm():
    return FakeLLM()


# ------------------------------------------------------------------ поддельный Telegram

class FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body, self.status_code, self.text = body, status, json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, fail_with: str = ""):
        self.posts: list[dict] = []
        self.fail_with = fail_with

    def post(self, url, data=None, files=None, timeout=None):
        entry = {"url": url, "data": dict(data or {}), "files": {}}
        for key, (name, fh, ctype) in (files or {}).items():
            entry["files"][key] = (name, len(fh.read()), ctype)
        self.posts.append(entry)
        if self.fail_with:
            return FakeResponse({"ok": False, "description": self.fail_with}, 400)
        return FakeResponse({"ok": True, "result": {}})


@pytest.fixture
def fake_session():
    return FakeSession()
