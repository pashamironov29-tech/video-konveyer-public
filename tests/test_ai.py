"""DeepSeek-клиент, перевод (мост к subs.py и встроенный), пост."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from conftest import FAKE_KEY, FakeLLM
from konveyer.config import LLMCfg
from konveyer.jobs import Job
from konveyer.llm import DeepSeek, chat_json, extract_json, make_llm
from konveyer.post import (_with_hashtags, fallback_post, load_style, make_post, parse_post,
                           post_prompt, render_md)
from konveyer.transcript import Transcript
from konveyer.translate import _internal_translate, load_subs, translate_texts
from konveyer.util import KonveyerError, read_json

REQ = httpx.Request("POST", "https://api.deepseek.com/chat/completions")


# ------------------------------------------------------------------ DeepSeek

class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item))])


def client(script):
    comp = FakeCompletions(script)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


def status_error(cls, code):
    return cls("ошибка", response=httpx.Response(code, request=REQ), body=None)


@pytest.mark.parametrize("raw, value", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": [1, 2]}\n```', {"a": [1, 2]}),
    ('Вот ответ: {"a": "б"} — готово', {"a": "б"}),
    ('[1, 2, 3]', [1, 2, 3]),
])
def test_extract_json(raw, value):
    assert extract_json(raw) == value


def test_extract_json_garbage():
    with pytest.raises(ValueError):
        extract_json("никакого json")


def test_deepseek_request_shape():
    c, comp = client(['{"ok": true}', "текст"])
    ds = DeepSeek(LLMCfg(), FAKE_KEY, client=c, sleep=lambda s: None)
    assert ds.chat("сис", "юзер", temperature=0.3) == '{"ok": true}'
    kw = comp.kwargs[0]
    assert kw["model"] == "deepseek-chat" and kw["temperature"] == 0.3
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["messages"][0] == {"role": "system", "content": "сис"}
    assert ds.chat("с", "ю", json_mode=False) == "текст"
    assert "response_format" not in comp.kwargs[1] and comp.kwargs[1]["temperature"] == 0.9


def test_deepseek_retries_then_succeeds():
    sleeps = []
    c, comp = client([openai.APIConnectionError(request=REQ), status_error(openai.InternalServerError, 500), "{}"])
    ds = DeepSeek(LLMCfg(attempts=3), FAKE_KEY, client=c, sleep=sleeps.append)
    assert ds.chat("с", "ю") == "{}"
    assert sleeps == [2, 4] and len(comp.kwargs) == 3


def test_deepseek_gives_up():
    c, _ = client([openai.APIConnectionError(request=REQ)] * 2)
    ds = DeepSeek(LLMCfg(attempts=2), FAKE_KEY, client=c, sleep=lambda s: None)
    with pytest.raises(KonveyerError, match="после 2 попыток"):
        ds.chat("с", "ю")


def test_deepseek_401_and_402_no_retry():
    c, comp = client([status_error(openai.AuthenticationError, 401)])
    with pytest.raises(KonveyerError, match="401"):
        DeepSeek(LLMCfg(), FAKE_KEY, client=c, sleep=lambda s: None).chat("с", "ю")
    assert len(comp.kwargs) == 1
    c, comp = client([status_error(openai.APIStatusError, 402)])
    with pytest.raises(KonveyerError, match="балансе"):
        DeepSeek(LLMCfg(), FAKE_KEY, client=c, sleep=lambda s: None).chat("с", "ю")
    assert len(comp.kwargs) == 1


def test_chat_json_reasks():
    class Flaky:
        n = 0

        def chat(self, system, user, json_mode=True, temperature=None):
            self.n += 1
            return "мусор" if self.n == 1 else '{"x": 1}'
    f = Flaky()
    assert chat_json(f, "с", "ю") == {"x": 1} and f.n == 2
    with pytest.raises(KonveyerError, match="не JSON"):
        chat_json(FakeLLM(broken=True), "с", "ю")


def test_make_llm_needs_env(settings, quiet_log, monkeypatch):
    assert make_llm(settings, quiet_log) is None
    assert make_llm(settings, quiet_log, enabled=False) is None
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    llm = make_llm(settings, quiet_log)
    assert isinstance(llm, DeepSeek)
    assert FAKE_KEY not in repr(vars(llm.cfg))


# ------------------------------------------------------------------ перевод

def test_internal_translate_retries_missing(quiet_log):
    calls = []

    def call(batch):
        calls.append(dict(batch))
        if len(calls) == 1:                      # первый ответ — только часть
            return json.dumps({"1": "раз"})
        return json.dumps({str(k): f"т{k}" for k in batch})
    out = _internal_translate(["one", "two", "three"], call, quiet_log)
    assert out == ["раз", "т2", "т3"]
    assert calls[1] == {2: "two", 3: "three"}


def test_internal_translate_keeps_original_on_failure(quiet_log):
    out = _internal_translate(["one", "two"], lambda b: "мусор", quiet_log, attempts=2)
    assert out == ["one", "two"]


def test_internal_translate_batches(quiet_log):
    sizes = []

    def call(batch):
        sizes.append(len(batch))
        return json.dumps({str(k): v.upper() for k, v in batch.items()})
    texts = [f"t{i}" for i in range(95)]
    out = _internal_translate(texts, call, quiet_log, batch=40)
    assert sizes == [40, 40, 15] and out[94] == "T94"


def test_translate_with_fake_llm(quiet_log):
    llm = FakeLLM()
    assert translate_texts(["hello", "world"], llm, quiet_log) == ["РУ hello", "РУ world"]
    assert translate_texts([], llm, quiet_log) == []


def test_translate_uses_subs_bridge(tmp_path, quiet_log):
    (tmp_path / "subs.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Cue:\n    text: str\n"
        'SYSTEM_PROMPT = "Промпт субтитров из subs.py"\n'
        "def translate_texts(texts, translate_batch, batch_size=40, attempts=3):\n"
        "    import json\n"
        "    raw = translate_batch({i + 1: t for i, t in enumerate(texts)})\n"
        "    data = json.loads(raw)\n"
        "    return [data[str(i + 1)] for i in range(len(texts))]\n", encoding="utf-8")
    llm = FakeLLM()
    out = translate_texts(["hi"], llm, quiet_log, str(tmp_path))
    assert out == ["РУ hi"]
    assert llm.calls[0][0] == "Промпт субтитров из subs.py"
    assert load_subs(str(tmp_path)) is not None


def test_load_subs_missing_or_broken(tmp_path):
    assert load_subs(str(tmp_path / "нет")) is None
    bad = tmp_path / "битый"
    bad.mkdir()
    (bad / "subs.py").write_text("raise RuntimeError('нет')\n", encoding="utf-8")
    assert load_subs(str(bad)) is None


def test_real_subs_py_bridge_if_present(quiet_log):
    """Ваш настоящий subs.py (папка в KONVEYER_SUBS_DIR) подключается и переводит."""
    import os
    real = os.environ.get("KONVEYER_SUBS_DIR", "")
    from pathlib import Path
    if not real or not (Path(real) / "subs.py").exists():
        pytest.skip("subs.py не найден")
    mod = load_subs(real)
    assert mod is not None, "subs.py есть, но не загрузился"
    llm = FakeLLM()
    assert translate_texts(["hello", "world"], llm, quiet_log, real) == ["РУ hello", "РУ world"]
    assert llm.calls[0][0] == mod.SYSTEM_PROMPT


# ------------------------------------------------------------------ пост

def test_with_hashtags():
    assert _with_hashtags("Робот  учится", "#shorts #россия") == "Робот учится #shorts #россия"
    assert _with_hashtags("Робот #Shorts", "#shorts #россия") == "Робот #Shorts #россия"


def test_parse_post(settings):
    p = parse_post({"post": " текст ", "shorts_title": "Хук", "comment": "Вопрос?"}, settings)
    assert p["post"].startswith("🤖 текст\n\n#ИИ ") and p["ai"] is True   # эмодзи и хештеги
    assert p["shorts_title"] == "Хук #shorts #россия" and p["comment"] == "Вопрос?"
    with pytest.raises(KonveyerError):
        parse_post({"post": ""}, settings)
    settings.post.shorts_pack = False
    assert "shorts_title" not in parse_post({"post": "т"}, settings)


def test_fallback_post_and_md(settings):
    p = fallback_post({"uploader": "Автор", "title": "Ролик"}, [{"title": "Робот|учится"}], settings)
    assert p["ai"] is False and "Автор" in p["post"] and "Робот учится" in p["post"]
    md = render_md(p)
    assert "## YouTube Shorts" in md and "#shorts" in md and "Черновик без ИИ" in md


def test_post_prompt_includes_style(settings, tmp_path):
    style = tmp_path / "ПРОМПТ.md"
    style.write_text("Пиши с иронией, как взрослый паблик.", encoding="utf-8")
    settings.post.style_files = [str(style), str(tmp_path / "нет.md")]
    assert "иронией" in load_style(settings.post.style_files)
    tr = Transcript.from_dict({"language": "en", "segments": [{"start": 0, "end": 1, "text": "hello"}]})
    system, user = post_prompt({"title": "Т", "uploader": "А"}, tr, [{"id": 1, "title": "x|y", "bottom": "z"}], settings)
    assert "иронией" in system and "#shorts #россия" in system
    assert "клип 1: x y / z" in user and "hello" in user


def test_make_post_ai_and_fallback(settings, quiet_log):
    job = Job.create(settings, "a.mp4")
    job.data["meta"] = {"title": "Ролик", "uploader": "Автор"}
    job.data["clips"] = [{"id": 1, "title": "Робот|учится", "bottom": ""}]
    post = make_post(job, settings, quiet_log, FakeLLM(), Transcript())
    assert post["ai"] and "Тестовый пост" in job.post_md.read_text(encoding="utf-8")
    assert read_json(job.post_json)["shorts_title"].endswith("#shorts #россия")
    post = make_post(job, settings, quiet_log, FakeLLM(broken=True), Transcript())
    assert not post["ai"] and "Черновик без ИИ" in job.post_md.read_text(encoding="utf-8")
    assert not make_post(job, settings, quiet_log, None, Transcript())["ai"]
