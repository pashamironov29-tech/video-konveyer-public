"""Речь ненадёжна, проверка перед Telegram, расход DeepSeek, каскад лиц, deno."""
from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np

from conftest import FAKE_KEY, FakeLLM, FakeSession, FakeTranscriber, needs_ffmpeg
from konveyer import cli, hardware
from konveyer.facetrack import FaceFinder, _try_haar
from konveyer.hardware import Encoder
from konveyer.jobs import Job
from konveyer.layout import geometry
from konveyer.log import Log
from konveyer.post import post_prompt
from konveyer.quality import assess_publish
from konveyer.render import Renderer
from konveyer.speech import SKIP_CAPTIONS, assess_speech
from konveyer.telegram import Telegram, send_job
from konveyer.transcript import Segment, Transcript, Word
from konveyer.usage import estimate_cost, format_usd, read_usage, summary_line
from konveyer.llm import DeepSeek
from konveyer.config import LLMCfg
from konveyer.usage import UsageBook


def _seg(start, end, text, no_speech=None, logprob=None, compression=None, words=None):
    if words is None:
        words = [Word(start, end, " " + text)]
    return Segment(start, end, text, words, no_speech, logprob, compression)


def music_transcript() -> Transcript:
    """17 с музыки: ja 74% и «ルルル», как в ролике из X."""
    return Transcript(
        "ja",
        [_seg(0, 8, "ルルルルルル", 0.82, -1.05, 2.9),
         _seg(8, 16, "生まれるのはシンマゾー", 0.71, -0.92, 1.6)],
        language_probability=0.74, vad=True)


def dense_speech(lang: str, prob: float, n: int = 80, duration: float = 84.0) -> Transcript:
    words: list[Word] = []
    segs: list[Segment] = []
    t = 0.2
    for i in range(1, n + 1):
        words.append(Word(round(t, 3), round(t + 0.3, 3), f" слово{i}"))
        t += 0.34
        if i % 8 == 0:
            segs.append(Segment(words[0].start, words[-1].end, "".join(w.word for w in words).strip(),
                                list(words), 0.04, -0.22, 1.35))
            words = []
    if words:
        segs.append(Segment(words[0].start, words[-1].end, "".join(w.word for w in words).strip(),
                            list(words), 0.04, -0.22, 1.35))
    assert segs[-1].end < duration
    return Transcript(lang, segs, language_probability=prob, vad=True)


# ------------------------------------------------------------------ речь

def test_music_hallucination_is_unreliable(settings):
    verdict = assess_speech(music_transcript(), 17, settings.speech)
    assert not verdict.ok
    assert "повтор символов" in verdict.reason


def test_real_speech_stays(settings):
    steve = assess_speech(dense_speech("en", 1.0, 80, 84), 84, settings.speech)
    news = assess_speech(dense_speech("ru", 1.0, 200, 257), 257, settings.speech)
    assert steve.ok and steve.reason == ""
    assert news.ok and news.reason == ""


def test_low_language_prob_alone_does_not_reject(settings):
    """74% языка при нормальных остальных сигналах — ещё не мусор."""
    words = [Word(i * 0.4, i * 0.4 + 0.3, f" слово{i}") for i in range(1, 13)]
    tr = Transcript("ja", [Segment(0, 5, "разговор о погоде сегодня вечером у дома", words,
                                   0.1, -0.3, 1.2)], language_probability=0.74, vad=True)
    verdict = assess_speech(tr, 12, settings.speech)
    assert verdict.ok
    assert "низкая вероятность языка" in verdict.weak


def test_two_weak_signals_reject(settings):
    words = [Word(i * 0.4, i * 0.4 + 0.3, f" слово{i}") for i in range(1, 13)]
    tr = Transcript("ja", [Segment(0, 5, "разговор о погоде сегодня вечером у дома", words,
                                   0.75, -0.4, 1.2)], language_probability=0.74, vad=True)
    verdict = assess_speech(tr, 12, settings.speech)
    assert not verdict.ok
    assert "низкая вероятность языка" in verdict.reason
    assert "высокая вероятность тишины" in verdict.reason


def test_word_repeat_and_filler(settings):
    rep = Transcript("en", [_seg(0, 5, "you you you you you you")], language_probability=1.0)
    assert not assess_speech(rep, 5, settings.speech).ok
    filler = Transcript("en", [_seg(0, 3, "thank you thank you")], language_probability=0.4, vad=True)
    assert not assess_speech(filler, 10, settings.speech).ok
    polite = Transcript("en", [_seg(0, 2, "thank you")], language_probability=1.0)
    assert assess_speech(polite, 3, settings.speech).ok


def test_vad_low_coverage_is_weak(settings):
    tr = Transcript("en", [_seg(0, 0.4, "hi there friend")], language_probability=1.0, vad=True)
    verdict = assess_speech(tr, 30, settings.speech)
    assert "VAD почти не нашёл речь" in verdict.weak
    assert verdict.ok                      # один слабый сигнал речь не выключает


def test_old_transcript_without_metrics_stays_reliable():
    tr = Transcript.from_dict({"language": "en", "segments": [
        {"start": 0, "end": 1, "text": "hello world", "words": [{"start": 0, "end": 1, "word": " hello"}]}]})
    assert tr.reliable and tr.language_probability is None
    back = Transcript.from_dict(tr.to_dict())
    assert back == tr


def test_captions_skipped_unless_forced(settings, quiet_log, fast_encoder):
    job = Job.create(settings, "a.mp4")
    job.data["meta"] = {"duration": 17, "width": 1920, "height": 1080, "has_audio": True}
    tr = music_transcript()
    tr.reliable = False
    tr.speech_reason = "повтор символов"
    renderer = Renderer(job, settings, quiet_log, tr, {}, None, encoder=fast_encoder)
    ass, note = renderer.captions(tr.words, None, geometry("blur", 1920, 1080))
    assert ass is None and note == SKIP_CAPTIONS
    settings.speech.force = True
    ass, note = renderer.captions(tr.words, None, geometry("blur", 1920, 1080))
    assert ass and "ルルル" in ass and note.startswith("оригинал")


def test_post_does_not_see_hallucinated_speech(settings):
    tr = music_transcript()
    tr.reliable = False
    _, user = post_prompt({"title": "Трек", "uploader": "Автор"}, tr, [], settings)
    assert "ルル" not in user and "シンマゾー" not in user
    assert "не выдумывай" in user


def test_speech_thresholds_come_from_toml(tmp_path):
    from konveyer.config import load_settings
    (tmp_path / "k.toml").write_text(
        "[speech]\nmin_language_prob = 0.5\nweak_to_reject = 3\n"
        "[llm]\nprice_cache_hit = 0.01\n", encoding="utf-8")
    s = load_settings(tmp_path / "k.toml", root=tmp_path)
    assert s.speech.min_language_prob == 0.5 and s.speech.weak_to_reject == 3
    assert s.llm.price_cache_hit == 0.01
    words = [Word(i * 0.4, i * 0.4 + 0.3, f" слово{i}") for i in range(1, 13)]
    tr = Transcript("ja", [Segment(0, 5, "разговор о погоде сегодня вечером у дома", words,
                                   0.75, -0.4, 1.2)], language_probability=0.74, vad=True)
    assert assess_speech(tr, 12, s.speech).ok      # порог пачки поднят до 3


# ------------------------------------------------------------------ проверка перед отправкой

def _ready_job(settings, title="Новости дня", post=None):
    job = Job.create(settings, "a.mp4")
    job.data["meta"] = {"title": "Ролик", "uploader": "Автор", "description": "Короткий ролик автора."}
    job.data["clips"] = [{"id": 1, "title": title, "kicker": "Автор", "bottom": "Без паники", "file": ""}]
    text = post or ("Автор выложил короткий ролик. Из названия больше ничего не следует, "
                    "и придумывать сюжет тут не из чего. Напиши, если узнал, что это было.")
    job.post_md.write_text(text, encoding="utf-8")
    job.save()
    return job


def test_local_review_sends_only_notice(settings):
    job = _ready_job(settings, title="р р р р р", post="р р р р р р")
    log = Log(job.log_file, echo=False)
    result = assess_publish(job, settings, log, None)
    assert not result.ok and result.source == "local" and "повтор" in result.reason
    session = FakeSession()
    sent = send_job(job, settings, log, tg=Telegram("123:fake", "1", session), llm=None)
    assert sent == {"clips": [], "post": False, "review": True}
    assert len(session.posts) == 1
    assert session.posts[0]["url"].endswith("/sendMessage")
    assert "На проверку" in session.posts[0]["data"]["text"]
    assert "Клипы не отправлены" in session.posts[0]["data"]["text"]
    assert Job.load(job.dir).data["state"] == "review"


def test_deepseek_rejects_invention(settings):
    class Nope:
        def chat(self, system, user, json_mode=True, temperature=None):
            assert "Материалы" in user and temperature == 0
            return '{"ok": false, "reason": "пост выдумывает хип-хоп"}'

    job = _ready_job(settings)
    result = assess_publish(job, settings, Log(None, echo=False), Nope())
    assert not result.ok and result.source == "deepseek" and "хип-хоп" in result.reason


def test_no_key_and_llm_error_do_not_crash(settings):
    job = _ready_job(settings)
    log = Log(None, echo=False)
    assert assess_publish(job, settings, log, None).ok

    class Boom:
        def chat(self, *args, **kwargs):
            raise RuntimeError("сеть пропала")

    result = assess_publish(job, settings, log, Boom())
    assert result.ok and result.source == "local"


@needs_ffmpeg
def test_force_send_delivers_clips(settings, wide_video):
    job = _ready_job(settings, title="р р р р р", post="р р р р р р")
    clip_dir = job.clip_dir(1)
    clip_dir.mkdir(parents=True)
    shutil.copy(wide_video, clip_dir / "clip.mp4")
    job.data["clips"][0]["file"] = "clips/clip_01/clip.mp4"
    job.save()
    session = FakeSession()
    log = Log(job.log_file, echo=False)
    settings.telegram.shorts_post = True        # общий пост после Shorts — только по настройке
    sent = send_job(job, settings, log, tg=Telegram("123:fake", "1", session), force=True, llm=None)
    assert sent["clips"] == [1] and sent["post"] is True
    assert any(p["url"].endswith("/sendVideo") for p in session.posts)
    assert "принудительно" in job.log_file.read_text(encoding="utf-8")
    assert Job.load(job.dir).data["state"] == "done"


# ------------------------------------------------------------------ токены DeepSeek

def _completion(content, prompt, completion, cache=None, via_extra=False):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    if via_extra:
        usage.model_extra = {"prompt_cache_hit_tokens": cache}
    elif cache is not None:
        usage.prompt_cache_hit_tokens = cache
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


class _Script:
    def __init__(self, items):
        self.items = list(items)

    def create(self, **kwargs):
        return self.items.pop(0)


def test_usage_sums_stages_and_cost(settings):
    settings.llm.price_input = 0.15
    settings.llm.price_output = 0.60
    settings.llm.price_cache_hit = 0.003
    job = Job.create(settings, "a.mp4")
    log = Log(job.log_file, echo=False)
    book = UsageBook(job, settings, log)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Script([
        _completion('{"ok":1}', 1000, 100, 400),
        _completion('{"ok":2}', 200, 50, 0),
    ])))
    ds = DeepSeek(LLMCfg(), FAKE_KEY, client=client, sleep=lambda s: None, book=book)
    ds.stage = "moments"
    ds.chat("с", "ю")
    ds.stage = "post"
    ds.chat("с", "ю")
    book.flush(final=True)
    data = Job.load(job.dir).data["deepseek"]
    assert data["prompt_tokens"] == 1200
    assert data["completion_tokens"] == 150
    assert data["prompt_cache_hit_tokens"] == 400
    assert data["total_tokens"] == 1350          # cache hit уже внутри prompt
    assert data["stages"]["moments"]["prompt_tokens"] == 1000
    assert data["stages"]["post"]["completion_tokens"] == 50
    cost = estimate_cost(1200, 150, 400, 0.15, 0.60, 0.003)
    assert data["cost_usd"] == round(cost, 8)
    text = job.log_file.read_text(encoding="utf-8")
    assert "DeepSeek [моменты]:" in text and "DeepSeek [пост]:" in text
    assert summary_line(data, settings) in text
    assert format_usd(cost) in text and "$" in text


def test_usage_without_prices_skips_dollars(settings):
    settings.llm.price_input = settings.llm.price_output = settings.llm.price_cache_hit = 0
    job = Job.create(settings, "a.mp4")
    log = Log(job.log_file, echo=False)
    book = UsageBook(job, settings, log)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Script([
        _completion("{}", 10, 5, via_extra=True, cache=4),
        _completion("{}", 1, 1),          # без usage-полей cache
    ])))
    # второй ответ без prompt_cache_hit_tokens — не падаем
    ds = DeepSeek(settings.llm, FAKE_KEY, client=client, sleep=lambda s: None, book=book)
    ds.stage = "quality"
    ds.chat("с", "ю")
    ds.chat("с", "ю")
    book.flush(final=True)
    line = Job.load(job.dir).data["deepseek"]["summary"]
    assert line == "DeepSeek: 17 токенов"
    assert "$" not in line
    assert read_usage(None) is None


def test_missing_usage_does_not_crash(settings):
    job = Job.create(settings, "a.mp4")
    book = UsageBook(job, settings, Log(None, echo=False))
    bare = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Script([bare])))
    ds = DeepSeek(settings.llm, FAKE_KEY, client=client, sleep=lambda s: None, book=book)
    assert ds.chat("с", "ю") == "{}"
    assert "deepseek" not in job.data


# ------------------------------------------------------------------ лица и deno

def test_haar_missing_does_not_crash(monkeypatch):
    import sys
    import types

    monkeypatch.setattr("konveyer.facetrack._try_mediapipe", lambda: None)
    fake = types.ModuleType("cv2")
    monkeypatch.setitem(sys.modules, "cv2", fake)
    assert _try_haar() is None
    finder = FaceFinder()
    try:
        assert not finder.available
        assert finder.center_x(np.zeros((40, 40, 3), np.uint8)) is None
    finally:
        finder.close()


def test_haar_empty_classifier_is_skipped(monkeypatch, tmp_path):
    import sys
    import types

    monkeypatch.setattr("konveyer.facetrack._try_mediapipe", lambda: None)
    xml = tmp_path / "haarcascade_frontalface_default.xml"
    xml.write_text("<cascade/>", encoding="utf-8")

    class EmptyCascade:
        def __init__(self, path):
            self.path = path

        def empty(self):
            return True

    fake = types.ModuleType("cv2")
    fake.data = SimpleNamespace(haarcascades=str(tmp_path) + "/")
    fake.CascadeClassifier = EmptyCascade
    monkeypatch.setitem(sys.modules, "cv2", fake)
    assert _try_haar() is None


def test_doctor_mentions_deno(capsys, tmp_path, monkeypatch):
    real = shutil.which

    def which(name):
        if name == "deno":
            return None
        return real(name)

    monkeypatch.setattr(cli.shutil, "which", which)
    monkeypatch.setattr(hardware, "cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware, "detect_encoder", lambda *a, **k: Encoder("libx264"))
    code = cli.main(["--set", f"paths.jobs_dir={tmp_path / 'jobs'}", "doctor"])
    out = capsys.readouterr().out
    assert code == 0
    assert "winget install DenoLand.Deno" in out and "deno.land" in out


def test_cli_force_flags():
    a = cli.build_parser().parse_args(["run", "https://youtu.be/x", "--force-captions", "--force-send",
                                       "--no-titles"])
    assert cli._job_options(a)["force_captions"] is True and a.force_send
    assert cli._job_options(a)["titles"] is False
    send = cli.build_parser().parse_args(["send", "last", "--force-send"])
    assert send.force_send
    clip = cli.build_parser().parse_args(["clip", "last", "1", "--no-titles"])
    assert clip.no_titles


def test_no_titles_does_not_ask_deepseek(settings, quiet_log):
    from konveyer.moments import pick_moments
    from konveyer.transcript import Segment as Seg

    settings.render.titles = False
    settings.moments.whole_if_shorter = 30
    job = Job.create(settings, "a.mp4")
    job.data["meta"] = {"duration": 20, "title": "Ролик про робота", "uploader": "Автор",
                        "width": 1920, "height": 1080}
    job.save()
    tr = Transcript("ru", [Seg(0, 5, "раз два три четыре пять шесть семь")])
    llm = FakeLLM()
    log = Log(job.log_file, echo=False)
    pick_moments(job, settings, log, llm, tr, {"hop": 0.5, "loudness": [], "speech": [], "scenes": []})
    assert llm.calls == []
    clip = job.data["clips"][0]
    assert clip["title"] == "" and clip["kicker"] == "" and clip["bottom"] == ""
    assert "надписи выключены" in job.log_file.read_text(encoding="utf-8")
    from konveyer.moments import moments_prompt
    system, _ = moments_prompt(tr, job.data["meta"], settings.moments, "группа", 1000, 20, with_titles=False)
    assert "kicker" not in system and "bottom" not in system
    system_on, _ = moments_prompt(tr, job.data["meta"], settings.moments, "группа", 1000, 20, with_titles=True)
    assert "kicker" in system_on and "от 15 до 60 секунд" in system_on


def test_clip_no_titles_clears_three_fields(settings, monkeypatch):
    job = Job.create(settings, "a.mp4")
    job.data["clips"] = [{"id": 1, "start": 0.0, "end": 8.0, "title": "Было|тут",
                          "kicker": "Сверху", "bottom": "Снизу", "file": "clips/clip_01/clip.mp4"}]
    job.data["meta"] = {"duration": 10, "width": 640, "height": 360}
    job.save()

    def rerender(self, ids):
        return self.job

    monkeypatch.setattr("konveyer.pipeline.Pipeline.rerender", rerender)
    code = cli.main(["--config", str(settings.root / "нет.toml"),
                     "--set", f"paths.jobs_dir={settings.jobs_dir}",
                     "clip", job.id, "1", "--title", "неважно", "--no-titles"])
    assert code == 0
    clip = Job.load(job.dir).data["clips"][0]
    assert clip["title"] == "" and clip["kicker"] == "" and clip["bottom"] == ""


@needs_ffmpeg
def test_music_pipeline_skips_captions(settings, wide_video, fast_encoder):
    from test_pipeline import run_job

    settings.render.width, settings.render.height = 540, 960
    settings.render.caption_size = 40
    settings.moments.whole_if_shorter = 5
    settings.moments.min_sec, settings.moments.max_sec = 5, 8
    llm = FakeLLM()
    tr = music_transcript()
    job, _ = run_job(settings, wide_video, llm=llm, transcriber=FakeTranscriber(tr), encoder=fast_encoder)
    data = Job.load(job.dir).data
    assert data["meta"]["speech_note"] == SKIP_CAPTIONS
    assert all(c["captions"] == SKIP_CAPTIONS for c in data["clips"])
    assert not (job.dir / "clips/clip_01/captions.ass").exists()
    log = job.log_file.read_text(encoding="utf-8")
    assert SKIP_CAPTIONS in log
    fed = " ".join(user for _system, user in llm.calls)
    assert "ルル" not in fed and "シンマゾー" not in fed
