"""Русская озвучка: SSML, подгонка по времени, микс, пропуск без речи, флаги и конфиг.

Silero, XTTS и demucs не запускаются: движок подменён синусом нужной длины.
"""
from __future__ import annotations

import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import FAKE_KEY, FakeLLM, FakeTranscriber, english_transcript, needs_ffmpeg
from konveyer import cli, voice as voicemod
from konveyer.captions import Cue
from konveyer.config import LLMCfg, load_settings
from konveyer.jobs import Job
from konveyer.layout import build_graph, geometry
from konveyer.llm import DeepSeek
from konveyer.log import Log
from konveyer.pipeline import Pipeline, settings_for_job
from konveyer.transcript import Segment, Transcript, Word
from konveyer.usage import UsageBook
from konveyer.util import KonveyerError, ffprobe, read_json
from konveyer.voice import (VoiceOver, XTTSEngine, SileroEngine, atempo_chain, build_ssml,
                            clean_text, duck_expr, phrases_from_cues, plan_fit, separate_music,
                            shorten_phrase, target_chars, voice_gate)

PER_CHAR = 0.06       # поддельный голос: 60 мс на символ


class FakeEngine:
    name = "silero"
    rate = 48000
    tag = "fake:aidar"

    def __init__(self):
        self.calls: list[list[str]] = []

    def synth(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            n = int(self.rate * PER_CHAR * len(t))
            out.append((0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / self.rate)).astype(np.float32))
        return out


# ------------------------------------------------------------------ SSML

def test_ssml_intonation_and_pauses():
    ssml = build_ssml("Ты это видел? Невероятно! Идём дальше. Конец")
    assert ssml.startswith("<speak>") and ssml.endswith("</speak>")
    assert '<prosody pitch="high">Ты это видел?</prosody>' in ssml
    assert '<prosody rate="fast" pitch="high">Невероятно!</prosody>' in ssml
    assert ssml.count('<break time="300ms"/>') == 3
    assert ssml.removesuffix("</speak>").endswith("Конец")


def test_ssml_no_break_after_last_and_ellipsis():
    assert build_ssml("Одна фраза.") == "<speak>Одна фраза.</speak>"
    assert '<break time="500ms"/>' in build_ssml("Ну… ладно.")


def test_ssml_escapes_and_cleans():
    ssml = build_ssml("Том & Джерри <3 «кот» — друг 🙂 isn't")
    assert "Том и Джерри" in ssml and "&apos;" in ssml
    assert "<3" not in ssml and "«" not in ssml and "🙂" not in ssml
    assert clean_text("Это — «тест» (почти)!") == "Это, тест почти!"


# ------------------------------------------------------------------ подгонка

def test_plan_fit():
    assert plan_fit(1.0, 2.0, 1.3) == ("ok", 1.0)
    assert plan_fit(2.4, 2.0, 1.3) == ("tempo", 1.2)
    assert plan_fit(2.6, 2.0, 1.3) == ("tempo", 1.3)
    assert plan_fit(2.7, 2.0, 1.3) == ("shorten", 1.3)
    assert target_chars("а" * 100, 4.0, 2.0, 1.3) == int(100 * 2.0 * 1.3 / 4.0 * 0.9)


def test_phrases_slots_until_next_cue():
    cues = [Cue(0.0, 1.0, "Раз."), Cue(2.0, 3.0, "Два."), Cue(3.1, 3.2, "Три."), Cue(3.2, 4.0, "..."), Cue(4.2, 4.6, "")]
    ph = phrases_from_cues(cues, 5.0)
    assert [p.text for p in ph] == ["Раз.", "Два."]          # «три» — окно 0.05 с, без букв — нет
    assert ph[0].slot == pytest.approx(1.95) and ph[1].slot == pytest.approx(1.05)
    last = phrases_from_cues([Cue(1.0, 2.0, "Последняя.")], 4.0)
    assert last[0].slot == pytest.approx(2.95)


def test_atempo_chain_and_duck_expr():
    assert atempo_chain(1.25) == "atempo=1.2500"
    assert atempo_chain(2.5) == "atempo=2.0,atempo=1.2500"
    assert duck_expr([], -18) == "" and duck_expr([(0, 1)], 0) == ""
    expr = duck_expr([(1.0, 2.0), (2.2, 3.0), (5.0, 6.0)], -18, 0.15)
    assert expr.startswith("1-0.8741*max(")                      # 10^(-18/20) = 0.1259
    assert expr.count("clip(") == 2                              # 1–2 и 2.2–3 слились
    assert "\\" not in expr


def test_graph_mixes_voice_and_ducks():
    g = geometry("blur", 1920, 1080)
    graph, has_a = build_graph(g, fps=30, src_h=1080, text_filters=[], ass_file=None, keep_expr=None,
                               crop_x_expr=None, has_audio=True, duck="1-0.5*t", voice_in="1:a",
                               out_dur=5)
    assert has_a and "[0:a]aresample=48000,volume='1-0.5*t':eval=frame[a0]" in graph
    assert "[a0][vo]amix=inputs=2:duration=first" in graph and graph.endswith("[a]")
    graph, _ = build_graph(g, fps=30, src_h=1080, text_filters=[], ass_file=None, keep_expr=None,
                           crop_x_expr=None, has_audio=True, audio_in="2:a", voice_in="1:a")
    assert "[2:a]aresample" in graph and "[0:a]" not in graph
    graph, has_a = build_graph(g, fps=30, src_h=1080, text_filters=[], ass_file=None, keep_expr=None,
                               crop_x_expr=None, has_audio=False, voice_in="1:a", out_dur=4.5)
    assert has_a and "apad=whole_dur=4.500[a]" in graph
    graph, has_a = build_graph(g, fps=30, src_h=1080, text_filters=[], ass_file=None, keep_expr=None,
                               crop_x_expr=None, has_audio=False)
    assert not has_a and "[a]" not in graph


@needs_ffmpeg
def test_voiceover_fits_tempo_and_shortens(settings, tmp_path):
    settings.voice.voice_mode = "fit"
    settings.voice.speaker = "aidar"          # заглушка притворяется Silero, в заметке — его голос
    folder = tmp_path / "клип"
    folder.mkdir()
    cues = [Cue(0.0, 1.0, "Привет всем"),                       # 0.66 с в окне 1.95 — как есть
            Cue(2.0, 3.0, "Это фраза подлиннее, ей нужно ускорение."),   # 2.4 с → ×1.23
            Cue(4.0, 5.0, "А эта фраза совсем длинная и никак не помещается в своё окно без сокращения.")]
    engine, llm = FakeEngine(), FakeLLM()
    vo = VoiceOver(tmp_path, settings, Log(None, echo=False), llm, engine=engine)
    track = vo.make(folder, cues, 6.0, 1)
    st = {p.text[:5]: p for p in track.phrases}
    assert st["Приве"].status == "ok" and st["Приве"].tempo == 1.0
    assert st["Это ф"].status == "tempo" and 1.0 < st["Это ф"].tempo <= 1.3
    assert st["А эта"].status == "short" and st["А эта"].spoken == "Коротко."
    for p in track.phrases:
        assert p.duration <= p.slot + 0.01
    with wave.open(str(folder / "voice.wav"), "rb") as w:
        assert w.getframerate() == 48000 and w.getnframes() == 6 * 48000
    saved = read_json(folder / "voice.json")
    assert [p["status"] for p in saved["phrases"]] == ["ok", "tempo", "short"]
    assert "ускорено 1" in track.note and "сокращено 1" in track.note and track.note.startswith("aidar")
    assert len([c for c in llm.calls if "сокращаешь" in c[0]]) == 1

    # повтор: всё из кеша — ни синтеза, ни DeepSeek
    calls, llm_calls = len(engine.calls), len(llm.calls)
    VoiceOver(tmp_path, settings, Log(None, echo=False), llm, engine=engine).make(folder, cues, 6.0, 1)
    assert len(engine.calls) == calls and len(llm.calls) == llm_calls


@needs_ffmpeg
def test_voiceover_without_llm_cuts_to_slot(settings, tmp_path):
    settings.voice.voice_mode = "fit"
    cues = [Cue(0.0, 1.0, "Очень длинная фраза, которая не влезет в окно даже с ускорением никак.")]
    settings.voice.max_tempo = 1.3
    log_file = tmp_path / "job.log"
    vo = VoiceOver(tmp_path, settings, Log(log_file, echo=False), None, engine=FakeEngine())
    track = vo.make(tmp_path, cues, 2.0, 1)
    ph = track.phrases[0]
    kept = ph.spoken.split()
    assert ph.status == "cut" and ph.tempo == pytest.approx(1.3)
    assert ph.duration <= 2.15
    assert kept and kept == ph.text.split()[:len(kept)] and ph.dropped
    assert "обрезано 1" in track.note
    assert "обрезка по границе слова" in log_file.read_text(encoding="utf-8")


def test_shorten_counts_tokens_in_usage(settings):
    job = Job.create(settings, "a.mp4")
    book = UsageBook(job, settings, Log(job.log_file, echo=False))
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"text": "Коротко."}'))],
                           usage=SimpleNamespace(prompt_tokens=120, completion_tokens=8))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))
    ds = DeepSeek(LLMCfg(), FAKE_KEY, client=client, sleep=lambda s: None, book=book)
    assert shorten_phrase(ds, "Длинная фраза для проверки сокращения", 12, Log(None, echo=False)) == "Коротко."
    stage = job.data["deepseek"]["stages"]["voice"]
    assert stage["prompt_tokens"] == 120 and stage["completion_tokens"] == 8
    book.flush(final=True)
    assert "DeepSeek [озвучка]: 128 токенов" in job.log_file.read_text(encoding="utf-8")


def test_shorten_rejects_garbage():
    log = Log(None, echo=False)
    assert shorten_phrase(FakeLLM(broken=True), "Фраза подлиннее", 5, log) == ""

    class Longer:
        def chat(self, *a, **k):
            return json.dumps({"text": "Фраза подлиннее, чем была раньше"})

    assert shorten_phrase(Longer(), "Фраза подлиннее", 5, log) == ""


# ------------------------------------------------------------------ движки

def test_silero_engine_calls_apply_tts_with_ssml(settings):
    settings.voice.engine = "silero"
    settings.voice.speaker = "aidar"
    got = []

    class Model:
        def apply_tts(self, **kw):
            got.append(kw)
            return np.ones(4800, np.float32) * 0.5

    eng = SileroEngine(settings.voice, Log(None, echo=False), loader=lambda cfg: Model())
    out = eng.synth(["Привет! Как дела?"])
    assert out[0].shape == (4800,)
    kw = got[0]
    assert kw["ssml_text"].startswith("<speak>") and "prosody" in kw["ssml_text"]
    assert kw["speaker"] == "aidar" and kw["sample_rate"] == 48000
    assert kw["put_accent"] is True and kw["put_yo"] is True


def test_silero_missing_model_is_clear(settings, tmp_path):
    settings.voice.model_path = str(tmp_path / "нет" / "v4_ru.pt")
    with pytest.raises(KonveyerError, match="нет модели Silero.*model_path"):
        SileroEngine(settings.voice, Log(None, echo=False)).synth(["тест"])


def test_engine_failure_disables_voice_once(settings, tmp_path, monkeypatch):
    def boom(*a):
        raise KonveyerError("нет модели Silero: X")

    monkeypatch.setattr(voicemod, "make_engine", boom)
    log_file = tmp_path / "job.log"
    vo = VoiceOver(tmp_path, settings, Log(log_file, echo=False), None)
    assert vo.make(tmp_path, [Cue(0, 1, "Раз.")], 2.0, 1) is None
    assert vo.make(tmp_path, [Cue(0, 1, "Два.")], 2.0, 2) is None
    assert log_file.read_text(encoding="utf-8").count("озвучка выключена") == 1


def test_xtts_needs_sample_and_python(settings, tmp_path):
    log = Log(None, echo=False)
    with pytest.raises(KonveyerError, match="образец голоса"):
        XTTSEngine(settings.voice, log, tmp_path)
    sample = tmp_path / "голос.wav"
    sample.write_bytes(b"RIFF")
    settings.voice.sample = str(sample)
    settings.voice.xtts_python = str(tmp_path / "нет" / "python.exe")
    with pytest.raises(KonveyerError, match="xtts_python"):
        XTTSEngine(settings.voice, log, tmp_path)


def test_xtts_subprocess_request(settings, tmp_path, monkeypatch):
    sample = tmp_path / "голос.wav"
    sample.write_bytes(b"RIFF")
    settings.voice.sample = str(sample)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["req"] = json.loads(open(cmd[2], encoding="utf-8").read())
        return SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: TTS")

    monkeypatch.setattr(voicemod.subprocess, "run", fake_run)
    eng = XTTSEngine(settings.voice, Log(None, echo=False), tmp_path / "кеш")
    with pytest.raises(KonveyerError, match="XTTS упал.*COQUI_TOS_AGREED"):
        eng.synth(["Привет"])
    assert seen["cmd"][1].endswith("xtts_worker.py")
    assert seen["req"]["language"] == "ru" and seen["req"]["items"][0]["text"] == "Привет"
    assert seen["req"]["sample"] == str(sample.resolve())


def test_separate_without_demucs_does_not_crash(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(voicemod, "demucs_available", lambda: False)
    log_file = tmp_path / "job.log"
    assert separate_music(tmp_path / "a.mp4", 0, 5, tmp_path, settings.voice,
                          Log(log_file, echo=False)) is None
    assert "py -m pip install demucs" in log_file.read_text(encoding="utf-8")


# ------------------------------------------------------------------ когда озвучивать

def test_voice_gate(settings, fake_llm):
    tr = english_transcript(6, (10, 10))
    assert voice_gate(settings, tr, fake_llm) == ""
    assert "DEEPSEEK_API_KEY" in voice_gate(settings, tr, None)
    ru = Transcript("ru", [Segment(0, 1, "привет", [Word(0, 1, " привет")])])
    assert "уже на русском" in voice_gate(settings, ru, fake_llm)
    tr.reliable, tr.speech_reason = False, "повтор символа"
    assert voice_gate(settings, tr, fake_llm) == "озвучка пропущена: речи нет (повтор символа)"


# ------------------------------------------------------------------ конфиг и флаги

def test_voice_config_defaults_and_toml(tmp_path):
    s = load_settings(tmp_path / "нет.toml", root=tmp_path)
    v = s.voice
    assert (v.enabled, v.engine, v.speaker, v.max_tempo, v.duck_db, v.subs) == \
        (False, "edge", "brian", 1.3, -18.0, True)
    assert v.edge_rate == "+0%" and v.edge_pitch == "+0Hz" and v.edge_max_rate == 1.15
    assert v.voice_mode == "voiceover"
    assert v.voice_check is True and v.voice_check_min == 7
    assert v.expression["enabled"] is True and v.expression["question_pitch"] == "+8Hz"
    assert v.model_path.replace("\\", "/").endswith("models/silero/v4_ru.pt") and v.sample_rate == 48000
    toml = tmp_path / "konveyer.toml"
    toml.write_text('[voice]\nenabled = true\nspeaker = "dmitry"\nduck_db = -12\n', encoding="utf-8")
    s = load_settings(toml, root=tmp_path)
    assert s.voice.enabled and s.voice.speaker == "dmitry" and s.voice.duck_db == -12.0
    toml.write_text('[voice]\nengine = "silero"\nspeaker = "xenia"\n', encoding="utf-8")
    assert load_settings(toml, root=tmp_path).voice.speaker == "xenia"
    for bad in ('speaker = "robot"', 'speaker = "aidar"', 'speaker = "xenia"',
                "duck_db = 6", "max_tempo = 3", "voice_check_min = 0", 'engine = "gtts"',
                'voice_mode = "rush"',
                'engine = "silero"\nspeaker = "brian"'):
        toml.write_text(f"[voice]\n{bad}\n", encoding="utf-8")
        with pytest.raises(KonveyerError, match="voice"):
            load_settings(toml, root=tmp_path)


def test_example_toml_has_voice_section():
    from conftest import ROOT
    text = (ROOT / "konveyer.toml.example").read_text(encoding="utf-8")
    assert "[voice]" in text and 'speaker = "brian"' in text and 'engine = "edge"' in text
    assert "enabled = false" in text and "[voice.expression]" in text
    assert "voice_check = true" in text and "voice_check_min = 7" in text
    assert 'voice_mode = "voiceover"' in text
    loaded = load_settings(ROOT / "konveyer.toml.example", root=ROOT)
    assert loaded.voice.engine == "edge" and loaded.voice.speaker == "brian"
    assert loaded.voice.expression["exclamation_pitch"] == "+12Hz"
    assert loaded.voice.voice_check is True and loaded.voice.voice_check_min == 7
    assert loaded.voice.voice_mode == "voiceover"


def test_cli_voice_flags(settings, tmp_path):
    p = cli.build_parser()
    a = p.parse_args(["run", "u", "--voice", "--voice-speaker", "dmitry", "--voice-no-subs", "--separate"])
    assert cli._job_options(a, settings) == {"voice": True, "voice_speaker": "dmitry",
                                             "voice_subs": False, "separate": True}
    assert "voice" not in cli._job_options(p.parse_args(["run", "u"]), settings)
    assert cli._job_options(p.parse_args(["redo", "last", "--no-voice"]), settings) == {"voice": False}
    with pytest.raises(SystemExit):
        p.parse_args(["run", "u", "--voice-speaker", "robot"])
    with pytest.raises(SystemExit):
        p.parse_args(["run", "u", "--voice-speaker", "andrew"])
    with pytest.raises(KonveyerError, match="не подходит"):
        cli._job_options(p.parse_args(["run", "u", "--voice-speaker", "baya"]), settings)
    silero = cli._job_options(p.parse_args(
        ["run", "u", "--voice", "--voice-engine", "silero", "--voice-speaker", "baya"]), settings)
    assert silero["voice_engine"] == "silero" and silero["voice_speaker"] == "baya"
    with pytest.raises(KonveyerError, match="voice-sample"):
        cli._job_options(p.parse_args(["run", "u", "--voice", "--voice-engine", "xtts"]), settings)
    sample = tmp_path / "голос.wav"
    sample.write_bytes(b"RIFF")
    opts = cli._job_options(p.parse_args(["run", "u", "--voice", "--voice-sample", str(sample)]), settings)
    assert opts["voice_engine"] == "xtts" and opts["voice_sample"] == str(sample.resolve())
    with pytest.raises(KonveyerError, match="нет файла"):
        cli._job_options(p.parse_args(["run", "u", "--voice-sample", str(tmp_path / "нет.wav")]), settings)
    mode = cli._job_options(p.parse_args(["run", "u", "--voice", "--voice-mode", "fit"]), settings)
    assert mode["voice_mode"] == "fit"
    mode = cli._job_options(p.parse_args(
        ["redo", "last", "--voice", "--voice-mode", "voiceover"]), settings)
    assert mode["voice_mode"] == "voiceover"
    with pytest.raises(SystemExit):
        p.parse_args(["run", "u", "--voice-mode", "rush"])
    job = Job.create(settings, "a.mp4", {"voice_mode": "fit"})
    assert settings_for_job(settings, job).voice.voice_mode == "fit"
    a = p.parse_args(["clip", "last", "1", "--voice", "--voice-speaker", "emma"])
    assert cli._voice_options(a, settings) == {"voice": True, "voice_speaker": "emma"}
    a = p.parse_args(["clip", "last", "1", "--voice", "--voice-engine", "silero", "--voice-speaker", "eugene"])
    assert cli._voice_options(a, settings)["voice_speaker"] == "eugene"


def test_settings_for_job_voice(settings):
    job = Job.create(settings, "a.mp4", {"voice": True, "voice_speaker": "kseniya", "voice_subs": False,
                                         "separate": True})
    s = settings_for_job(settings, job)
    assert s.voice.enabled and s.voice.speaker == "kseniya" and not s.voice.subs and s.voice.separate
    assert not settings.voice.enabled


def test_clip_voice_saves_options(settings, monkeypatch):
    job = Job.create(settings, "a.mp4")
    job.data["clips"] = [{"id": 1, "start": 0.0, "end": 5.0, "file": "clips/clip_01/clip.mp4"}]
    job.save()
    monkeypatch.setattr(Pipeline, "rerender", lambda self, ids: job)
    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: settings)
    assert cli.main(["clip", job.id, "1", "--voice", "--voice-speaker", "emma"]) == 0
    assert Job.load(job.dir).data["options"] == {"voice": True, "voice_speaker": "emma"}


def test_doctor_voice_rows(settings, tmp_path, monkeypatch):
    settings.voice.model_path = str(tmp_path / "v4_ru.pt")
    rows = {name: (ok, note) for name, ok, note in cli.voice_rows(settings)}
    assert set(rows) == {"Edge TTS", "torch", "Silero v4_ru", "XTTS v2", "demucs"}
    assert rows["Silero v4_ru"][0] is False and "model_path" in rows["Silero v4_ru"][1]
    (tmp_path / "v4_ru.pt").write_bytes(b"x")
    rows = {name: ok for name, ok, _ in cli.voice_rows(settings)}
    assert rows["Silero v4_ru"] is True
    settings.voice.xtts_python = str(tmp_path / "нет.exe")
    ok, note = cli._xtts_status(settings)
    assert not ok and "xtts_python" in note
    names = [r[0] for r in cli.doctor_report(settings)]
    assert "Silero v4_ru" in names and "demucs" in names


# ------------------------------------------------------------------ сквозной прогон

@pytest.fixture
def small_voice(settings, monkeypatch):
    settings.voice.engine = "silero"
    settings.voice.speaker = "aidar"
    settings.render.width, settings.render.height = 540, 960
    settings.render.caption_size = 40
    settings.moments.whole_if_shorter = 30
    engine = FakeEngine()
    monkeypatch.setattr(voicemod, "make_engine", lambda *a: engine)
    settings._fake_engine = engine
    return settings


def _run(settings, source, options, llm, tr=None, encoder=None):
    job = Job.create(settings, str(source), options)
    Pipeline(job, settings, Log(job.log_file, echo=False), llm=llm,
             transcriber=FakeTranscriber(tr), encoder=encoder).run()
    return Job.load(job.dir)


@needs_ffmpeg
def test_pipeline_voice_mixes_into_clip(small_voice, wide_video, fast_encoder):
    llm = FakeLLM()
    job = _run(small_voice, wide_video, {"voice": True}, llm, encoder=fast_encoder)
    clip = job.data["clips"][0]
    folder = job.dir / "clips" / "clip_01"
    assert clip["voice"].startswith("aidar, закадр, фраз:")
    assert clip["captions"] == "перевод на русский" and (folder / "captions.ass").exists()
    graph = (folder / "filter.txt").read_text(encoding="utf-8")
    assert "amix=inputs=2" in graph and "volume='1-0.8741*" in graph
    assert (folder / "voice.wav").exists() and ffprobe(job.dir / clip["file"])["has_audio"]
    spoken = [t for call in small_voice._fake_engine.calls for t in call]
    # латиница и цифры уже словами; не влезшее DeepSeek сокращает до «Коротко.»
    assert spoken and all(t.startswith("РУ ворд") or t == "Коротко." for t in spoken)
    assert any(t.startswith("РУ ворд") for t in spoken)
    assert "word" in (folder / "captions.ass").read_text(encoding="utf-8")
    log = job.log_file.read_text(encoding="utf-8")
    assert "Озвучка" in log and "озвучка — aidar" in log


@needs_ffmpeg
def test_pipeline_voice_no_subs(small_voice, wide_video, fast_encoder):
    job = _run(small_voice, wide_video, {"voice": True, "voice_subs": False}, FakeLLM(),
               encoder=fast_encoder)
    clip = job.data["clips"][0]
    assert clip["captions"] == "убраны (озвучка)" and clip["voice"]
    assert not (job.dir / "clips" / "clip_01" / "captions.ass").exists()


@needs_ffmpeg
def test_pipeline_voice_skipped_without_speech(small_voice, wide_video, fast_encoder, monkeypatch):
    def never(*a):
        raise AssertionError("движок не должен грузиться без речи")

    monkeypatch.setattr(voicemod, "make_engine", never)
    music = Transcript("ja", [Segment(0, 8, "ルルルルルル", [Word(0, 8, " ルルルルルル")], 0.82, -1.05, 2.9)])
    job = _run(small_voice, wide_video, {"voice": True}, FakeLLM(), tr=music, encoder=fast_encoder)
    log = job.log_file.read_text(encoding="utf-8")
    assert log.count("озвучка пропущена: речи нет") == 1
    assert "voice" not in job.data["clips"][0]
    assert "amix" not in (job.dir / "clips" / "clip_01" / "filter.txt").read_text(encoding="utf-8")


@needs_ffmpeg
def test_pipeline_without_voice_unchanged(small_voice, wide_video, fast_encoder):
    job = _run(small_voice, wide_video, {}, FakeLLM(), encoder=fast_encoder)
    assert small_voice._fake_engine.calls == []
    graph = (job.dir / "clips" / "clip_01" / "filter.txt").read_text(encoding="utf-8")
    assert graph.endswith(";[0:a]aresample=48000[a]")
