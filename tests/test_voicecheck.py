"""Сверка озвучки с субтитрами: нормализация, пропуск, ворота перед Telegram."""
from __future__ import annotations

import json

from konveyer import cli
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.pipeline import settings_for_job
from konveyer.util import write_json
from konveyer.voice import Phrase
from konveyer.voicecheck import (align_heard, gate_issues, markup_in, normalize_said,
                                 review_voice, without_neighbor_edge)


def _phrase(text="привет мир", heard_ok=True):
    return Phrase(0.0, 1.0, text, spoken=text, duration=0.8, status="ok", origin=0.0)


def test_normalize_ignores_case_yo_punct_and_numbers():
    assert normalize_said("Привет, Ёлка!") == normalize_said("привет елка")
    assert normalize_said("в 2024 году") == normalize_said("в две тысячи двадцать четвертом году")
    assert normalize_said("5") == normalize_said("пять") == "#"
    assert normalize_said("двадцать пять") == "#"
    assert normalize_said("по-русски") == "по русски"


def test_align_heard_keeps_neighbor_first_word_out():
    phrases = [
        Phrase(0.0, 1.0, "думаю одно", spoken="думаю одно", duration=1.0, status="ok"),
        Phrase(1.02, 1.0, "из пользы", spoken="из пользы", duration=0.8, status="ok"),
    ]
    words = [
        (0.10, 0.40, "думаю"),
        (0.50, 0.90, "одно"),
        (1.00, 1.16, "из"),
        (1.30, 1.70, "пользы"),
    ]
    heard = align_heard(words, phrases)
    assert heard[0] == "думаю одно"
    assert heard[1] == "из пользы"


def test_without_neighbor_edge_drops_only_the_borrowed_word():
    assert without_neighbor_edge("думаю одно из", "думаю одно", "", "из пользы") == "думаю одно"
    assert without_neighbor_edge("из думаю одно", "думаю одно", "скажи из", "") == "думаю одно"
    assert without_neighbor_edge("скажи да", "скажи да", "", "да нет") == "скажи да"


def test_review_does_not_score_neighbor_word_as_extra(settings, tmp_path):
    settings.voice.voice_check = True
    folder = tmp_path / "clip_bleed"
    folder.mkdir()
    (folder / "voice.wav").write_bytes(b"RIFF")

    class LLM:
        def chat(self, system, user, json_mode=True, temperature=None):
            rows = json.loads(user)
            bad = any(r["услышано"] != r["субтитры"] for r in rows)
            return json.dumps({
                "score": 4 if bad else 9,
                "phrases": [{"i": r["i"], "ok": not bad,
                             "problem": "лишнее" if bad else ""} for r in rows],
            }, ensure_ascii=False)

    phrases = [
        Phrase(0.0, 1.0, "думаю одно", spoken="думаю одно", duration=1.0, status="ok"),
        Phrase(1.2, 1.0, "из пользы", spoken="из пользы", duration=0.8, status="ok"),
    ]
    payload = review_voice(folder, phrases, settings, Log(None, echo=False), LLM(), 1,
                           hear=lambda path, items: ["думаю одно из", "из пользы"])
    assert payload["score"] == 9 and payload["ok"] is True
    assert payload["phrases"][0]["problem"] == ""
    assert payload["phrases"][0]["heard"] == "думаю одно"


def test_markup_in_catches_break_tags():
    assert markup_in('меньше чем <break time="200ms"/>')
    assert markup_in("break time=")
    assert not markup_in("привет, как дела")


def test_review_voice_writes_json_and_log(settings, tmp_path):
    settings.voice.voice_check = True
    folder = tmp_path / "clip_01"
    folder.mkdir()
    (folder / "voice.wav").write_bytes(b"RIFF")
    log_file = tmp_path / "job.log"
    log = Log(log_file, echo=False)

    class LLM:
        def chat(self, system, user, json_mode=True, temperature=None):
            rows = json.loads(user)
            return json.dumps({"score": 9, "phrases": [
                {"i": rows[0]["i"], "ok": True, "problem": ""}]}, ensure_ascii=False)

    payload = review_voice(folder, [_phrase()], settings, log, LLM(), 1,
                           hear=lambda path, phrases: ["Привет, мир!"])
    assert payload["ok"] is True and payload["score"] == 9 and payload["skipped"] is False
    saved = json.loads((folder / "voice_check.json").read_text(encoding="utf-8"))
    assert saved["phrases"][0]["ok"] is True and saved["phrases"][0]["problem"] == ""
    assert "клип 1: проверка озвучки 9 из 10" in log_file.read_text(encoding="utf-8")


def test_review_voice_skips_without_whisper_or_deepseek(settings, tmp_path):
    settings.voice.voice_check = True
    folder = tmp_path / "clip_02"
    folder.mkdir()
    (folder / "voice.wav").write_bytes(b"RIFF")
    log_file = tmp_path / "job.log"
    log = Log(log_file, echo=False)

    def boom(path, phrases):
        raise RuntimeError("нет модели")

    skipped = review_voice(folder, [_phrase()], settings, log, object(), 2, hear=boom)
    assert skipped["skipped"] is True and skipped["ok"] is True
    no_key = review_voice(folder, [_phrase()], settings, log, None, 2,
                          hear=lambda path, phrases: ["привет"])
    assert no_key["skipped"] is True and "DeepSeek" in no_key["reason"]
    text = log_file.read_text(encoding="utf-8")
    assert text.count("проверка озвучки пропущена") == 2


def test_review_treats_letter_names_and_latin_as_normal(settings, tmp_path):
    """«зэ» и «джи-пи-ти» — как надо прочитать «З» и GPT, не ошибка букв."""
    settings.voice.voice_check = True
    folder = tmp_path / "clip_ze"
    folder.mkdir()
    (folder / "voice.wav").write_bytes(b"RIFF")

    class LLM:
        def chat(self, system, user, json_mode=True, temperature=None):
            row = json.loads(user)[0]
            same = row["субтитры"] == row["услышано"]
            return json.dumps({
                "score": 9 if same else 2,
                "phrases": [{"i": row["i"], "ok": same, "problem": "" if same else "буквы"}],
            }, ensure_ascii=False)

    payload = review_voice(
        folder, [_phrase("буква З и GPT")], settings, Log(None, echo=False), LLM(), 4,
        hear=lambda path, phrases: ["буква зэ и джи-пи-ти"])
    assert payload["score"] == 9 and payload["ok"] is True
    assert payload["phrases"][0]["problem"] == ""
    assert payload["phrases"][0]["text"] == "буква З и GPT"


def test_review_voice_forces_markup_and_caps_score(settings, tmp_path):
    settings.voice.voice_check = True
    folder = tmp_path / "clip_03"
    folder.mkdir()
    (folder / "voice.wav").write_bytes(b"RIFF")

    class LLM:
        def chat(self, system, user, json_mode=True, temperature=None):
            return json.dumps({"score": 9, "phrases": [{"i": 0, "ok": True, "problem": ""}]})

    payload = review_voice(folder, [_phrase("привет")], settings, Log(None, echo=False), LLM(), 3,
                           hear=lambda path, phrases: ['меньше чем <break time="200ms"/>'])
    assert payload["score"] == 4 and payload["ok"] is False
    assert payload["phrases"][0]["problem"] == "разметка"
    assert payload["phrases"][0]["critical"] is True


def _job_with_check(settings, payload):
    job = Job.create(settings, "a.mp4")
    job.data["clips"] = [{"id": 1, "file": "clips/clip_01/clip.mp4"}]
    folder = job.clip_dir(1)
    folder.mkdir(parents=True)
    write_json(folder / "voice_check.json", payload)
    return job


def test_gate_blocks_low_score_and_critical(settings):
    settings.voice.voice_check = True
    settings.voice.voice_check_min = 7
    job = _job_with_check(settings, {
        "skipped": False, "score": 6, "ok": False,
        "phrases": [{"i": 0, "ok": False, "problem": "обрыв", "critical": True}],
    })
    issues = gate_issues(job, settings)
    assert any("ниже порога 7" in item for item in issues)
    assert any("фраза оборвана" in item for item in issues)

    ok = _job_with_check(settings, {"skipped": False, "score": 7, "ok": True, "phrases": [
        {"i": 0, "ok": True, "problem": "", "critical": False}]})
    assert gate_issues(ok, settings) == []


def test_gate_ignores_skip_and_disabled_flag(settings):
    settings.voice.voice_check = True
    job = _job_with_check(settings, {"skipped": True, "score": 1, "ok": True, "phrases": []})
    assert gate_issues(job, settings) == []
    settings.voice.voice_check = False
    bad = _job_with_check(settings, {"skipped": False, "score": 1, "phrases": [
        {"problem": "разметка", "critical": True}]})
    assert gate_issues(bad, settings) == []


def test_doctor_voice_check_row(settings):
    settings.voice.voice_check = False
    name, ok, note = cli.voice_check_row(settings)
    assert name == "проверка озвучки" and ok and "выключена" in note
    settings.voice.voice_check = True
    settings.voice.voice_check_min = 7
    name, ok, note = cli.voice_check_row(settings)
    assert name == "проверка озвучки"
    assert "порог 7" in note or "faster-whisper" in note
    names = [row[0] for row in cli.doctor_report(settings)]
    assert "проверка озвучки" in names
    assert names.count("проверка озвучки") == 1


def test_no_voice_check_flag_reaches_job(settings):
    settings.voice.voice_check = True
    parser = cli.build_parser()
    args = parser.parse_args(["redo", "last", "--voice", "--no-voice-check"])
    opts = cli._job_options(args, settings)
    assert opts["voice_check"] is False
    tuned = settings_for_job(settings, Job.create(settings, "a.mp4", opts))
    assert tuned.voice.voice_check is False
    send_args = parser.parse_args(["send", "last", "--no-voice-check"])
    assert send_args.no_voice_check is True


def test_cmd_send_can_skip_voice_gate(settings, monkeypatch):
    settings.voice.voice_check = True
    job = Job.create(settings, "a.mp4")
    seen = {}

    def fake_send(job, tuned, log, **kwargs):
        seen["check"] = tuned.voice.voice_check

    monkeypatch.setattr("konveyer.telegram.send_job", fake_send)
    args = cli.build_parser().parse_args(["send", job.id, "--no-voice-check"])
    assert cli.cmd_send(args, settings) == 0
    assert seen["check"] is False


def _wav(path, seconds, cut_at=None, rate=16000):
    import wave
    import numpy as np
    n = int(seconds * rate)
    tt = np.arange(n) / rate
    env = np.ones(n)
    if cut_at is None:
        fade = (tt > seconds - 0.4)
        env[fade] = np.clip((seconds - 0.1 - tt[fade]) / 0.3, 0, 1)
    else:
        env[tt >= cut_at] = 0.0
    x = (0.3 * np.sin(2 * np.pi * 220 * tt) * env * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())


class _CutLLM:
    def chat(self, system, user, json_mode=True, temperature=None):
        rows = json.loads(user)
        return json.dumps({"score": 4, "phrases": [
            {"i": r["i"], "ok": False, "problem": "обрыв"} for r in rows]}, ensure_ascii=False)


def test_missing_last_word_with_whole_audio_is_only_a_warning(settings, tmp_path):
    settings.voice.voice_check = True
    folder = tmp_path / "clip_whole"
    folder.mkdir()
    _wav(folder / "voice.wav", 2.0)                       # голос затухает сам — фраза целая
    ph = Phrase(0.0, 2.0, "ради других агентов", spoken="ради других агентов",
                duration=2.0, status="ok")
    payload = review_voice(folder, [ph], settings, Log(None, echo=False), _CutLLM(), 1,
                           hear=lambda path, items: ["ради других"])
    assert payload["score"] >= 7 and payload["ok"] is True
    assert payload["phrases"][0]["problem"] == "" and payload["phrases"][0]["critical"] is False
    assert "не подтверждён" in payload["phrases"][0]["warning"]


def test_cut_confirmed_by_sound_stays_critical_without_cap(settings, tmp_path):
    settings.voice.voice_check = True
    folder = tmp_path / "clip_cut"
    folder.mkdir()
    _wav(folder / "voice.wav", 2.0, cut_at=1.5)           # голос громкий и срезан на 1,5 с
    ph = Phrase(0.0, 1.5, "ради других агентов", spoken="ради других агентов",
                duration=1.5, status="ok")
    payload = review_voice(folder, [ph], settings, Log(None, echo=False), _CutLLM(), 1,
                           hear=lambda path, items: ["ради других"])
    assert payload["phrases"][0]["problem"] == "обрыв" and payload["phrases"][0]["critical"] is True
    assert payload["ok"] is False
