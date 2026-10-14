"""Сторож озвучки целого видео: проверка по voice.json и энергии voice.wav, починка кругами."""
from __future__ import annotations

import numpy as np

from konveyer.guard import check_phrases, guard_full, holes, voice_activity
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.util import write_json
from konveyer.voicecheck import gate_issues


def _p(origin, start, dur, spk="S1", voice="", status="ok", cue_end=None, text="фраза"):
    return {"origin": origin, "start": start, "duration": dur, "speaker": spk, "voice": voice,
            "status": status, "cue_end": origin + 1.0 if cue_end is None else cue_end,
            "text": text, "dropped": ""}


def test_activity_and_holes():
    rate = 1000
    x = np.zeros(10 * rate, np.float32)
    x[: 2 * rate] = 0.2
    x[int(6.5 * rate):] = 0.2
    act = voice_activity(x, rate)
    assert len(act) == 2 and abs(act[0][1] - 2.0) < 0.06
    found = holes([(0.0, 10.0)], act)
    assert len(found) == 1 and abs(found[0][0] - 2.0) < 0.06 and abs(found[0][1] - 6.5) < 0.06
    assert holes([(0.0, 10.0)], [(0.0, 4.0), (6.5, 10.0)]) == []


def test_check_finds_every_kind():
    ph = [_p(0.0, 0.0, 2.0, voice="a"),
          _p(2.0, 2.0, 0.0, status="fail"),
          _p(4.0, 6.0, 1.0, voice="b"),
          _p(7.0, 7.0, 2.0, spk="S2", status="cut"),
          _p(8.5, 9.0, 1.0, spk="S1", voice="a")]
    kinds = {f["kind"] for f in check_phrases(ph, [(0, 2), (6, 10)])}
    assert {"voice", "silent", "lag", "cut", "overlap"} <= kinds
    good = [_p(0.0, 0.0, 1.0), _p(1.2, 1.3, 1.0, spk="S2", voice="d")]
    assert check_phrases(good, [(0.0, 2.3)]) == []


def test_guard_repairs_then_gate_is_clean(settings, tmp_path):
    job = Job.create(settings, "сторож.mp4")
    folder = job.dir / "full"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "voice.json", {"phrases": [_p(0.0, 0.0, 0.0, status="fail")]})
    calls = {"n": 0}

    def rerender():
        calls["n"] += 1
        write_json(folder / "voice.json", {"phrases": [_p(0.0, 0.0, 1.0)]})

    res = guard_full(job, settings, Log(tmp_path / "g.log", echo=False), rerender=rerender, fix=True)
    assert res["ok"] and res["rounds"] == 1 and calls["n"] == 1
    assert not any("сторож" in i for i in gate_issues(job, settings))
    assert "сторож: озвучка в порядке" in (tmp_path / "g.log").read_text(encoding="utf-8")


def test_guard_reports_only_and_never_blocks_send(settings, tmp_path):
    job = Job.create(settings, "сторож2.mp4")
    folder = job.dir / "full"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "voice.json", {"phrases": [_p(65.0, 65.0, 0.0, status="fail", text="дыра")]})
    calls = {"n": 0}
    res = guard_full(job, settings, Log(tmp_path / "g.log", echo=False),
                     rerender=lambda: calls.__setitem__("n", calls["n"] + 1))
    assert not res["ok"] and res["rounds"] == 0 and calls["n"] == 0
    assert res["issues"][0].startswith("1:05 без звука")
    assert gate_issues(job, settings) == []
    assert res["message"].splitlines()[1].startswith("• без звука: 1:05")

def test_guard_finds_break_inside_sentence_and_groups_message():
    from konveyer.guard import message
    ph = [_p(35.6, 35.9, 1.5, text="меня и беспокоит: эти"),
          _p(38.4, 38.4, 1.7, text="огромные рои агентов."),
          _p(40.5, 40.5, 1.0, text="Новое предложение."),
          _p(42.5, 42.5, 1.0, text="Ещё одно.")]
    for p in ph:
        p["speaker"] = "S2"
    found = check_phrases(ph, [(35.9, 42.0)])
    br = [f for f in found if f["kind"] == "break"]
    assert len(br) == 1 and br[0]["at"] == 38.4 and "эти | огромные" in br[0]["text"]
    text = message(found)
    assert text.startswith("Сторож:") and "разрыв фразы: 0:38 (1.0 с)" in text
