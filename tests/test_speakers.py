"""Спикеры: метки на словах, цвета субтитров, голоса Edge. Диаризация и сеть подменены."""
from __future__ import annotations

import builtins

import numpy as np
import pytest

from konveyer import cli
from konveyer.captions import (WHITE_TAG, Cue, build_ass, caption_ass, make_big_cues, make_cues)
from konveyer.config import EDGE_VOICES, load_settings
from konveyer.jobs import Job
from konveyer.log import Log
from konveyer.pipeline import settings_for_job
from konveyer.speakers import (PyannoteDiarizer, Turn, _gender, apply_turns, attach_speakers,
                               build_cast, canon_turns, edge_voice_for, hf_token,
                               resolve_overlaps, smooth_turns, smooth_words, speaker_color)
from konveyer.transcript import Segment, Transcript, Word
from konveyer.util import KonveyerError
from konveyer.voice import EdgeEngine, VoiceOver, sentence_groups


COLORS = ["#FFFFFF", "#FFE14A", "#7EC8FF"]
MIX = 48000


class Boom:
    def turns(self, audio, sample_rate):
        raise RuntimeError("cuda упала")


class Scripted:
    def __init__(self, turns):
        self.turns_in = turns
        self.calls = 0

    def turns(self, audio, sample_rate):
        self.calls += 1
        assert sample_rate == 16000
        assert len(audio) > 0
        return list(self.turns_in)


def _tr():
    return Transcript("ru", [Segment(0, 3, "раз два три", [
        Word(0.2, 0.6, " раз"),
        Word(1.2, 1.6, " два"),
        Word(1.7, 2.1, " три"),
    ])])


def test_canon_and_apply_by_midpoint():
    turns = canon_turns([
        Turn(1.0, 3.0, "GUEST"),
        Turn(0.0, 1.0, "SPEAKER_00"),
        Turn(3.0, 3.0, "пусто"),
    ])
    # порядок появления: GUEST первый в списке, потом SPEAKER_00
    assert [t.speaker for t in turns] == ["S1", "S2"]
    tr = _tr()
    n = apply_turns(tr, [Turn(0.0, 1.0, "SPEAKER_00"), Turn(1.0, 3.0, "GUEST")])
    words = tr.segments[0].words
    assert n == 3
    assert [w.speaker for w in words] == ["S1", "S2", "S2"]
    assert tr.segments[0].speaker == ""
    back = Transcript.from_dict(tr.to_dict())
    assert [w.speaker for w in back.words] == ["S1", "S2", "S2"]


def test_cues_split_on_speaker_and_big_does_not_join():
    words = [Word(0, 0.3, " раз", "S1"), Word(0.4, 0.7, " два", "S2"), Word(0.8, 1.1, " три", "S2")]
    cues = make_cues(words)
    assert [c.speaker for c in cues] == ["S1", "S2"]
    assert cues[0].text == "раз" and cues[1].text.startswith("два")
    big = make_big_cues(words)
    assert [c.speaker for c in big] == ["S1", "S2"]
    assert all(len(c.words) == 1 or c.speaker == "S2" for c in big)
    same = make_cues([Word(0, 0.3, " раз"), Word(0.4, 0.7, " два")])
    assert len(same) == 1 and same[0].speaker == ""


def test_karaoke_and_plain_and_big_colors():
    words = [Word(0.0, 0.4, " раз", "S1"), Word(0.5, 0.9, " два", "S2")]
    karaoke = build_ass(make_cues(words), 1080, 1920, karaoke=True, colors=COLORS)
    assert "&H00FFFFFF" in karaoke
    assert "&H004AE1FF" in karaoke
    events = [line for line in karaoke.splitlines() if line.startswith("Dialogue:")]
    yellow_lines = [line for line in events if "&H004AE1FF" in line]
    assert yellow_lines and WHITE_TAG in yellow_lines[0]
    plain = build_ass([Cue(0, 2, "Привет всем", speaker="S2")], 1280, 720,
                      karaoke=False, colors=COLORS)
    assert "PlayResX: 1280" in plain and "&H004AE1FF" in plain
    assert "{\\1c&H00FFFF&}" not in plain
    big = build_ass([Cue(0, 1, "Раз", speaker="S3")], 1080, 1920, style="big",
                    colors=COLORS, fonts=["Arial", "Arial", "Comic Sans MS"])
    assert "&H00FFC87E" in big and "{\\fnComic Sans MS}" in big
    alone = build_ass([Cue(0, 1, "Раз")], 1080, 1920, karaoke=False, colors=COLORS)
    assert "{\\1c" not in alone


def test_caption_ass_uses_settings_for_both_layouts(settings):
    cues = [Cue(0.2, 1.2, "раз два", [Word(0.2, 0.6, "раз", "S1"), Word(0.7, 1.1, "два", "S2")], "S1"),
            Cue(1.4, 2.0, "три", speaker="S2")]
    shorts = caption_ass(cues, settings, 1080, 1920, karaoke=True, align=8, margin_v=900)
    full = caption_ass(cues, settings, 1280, 720, karaoke=False, align=2, margin_v=80,
                       max_line=42, size=42)
    assert "&H004AE1FF" in shorts and "&H004AE1FF" in full
    settings.subs.speakers = False
    off = caption_ass(cues, settings, 1080, 1920, karaoke=True, align=2, margin_v=200)
    assert "{\\1c&H004AE1FF&}" not in off


def test_attach_falls_back_without_touching_words(settings, tmp_path):
    job = Job.create(settings, "ролик.mp4")
    tr = _tr()
    log = Log(tmp_path / "a.log", echo=False)
    attach_speakers(job, settings, log, tr, np.zeros(16, np.float32), 16000, diarizer=Boom())
    assert all(w.speaker == "" for w in tr.words)
    text = (tmp_path / "a.log").read_text(encoding="utf-8")
    assert "одним стилем" in text and "cuda" in text
    scripted = Scripted([Turn(0, 1, "A"), Turn(1, 4, "B")])
    attach_speakers(job, settings, log, tr, np.ones(16, np.float32), 16000, diarizer=scripted)
    assert scripted.calls == 1
    assert [w.speaker for w in tr.words] == ["S1", "S2", "S2"]
    meta = Job.load(job.dir).data["meta"]
    assert meta["speakers"] == ["S1", "S2"]
    assert meta["speaker_gender"] == {"S1": "", "S2": ""}
    assert meta["speaker_cast"] == {"S1": "brian", "S2": "ava"}
    settings.subs.speakers = False
    tr2 = _tr()
    attach_speakers(job, settings, log, tr2, np.ones(8), 16000, diarizer=Boom())
    assert all(w.speaker == "" for w in tr2.words)


def test_missing_token_does_not_import_pyannote(settings, monkeypatch):
    assert hf_token(settings) == ""
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if str(name).split(".")[0] in ("pyannote", "torch"):
            raise AssertionError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(KonveyerError, match="HF_TOKEN"):
        PyannoteDiarizer(settings).turns(np.zeros(8, np.float32), 16000)


def test_flags_config_and_voice_map(settings, tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(["run", "u", "--full", "--no-speakers", "--one-voice", "--voice"])
    opts = cli._job_options(args, settings)
    assert opts["speakers"] is False and opts["per_speaker"] is False and opts["full"] is True
    full_args = parser.parse_args(["full", "u", "--no-speakers", "--one-voice"])
    assert full_args.no_speakers and full_args.one_voice
    tuned = settings_for_job(settings, Job.create(settings, "a.mp4", opts))
    assert tuned.subs.speakers is False and tuned.voice.per_speaker is False
    assert edge_voice_for(settings, "S2") == EDGE_VOICES["ava"]
    assert edge_voice_for(settings, "S1") == ""
    settings.voice.per_speaker = False
    assert edge_voice_for(settings, "S2") == ""
    assert sentence_groups(["раз", "два."], ["S1", "S2"]) == [[0], [1]]
    assert sentence_groups(["раз", "два."]) == [[0, 1]]
    toml = tmp_path / "k.toml"
    toml.write_text(
        '[subs]\nspeaker_colors = ["white", "голубой"]\nspeaker_fonts = ["Segoe UI"]\n'
        '[voice]\nspeaker_voices = ["brian", "emma"]\n',
        encoding="utf-8")
    loaded = load_settings(toml, root=tmp_path)
    assert loaded.subs.speaker_colors == ["#FFFFFF", "#7EC8FF"]
    assert loaded.subs.speaker_fonts == ["Segoe UI"]
    assert loaded.voice.speaker_voices == ["brian", "emma"]
    assert loaded.voice.male_voices == ["brian", "florian", "dmitry"]
    assert loaded.voice.per_speaker is True
    toml.write_text('[voice]\nmale_voices = ["andrew"]\n', encoding="utf-8")
    with pytest.raises(KonveyerError, match="andrew"):
        load_settings(toml, root=tmp_path)
    toml.write_text('[voice]\nfemale_voices = ["brian"]\n', encoding="utf-8")
    with pytest.raises(KonveyerError, match="female_voices"):
        load_settings(toml, root=tmp_path)
    toml.write_text('[subs]\nspeaker_colors = ["нет такого"]\n', encoding="utf-8")
    with pytest.raises(KonveyerError, match="speaker_colors"):
        load_settings(toml, root=tmp_path)


def test_edge_switches_voice_per_speaker(settings, tmp_path):
    heard = []

    def fn(text, voice, rate, pitch, volume):
        heard.append(voice)
        return np.full(int(0.35 * MIX), 0.2, np.float32)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    settings.voice.voice_check = False
    over = VoiceOver(tmp_path, settings, Log(None, echo=False), None, engine=eng)
    cues = [Cue(0.0, 0.4, "Привет всем.", speaker="S1"),
            Cue(1.2, 1.6, "И тебе привет.", speaker="S2")]
    track = over.make(tmp_path, cues, 4.0, 1)
    assert track is not None
    assert EDGE_VOICES["brian"] in heard and EDGE_VOICES["ava"] in heard
    assert "brian+ava" in track.note

    heard.clear()
    settings.voice.per_speaker = False
    other = tmp_path / "one"
    other.mkdir()
    over2 = VoiceOver(other, settings, Log(None, echo=False), None, engine=eng)
    track2 = over2.make(other, cues, 4.0, 1)
    assert track2 is not None
    assert heard and set(heard) == {EDGE_VOICES["brian"]}


def _tone(hz: float, sec: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


def test_cast_follows_gender_and_stays_on_the_speaker(settings, tmp_path):
    assert build_cast(settings, ["S1"], {}) == {"S1": "brian"}
    assert build_cast(settings, ["S1", "S2"], {}) == {"S1": "brian", "S2": "ava"}
    assert build_cast(settings, ["S1", "S2"], {"S1": "female", "S2": "male"}) == {
        "S1": "ava", "S2": "brian"}
    four = build_cast(settings, ["S1", "S2", "S3", "S4"],
                      {name: "male" for name in ("S1", "S2", "S3", "S4")})
    assert [four[name] for name in ("S1", "S2", "S3", "S4")] == [
        "brian", "florian", "dmitry", "ava"]
    assert speaker_color(settings, "S1") == "#FFFFFF"
    assert speaker_color(settings, "S2") == "#FFE14A"

    sr = 16000
    audio = np.concatenate([_tone(120, 1.0, sr), _tone(220, 1.0, sr)])
    job = Job.create(settings, "голоса.mp4")
    tr = Transcript("en", [Segment(0, 2, "hi there", [
        Word(0.2, 0.6, " hi"),
        Word(1.2, 1.6, " there"),
    ])])
    log = Log(tmp_path / "g.log", echo=False)
    attach_speakers(job, settings, log, tr, audio, sr,
                    diarizer=Scripted([Turn(0, 1, "LOW"), Turn(1, 2, "HIGH")]))
    meta = Job.load(job.dir).data["meta"]
    assert meta["speaker_gender"] == {"S1": "male", "S2": "female"}
    assert meta["speaker_cast"] == {"S1": "brian", "S2": "ava"}
    note = (tmp_path / "g.log").read_text(encoding="utf-8")
    assert "S1 мужской → brian, #FFFFFF" in note
    assert "S2 женский → ava, #FFE14A" in note

    heard = []

    def fn(text, voice, rate, pitch, volume):
        heard.append(voice)
        return np.full(int(0.35 * MIX), 0.2, np.float32)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    over = VoiceOver(job.dir, settings, Log(None, echo=False), None, engine=eng)
    clip = tmp_path / "s2"
    clip.mkdir()
    track = over.make(clip, [Cue(0.0, 0.4, "И тебе привет.", speaker="S2")], 4.0, 1)
    assert track is not None and heard == [EDGE_VOICES["ava"]]

    alone = Job.create(settings, "одна.mp4")
    one = Transcript("en", [Segment(0, 1, "hello", [Word(0.2, 0.6, " hello")])])
    attach_speakers(alone, settings, log, one, _tone(220, 1.0, sr), sr,
                    diarizer=Scripted([Turn(0, 1, "HIGH")]))
    cast = Job.load(alone.dir).data["meta"]["speaker_cast"]
    assert cast == {"S1": "ava"}
    assert edge_voice_for(settings, "S1", cast) == EDGE_VOICES["ava"]


def test_saved_cast_is_not_rebuilt_for_one_clip(settings, tmp_path):
    job = Job.create(settings, "каст.mp4")
    job.data["meta"]["speakers"] = ["S1", "S2"]
    job.data["meta"]["speaker_cast"] = {"S1": "florian", "S2": "emma"}
    job.save()
    heard = []

    def fn(text, voice, rate, pitch, volume):
        heard.append(voice)
        return np.full(int(0.35 * MIX), 0.2, np.float32)

    eng = EdgeEngine(settings.voice, Log(None, echo=False), tmp_path, synth_fn=fn,
                     sleeper=lambda s: None)
    over = VoiceOver(job.dir, settings, Log(None, echo=False), None, engine=eng)
    clip = job.dir / "clip"
    clip.mkdir()
    track = over.make(clip, [Cue(0.0, 0.4, "Добрый вечер.", speaker="S2")], 4.0, 1)
    assert track is not None
    assert heard == [EDGE_VOICES["emma"]]
    assert edge_voice_for(settings, "S1", job.data["meta"]["speaker_cast"]) == EDGE_VOICES["florian"]
    assert speaker_color(settings, "S2") == "#FFE14A"


def test_smoothing_keeps_one_voice_per_speaker():
    turns = smooth_turns([Turn(0, 10, "S1"), Turn(10, 10.8, "S2"), Turn(10.8, 20, "S1"),
                          Turn(20, 21, "S2")])
    assert [t.speaker for t in turns] == ["S1", "S1", "S1", "S2"]
    words = [Word(0.0, 0.4, " a", "S3"), Word(0.5, 0.9, " b", "S3"), Word(1.0, 1.2, " who", ""),
             Word(1.3, 1.6, " c", "S3"), Word(1.7, 2.0, " x", "S1"), Word(2.1, 2.4, " d", "S3"),
             Word(3.0, 6.0, " long", "S1"), Word(6.1, 6.5, " e", "S3")]
    smooth_words(words)
    assert [w.speaker for w in words] == ["S3", "S3", "S3", "S3", "S3", "S3", "S1", "S3"]


def test_gender_is_conservative_and_speaker_map_wins(settings, tmp_path):
    assert _gender(None) == ""
    assert _gender(130) == "male" and _gender(185) == "male" and _gender(196) == "male"
    assert _gender(230) == "female"
    toml = tmp_path / "k.toml"
    toml.write_text('[voice]\nspeaker_map = { S2 = "dmitry" }\n', encoding="utf-8")
    s = load_settings(toml, root=tmp_path)
    assert s.voice.speaker_map == {"S2": "dmitry"}
    cast = build_cast(s, ["S1", "S2"], {"S1": "male", "S2": "female"})
    assert cast == {"S1": "brian", "S2": "dmitry"}
    assert edge_voice_for(s, "S2", {"S2": "ava"}) == EDGE_VOICES["dmitry"]
    s2 = load_settings(toml, overrides={"voice.speaker_map": "S3=florian"}, root=tmp_path)
    assert s2.voice.speaker_map == {"S2": "dmitry", "S3": "florian"}
    toml.write_text('[voice]\nspeaker_map = { S2 = "andrew" }\n', encoding="utf-8")
    with pytest.raises(KonveyerError, match="speaker_map"):
        load_settings(toml, root=tmp_path)


def test_overlap_handover_goes_to_new_speaker_backchannel_to_owner():
    turns = resolve_overlaps([Turn(33.3, 81.3, "A"), Turn(78.4, 80.7, "B"),
                              Turn(96.7, 189.9, "A"), Turn(187.7, 190.8, "B"),
                              Turn(191.5, 212.2, "B")])
    got = [(round(t.start, 1), round(t.end, 1), t.speaker) for t in turns]
    assert got == [(33.3, 81.3, "A"), (96.7, 187.7, "A"), (187.7, 190.8, "B"), (191.5, 212.2, "B")]
