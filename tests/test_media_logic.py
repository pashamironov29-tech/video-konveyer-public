"""timeline, captions, layout, facetrack, analysis — чистая логика без рендера."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_ffmpeg
from konveyer.analysis import loudness_db, parse_scene_times, speech_density
from konveyer.captions import (YELLOW_TAG, Cue, ass_escape, ass_time, build_ass, line_groups,
                               make_cues, remap_cues)
from konveyer.facetrack import crop_track, crop_x_expr
from konveyer.layout import (build_graph, choose_layout, drawtext, even, geometry, text_layout)
from konveyer.timeline import keep_intervals, kept_duration, map_range, map_time, select_expr
from konveyer.transcript import Segment, Transcript, Word

FONT = "C:/Windows/Fonts/arialbd.ttf"


# ------------------------------------------------------------------ timeline

def words_at(*spans):
    return [Word(a, b, f" w{i}") for i, (a, b) in enumerate(spans)]


def test_keep_intervals_cuts_pause():
    ws = words_at((0.0, 1.0), (3.0, 4.0))
    iv = keep_intervals(ws, 5.0, pause=0.6, pad=0.12)
    assert iv == [(0.0, 1.12), (2.88, 4.12)]
    assert kept_duration(iv) == pytest.approx(2.36)


def test_keep_intervals_small_gaps_stay():
    ws = words_at((0.1, 0.5), (0.8, 1.2), (1.5, 2.0))
    assert keep_intervals(ws, 2.3, pause=0.6) == [(0.0, 2.3)]


def test_keep_intervals_leading_silence():
    ws = words_at((2.0, 3.0))
    assert keep_intervals(ws, 3.2, pause=0.6, pad=0.1) == [(1.9, 3.2)]


def test_keep_intervals_empty():
    assert keep_intervals([], 4.0) == [(0.0, 4.0)]


def test_map_time_and_range():
    iv = [(0.0, 1.0), (3.0, 4.0)]
    assert map_time(0.5, iv) == 0.5
    assert map_time(3.5, iv) == 1.5
    assert map_time(2.0, iv) is None
    assert map_range(0.5, 3.5, iv) == (0.5, 1.5)
    assert map_range(1.5, 2.5, iv) is None
    assert map_range(1.5, 3.2, iv) == (1.0, 1.2)


def test_select_expr():
    assert select_expr([(0, 1), (2.5, 3)]) == "between(t,0.000,1.000)+between(t,2.500,3.000)"


# ------------------------------------------------------------------ captions

def test_make_cues_split_by_gap_and_length():
    ws = [Word(i * 0.3, i * 0.3 + 0.25, f" слово{i}") for i in range(4)]
    ws += [Word(5.0, 5.3, " после"), Word(5.4, 5.7, " паузы")]
    cues = make_cues(ws, max_chars=38)
    assert [c.text for c in cues] == ["слово0 слово1 слово2 слово3", "после паузы"]
    assert cues[0].end <= cues[1].start
    assert all(len(c.text) <= 38 for c in make_cues([Word(i, i + 0.5, " длинноеслово") for i in range(10)]))


def test_make_cues_sentence_end():
    ws = [Word(0, .3, " Привет"), Word(.4, .7, " всем,"), Word(.8, 1.1, " друзья."), Word(1.2, 1.5, " Начнём")]
    assert [c.text for c in make_cues(ws)] == ["Привет всем, друзья.", "Начнём"]


def test_remap_cues():
    cues = [Cue(0.2, 0.9, "раз", [Word(0.2, 0.9, "раз")]), Cue(1.5, 2.5, "пауза"),
            Cue(3.1, 3.9, "два", [Word(3.1, 3.9, "два")])]
    out = remap_cues(cues, [(0.0, 1.0), (3.0, 4.0)])
    assert [c.text for c in out] == ["раз", "два"]
    assert out[1].start == pytest.approx(1.1) and out[1].words[0].end == pytest.approx(1.9)


def test_ass_helpers():
    assert ass_time(3661.5) == "1:01:01.50"
    assert ass_time(-1) == "0:00:00.00"
    assert ass_escape("a{b}\\c\nd") == "a(b)/c d"
    assert line_groups(["коротко"], 20) == [[0]]
    assert line_groups("один два три четыре пять".split(), 14) == [[0, 1, 2], [3, 4]]


def test_build_ass_karaoke():
    ws = [Word(0.0, 0.4, " раз"), Word(0.5, 0.9, " два"), Word(1.0, 1.4, " три")]
    ass = build_ass(make_cues(ws), 1080, 1920, size=74, align=8, margin_v=900, karaoke=True)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert ",8,60,60,900,1" in ass
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 3
    assert all(line.count(YELLOW_TAG) == 1 for line in events)
    assert f"{YELLOW_TAG}два" in events[1]


def test_build_ass_plain_and_upper():
    cues = [Cue(0, 2, "переведённая {реплика}")]
    ass = build_ass(cues, 1080, 1920, karaoke=True, upper=True)
    line = [row for row in ass.splitlines() if row.startswith("Dialogue:")][0]
    assert YELLOW_TAG not in line and "ПЕРЕВЕДЁННАЯ (РЕПЛИКА)" in line.replace("\\N", " ")
    assert build_ass([], 1080, 1920).rstrip().endswith("Text")


# ------------------------------------------------------------------ layout

@pytest.mark.parametrize("req, w, h, mode", [
    ("auto", 1920, 1080, "blur"), ("auto", 1080, 1920, "full"), ("auto", 720, 900, "full"),
    ("blur", 1080, 1920, "full"), ("smart", 1080, 1920, "full"), ("smart", 1920, 1080, "smart"),
    ("full", 1920, 1080, "full"), ("blur", 1000, 1000, "blur"),
])
def test_choose_layout(req, w, h, mode):
    assert choose_layout(req, w, h) == mode


def test_geometry_blur_and_smart():
    g = geometry("blur", 1920, 1080)
    assert (g.video_h, g.video_y) == (608, 656)
    assert g.caption_align == 8 and g.caption_margin_v == 656 + 608 + 20
    s = geometry("smart", 1920, 1080)
    assert s.crop_w == even(1080 * 1080 / 1920) == 608 and s.caption_align == 2
    assert geometry("full", 1080, 1920).video_h == 1920


def test_text_layout_blur_positions():
    g = geometry("blur", 1920, 1080)
    items = text_layout(g, "Автор канала", "Робот|учится", "Без паники", FONT, FONT)
    texts = [i.text for i in items]
    assert texts == ["Автор канала", "Робот", "учится", "Без паники"]
    assert items[0].y == 64 and items[0].size <= 42
    assert all(i.y + i.size <= g.video_y for i in items[:3])
    assert items[-1].color == "yellow" and items[-1].y > g.video_y + g.video_h
    assert all(i.size <= 118 for i in items)


def test_text_layout_full_top_stack():
    g = geometry("full", 1080, 1920)
    items = text_layout(g, "", "Длинный заголовок для проверки", "", FONT, FONT)
    assert 1 <= len(items) <= 2 and items[0].y == 110
    assert all(i.y < 1920 * 0.40 for i in items)


def test_drawtext_uses_textfile():
    g = geometry("blur", 1920, 1080)
    f = drawtext(text_layout(g, "", "Тест", "", FONT, FONT)[0], "t_0.txt")
    assert "textfile='t_0.txt'" in f and "expansion=none" in f and r"C\:/Windows" in f


def test_build_graph_variants():
    g = geometry("blur", 1920, 1080)
    graph, has_a = build_graph(g, fps=30, src_h=1080, text_filters=["drawtext=x"], ass_file="captions.ass",
                               keep_expr=None, crop_x_expr=None, has_audio=True)
    assert has_a and "gblur=sigma=6" in graph and "overlay=0:656" in graph
    assert graph.index("drawtext=x") < graph.index("ass='captions.ass'") < graph.index("[v]")
    assert graph.endswith("[0:a]aresample=48000[a]")

    s = geometry("smart", 1920, 1080)
    graph, has_a = build_graph(s, fps=30, src_h=1080, text_filters=[], ass_file=None,
                               keep_expr="between(t,0,1)", crop_x_expr="if(lt(t,2),100,300)", has_audio=False)
    assert not has_a and "[a]" not in graph
    assert "crop=608:1080:'if(lt(t,2),100,300)':0" in graph
    assert "select='between(t,0,1)',setpts=N/(30*TB)" in graph


def test_build_graph_audio_follows_cut():
    g = geometry("full", 1080, 1920)
    graph, _ = build_graph(g, fps=25, src_h=1920, text_filters=[], ass_file=None,
                           keep_expr="between(t,0,1)", crop_x_expr=None, has_audio=True)
    assert "aselect='between(t,0,1)',asetpts=N/SR/TB" in graph


# ------------------------------------------------------------------ facetrack

def test_crop_track_follows_face_per_scene():
    samples = [(0.0, 0.3), (0.5, 0.32), (1.0, 0.28), (2.5, 0.8), (3.0, 0.82), (3.5, None)]
    track = crop_track(samples, [2.0], 4.0, src_w=1920, crop_w=608)
    assert [t for t, _ in track] == [0.0, 2.0]
    x0, x1 = track[0][1], track[1][1]
    assert x0 == int(round(0.3 * 1920 - 304)) - (int(round(0.3 * 1920 - 304)) % 2)
    assert x1 > x0 and x1 <= 1920 - 608 and x1 % 2 == 0


def test_crop_track_no_faces_centers():
    assert crop_track([(0, None)], [], 3.0, 1920, 608) == [(0.0, 656)]


def test_crop_track_clamps_and_merges():
    samples = [(0.0, 0.99), (2.0, 0.98)]
    track = crop_track(samples, [1.5], 3.0, 1920, 608)
    assert track == [(0.0, 1312)]


def test_crop_x_expr():
    assert crop_x_expr([(0.0, 100)]) == "100"
    assert crop_x_expr([(0.0, 100), (2.0, 300), (5.0, 50)]) == \
        "if(lt(t,2.000),100,if(lt(t,5.000),300,50))"


def test_face_finder_blank_frame():
    from konveyer.facetrack import FaceFinder
    f = FaceFinder()
    try:
        assert f.center_x(np.zeros((180, 320, 3), np.uint8)) is None
    finally:
        f.close()


# ------------------------------------------------------------------ analysis

def test_loudness_db():
    sr = 16000
    tone = (0.5 * np.sin(np.linspace(0, 2 * np.pi * 440, sr))).astype(np.float32)
    audio = np.concatenate([np.zeros(sr, np.float32), tone])
    db = loudness_db(audio, sr, 0.5)
    assert len(db) == 4
    assert db[0] == pytest.approx(-120.0) and db[3] == pytest.approx(-9.03, abs=0.2)
    assert loudness_db(np.zeros(0, np.float32)) == []
    assert len(loudness_db(np.ones(100, np.float32), sr, 0.5)) == 1


def test_parse_scene_times():
    err = ("[Parsed_showinfo_2 @ 0] n:0 pts:0 pts_time:0.1 duration:...\n"
           "[Parsed_showinfo_2 @ 0] n:1 pts:512 pts_time:4.2 \n"
           "[Parsed_showinfo_2 @ 0] n:2 pts:900 pts_time:12.5\n pts_time:4.2")
    assert parse_scene_times(err) == [4.2, 12.5]


def test_speech_density():
    tr = Transcript("ru", [Segment(0, 2, "", [Word(0.1, 0.3, "a"), Word(0.6, 0.8, "b"), Word(0.7, 0.9, "c")])])
    assert speech_density(tr, 2.0, 0.5) == [2.0, 4.0, 0.0, 0.0]


@needs_ffmpeg
def test_detect_scenes_on_cut(tmp_path):
    import subprocess
    from konveyer.analysis import detect_scenes
    out = tmp_path / "склейка.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=320x180:d=2:r=25",
                    "-f", "lavfi", "-i", "color=c=white:s=320x180:d=2:r=25",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", out.name],
                   check=True, cwd=tmp_path)
    scenes = detect_scenes(out)
    assert len(scenes) == 1 and scenes[0] == pytest.approx(2.0, abs=0.1)
