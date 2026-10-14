"""Linux и Steam Deck: программы из PATH, пути через pathlib, без обязательных .exe."""
from __future__ import annotations

from pathlib import Path

import pytest

from konveyer.config import RenderCfg
from konveyer.ingest import ytdlp_command
from konveyer.jobs import label_from_source
from konveyer.layout import resolve_text_font
from konveyer.util import KonveyerError, ffprobe, need_tool


def test_need_tool_uses_path_and_does_not_add_exe(monkeypatch):
    seen = []

    def which(name):
        seen.append(name)
        return f"/usr/bin/{name}"

    monkeypatch.setattr("konveyer.util.shutil.which", which)
    assert need_tool("ffmpeg") == "/usr/bin/ffmpeg"
    assert need_tool("yt-dlp") == "/usr/bin/yt-dlp"
    assert seen == ["ffmpeg", "yt-dlp"]
    monkeypatch.setattr("konveyer.util.shutil.which", lambda name: None)
    with pytest.raises(KonveyerError, match="PATH"):
        need_tool("ffprobe")


def test_ytdlp_command_binary_comes_from_path(monkeypatch, settings, tmp_path):
    monkeypatch.setattr("konveyer.ingest.need_tool", lambda name: "/usr/bin/yt-dlp")
    cmd = ytdlp_command("https://youtu.be/abc", tmp_path, settings)
    assert cmd[0] == "/usr/bin/yt-dlp"
    assert all(not str(part).endswith(".exe") for part in cmd)
    assert Path(cmd[cmd.index("-P") + 1]) == tmp_path


def test_ffprobe_calls_binary_from_path(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, what, **kwargs):
        calls.append(list(cmd))

        class Proc:
            stdout = ('{"streams":[{"codec_type":"video","width":8,"height":4,'
                      '"avg_frame_rate":"30/1"}],"format":{"duration":"1.5"}}')

        return Proc()

    monkeypatch.setattr("konveyer.util.need_tool", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("konveyer.util.run", fake_run)
    info = ffprobe(tmp_path / "clip.mp4")
    assert calls[0][0] == "/usr/bin/ffprobe"
    assert info["width"] == 8 and info["height"] == 4 and info["duration"] == 1.5


def test_render_font_defaults_are_empty():
    cfg = RenderCfg()
    assert cfg.font_title == "" and cfg.font_text == ""


def test_linux_font_candidates_skip_windows(monkeypatch):
    import konveyer.layout as layout

    monkeypatch.setattr(layout.sys, "platform", "linux")
    paths = [str(p) for p in layout.font_candidates()]
    assert paths
    assert all(not p.startswith("C:") and "Windows" not in p for p in paths)
    assert any("DejaVuSans-Bold.ttf" in p for p in paths)
    assert any("NotoSans-Bold.ttf" in p for p in paths)


def test_empty_font_prefers_system_then_bundled(tmp_path, monkeypatch):
    import konveyer.layout as layout

    bundled = tmp_path / "assets" / "fonts" / "Bundled.ttf"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"b")
    system = tmp_path / "DejaVuSans-Bold.ttf"
    system.write_bytes(b"s")
    monkeypatch.setattr(layout, "font_candidates", lambda: [system])
    assert resolve_text_font("", tmp_path) == str(system)
    monkeypatch.setattr(layout, "font_candidates", lambda: [])
    assert resolve_text_font("", tmp_path) == str(bundled)


def test_label_from_posix_and_windows_paths():
    assert label_from_source("/home/deck/Videos/ролик.mp4") == "ролик"
    assert label_from_source(r"D:\videos\ролик.mp4") == "ролик"


def test_cookies_file_is_joined_from_project_root(settings, tmp_path, monkeypatch):
    monkeypatch.setattr("konveyer.ingest.need_tool", lambda name: "yt-dlp")
    settings.root = tmp_path
    settings.ingest.cookies_file = "cookies/youtube_cookies.txt"
    cmd = ytdlp_command("https://youtu.be/abc", tmp_path, settings)
    assert "--cookies" not in cmd
    dest = tmp_path / "cookies" / "youtube_cookies.txt"
    dest.parent.mkdir()
    dest.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    cmd = ytdlp_command("https://youtu.be/abc", tmp_path, settings)
    assert Path(cmd[cmd.index("--cookies") + 1]) == dest
