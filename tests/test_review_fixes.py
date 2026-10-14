"""Review fixes: fonts fallback, web numeric validation, voice list dedupe, EN help everywhere."""
import json
import re
import threading
import urllib.error
import urllib.request

import pytest

from konveyer import i18n
from konveyer.layout import resolve_text_font


# ---------------------------------------------------------------- fonts (P3)

def test_font_configured_path_wins(tmp_path):
    f = tmp_path / "my.ttf"
    f.write_bytes(b"x" * 10)
    assert resolve_text_font(str(f), tmp_path) == str(f)
    assert resolve_text_font("my.ttf", tmp_path) == str(f)          # relative to root


def test_font_falls_back_to_assets(tmp_path):
    fonts = tmp_path / "assets" / "fonts"
    fonts.mkdir(parents=True)
    (fonts / "Bundled.ttf").write_bytes(b"x")
    assert resolve_text_font("C:/nope/ariblk.ttf", tmp_path).endswith("Bundled.ttf")


def test_font_none_when_nothing(tmp_path, monkeypatch):
    import konveyer.layout as layout
    monkeypatch.setattr(layout, "SYSTEM_FONTS", ())
    assert resolve_text_font("/nope/x.ttf", tmp_path) is None


# ---------------------------------------------------------------- voice list (P8)

def test_voice_list_dedupe_casefold():
    from konveyer.config import EDGE_SPEAKERS, _check_voice_list
    out = _check_voice_list(["Brian", "brian", "AVA", "ava"], "speaker_voices", EDGE_SPEAKERS, 8)
    assert out == ["brian", "ava"]


# ---------------------------------------------------------------- EN help (P6)

def test_every_subcommand_help_has_no_cyrillic_in_en():
    import argparse
    from konveyer.cli import build_parser
    i18n.set_lang("en")
    try:
        p = build_parser()
        i18n.translate_parser(p)
        sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        assert len(sub.choices) >= 15
        for name, sp in sub.choices.items():
            assert not re.search("[А-Яа-яЁё]", sp.format_help()), name
        assert not re.search("[А-Яа-яЁё]", p.format_help())
    finally:
        i18n.set_lang("ru")


# ---------------------------------------------------------------- web numbers (P7)

def test_num_param_rules():
    from konveyer.web.server import num_param
    assert num_param({}, "count", 1, 10, True) is None
    assert num_param({"count": "3"}, "count", 1, 10, True) == 3
    assert num_param({"count": 4.0}, "count", 1, 10, True) == 4
    for bad in (2.5, "nan", float("inf"), "abc", True, 0, 11):
        with pytest.raises(ValueError):
            num_param({"count": bad}, "count", 1, 10, True)
    with pytest.raises(ValueError):
        num_param({"zoom": float("nan")}, "zoom", 1.0, 2.0)


@pytest.fixture
def server(settings):
    from konveyer.web.server import App, make_server
    app = App(settings, runner=lambda *a: None)
    srv, _ = make_server(settings, app, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()
    app.stop()


def _post(url, raw: str):
    r = urllib.request.Request(url, data=raw.encode(), method="POST",
                               headers={"X-Konveyer": "1", "Content-Type": "application/json",
                                        "X-Konveyer-Lang": "en"})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.mark.parametrize("body", ['{"source": "x.mp4", "count": NaN}',
                                  '{"source": "x.mp4", "count": 2.5}',
                                  '{"source": "x.mp4", "count": 99}',
                                  '{"source": "x.mp4", "zoom": Infinity}',
                                  '{"source": "x.mp4", "min": 50, "max": 20}'])
def test_web_rejects_bad_numbers(server, body):
    status, data = _post(server + "/api/jobs", body)
    assert status == 400 and data["error"] and not re.search("[А-Яа-яЁё]", data["error"])
