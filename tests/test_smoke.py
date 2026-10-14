"""Smoke: all modules import, i18n EN covers all CLI help, example configs load."""
import importlib
import pkgutil
from pathlib import Path

import konveyer
from konveyer import i18n

ROOT = Path(__file__).resolve().parent.parent


def test_all_modules_import():
    for m in pkgutil.walk_packages(konveyer.__path__, "konveyer."):
        if m.name.endswith(("xtts_worker", "__main__")):
            continue
        importlib.import_module(m.name)


def test_cli_help_fully_english():
    import re
    from konveyer.cli import build_parser
    i18n.set_lang("en")
    try:
        p = build_parser()
        i18n.translate_parser(p)
        texts = [p.format_help()]
        for a in p._actions:
            ch = getattr(a, "choices", None)
            for sub in (ch.values() if isinstance(ch, dict) else ()):
                if hasattr(sub, "format_help"):
                    texts.append(sub.format_help())
        assert not re.search("[А-Яа-яЁё]", "\n".join(texts))
    finally:
        i18n.set_lang("ru")


def test_example_config_and_env_example():
    from konveyer.config import load_settings
    s = load_settings(ROOT / "konveyer.toml.example", root=ROOT)
    assert s.ui.lang in ("ru", "en")
    keys = {l.split("=")[0] for l in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")}
    from konveyer.envkeys import SECRET_NAMES
    assert {"DEEPSEEK_API_KEY", "TG_BOT_TOKEN", "TG_CHAT_ID", "HF_TOKEN"} <= keys
    assert set(SECRET_NAMES) - keys <= {"HUGGING_FACE_HUB_TOKEN"}
