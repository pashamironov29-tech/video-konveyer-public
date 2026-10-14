"""Interface language switch: ru (default) / en."""
from konveyer import i18n
from konveyer.cli import build_parser


def test_ru_is_untouched():
    assert i18n.tr("задание не найдено", "ru") == "задание не найдено"


def test_en_exact_and_dynamic():
    assert i18n.tr("задание не найдено", "en") == "job not found"
    i18n.set_lang("en")
    try:
        assert i18n.tr("клип 3: готово за 12.5 с") == "clip 3: done in 12.5 s"
        p = build_parser()
        i18n.translate_parser(p)
        assert "Video -> vertical" in p.format_help()
    finally:
        i18n.set_lang("ru")
