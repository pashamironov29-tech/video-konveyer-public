"""Командная строка: python -m konveyer <команда> ..."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from konveyer import __version__
from konveyer.config import (COVER_MODES, SUBS_STYLES, VOICE_ENGINES, VOICE_SPEAKERS,
                             load_settings, speaker_error)
from konveyer.envkeys import SECRET_NAMES, has_secret
from konveyer.jobs import STEP_NAMES, STEPS, Job, find_job, list_jobs
from konveyer.log import Log
from konveyer import i18n
from konveyer.i18n import tr
import builtins as _builtins


def print(*args, **kwargs):  # noqa: A001 — interface output goes through the translator
    _builtins.print(*(tr(a) if isinstance(a, str) else a for a in args), **kwargs)
from konveyer.util import KonveyerError, fmt_time


def _add_job_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--count", type=int, help="сколько клипов (по умолчанию 3)")
    p.add_argument("--min", type=float, help="мин. длина клипа, с")
    p.add_argument("--max", type=float, help="макс. длина клипа, с")
    p.add_argument("--layout", choices=("auto", "blur", "full", "smart"),
                   help="blur — размытый фон, full — на весь экран, smart — следить за лицом")
    p.add_argument("--tighten", action="store_true", help="вырезать паузы в речи")
    p.add_argument("--no-captions", action="store_true", help="без субтитров")
    p.add_argument("--no-titles", action="store_true",
                   help="без надписей: ни заголовка, ни строки сверху, ни жёлтой подписи")
    p.add_argument("--force-captions", action="store_true",
                   help="субтитры даже если речь ненадёжна или её нет")
    p.add_argument("--no-translate", action="store_true", help="не переводить субтитры")
    p.add_argument("--no-llm", action="store_true", help="без DeepSeek (эвристика и шаблон поста)")
    p.add_argument("--cpu", action="store_true", help="Whisper только на процессоре")
    p.add_argument("--zoom", type=float, help="зум картинки 1.0–2.0, по умолчанию 1.0 — кадр целиком; 1.5 — старый зум")
    p.add_argument("--zoom-center", action="store_true",
                   help="зум ровно по центру кадра, без слежения за лицом")
    p.add_argument("--subs-style", choices=SUBS_STYLES,
                   help="normal — обычные субтитры, big — крупные по 1–2 слова")
    p.add_argument("--no-speakers", action="store_true",
                   help="не различать спикеров: один цвет субтитров и один голос")
    p.add_argument("--score", action="store_true",
                   help="перед Telegram оценить момент DeepSeek от 1 до 10")
    p.add_argument("--vision-check", action="store_true",
                   help="перед Telegram показать кадры локальной модели Ollama")
    _add_voice_options(p)
    _add_cover_flags(p)


def _add_cover_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-cover", action="store_true", help="не делать обложки")
    p.add_argument("--cover-mode", choices=COVER_MODES,
                   help="frame — кадр из видео, ai — нейросеть, both — обе (по умолчанию)")


def _cover_options(a) -> dict:
    opts = {}
    if getattr(a, "no_cover", False):
        opts["cover"] = False
    mode = getattr(a, "cover_mode", None)
    if mode:
        opts["cover_mode"] = mode
    return opts


def _add_voice_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("озвучка")
    g.add_argument("--voice", action="store_true", default=None,
                   help="русская озвучка переведённых реплик")
    g.add_argument("--no-voice", dest="voice", action="store_false", help="выключить озвучку")
    g.add_argument("--voice-speaker", choices=VOICE_SPEAKERS,
                   help="голос Edge: brian (по умолчанию), dmitry, svetlana, ava, emma, "
                        "florian, seraphina; Silero: aidar, eugene, baya, kseniya, xenia "
                        "(только вместе с --voice-engine silero)")
    g.add_argument("--voice-engine", choices=VOICE_ENGINES,
                   help="edge (по умолчанию), silero или xtts — клон голоса по --voice-sample")
    g.add_argument("--voice-sample", type=Path, help="xtts: чистая запись голоса 10–30 с, wav")
    g.add_argument("--voice-mode", choices=("voiceover", "fit"),
                   help="voiceover — закадровый перевод в естественном темпе (по умолчанию); "
                        "fit — подогнать фразу под окно субтитра")
    g.add_argument("--voice-flat", action="store_true",
                   help="Edge без смены тона: вопрос и восклицание тем же голосом")
    g.add_argument("--voice-no-subs", action="store_true", help="с озвучкой убрать субтитры")
    g.add_argument("--separate", action="store_true",
                   help="demucs: убрать голос оригинала, оставить музыку и шумы")
    g.add_argument("--no-voice-check", action="store_true",
                   help="не проверять озвучку через Whisper и DeepSeek")
    g.add_argument("--one-voice", action="store_true",
                   help="выключить разные голоса: все спикеры одним. По умолчанию голоса разные")


def _voice_options(a, settings=None) -> dict:
    opts = {}
    if getattr(a, "voice", None) is not None:
        opts["voice"] = bool(a.voice)
    if getattr(a, "voice_speaker", None):
        opts["voice_speaker"] = a.voice_speaker
    if getattr(a, "voice_engine", None):
        opts["voice_engine"] = a.voice_engine
    if getattr(a, "voice_sample", None):
        sample = Path(a.voice_sample).expanduser().resolve()
        if not sample.is_file():
            raise KonveyerError(f"нет файла с образцом голоса: {sample}")
        opts["voice_sample"] = str(sample)
    if getattr(a, "voice_no_subs", False):
        opts["voice_subs"] = False
    if getattr(a, "separate", False):
        opts["separate"] = True
    if getattr(a, "voice_mode", None):
        opts["voice_mode"] = a.voice_mode
    if getattr(a, "voice_flat", False):
        opts["voice_flat"] = True
    if getattr(a, "no_voice_check", False):
        opts["voice_check"] = False
    if getattr(a, "one_voice", False):
        opts["per_speaker"] = False
    if "voice_sample" in opts and "voice_engine" not in opts:
        opts["voice_engine"] = "xtts"
    has_sample = "voice_sample" in opts or bool(settings and settings.voice.sample)
    if opts.get("voice_engine") == "xtts" and not has_sample:
        raise KonveyerError("--voice-engine xtts нужен вместе с --voice-sample путь.wav")
    if "voice_speaker" in opts or "voice_engine" in opts:
        engine = opts.get("voice_engine") or (settings.voice.engine if settings else "edge")
        speaker = opts.get("voice_speaker") or (settings.voice.speaker if settings else "brian")
        err = speaker_error(engine, speaker)
        if err:
            raise KonveyerError(err)
    return opts


def _job_options(a, settings=None) -> dict:
    opts = {}
    for key in ("count", "min", "max", "layout"):
        if getattr(a, key, None) is not None:
            opts[key] = getattr(a, key)
    if getattr(a, "tighten", False):
        opts["tighten"] = True
    if getattr(a, "no_captions", False):
        opts["captions"] = False
    if getattr(a, "no_titles", False):
        opts["titles"] = False
    if getattr(a, "force_captions", False):
        opts["force_captions"] = True
    if getattr(a, "no_translate", False):
        opts["translate"] = False
    if getattr(a, "no_llm", False):
        opts["llm"] = False
    if getattr(a, "cpu", False):
        opts["device"] = "cpu"
    if getattr(a, "zoom", None) is not None:
        z = float(a.zoom)
        if not 1.0 <= z <= 2.0:
            raise KonveyerError("--zoom от 1.0 до 2.0")
        opts["zoom"] = round(z, 2)
    if getattr(a, "zoom_center", False):
        opts["zoom_center"] = True
    if getattr(a, "subs_style", None):
        opts["subs_style"] = a.subs_style
    if getattr(a, "no_speakers", False):
        opts["speakers"] = False
    if getattr(a, "score", False):
        opts["score"] = True
    if getattr(a, "vision_check", False):
        opts["vision_check"] = True
    opts.update(_voice_options(a, settings))
    opts.update(_span_options(a))
    opts.update(_cover_options(a))
    return opts


def _span_options(a) -> dict:
    """--full и отрезок --from/--to. Время только вместе с целым видео."""
    from konveyer.moments import parse_time

    opts = {}
    if getattr(a, "full", False):
        opts["full"] = True
    start = getattr(a, "time_from", None) or ""
    end = getattr(a, "time_to", None) or ""
    if (start or end) and not opts.get("full"):
        raise KonveyerError("--from и --to только вместе с --full (или команда full)")
    try:
        if start:
            opts["full_from"] = round(parse_time(start), 3)
        if end:
            opts["full_to"] = round(parse_time(end), 3)
    except ValueError as e:
        raise KonveyerError(str(e)) from e
    if "full_from" in opts and "full_to" in opts and opts["full_to"] <= opts["full_from"]:
        raise KonveyerError("--to должен быть позже --from")
    return opts


def _add_full_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--from", dest="time_from", default="", metavar="ВРЕМЯ",
                   help="начало отрезка: секунды или 1:02:03 (только с --full)")
    p.add_argument("--to", dest="time_to", default="", metavar="ВРЕМЯ",
                   help="конец отрезка")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="konveyer",
                                 description="Ролик -> вертикальные клипы 9:16 + пост ВК + Telegram")
    ap.add_argument("--config", type=Path, help="путь к konveyer.toml")
    ap.add_argument("--lang", choices=i18n.LANGS,
                    help="язык интерфейса: ru или en (иначе KONVEYER_LANG или [ui] lang)")
    ap.add_argument("--set", action="append", default=[], metavar="РАЗДЕЛ.КЛЮЧ=ЗНАЧЕНИЕ",
                    help="переопределить настройку, например --set render.crf=18")
    ap.add_argument("--version", action="version", version=f"konveyer {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="полный конвейер по ссылке или файлу")
    p.add_argument("source", help="ссылка (YouTube, X, Instagram, TikTok...) или путь к видео")
    p.add_argument("--full", action="store_true",
                   help="целое видео 16:9 для ВК, без нарезки на Shorts")
    _add_full_flags(p)
    _add_job_options(p)
    p.add_argument("--tg", action="store_true", help="в конце отправить клипы и пост в Telegram")
    p.add_argument("--force-send", action="store_true",
                   help="отправить в Telegram даже если проверка текстов не пройдена")
    p.add_argument("--label", default="", help="метка для имени папки задания")

    p = sub.add_parser("full", help="целое видео 16:9 для ВК, без нарезки на клипы")
    p.add_argument("source", help="ссылка или путь к видео")
    _add_full_flags(p)
    p.add_argument("--no-captions", action="store_true", help="без субтитров")
    p.add_argument("--no-speakers", action="store_true",
                   help="не различать спикеров: один цвет субтитров и один голос")
    p.add_argument("--force-captions", action="store_true",
                   help="субтитры даже если речь ненадёжна или её нет")
    p.add_argument("--no-translate", action="store_true", help="не переводить субтитры")
    p.add_argument("--no-llm", action="store_true", help="без DeepSeek (эвристика и шаблон поста)")
    p.add_argument("--cpu", action="store_true", help="Whisper только на процессоре")
    p.add_argument("--score", action="store_true",
                   help="перед Telegram оценить момент DeepSeek от 1 до 10")
    p.add_argument("--vision-check", action="store_true",
                   help="перед Telegram показать кадры локальной модели Ollama")
    _add_voice_options(p)
    _add_cover_flags(p)
    p.add_argument("--tg", action="store_true", help="в конце отправить в Telegram")
    p.add_argument("--force-send", action="store_true",
                   help="отправить в Telegram даже если проверка текстов не пройдена")
    p.add_argument("--label", default="", help="метка для имени папки задания")

    sub.add_parser("jobs", help="список заданий")

    p = sub.add_parser("show", help="подробности задания")
    p.add_argument("job", nargs="?", default="last", help="id, начало id или last")

    p = sub.add_parser("redo", help="переделать с указанного шага")
    p.add_argument("job", nargs="?", default="last")
    p.add_argument("--from", dest="from_step", choices=STEPS, default="moments")
    _add_job_options(p)
    p.add_argument("--tg", action="store_true")
    p.add_argument("--resend", action="store_true",
                   help="с --tg: отправить заново то, что уже ушло")
    p.add_argument("--force-send", action="store_true",
                   help="отправить в Telegram даже если проверка текстов не пройдена")

    p = sub.add_parser("clip", help="поправить клип (номер) или нарезать Shorts по таймкодам")
    p.add_argument("job", help="задание (id, last, папка) или ссылка/файл для ручных таймкодов")
    p.add_argument("clip", nargs="+", metavar="КЛИП|КУСОК",
                   help="номер клипа — поправить его; или куски 2:10-2:50 [5:00-5:40 ...] — "
                        "Shorts ровно по ним (границы к концу предложения ±1,5 с)")
    p.add_argument("--tg", action="store_true", help="ручные Shorts: отправить в Telegram")
    p.add_argument("--force-send", action="store_true",
                   help="отправить даже если проверка текстов не пройдена")
    p.add_argument("--label", default="", help="метка для имени папки нового задания")
    p.add_argument("--no-captions", action="store_true", help="без субтитров")
    p.add_argument("--no-speakers", action="store_true",
                   help="не различать спикеров: один цвет субтитров и один голос")
    _add_cover_flags(p)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--title", help="белый заголовок, строки через |")
    p.add_argument("--kicker", help="мелкая строка сверху")
    p.add_argument("--bottom", help="жёлтая подпись, строки через |")
    p.add_argument("--no-titles", action="store_true",
                   help="убрать заголовок, строку сверху и жёлтую подпись "
                        "(пустой --title из PowerShell не доходит)")
    p.add_argument("--layout", choices=("auto", "blur", "full", "smart"))
    p.add_argument("--zoom", type=float, help="зум картинки 1.0–2.0")
    p.add_argument("--zoom-center", action="store_true",
                   help="зум ровно по центру кадра, без слежения за лицом")
    p.add_argument("--subs-style", choices=SUBS_STYLES,
                   help="normal — обычные субтитры, big — крупные по 1–2 слова")
    _add_voice_options(p)

    p = sub.add_parser("cover", help="перегенерировать только обложки")
    p.add_argument("job", nargs="?", default="last")
    p.add_argument("--title", default="", help="свой текст вместо заголовка от DeepSeek")
    p.add_argument("--clip", type=int, help="только этот клип (для Shorts)")
    p.add_argument("--cover-mode", choices=COVER_MODES,
                   help="frame — кадр из видео, ai — нейросеть, both — обе")

    p = sub.add_parser("post", help="переписать пост")
    p.add_argument("job", nargs="?", default="last")

    p = sub.add_parser("describe", help="переписать описания Shorts (description.md), без отправки")
    p.add_argument("job", nargs="?", default="last")
    p.add_argument("--clip", type=int, action="append", help="только этот клип (можно несколько)")

    p = sub.add_parser("send", help="отправить в Telegram")
    p.add_argument("job", nargs="?", default="last")
    p.add_argument("--clip", type=int, action="append", help="только этот клип (можно несколько)")
    p.add_argument("--no-clips", action="store_true")
    p.add_argument("--no-post", action="store_true")
    p.add_argument("--shorts-post", action="store_true",
                   help="после Shorts отправить и общий пост (по умолчанию только описания клипов)")
    p.add_argument("--resend", action="store_true",
                   help="отправить заново то, что уже ушло (по умолчанию повтор не шлётся)")
    p.add_argument("--force-send", action="store_true",
                   help="отправить даже если проверка текстов не пройдена")
    p.add_argument("--score", action="store_true",
                   help="оценить момент DeepSeek и не слать клип ниже порога")
    p.add_argument("--vision-check", action="store_true",
                   help="проверить кадры локальной моделью Ollama")
    p.add_argument("--no-voice-check", action="store_true",
                   help="не заворачивать клип из-за проверки озвучки")

    p = sub.add_parser("guard", help="сторож: проверить озвучку целого видео (отчёт)")
    p.add_argument("job", nargs="?", default="last")
    p.add_argument("--fix", action="store_true",
                   help="переозвучить и пересобрать при замечаниях (до 2 кругов)")

    p = sub.add_parser("web", help="веб-интерфейс в браузере")
    p.add_argument("--port", type=int)
    p.add_argument("--no-open", action="store_true", help="не открывать браузер")

    sub.add_parser("doctor", help="проверить ffmpeg, yt-dlp, GPU, ключи, шрифты")
    sub.add_parser("config", help="показать действующие настройки")
    p = sub.add_parser("cleanup", help="очистка папок заданий: исходники и временные файлы "
                                       "отправленных, старые задания целиком")
    p.add_argument("--dry-run", action="store_true", help="только показать, что и сколько удалится")
    p.add_argument("--job", help="только это задание (исходники и временные файлы)")

    return ap


def _overrides(items: list[str]) -> dict:
    out = {}
    for it in items:
        if "=" not in it or "." not in it.split("=", 1)[0]:
            raise KonveyerError(f"--set ожидает РАЗДЕЛ.КЛЮЧ=ЗНАЧЕНИЕ, получено: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ------------------------------------------------------------------ команды

def cmd_full(a, settings) -> int:
    a.full = True
    return cmd_run(a, settings)


def cmd_run(a, settings) -> int:
    from konveyer.pipeline import Pipeline

    opts = _job_options(a, settings)
    if a.force_send:
        opts["force_send"] = True
    job = Job.create(settings, a.source, opts, a.label)
    log = Log(job.log_file)
    log.info(f"задание: {job.id}")
    Pipeline(job, settings, log).run(send=a.tg or a.force_send, force_send=a.force_send)
    return 0


def cmd_jobs(a, settings) -> int:
    jobs = list_jobs(settings)
    if not jobs:
        print("Заданий пока нет.")
        return 0
    for j in jobs:
        s = j.summary()
        state = {"done": "готово", "failed": "ошибка", "running": "идёт", "new": "новое",
                 "review": "проверка", "queued": "в очереди"}.get(s["state"], s["state"])
        print(f"{s['id']:<44} {state:<7} клипов: {s['clips']}  {str(s['title'])[:60]}")
    return 0


def cmd_show(a, settings) -> int:
    job = find_job(settings, a.job)
    d, m = job.data, job.data.get("meta", {})
    print(f"Задание: {job.id}\nПапка:   {job.dir}\nИсточник: {d['source']}\nСостояние: {d['state']}"
          + (f" (шаг {d['current']})" if d.get("current") else ""))
    if d.get("error"):
        label = "На проверку" if d.get("state") == "review" else "Ошибка"
        print(f"{label}: {d['error']}")
    if m:
        print(f"Видео: {m.get('width')}x{m.get('height')}, {fmt_time(m.get('duration', 0))}, "
              f"автор: {m.get('uploader') or '-'}, язык: {m.get('language') or '-'}")
    if m.get("speech_note"):
        print(m["speech_note"] + (f" ({m.get('speech_reason')})" if m.get("speech_reason") else ""))
    summary = (d.get("deepseek") or {}).get("summary")
    if summary:
        print(summary)
    for step in STEPS:
        st = d["steps"].get(step, {})
        print(f"  {STEP_NAMES[step]:<16} {st.get('status', '—'):<7} {st.get('sec', '')}")
    for c in d.get("clips", []):
        print(f"  клип {c['id']}: {c['start']:.1f}–{c['end']:.1f} с, «{c.get('title', '')}» "
              f"{'→ ' + c['file'] if c.get('file') else '(не отрендерен)'}"
              + (f"\n    озвучка: {c['voice']}" if c.get("voice") else ""))
        covers = c.get("covers") or {}
        if covers.get("title"):
            who = f", {covers['speaker']}" if covers.get("speaker") else ""
            print(f"    обложка: «{covers['title']}»{who}")
        for kind, label in (("frame", "кадр"), ("ai", "нейросеть")):
            if covers.get(kind):
                print(f"    {label}: {covers[kind]}")
    if job.post_md.exists():
        print(f"Пост: {job.post_md}")
    return 0


def cmd_redo(a, settings) -> int:
    from konveyer.pipeline import Pipeline

    job = find_job(settings, a.job)
    opts = job.data.setdefault("options", {})
    opts.update(_job_options(a, settings))
    if a.force_send:
        opts["force_send"] = True
    job.invalidate_from(a.from_step)
    Pipeline(job, settings).run(send=a.tg or a.force_send, force_send=a.force_send,
                                resend=a.resend)
    return 0


def cmd_clip(a, settings) -> int:
    from konveyer.manual import is_spans, run_manual
    from konveyer.pipeline import Pipeline

    items = [str(x) for x in a.clip]
    if is_spans(items):
        job = run_manual(settings, a.job, items, _job_options(a, settings), a.label,
                         send=a.tg or a.force_send, force_send=a.force_send)
        print(f"Задание: {job.id}")
        for c in job.data.get("clips") or []:
            if c.get("file"):
                print(f"Готово: {job.dir / c['file']}")
        return 0
    if len(items) != 1 or not items[0].isdigit():
        raise KonveyerError("после задания — номер клипа (clip last 2) "
                            "или куски вида 2:10-2:50 (clip last 2:10-2:50 5:00-5:40)")
    a.clip = int(items[0])
    job = find_job(settings, a.job)
    clip = job.clip(a.clip)
    for key in ("start", "end", "title", "kicker", "bottom"):
        value = getattr(a, key)
        if value is not None:
            clip[key] = value
    if a.no_titles:
        clip["title"] = clip["kicker"] = clip["bottom"] = ""
    if a.layout:
        clip["layout_req"] = "" if a.layout == "auto" else a.layout
    if a.zoom is not None:
        if not 1.0 <= a.zoom <= 2.0:
            raise KonveyerError("--zoom от 1.0 до 2.0")
        clip["zoom"] = round(float(a.zoom), 2)
        job.data.setdefault("options", {})["zoom"] = clip["zoom"]
    if a.subs_style:
        job.data.setdefault("options", {})["subs_style"] = a.subs_style
    if clip["end"] <= clip["start"]:
        raise KonveyerError("конец клипа должен быть позже начала")
    voice = _voice_options(a, settings)
    if voice:
        # настройки озвучки общие для задания: следующий redo озвучит так же
        job.data.setdefault("options", {}).update(voice)
    job.save()
    Pipeline(job, settings).rerender([a.clip])
    print(f"Готово: {job.dir / clip['file']}")
    return 0


def cmd_cover(a, settings) -> int:
    from konveyer.cover import make_covers
    from konveyer.llm import make_llm
    from konveyer.pipeline import load_transcript, settings_for_job

    job = find_job(settings, a.job)
    tuned = settings_for_job(settings, job)
    tuned.cover.enabled = True
    if a.cover_mode:
        tuned.cover.mode = a.cover_mode
    log = Log(job.log_file)
    llm = None if (job.data.get("options") or {}).get("llm") is False else make_llm(
        tuned, log, job=job, warn_missing=False)
    make_covers(job, tuned, log, llm, load_transcript(job),
                title=a.title or None, only=a.clip)
    job.mark("cover", "done")
    print("Обложки готовы:")
    for clip in job.data.get("clips") or []:
        if a.clip and clip.get("id") != a.clip:
            continue
        for kind in ("frame", "ai"):
            rel = str((clip.get("covers") or {}).get(kind) or "")
            if rel:
                print(f"  {job.dir / rel}")
    return 0


def cmd_post(a, settings) -> int:
    from konveyer.pipeline import Pipeline

    job = find_job(settings, a.job)
    job.data["steps"].pop("post", None)
    Pipeline(job, settings).run(steps=("post",))
    print(job.post_md.read_text(encoding="utf-8"))
    return 0


def cmd_guard(a, settings) -> int:
    from konveyer.guard import guard_full
    from konveyer.pipeline import Pipeline

    job = find_job(settings, a.job)
    pipe = Pipeline(job, settings)
    res = guard_full(job, pipe.s, pipe.log, rerender=pipe.rerender_full if a.fix else None,
                     fix=bool(a.fix))
    print("сторож: " + ("в порядке" if res["ok"] else "есть замечания"))
    for line in res["issues"]:
        print("  " + line)
    return 0 if res["ok"] else 1


def cmd_describe(a, settings) -> int:
    from konveyer.clipdesc import make_descriptions
    from konveyer.llm import make_llm
    from konveyer.pipeline import load_transcript, settings_for_job

    job = find_job(settings, a.job)
    log = Log(job.log_file)
    tuned = settings_for_job(settings, job)
    tuned.post.clip_desc = True
    if not any(c.get("file") for c in job.data.get("clips") or []):
        print("в задании нет клипов Shorts")
        return 1
    llm = make_llm(tuned, log, job=job)
    out = make_descriptions(job, tuned, log, llm, load_transcript(job), only=a.clip)
    book = getattr(llm, "book", None)
    if book is not None:
        book.flush(final=True)
    for cid, text in out.items():
        print(f"--- клип {cid} ---\n{text}\n")
    return 0


def cmd_send(a, settings) -> int:
    from konveyer.pipeline import settings_for_job
    from konveyer.telegram import send_job

    job = find_job(settings, a.job)
    log = Log(job.log_file)
    tuned = settings_for_job(settings, job)
    if a.score:
        tuned.quality.score = True
    if a.vision_check:
        tuned.quality.vision = True
    if getattr(a, "no_voice_check", False):
        tuned.voice.voice_check = False
    if getattr(a, "shorts_post", False):
        tuned.telegram.shorts_post = True
    send_job(job, tuned, log, clips=not a.no_clips,
             post=not a.no_post, only=a.clip, force=a.force_send, resend=a.resend)
    return 0


def cmd_cleanup(a, settings) -> int:
    from konveyer.cleanup import report

    dirs = [find_job(settings, a.job).dir] if a.job else None
    print(report(settings, dry_run=a.dry_run, job_dirs=dirs))
    return 0


def cmd_web(a, settings) -> int:
    from konveyer.web.server import serve

    if a.port:
        settings.web.port = a.port
    serve(settings, open_browser=not a.no_open)
    return 0


def doctor_report(settings) -> list[tuple[str, bool, str]]:
    from konveyer.hardware import cuda_devices, detect_encoder

    rows: list[tuple[str, bool, str]] = []
    for tool in ("ffmpeg", "ffprobe", "yt-dlp"):
        path = shutil.which(tool)
        ver = ""
        if path:
            try:
                arg = "--version" if tool == "yt-dlp" else "-version"
                ver = subprocess.run([path, arg], capture_output=True, text=True,
                                     timeout=20).stdout.splitlines()[0][:60]
            except Exception:
                ver = path
        rows.append((tool, bool(path), ver or "не найден в PATH"))
    deno = shutil.which("deno")
    if deno:
        ver = deno
        try:
            out = subprocess.run([deno, "--version"], capture_output=True, text=True, timeout=20).stdout
            ver = (out.splitlines() or [deno])[0][:60]
        except Exception:
            pass
        rows.append(("deno", True, ver))
    else:
        rows.append(("deno", False, "нет в PATH — для YouTube: winget install DenoLand.Deno"))
    if shutil.which("ffmpeg"):
        enc = detect_encoder(settings.render.encoder, settings.render.crf, settings.render.preset)
        rows.append(("видеокодировщик", True, enc.title))
    cuda = cuda_devices()
    rows.append(("CUDA для Whisper", cuda > 0, f"устройств: {cuda}" if cuda else
                 "нет — Whisper на процессоре (модель " + settings.whisper.model_cpu + ")"))
    for mod, why in (("faster_whisper", "распознавание"), ("openai", "DeepSeek"),
                     ("requests", "Telegram"), ("mediapipe", "слежение за лицом"),
                     ("num2words", "цифры словами в озвучке")):
        try:
            __import__(mod)
            rows.append((mod, True, why))
        except Exception:
            rows.append((mod, False, f"не установлен ({why})"))
    rows += voice_rows(settings)
    rows.append(voice_check_row(settings))
    from konveyer.mediaq import ollama_status
    ok, note = ollama_status(settings)
    rows.append(("Ollama", ok, note))
    from konveyer.speakers import diarize_note
    ok, note = diarize_note(settings)
    rows.append(("диаризация", ok, note))
    for name in SECRET_NAMES:
        rows.append((name, has_secret(name), "задан" if has_secret(name) else "не задан"))
    from konveyer.layout import resolve_text_font
    for f in (settings.render.font_title, settings.render.font_text):
        found = resolve_text_font(f, settings.root)
        rows.append(("шрифт", bool(found), found or f"{f} — нет, надписи будут выключены"))
    from konveyer.cover import bundled_font
    font = Path(settings.cover.font) if settings.cover.font else bundled_font()
    if settings.cover.font and not font.is_absolute():
        font = settings.root / font
    rows.append(("шрифт обложки", font.is_file(), str(font)))
    try:
        import PIL
        rows.append(("pillow", True, f"обложки, pillow {getattr(PIL, '__version__', '')}".strip()))
    except Exception:
        rows.append(("pillow", False, "не установлен: py -m pip install pillow"))
    subs = Path(settings.paths.subs_dir) / "subs.py"
    rows.append(("subs.py (перевод)", subs.exists(), str(subs) if subs.exists() else "нет — встроенный перевод"))
    styles = [f for f in settings.post.style_files if Path(f).exists()]
    rows.append(("стиль поста", bool(styles), ", ".join(Path(f).name for f in styles) or "нет — общий стиль"))
    rows.append(("папка заданий", True, str(settings.jobs_dir)))
    return rows


XTTS_HINT = ("нет в этом python (на 3.14 coqui-tts не ставится) — отдельный venv на Python 3.12 "
             "и [voice] xtts_python, см. README, «Озвучка»")


def _xtts_status(settings) -> tuple[bool, str]:
    import importlib.util

    py = settings.voice.xtts_python
    if not py:
        if importlib.util.find_spec("TTS") is not None:
            return True, f"есть в текущем python {sys.version_info.major}.{sys.version_info.minor}"
        return False, XTTS_HINT
    if not Path(py).is_file():
        return False, f"нет файла {py} ([voice] xtts_python)"
    try:
        proc = subprocess.run([py, "-c", "import TTS, torch; print(TTS.__version__, torch.__version__)"],
                              capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, f"{py} не запускается ({e})"
    if proc.returncode != 0:
        return False, f"в {py} нет coqui-tts: {py} -m pip install coqui-tts"
    return True, f"coqui-tts/torch {proc.stdout.strip()[:40]} в {py}"


def edge_tts_status(timeout: float = 5.0) -> tuple[bool, str]:
    """Установлен ли edge-tts и отвечает ли сервис. Сеть — не дольше timeout секунд."""
    import importlib.util

    if importlib.util.find_spec("edge_tts") is None:
        return False, "не установлен: py -m pip install edge-tts"
    try:
        import edge_tts
    except Exception as e:
        return False, f"не импортируется ({str(e)[:80]})"
    ver = getattr(edge_tts, "__version__", "") or "установлен"
    return True, f"edge-tts {ver}, {_edge_service(edge_tts, timeout)}"


def _edge_service(edge_tts, timeout: float) -> str:
    import asyncio

    async def ping():
        return await asyncio.wait_for(edge_tts.list_voices(), timeout)

    try:
        voices = asyncio.run(ping())
        n = len(voices) if voices is not None else 0
        return f"сервис отвечает, голосов: {n}"
    except Exception:
        return "сервис не ответил за 5 с"


def voice_check_row(settings) -> tuple[str, bool, str]:
    """doctor: включена ли сверка озвучки и есть ли faster-whisper."""
    import importlib.util

    cfg = settings.voice
    if not cfg.voice_check:
        return ("проверка озвучки", True, "выключена (voice_check = false или --no-voice-check)")
    if importlib.util.find_spec("faster_whisper") is None:
        return ("проверка озвучки", False,
                "включена, нет faster-whisper — проверка будет пропускаться")
    return ("проверка озвучки", True, f"включена, порог {cfg.voice_check_min}")


def voice_rows(settings) -> list[tuple[str, bool, str]]:
    """Озвучка: Edge, torch, файл Silero, XTTS, demucs. Ничего из этого не критично."""
    import importlib.util

    rows: list[tuple[str, bool, str]] = []
    ok, note = edge_tts_status()
    rows.append(("Edge TTS", ok, note))
    try:
        import torch

        cuda = "CUDA" if torch.cuda.is_available() else "процессор"
        rows.append(("torch", True, f"{torch.__version__}, {cuda} (озвучка Silero)"))
    except Exception as e:
        rows.append(("torch", False, f"не импортируется ({str(e)[:60]}) — Silero не заработает"))
    model = Path(settings.voice.model_path)
    rows.append(("Silero v4_ru", model.is_file(),
                 str(model) if model.is_file() else f"нет файла {model} — [voice] model_path"))
    ok, note = _xtts_status(settings)
    rows.append(("XTTS v2", ok, note))
    has_demucs = importlib.util.find_spec("demucs") is not None
    rows.append(("demucs", has_demucs, "есть (--separate)" if has_demucs
                 else "не установлен (--separate): py -m pip install demucs"))
    return rows


def cmd_doctor(a, settings) -> int:
    rows = doctor_report(settings)
    for name, ok, note in rows:
        print(f"[{'OK' if ok else '--'}] {tr(name):<20} {tr(note)}")
    critical = {"ffmpeg", "ffprobe"}
    return 0 if all(ok for name, ok, _ in rows if name in critical) else 1


def cmd_config(a, settings) -> int:
    print(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2))
    return 0


COMMANDS = {"run": cmd_run, "full": cmd_full, "jobs": cmd_jobs, "show": cmd_show,
            "redo": cmd_redo, "clip": cmd_clip, "cover": cmd_cover,
            "post": cmd_post, "describe": cmd_describe, "send": cmd_send, "guard": cmd_guard, "web": cmd_web, "doctor": cmd_doctor,
            "config": cmd_config, "cleanup": cmd_cleanup}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = list(sys.argv[1:] if argv is None else argv)
    for i, x in enumerate(args):                      # язык нужен ещё до --help
        if x == "--lang" and i + 1 < len(args):
            i18n.set_lang(args[i + 1])
        elif x.startswith("--lang="):
            i18n.set_lang(x.split("=", 1)[1])
    parser = build_parser()
    i18n.translate_parser(parser)
    a = parser.parse_args(args)
    try:
        settings = load_settings(a.config, _overrides(a.set))
        i18n.init_from_settings(settings)
        if a.cmd not in ("cleanup", "config", "doctor"):
            from konveyer.cleanup import auto_cleanup

            auto_cleanup(settings, Log(None))   # старые отправленные задания — при каждом запуске
        return COMMANDS[a.cmd](a, settings)
    except KonveyerError as e:
        print(f"\nОШИБКА: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nОстановлено вручную.", file=sys.stderr)
        return 130
