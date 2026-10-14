"""Проверка русской озвучки: что сказал Whisper, против текста субтитров.

После voice.wav клипа faster-whisper (язык ru) возвращает услышанное, DeepSeek
сравнивает фразы. Итог — clips/clip_XX/voice_check.json и строка в журнале.
Нет Whisper или DeepSeek — проверка пропускается, рендер не падает.
Перед отправкой низкая оценка или критичная проблема не пускает клип, как
субтитры за краем кадра.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from konveyer.config import Settings
from konveyer.jobs import Job
from konveyer.llm import chat_json, set_stage
from konveyer.saytext import prepare_for_speech
from konveyer.log import Log
from konveyer.util import read_json, write_json

CRITICAL = frozenset({"обрыв", "английский", "буквы", "разметка"})
# потолок оценки 4 — только за то, что слышно наверняка: английский, буквы, разметка
CAP_PROBLEMS = frozenset({"английский", "буквы", "разметка"})
CUT_FRAME = 0.03       # кадр для замера среза громкости на конце фразы, с
CUT_BODY = 0.5         # с какой частью речи перед концом сравниваем
CUT_LOUD = 0.5         # последний кадр не тише половины речи перед ним — ещё говорит
CUT_DROP = 0.2         # а сразу после конца — тише в 5 раз: срез, а не затухание
CUT_MIN_RMS = 0.01     # около −40 dBFS, как у сторожа
_PROBLEMS = CRITICAL | {"лишнее"}
_ALIAS = {
    "недоговорено": "обрыв",
    "обрыв/недоговорено": "обрыв",
    "обрезано": "обрыв",
}

_LABELS = {
    "обрыв": "фраза оборвана",
    "английский": "английский вместо русского",
    "буквы": "неверно прочитаны буквы или цифры",
    "разметка": "прочитана разметка",
}

# Слово целиком: числительное или порядковое. После нормализации ё уже стала е.
_NUM = re.compile(
    r"^(?:"
    r"нол[ьяею]*|нуля|"
    r"один|одна|одно|одну|одним|одного|одной|одному|"
    r"два|две|двух|двумя|"
    r"три|трех|тремя|"
    r"четыре|четырех|четырьмя|"
    r"пять|пяти|пятью|шесть|шести|шестью|семь|семи|семью|"
    r"восемь|восьми|восемью|девять|девяти|девятью|"
    r"десять|десяти|десятью|"
    r"одиннадцать\w*|двенадцать\w*|тринадцать\w*|четырнадцать\w*|"
    r"пятнадцать\w*|шестнадцать\w*|семнадцать\w*|восемнадцать\w*|девятнадцать\w*|"
    r"двадцать\w*|тридцать\w*|сорок\w*|"
    r"пятьдесят\w*|шестьдесят\w*|семьдесят\w*|восемьдесят\w*|девяносто\w*|"
    r"сто|ста|"
    r"двести|двухсот\w*|триста|трехсот\w*|четыреста|четырехсот\w*|"
    r"пятьсот\w*|шестьсот\w*|семьсот\w*|восемьсот\w*|девятьсот\w*|"
    r"тысяч\w*|миллион\w*|миллиард\w*|"
    r"перв\w*|втор\w*|трет\w*|четверт\w*|"
    r"\d+"
    r")$"
)

VOICE_CHECK_SYSTEM = (
    "Ты сравниваешь русскую озвучку с текстом субтитров по фразам. "
    "Тексты уже нормализованы: нижний регистр, без пунктуации, буква ё записана как е, "
    "числа заменены на знак #. Сторона субтитров уже в том виде, как фразу надо прочитать: "
    "одиночная «З» записана «зэ», латиница и годы — русскими словами (джи пи ти). "
    "Такое чтение — норма, не ошибка букв. Не считай ошибкой мелкие расхождения распознавания: "
    "окончание, одна буква, лишний или пропущенный короткий предлог, если слово то же. "
    "Верни ТОЛЬКО JSON {\"score\": целое от 1 до 10, "
    "\"phrases\": [{\"i\": 0, \"ok\": true, \"problem\": \"\"}]}. "
    "problem пустой или одно из: обрыв, английский, буквы, разметка, лишнее. "
    "обрыв — фраза не договорена или обрезана. "
    "английский — вместо русского слышен английский. "
    "буквы — неверно прочитаны буквы или цифры. "
    "разметка — прочитаны теги, слово break, time или другой служебный мусор. "
    "лишнее — сказаны слова, которых нет в субтитре. "
    "score: 10 — всё совпало, 1 — озвучка непригодна. Не переписывай фразы."
)

_WHISPER: dict = {"model": None, "key": None}


def normalize_said(text: str) -> str:
    """Сравнение без регистра, пунктуации, ё/е и без спора «5» против «пять»."""
    t = (text or "").casefold().replace("ё", "е")
    t = t.replace("-", " ")
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    words: list[str] = []
    for word in t.split():
        words.append("#" if word.isdigit() or _NUM.fullmatch(word) else word)
    out: list[str] = []
    for word in words:
        if word == "#" and out and out[-1] == "#":
            continue
        out.append(word)
    return " ".join(out)


def markup_in(text: str) -> bool:
    raw = text or ""
    low = raw.casefold()
    return "<" in raw or ">" in raw or "break" in low or "time=" in low


def _same_token(a: str, b: str) -> bool:
    left, right = normalize_said(a), normalize_said(b)
    return bool(left) and left == right


def align_heard(words: list[tuple[float, float, str]], phrases) -> list[str]:
    """Слова Whisper (start, end, text) разложить по окнам фраз.

    Окно не заходит на начало следующей фразы: иначе её первое слово
    попадает в предыдущую и проверка считает это «лишним».
    """
    spans: list[tuple[float, float] | None] = []
    for i, ph in enumerate(phrases):
        if getattr(ph, "status", "") == "fail" or float(getattr(ph, "duration", 0) or 0) <= 0:
            spans.append(None)
            continue
        a = float(ph.start) - 0.04
        b = float(ph.start) + float(ph.duration)
        if i + 1 < len(phrases):
            nxt = phrases[i + 1]
            if (getattr(nxt, "status", "") != "fail"
                    and float(getattr(nxt, "duration", 0) or 0) > 0):
                b = min(b + 0.03, float(nxt.start))
            else:
                b += 0.12
        else:
            b += 0.12
        if i > 0:
            prev = phrases[i - 1]
            if (getattr(prev, "status", "") != "fail"
                    and float(getattr(prev, "duration", 0) or 0) > 0):
                a = max(a, float(prev.start) + float(prev.duration))
        if b < a:
            b = a
        spans.append((a, b))
    buckets: list[list[str]] = [[] for _ in phrases]
    for start, end, word in words:
        text = (word or "").strip()
        if not text:
            continue
        mid = (float(start) + float(end)) / 2.0
        best_i, best_d = None, 1e9
        for i, span in enumerate(spans):
            if span is None:
                continue
            lo, hi = span
            if lo - 1e-6 <= mid <= hi + 1e-6:
                center = float(phrases[i].start) + float(phrases[i].duration) / 2.0
                dist = abs(mid - center)
                if dist < best_d:
                    best_i, best_d = i, dist
        if best_i is not None:
            buckets[best_i].append(text)
    return [" ".join(parts) for parts in buckets]


def without_neighbor_edge(heard: str, current: str, prev: str, nxt: str) -> str:
    """Убрать с края слово, которое принадлежит соседней фразе, а не этой.

    Если своя фраза сама так кончается или начинается, слово оставляем.
    """
    words = [w for w in (heard or "").split() if w]
    cur = [w for w in (current or "").split() if w]
    prev_w = [w for w in (prev or "").split() if w]
    next_w = [w for w in (nxt or "").split() if w]
    if words and next_w and _same_token(words[-1], next_w[0]):
        if not cur or not _same_token(cur[-1], words[-1]):
            words = words[:-1]
    if words and prev_w and _same_token(words[0], prev_w[-1]):
        if not cur or not _same_token(cur[0], words[0]):
            words = words[1:]
    return " ".join(words)


def hear_voice(path: Path, phrases, settings: Settings, log: Log) -> list[str]:
    """Распознать voice.wav языком ru. Модель та же, что у ролика; при сбое CUDA — CPU."""
    from faster_whisper import WhisperModel

    from konveyer.transcribe import load_audio, pick_device

    audio = load_audio(path)
    device, model_name, compute = pick_device(settings)
    key = (device, model_name, compute)
    model = _WHISPER["model"] if _WHISPER["key"] == key else None
    if model is None:
        log.info(f"проверка озвучки: модель {model_name} ({device})")
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute)
        except Exception:
            if device == "cpu":
                raise
            w = settings.whisper
            device, model_name, compute = "cpu", w.model_cpu, w.compute_cpu
            log.info(f"проверка озвучки: CUDA не вышла, модель {model_name} (cpu)")
            model = WhisperModel(model_name, device=device, compute_type=compute)
            key = (device, model_name, compute)
        _WHISPER["model"] = model
        _WHISPER["key"] = key
    segments, _info = model.transcribe(
        audio, language="ru", vad_filter=False, word_timestamps=True,
        beam_size=max(int(settings.whisper.beam_size), 1))
    words: list[tuple[float, float, str]] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append((float(w.start or 0), float(w.end or 0), w.word or ""))
        elif (seg.text or "").strip():
            words.append((float(seg.start or 0), float(seg.end or 0), seg.text or ""))
    return align_heard(words, phrases)


def _rms(x) -> float:
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def cut_by_sound(samples, rate: int, end: float) -> bool:
    """Звук подтверждает обрыв: голос громкий до самого конца фразы и пропадает за один кадр.

    Естественный конец затухает (Edge оставляет хвост тишины), срезанный — нет.
    Тот же подход, что у сторожа на стыках кусков: мерим звук, а не текст.
    """
    import numpy as np

    x = np.asarray(samples, np.float32).reshape(-1)
    n = max(int(rate * CUT_FRAME), 1)
    e = int(round(float(end) * rate))
    if e - 2 * n <= 0 or e > x.size:
        return False
    before = _rms(x[e - n:e])
    body = _rms(x[max(0, e - int(CUT_BODY * rate)):e - n])
    after = _rms(x[e:e + n]) if e < x.size else 0.0
    return before >= CUT_MIN_RMS and before >= CUT_LOUD * body and after <= CUT_DROP * before


def _tc(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _load_voice(folder: Path):
    """voice.wav клипа для замеров. Нет или не читается — None: обрыв не подтверждён."""
    try:
        from konveyer.voice import read_wav
        return read_wav(Path(folder) / "voice.wav")
    except Exception:
        return None


def _problem(raw: str) -> str:
    text = _ALIAS.get((raw or "").strip(), (raw or "").strip())
    return text if text in _PROBLEMS else ""


def review_voice(folder: Path, phrases, settings: Settings, log: Log, llm, clip_id,
                 hear=None, audio: Path | None = None) -> dict:
    """Сверить озвучку клипа с текстом. Файл пишется всегда, кроме сбоя записи.

    audio — другой wav, если слушать надо не весь voice.wav, а короткую выборку.
    """
    path = Path(audio) if audio is not None else Path(folder) / "voice.wav"
    dest = Path(folder) / "voice_check.json"
    minimum = settings.voice.voice_check_min

    def skip(reason: str) -> dict:
        payload = {"skipped": True, "reason": reason, "score": None, "ok": True,
                   "phrases": [], "min_score": minimum}
        write_json(dest, payload)
        log.warn(f"клип {clip_id}: проверка озвучки пропущена ({reason})")
        return payload

    if not path.is_file():
        return skip("нет voice.wav")
    try:
        heard = list(hear(path, phrases) if hear is not None else hear_voice(path, phrases, settings, log))
    except Exception as e:
        return skip(f"Whisper недоступен ({str(e)[:140]})")
    if llm is None:
        return skip("нет DeepSeek")
    if len(heard) < len(phrases):
        heard = heard + [""] * (len(phrases) - len(heard))
    speech = []
    for ph in phrases:
        if ph.status == "fail" or ph.duration <= 0:
            speech.append("")
        else:
            speech.append(prepare_for_speech(ph.spoken or ph.text, {**settings.voice.replacements,
                                                          **settings.voice.pronounce}))
    rows = []
    for i, ph in enumerate(phrases):
        if ph.status == "fail" or ph.duration <= 0:
            continue
        expected = ph.spoken or ph.text
        # «З» и GPT в субтитре — это «зэ» и «джи-пи-ти» в голосе. Сверяем чтение, не экран.
        said = speech[i]
        heard_i = without_neighbor_edge(
            heard[i], said, speech[i - 1] if i else "", speech[i + 1] if i + 1 < len(speech) else "")
        rows.append({"i": i, "text": expected, "heard": heard_i,
                     "norm_text": normalize_said(said),
                     "norm_heard": normalize_said(heard_i)})
    if not rows:
        return skip("нет фраз")
    user = json.dumps(
        [{"i": r["i"], "субтитры": r["norm_text"], "услышано": r["norm_heard"]} for r in rows],
        ensure_ascii=False)
    set_stage(llm, "voice_check")
    try:
        data = chat_json(llm, VOICE_CHECK_SYSTEM, user, temperature=0)
    except Exception as e:
        return skip(f"DeepSeek недоступен ({str(e)[:140]})")
    if not isinstance(data, dict):
        return skip("DeepSeek вернул не объект")
    try:
        score = int(data.get("score"))
    except (TypeError, ValueError):
        return skip("DeepSeek не дал оценку")
    score = min(10, max(1, score))
    by_i: dict[int, dict] = {}
    for item in data.get("phrases") or []:
        if isinstance(item, dict) and "i" in item:
            try:
                by_i[int(item["i"])] = item
            except (TypeError, ValueError):
                continue
    phrases_out = []
    warnings: list[str] = []
    sound = None
    for r in rows:
        item = by_i.get(r["i"], {})
        problem = _problem(str(item.get("problem") or ""))
        if item.get("ok") is False and not problem:
            problem = "лишнее"
        if markup_in(r["heard"]):
            problem = "разметка"
        warning = ""
        if problem == "обрыв":
            # Whisper часто теряет последнее слово целой фразы. Обрыв — только если слышно срез.
            ph = phrases[r["i"]]
            end = float(ph.start) + float(ph.duration)
            if sound is None:
                sound = _load_voice(folder) or ((), 1)
            if not cut_by_sound(sound[0], sound[1], end):
                problem = ""
                warning = f"обрыв не подтверждён звуком ({_tc(end)})"
                warnings.append(warning)
        ok = not problem and (item.get("ok") is not False or bool(warning))
        out = {"i": r["i"], "text": r["text"], "heard": r["heard"],
               "ok": ok, "problem": problem, "critical": problem in CRITICAL}
        if warning:
            out["warning"] = warning
        phrases_out.append(out)
    if any(p["problem"] in CAP_PROBLEMS for p in phrases_out):
        score = min(score, 4)
    elif warnings and not any(p["problem"] for p in phrases_out):
        # единственная претензия — неподтверждённый обрыв: оценку не режем
        score = max(score, minimum)
    payload = {
        "skipped": False,
        "score": score,
        "min_score": minimum,
        "ok": score >= minimum and not any(p["critical"] for p in phrases_out),
        "phrases": phrases_out,
    }
    write_json(dest, payload)
    names = [p["problem"] for p in phrases_out if p["problem"]]
    tail = ""
    if names:
        tail = " — " + ", ".join(dict.fromkeys(names))
    log.info(f"клип {clip_id}: проверка озвучки {score} из 10{tail}")
    for w in warnings[:5]:
        log.info(f"клип {clip_id}: предупреждение — {w}")
    return payload


def sample_groups(phrases, n: int, window: float) -> list[list[int]]:
    """Индексы фраз: несколько окон по ролику, каждое около window секунд речи.

    Короткий ролик, который и так короче одного окна, возвращается целиком.
    Окна не делят одну и ту же фразу дважды.
    """
    active = []
    for i, ph in enumerate(phrases):
        if float(getattr(ph, "duration", 0) or 0) <= 0.05:
            continue
        if getattr(ph, "status", "") == "fail":
            continue
        active.append(i)
    if not active:
        return []
    n = max(int(n), 1)
    window = max(float(window), 0.5)
    total_dur = sum(float(phrases[i].duration) for i in active)
    if total_dur <= window * 1.15:
        return [active]
    first = float(phrases[active[0]].start)
    last = max(float(phrases[i].start) + float(phrases[i].duration) for i in active)
    span = max(last - first, 0.1)
    centers = [first + span * (i + 0.5) / n for i in range(n)]
    used: set[int] = set()
    groups: list[list[int]] = []
    for center in centers:
        nearest = min(active, key=lambda i: abs((float(phrases[i].start)
                                                 + float(phrases[i].duration) / 2) - center))
        pos = active.index(nearest)
        chosen = [nearest]
        dur = float(phrases[nearest].duration)
        left, right = pos - 1, pos + 1
        while dur < window and (left >= 0 or right < len(active)):
            take_left = left >= 0
            take_right = right < len(active)
            if take_left and take_right:
                pick_left = abs(float(phrases[active[left]].start) - center) <= abs(
                    float(phrases[active[right]].start) - center)
            else:
                pick_left = take_left
            if pick_left:
                chosen.insert(0, active[left])
                dur += float(phrases[active[left]].duration)
                left -= 1
            else:
                chosen.append(active[right])
                dur += float(phrases[active[right]].duration)
                right += 1
        chosen = [i for i in chosen if i not in used]
        if not chosen:
            continue
        used.update(chosen)
        groups.append(chosen)
    return groups


def build_sample_wav(src: Path, phrases, groups: list[list[int]], dest: Path,
                     gap: float = 0.25):
    """Склеить окна в короткий wav и вернуть фразы со временем от начала склейки."""
    from dataclasses import replace

    import numpy as np

    from konveyer.voice import read_wav, write_wav

    data, rate = read_wav(src)
    pieces = []
    shifted = []
    cursor = 0.0
    gap_n = int(rate * gap)
    for g_i, group in enumerate(groups):
        for idx in group:
            ph = phrases[idx]
            a = max(int(float(ph.start) * rate), 0)
            b = min(int((float(ph.start) + float(ph.duration)) * rate), int(data.size))
            chunk = data[a:b]
            if chunk.size == 0:
                continue
            pieces.append(chunk)
            dur = chunk.size / rate
            shifted.append(replace(ph, start=round(cursor, 3), origin=round(cursor, 3),
                                   duration=round(dur, 3)))
            cursor += dur
        if g_i + 1 < len(groups) and gap_n > 0:
            pieces.append(np.zeros(gap_n, np.float32))
            cursor += gap
    if not pieces:
        return []
    write_wav(dest, np.concatenate(pieces), rate)
    return shifted


def review_voice_sample(folder: Path, phrases, settings: Settings, log: Log, llm, clip_id, *,
                        samples: int, window: float, hear=None) -> dict:
    """Проверка озвучки по нескольким фрагментам, а не по всему ролику."""
    groups = sample_groups(phrases, samples, window)
    sample = Path(folder) / "voice_sample.wav"
    if not groups:
        return review_voice(folder, phrases, settings, log, llm, clip_id, hear=hear)
    shifted = build_sample_wav(Path(folder) / "voice.wav", phrases, groups, sample)
    if not shifted:
        return review_voice(folder, phrases, settings, log, llm, clip_id, hear=hear)
    heard_sec = sum(float(p.duration) for p in shifted)
    log.info(f"клип {clip_id}: проверка озвучки по выборке — {len(groups)} фрагм., "
             f"{len(shifted)} фраз, около {heard_sec:.0f} с, не весь ролик")
    result = review_voice(folder, shifted, settings, log, llm, clip_id, hear=hear, audio=sample)
    result["sampled"] = True
    result["fragments"] = len(groups)
    result["sample_phrases"] = len(shifted)
    write_json(Path(folder) / "voice_check.json", result)
    return result


def _check_path(job: Job, clip: dict) -> Path:
    rel = str(clip.get("file") or "")
    if rel and (clip.get("full") or (job.dir / rel).parent.joinpath("voice_check.json").is_file()):
        return (job.dir / rel).parent / "voice_check.json"
    try:
        return job.clip_dir(int(clip["id"])) / "voice_check.json"
    except (KeyError, TypeError, ValueError):
        return job.dir / "voice_check.json"


def voice_notes(job: Job, settings: Settings) -> str:
    """Текст «Сторож: …» после целого видео. Без оценок моделей; фразы без звука — первыми."""
    guard = job.data.get("guard") or {}
    text = str(guard.get("message") or "")
    lost = [m.group(1) for c in job.data.get("clips") or []
            for m in [re.search(r"не озвучено (\d+)", str(c.get("voice") or ""))] if m]
    if lost and "без звука" not in text:
        head = f"• без звука: {sum(int(x) for x in lost)} фраз"
        text = ("Сторож: замечания по озвучке\n" + head + ("\n" + text.split("\n", 1)[1]
                                                          if "\n" in text else ""))
    heard: list[str] = []
    if settings.voice.voice_check:
        for clip in job.data.get("clips") or []:
            data = read_json(_check_path(job, clip), {})
            for ph in (data.get("phrases") or []) if isinstance(data, dict) else []:
                label = _LABELS.get(str((ph or {}).get("problem") or "")) if isinstance(ph, dict) else None
                if label and label not in heard and (ph.get("critical") or ph.get("problem") in CRITICAL):
                    heard.append(label)
    if heard:
        text = (text or "Сторож: замечания по озвучке") + "\n• на слух: " + ", ".join(heard[:3])
    return text


def gate_issues(job: Job, settings: Settings) -> list[str]:
    """Замечания для проверки перед Telegram. Пропуск и выключенный флаг — не отказ.

    Целое видео озвучка не держит: замечания уходят отдельным сообщением (voice_notes).
    """
    issues: list[str] = []
    if (job.data.get("options") or {}).get("full"):
        return issues
    for clip in job.data.get("clips") or []:
        lost = re.search(r"не озвучено (\d+)", str(clip.get("voice") or ""))
        if lost:
            issues.append(f"клип {clip.get('id')}: ОЗВУЧКА НЕПОЛНАЯ — без звука {lost.group(1)} фраз")
    if not settings.voice.voice_check:
        return issues
    for clip in job.data.get("clips") or []:
        try:
            cid = int(clip["id"])
        except (KeyError, TypeError, ValueError):
            continue
        path = _check_path(job, clip)
        if not path.is_file():
            continue
        data = read_json(path, {})
        if not isinstance(data, dict) or data.get("skipped"):
            continue
        try:
            score = float(data.get("score"))
        except (TypeError, ValueError):
            score = None
        if score is not None and score < settings.voice.voice_check_min:
            shown = int(score) if score == int(score) else score
            issues.append(
                f"клип {cid}: озвучка {shown} из 10, ниже порога {settings.voice.voice_check_min}")
        seen: set[str] = set()
        for ph in data.get("phrases") or []:
            if not isinstance(ph, dict):
                continue
            problem = str(ph.get("problem") or "")
            if problem in _LABELS and problem not in seen and (ph.get("critical") or problem in CRITICAL):
                seen.add(problem)
                issues.append(f"клип {cid}: озвучка — {_LABELS[problem]}")
    return issues
