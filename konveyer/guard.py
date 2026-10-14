"""Сторож целого видео: озвучка проверяется по фактам (voice.json и звук voice.wav) до Telegram.

Не оценка нейросети, а измерения: каждая фраза со звуком; нет дыр дольше HOLE_SEC, где
идёт речь оригинала с субтитром, а русского голоса нет; начало фразы отстаёт не больше
LAG_SEC; фраза не залезает на реплику другого спикера; у спикера один голос; концы не срезаны;
предложение одного спикера не рвётся паузой дольше BREAK_SEC.
Только отчёт: отправку не держит и сам ничего не переделывает. После видео в Telegram
уходит одно короткое сообщение «Сторож: …». Починка кругами — только py -m konveyer guard --fix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from konveyer.log import Log
from konveyer.util import read_json

HOLE_SEC = 3.0
LAG_SEC = 1.5
OVERLAP_SEC = 0.3
ROUNDS = 2
BREAK_SEC = 0.8        # пауза внутри предложения одного спикера — «разрыв фразы»
SHOW = 10              # столько самых заметных таймкодов в сообщении
# порядок важности; первым — фразы совсем без звука
KINDS = {"silent": "без звука", "fallback": "запасной голос", "hole": "нет русского голоса", "break": "разрыв фразы",
         "cut": "обрезан конец", "voice": "сменился голос", "overlap": "наезд на другого спикера",
         "lag": "отставание"}
FRAME_SEC = 0.05
ACTIVE_RMS = 0.01      # около −40 dBFS
BRIDGE_SEC = 0.35      # паузы внутри речи короче этого речь не прерывают


def tc(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def voice_activity(samples: np.ndarray, rate: int) -> list[tuple[float, float]]:
    """Где в русской дорожке реально есть голос, по энергии кадров."""
    x = np.asarray(samples, np.float32).reshape(-1)
    n = max(int(rate * FRAME_SEC), 1)
    frames = x[: x.size // n * n].reshape(-1, n)
    if frames.size == 0:
        return []
    loud = np.sqrt((frames ** 2).mean(axis=1)) > ACTIVE_RMS
    out: list[list[float]] = []
    for i, on in enumerate(loud):
        if not on:
            continue
        a, b = i * FRAME_SEC, (i + 1) * FRAME_SEC
        if out and a - out[-1][1] <= BRIDGE_SEC:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _merge(spans: list[tuple[float, float]], bridge: float = 0.0) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1] + bridge:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def holes(windows: list[tuple[float, float]], active: list[tuple[float, float]],
          limit: float = HOLE_SEC) -> list[tuple[float, float]]:
    """Куски речи оригинала (окна субтитров) без русского голоса дольше limit."""
    out = []
    act = _merge(active)
    for a, b in _merge(windows, bridge=0.3):
        t = a
        for x, y in act:
            if y <= t or x >= b:
                continue
            if x - t > limit:
                out.append((t, x))
            t = max(t, y)
        if b - t > limit:
            out.append((t, b))
    return out


def silence_at(active: list[tuple[float, float]], t: float) -> float | None:
    """Тишина русской дорожки на стыке t: от конца речи до t до начала речи после t.

    Edge кладёт тишину в начало и конец файла, поэтому смотрим звук, а не voice.json.
    """
    if not active:
        return None
    spans = _merge(active)
    for (a0, b0), (a1, b1) in zip(spans, spans[1:]):
        if t - 0.6 <= b0 <= t + 0.3 and a1 >= t - 0.1:
            return a1 - b0
    return 0.0


def check_phrases(phrases: list[dict], active: list[tuple[float, float]]) -> list[dict]:
    """Список нарушений: {"kind", "at", "text"}."""
    found: list[dict] = []

    from konveyer.voice import continues_sentence

    def add(kind: str, at: float, text: str, value: float = 0.0) -> None:
        found.append({"kind": kind, "at": round(float(at), 2), "text": text,
                      "value": round(float(value), 2)})

    voices: dict[str, set] = {}
    for p in phrases:
        if p.get("speaker"):
            voices.setdefault(p["speaker"], set()).add(p.get("voice") or "")
    for spk, vs in sorted(voices.items()):
        if len(vs) > 1:
            add("voice", 0.0, f"{spk}: голосов {len(vs)}")
    for i, p in enumerate(phrases):
        origin = float(p.get("origin", p.get("start", 0.0)) or 0.0)
        start = float(p.get("start") or 0.0)
        dur = float(p.get("duration") or 0.0)
        text = str(p.get("text") or "")[:40]
        if p.get("status") == "fail" or dur <= 0:
            add("silent", origin, f"без звука: «{text}»")
            continue
        if start - origin > LAG_SEC:
            add("lag", origin, f"отстаёт на {start - origin:.1f} с: «{text}»", start - origin)
        nxt = phrases[i + 1] if i + 1 < len(phrases) else None
        if nxt and (nxt.get("speaker") or "") != (p.get("speaker") or ""):
            n_origin = float(nxt.get("origin", nxt.get("start", 0.0)) or 0.0)
            if start + dur > n_origin + OVERLAP_SEC:
                add("overlap", origin, f"залезает на реплику {nxt.get('speaker') or '?'} "
                                       f"на {start + dur - n_origin:.1f} с: «{text}»",
                    start + dur - n_origin)
        if (nxt and (nxt.get("speaker") or "") == (p.get("speaker") or "")
                and continues_sentence(str(p.get("text") or ""))
                and nxt.get("status") != "fail" and float(nxt.get("duration") or 0.0) > 0):
            pause = float(nxt.get("start") or 0.0) - (start + dur)
            # внутри одного запроса Edge пауз без знака не делает — слушаем только стыки кусков
            heard = (silence_at(active, float(nxt.get("start") or 0.0))
                     if abs(pause) > 0.02 else None)
            if heard is not None:
                pause = max(pause, heard)
            if pause > BREAK_SEC:
                tail = " ".join(str(p.get("text") or "").split()[-2:])
                head = " ".join(str(nxt.get("text") or "").split()[:2])
                add("break", float(nxt.get("start") or 0.0),
                    f"разрыв фразы {pause:.1f} с: «…{tail} | {head}…»", pause)
        if p.get("fallback"):
            add("fallback", origin, f"запасной голос {p['fallback']}: «{text}»")
        if p.get("status") == "cut" or p.get("dropped"):
            add("cut", origin, f"обрезан конец: «{text}»")
    windows = [(float(p.get("origin", 0.0)), float(p.get("cue_end") or p.get("origin", 0.0)))
               for p in phrases]
    for a, b in holes(windows, active):
        add("hole", a, f"нет русского голоса {tc(a)}–{tc(b)} ({b - a:.1f} с)", b - a)
    return sorted(found, key=lambda f: f["at"])


def check_folder(folder: Path) -> list[dict]:
    from konveyer.voice import read_wav

    data = read_json(Path(folder) / "voice.json", {})
    phrases = data.get("phrases") if isinstance(data, dict) else None
    if not phrases:
        return [{"kind": "silent", "at": 0.0, "text": "нет voice.json — озвучки нет"}]
    wav = Path(folder) / "voice.wav"
    active: list[tuple[float, float]] = []
    if wav.is_file():
        samples, rate = read_wav(wav)
        active = voice_activity(samples, rate)
    return check_phrases(phrases, active)


def lines(found: list[dict], limit: int = 10) -> list[str]:
    out = [f"{tc(f['at'])} {f['text']}" for f in found[:limit]]
    if len(found) > limit:
        out.append(f"и ещё {len(found) - limit}")
    return out


def worst(found: list[dict], limit: int = SHOW) -> list[dict]:
    """Самые заметные: сначала по важности вида, внутри — по величине."""
    order = list(KINDS)
    ranked = sorted(found, key=lambda f: (order.index(f["kind"]) if f["kind"] in order else 99,
                                          -float(f.get("value") or 0.0)))
    picked: list[dict] = []
    for kind in order:                      # каждый вид хоть одним таймкодом
        first = next((f for f in ranked if f["kind"] == kind), None)
        if first is not None and len(picked) < limit:
            picked.append(first)
    for f in ranked:
        if len(picked) >= limit:
            break
        if f not in picked:
            picked.append(f)
    return sorted(picked, key=lambda f: f["at"])


def message(found: list[dict], limit: int = SHOW) -> str:
    """Одно короткое сообщение в Telegram: виды по важности, таймкоды по порядку."""
    if not found:
        return ""
    shown = worst(found, limit)
    rows = []
    for kind, label in KINDS.items():
        all_of = [f for f in found if f["kind"] == kind]
        if not all_of:
            continue
        mine = [f for f in shown if f["kind"] == kind]
        if kind == "voice":
            marks = [f["text"] for f in mine]
        elif kind == "fallback":
            marks = [tc(f["at"]) for f in mine]
        elif kind == "silent":
            marks = [f"{tc(f['at'])} {f['text'].split(': ', 1)[-1]}" for f in mine]
        else:
            marks = [f"{tc(f['at'])} ({float(f.get('value') or 0):.1f} с)" for f in mine]
        more = len(all_of) - len(mine)
        row = f"• {label}" + (f" — {len(all_of)}" if len(all_of) > 1 else "")
        if marks:
            row += ": " + ", ".join(marks)
        if more > 0:
            row += f"{'' if marks else ':'} и ещё {more}"
        rows.append(row)
    return "Сторож: замечания по озвучке\n" + "\n".join(rows)


def guard_full(job, settings, log: Log, rerender=None, fix: bool = False) -> dict:
    """Проверить озвучку целого видео. По умолчанию только отчёт; fix — чинить до ROUNDS кругов."""
    from konveyer import voice as voicemod

    folder = job.dir / "full"
    found = check_folder(folder)
    rounds = 0
    drift0, lag0, rate0 = voicemod.DRIFT_SEC, settings.full.lag_sec, settings.voice.edge_max_rate
    try:
        while found and fix and rerender is not None and rounds < ROUNDS:
            rounds += 1
            kinds = sorted({f["kind"] for f in found})
            log.warn(f"сторож: нарушений {len(found)} ({', '.join(kinds)}), круг {rounds}: "
                     f"переозвучиваю и пересобираю")
            if {"hole", "lag", "overlap"} & set(kinds):
                voicemod.DRIFT_SEC = max(0.8, voicemod.DRIFT_SEC / 2)
                settings.full.lag_sec = max(0.8, min(settings.full.lag_sec, LAG_SEC) - 0.3 * (rounds - 1))
                settings.voice.edge_max_rate = min(1.3, settings.voice.edge_max_rate + 0.07)
            for old in job.dir.glob("voice_cache/*.fallback.*"):
                old.unlink(missing_ok=True)
            rerender()
            found = check_folder(folder)
    finally:
        voicemod.DRIFT_SEC = drift0
        settings.full.lag_sec = lag0
        settings.voice.edge_max_rate = rate0
    counts = {k: sum(f["kind"] == k for f in found) for k in KINDS if any(f["kind"] == k for f in found)}
    result = {"ok": not found, "rounds": rounds, "counts": counts,
              "issues": lines(worst(found), SHOW), "message": message(found)}
    job.data["guard"] = result
    job.save()
    if found:
        log.warn("СТОРОЖ (только отчёт, отправку не держит): "
                 + ", ".join(f"{KINDS[k]} {n}" for k, n in counts.items()))
        for line in lines(found, 400):
            log.info(f"    {line}")
    else:
        log.info("сторож: озвучка в порядке")
    return result
