"""Кто говорит: диаризация pyannote и метки S1, S2 на словах.

Нет токена, пакета или сам разбор упал — слова остаются без метки.
Субтитры тогда одним стилем, задание не падает.

Голос Edge подбирается по полу и закрепляется за меткой на весь ролик.
Цвет субтитров и голос смотрят на одну и ту же метку S1, S2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from konveyer.config import EDGE_VOICES, Settings
from konveyer.envkeys import get_secret, mask
from konveyer.log import Log
from konveyer.transcript import Transcript
from konveyer.util import KonveyerError, read_json


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


def hf_token(settings: Settings | None = None) -> str:
    """Сначала окружение, потом [subs] hf_token. В журнал значение не попадает."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = get_secret(name)
        if value:
            return value
    if settings is not None:
        return str(getattr(settings.subs, "hf_token", "") or "").strip()
    return ""


def _scrub(text: str, token: str) -> str:
    if token and len(token) >= 4 and token in text:
        text = text.replace(token, "***")
    return text


def canon_turns(turns: list[Turn]) -> list[Turn]:
    """Порядок появления → S1, S2, … Пустые и нулевые окна выкидываются."""
    order: list[str] = []
    for turn in turns:
        name = str(turn.speaker or "").strip()
        if name and name not in order:
            order.append(name)
    mapping = {raw: f"S{i + 1}" for i, raw in enumerate(order)}
    out = []
    for turn in turns:
        name = mapping.get(str(turn.speaker or "").strip(), "")
        if not name or turn.end <= turn.start:
            continue
        out.append(Turn(float(turn.start), float(turn.end), name))
    return out


SMOOTH_SEC = 1.5        # короче — выброс диаризации, если вокруг тот же спикер
FEMALE_F0 = 200.0       # женский голос только выше этого; неуверенно — мужской


def smooth_turns(turns: list[Turn], min_len: float = SMOOTH_SEC) -> list[Turn]:
    """Короткий кусок между двумя кусками одного спикера отдаём этому спикеру."""
    out = [Turn(t.start, t.end, t.speaker) for t in sorted(turns, key=lambda t: (t.start, t.end))]
    for i in range(1, len(out) - 1):
        t = out[i]
        prev, nxt = out[i - 1].speaker, out[i + 1].speaker
        if t.end - t.start < min_len and prev == nxt and t.speaker != prev:
            t.speaker = prev
    return out


def smooth_words(words: list, min_len: float = SMOOTH_SEC) -> None:
    """Слова без метки — к соседу. Короткая вставка между словами одного спикера — к нему.

    Так голос и цвет не скачут на одном слове или перебивке.
    """
    words = list(words)
    if not any(w.speaker for w in words):
        return
    last = ""
    for w in words:
        if w.speaker:
            last = w.speaker
        elif last:
            w.speaker = last
    last = ""
    for w in reversed(words):
        if w.speaker:
            last = w.speaker
        else:
            w.speaker = last
    runs: list[list[int]] = []
    for i, w in enumerate(words):
        if runs and words[runs[-1][0]].speaker == w.speaker:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    labels = [words[a].speaker for a, _b in runs]
    for k in range(1, len(runs) - 1):
        a, b = runs[k]
        if (labels[k - 1] == labels[k + 1] and labels[k] != labels[k - 1]
                and words[b].end - words[a].start < min_len):
            labels[k] = labels[k - 1]
            for w in words[a:b + 1]:
                w.speaker = labels[k]


def label_at(t: float, turns: list[Turn]) -> str:
    """Спикер в момент t. Несколько окон — берём самое короткое."""
    hits = [turn for turn in turns if turn.start - 0.02 <= t <= turn.end + 0.02]
    if not hits:
        return ""
    hits.sort(key=lambda turn: (turn.end - turn.start, abs(t - (turn.start + turn.end) / 2)))
    return hits[0].speaker


def apply_turns(tr: Transcript, turns: list[Turn]) -> int:
    """Повесить S1/S2 на слова. Возвращает, сколько слов получили метку."""
    labeled = smooth_turns(canon_turns(turns))
    n = 0
    for seg in tr.segments:
        for word in seg.words or []:
            word.speaker = label_at((word.start + word.end) / 2, labeled)
    smooth_words([w for seg in tr.segments for w in (seg.words or [])])
    for seg in tr.segments:
        if seg.words:
            n += sum(1 for w in seg.words if w.speaker)
            names = [w.speaker for w in seg.words if w.speaker]
            seg.speaker = names[0] if names and all(x == names[0] for x in names) else ""
        elif seg.text.strip():
            seg.speaker = label_at((seg.start + seg.end) / 2, labeled)
            if seg.speaker:
                n += 1
    return n


def speaker_names(tr: Transcript) -> list[str]:
    found: list[str] = []
    for word in tr.words:
        if word.speaker and word.speaker not in found:
            found.append(word.speaker)
    return found


def edge_voice_for(settings: Settings, label: str, cast: dict | None = None) -> str:
    """Полное имя голоса Edge для спикера или пусто — оставить голос по умолчанию.

    Только Edge и только когда voice.per_speaker включён (это обычный путь).
    Silero и XTTS не трогаем. cast — короткие имена, закреплённые за S1, S2 на весь ролик.
    Без каста голос берётся из палитры по номеру метки, чтобы клип из одного S2
    не получил голос S1.
    """
    cfg = settings.voice
    if not settings.subs.speakers or not cfg.per_speaker or cfg.engine != "edge":
        return ""
    if not label:
        return ""
    short = fixed_voices(settings).get(label, "")
    if not short and cast is not None:
        short = str(cast.get(label) or "").strip()
    if not short:
        voices = _names(cfg.speaker_voices)
        slot = _slot(label)
        if slot < 0 or not voices:
            return ""
        short = voices[slot % len(voices)]
    full = EDGE_VOICES.get(short, "")
    if not full or full == EDGE_VOICES.get(cfg.speaker, ""):
        return ""
    return full


def fixed_voices(settings: Settings) -> dict[str, str]:
    """[voice] speaker_map: явный голос на метку, например S2 = "dmitry". Пол не проверяем."""
    raw = getattr(settings.voice, "speaker_map", None) or {}
    return {str(k).strip(): str(v).strip() for k, v in raw.items()
            if str(k).strip() and str(v).strip() in EDGE_VOICES}


def speaker_color(settings: Settings, label: str) -> str:
    """Тот же слот, что у цвета субтитров: S1 — первый цвет, S2 — второй."""
    colors = [str(c) for c in (settings.subs.speaker_colors or []) if str(c).strip()]
    if not colors:
        return "#FFFFFF"
    slot = _slot(label)
    if slot < 0:
        return colors[0]
    return colors[slot % len(colors)]


def build_cast(settings: Settings, labels: list[str],
               genders: dict[str, str] | None = None) -> dict[str, str]:
    """Короткое имя Edge на каждую метку. Голоса не повторяются.

    Пол известен — мужской или женский набор. Неизвестен — разные голоса из палитры.
    Один спикер без пола остаётся на голосе по умолчанию. Известный пол может его сменить.
    Голос по умолчанию берём первым, если он подходит этому полу и ещё свободен.
    """
    genders = genders or {}
    cfg = settings.voice
    male = _names(cfg.male_voices) or ["brian", "florian", "dmitry"]
    female = _names(cfg.female_voices) or ["ava", "emma", "seraphina", "svetlana"]
    palette = _names(cfg.speaker_voices) or list(dict.fromkeys([*male, *female]))
    default = cfg.speaker if cfg.speaker in EDGE_VOICES else ""
    used: set[str] = set()
    ordered = sorted({str(x) for x in labels if str(x)}, key=lambda name: (_slot(name), name))

    def take(pool: list[str], prefer: bool) -> str:
        if prefer and default and default in pool and default not in used:
            used.add(default)
            return default
        for name in pool:
            if name not in used:
                used.add(name)
                return name
        return ""

    cast: dict[str, str] = {}
    fixed = fixed_voices(settings)
    for label in ordered:
        if label in fixed:
            cast[label] = fixed[label]
            used.add(fixed[label])
    only = len(ordered) == 1
    for label in ordered:
        if label in cast:
            continue
        gender = genders.get(label, "")
        if gender == "male":
            voice = take(male, True) or take(female, False) or take(palette, False)
        elif gender == "female":
            voice = take(female, True) or take(male, False) or take(palette, False)
        elif only:
            voice = default or (palette[0] if palette else "brian")
            used.add(voice)
        else:
            voice = take(palette, True) or take(male, False) or take(female, False)
        cast[label] = voice or "brian"
    return cast


def load_cast(job_dir: Path) -> dict[str, str] | None:
    """Каст из job.json. None — каста ещё нет, тогда палитра по номеру метки."""
    data = read_json(Path(job_dir) / "job.json")
    if not data:
        return None
    meta = data.get("meta") or {}
    if "speaker_cast" not in meta:
        return None
    raw = meta.get("speaker_cast") or {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        name = str(value or "").strip()
        if name in EDGE_VOICES:
            out[str(key)] = name
    return out


def speaker_genders(audio, sample_rate: int, turns: list[Turn]) -> dict[str, str]:
    """male / female / "" по медиане F0. Мало звука или неясно — пустая строка."""
    import numpy as np
    wave = np.asarray(audio, np.float32).reshape(-1)
    sr = int(sample_rate or 0)
    if sr <= 0 or wave.size == 0:
        return {}
    found: dict[str, str] = {}
    for label, chunk in _chunks(wave, sr, canon_turns(turns)).items():
        found[label] = _gender(_spread_f0(chunk, sr))
    return found


def _spread_f0(wave, sample_rate: int, windows: int = 30, span: float = 2.0) -> float | None:
    """Медиана F0 по окнам через весь голос спикера, а не по первым секундам."""
    import numpy as np
    size = int(span * sample_rate)
    n = int(wave.size)
    if n < 2 * size:
        return _median_f0(wave, sample_rate)
    vals = []
    for i in range(windows):
        start = int(i * (n - size) / max(windows - 1, 1))
        f0 = _median_f0(wave[start:start + size], sample_rate)
        if f0:
            vals.append(f0)
    if len(vals) < 3:
        return _median_f0(wave, sample_rate)
    return float(np.median(np.asarray(vals, np.float32)))


def _names(values) -> list[str]:
    out = []
    for value in values or []:
        name = str(value or "").strip()
        if name in EDGE_VOICES and name not in out:
            out.append(name)
    return out


def _chunks(wave, sample_rate: int, turns: list[Turn]) -> dict[str, object]:
    import numpy as np
    parts: dict[str, list] = {}
    n = int(wave.size)
    for turn in turns:
        a = max(0, int(turn.start * sample_rate))
        b = min(n, int(turn.end * sample_rate))
        if b > a:
            parts.setdefault(turn.speaker, []).append(wave[a:b])
    return {label: np.concatenate(bits) for label, bits in parts.items() if bits}


def _median_f0(wave, sample_rate: int) -> float | None:
    """Медиана основного тона, Гц. Нужно около 0.4 с и несколько звонких кадров."""
    import numpy as np
    wave = np.asarray(wave, np.float32).reshape(-1)
    if wave.size < int(0.4 * sample_rate):
        return None
    frame = max(int(0.04 * sample_rate), 8)
    hop = max(frame // 2, 1)
    min_lag = max(1, int(sample_rate / 320))
    max_lag = min(frame - 2, int(sample_rate / 70))
    if max_lag <= min_lag:
        return None
    window = np.hanning(frame).astype(np.float32)
    pitches = []
    for start in range(0, wave.size - frame + 1, hop):
        chunk = wave[start:start + frame]
        if float(np.sqrt(np.dot(chunk, chunk) / frame)) < 0.01:
            continue
        chunk = (chunk - float(chunk.mean())) * window
        corr = np.correlate(chunk, chunk, mode="full")
        corr = corr[len(corr) // 2:]
        if corr[0] <= 1e-8:
            continue
        zone = corr[min_lag:max_lag + 1]
        peak = int(np.argmax(zone)) + min_lag
        if corr[peak] / corr[0] < 0.45:
            continue
        pitches.append(sample_rate / peak)
        if len(pitches) >= 40:
            break
    if len(pitches) < 5:
        return None
    return float(np.median(np.asarray(pitches, np.float32)))


def _gender(f0: float | None) -> str:
    if f0 is None:
        return ""
    # Неуверенно — мужской: высокий мужской голос не должен получить женский тембр.
    return "female" if f0 >= FEMALE_F0 else "male"


_GENDER_RU = {"male": "мужской", "female": "женский", "": "пол не ясен"}


def _cast_note(settings: Settings, names: list[str], genders: dict[str, str],
               cast: dict[str, str]) -> str:
    bits = []
    for name in names:
        bits.append(
            f"{name} {_GENDER_RU.get(genders.get(name, ''), 'пол не ясен')} → "
            f"{cast.get(name, '')}, {speaker_color(settings, name)}")
    return "; ".join(bits)


def _slot(label: str) -> int:
    import re
    found = re.fullmatch(r"S(\d+)", str(label or "").strip())
    if not found:
        return 0 if label else -1
    return max(int(found.group(1)) - 1, 0)


class PyannoteDiarizer:
    """pyannote/speaker-diarization-3.1. Модель закрытая: нужен HF_TOKEN и согласие на странице модели."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipe = None

    def turns(self, audio, sample_rate: int) -> list[Turn]:
        token = hf_token(self.settings)
        if not token:
            raise KonveyerError("нет HF_TOKEN")
        try:
            import numpy as np
            import torch
            from pyannote.audio import Pipeline
        except Exception as e:
            raise KonveyerError(
                "нет пакета pyannote.audio (и torch). "
                "На ПК: py -m pip install torch --index-url https://download.pytorch.org/whl/cu128 "
                "&& py -m pip install pyannote.audio"
            ) from e
        wave = np.asarray(audio, np.float32).reshape(-1)
        if wave.size == 0:
            return []
        tensor = torch.from_numpy(wave).unsqueeze(0)
        try:
            result = self._pipeline(token, Pipeline, torch)(
                {"waveform": tensor, "sample_rate": int(sample_rate)})
        except Exception as e:
            raise KonveyerError(_scrub(str(e), token)[:300]) from e
        return [Turn(a, b, name) for a, b, name in _iter_turns(result)]

    def _pipeline(self, token: str, Pipeline, torch):
        if self._pipe is not None:
            return self._pipe
        model = self.settings.subs.diarize_model
        try:
            pipe = Pipeline.from_pretrained(model, token=token)
        except TypeError:
            pipe = Pipeline.from_pretrained(model, use_auth_token=token)
        if pipe is None:
            raise KonveyerError(f"не загрузилась модель {model}")
        device = self.settings.subs.diarize_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipe.to(torch.device("cuda"))
        self._pipe = pipe
        return pipe


def resolve_overlaps(turns: list[Turn]) -> list[Turn]:
    """Перекрытия речи без потери смены спикера.

    Короткая вставка целиком внутри чужой реплики («угу», «да») — владельцу реплики.
    Передача слова (новый говорит и после конца старого) — новому с момента его начала:
    exclusive-разметка pyannote в этом месте часто оставляет фразу старому спикеру.
    """
    out: list[Turn] = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        t = Turn(float(t.start), float(t.end), str(t.speaker))
        keep = True
        for o in out:
            if o.speaker == t.speaker or o.end <= t.start:
                continue
            if t.end <= o.end:
                keep = False
                break
            o.end = t.start
        if keep:
            out.append(t)
    return [t for t in out if t.end > t.start]


def _iter_turns(result):
    ann = getattr(result, "speaker_diarization", None)
    if ann is None:
        ann = getattr(result, "exclusive_speaker_diarization", None)
    if ann is None:
        ann = result
    raw = [Turn(float(turn.start), float(turn.end), str(label))
           for turn, _track, label in ann.itertracks(yield_label=True)]
    for t in resolve_overlaps(raw):
        yield t.start, t.end, t.speaker


def attach_speakers(job, settings: Settings, log: Log, tr: Transcript, audio, sample_rate: int,
                    diarizer=None) -> Transcript:
    """Разметить расшифровку. Любой сбой — одна строка в журнале и прежние слова."""
    if not settings.subs.speakers:
        log.info("разные спикеры выключены")
        return tr
    if tr.empty or audio is None:
        return tr
    token = hf_token(settings)
    try:
        engine = diarizer if diarizer is not None else PyannoteDiarizer(settings)
        turns = list(engine.turns(audio, sample_rate) or [])
    except Exception as e:
        log.info(f"диаризация не вышла ({mask(_scrub(str(e), token))[:220]}) — субтитры одним стилем")
        return tr
    if not canon_turns(turns):
        log.info("диаризация не нашла спикеров — субтитры одним стилем")
        return tr
    n = apply_turns(tr, turns)
    names = speaker_names(tr)
    if not names:
        log.info("диаризация не нашла спикеров — субтитры одним стилем")
        return tr
    genders = speaker_genders(audio, sample_rate, turns)
    cast = build_cast(settings, names, genders)
    from konveyer.util import write_json
    write_json(job.transcript_file, tr.to_dict())
    meta = job.data.setdefault("meta", {})
    meta["speakers"] = names
    meta["speaker_gender"] = {name: genders.get(name, "") for name in names}
    meta["speaker_cast"] = {name: cast[name] for name in names if name in cast}
    job.save()
    log.info(f"спикеры: {_cast_note(settings, names, genders, cast)}; слов с меткой: {n}")
    return tr


def diarize_note(settings: Settings) -> tuple[bool, str]:
    """Строка для doctor: пакет и токен, без значения токена."""
    token = bool(hf_token(settings))
    try:
        import pyannote.audio  # noqa: F401
        pkg = True
    except Exception:
        pkg = False
    model = settings.subs.diarize_model
    if pkg and token:
        return True, f"pyannote, {model}"
    if not pkg:
        return False, "нет pyannote.audio — субтитры одним стилем, нужен HF_TOKEN"
    return False, "нет HF_TOKEN — субтитры одним стилем"
