"""Субтитры для вертикали: короткие реплики, 1–2 строки, ASS с подсветкой слова."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from konveyer.timeline import Interval, map_range
from konveyer.transcript import Word

WHITE = "&H00FFFFFF"
YELLOW_TAG = r"{\1c&H00FFFF&}"   # ASS: &HBBGGRR, жёлтый = R+G
WHITE_TAG = r"{\1c&HFFFFFF&}"


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)   # пусто — без караоке (например, перевод)
    speaker: str = ""                                  # S1, S2… пусто — цвет по умолчанию


def _speaker(word: Word) -> str:
    return str(getattr(word, "speaker", "") or "")


def make_cues(words: list[Word], max_chars: int = 38, max_words: int = 8,
              max_dur: float = 3.2, gap: float = 0.7) -> list[Cue]:
    """Слова -> реплики, которые читаются за пару секунд. Смена спикера рвёт реплику."""
    cues: list[Cue] = []
    cur: list[Word] = []

    def flush():
        if cur:
            text = " ".join(w.word.strip() for w in cur).strip()
            if text:
                cues.append(Cue(cur[0].start, cur[-1].end, text, list(cur), _speaker(cur[0])))
            cur.clear()

    for w in words:
        if not w.word.strip():
            continue
        if cur:
            text_len = len(" ".join(x.word.strip() for x in cur + [w]))
            if (_speaker(w) != _speaker(cur[0]) or text_len > max_chars or len(cur) >= max_words
                    or w.end - cur[0].start > max_dur or w.start - cur[-1].end > gap):
                flush()
        cur.append(w)
        if re.search(r"[.!?…]$", w.word.strip()) and len(" ".join(x.word.strip() for x in cur)) >= 12:
            flush()
    flush()
    # растянуть реплику до следующей (до 0.4 с), чтобы текст не мигал
    for a, b in zip(cues, cues[1:]):
        a.end = max(a.end, min(a.end + 0.4, b.start - 0.02))
    if cues:
        cues[-1].end += 0.3
    return cues


def remap_cues(cues: list[Cue], intervals: list[Interval]) -> list[Cue]:
    """Перенести реплики на время после вырезания пауз."""
    out = []
    for c in cues:
        r = map_range(c.start, c.end, intervals)
        if not r:
            continue
        words = []
        for w in c.words:
            wr = map_range(w.start, w.end, intervals)
            if wr:
                words.append(Word(wr[0], wr[1], w.word, _speaker(w)))
        out.append(Cue(r[0], r[1], c.text, words if c.words else [], c.speaker))
    return out


def ass_time(t: float) -> str:
    cs = int(round(max(t, 0.0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def line_groups(tokens: list[str], max_line: int, max_lines: int = 3) -> list[list[int]]:
    """Разбить токены на строки (индексы), стараясь уравнять 2 строки."""
    total = len(" ".join(tokens))
    if total <= max_line or len(tokens) == 1:
        return [list(range(len(tokens)))]
    best, best_cost = None, None
    for i in range(1, len(tokens)):
        a, b = len(" ".join(tokens[:i])), len(" ".join(tokens[i:]))
        cost = max(a, b)
        if best_cost is None or cost < best_cost:
            best, best_cost = i, cost
    if best_cost is not None and best_cost <= max_line:
        return [list(range(best)), list(range(best, len(tokens)))]
    groups, cur, cur_len = [], [], 0
    for i, t in enumerate(tokens):
        add = len(t) + (1 if cur else 0)
        if cur and cur_len + add > max_line and len(groups) < max_lines - 1:
            groups.append(cur)
            cur, cur_len = [], 0
            add = len(t)
        cur.append(i)
        cur_len += add
    groups.append(cur)
    return groups


def render_tokens(tokens: list[str], groups: list[list[int]], highlight: int | None,
                  base: str = "", hi: str = YELLOW_TAG) -> str:
    lines = []
    for g in groups:
        parts = []
        for i in g:
            tok = ass_escape(tokens[i])
            if i == highlight:
                parts.append(f"{hi}{tok}{base or WHITE_TAG}")
            elif base:
                parts.append(f"{base}{tok}")
            else:
                parts.append(tok)
        lines.append(" ".join(parts))
    return r"\N".join(lines)


def make_big_cues(words: list[Word], max_chars: int = 18, max_gap: float = 0.35) -> list[Cue]:
    """По 1–2 слова строго на своих таймингах Whisper, без растяжки до следующей реплики."""
    clean = [w for w in words if w.word.strip()]
    cues: list[Cue] = []
    i = 0
    while i < len(clean):
        take = 1
        if i + 1 < len(clean):
            a = clean[i].word.strip()
            b = clean[i + 1].word.strip()
            gap = clean[i + 1].start - clean[i].end
            if (gap < max_gap and len(a) <= 14 and len(b) <= 14 and len(f"{a} {b}") <= max_chars
                    and not re.search(r"[.!?…]$", a)
                    and _speaker(clean[i]) == _speaker(clean[i + 1])):
                take = 2
        chunk = clean[i:i + take]
        text = " ".join(w.word.strip() for w in chunk)
        cues.append(Cue(chunk[0].start, chunk[-1].end, text, list(chunk), _speaker(chunk[0])))
        i += take
    return cues


def spread_phrase(text: str, start: float, end: float, max_chars: int = 18,
                  speaker: str = "") -> list[Cue]:
    """Перевод или озвученная фраза: 1–2 слова, время делится по длине внутри окна оригинала."""
    words = (text or "").split()
    if not words or end <= start:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(words):
        if (i + 1 < len(words) and len(words[i]) <= 14 and len(words[i + 1]) <= 14
                and len(f"{words[i]} {words[i + 1]}") <= max_chars):
            chunks.append(f"{words[i]} {words[i + 1]}")
            i += 2
        else:
            chunks.append(words[i])
            i += 1
    if len(chunks) == 1:
        return [Cue(start, end, chunks[0], speaker=speaker)]
    dur = max(end - start, 0.05)
    weights = [max(len(c), 1) for c in chunks]
    total = sum(weights)
    t = start
    out: list[Cue] = []
    for c, w in zip(chunks, weights):
        span = dur * w / total
        out.append(Cue(t, min(t + span, end), c, speaker=speaker))
        t += span
    if out:
        out[-1].end = end
    return out


BIG_SIZE = 120


def big_anchor(width: int, height: int, video_y: int = 0, video_h: int = 0,
               size: int = BIG_SIZE) -> tuple[int, int]:
    """Центр нижней половины кадра, но на картинке, если полоса зума её не достаёт."""
    y = int(height * 0.75)
    if video_h > 0:
        bot = video_y + video_h
        y = min(y, bot - max(size // 2, 36))
        y = max(y, video_y + video_h // 2)
    half = size // 2 + 12
    y = min(max(y, half), height - half)
    return width // 2, int(y)


def ass_bgr(color: str) -> str:
    """#RRGGBB → &H00BBGGRR для PrimaryColour и тега \\1c."""
    text = str(color or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        text = "FFFFFF"
    rr, gg, bb = text[0:2], text[2:4], text[4:6]
    return f"&H00{bb}{gg}{rr}".upper()


def _yellowish(ass: str) -> bool:
    digits = ass.replace("&H", "").replace("&", "")
    if len(digits) < 6:
        return False
    bb = int(digits[-6:-4], 16)
    gg = int(digits[-4:-2], 16)
    rr = int(digits[-2:], 16)
    return rr > 180 and gg > 160 and bb < 140


def _speaker_slot(label: str) -> int:
    import re
    found = re.fullmatch(r"S(\d+)", str(label or "").strip())
    if not found:
        return 0 if label else -1
    return max(int(found.group(1)) - 1, 0)


def _speaker_tags(cue: Cue, colors: list[str] | None, fonts: list[str] | None
                  ) -> tuple[str, str, str]:
    """Теги ASS: цвет реплики, цвет подсветки слова, шрифт. Пусто — как раньше, белый."""
    if not cue.speaker or not colors:
        return "", YELLOW_TAG, ""
    slot = _speaker_slot(cue.speaker)
    if slot < 0:
        return "", YELLOW_TAG, ""
    ass = ass_bgr(colors[slot % len(colors)])
    base = "{\\1c" + ass + "&}"
    hi = WHITE_TAG if _yellowish(ass) else YELLOW_TAG
    font = ""
    if fonts:
        name = str(fonts[slot % len(fonts)] or "").strip()
        name = name.replace("\\", "").replace("{", "").replace("}", "")
        if name:
            font = "{\\fn" + name + "}"
    return base, hi, font


def _build_ass_big(cues: list[Cue], width: int, height: int, *, font: str, size: int,
                   anchor: tuple[int, int] | None, colors: list[str] | None = None,
                   fonts: list[str] | None = None) -> str:
    from konveyer.textfit import fit_font_size

    outline = max(8, size // 10)
    cx, cy = anchor or big_anchor(width, height, size=size)
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{font},{size},{WHITE},&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,"
        f"1,{outline},0,5,40,40,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for c in cues:
        if c.end - c.start < 0.02:
            continue
        text = ass_escape(c.text.strip().upper())
        if not text:
            continue
        base, _hi, font_tag = _speaker_tags(c, colors, fonts)
        text = font_tag + base + text
        fs = fit_font_size(c.text.strip().upper(), width - 120, size, 0.62, min_size=56)
        half = fs // 2 + 8
        y = min(max(cy, half), height - half)
        events.append(f"Dialogue: 0,{ass_time(c.start)},{ass_time(c.end)},Cap,,0,0,0,,"
                      f"{{\\an5\\pos({cx},{y})\\fs{fs}\\bord{max(6, fs // 12)}}}{text}")
    return header + "\n".join(events) + ("\n" if events else "")


def build_ass(cues: list[Cue], width: int, height: int, *, font: str = "Arial", size: int = 74,
              align: int = 2, margin_v: int = 400, karaoke: bool = True, upper: bool = False,
              max_line: int = 20, style: str = "normal",
              anchor: tuple[int, int] | None = None,
              colors: list[str] | None = None, fonts: list[str] | None = None,
              outline: int | None = None, max_lines: int = 3, box: str | None = None) -> str:
    """box — цвет подложки (&HAABBGGRR): BorderStyle=3, в libass коробка берёт OutlineColour,
    Outline — её поля; тень у коробки выключена."""
    if style == "big":
        return _build_ass_big(cues, width, height, font=font, size=max(size, BIG_SIZE),
                              anchor=anchor, colors=colors, fonts=fonts)
    outline = outline or max(3, size // 12)
    border, edge, back, shadow = 1, "&H00000000", "&H80000000", 2
    if box:
        border, edge, back, shadow = 3, box, box, 0
        outline = max(6, size // 7)
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{font},{size},{WHITE},&H0000FFFF,{edge},{back},-1,0,0,0,100,100,0,0,"
        f"{border},{outline},{shadow},{align},60,60,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for c in cues:
        if c.end - c.start < 0.05:
            continue
        if karaoke and c.words:
            tokens = [w.word.strip() for w in c.words]
            if upper:
                tokens = [t.upper() for t in tokens]
            groups = line_groups(tokens, max_line, max_lines)
            base, hi, font_tag = _speaker_tags(c, colors, fonts)
            for i, w in enumerate(c.words):
                a = c.start if i == 0 else w.start
                b = c.words[i + 1].start if i + 1 < len(c.words) else c.end
                if b - a < 0.01:
                    continue
                events.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,"
                              f"{font_tag}{render_tokens(tokens, groups, i, base, hi)}")
        else:
            text = c.text.upper() if upper else c.text
            tokens = text.split()
            if not tokens:
                continue
            base, hi, font_tag = _speaker_tags(c, colors, fonts)
            events.append(f"Dialogue: 0,{ass_time(c.start)},{ass_time(c.end)},Cap,,0,0,0,,"
                          f"{font_tag}{render_tokens(tokens, line_groups(tokens, max_line, max_lines), None, base, hi)}")
    return header + "\n".join(events) + ("\n" if events else "")


def caption_ass(cues: list[Cue], settings, width: int, height: int, *, karaoke: bool,
                align: int, margin_v: int, style: str = "normal",
                anchor: tuple[int, int] | None = None, max_line: int = 20,
                size: int | None = None, outline: int | None = None, max_lines: int = 3,
                box: str | None = None) -> str:
    """Общий вход для Shorts и --full: палитра из [subs], если спикеры включены."""
    subs = settings.subs
    colors = list(subs.speaker_colors) if subs.speakers else None
    if colors and len(colors) > 1:
        # главные спикеры по времени речи — свои цвета (слоты 2..N), остальные — первый цвет (белый)
        import dataclasses as _dc
        tot: dict[str, float] = {}
        for c in cues:
            if c.speaker:
                tot[c.speaker] = tot.get(c.speaker, 0.0) + max(c.end - c.start, 0.0)
        top = sorted(tot, key=lambda k: -tot[k])[:len(colors) - 1]
        remap = {lab: f"S{i + 2}" for i, lab in enumerate(top)}
        cues = [_dc.replace(c, speaker=remap.get(c.speaker, "S1")) if c.speaker else c for c in cues]
    fonts = list(subs.speaker_fonts) if subs.speakers and subs.speaker_fonts else None
    r = settings.render
    return build_ass(cues, width, height, font=r.caption_font, size=size or r.caption_size,
                     align=align, margin_v=margin_v, karaoke=karaoke, upper=r.caption_upper,
                     max_line=max_line, style=style, anchor=anchor, colors=colors, fonts=fonts,
                     outline=outline, max_lines=max_lines, box=box)
