"""Сжатие пауз (идея auto-editor): оставить только куски с речью.

Работает по таймингу слов Whisper: паузы длиннее pause секунд вырезаются,
по краям остаётся запас pad, чтобы не съедать начало и конец слов.
Все времена — относительно начала клипа.
"""
from __future__ import annotations

from konveyer.transcript import Word

Interval = tuple[float, float]


def keep_intervals(words: list[Word], duration: float, pause: float = 0.6,
                   pad: float = 0.12) -> list[Interval]:
    if not words or duration <= 0:
        return [(0.0, duration)]
    ws = sorted(words, key=lambda w: w.start)
    out: list[list[float]] = []
    a = 0.0 if ws[0].start <= pause else max(0.0, ws[0].start - pad)
    for cur, nxt in zip(ws, ws[1:]):
        if nxt.start - cur.end > pause:
            out.append([a, cur.end + pad])
            a = nxt.start - pad
    last_end = ws[-1].end
    b = duration if duration - last_end <= pause else min(duration, last_end + pad)
    out.append([a, b])
    merged: list[list[float]] = []
    for x, y in out:
        x, y = max(0.0, x), min(duration, y)
        if y - x < 0.05:
            continue
        if merged and x <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], y)
        else:
            merged.append([x, y])
    return [(round(x, 3), round(y, 3)) for x, y in merged] or [(0.0, duration)]


def kept_duration(intervals: list[Interval]) -> float:
    return sum(b - a for a, b in intervals)


def map_time(t: float, intervals: list[Interval]) -> float | None:
    """Время в исходном клипе -> время после вырезания пауз. None — момент вырезан."""
    acc = 0.0
    for a, b in intervals:
        if t < a:
            return None
        if t <= b:
            return acc + (t - a)
        acc += b - a
    return None


def map_range(start: float, end: float, intervals: list[Interval]) -> tuple[float, float] | None:
    """Отрезок -> отрезок после вырезания; концы, попавшие в паузу, прижимаются к речи."""
    acc = 0.0
    new_start = new_end = None
    for a, b in intervals:
        if new_start is None and end >= a and start <= b:
            new_start = acc + (max(start, a) - a)
        if new_start is not None and start <= b:
            if end <= b:
                new_end = acc + (max(end, a) - a)
                break
            new_end = acc + (b - a)
        acc += b - a
    if new_start is None or new_end is None or new_end - new_start < 0.05:
        return None
    return round(new_start, 3), round(new_end, 3)


def select_expr(intervals: list[Interval]) -> str:
    """Выражение для фильтров select/aselect: between(t,a,b)+between(t,c,d)..."""
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in intervals)


def append_source_tail(intervals: list[Interval] | None, clip_dur: float,
                       take: float) -> list[Interval] | None:
    """Добавить к вырезанным паузам хвост исходника сразу после конца клипа.

    Паузы внутри клипа остаются вырезанными. Продолжение берётся с clip_dur,
    а не с последнего оставленного кадра: клип кончился там, исходник идёт дальше.
    """
    if not intervals or take <= 0.02 or clip_dur < 0:
        return intervals
    start = float(clip_dur)
    if intervals[-1][1] >= clip_dur - 0.001:
        start = max(float(intervals[-1][1]), start)
    end = float(clip_dur) + float(take)
    if end - start < 0.02:
        return intervals
    return [*intervals, (round(start, 3), round(end, 3))]
