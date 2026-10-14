"""Перенос строк и подбор размера шрифта под ширину кадра."""
from __future__ import annotations


def wrap_lines(text: str, max_chars: int, max_lines: int | None = None) -> list[str]:
    """Жадный перенос по словам; слишком длинное слово не рвём."""
    words = (text or "").split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if cur and len(cand) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    if max_lines and len(lines) > max_lines:
        head = lines[: max_lines - 1]
        rest = " ".join(lines[max_lines - 1:])
        lines = head + [rest]
    return lines


def balance_two(text: str, max_chars: int) -> list[str]:
    """1 строка, если влезает; иначе 2 примерно равные строки (красивее, чем жадно).
    Не влезает и в две — всё равно две: шрифт потом уменьшится, а слова не теряются."""
    text = " ".join((text or "").split())
    if len(text) <= max_chars or " " not in text:
        return [text] if text else []
    words = text.split()
    best, best_cost = [text], None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        cost = max(len(a), len(b))
        if best_cost is None or cost < best_cost:
            best, best_cost = [a, b], cost
    return best


def split_text_lines(text: str, max_lines: int, max_chars: int) -> list[str]:
    """Строки заголовка: «первая|вторая» или перенос, не больше max_lines."""
    raw = (text or "").replace("\\n", "\n")
    if "|" in raw or "\n" in raw:
        parts = [p.strip() for p in raw.replace("\n", "|").split("|") if p.strip()]
    else:
        parts = balance_two(raw, max_chars) if max_lines == 2 else wrap_lines(raw, max_chars, max_lines)
    return parts[:max_lines]


def fit_font_size(text: str, max_width: int, max_size: int, ratio: float = 0.62,
                  min_size: int = 28) -> int:
    """Размер, при котором строка влезает в max_width. ratio — средняя ширина символа / размер."""
    size = int(max_width / (max(len(text), 1) * ratio))
    return max(min_size, min(max_size, size))
