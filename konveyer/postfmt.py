"""Оформление постов: эмодзи в начале каждого абзаца и строка хештегов в конце.

Промпт просит модель так писать, а код гарантирует результат: абзацы через пустую строку,
у абзаца без эмодзи — подходящий по смыслу, в конце 6–10 хештегов, постоянные — всегда.
У таймкодов эмодзи нет; перед «Таймкоды:» — ⏱.
"""
from __future__ import annotations

import re
import unicodedata

CLOCK = "⏱"
MIN_TAGS = 6
DEFAULT_TAGS = "#ИИ #нейросети #искусственныйинтеллект #интервью #технологии"
# Платформы в тексте не называем: ни «X», ни «в X», ни «Instagram». Источник — только ник.
SOURCE_RULE = (
    "Платформы не называй: не пиши «X», «в X», «из X», «Twitter», «Instagram», «Инстаграм» — "
    "ни в тексте, ни в хештегах. Источник указывается только ником автора (например "
    "«Источник: @AlertaNews24»), его добавит код — сам строку «Источник» не пиши."
)
_PLATFORM_TAGS = {"x", "х", "twitter", "твиттер", "instagram", "инстаграм", "insta", "инста"}
_PLAT_WORD = r"(?:Twitter|Твиттер\w*|Instagram|Инстаграм\w*|Insta\b)"
_PLAT = rf"(?:{_PLAT_WORD}|(?<![\w@#.-])[XХ](?![\w-]))"
_PREP = r"(?i:\b(?:в|во|из|на|с|со|через)\s+)"
_VENUE = r"(?i:(?:соц\.?\s*сет\w*|сет[иья]\w*|платформ\w*|приложени\w*)\s+)"
_PLATFORM_RES = [
    re.compile(rf"(?i:\s+(?:и|или)\s+)(?:{_PREP}{_VENUE}?)?{_PLAT_WORD}"),     # «… и Instagram»
    re.compile(rf"\s*\(\s*(?:бывш(?:ий|\.)\s+)?{_PLAT}\s*\)"),          # (X), (бывший Twitter)
    re.compile(rf"\s*,?\s*(?:{_PREP}{_VENUE}?|{_VENUE}){_PLAT}(?:\s*\(\s*{_PLAT}\s*\))?(?![\w])"),
    re.compile(rf"\s*,?\s*{_PREP}?{_VENUE}?{_PLAT_WORD}"),
]
_HANDLE_RE = re.compile(r"^@?([A-Za-z0-9_.]{2,30})$")
_URL_HANDLE = [
    re.compile(r"(?:x|twitter)\.com/([A-Za-z0-9_]{2,30})/status", re.I),
    re.compile(r"instagram\.com/(?!p/|reels?/|tv/|stories/)([A-Za-z0-9_.]{2,30})", re.I),
    re.compile(r"tiktok\.com/@([A-Za-z0-9_.]{2,30})", re.I),
    re.compile(r"youtube\.com/@([A-Za-z0-9_.-]{2,30})", re.I),
]
_SOURCE_LINE = re.compile(r"^\W*(?:Источник|Автор|Source)\s*:\s*(.*)$", re.I)
# подсказка модели (вставляется в промпт поста)
PROMPT_RULES = (
    "Оформление: каждый абзац начинай с одного подходящего по смыслу эмодзи "
    "(например 🤖 ⚠️ 🐝 🛑 💡 📌), абзацы разделяй пустой строкой. Хештеги в текст поста "
    "не пиши — дай их отдельно в поле \"hashtags\": 6–10 штук, обязательно {base} и "
    "тематические по ролику: компании и имена, слитно, без пробелов (например #Anthropic "
    "#OpenAI #Амодеи). Только то, что есть в материалах. " + SOURCE_RULE
)
# эмодзи по смыслу абзаца, если модель его не поставила; первый абзац — 🤖
_PICK = [
    (r"\?\s*$", "❓"),
    (r"опас|угроз|риск|тревож|пугает|страш|атак|взлом", "⚠️"),
    (r"замедл|останов|стоп|запрет|пауз", "🛑"),
    (r"рой|роя|агент", "🐝"),
    (r"итог|вывод|значит|главное", "💡"),
    (r"видео|ролик", "🎬"),
]
_TAG_RE = re.compile(r"#[\w]+", re.UNICODE)



def is_emoji(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch[0])
    if cp >= 0x1F000 or 0x2190 <= cp <= 0x2BFF or cp in (0x00A9, 0x00AE, 0x203C, 0x2049):
        return unicodedata.category(ch[0]) in ("So", "Sm", "Sk", "Cn") or cp >= 0x1F000
    return False


def starts_with_emoji(text: str) -> bool:
    s = (text or "").lstrip()
    return bool(s) and is_emoji(s[0])


def strip_decor(text: str) -> str:
    """Текст без хештегов и эмодзи — для проверки на выдумки и местных признаков."""
    out = _TAG_RE.sub(" ", text or "")
    out = "".join(c for c in out if not (is_emoji(c) or ord(c) in (0xFE0F, 0x200D)))
    return "\n".join(" ".join(line.split()) for line in out.splitlines())


def clean_tag(raw: str) -> str:
    word = re.sub(r"[^\w]", "", str(raw or "").replace("#", ""), flags=re.UNICODE)
    if word.casefold() in _PLATFORM_TAGS:
        return ""                      # #X, #Instagram — платформы не называем
    return f"#{word}" if word and not word.isdigit() else ""


def _as_handle(raw) -> str:
    m = _HANDLE_RE.match(str(raw or "").strip())
    return m.group(1) if m and not m.group(1).replace(".", "").isdigit() else ""


def source_handle(meta: dict | None) -> str:
    """Ник автора без платформы: @AlertaNews24. Из meta.handle, uploader_id или ссылки."""
    meta = meta or {}
    for key in ("handle", "uploader_id"):
        h = _as_handle(meta.get(key))
        if h:
            return f"@{h}"
    url = str(meta.get("url") or "")
    for rx in _URL_HANDLE:
        m = rx.search(url)
        if m and _as_handle(m.group(1)):
            return f"@{m.group(1)}"
    h = _as_handle(meta.get("uploader"))
    return f"@{h}" if h else ""


def source_line(meta: dict | None) -> str:
    h = source_handle(meta)
    return f"Источник: {h}" if h else ""


def scrub_platforms(text: str) -> str:
    """Убрать из текста «в X», «Instagram», «(бывший Twitter)», #X/#Instagram."""
    if not text:
        return text or ""
    out_lines = []
    for line in str(text).splitlines():
        line = _TAG_RE.sub(lambda m: m.group(0) if clean_tag(m.group(0)) else "", line)
        for rx in _PLATFORM_RES:
            line = rx.sub("", line)
        line = re.sub(r"[ \t]{2,}", " ", line)
        line = re.sub(r"\s+([,.;:!?»)])", r"\1", line)
        line = re.sub(r"\(\s*\)", "", line)
        line = re.sub(r"«\s*»", "", line)
        line = re.sub(r"^(\W*?)[,;:]\s*", r"\1", line) if line.strip()[:1] in ",;:" else line
        out_lines.append(line.rstrip())
    return "\n".join(out_lines)


def fix_source(text: str, meta: dict | None = None, add: bool = False) -> str:
    """Чистка платформ + строка «Источник: @ник» (только ник). add — дописать, если её нет."""
    text = scrub_platforms(text)
    want = source_line(meta)
    lines, found = [], False
    for line in text.splitlines():
        m = _SOURCE_LINE.match(line)
        if m and line.lstrip()[:1] != "#":
            rest = m.group(1)
            nick = re.search(r"@([A-Za-z0-9_.]{2,30})", rest)
            if found:
                continue
            if nick:
                line, found = f"Источник: @{nick.group(1)}", True
            elif want:
                line, found = want, True
            elif rest.strip():
                line, found = f"Источник: {rest.strip()}", True
            else:
                continue
        lines.append(line)
    out = "\n".join(lines).strip()
    if add and want and not found:
        out = (out + "\n\n" if out else "") + want
    return out


def split_tags(text: str) -> tuple[str, list[str]]:
    """Отрезать хвостовые строки, где только хештеги. Вернуть текст и эти хештеги."""
    lines = (text or "").rstrip().splitlines()
    tags: list[str] = []
    while lines:
        line = lines[-1].strip()
        if not line:
            lines.pop()
            continue
        if _TAG_RE.sub("", line).strip(" ,.;·|") == "" and _TAG_RE.search(line):
            tags = _TAG_RE.findall(line) + tags
            lines.pop()
            continue
        break
    return "\n".join(lines).rstrip(), tags


def meta_tags(meta: dict) -> list[str]:
    """Тематика из материалов, если модель хештегов не дала: канал/автор одним словом."""
    who = str((meta or {}).get("uploader") or "").strip()
    tag = clean_tag(who) if who and len(who.split()) <= 2 else ""
    return [tag] if tag and len(tag) <= 25 else []


def merge_tags(base: str, *groups, limit: int = 10) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    fixed = [t for t in (clean_tag(x) for x in str(base or "").split()) if t]
    for tag in fixed + [clean_tag(t) for g in groups for t in (g or [])]:
        key = tag.casefold()
        if tag and key not in seen:
            seen.add(key)
            out.append(tag)
    return out[:max(limit, len(fixed))]


def pick_emoji(paragraph: str, first: bool) -> str:
    if first:
        return "🤖"
    low = paragraph.lower()
    for pattern, emo in _PICK:
        if re.search(pattern, low):
            return emo
    return "📌"


def paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", (text or "").strip())
    return [b.strip() for b in blocks if b.strip()]


def decorate_body(text: str, emoji: bool = True) -> str:
    """Абзацы через пустую строку; у каждого в начале эмодзи."""
    out = []
    for i, block in enumerate(paragraphs(text)):
        if emoji and not starts_with_emoji(block) and not _TAG_RE.fullmatch(block.split()[0]):
            block = f"{pick_emoji(block, i == 0)} {block}"
        out.append(block)
    return "\n\n".join(out)


def tags_for(post_cfg, model_tags=(), meta: dict | None = None, body_tags=()) -> list[str]:
    if not getattr(post_cfg, "hashtags_on", True):
        return []
    tags = merge_tags(post_cfg.hashtags, model_tags, body_tags, limit=10)
    if len(tags) < MIN_TAGS:
        tags = merge_tags(post_cfg.hashtags, tags, meta_tags(meta or {}), limit=10)
    return tags


def decorate_post(text: str, post_cfg, model_tags=(), meta: dict | None = None) -> tuple[str, list[str]]:
    """Тело поста с эмодзи и список хештегов (хештеги, что модель вписала в текст, — в список)."""
    body, body_tags = split_tags(text)
    body = decorate_body(body, getattr(post_cfg, "emoji", True))
    return body, tags_for(post_cfg, model_tags, meta, body_tags)


def with_tags(body: str, tags: list[str]) -> str:
    return body.rstrip() + ("\n\n" + " ".join(tags) if tags else "")


def model_tags(data) -> list[str]:
    raw = data.get("hashtags") if isinstance(data, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    # модель иногда кладёт все теги одной строкой: "#Anthropic #OpenAI" — режем на слова
    pieces = [p for x in (raw or []) for p in re.split(r"[\s,;]+|(?=#)", str(x)) if p]
    return [t for t in (clean_tag(x) for x in pieces) if t]
