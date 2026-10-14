"""Проверка текстов перед Telegram.

Сначала местные признаки (повторы, непонятные символы, русский язык заголовков и поста).
Если они чистые и есть ключ DeepSeek — один дешёвый запрос: осмысленно ли, нет ли выдумок.
Ключа нет или модель не ответила — остаётся местный результат, без падения.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from konveyer.config import Settings
from konveyer.jobs import Job
from konveyer.llm import LLM, chat_json, set_stage
from konveyer.log import Log
from konveyer.moments import meta_block
from konveyer.speech import NO_SPEECH_MATERIAL, char_run_ratio, top_word_ratio
from konveyer.transcript import Transcript
from konveyer.util import read_json

# Доли ниже — местный отказ. Совпадают по духу с [speech], но смотрят уже готовые тексты.
MAX_CHAR_RUN = 0.30
MAX_TOP_WORD = 0.60
MIN_WORDS_REPEAT = 4
MAX_WEIRD = 0.12
MIN_CYRILLIC = 0.45
MIN_LETTERS_LANG = 30

_CYR = re.compile(r"[\u0400-\u04FF]")
_ASS_TAG = re.compile(r"\{[^}]*\}")

QUALITY_SYSTEM = (
    "Ты проверяешь тексты перед публикацией ролика. Основания — блоки «Материалы», "
    "«Речь (английский оригинал)» и «Русский перевод»: это одна и та же речь целиком. "
    "Выпиши каждое конкретное утверждение поста и заголовков (факт, число, имя, событие) и для "
    "каждого процитируй дословно место-основание из английской речи или русского перевода. "
    "Выдумка — только то, чему нет основания ни в одном из них; пересказ своими словами — не выдумка. "
    "Ответь JSON {\"claims\": [{\"claim\": \"...\", \"quote\": \"дословная цитата или пусто\"}], "
    "\"ok\": true или false, \"reason\": \"одна короткая фраза по-русски\"}. "
    "ok=false, если есть утверждение без цитаты, или тексты — мусор, одни повторы, не на том языке. "
    "Если речи нет и пост честно не выдумывает сюжет — ok=true. Не переписывай тексты. "
    "Заголовок темы у таймкода пиши как claim «Таймкод M:SS — заголовок» и тоже ищи ему цитату. "
    "Служебные строки (длина ролика, имя канала, слова «таймкоды», «темы»), эмодзи и хештеги — "
    "не утверждения."
)


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: str
    source: str          # local | deepseek


def _letters(text: str) -> str:
    return "".join(c for c in text if c.isalpha())


def cyrillic_ratio(text: str) -> float:
    letters = _letters(text)
    if not letters:
        return 1.0
    return len(_CYR.findall(letters)) / len(letters)


def weird_char_ratio(text: str) -> float:
    chars = [c for c in (text or "") if not c.isspace()]
    if not chars:
        return 0.0
    weird = 0
    for c in chars:
        if c.isalnum() or c in ".,!?:;…—–-\"'«»()[]/%+№#@&*_":
            continue
        if ord(c) >= 0x1F300:          # эмодзи
            continue
        weird += 1
    return weird / len(chars)


def ass_plain(path: Path) -> str:
    """Текст реплик из ASS. Караоке повторяет одну и ту же строку — оставляем уникальные подряд."""
    lines: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in raw.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        text = _ASS_TAG.sub("", parts[9]).replace("\\N", " ").replace("\\n", " ")
        text = " ".join(text.split())
        if text and (not lines or lines[-1] != text):
            lines.append(text)
    return "\n".join(lines)


def collect_texts(job: Job) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    titles = []
    captions = []
    for clip in job.data.get("clips") or []:
        bits = " ".join(str(clip.get(k) or "").replace("|", " ") for k in ("kicker", "title", "bottom"))
        bits = " ".join(bits.split())
        if bits:
            titles.append(bits)
        rel = str(clip.get("file") or "")
        ass = (job.dir / rel).parent / "captions.ass" if rel else job.clip_dir(int(clip["id"])) / "captions.ass"
        if not ass.is_file():
            ass = job.clip_dir(int(clip["id"])) / "captions.ass"
        if ass.is_file():
            plain = ass_plain(ass)
            if plain:
                captions.append(plain)
    if titles:
        items.append(("заголовки", "\n".join(titles)))
    if captions:
        items.append(("субтитры", "\n".join(captions)))
    if job.post_md.is_file():
        from konveyer.postfmt import strip_decor
        # эмодзи и хештеги — оформление, а не утверждения: проверке их не показываем
        post = strip_decor(job.post_md.read_text(encoding="utf-8")).strip()
        if post:
            items.append(("пост", post))
    from konveyer.clipdesc import read_description
    from konveyer.postfmt import strip_decor
    descs = [f"клип {c['id']}: " + strip_decor(read_description(job, c)).strip()
             for c in job.data.get("clips") or [] if read_description(job, c)]
    if descs:
        items.append(("описания Shorts", "\n\n".join(descs)))
    return items


def local_issues(items: list[tuple[str, str]]) -> list[str]:
    issues: list[str] = []
    for name, text in items:
        if len([c for c in text if not c.isspace()]) >= 6 and char_run_ratio(text) >= MAX_CHAR_RUN:
            issues.append(f"{name}: повтор символов")
        ratio, n = top_word_ratio(text)
        if n >= MIN_WORDS_REPEAT and ratio >= MAX_TOP_WORD:
            issues.append(f"{name}: повтор слов")
        if len(text) >= 8 and weird_char_ratio(text) >= MAX_WEIRD:
            issues.append(f"{name}: много непонятных символов")
        if name in ("пост", "заголовки") and len(_letters(text)) >= MIN_LETTERS_LANG:
            if cyrillic_ratio(text) < MIN_CYRILLIC:
                issues.append(f"{name}: не на русском")
        if name == "субтитры" and len(_letters(text)) >= MIN_LETTERS_LANG:
            # субтитры после перевода и русская речь должны быть кириллицей;
            # оригинал на другом языке сюда попадает тоже — смотрим только явный мусор выше,
            # а язык субтитров отдаём DeepSeek, если он есть. Местно режем только повторы.
            pass
    return issues


def _clip_digest(items: list[tuple[str, str]], limit: int = 1800) -> str:
    parts = []
    for name, text in items:
        parts.append(f"{name}:\n{text[:limit]}")
    return "\n\n".join(parts)


def _speech_block(job: Job, limit: int = 1500) -> str:
    tr = Transcript.from_dict(read_json(job.transcript_file, None))
    if tr.empty or not tr.reliable:
        return NO_SPEECH_MATERIAL
    return tr.text()[:limit]


def _russian_block(job: Job, items: list[tuple[str, str]], limit: int) -> str:
    """Русский перевод речи целиком: фразы озвучки целого видео или субтитры."""
    voice = read_json(job.dir / "full" / "voice.json", {})
    lines = [str(p.get("text") or "") for p in (voice.get("phrases") or [])] \
        if isinstance(voice, dict) else []
    text = " ".join(" ".join(lines).split())
    if not text:
        text = "\n".join(body for name, body in items if name == "субтитры")
    return text[:limit] or "-"


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[0-9a-zа-яё]+", (text or "").casefold().replace("ё", "е"))


def quote_found(quote: str, *sources: str) -> bool:
    """Цитата правда есть в речи: не меньше 60% её слов встречаются в источнике."""
    words = _norm_words(quote)
    if not words:
        return False
    for src in sources:
        bag = set(_norm_words(src))
        if bag and sum(w in bag for w in words) >= 0.6 * len(words):
            return True
    return False


HEADING_RE = re.compile(r"^\s*(?:таймкод\s*)?\d{1,2}:\d{2}(?::\d{2})?\s*[—–-]\s*", re.I)


def _key_tokens(text: str) -> list[str]:
    """Конкретика заголовка: числа и имена (слово с заглавной не в начале)."""
    words = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text or "")
    keys = []
    for i, w in enumerate(words):
        if w.isdigit() or (i > 0 and any(ch.isupper() for ch in w)):
            keys.append(w.casefold().replace("ё", "е"))
    return keys


def heading_ok(claim: str, *sources: str) -> bool:
    """Заголовок темы — пересказ: выдумка, только если в нём число или имя, которого нет в речи."""
    m = HEADING_RE.match(claim or "")
    if not m:
        return False
    keys = _key_tokens(claim[m.end():])
    bag = set()
    for src in sources:
        bag.update(_norm_words(src))
    return all(k in bag for k in keys)


def unsupported_claims(data: dict, *sources: str) -> list[str]:
    """Утверждения без цитаты или с цитатой, которой нет ни в речи, ни в переводе, ни в описании."""
    out = []
    for item in data.get("claims") or []:
        if not isinstance(item, dict):
            continue
        claim = " ".join(str(item.get("claim") or "").split())
        from konveyer.postfmt import strip_decor
        if not strip_decor(claim).strip(" #.,:—-"):
            continue          # хештег или эмодзи — оформление, не утверждение
        if not claim or quote_found(str(item.get("quote") or ""), *sources):
            continue
        if heading_ok(claim, *sources):
            continue
        out.append(claim[:120])
    return out


def assess_publish(job: Job, settings: Settings, log: Log, llm: LLM | None) -> QualityResult:
    items = collect_texts(job)
    issues = local_issues(items)
    from konveyer.mediaq import media_issues
    try:
        issues.extend(media_issues(job, settings, log, llm))
    except Exception as e:
        log.info(f"проверка клипов не удалась ({str(e)[:200]})")
    if issues:
        reason = "; ".join(issues)
        log.info(f"проверка перед отправкой не пройдена: {reason}")
        return QualityResult(False, reason, "local")
    if llm is None or not items:
        log.info("проверка перед отправкой: местные признаки в порядке"
                 + ("" if llm else ", DeepSeek нет — выдумки не проверял"))
        return QualityResult(True, "", "local")

    meta = job.data.get("meta") or {}
    limit = max(int(settings.llm.max_transcript_chars), 1500)
    english = _speech_block(job, limit)
    russian = _russian_block(job, items, limit)
    publish = [(n, b) for n, b in items if n != "субтитры"] or items
    material = meta_block(meta) or ""
    user = (f"Материалы:\n{material or '(нет названия и описания)'}\n\n"
            f"Речь (английский оригинал):\n{english}\n\n"
            f"Русский перевод:\n{russian}\n\n"
            f"Тексты на публикацию:\n{_clip_digest(publish, 6000)}")
    set_stage(llm, "quality")
    try:
        data = chat_json(llm, QUALITY_SYSTEM, user, temperature=0)
    except Exception as e:
        log.warn(f"проверка DeepSeek не удалась ({e}) — оставляю местную, она пройдена")
        return QualityResult(True, "", "local")
    if isinstance(data, dict) and data.get("claims"):
        bad = unsupported_claims(data, english, russian, material)
        if bad:
            reason = "без основания в речи: " + "; ".join(bad[:5])
            log.info(f"проверка перед отправкой не пройдена (DeepSeek): {reason}")
            return QualityResult(False, reason, "deepseek")
        log.info(f"проверка перед отправкой: DeepSeek — у всех {len(data['claims'])} "
                 f"утверждений есть цитата из речи")
        return QualityResult(True, "", "deepseek")
    if isinstance(data, dict) and isinstance(data.get("ok"), bool):
        if data["ok"]:
            log.info("проверка перед отправкой: DeepSeek — тексты в порядке")
            return QualityResult(True, "", "deepseek")
        reason = str(data.get("reason") or "DeepSeek: тексты лучше не публиковать").strip()
        log.info(f"проверка перед отправкой не пройдена (DeepSeek): {reason}")
        return QualityResult(False, reason, "deepseek")
    log.warn("проверка DeepSeek не дала понятного ответа — оставляю местную, она пройдена")
    return QualityResult(True, "", "local")


def review_notice(job: Job, reason: str) -> str:
    title = (job.data.get("meta") or {}).get("title") or job.id
    text = f"На проверку: {title}. {reason} Клипы не отправлены."
    text = " ".join(text.split())
    if len(text) > 400:
        text = text[:397] + "…"
    return text
