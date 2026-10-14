"""Текст для синтеза: цифры словами, латиница по-русски, одиночные буквы по названию.

«ИИ» и другие аббревиатуры из двух одинаковых заглавных читаются раздельно: «и-и», «эс-эс».

Субтитры этим не трогаем — только строку, которую отдаём Edge, Silero или XTTS.
num2words (lang=ru) умеет падежи: чистый Python, без бинарных модулей, на 3.14 ставится.
Если пакета нет, остаётся короткий запасной ряд для целых — конвейер не падает.
Одиночная «З» читается «зэ»: иначе многоязычный Edge принимает её за цифру 3.
"""
from __future__ import annotations

import re

# Частые слова. Ключ как в речи, значение — как читать. Конфиг дополняет и перекрывает.
DEFAULT_REPLACEMENTS = {
    "GPT": "джи-пи-ти",
    "AI": "эй-ай",
    "ChatGPT": "чат джи-пи-ти",
    "API": "эй-пи-ай",
    "CPU": "си-пи-ю",
    "GPU": "джи-пи-ю",
    "IT": "ай-ти",
    "PC": "пи-си",
    "OK": "окей",
    "USB": "ю-эс-би",
    "SMS": "эс-эм-эс",
    "WiFi": "вай-фай",
    "YouTube": "Ютуб",
    "iPhone": "айфон",
    "RTX": "эр-ти-экс",
    # Компании, продукты и люди из новостей про ИИ. Только для голоса: субтитры и пост — латиницей.
    "Hugging Face": "Хаггинг Фейс", "HuggingFace": "Хаггинг Фейс",
    "OpenAI": "Опен Эй Ай", "Anthropic": "Энтропик", "Claude": "Клод",
    "Gemini": "Джеминай", "Google": "Гугл", "DeepMind": "ДипМайнд", "DeepSeek": "ДипСик",
    "Meta": "Мета", "Microsoft": "Майкрософт", "Nvidia": "Энвидиа", "Tesla": "Тесла",
    "xAI": "Икс Эй Ай", "Grok": "Грок", "Llama": "Лама", "Mistral": "Мистраль",
    "Perplexity": "Перплексити", "Midjourney": "Миджорни", "Sora": "Сора",
    "Copilot": "Копайлот", "GitHub": "ГитХаб", "Apple": "Эппл", "Amazon": "Амазон",
    "AGI": "эй-джи-ай", "LLM": "эл-эл-эм", "CNN": "си-эн-эн", "CEO": "гендиректор",
    "AMD": "эй-эм-ди", "Intel": "Интел", "Samsung": "Самсунг", "Huawei": "Хуавей",
    "Baidu": "Байду", "Alibaba": "Алибаба", "Qwen": "Квен", "Runway": "Ранвей",
    "Stable Diffusion": "Стейбл Диффьюжн", "Facebook": "Фейсбук", "Instagram": "Инстаграм",
    "WhatsApp": "Ватсап", "Telegram": "Телеграм", "Twitter": "Твиттер", "TikTok": "Тикток",
    "Netflix": "Нетфликс", "Android": "Андроид", "iOS": "ай-оу-эс", "Bing": "Бинг",
    "Skynet": "Скайнет", "Terminator": "Терминатор", "Al Qaeda": "Аль-Каида",
    "Lockheed Martin": "Локхид Мартин", "Northrop Grumman": "Нортроп Грумман",
    "Northrop": "Нортроп", "Grumman": "Грумман", "Lockheed": "Локхид",
    "Sam Altman": "Сэм Альтман", "Altman": "Альтман",
    "Dario Amodei": "Дарио Амодей", "Amodei": "Амодей", "Dario": "Дарио",
    "Elon Musk": "Илон Маск", "Musk": "Маск", "Jensen Huang": "Дженсен Хуанг",
    "Tristan Harris": "Тристан Харрис", "Harris": "Харрис",
    "Anderson Cooper": "Эндерсон Купер", "Anderson": "Эндерсон", "Cooper": "Купер",
    "Mark Zuckerberg": "Марк Цукерберг", "Zuckerberg": "Цукерберг",
    "Sundar Pichai": "Сундар Пичаи", "Satya Nadella": "Сатья Наделла",
    "Demis Hassabis": "Демис Хассабис", "Ilya Sutskever": "Илья Суцкевер",
    "Geoffrey Hinton": "Джеффри Хинтон", "Yann LeCun": "Ян Лекун",
    "Bill Gates": "Билл Гейтс", "Gates": "Гейтс", "Trump": "Трамп", "Dean Ball": "Дин Болл",
}

# Буква латиницы → как её говорят по-русски. «GPT» → «джи-пи-ти», «AI» → «эй-ай».
_LETTERS = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф",
    "g": "джи", "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эль",
    "m": "эм", "n": "эн", "o": "о", "p": "пи", "q": "кью", "r": "эр",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл-ю", "x": "икс",
    "y": "уай", "z": "зет",
}

_DIGRAPHS = (
    ("shch", "щ"), ("sch", "щ"), ("yo", "ё"), ("zh", "ж"), ("kh", "х"),
    ("ts", "ц"), ("ch", "ч"), ("sh", "ш"), ("yu", "ю"), ("ya", "я"),
    ("ye", "е"), ("ph", "ф"), ("th", "т"),
)
_CHAR = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
}

# Предлог перед числом → падеж num2words (n g d a i p).
_PREP_CASE = {
    "без": "g", "до": "g", "из": "g", "от": "g", "у": "g", "для": "g",
    "около": "g", "после": "g", "с": "g", "со": "g",
    "к": "d", "ко": "d", "по": "d",
    "над": "i", "под": "i", "перед": "i", "между": "i",
    "в": "p", "во": "p", "на": "p", "о": "p", "об": "p", "при": "p",
}

_PREPS = "|".join(sorted(_PREP_CASE, key=len, reverse=True))
_YEAR = re.compile(
    rf"(?i)(?P<pre>\b(?:{_PREPS})\s+)?(?P<num>\d{{3,4}})(?P<post>\s+год(?:у|а)?\b)"
)
_PERCENT = re.compile(rf"(?i)(?P<pre>\b(?:{_PREPS})\s+)?(?P<num>\d+(?:[.,]\d+)?)\s*%")
_NUMBER = re.compile(
    rf"(?i)(?P<pre>\b(?:{_PREPS})\s+)?(?P<num>\d+(?:[.,]\d+)?)(?![\d%])"
)
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# Школьные названия. «З» → «зэ», не «три». Предлоги из одной буквы не трогаем:
# «в городе» нельзя читать «вэ городе». «о» похожа на ноль, но это предлог,
# а название буквы — то же «о».
_CYR_NAMES = {
    "а": "а", "б": "бэ", "в": "вэ", "г": "гэ", "д": "дэ", "е": "е", "ё": "ё",
    "ж": "жэ", "з": "зэ", "и": "и", "й": "и краткое", "к": "ка", "л": "эль",
    "м": "эм", "н": "эн", "о": "о", "п": "пэ", "р": "эр", "с": "эс", "т": "тэ",
    "у": "у", "ф": "эф", "х": "ха", "ц": "цэ", "ч": "че", "ш": "ша", "щ": "ща",
    "ъ": "твёрдый знак", "ы": "ы", "ь": "мягкий знак", "э": "э", "ю": "ю", "я": "я",
}
_KEEP_ONE = frozenset("авиксоуя")
_CYR_ONE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])([А-Яа-яЁё])(?![0-9A-Za-zА-Яа-яЁё])"
)
# «ИИ» Edge склеивает в один звук «и». Две одинаковые заглавные — по названию через дефис.
_DOUBLED_ABBR = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])([А-ЯЁ])\1(?![0-9A-Za-zА-Яа-яЁё])"
)
# Метка без цифр и латиницы: иначе следующий проход прочитает её как число или слово.
_MARK_ALPH = "абвгдежзийклмнопрстуфхцчшщыэюя"
_HOLD = re.compile("\ue000([" + _MARK_ALPH + "]+)\ue001")


def _mark(i: int) -> str:
    n = i + 1
    chars = []
    while n:
        n, r = divmod(n - 1, len(_MARK_ALPH))
        chars.append(_MARK_ALPH[r])
    return "\ue000" + "".join(reversed(chars)) + "\ue001"


def _unmark(token: str) -> int:
    n = 0
    for ch in token:
        n = n * len(_MARK_ALPH) + _MARK_ALPH.index(ch) + 1
    return n - 1


def _lookup(token: str, table: dict) -> str:
    if token in table and table[token]:
        return str(table[token])
    low = token.casefold()
    for key, val in table.items():
        if key.casefold() == low and val:
            return str(val)
    return ""


def _n2w(number, *, case: str = "n", ordinal: bool = False, gender: str = "m") -> str:
    kwargs = {"lang": "ru", "case": case, "gender": gender, "animate": False}
    if ordinal:
        kwargs["to"] = "ordinal"
    try:
        from num2words import num2words
        return str(num2words(number, **kwargs))
    except Exception:
        return _fallback(number, case)


def _fallback(number, case: str) -> str:
    """Если num2words не импортировался: хотя бы именительный для небольших целых."""
    small = {
        0: "ноль", 1: "один", 2: "два", 3: "три", 4: "четыре", 5: "пять",
        6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять",
        11: "одиннадцать", 12: "двенадцать", 13: "тринадцать", 14: "четырнадцать",
        15: "пятнадцать", 16: "шестнадцать", 17: "семнадцать", 18: "восемнадцать",
        19: "девятнадцать", 20: "двадцать", 30: "тридцать", 40: "сорок",
        50: "пятьдесят", 60: "шестьдесят", 70: "семьдесят", 80: "восемьдесят",
        90: "девяносто", 100: "сто", 200: "двести", 1000: "тысяча",
    }
    try:
        n = int(number)
    except (TypeError, ValueError):
        return str(number)
    if case == "n" and n in small:
        return small[n]
    return str(n)


def _parse_num(raw: str):
    text = raw.replace(",", ".")
    if "." in text:
        return float(text)
    return int(text)


def _year_case(pre: str, post: str) -> str:
    noun = post.strip().casefold()
    prep = (pre or "").strip().casefold()
    if noun == "году":
        if prep in ("к", "ко"):
            return "d"
        return "p"
    if noun == "года":
        return "g"
    return "n"


def _spell_num(raw: str, case: str = "n", ordinal: bool = False) -> str:
    try:
        number = _parse_num(raw)
    except ValueError:
        return raw
    return _n2w(number, case=case, ordinal=ordinal)


def _is_abbrev(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters) and len(letters) <= 8


def _spell_letters(token: str) -> str:
    return "-".join(_LETTERS[c.lower()] for c in token if c.isalpha())


def _translit(token: str) -> str:
    s = token.casefold()
    out = []
    i = 0
    while i < len(s):
        chunk = s[i:i + 4]
        hit = ""
        for dig, ru in _DIGRAPHS:
            if chunk.startswith(dig):
                hit = ru
                i += len(dig)
                break
        if hit:
            out.append(hit)
            continue
        ch = s[i]
        if ch == "c" and i + 1 < len(s) and s[i + 1] in "eiy":
            out.append("с")
        else:
            out.append(_CHAR.get(ch, ch))
        i += 1
    return "".join(out)


def _latin_token(token: str, table: dict) -> str:
    known = _lookup(token, table)
    if known:
        return known
    parts = re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)?", token)
    said = []
    for part in parts:
        if part[0].isdigit():
            said.append(_spell_num(part))
        elif _is_abbrev(part):
            said.append(_spell_letters(part))
        else:
            said.append(_translit(part))
    return " ".join(p for p in said if p)


_RU_CONS = set("бвгджзйклмнпрстфхцчшщ")


def _dict_pattern(table: dict) -> re.Pattern | None:
    """Ключи словаря целыми словами, без учёта регистра. Пробелы в ключе — любые пробелы.

    После латиницы может идти русское окончание: «OpenAI-ем», «Anthropic'а», «Google-а».
    """
    keys = sorted((k for k, v in table.items() if k and v), key=len, reverse=True)
    if not keys:
        return None
    alts = "|".join(re.escape(k).replace("\\ ", r"\s+") for k in keys)
    return re.compile(rf"(?<![A-Za-z0-9])(?P<key>{alts})(?:[-'’]?(?P<end>[а-яё]{{1,4}}))?"
                      rf"(?![A-Za-z0-9А-Яа-яЁё])", re.IGNORECASE)


def _hold_map(text: str, table: dict) -> tuple[str, list[str]]:
    """Подставить словарь и спрятать замены, чтобы их не перевели второй раз."""
    if not table:
        return text, []
    pattern = _dict_pattern(table)
    if pattern is None:
        return text, []
    saved: list[str] = []

    def sub(m: re.Match) -> str:
        val = _lookup(" ".join(m.group("key").split()), table)
        if not val:
            return m.group(0)
        end = m.group("end") or ""
        if end:
            val += ("" if val[-1].casefold() in _RU_CONS else "-") + end
        saved.append(val)
        return _mark(len(saved) - 1)

    return pattern.sub(sub, text), saved


def _restore(text: str, saved: list[str]) -> str:
    def sub(m: re.Match) -> str:
        try:
            idx = _unmark(m.group(1))
        except ValueError:
            return m.group(0)
        return saved[idx] if 0 <= idx < len(saved) else m.group(0)
    return _HOLD.sub(sub, text)


def _spell_doubled_abbr(text: str) -> str:
    """«ИИ» → «и-и», «СС» → «эс-эс». Строчные «ее» — местоимение, его не трогаем."""

    def sub(m: re.Match) -> str:
        name = _CYR_NAMES.get(m.group(1).casefold(), m.group(1).casefold())
        return f"{name}-{name}"

    return _DOUBLED_ABBR.sub(sub, text)


def _spell_cyr_letters(text: str) -> str:
    """Одиночная кириллическая буква — её русское название, не цифра."""

    def sub(m: re.Match) -> str:
        ch = m.group(1)
        low = ch.casefold()
        if low in _KEEP_ONE:
            return ch
        name = _CYR_NAMES.get(low, "")
        if not name or name == low:
            return ch
        return name

    return _CYR_ONE.sub(sub, text)


def latin_left(text: str, replacements: dict | None = None) -> list[str]:
    """Латиница, которой нет в словаре: она уйдёт в голос транслитом или по буквам."""
    table = dict(DEFAULT_REPLACEMENTS)
    if replacements:
        table.update({str(k): str(v) for k, v in replacements.items() if str(k) and str(v)})
    held, _saved = _hold_map(text or "", table)
    return list(dict.fromkeys(m.group(0) for m in re.finditer(r"[A-Za-z][A-Za-z']*", held)))


def prepare_for_speech(text: str, replacements: dict | None = None) -> str:
    """«В 2024 году вышел GPT» → «В две тысячи двадцать четвёртом году вышел джи-пи-ти».

    «буква З» → «буква зэ». Цифра 3 по-прежнему «три».
    """
    table = dict(DEFAULT_REPLACEMENTS)
    if replacements:
        table.update({str(k): str(v) for k, v in replacements.items() if str(k) and str(v)})
    raw = text or ""
    # «GPT-4» и «word1» — число отдельно, иначе оно прилипает к слову.
    raw = re.sub(r"(?<=[A-Za-zА-Яа-яЁё])-(?=\d)", " ", raw)
    raw = re.sub(r"(?<=[A-Za-zА-Яа-яЁё])(?=\d)", " ", raw)
    raw = re.sub(r"(?<=\d)(?=[A-Za-zА-Яа-яЁё])", " ", raw)
    held, saved = _hold_map(raw, table)

    def year_sub(m: re.Match) -> str:
        pre = m.group("pre") or ""
        post = m.group("post") or ""
        words = _spell_num(m.group("num"), _year_case(pre, post), ordinal=True)
        return f"{pre}{words}{post}"

    def percent_sub(m: re.Match) -> str:
        # «на 50%» читается «на пятьдесят процентов», без склонения числительного.
        pre = m.group("pre") or ""
        return f"{pre}{_spell_num(m.group('num'))} процентов"

    def num_sub(m: re.Match) -> str:
        pre = m.group("pre") or ""
        case = _PREP_CASE.get(pre.strip().casefold(), "n")
        return f"{pre}{_spell_num(m.group('num'), case)}"

    out = _YEAR.sub(year_sub, held)
    out = _PERCENT.sub(percent_sub, out)
    out = _NUMBER.sub(num_sub, out)

    def lat_sub(m: re.Match) -> str:
        return _latin_token(m.group(0), table)

    out = _LATIN.sub(lat_sub, out)
    out = _restore(out, saved)
    out = _spell_doubled_abbr(out)
    out = _spell_cyr_letters(out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()
