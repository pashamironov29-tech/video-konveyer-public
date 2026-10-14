"""Настройки: значения по умолчанию + konveyer.toml + флаги командной строки.

Секретов здесь нет и быть не должно: ключи читаются только из переменных окружения
(см. envkeys.py).
"""
from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from konveyer.util import KonveyerError

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsCfg:
    jobs_dir: str = "jobs"                       # относительно корня проекта или абсолютный
    subs_dir: str = ""                           # необязательно: папка с вашим subs.py (свой промпт перевода)


@dataclass
class IngestCfg:
    max_height: int = 1080                       # для вертикали 1080x1920 больше не нужно
    cookies_from_browser: str = ""               # "firefox" / "chrome" — для Instagram и X
    cookies_file: str = ""                       # cookies.txt (Netscape) для yt-dlp, напр. cookies/youtube_cookies.txt
    link_local: bool = False                     # True — жёсткая ссылка на локальный файл вместо копии


@dataclass
class WhisperCfg:
    device: str = "auto"                         # auto | cuda | cpu
    model_gpu: str = "large-v3"
    model_cpu: str = "small"
    compute_gpu: str = "float16"
    compute_cpu: str = "int8"
    language: str = ""                           # "" — определить самому
    beam_size: int = 5
    vad: bool = True


@dataclass
class SpeechCfg:
    """Когда расшифровке нельзя верить. Сильный сигнал один режет субтитры, слабые — пачкой."""
    force: bool = False                         # субтитры даже если речь ненадёжна
    max_compression_ratio: float = 2.4          # сильный: порог самого Whisper
    max_char_run_ratio: float = 0.30            # сильный: доля символов в повторе 3+ подряд
    max_top_word_ratio: float = 0.60            # сильный: одно слово занимает столько слов
    min_words_for_repeat: int = 4
    combo_no_speech: float = 0.80               # сильный вместе с combo_logprob
    combo_logprob: float = -0.8
    weak_to_reject: int = 2                     # сколько слабых сигналов нужно, чтобы отказать
    min_language_prob: float = 0.85             # слабый: вероятность языка ниже
    max_no_speech_prob: float = 0.60            # слабый: порог тишины Whisper
    min_avg_logprob: float = -1.0               # слабый: средняя уверенность ниже
    min_speech_ratio: float = 0.12              # слабый: доля времени с речью (VAD/сегменты)
    min_duration_for_ratio: float = 8.0         # короче — долю времени не смотрим
    min_unique_word_ratio: float = 0.35         # слабый: разных слов меньше этой доли
    min_words_for_unique: int = 6


@dataclass
class MomentsCfg:
    count: int = 3
    min_sec: float = 15.0
    max_sec: float = 60.0
    whole_if_shorter: float = 75.0               # ролик короче — берём целиком одним клипом
    use_llm: bool = True


@dataclass
class RenderCfg:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    layout: str = "auto"                         # auto | blur | full | smart
    captions: bool = True
    titles: bool = True                          # белый заголовок, строка сверху и жёлтая подпись
    karaoke: bool = True                         # подсветка слова (если речь уже русская)
    translate: bool = True                       # перевести субтитры на русский
    caption_upper: bool = False
    caption_size: int = 74
    caption_font: str = "Arial"
    tighten: bool = False                        # вырезать паузы в речи
    pause_sec: float = 0.6
    keep_pad: float = 0.12
    encoder: str = "auto"                        # auto | h264_nvenc | h264_qsv | h264_amf | libx264
    crf: int = 20
    preset: str = "medium"
    font_title: str = ""                         # пусто — системный шрифт, затем assets/fonts
    font_text: str = ""
    blur_sigma: float = 6.0
    cover: bool = True
    zoom: float = 1.0                        # 1.0 — кадр 16:9 целиком (по умолчанию), 1.5 — старый зум: крупнее, бока обрезаны
    zoom_center: bool = False                # зум строго по центру кадра, без слежения за лицом/героем


# Короткие имена Edge → полное имя нейронного голоса. brian первый: он по умолчанию.
# Andrew намеренно нет.
EDGE_VOICES = {
    "brian": "en-US-BrianMultilingualNeural",
    "dmitry": "ru-RU-DmitryNeural",
    "svetlana": "ru-RU-SvetlanaNeural",
    "ava": "en-US-AvaMultilingualNeural",
    "emma": "en-US-EmmaMultilingualNeural",
    "florian": "de-DE-FlorianMultilingualNeural",
    "seraphina": "de-DE-SeraphinaMultilingualNeural",
}
EDGE_SPEAKERS = tuple(EDGE_VOICES)
SILERO_SPEAKERS = ("aidar", "eugene", "baya", "kseniya", "xenia")   # первые два мужские
# В --voice-speaker оба семейства: несовпадение с движком ловит speaker_error, не argparse.
VOICE_SPEAKERS = EDGE_SPEAKERS + SILERO_SPEAKERS
VOICE_ENGINES = ("edge", "silero", "xtts")
SUBS_STYLES = ("normal", "big")

_PCT = re.compile(r"^([+-]?\d+(?:\.\d+)?)%$")
_HZ = re.compile(r"^([+-]?\d+(?:\.\d+)?)Hz$")


def default_expression() -> dict:
    """Сдвиги Edge относительно [voice] edge_rate / edge_pitch. Включено, пока нет --voice-flat."""
    return {
        "enabled": True,
        "pause_ms": 300,                         # пауза между предложениями, как break у Silero
        "ellipsis_ms": 500,
        "serious_chars": 80,                     # длинное утверждение без ?!… — чуть ниже и медленнее
        "question_rate": "+0%",
        "question_pitch": "+8Hz",
        "question_volume": "+0%",
        "exclamation_rate": "+8%",
        "exclamation_pitch": "+12Hz",
        "exclamation_volume": "+0%",
        "serious_rate": "-5%",
        "serious_pitch": "-4Hz",
        "serious_volume": "+0%",
        "ellipsis_rate": "-8%",
        "ellipsis_pitch": "+0Hz",
        "ellipsis_volume": "-10%",
        "normal_rate": "+0%",
        "normal_pitch": "+0Hz",
        "normal_volume": "+0%",
    }


def parse_percent(value: str) -> float:
    m = _PCT.fullmatch(str(value).strip())
    if not m:
        raise KonveyerError(f"ожидалось значение вроде +0% или -5%, получено: {value}")
    return float(m.group(1))


def parse_hz(value: str) -> float:
    m = _HZ.fullmatch(str(value).strip())
    if not m:
        raise KonveyerError(f"ожидалось значение вроде +0Hz или +8Hz, получено: {value}")
    return float(m.group(1))


def format_percent(factor: float) -> str:
    pct = (factor - 1.0) * 100.0
    if abs(pct) < 0.05:
        return "+0%"
    rounded = round(pct, 1)
    if abs(rounded - round(rounded)) < 1e-6:
        return f"{int(round(rounded)):+d}%"
    return f"{rounded:+.1f}%"


def format_hz(value: float) -> str:
    if abs(value) < 0.05:
        return "+0Hz"
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value)):+d}Hz"
    return f"{value:+.1f}Hz"


def combine_rates(*parts: str | float) -> str:
    """Сложить темпы: проценты Edge и коэффициент ускорения перемножаются."""
    factor = 1.0
    for part in parts:
        if isinstance(part, (int, float)):
            factor *= float(part)
        else:
            factor *= 1.0 + parse_percent(part) / 100.0
    return format_percent(factor)


def combine_hz(*parts: str) -> str:
    return format_hz(sum(parse_hz(p) for p in parts))


def speaker_error(engine: str, speaker: str) -> str | None:
    """Пустая строка смысла нет: None — голос подходит движку. XTTS голос не выбирает."""
    if engine == "xtts":
        return None
    if engine == "edge" and speaker not in EDGE_VOICES:
        return (
            f"voice.speaker: голос «{speaker}» не подходит движку edge. "
            f"Для Edge: {', '.join(EDGE_SPEAKERS)}. "
            f"Голоса Silero ({', '.join(SILERO_SPEAKERS)}) — с --voice-engine silero."
        )
    if engine == "silero" and speaker not in SILERO_SPEAKERS:
        return (
            f"voice.speaker: голос «{speaker}» не подходит движку silero. "
            f"Для Silero: {', '.join(SILERO_SPEAKERS)}. "
            f"Голоса Edge ({', '.join(EDGE_SPEAKERS)}) — с --voice-engine edge."
        )
    return None


_EXPR_BOOL = {"enabled"}
_EXPR_INT = {"pause_ms", "ellipsis_ms", "serious_chars"}


def merge_expression(current: dict, value: dict) -> dict:
    """[voice.expression] дополняет значения по умолчанию, типы не превращает в строки."""
    merged = dict(current)
    for key, item in value.items():
        if key in _EXPR_BOOL:
            merged[key] = _coerce(item, True)
        elif key in _EXPR_INT:
            merged[key] = _coerce(item, 0)
        else:
            merged[key] = "" if item is None else str(item)
    return merged


@dataclass
class VoiceCfg:
    """Русская озвучка клипов поверх приглушённого оригинала."""
    enabled: bool = False
    engine: str = "edge"                        # edge | silero | xtts
    model_path: str = "models/silero/v4_ru.pt"   # v4_ru.pt, путь относительно проекта или абсолютный
    speaker: str = "brian"
    sample_rate: int = 48000
    device: str = "auto"                         # auto | cuda | cpu (для Silero)
    subs: bool = True                            # субтитры остаются вместе с озвучкой
    max_tempo: float = 1.3                       # сильнее ускорять — уже неразборчиво
    shorten: bool = True                         # не влезает и после ускорения — DeepSeek сокращает
    duck_db: float = -18.0                       # насколько приглушать оригинал под озвучкой
    duck_fade: float = 0.15                      # плавный спуск/подъём громкости, с
    gain_db: float = 0.0                         # громкость самой озвучки
    separate: bool = False                       # demucs: убрать голос, оставить музыку и шумы
    demucs_model: str = "htdemucs"
    sample: str = ""                             # xtts: чистая запись голоса 10–30 с (wav)
    xtts_python: str = ""                        # python отдельного venv с XTTS; "" — текущий
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_timeout: float = 1800.0
    edge_rate: str = "+0%"                       # базовый темп Edge, к нему прибавляется интонация
    edge_pitch: str = "+0Hz"
    edge_max_rate: float = 1.15                  # ускорение пересинтезом Edge, до +15%
    edge_workers: int = 4                        # сколько фраз Edge озвучивает одновременно
    edge_timeout: float = 20.0                   # один запрос к Edge дольше — неудача
    edge_breaker: int = 12                       # столько неудач подряд — отдых для всех, без 2-минутных ожиданий
    edge_rest_sec: float = 30.0                  # длина такого отдыха
    edge_budget_min: float = 45.0                # лимит на озвучку задания; дальше — запасной голос (0 — без лимита)
    voice_mode: str = "voiceover"               # voiceover — закадр в естественном темпе; fit — подогнать под окно
    strip_commas: bool = True                    # перед синтезом убрать запятые, тире и многоточия; точки и вопросы остаются
    # Частые слова для озвучки: «GPT» → «джи-пи-ти». Пустое значение слово не подменяет.
    replacements: dict = field(default_factory=dict)
    # Произношение для голоса: [voice.pronounce] или файл pronunciations.toml рядом с конфигом.
    # Дополняет и перекрывает встроенный словарь (saytext.DEFAULT_REPLACEMENTS).
    pronounce: dict = field(default_factory=dict)
    expression: dict = field(default_factory=default_expression)
    voice_check: bool = True                    # Whisper + DeepSeek после озвучки клипа
    voice_check_min: int = 7                    # ниже — клип не уходит в Telegram
    # Разные голоса Edge — обычный путь, включено. --one-voice выключает.
    per_speaker: bool = True
    # Если пол не удалось определить: разные голоса из этой палитры, по метке S1, S2.
    speaker_voices: list = field(default_factory=lambda: [
        "brian", "ava", "florian", "emma", "dmitry", "seraphina", "svetlana"])
    male_voices: list = field(default_factory=lambda: ["brian", "florian", "dmitry"])
    female_voices: list = field(default_factory=lambda: ["ava", "emma", "seraphina", "svetlana"])
    # Явный голос на метку спикера, пол не проверяется: {S2 = "dmitry"}; в --set: S2=dmitry,S3=florian
    speaker_map: dict = field(default_factory=dict)


@dataclass
class SubsCfg:
    """Стиль субтитров. normal — как раньше, big — 1–2 слова крупно в нижней половине.

    speakers: разный цвет и свой голос на каждого. Голос по полу, если его слышно,
    и закреплён за меткой на весь ролик. Цвет и голос смотрят на одну метку S1, S2.
    speaker_fonts пустой — шрифт один, из [render] caption_font.
    """
    style: str = "normal"                    # normal | big
    speakers: bool = True
    speaker_colors: list = field(default_factory=lambda: ["#FFFFFF", "#FFE14A", "#7EC8FF"])
    speaker_fonts: list = field(default_factory=list)
    diarize_model: str = "pyannote/speaker-diarization-3.1"
    diarize_device: str = "auto"             # auto | cuda | cpu
    hf_token: str = ""                       # лучше HF_TOKEN в окружении, файл конфига не коммитить


@dataclass
class QualityCfg:
    """Проверка клипа перед Telegram. Кадры, субтитры, звук и ffprobe включены всегда.
    Оценка DeepSeek и Ollama — только флагами, по умолчанию выключены."""
    min_score: int = 6                       # клип ниже этого балла не уходит, если включён --score
    score: bool = False
    vision: bool = False
    ollama_url: str = "http://127.0.0.1:11434"
    vision_model: str = "qwen2.5vl"
    vision_timeout: float = 90.0
    shorts_max_sec: float = 60.0
    silence_sec: float = 1.0                 # тишина длиннее этого в начале или конце — замечание
    min_mean_db: float = -36.0               # тише — «слишком тихо»
    clip_db: float = -0.5                    # max_volume выше — клиппинг


@dataclass
class LLMCfg:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.9
    timeout: float = 120.0
    attempts: int = 3
    max_transcript_chars: int = 60000
    # Доллары за 1 млн токенов. 0 у всех трёх — сумму в журнале не пишем, только токены.
    # Непик Flash на октябрь 2026 (deepseek-chat тарифицируется как Flash). Пик примерно ×2.
    price_input: float = 0.15                   # вход, cache miss
    price_output: float = 0.60
    price_cache_hit: float = 0.003              # вход, prompt_cache_hit_tokens


@dataclass
class PostCfg:
    group: str = "My channel"
    style_files: list = field(default_factory=lambda: [
        "prompts/post_prompt.md",
        "prompts/post_example.md",
    ])
    shorts_pack: bool = True
    shorts_hashtags: str = "#shorts #россия"
    emoji: bool = True              # каждый абзац поста — с эмодзи, перед «Таймкоды:» — ⏱
    hashtags_on: bool = True        # строка 6–10 хештегов в конце поста
    hashtags: str = "#ИИ #нейросети #искусственныйинтеллект #интервью #технологии"  # всегда
    clip_desc: bool = True          # своё описание к каждому Shorts (clips/clip_NN/description.md)


@dataclass
class TelegramCfg:
    max_mb: float = 50.0
    send_cover: bool = False
    send_post: bool = True
    shorts_post: bool = False       # общий пост после Shorts (у каждого клипа своё описание)


@dataclass
class WebCfg:
    host: str = "127.0.0.1"         # только loopback: у интерфейса нет пароля
    port: int = 8765


WEB_HOSTS = ("127.0.0.1", "localhost", "::1")


COVER_MODES = ("frame", "ai", "both")
_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$|^[A-Za-z]{3,20}$")


@dataclass
class CoverCfg:
    """Обложка ВК. frame — кадр из ролика (по умолчанию), ai — фон нейросети, both — обе."""
    enabled: bool = True
    mode: str = "frame"                      # frame | ai | both; ИИ-фон (Pollinations) выключен
    font: str = ""                           # пусто — assets/fonts/RussoOne-Regular.ttf
    max_words: int = 5
    text_color: str = "#FFFFFF"
    accent_color: str = "#FFE14A"            # плашка с именем спикера
    stroke_color: str = "#000000"
    candidates: int = 12                     # сколько кадров смотреть, выбирается лучший
    jpeg_max_mb: float = 2.0
    provider: str = "pollinations"
    pollinations_url: str = "https://image.pollinations.ai/prompt"
    ai_timeout: float = 40.0
    ai_attempts: int = 3


@dataclass
class FullCfg:
    """Целое видео 16:9 для ВК: без нарезки на Shorts, без зума и без размытого фона."""
    max_height: int = 1080                 # исходник не выше 1080p и не увеличивается
    caption_size: int = 0                  # 0 — от высоты кадра; иначе размер шрифта субтитров
    caption_margin: int = 0                # 0 — отступ снизу от высоты; иначе пиксели
    caption_box: bool = True               # полупрозрачная тёмная подложка под субтитрами
    max_chars: int = 42                    # сколько символов в одной реплике субтитров
    max_line: int = 36                     # символов в строке на экране
    translate_batch: int = 30              # реплик в одном запросе перевода
    translate_chars: int = 6000            # и символов в том же запросе, что короче
    lag_sec: float = 2.0                   # отставание русского, дальше которого не уходим
    tail_sec: float = 3.0                  # стоп-кадр в конце, не дольше
    voice_rate: str = "+8%"                # базовый темп закадра целого ролика
    fade_sec: float = 0.5                  # фейд картинки, если в конце дописываем последний кадр
    telegram_max_mb: float = 50.0          # больше — в чат сжатая копия или только путь
    telegram_compress: bool = True         # если файл больше порога — video_tg.mp4 около 48 МБ
    telegram_target_mb: float = 48.0       # во сколько метить сжатую копию
    min_video_kbps: int = 300              # ниже этого битрейта ролик не жмём, в чат только путь
    voice_check_samples: int = 4           # сколько фрагментов слушать, не весь ролик
    voice_check_window: float = 18.0       # длина одного фрагмента, секунды
    topics: int = 8                        # таймкодов в посте
    progress_sec: float = 20.0             # как часто писать в журнал, что кодирование ещё идёт
    voice_batch: int = 20                  # фраз озвучки между строками прогресса


@dataclass
class UiCfg:
    """Язык интерфейса / interface language: "ru" или "en"."""
    lang: str = "ru"


@dataclass
class CleanupCfg:
    """Автоочистка: после успешной отправки — исходники и временные файлы,
    задания старше keep_days — целиком (только уже отправленные)."""
    enabled: bool = True
    keep_days: float = 3.0


@dataclass
class Settings:
    paths: PathsCfg = field(default_factory=PathsCfg)
    ingest: IngestCfg = field(default_factory=IngestCfg)
    whisper: WhisperCfg = field(default_factory=WhisperCfg)
    speech: SpeechCfg = field(default_factory=SpeechCfg)
    moments: MomentsCfg = field(default_factory=MomentsCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    subs: SubsCfg = field(default_factory=SubsCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    voice: VoiceCfg = field(default_factory=VoiceCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    post: PostCfg = field(default_factory=PostCfg)
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    web: WebCfg = field(default_factory=WebCfg)
    cover: CoverCfg = field(default_factory=CoverCfg)
    full: FullCfg = field(default_factory=FullCfg)
    cleanup: CleanupCfg = field(default_factory=CleanupCfg)
    ui: UiCfg = field(default_factory=UiCfg)
    root: Path = ROOT

    @property
    def jobs_dir(self) -> Path:
        p = Path(self.paths.jobs_dir)
        return p if p.is_absolute() else self.root / p

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("root", None)
        token = str((d.get("subs") or {}).get("hf_token") or "")
        if token:
            d["subs"]["hf_token"] = "***"
        return d


SECTIONS = [f.name for f in dataclasses.fields(Settings) if f.name != "root"]


def _coerce(value, current):
    """Привести строку из CLI к типу текущего значения."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on", "да"):
            return True
        if s in ("0", "false", "no", "off", "нет"):
            return False
        raise KonveyerError(f"ожидалось да/нет, получено: {value}")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise KonveyerError(f"ожидалось целое число, получено: {value}") from None
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise KonveyerError(f"ожидалось число, получено: {value}") from None
    if isinstance(current, list):
        return list(value) if isinstance(value, (list, tuple)) else [str(value)]
    if isinstance(current, dict):
        if isinstance(value, str) and "=" in value:
            pairs = [p.split("=", 1) for p in value.split(",") if p.strip()]
            value = {k.strip(): v.strip() for k, v in pairs}
        if not isinstance(value, dict):
            raise KonveyerError(f"ожидался словарь, получено: {value}")
        return {str(k): "" if v is None else str(v) for k, v in value.items()}
    return str(value)


def apply(settings: Settings, section: str, key: str, value) -> None:
    if section not in SECTIONS:
        raise KonveyerError(f"в конфиге нет раздела [{section}]. Есть: {', '.join(SECTIONS)}")
    obj = getattr(settings, section)
    if not hasattr(obj, key):
        names = ", ".join(f.name for f in dataclasses.fields(obj))
        raise KonveyerError(f"в разделе [{section}] нет ключа {key}. Есть: {names}")
    if section == "voice" and key == "expression":
        if not isinstance(value, dict):
            raise KonveyerError("voice.expression: ожидался раздел [voice.expression]")
        obj.expression = merge_expression(obj.expression, value)
        return
    current = getattr(obj, key)
    new = _coerce(value, current)
    if isinstance(current, dict) and isinstance(new, dict):
        merged = dict(current)
        merged.update(new)
        setattr(obj, key, merged)
        return
    setattr(obj, key, new)


def load_settings(path: Path | None = None, overrides: dict | None = None,
                  root: Path | None = None) -> Settings:
    """overrides: {"render.layout": "blur", ...} — то, что пришло из флагов CLI."""
    settings = Settings(root=root or ROOT)
    if path is None:
        env = os.environ.get("KONVEYER_CONFIG", "").strip()
        path = Path(env) if env else settings.root / "konveyer.toml"
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8-sig"))   # Блокнот/PowerShell пишут BOM
        except tomllib.TOMLDecodeError as e:
            raise KonveyerError(f"ошибка в {path.name}: {e}") from e
        for section, values in data.items():
            if not isinstance(values, dict):
                raise KonveyerError(f"{path.name}: «{section}» должен быть разделом [{section}]")
            for key, value in values.items():
                apply(settings, section, key, value)
    extra = settings.root / "pronunciations.toml"
    if extra.is_file():
        try:
            words = tomllib.loads(extra.read_text(encoding="utf-8-sig"))
        except tomllib.TOMLDecodeError as e:
            raise KonveyerError(f"ошибка в {extra.name}: {e}") from e
        words = words.get("pronounce", words) if isinstance(words.get("pronounce"), dict) else words
        settings.voice.pronounce = {**settings.voice.pronounce,
                                    **{str(k): str(v) for k, v in words.items()
                                       if not isinstance(v, dict) and str(v).strip()}}
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        apply(settings, section, key, value)
    _resolve_paths(settings)
    _validate(settings)
    return settings


def _resolve_paths(s: Settings) -> None:
    """Относительные пути — от корня проекта, а не от текущей папки."""
    def rel(p: str) -> str:
        if not p:
            return p
        q = Path(p)
        return str(q if q.is_absolute() else s.root / q)
    s.voice.model_path = rel(s.voice.model_path)
    s.post.style_files = [rel(str(f)) for f in s.post.style_files]
    if s.paths.subs_dir:
        s.paths.subs_dir = rel(s.paths.subs_dir)


def _validate(s: Settings) -> None:
    if s.ui.lang not in ("ru", "en"):
        raise KonveyerError("ui.lang: ru или en")
    if str(s.web.host).strip().lower() not in WEB_HOSTS:
        raise KonveyerError("web.host: только 127.0.0.1, localhost или ::1 — "
                            "у веб-интерфейса нет пароля, наружу его открывать нельзя")
    if not 0 <= int(s.web.port) <= 65535:
        raise KonveyerError("web.port от 0 до 65535")
    if s.cleanup.keep_days < 0:
        raise KonveyerError("cleanup.keep_days: 0 или больше (0 — старые задания не удалять)")
    if s.render.layout not in ("auto", "blur", "full", "smart"):
        raise KonveyerError("render.layout: auto, blur, full или smart")
    if s.whisper.device not in ("auto", "cuda", "cpu"):
        raise KonveyerError("whisper.device: auto, cuda или cpu")
    if s.moments.count < 1:
        raise KonveyerError("moments.count должен быть ≥ 1")
    if not 3 <= s.moments.min_sec <= s.moments.max_sec:
        raise KonveyerError("нужно 3 ≤ moments.min_sec ≤ moments.max_sec")
    if s.render.width % 2 or s.render.height % 2:
        raise KonveyerError("render.width и render.height должны быть чётными")
    sp = s.speech
    if sp.weak_to_reject < 1:
        raise KonveyerError("speech.weak_to_reject должен быть ≥ 1")
    if sp.min_words_for_repeat < 1 or sp.min_words_for_unique < 1:
        raise KonveyerError("пороги числа слов в [speech] должны быть ≥ 1")
    if sp.max_compression_ratio <= 0:
        raise KonveyerError("speech.max_compression_ratio должен быть > 0")
    for name in ("max_char_run_ratio", "max_top_word_ratio", "min_language_prob",
                 "max_no_speech_prob", "min_speech_ratio", "min_unique_word_ratio",
                 "combo_no_speech"):
        if not 0 <= getattr(sp, name) <= 1:
            raise KonveyerError(f"speech.{name} должен быть от 0 до 1")
    for name in ("price_input", "price_output", "price_cache_hit"):
        if getattr(s.llm, name) < 0:
            raise KonveyerError(f"llm.{name} не может быть отрицательной")
    v = s.voice
    if v.engine not in VOICE_ENGINES:
        raise KonveyerError(f"voice.engine: {' или '.join(VOICE_ENGINES)}")
    err = speaker_error(v.engine, v.speaker)
    if err:
        raise KonveyerError(err)
    try:
        parse_percent(v.edge_rate)
    except KonveyerError as e:
        raise KonveyerError(f"voice.edge_rate: {e}") from e
    try:
        parse_hz(v.edge_pitch)
    except KonveyerError as e:
        raise KonveyerError(f"voice.edge_pitch: {e}") from e
    if not 1.0 <= v.edge_max_rate <= 2.0:
        raise KonveyerError("voice.edge_max_rate должен быть от 1.0 до 2.0")
    if v.voice_mode not in ("voiceover", "fit"):
        raise KonveyerError("voice.voice_mode: voiceover или fit")
    if not 1 <= int(v.voice_check_min) <= 10:
        raise KonveyerError("voice.voice_check_min от 1 до 10")
    _check_expression(v.expression)
    if not 1.0 <= v.max_tempo <= 2.0:
        raise KonveyerError("voice.max_tempo должен быть от 1.0 до 2.0")
    if v.duck_db > 0:
        raise KonveyerError("voice.duck_db — приглушение, значение ≤ 0 (например -18)")
    if v.device not in ("auto", "cuda", "cpu"):
        raise KonveyerError("voice.device: auto, cuda или cpu")
    if v.sample_rate not in (8000, 24000, 48000):
        raise KonveyerError("voice.sample_rate для Silero: 8000, 24000 или 48000")
    if not 1.0 <= s.render.zoom <= 2.0:
        raise KonveyerError("render.zoom должен быть от 1.0 до 2.0")
    if s.subs.style not in SUBS_STYLES:
        raise KonveyerError("subs.style: normal или big")
    _check_subs(s.subs)
    _check_speaker_voices(s.voice)
    q = s.quality
    if not 1 <= q.min_score <= 10:
        raise KonveyerError("quality.min_score от 1 до 10")
    if q.shorts_max_sec < 1:
        raise KonveyerError("quality.shorts_max_sec должен быть ≥ 1")
    if not q.ollama_url.startswith("http"):
        raise KonveyerError("quality.ollama_url должен начинаться с http")
    _check_cover(s.cover)
    _check_full(s.full)


def _check_cover(c: CoverCfg) -> None:
    if c.mode not in COVER_MODES:
        raise KonveyerError("cover.mode: frame, ai или both")
    if not 2 <= int(c.max_words) <= 8:
        raise KonveyerError("cover.max_words от 2 до 8")
    for name in ("text_color", "accent_color", "stroke_color"):
        if not _COLOR.fullmatch(str(getattr(c, name)).strip()):
            raise KonveyerError(f"cover.{name}: цвет вроде #FFFFFF или white")
    if not 3 <= int(c.candidates) <= 40:
        raise KonveyerError("cover.candidates от 3 до 40")
    if not 0.3 <= float(c.jpeg_max_mb) <= 8:
        raise KonveyerError("cover.jpeg_max_mb от 0.3 до 8")
    if not str(c.provider).strip():
        raise KonveyerError("cover.provider не может быть пустым")
    if not str(c.pollinations_url).startswith("http"):
        raise KonveyerError("cover.pollinations_url должен начинаться с http")
    if not 1 <= float(c.ai_timeout) <= 180:
        raise KonveyerError("cover.ai_timeout от 1 до 180 секунд")
    if not 1 <= int(c.ai_attempts) <= 6:
        raise KonveyerError("cover.ai_attempts от 1 до 6")


_NAMED_COLORS = {
    "white": "#FFFFFF", "yellow": "#FFE14A", "cyan": "#7EC8FF", "blue": "#7EC8FF",
    "белый": "#FFFFFF", "жёлтый": "#FFE14A", "желтый": "#FFE14A", "голубой": "#7EC8FF",
}


def normalize_color(value) -> str:
    text = str(value or "").strip()
    named = _NAMED_COLORS.get(text.casefold())
    if named:
        return named
    hexpart = text[1:] if text.startswith("#") else text
    if re.fullmatch(r"[0-9A-Fa-f]{3}", hexpart):
        hexpart = "".join(ch * 2 for ch in hexpart)
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", hexpart):
        raise KonveyerError(
            f"subs.speaker_colors: цвет «{text}» — нужен #RRGGBB или white/yellow/cyan")
    return "#" + hexpart.upper()


def _check_subs(subs: SubsCfg) -> None:
    if subs.diarize_device not in ("auto", "cuda", "cpu"):
        raise KonveyerError("subs.diarize_device: auto, cuda или cpu")
    if not str(subs.diarize_model or "").strip():
        raise KonveyerError("subs.diarize_model не может быть пустым")
    colors = list(subs.speaker_colors or [])
    if not 1 <= len(colors) <= 8:
        raise KonveyerError("subs.speaker_colors: от 1 до 8 цветов")
    subs.speaker_colors = [normalize_color(c) for c in colors]
    fonts = list(subs.speaker_fonts or [])
    if len(fonts) > 8:
        raise KonveyerError("subs.speaker_fonts: не больше 8")
    clean = []
    for font in fonts:
        name = str(font or "").strip()
        if not name or len(name) > 40 or any(ch in name for ch in "\\{}"):
            raise KonveyerError(f"subs.speaker_fonts: шрифт «{font}» не подходит")
        clean.append(name)
    subs.speaker_fonts = clean
    token = str(subs.hf_token or "").strip()
    if token and len(token) < 8:
        raise KonveyerError("subs.hf_token слишком короткий; лучше задать HF_TOKEN в окружении")
    subs.hf_token = token


_MALE_VOICES = ("brian", "florian", "dmitry")
_FEMALE_VOICES = ("ava", "emma", "seraphina", "svetlana")


def _check_voice_list(values, key: str, allowed: tuple[str, ...], limit: int) -> list[str]:
    names = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not 1 <= len(names) <= limit:
        raise KonveyerError(f"voice.{key}: от 1 до {limit} голосов")
    canon = {a.casefold(): a for a in allowed}
    clean, seen = [], set()
    for name in names:
        key_cf = name.casefold()
        if key_cf == "andrew":
            raise KonveyerError(f"voice.{key}: голос andrew не используем")
        if key_cf not in canon:
            raise KonveyerError(
                f"voice.{key}: «{name}» не подходит. Можно: {', '.join(allowed)}.")
        if key_cf not in seen:
            seen.add(key_cf)
            clean.append(canon[key_cf])
    return clean


def _check_speaker_voices(voice: VoiceCfg) -> None:
    voice.speaker_voices = _check_voice_list(voice.speaker_voices, "speaker_voices", EDGE_SPEAKERS, 8)
    voice.male_voices = _check_voice_list(voice.male_voices, "male_voices", _MALE_VOICES, 3)
    voice.female_voices = _check_voice_list(voice.female_voices, "female_voices", _FEMALE_VOICES, 4)
    clean = {}
    for label, name in (voice.speaker_map or {}).items():
        label, name = str(label).strip(), str(name).strip()
        if not re.fullmatch(r"S\d+", label):
            raise KonveyerError(f"voice.speaker_map: метка «{label}» — нужно S1, S2…")
        if name not in EDGE_SPEAKERS:
            raise KonveyerError(
                f"voice.speaker_map: «{name}» не подходит. Можно: {', '.join(EDGE_SPEAKERS)}.")
        clean[label] = name
    voice.speaker_map = clean


def _check_full(f: FullCfg) -> None:
    if f.max_height < 144 or f.max_height > 1080 or f.max_height % 2:
        raise KonveyerError("full.max_height — чётное число от 144 до 1080")
    if f.caption_size != 0 and not 16 <= f.caption_size <= 96:
        raise KonveyerError("full.caption_size: 0 (само) или от 16 до 96")
    if f.caption_margin != 0 and not 8 <= f.caption_margin <= 400:
        raise KonveyerError("full.caption_margin: 0 (само) или от 8 до 400")
    if not 12 <= f.max_chars <= 80:
        raise KonveyerError("full.max_chars от 12 до 80")
    if not 12 <= f.max_line <= 80:
        raise KonveyerError("full.max_line от 12 до 80")
    if not 1 <= f.translate_batch <= 80:
        raise KonveyerError("full.translate_batch от 1 до 80")
    if not 500 <= f.translate_chars <= 20000:
        raise KonveyerError("full.translate_chars от 500 до 20000")
    if not 0.3 <= f.lag_sec <= 15:
        raise KonveyerError("full.lag_sec от 0.3 до 15 секунд")
    if not 0 <= f.tail_sec <= 8:
        raise KonveyerError("full.tail_sec от 0 до 8 секунд")
    try:
        rate = 1.0 + parse_percent(f.voice_rate) / 100.0
    except KonveyerError as e:
        raise KonveyerError(f"full.voice_rate: {e}") from e
    if not 0.85 <= rate <= 1.30:
        raise KonveyerError("full.voice_rate от -15% до +30%")
    if not 0 <= f.fade_sec <= 3:
        raise KonveyerError("full.fade_sec от 0 до 3 секунд")
    if not 1 <= f.telegram_max_mb <= 2000:
        raise KonveyerError("full.telegram_max_mb от 1 до 2000")
    if not 4 <= f.telegram_target_mb <= 2000:
        raise KonveyerError("full.telegram_target_mb от 4 до 2000")
    if not 150 <= int(f.min_video_kbps) <= 4000:
        raise KonveyerError("full.min_video_kbps от 150 до 4000")
    if not 1 <= f.voice_check_samples <= 12:
        raise KonveyerError("full.voice_check_samples от 1 до 12")
    if not 3 <= f.voice_check_window <= 120:
        raise KonveyerError("full.voice_check_window от 3 до 120 секунд")
    if not 1 <= f.topics <= 30:
        raise KonveyerError("full.topics от 1 до 30")
    if not 2 <= f.progress_sec <= 600:
        raise KonveyerError("full.progress_sec от 2 до 600 секунд")
    if not 1 <= f.voice_batch <= 200:
        raise KonveyerError("full.voice_batch от 1 до 200")


def _check_expression(expr: dict) -> None:
    if not isinstance(expr, dict):
        raise KonveyerError("voice.expression: ожидался раздел [voice.expression]")
    if not isinstance(expr.get("enabled", True), bool):
        raise KonveyerError("voice.expression.enabled: да или нет")
    for name in ("pause_ms", "ellipsis_ms"):
        if int(expr.get(name, 0)) < 0:
            raise KonveyerError(f"voice.expression.{name} не может быть отрицательным")
    if int(expr.get("serious_chars", 1)) < 1:
        raise KonveyerError("voice.expression.serious_chars должен быть ≥ 1")
    for key, val in expr.items():
        if not isinstance(val, str):
            continue
        try:
            if key.endswith("_rate") or key.endswith("_volume"):
                parse_percent(val)
            elif key.endswith("_pitch"):
                parse_hz(val)
        except KonveyerError as e:
            raise KonveyerError(f"voice.expression.{key}: {e}") from e
