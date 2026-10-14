# AGENTS.md — памятка для разработчиков и ИИ-агентов

## Назначение

Видео-конвейер: ссылка (YouTube, X, Instagram, TikTok, VK…) или локальный файл → лучшие моменты →
вертикальные Shorts 9:16 (или целое видео 16:9) с заголовками, караоке-субтитрами, переводом на русский,
русской озвучкой по желанию, обложками → текст поста ВК и пакет для Shorts → отправка в Telegram-бот
(только отправка). Интерфейсы: CLI (`python -m konveyer`) и локальный веб-интерфейс (stdlib, 127.0.0.1).

## Шаги конвейера

`ingest → transcribe → analyze → moments → render → cover → post → send` (`konveyer/jobs.py: STEPS`).
Готовые шаги пропускаются; `redo --from <шаг>` сбрасывает шаг и всё после него.
Задание = папка `jobs/<id>/` + состояние в `job.json` + журнал `job.log`.

1. **ingest** — yt-dlp (до `[ingest] max_height`) или локальный файл (копия; жёсткая ссылка — `[ingest] link_local`),
   метаданные автора. Файл, который ffprobe не открыл, удаляется из задания.
2. **transcribe** — faster-whisper с таймингом слов; CUDA → CPU при ошибке; оценка надёжности речи (`[speech]`).
3. **analyze** — громкость RMS, склейки (`select=scene`), плотность речи.
4. **moments** — DeepSeek выбирает куски; без ключа — эвристика; границы к словам/склейкам.
5. **render** — ffmpeg: раскладки blur/full/smart, зум, ASS-караоке, цвета спикеров, перевод, озвучка, сжатие пауз,
   аппаратный кодировщик (NVENC/QSV/AMF) с откатом на libx264, проверки качества.
6. **cover** — кадр или нейросеть (Pollinations), заголовок от DeepSeek, шрифт Russo One.
7. **post** — пост ВК, описания Shorts, хештеги, таймкоды; без ключа — шаблон.
8. **send** — Telegram: клипы, обложки, пост; пережатие под лимит; замок от повторной отправки.

Режим `full` — целое видео 16:9 без нарезки (`fullvideo.py`), с переводом/озвучкой целиком и сторожем (`guard.py`).

## Карта модулей (`konveyer/`)

| Модуль | Что делает |
|---|---|
| `cli.py`, `__main__.py` | argparse, команды, `doctor` |
| `config.py` | dataclass-настройки: значения по умолчанию + `konveyer.toml` + `--set`, проверка |
| `envkeys.py` | секреты только из окружения/`.env`, маскировка в журнале |
| `i18n.py` | язык интерфейса RU/EN, словарь переводов, перевод argparse |
| `log.py` | журнал в консоль и `job.log` через маскировку и `tr()` |
| `jobs.py` | задания, шаги, поиск задания (`last`, префикс id) |
| `pipeline.py` | порядок шагов, перерендер |
| `ingest.py`, `transcribe.py`, `transcript.py`, `speech.py` | скачивание, распознавание, надёжность речи |
| `analysis.py`, `moments.py`, `manual.py`, `timeline.py` | сигналы, выбор моментов, ручные таймкоды, сжатие пауз |
| `render.py`, `layout.py`, `captions.py`, `textfit.py`, `facetrack.py`, `hardware.py` | рендер, раскладки, субтитры, лицо, кодировщик |
| `translate.py`, `llm.py`, `usage.py` | перевод, клиент DeepSeek (OpenAI API), учёт токенов |
| `voice.py`, `saytext.py`, `voicecheck.py`, `xtts_worker.py`, `speakers.py` | озвучка Edge/Silero/XTTS, текст для синтеза, проверка, диаризация |
| `cover.py`, `post.py`, `postfmt.py`, `clipdesc.py`, `quality.py`, `mediaq.py` | обложки, посты, проверки перед отправкой |
| `telegram.py`, `cleanup.py`, `fullvideo.py`, `guard.py` | отправка, автоочистка, целое видео, сторож |
| `web/server.py`, `web/static/index.html` | веб-интерфейс и API (POST требует `X-Konveyer: 1`, любой запрос — loopback `Host`) |

## Команды

```
py -m konveyer [--lang ru|en] [--config PATH] [--set раздел.ключ=значение] <команда>
run <ссылка|файл>   full <ссылка|файл> [--from --to]   jobs   show <id>   redo <id> --from <шаг>
clip <id> <номер | 2:10-2:50 ...>   cover <id>   post <id>   describe <id>   send <id>
guard <id>   web [--port] [--no-open]   doctor   config   cleanup [--dry-run]
```
`py -m konveyer --help` / `<команда> --help` — все флаги.

## Конфигурация

- **`.env`** (образец `.env.example`, в git не попадает): `DEEPSEEK_API_KEY`, `TG_BOT_TOKEN`, `TG_CHAT_ID`, `HF_TOKEN`,
  `KONVEYER_LANG`. Загружается `envkeys.load_dotenv()`; настоящие переменные окружения важнее. Все ключи необязательны.
- **`konveyer.toml`** (копия `konveyer.toml.example`, в git не попадает): разделы `[ui] [paths] [ingest] [whisper] [speech]
  [moments] [render] [subs] [voice] [voice.expression] [quality] [full] [llm] [post] [telegram] [cover] [web] [cleanup]`.
  Секреты в конфиг не кладём. Относительные пути — от корня проекта (`config._resolve_paths`).
- Новый ключ: поле в dataclass в `config.py` + проверка в `_validate` + строка с комментарием в `konveyer.toml.example`.
- cookies: `[ingest] cookies_file = "cookies/…txt"` или `cookies_from_browser`; файлы cookies не коммитим.

## Язык интерфейса (i18n) — правила

- Приоритет: `--lang` > `KONVEYER_LANG` > `[ui] lang` > `ru`.
- Исходный язык строк — русский. Английский — в словаре `_EN` в `konveyer/i18n.py`.
- **Каждая новая строка интерфейса** (help, `log.info/warn/step`, `KonveyerError`, `print` в CLI, ошибки веб-API,
  строки `doctor`) должна иметь пару RU+EN: добавьте русский текст и перевод в `_EN`. Для f-строк добавляйте
  постоянные куски (фрагменты от 3 символов переводятся по границам слов).
- Веб-интерфейс: русский текст в `index.html`, перевод — в объекте `EN` (точные строки) или `RULES` (шаблоны).
  Пользовательские данные (названия, пост, журнал) не переводятся.
- Язык интерфейса **не меняет результат**: перевод и озвучка — на русский.
- `tests/test_smoke.py` проверяет, что в английской справке CLI нет кириллицы.

## Тесты

```
py -m pip install -r requirements-dev.txt   # requirements.txt + pytest
py -m pytest -q
```
Необязательные пакеты (`mediapipe`) — в `requirements-optional.txt`; тесты без них проходят. CI — `.github/workflows/tests.yml`.
Внешние сервисы (Whisper, DeepSeek, Edge, Silero, Ollama, Telegram, сеть) в тестах подменены; часть тестов требует ffmpeg
(`@needs_ffmpeg`). Около 470 тестов; `test_i18n.py` и `test_smoke.py` — проверки импорта, i18n и примеров конфигов.

**Известные падения:** нет. Раньше нестабильны были `test_full.py::test_block_keeps_quiet_onset_and_does_not_overlap`
и `test_edge.py::test_deferred_block_recovers_with_main_voice_no_fallback`: Edge синтезирует фразы в параллельных
потоках, а тесты зависели от порядка вызовов. Правило: в тестах с подменённым синтезом не полагайтесь на порядок
вызовов — выбирайте ответ по тексту фразы или ставьте `settings.voice.edge_workers = 1`.

## Соглашения

- Python 3.12+, `from __future__ import annotations`, стандартная библиотека где можно; тяжёлые пакеты
  (torch, pyannote, mediapipe, demucs, coqui-tts) — необязательные, импорт внутри функций, без них — понятный откат.
- Ошибки для пользователя — `KonveyerError` с коротким текстом и подсказкой.
- Все сообщения — через `Log` (маскирует секреты). Значения секретов никогда не печатаются.
- Внешние программы — через `util.run`/`need_tool` (ffmpeg, ffprobe, yt-dlp, deno).
- Шрифты drawtext — только через `layout.resolve_text_font` (конфиг → `assets/fonts` → DejaVu/Liberation/Arial);
  нет шрифта — предупреждение и клип без надписей, не падение. Обложки — `cover.resolve_font`.
- Числа из веб-API — только через `web.server.num_param` (диапазон, без nan/inf, целые — целые).
- Списки голосов сравниваются без учёта регистра (`casefold`), дубли убираются.
- Веб — только loopback: `[web] host` проверяется в `_validate`, заголовок `Host` — в каждом запросе
  (`server.allowed_host`); через веб принимаются только ссылки и видеофайлы (`server.check_source`). Telegram — только отправка.
- Одно задание — один процесс: `job.json` пишется атомарно (`util.write_json`), но веб и CLI на одном задании
  одновременно не согласуются — последняя запись побеждает. Внутри веба задание в очереди/работе даёт 409 (`Busy`).
- Длинный граф ffmpeg — через `util.filter_complex_args` (файлом, иначе упрёмся в ~32 КБ командной строки Windows).
- Пути внутри фильтров ffmpeg — `util.ffmpeg_filter_path` в одинарных кавычках.
- Пути — `pathlib`. `ffmpeg`, `ffprobe`, `yt-dlp`, `deno` ищутся в PATH (`util.need_tool` / `shutil.which`), без зашитых `.exe`.
  Шрифты: конфиг → системные (Windows через `%WINDIR%`, Linux/SteamOS — `/usr/share/fonts`) → `assets/fonts`.
  `.bat` — только удобный запуск на Windows; на Linux команды из README.

## Правила репозитория

- **Никаких секретов**: ключи, токены, chat id, cookies, `.env`, `konveyer.toml` — не коммитить.
- **Никаких личных данных**: абсолютных путей пользователя, имён машин, имён людей, названий своих ботов/каналов, IP, почт.
  Пути — относительные или настраиваемые.
- Не коммитить результаты и медиа: `jobs/`, `*.log`, видео/аудио, модели (`models/`, `*.pt`), venv.
- Перед коммитом: `py -m pytest -q` и поиск по шаблонам `sk-`, `AIza`, `hf_`, токенов вида `123456:ABC…`, `Users\`, `@gmail`.
