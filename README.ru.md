# Видео-конвейер

[English](README.md) · **Русский**

Локальный конвейер для видео: даёте ссылку (YouTube, X, Instagram, TikTok, VK…) или файл — получаете
вертикальные Shorts 9:16 или целое видео 16:9 с переведёнными субтитрами, озвучкой по желанию, обложками,
текстом поста и (по желанию) отправкой всего в ваш Telegram-бот.

Шаги: **скачивание → распознавание → анализ → лучшие моменты → рендер → обложка → пост → отправка**

- **Скачивание** — `yt-dlp` (лучшее качество до 1080p) или локальный файл.
- **Распознавание** — `faster-whisper` (видеокарта при наличии CUDA, иначе процессор), проверка надёжности речи.
- **Анализ / моменты** — склейки, громкость, плотность речи; лучшие куски выбирает DeepSeek (без ключа — эвристика).
- **Рендер** — ffmpeg, раскладки `auto | blur | full | smart` (слежение за лицом через mediapipe), зум, караоке-субтитры,
  цвета спикеров (диаризация pyannote), заголовки.
- **Перевод** — субтитры переводятся на русский через DeepSeek.
- **Озвучка** — русская озвучка через Edge TTS (по умолчанию), Silero или XTTS v2 (клон голоса), приглушение оригинала,
  удаление голоса через demucs, проверка озвучки Whisper'ом.
- **Обложка и пост** — обложка из кадра или нейросети (Pollinations), пост ВК / описания Shorts с таймкодами и хештегами.
- **Telegram** — бот только на отправку: клипы, обложки, пост. Большие файлы пережимаются.
- **Веб-интерфейс** (`python -m konveyer web`) и полный **CLI**. Автоочистка старых заданий.

## Язык интерфейса (RU / EN)

Интерфейс (справка CLI, журнал, ошибки, doctor, веб-интерфейс) — на русском (по умолчанию) или английском.
Способы (по убыванию приоритета):

1. Флаг: `python -m konveyer --lang en run <ссылка>`
2. Переменная окружения `KONVEYER_LANG=en` (можно в `.env`)
3. Конфиг: `[ui] lang = "en"` в `konveyer.toml`
4. Веб-интерфейс: переключатель **RU / EN** справа вверху (запоминается в браузере)

Язык интерфейса не меняет результат: перевод и озвучка, как и раньше, — на русский.

## Требования

- Windows 10/11, Linux (включая Steam Deck) и macOS. Пути — через `pathlib`. `ffmpeg`, `ffprobe` и `yt-dlp` ищутся в `PATH`, зашитых `.exe` нет.
- Python 3.12+
- `ffmpeg` и `ffprobe` в PATH
- `yt-dlp` (ставится из requirements) и **Deno** для YouTube (Windows: `winget install DenoLand.Deno`; Linux: скрипт установки Deno, см. ниже)
- По желанию: видеокарта NVIDIA + PyTorch с CUDA, ключ DeepSeek, Telegram-бот, токен Hugging Face (диаризация), Ollama (проверка кадров)

## Установка

ffmpeg и Deno (Windows):

```bat
winget install Gyan.FFmpeg
winget install DenoLand.Deno
```

Linux (Debian/Ubuntu):

```bash
sudo apt install ffmpeg python3-venv
curl -fsSL https://deno.land/install.sh | sh
```

Проект:

```bat
git clone https://github.com/pashamironov29-tech/video-konveyer-public.git video-konveyer
cd video-konveyer
py -m pip install -r requirements.txt
copy konveyer.toml.example konveyer.toml
copy .env.example .env
py -m konveyer doctor
```

На Linux вместо `py` — `python3`, вместо `copy` — `cp`. Полные шаги для Linux и Steam Deck —
в разделе [Запуск на Linux](#running-on-linux--запуск-на-linux). Все команды запускайте из папки репозитория:
пакет `konveyer` не ставится через pip, `python -m konveyer` находит его в текущей папке.

`INSTALL.bat` ставит зависимости, копирует образцы конфига (если их ещё нет) и запускает `doctor`. Дополнительно:

```bat
py -m pip install -r requirements-optional.txt                              :: mediapipe для layout = "smart"
py -m pip install torch --index-url https://download.pytorch.org/whl/cu128   :: Whisper/Silero на видеокарте
py -m pip install pyannote.audio                                            :: диаризация (нужен HF_TOKEN)
py -m pip install demucs                                                    :: --separate
```

## Озвучка

`[voice] engine` в `konveyer.toml`, включить — `--voice` или `[voice] enabled = true`.

- **edge** (по умолчанию) — Edge TTS, нужен интернет, ставится из requirements.
- **silero** — офлайн: нужен PyTorch и файл модели `v4_ru.pt` в `models/silero/` (или `[voice] model_path`).
- **xtts** — XTTS v2, клон голоса по записи `[voice] sample` (10–30 с, только с согласия владельца голоса).
  coqui-tts не ставится на новые Python, поэтому для него — отдельное окружение на Python 3.12:

```bash
python3 -m venv xtts-venv
```

Поставьте coqui-tts в это окружение и укажите его интерпретатор в `[voice] xtts_python`
(Linux: `xtts-venv/bin/python`, Windows: `xtts-venv/Scripts/python.exe`). При первом запуске модель (~2 ГБ)
скачивается с Hugging Face. `python -m konveyer doctor` покажет, видит ли конвейер XTTS.

## Что уходит в сеть

Конвейер работает на вашем компьютере, но некоторые шаги обращаются к внешним сервисам:

| Сервис | Что отправляется | Когда |
|---|---|---|
| Сайт источника (через yt-dlp) | ссылка, cookies если вы их указали | скачивание по ссылке |
| DeepSeek (`api.deepseek.com`) | текст расшифровки, название ролика | перевод, выбор моментов, заголовки, пост — если задан `DEEPSEEK_API_KEY` |
| Microsoft Edge TTS | текст озвучки | `engine = "edge"` |
| Pollinations (`image.pollinations.ai`) | короткое описание картинки | `[cover] mode = "ai"` или `"both"` |
| Telegram (`api.telegram.org`) | готовые клипы, обложки, пост | команда `send` / автоотправка |
| Hugging Face | скачивание моделей Whisper, XTTS, pyannote (с `HF_TOKEN`) | первый запуск |
| GitHub | шрифт Russo One, если его нет в `assets/fonts` | обложки |
| Ollama | кадры клипа | только локальный адрес `[quality] ollama_url` |

Без сети: локальный файл вместо ссылки, без `DEEPSEEK_API_KEY` (моменты — эвристикой, без перевода),
`engine = "silero"`, `[cover] mode = "frame"`, без `send`. Модели Whisper скачайте заранее.

## Безопасность

- Веб-интерфейс без пароля и слушает только этот компьютер: `[web] host` — `127.0.0.1`, `localhost` или `::1`,
  другие адреса конфиг не примет. Запросы с чужим заголовком `Host` отклоняются (защита от DNS rebinding).
- Через веб можно указать ссылку или видеофайл; другие файлы не принимаются.
- Локальный файл копируется в папку задания. Чтобы не тратить место на большие файлы — `[ingest] link_local = true`
  (жёсткая ссылка на том же диске; учтите: правка файла в задании меняет и исходник).
- Не запускайте веб и CLI на одном и том же задании одновременно: `job.json` пишется атомарно, но правки
  двух процессов не сливаются — последняя запись побеждает.
- Об уязвимостях — см. [SECURITY.md](SECURITY.md).

## Куда положить свои ключи, cookies и пути

В коде и конфиге секретов нет. Свои значения кладите сюда (всё в .gitignore):

| Что | Куда |
|---|---|
| `DEEPSEEK_API_KEY`, `TG_BOT_TOKEN`, `TG_CHAT_ID`, `HF_TOKEN` | `.env` (образец — `.env.example`) или переменные окружения |
| cookies YouTube / Instagram / X | `cookies/youtube_cookies.txt` + `[ingest] cookies_file = "cookies/youtube_cookies.txt"` или `[ingest] cookies_from_browser = "firefox"` |
| Настройки | `konveyer.toml` (копия `konveyer.toml.example`) |
| Модель Silero `v4_ru.pt` | `models/silero/v4_ru.pt` или `[voice] model_path` |
| python для XTTS | `[voice] xtts_python`: Linux `xtts-venv/bin/python`, Windows `xtts-venv/Scripts/python.exe` |
| Свой промпт и пример поста | `prompts/post_prompt.md`, `prompts/post_example.md` (`[post] style_files`) |
| Свой `subs.py` с промптом перевода | `[paths] subs_dir` |
| Название канала в постах | `[post] group`, `hashtags` |

Относительные пути считаются от папки проекта. Не коммитьте `.env`, `konveyer.toml` и файлы cookies.

## Использование

```bat
py -m konveyer run "https://youtu.be/..."             :: Shorts (по умолчанию 3 клипа)
py -m konveyer run video.mp4 --count 5 --voice        :: с русской озвучкой
py -m konveyer full "https://..." --from 1:00 --to 5:00 --voice
py -m konveyer jobs                                   :: список заданий
py -m konveyer show last                              :: подробности
py -m konveyer redo last --from render                :: переделать с шага
py -m konveyer clip last 2:10-2:50 5:00-5:40          :: Shorts по ручным таймкодам
py -m konveyer send last                              :: отправить в Telegram
py -m konveyer web                                    :: веб-интерфейс http://127.0.0.1:8765
py -m konveyer cleanup --dry-run
py -m konveyer --help                                 :: все команды и флаги
```

Или `RUN.bat` (спросит ссылку) / `WEB.bat`. Результаты — в `jobs/<id>/`.

Тесты:

```bat
py -m pip install -r requirements-dev.txt
py -m ruff check .
py -m pytest -q
```

## Running on Linux / Запуск на Linux

Те же команды работают на Linux. Программы ищутся в `PATH` (`ffmpeg`, `ffprobe`, `yt-dlp`, `deno`); пути внутри проекта — `pathlib`, относительно папки репозитория.

Debian / Ubuntu:

```bash
sudo apt install ffmpeg python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp konveyer.toml.example konveyer.toml
cp .env.example .env
python -m konveyer doctor
```

Для YouTube нужен Deno в `PATH`:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

Откройте новый терминал, чтобы `~/.deno/bin` попал в `PATH`, и снова запустите `python -m konveyer doctor`.

Steam Deck (режим рабочего стола): у SteamOS корень только для чтения, поэтому ставь не в систему, а в distrobox:

```bash
distrobox create -n konveyer -i ubuntu:24.04
distrobox enter konveyer
sudo apt install ffmpeg python3-venv
```

Дальше внутри контейнера — те же шаги, что для Ubuntu. `doctor` должен увидеть `ffmpeg`, `ffprobe` и `yt-dlp`. Надписи берут системный шрифт, если он установлен (DejaVu, Liberation или Noto на SteamOS — обычно в `/usr/share/fonts`), иначе `assets/fonts` (Russo One). Свой шрифт: `[render] font_title` и `[render] font_text` — путь к `.ttf`.

## How this was built and tested / Как это сделано и протестировано

- Я учу Python. Код в основном написан с помощью ИИ.
- Я пользуюсь этим пайплайном каждый день.
- Он стабильно работает по несколько дней подряд.
- Тесты и CI запускаются на каждый пуш: GitHub Actions гоняет ruff и pytest на Ubuntu и Windows.

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
```

## Права

Скачивайте и публикуйте только то, на что у вас есть права. Клонирование голоса (XTTS) — только с согласия владельца голоса;
у модели XTTS своя лицензия (CPML). Шрифты в `assets/fonts` — под SIL OFL (`assets/fonts/OFL.txt`).

## Лицензия

[MIT](LICENSE)
