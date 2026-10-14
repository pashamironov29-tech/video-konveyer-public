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

- Windows 10/11 (основная платформа; Linux/macOS в целом работают, но шрифты по умолчанию — `C:/Windows/Fonts`, поменяйте `[render] font_title/font_text`)
- Python 3.12+
- `ffmpeg` и `ffprobe` в PATH
- `yt-dlp` (ставится из requirements) и **Deno** для YouTube (`winget install DenoLand.Deno`)
- По желанию: видеокарта NVIDIA + PyTorch с CUDA, ключ DeepSeek, Telegram-бот, токен Hugging Face (диаризация), Ollama (проверка кадров)

## Установка

```bat
git clone <этот репозиторий> video-konveyer
cd video-konveyer
py -m pip install -r requirements.txt
copy konveyer.toml.example konveyer.toml
copy .env.example .env
py -m konveyer doctor
```

`INSTALL.bat` делает установку и `doctor`. Дополнительно:

```bat
py -m pip install torch --index-url https://download.pytorch.org/whl/cu128   :: Whisper/Silero на видеокарте
py -m pip install pyannote.audio                                            :: диаризация (нужен HF_TOKEN)
py -m pip install demucs                                                    :: --separate
```

## Куда положить свои ключи, cookies и пути

В коде и конфиге секретов нет. Свои значения кладите сюда (всё в .gitignore):

| Что | Куда |
|---|---|
| `DEEPSEEK_API_KEY`, `TG_BOT_TOKEN`, `TG_CHAT_ID`, `HF_TOKEN` | `.env` (образец — `.env.example`) или переменные окружения |
| cookies YouTube / Instagram / X | `cookies/youtube_cookies.txt` + `[ingest] cookies_file = "cookies/youtube_cookies.txt"` или `[ingest] cookies_from_browser = "firefox"` |
| Настройки | `konveyer.toml` (копия `konveyer.toml.example`) |
| Модель Silero `v4_ru.pt` | `models/silero/v4_ru.pt` или `[voice] model_path` |
| python для XTTS | `[voice] xtts_python = "xtts-venv/Scripts/python.exe"` |
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

Тесты: `py -m pytest -q`.

## Права

Скачивайте и публикуйте только то, на что у вас есть права. Клонирование голоса (XTTS) — только с согласия владельца голоса;
у модели XTTS своя лицензия (CPML). Шрифты в `assets/fonts` — под SIL OFL (`assets/fonts/OFL.txt`).

## Лицензия

[MIT](LICENSE)
