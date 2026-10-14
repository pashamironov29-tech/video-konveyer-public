# Video Konveyer

**English** · [Русский](README.ru.md)

A local video pipeline: give it a link (YouTube, X, Instagram, TikTok, VK…) or a video file and it produces
vertical 9:16 Shorts or a full 16:9 video with translated subtitles, an optional voiceover, covers, a post text,
and (optionally) sends everything to your Telegram bot.

Pipeline steps: **download → transcribe → analyze → best moments → render → cover → post → send**

- **Download** — `yt-dlp` (best quality up to 1080p), or a local file.
- **Transcribe** — `faster-whisper` (GPU if CUDA is available, otherwise CPU), speech-quality guard.
- **Analyze / moments** — scene cuts, loudness, speech density; DeepSeek picks the best moments (or a heuristic without a key).
- **Render** — ffmpeg, layouts `auto | blur | full | smart` (face tracking with mediapipe), zoom, karaoke subtitles,
  per-speaker subtitle colors (pyannote diarization), titles.
- **Translation** — subtitles are translated into Russian via DeepSeek.
- **Voiceover** — Russian voiceover with Edge TTS (default), Silero or XTTS v2 (voice clone), ducking of the original,
  optional demucs voice removal, a Whisper-based voiceover check.
- **Cover & post** — cover from a frame or an AI image (Pollinations), VK post / Shorts descriptions with timestamps and hashtags.
- **Telegram** — send-only bot: clips, covers, post. Large files are recompressed.
- **Web UI** (`python -m konveyer web`) and a full **CLI**. Automatic cleanup of old jobs.

## Interface language (RU / EN)

The interface (CLI help, logs, errors, doctor, web UI) is available in Russian (default) and English.
Choose one of (highest priority first):

1. CLI flag: `python -m konveyer --lang en run <url>`
2. Environment variable: `KONVEYER_LANG=en` (also works from `.env`)
3. Config: `[ui] lang = "en"` in `konveyer.toml`
4. Web UI: the **RU / EN** switch in the top-right corner (remembered in the browser)

The interface language does not change what the pipeline produces: translation and voiceover target Russian as before.

## Requirements

- Windows 10/11 (primary target; Linux/macOS mostly work, but default fonts point to `C:/Windows/Fonts` — change `[render] font_title/font_text`)
- Python 3.12+ (`tomllib` is required)
- `ffmpeg` and `ffprobe` in PATH
- `yt-dlp` (installed from requirements) and **Deno** for YouTube (`winget install DenoLand.Deno`)
- Optional: NVIDIA GPU + CUDA PyTorch, DeepSeek API key, Telegram bot, Hugging Face token (diarization), Ollama (frame check)

## Install

```bat
git clone <this repo> video-konveyer
cd video-konveyer
py -m pip install -r requirements.txt
copy konveyer.toml.example konveyer.toml
copy .env.example .env
py -m konveyer doctor
```

`INSTALL.bat` does the pip install + `doctor` step. Optional extras:

```bat
py -m pip install torch --index-url https://download.pytorch.org/whl/cu128   :: GPU Whisper / Silero
py -m pip install pyannote.audio                                            :: speaker diarization (needs HF_TOKEN)
py -m pip install demucs                                                    :: --separate
```

## Your keys, cookies and paths

Nothing secret lives in the code or in the config. Put your own values here (all git-ignored):

| What | Where |
|---|---|
| `DEEPSEEK_API_KEY`, `TG_BOT_TOKEN`, `TG_CHAT_ID`, `HF_TOKEN` | `.env` (see `.env.example`) or real environment variables |
| YouTube / Instagram / X cookies | `cookies/youtube_cookies.txt` + `[ingest] cookies_file = "cookies/youtube_cookies.txt"`, or `[ingest] cookies_from_browser = "firefox"` |
| Settings | `konveyer.toml` (copy of `konveyer.toml.example`) |
| Silero model `v4_ru.pt` | `models/silero/v4_ru.pt` or `[voice] model_path` |
| XTTS python | `[voice] xtts_python = "xtts-venv/Scripts/python.exe"` |
| Your post prompt / example | `prompts/post_prompt.md`, `prompts/post_example.md` (`[post] style_files`) |
| Your own `subs.py` translator prompt | `[paths] subs_dir` |
| Channel name used in posts | `[post] group`, `hashtags` |

Relative paths are resolved from the project folder. Never commit `.env`, `konveyer.toml` or cookie files.

## Usage

```bat
py -m konveyer run "https://youtu.be/..."             :: Shorts (3 clips by default)
py -m konveyer run video.mp4 --count 5 --voice        :: with Russian voiceover
py -m konveyer full "https://..." --from 1:00 --to 5:00 --voice
py -m konveyer jobs                                   :: list jobs
py -m konveyer show last                              :: job details
py -m konveyer redo last --from render                :: redo from a step
py -m konveyer clip last 2:10-2:50 5:00-5:40          :: Shorts from manual timestamps
py -m konveyer send last                              :: send to Telegram
py -m konveyer web                                    :: web UI at http://127.0.0.1:8765
py -m konveyer cleanup --dry-run
py -m konveyer --lang en --help                       :: all commands and flags
```

Or double-click `RUN.bat` (asks for a link) / `WEB.bat`. Results are in `jobs/<id>/`.

Tests: `py -m pytest -q`.

## Legal

Only download and republish content you have the rights to. Voice cloning (XTTS) only with the speaker's consent;
the XTTS model has its own license (CPML). Fonts in `assets/fonts` are under the SIL OFL (`assets/fonts/OFL.txt`).

## License

[MIT](LICENSE)
