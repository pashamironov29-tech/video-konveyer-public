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

- Windows 10/11, Linux (including Steam Deck) and macOS. Paths go through `pathlib`. `ffmpeg`, `ffprobe` and `yt-dlp` are found on `PATH` — nothing is hardcoded as `.exe`.
- Python 3.12+ (`tomllib` is required)
- `ffmpeg` and `ffprobe` in PATH
- `yt-dlp` (installed from requirements) and **Deno** for YouTube (Windows: `winget install DenoLand.Deno`; Linux: the Deno install script, see below)
- Optional: NVIDIA GPU + CUDA PyTorch, DeepSeek API key, Telegram bot, Hugging Face token (diarization), Ollama (frame check)

## Install

ffmpeg and Deno (Windows):

```bat
winget install Gyan.FFmpeg
winget install DenoLand.Deno
```

Linux (Debian/Ubuntu):

```bash
sudo apt install ffmpeg python3-venv
curl -fsSL https://deno.land/install.sh | sh
```

The project:

```bat
git clone https://github.com/pashamironov29-tech/video-konveyer-public.git video-konveyer
cd video-konveyer
py -m pip install -r requirements.txt
copy konveyer.toml.example konveyer.toml
copy .env.example .env
py -m konveyer doctor
```

On Linux use `python3` instead of `py` and `cp` instead of `copy`. The full Linux and Steam Deck steps are in
[Running on Linux](#running-on-linux--запуск-на-linux). Run every command from the repository folder:
the `konveyer` package is not pip-installed, `python -m konveyer` finds it in the current folder.

`INSTALL.bat` installs the requirements, copies the config templates (if missing) and runs `doctor`. Optional extras:

```bat
py -m pip install -r requirements-optional.txt                              :: mediapipe for layout = "smart"
py -m pip install torch --index-url https://download.pytorch.org/whl/cu128   :: GPU Whisper / Silero
py -m pip install pyannote.audio                                            :: speaker diarization (needs HF_TOKEN)
py -m pip install demucs                                                    :: --separate
```

## Voiceover

`[voice] engine` in `konveyer.toml`; turn it on with `--voice` or `[voice] enabled = true`.

- **edge** (default) — Edge TTS, needs internet, installed from requirements.
- **silero** — offline: needs PyTorch and the `v4_ru.pt` model in `models/silero/` (or `[voice] model_path`).
- **xtts** — XTTS v2 voice clone from `[voice] sample` (10–30 s, only with the speaker's consent).
  coqui-tts does not install on recent Python versions, so give it a separate Python 3.12 environment:

```bash
python3 -m venv xtts-venv
```

Then install coqui-tts into that environment and point `[voice] xtts_python` at its interpreter
(Linux: `xtts-venv/bin/python`, Windows: `xtts-venv/Scripts/python.exe`). On first use the model (~2 GB)
is downloaded from Hugging Face. `python -m konveyer doctor` shows whether XTTS is usable.

## What goes over the network

The pipeline runs on your machine, but some steps call external services:

| Service | What is sent | When |
|---|---|---|
| Source site (via yt-dlp) | the link, your cookies if configured | downloading a link |
| DeepSeek (`api.deepseek.com`) | transcript text, video title | translation, moments, titles, post — if `DEEPSEEK_API_KEY` is set |
| Microsoft Edge TTS | voiceover text | `engine = "edge"` |
| Pollinations (`image.pollinations.ai`) | a short image prompt | `[cover] mode = "ai"` or `"both"` |
| Telegram (`api.telegram.org`) | finished clips, covers, post | `send` / auto-send |
| Hugging Face | downloads Whisper, XTTS, pyannote models (with `HF_TOKEN`) | first run |
| GitHub | the Russo One font if missing from `assets/fonts` | covers |
| Ollama | clip frames | local `[quality] ollama_url` only |

Offline: use a local file instead of a link, no `DEEPSEEK_API_KEY` (heuristic moments, no translation),
`engine = "silero"`, `[cover] mode = "frame"`, no `send`. Download Whisper models in advance.

## Security

- The web UI has no password and listens on this machine only: `[web] host` must be `127.0.0.1`, `localhost`
  or `::1`, other values are rejected. Requests with a foreign `Host` header are refused (DNS rebinding protection).
- The web UI accepts a link or a video file; other files are rejected.
- A local file is copied into the job folder. To save space on big files set `[ingest] link_local = true`
  (a hard link on the same drive; note that editing the job's copy then changes the original too).
- Do not run the web UI and the CLI on the same job at the same time: `job.json` is written atomically,
  but changes from two processes are not merged — the last write wins.
- Reporting vulnerabilities: see [SECURITY.md](SECURITY.md).

## Your keys, cookies and paths

Nothing secret lives in the code or in the config. Put your own values here (all git-ignored):

| What | Where |
|---|---|
| `DEEPSEEK_API_KEY`, `TG_BOT_TOKEN`, `TG_CHAT_ID`, `HF_TOKEN` | `.env` (see `.env.example`) or real environment variables |
| YouTube / Instagram / X cookies | `cookies/youtube_cookies.txt` + `[ingest] cookies_file = "cookies/youtube_cookies.txt"`, or `[ingest] cookies_from_browser = "firefox"` |
| Settings | `konveyer.toml` (copy of `konveyer.toml.example`) |
| Silero model `v4_ru.pt` | `models/silero/v4_ru.pt` or `[voice] model_path` |
| XTTS python | `[voice] xtts_python`: Linux `xtts-venv/bin/python`, Windows `xtts-venv/Scripts/python.exe` |
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

Tests:

```bat
py -m pip install -r requirements-dev.txt
py -m ruff check .
py -m pytest -q
```

## Running on Linux / Запуск на Linux

The same commands work on Linux. Tools are looked up on `PATH` (`ffmpeg`, `ffprobe`, `yt-dlp`, `deno`); paths inside the project are `pathlib` paths, relative to the repository folder.

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

YouTube needs Deno on `PATH`:

```bash
curl -fsSL https://deno.land/install.sh | sh
```

Open a new shell so `~/.deno/bin` is on `PATH`, then run `python -m konveyer doctor` again.

Steam Deck (Desktop Mode): SteamOS keeps the root filesystem read-only, so install into a distrobox instead of the system:

```bash
distrobox create -n konveyer -i ubuntu:24.04
distrobox enter konveyer
sudo apt install ffmpeg python3-venv
```

Inside the container, follow the Ubuntu steps above. `doctor` should show `ffmpeg`, `ffprobe` and `yt-dlp` as found. Overlay text uses a system font if one is installed (DejaVu, Liberation or Noto on SteamOS — usually under `/usr/share/fonts`) and otherwise `assets/fonts` (Russo One). To pick a face, set `[render] font_title` and `[render] font_text` to a `.ttf` path.

## How this was built and tested / Как это сделано и протестировано

- I am learning Python. Most of this code was written with help from AI.
- I use this pipeline every day.
- It keeps running stably for several days in a row.
- Tests and CI run on every push: GitHub Actions runs ruff and pytest on Ubuntu and Windows.

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
```

## Legal

Only download and republish content you have the rights to. Voice cloning (XTTS) only with the speaker's consent;
the XTTS model has its own license (CPML). Fonts in `assets/fonts` are under the SIL OFL (`assets/fonts/OFL.txt`).

## License

[MIT](LICENSE)
