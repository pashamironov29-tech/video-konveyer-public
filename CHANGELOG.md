# Changelog

## Unreleased

### Security
- Web UI: requests with a non-loopback `Host` header are rejected (DNS rebinding); `[web] host` accepts only
  `127.0.0.1`, `localhost` or `::1`.
- Web UI: a job source must be a link or an existing video file; a local file that ffprobe cannot read is removed
  from the job, so arbitrary files are no longer served through `/files`.
- Web UI: invalid or negative `Content-Length` is rejected; `jobs_dir` is no longer exposed in `/api/status`.

### Fixed
- Two web requests for the same job could both get queued; now the second gets 409.
- Parallel Edge TTS could write the same cache file from two threads; duplicates are synthesized once and WAV files are written atomically.
- Long ffmpeg filter graphs are passed as a file (Windows command line limit).
- Paths with an apostrophe broke `ass=` / `fontfile=` filters.
- Two jobs created in the same second could share a folder.
- `[ui] lang` from `konveyer.toml` now applies to `--help`.
- `konveyer.toml` saved with a UTF-8 BOM (Notepad, PowerShell 5) is read correctly.
- Windows: `job.json` read/replace retries when another thread holds the file.
- Web UI follows the server language again after it changes, instead of a stale browser choice.

### Changed
- A local source file is copied into the job by default; `[ingest] link_local = true` restores the hard link.
- `mediapipe` moved to `requirements-optional.txt`, `pytest` to `requirements-dev.txt`.
- README: real clone URL, ffmpeg/Deno install, Linux and Steam Deck notes, XTTS setup, network usage, security notes.
- Paths stay on `pathlib`. `ffmpeg` / `ffprobe` / `yt-dlp` come from `PATH`. Default overlay fonts are no longer `C:/Windows/Fonts`.
- CI (`.github/workflows/tests.yml`): ruff and pytest on Ubuntu and Windows. Added `SECURITY.md`.

## 1.0.0

- Initial public release: bilingual (RU/EN) video pipeline.
