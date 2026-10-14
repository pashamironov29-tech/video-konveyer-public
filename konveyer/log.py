"""Журнал: в консоль и в job.log, всё через маскировку секретов."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from konveyer.envkeys import mask
from konveyer.i18n import get_lang, tr


class Log:
    def __init__(self, file: Path | None = None, echo: bool = True):
        self.file = file
        self.echo = echo
        self._lock = threading.Lock()

    def _write(self, line: str) -> None:
        line = mask(line)
        with self._lock:
            if self.echo:
                try:
                    print(line, flush=True)
                except UnicodeEncodeError:
                    enc = sys.stdout.encoding or "utf-8"
                    print(line.encode(enc, "replace").decode(enc), flush=True)
            if self.file:
                self.file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.file, "a", encoding="utf-8") as f:
                    f.write(time.strftime("%H:%M:%S ") + line + "\n")

    def step(self, msg: str) -> None:
        self._write(f"\n==> {tr(msg)}")

    def info(self, msg: str) -> None:
        self._write(f"    {tr(msg)}")

    def warn(self, msg: str) -> None:
        self._write(f"    {'WARNING' if get_lang() == 'en' else 'ВНИМАНИЕ'}: {tr(msg)}")

    def child(self, file: Path) -> "Log":
        return Log(file, self.echo)


def tail(file: Path, lines: int = 80) -> str:
    try:
        text = file.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(text.splitlines()[-lines:])
