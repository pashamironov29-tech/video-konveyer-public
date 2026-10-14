"""Озвучка XTTS v2 в отдельном python (venv с coqui-tts). konveyer здесь не импортируется.

Запуск: python xtts_worker.py request.json
request.json: {"model": ..., "sample": "голос.wav", "language": "ru",
               "items": [{"text": "...", "out": "путь.wav"}, ...]}
Лицензия модели (CPML) принимается заранее: setx COQUI_TOS_AGREED 1.
"""
import json
import sys


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        req = json.load(f)
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(req["model"]).to(device)
    for item in req["items"]:
        try:
            tts.tts_to_file(text=item["text"], speaker_wav=req["sample"],
                            language=req.get("language", "ru"), file_path=item["out"])
        except Exception as e:
            print(f"не озвучено: {item['text'][:40]}: {e}", file=sys.stderr)
    print(f"ok {device}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
