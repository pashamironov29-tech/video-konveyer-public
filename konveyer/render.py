"""Шаг 5: рендер вертикальных клипов и обложек."""
from __future__ import annotations

import os
import time
from pathlib import Path

from konveyer.captions import (BIG_SIZE, big_anchor, caption_ass, make_big_cues, make_cues,
                               remap_cues, spread_phrase)
from konveyer.config import Settings
from konveyer.facetrack import BLUR_FALLBACK, FaceFinder, face_crop_expr, subject_frac_expr
from konveyer.hardware import Encoder, detect_encoder, encoder_args
from konveyer.jobs import Job
from konveyer.layout import resolve_text_font, build_graph, choose_layout, drawtext, geometry, text_layout
from konveyer.llm import LLM
from konveyer.log import Log
from konveyer.speech import SKIP_CAPTIONS
from konveyer.timeline import append_source_tail, kept_duration, keep_intervals, select_expr
from konveyer.transcript import Transcript, Word
from konveyer.translate import translate_texts
from konveyer.util import KonveyerError, need_tool, read_json, run, write_json
from konveyer.voice import (VoiceOver, align_cues_to_voice, duck_expr, plan_clip_tail,
                            separate_music, voice_gate)


def clip_words(tr: Transcript, start: float, end: float) -> list[Word]:
    """Слова клипа во времени клипа (0 = начало клипа)."""
    out = []
    for w in tr.words_between(start, end):
        a, b = max(w.start, start) - start, min(w.end, end) - start
        if b > a:
            out.append(Word(round(a, 3), round(b, 3), w.word, w.speaker))
    return out


class Renderer:
    def __init__(self, job: Job, settings: Settings, log: Log, tr: Transcript,
                 analysis: dict, llm: LLM | None, encoder: Encoder | None = None,
                 voice: VoiceOver | None = None):
        self.job, self.s, self.log, self.tr, self.analysis, self.llm = job, settings, log, tr, analysis, llm
        r = settings.render
        self.encoder = encoder or detect_encoder(r.encoder, r.crf, r.preset)
        self._tr_cache_file = job.dir / "translations.json"
        self._tr_cache: dict[str, str] = read_json(self._tr_cache_file, {}) or {}
        self.voice = voice
        self._voiced = False
        self._voice_phrases: list = []
        self._finder: FaceFinder | None = None

    def finder(self) -> FaceFinder:
        if self._finder is None:
            self._finder = FaceFinder()
        return self._finder

    def close(self) -> None:
        if self._finder is not None:
            self._finder.close()
            self._finder = None

    def _zoom(self, clip: dict) -> float:
        raw = clip.get("zoom", None)
        if raw in (None, ""):
            return float(self.s.render.zoom)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(self.s.render.zoom)

    # ---------- субтитры
    def _translate(self, texts: list[str]) -> list[str]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._tr_cache]
        if missing:
            done = translate_texts(missing, self.llm, self.log, self.s.paths.subs_dir)
            self._tr_cache.update(zip(missing, done))
            write_json(self._tr_cache_file, self._tr_cache)
        return [self._tr_cache.get(t, t) for t in texts]

    def subtitle_cues(self, words: list[Word], intervals) -> list:
        """Текст и окна субтитров клипа. Озвучка Edge читает эти же строки."""
        if not words:
            return []
        big = self.s.subs.style == "big"
        cues = make_big_cues(words) if big else make_cues(words)
        lang = self.tr.language
        if lang and lang != "ru" and self.s.render.translate and self.llm:
            translated = self._translate([c.text for c in cues])
            if big:
                spread = []
                for c, t in zip(cues, translated):
                    spread.extend(spread_phrase(t, c.start, c.end, speaker=c.speaker) or [c])
                cues = spread
            else:
                for c, t in zip(cues, translated):
                    c.text, c.words = t, []
        if intervals:
            cues = remap_cues(cues, intervals)
        return cues

    def captions(self, words: list[Word], intervals, geo) -> tuple[str | None, str]:
        r = self.s.render
        if not r.captions:
            return None, "выключены"
        if self._voiced and not self.s.voice.subs:
            return None, "убраны (озвучка)"
        if not self.tr.reliable and not self.s.speech.force:
            return None, SKIP_CAPTIONS
        if not words:
            return None, "нет"
        big = self.s.subs.style == "big"
        karaoke = r.karaoke and not big
        lang = self.tr.language
        note = f"оригинал ({lang or '?'})"
        if lang and lang != "ru" and r.translate:
            if self.llm:
                karaoke = False
                note = "перевод на русский"
            else:
                self.log.warn("нет ключа DeepSeek — субтитры останутся на языке оригинала")
        elif lang == "ru":
            note = "русские" + (", подсветка слов" if karaoke else "")
        if big:
            note = "крупные, " + note
        cues = align_cues_to_voice(
            self.subtitle_cues(words, intervals), self._voice_phrases,
            to_voice=self.s.voice.voice_mode == "voiceover")
        who = list(dict.fromkeys(c.speaker for c in cues if c.speaker))
        if who and self.s.subs.speakers:
            note += ", спикеры: " + ", ".join(who)
        anchor = None
        size = r.caption_size
        if big:
            size = max(r.caption_size, BIG_SIZE)
            anchor = big_anchor(geo.W, geo.H, geo.video_y, geo.video_h, size)
        ass = caption_ass(cues, self.s, geo.W, geo.H, karaoke=karaoke,
                          align=geo.caption_align, margin_v=geo.caption_margin_v,
                          style="big" if big else "normal", anchor=anchor, size=size)
        return ass, note

    # ---------- озвучка
    def voice_cues(self, words: list[Word], intervals) -> list:
        """Те же строки, что написаны в субтитрах клипа. Отдельного перевода нет."""
        return self.subtitle_cues(words, intervals)

    def make_voice(self, clip: dict, folder: Path, words, intervals, start: float, dur: float,
                   out_dur: float) -> dict:
        """Озвучка клипа → входы ffmpeg и параметры графа. Пусто — звук как раньше."""
        self._voiced = False
        self._voice_phrases = []
        for old in ("voice.wav", "voice.json", "voice_check.json"):
            (folder / old).unlink(missing_ok=True)
        if self.voice is None or not words:
            return {}
        try:
            track = self.voice.make(folder, self.voice_cues(words, intervals), out_dur, clip["id"])
        except KonveyerError as e:
            self.log.warn(f"клип {clip['id']}: озвучка не получилась ({str(e)[:300]}) — клип без неё")
            track = None
        if track is None:
            return {}
        self._voiced = True
        self._voice_phrases = list(track.phrases)
        inputs = [track.path.name]
        audio_in = "0:a"
        if self.s.voice.separate and self.job.data["meta"].get("has_audio"):
            music = separate_music(self.job.source_video, start, dur, folder, self.s.voice, self.log)
            if music is not None:
                inputs.append(music.name)
                audio_in = "2:a"
        duck = duck_expr(track.intervals, self.s.voice.duck_db, self.s.voice.duck_fade)
        clip["voice"] = track.note + (", голос оригинала убран" if audio_in != "0:a" else "")
        self.log.info(f"клип {clip['id']}: озвучка — {clip['voice']}")
        return {"inputs": inputs, "audio_in": audio_in, "duck": duck, "voice_in": "1:a",
                "voice_end": track.end}

    # ---------- один клип
    def render(self, clip: dict) -> dict:
        meta = self.job.data["meta"]
        r = self.s.render
        src = self.job.source_video
        start, end = float(clip["start"]), float(clip["end"])
        if not 0 <= start < end:
            raise KonveyerError(f"клип {clip['id']}: неверные границы {start}–{end}")
        end = min(end, float(meta["duration"]))
        dur = end - start
        folder = self.job.clip_dir(clip["id"])
        folder.mkdir(parents=True, exist_ok=True)

        layout = choose_layout(clip.get("layout_req") or self.job.data.get("options", {}).get("layout")
                               or r.layout, meta["width"], meta["height"], r.width, r.height)
        zoom = self._zoom(clip)
        if layout == "smart" and not self.finder().available:
            self.log.warn(BLUR_FALLBACK)
            layout = "blur"
        geo = geometry(layout, meta["width"], meta["height"], r.width, r.height, r.caption_size,
                       zoom=zoom)

        words = clip_words(self.tr, start, end)
        intervals = keep_intervals(words, dur, r.pause_sec, r.keep_pad) if r.tighten and words else None
        if intervals and len(intervals) == 1 and intervals[0] == (0.0, dur):
            intervals = None
        out_dur = kept_duration(intervals) if intervals else dur
        clip.pop("voice", None)
        voice = self.make_voice(clip, folder, words, intervals, start, dur, out_dur)
        take, hold = 0.0, 0.0
        if self.s.voice.voice_mode == "voiceover":
            voice_end = float(voice.get("voice_end") or 0.0)
            take, hold = plan_clip_tail(float(meta["duration"]), end, out_dur, voice_end,
                                        tightened=bool(intervals))
        input_dur = dur + take
        out_dur = out_dur + take + hold
        kept = append_source_tail(intervals, dur, take)
        if take or hold:
            how = []
            if take:
                how.append(f"исходник +{take:.1f} с")
            if hold:
                how.append(f"затемнение +{hold:.1f} с")
            self.log.info(f"клип {clip['id']}: закадр длиннее ролика, продлеваю "
                          f"({', '.join(how)})")
        ass, cap_note = self.captions(words, intervals, geo)
        ass_name = None
        if ass:
            ass_name = "captions.ass"
            (folder / ass_name).write_text(ass, encoding="utf-8")

        if r.titles:
            kicker, title, bottom = clip.get("kicker", ""), clip.get("title", ""), clip.get("bottom", "")
        else:
            kicker = title = bottom = ""
        font_title = resolve_text_font(r.font_title, self.s.root, self.log)
        font_text = resolve_text_font(r.font_text, self.s.root, self.log) or font_title
        if not font_title and (kicker or title or bottom):
            self.log.warn("нет ни одного шрифта для надписей (render.font_title, assets/fonts, "
                          "DejaVu/Liberation) — клип без надписей")
            kicker = title = bottom = ""
        items = text_layout(geo, kicker, title, bottom, font_title or "", font_text or "",
                            captions_on=bool(ass), caption_size=r.caption_size)
        for old in folder.glob("t_*.txt"):
            old.unlink()
        filters = []
        for i, item in enumerate(items):
            name = f"t_{i}.txt"
            (folder / name).write_text(item.text, encoding="utf-8")
            filters.append(drawtext(item, name))

        crop_expr = None
        zoom_frac = None
        scenes_rel = [t - start for t in self.analysis.get("scenes", []) if start < t < end]
        if layout == "smart":
            crop_expr, found = face_crop_expr(src, start, end, scenes_rel, meta["width"],
                                              meta["height"], geo.crop_w, finder=self.finder())
            self.log.info(f"слежение за лицом: лицо найдено на {found} кадрах")
        elif zoom > 1.001 and layout in ("blur", "full") and r.zoom_center:
            self.log.info(f"зум {zoom:.1f}: по центру кадра, без слежения")
        elif zoom > 1.001 and layout in ("blur", "full"):
            zoom_frac, where = subject_frac_expr(src, start, end, scenes_rel, meta["width"],
                                                 meta["height"], self.finder())
            self.log.info(f"зум {zoom:.1f}: центр обрезки — {where}")

        graph, has_a = build_graph(geo, fps=r.fps, src_h=meta["height"], text_filters=filters,
                                   ass_file=ass_name,
                                   keep_expr=select_expr(kept) if kept else None,
                                   crop_x_expr=crop_expr, has_audio=bool(meta.get("has_audio")),
                                   blur_sigma=r.blur_sigma, audio_in=voice.get("audio_in", "0:a"),
                                   duck=voice.get("duck", ""), voice_in=voice.get("voice_in", ""),
                                   out_dur=out_dur, zoom_frac=zoom_frac,
                                   pad_video=hold, pad_audio=hold)
        (folder / "filter.txt").write_text(graph, encoding="utf-8")
        out = folder / "clip.mp4"
        part = folder / "clip.part.mp4"
        self._encode(src, start, input_dur, graph, has_a, part, folder, voice.get("inputs", []))
        os.replace(part, out)

        info = {"file": out.relative_to(self.job.dir).as_posix(), "layout": layout,
                "encoder": self.encoder.name, "out_duration": round(out_dur, 2),
                "captions": cap_note, "cover": "", "rendered_at": round(time.time()),
                "zoom": round(zoom, 2)}
        if r.cover:
            cover = folder / "cover.jpg"
            run([need_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{min(1.0, out_dur / 3):.2f}", "-i", out.name, "-frames:v", "1",
                 "-q:v", "2", cover.name], "ffmpeg (обложка)", cwd=folder)
            info["cover"] = cover.relative_to(self.job.dir).as_posix()
        clip.update(info)
        cut = f", паузы вырезаны: {dur:.1f} → {out_dur:.1f} с" if intervals else ""
        self.log.info(f"клип {clip['id']}: {layout}, зум {zoom:.1f}, субтитры: {cap_note}, "
                      f"кодировщик: {self.encoder.title}{cut}")
        return clip

    def _cmd(self, src: Path, start: float, dur: float, graph: str, has_a: bool,
             out: Path, enc: Encoder, extra: list[str] | None = None) -> list[str]:
        cmd = [need_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src.resolve())]
        for name in extra or []:
            cmd += ["-i", name]
        cmd += ["-filter_complex", graph, "-map", "[v]"]
        if has_a:
            cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
        cmd += [*enc.args, "-r", str(self.s.render.fps), "-movflags", "+faststart", out.name]
        return cmd

    def _encode(self, src, start, dur, graph, has_a, out, folder, extra=None) -> None:
        try:
            run(self._cmd(src, start, dur, graph, has_a, out, self.encoder, extra),
                "ffmpeg (рендер клипа)", cwd=folder, timeout=7200)
        except KonveyerError as e:
            if not self.encoder.hardware:
                raise
            self.log.warn(f"{self.encoder.title} не справился, рендерю на процессоре ({str(e)[:200]})")
            r = self.s.render
            self.encoder = Encoder("libx264", encoder_args("libx264", r.crf, r.preset))
            run(self._cmd(src, start, dur, graph, has_a, out, self.encoder, extra),
                "ffmpeg (рендер клипа)", cwd=folder, timeout=7200)


def render_job(job: Job, settings: Settings, log: Log, tr: Transcript, analysis: dict,
               llm: LLM | None, only: list[int] | None = None,
               encoder: Encoder | None = None) -> None:
    clips = job.data.get("clips") or []
    if not clips:
        raise KonveyerError("нет моментов для рендера — сначала шаг «лучшие моменты»")
    if settings.render.captions and not tr.reliable and not settings.speech.force:
        log.info(SKIP_CAPTIONS)
    elif settings.render.captions and not tr.reliable and settings.speech.force:
        log.info("речь ненадёжна — субтитры включены принудительно")
    voice = None
    if settings.voice.enabled:
        why = voice_gate(settings, tr, llm)
        if why:
            log.info(why)
        else:
            log.step("Озвучка")
            voice = VoiceOver(job.dir, settings, log, llm)
    renderer = Renderer(job, settings, log, tr, analysis, llm, encoder, voice)
    log.info(f"кодировщик: {renderer.encoder.title}")
    try:
        for clip in clips:
            if only and clip["id"] not in only:
                continue
            renderer.render(clip)
            job.save()
    finally:
        renderer.close()
