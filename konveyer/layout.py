"""Раскладка вертикального кадра и граф фильтров ffmpeg.

Режимы:
- blur  — исходник целиком по центру, сверху и снизу его размытая копия, тексты на полях
          (мелкая белая строка, белый заголовок, жёлтая подпись);
- full  — кадр заполняет весь экран (для вертикальных исходников из Instagram/TikTok);
- smart — из горизонтального кадра вырезается вертикальное окно, которое следит за лицом.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dataclasses import dataclass

from konveyer.textfit import fit_font_size, split_text_lines
from konveyer.util import ffmpeg_filter_path


def even(x: float) -> int:
    v = int(round(x))
    return v - (v % 2)


@dataclass
class Geometry:
    mode: str
    W: int
    H: int
    video_h: int          # высота картинки в кадре (blur) или H
    video_y: int          # верх картинки в кадре
    crop_w: int = 0       # smart: ширина окна в пикселях исходника
    caption_align: int = 2
    caption_margin_v: int = 400
    zoom: float = 1.0
    fg_w: int = 0         # blur+зум: ширина картинки до обрезки боков; 0 — картинка целиком


@dataclass
class TextItem:
    text: str
    size: int
    color: str
    border: int
    y: int
    font: str


def choose_layout(requested: str, src_w: int, src_h: int, W: int = 1080, H: int = 1920) -> str:
    vertical = src_w / max(src_h, 1) <= 0.8
    if requested == "auto":
        return "full" if vertical else "blur"
    if requested in ("blur", "smart") and vertical:
        return "full"
    if requested == "blur" and even(W * src_h / src_w) > H - 400:
        return "full"
    return requested


def geometry(mode: str, src_w: int, src_h: int, W: int = 1080, H: int = 1920,
             caption_size: int = 74, zoom: float = 1.0) -> Geometry:
    zoom = 1.0 if zoom < 1 else zoom
    if mode == "blur":
        base = min(even(W * src_h / max(src_w, 1)), H)
        if zoom <= 1.001:
            vh, fg_w, z = base, 0, 1.0
        else:
            vh = min(even(base * zoom), H)
            scale = vh / max(base, 1)
            fg_w = even(W * scale)
            if fg_w <= W:
                fg_w = 0
            z = zoom
        vy = even((H - vh) / 2)
        margin = vy + vh + 20
        # субтитры под картинкой не должны уехать за нижний край
        need = int(caption_size * 2.4) + 12
        if H - 8 - margin < need:
            margin = max(8, H - 8 - need)
        return Geometry("blur", W, H, vh, vy, caption_align=8, caption_margin_v=margin,
                        zoom=z, fg_w=fg_w)
    crop_w = 0
    z = zoom if zoom > 1.001 else 1.0
    if mode == "smart":
        crop_w = min(even(src_h * W / H), src_w - (src_w % 2))
        if z > 1 and crop_w > 0:
            crop_w = min(max(even(crop_w / z), 64), src_w - (src_w % 2))
    return Geometry(mode, W, H, H, 0, crop_w=crop_w, caption_align=2,
                    caption_margin_v=int(H * 0.22), zoom=z)


def _fit_block(lines: list[str], max_size: int, room: int, width: int) -> tuple[int, int, int]:
    """(размер, промежуток, высота блока) для строк, чтобы влезли в room по высоте."""
    if not lines:
        return 0, 0, 0
    size = min(fit_font_size(line, width, max_size, 0.70) for line in lines)
    gap = max(8, size // 10)
    block = len(lines) * size + (len(lines) - 1) * gap
    while block > room and size > 36:
        size -= 2
        gap = max(8, size // 10)
        block = len(lines) * size + (len(lines) - 1) * gap
    return size, gap, block


# Каталоги дистрибутивов, не личные пути. Debian/Ubuntu, Arch и SteamOS (Steam Deck), macOS.
SYSTEM_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/NotoSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def font_candidates() -> list[Path]:
    """Системные шрифты текущей ОС. На Windows — из %WINDIR%\\Fonts, без зашитого диска."""
    paths: list[Path] = []
    if sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        paths.extend(fonts / name for name in ("ariblk.ttf", "arialbd.ttf", "arial.ttf"))
    paths.extend(Path(item) for item in SYSTEM_FONTS)
    return paths


def _bundled_fonts(root: Path) -> list[Path]:
    fonts_dir = root / "assets" / "fonts"
    if not fonts_dir.is_dir():
        return []
    return sorted(fonts_dir.glob("*.ttf")) + sorted(fonts_dir.glob("*.otf"))


def resolve_text_font(configured: str, root: Path, log=None, label: str = "") -> str | None:
    """Шрифт для drawtext. Пустой конфиг: системный шрифт, затем assets/fonts.
    Заданный, но отсутствующий файл: assets/fonts, затем система.
    None — шрифта нет нигде (надписи тогда выключаются, рендер не падает)."""
    custom = (configured or "").strip()
    bundled = _bundled_fonts(root)
    system = font_candidates()
    if custom:
        path = Path(custom)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return str(path)
        order = [*bundled, *system]
    else:
        order = [*system, *bundled]
    for cand in order:
        if cand.is_file():
            if custom and log:
                log.warn(f"шрифт надписей не найден ({custom}), беру {cand.name}")
            return str(cand)
    return None


def text_layout(geo: Geometry, kicker: str, title: str, bottom: str, font_title: str,
                font_text: str, captions_on: bool = True, caption_size: int = 74) -> list[TextItem]:
    W, H = geo.W, geo.H
    width = W - 120
    items: list[TextItem] = []
    title_lines = split_text_lines(title, 2, 18)
    bottom_lines = split_text_lines(bottom, 2, 18)
    kicker = (kicker or "").strip()[:48]

    if geo.mode == "blur":
        top_room_end = geo.video_y - 40
        y = 64
        # при зуме верхняя полоса ниже, но кикер всё равно ставим, если влезает строка
        if kicker and top_room_end - y >= 48:
            ks = fit_font_size(kicker, width, 42, 0.55)
            items.append(TextItem(kicker, ks, "white", 3, y, font_text))
            y += ks + 36
        size, gap, block = _fit_block(title_lines, 118, top_room_end - y, width)
        yy = y + max(0, (top_room_end - y - block) // 2)
        for line in title_lines:
            items.append(TextItem(line, size, "white", max(6, size // 12), yy, font_title))
            yy += size + gap
        room_top = geo.video_y + geo.video_h + 40
        if captions_on:
            room_top += int(caption_size * 2.5) + 10
        room_bot = H - 48
        size, gap, block = _fit_block(bottom_lines, 108, room_bot - room_top, width)
        yy = room_top + max(0, (room_bot - room_top - block) // 2)
        for line in bottom_lines:
            items.append(TextItem(line, size, "yellow", max(6, size // 12), yy, font_title))
            yy += size + gap
        for it in items:
            if it.y < 4:
                it.y = 4
            over = it.y + it.size - (H - 4)
            if over > 0:
                it.y = max(4, it.y - over)
        return items

    # full / smart: всё сверху стопкой, ниже — субтитры у нижней трети
    y = 110
    limit = int(H * 0.40)
    if kicker:
        ks = fit_font_size(kicker, width, 44, 0.55)
        items.append(TextItem(kicker, ks, "white", 4, y, font_text))
        y += ks + 24
    size, gap, block = _fit_block(title_lines, 104, limit - y, width)
    for line in title_lines:
        items.append(TextItem(line, size, "white", max(7, size // 10), y, font_title))
        y += size + gap
    if title_lines:
        y += 16
    size, gap, block = _fit_block(bottom_lines, 88, max(limit + 120 - y, 80), width)
    for line in bottom_lines:
        items.append(TextItem(line, size, "yellow", max(7, size // 10), y, font_title))
        y += size + gap
    for it in items:
        if it.y < 4:
            it.y = 4
        over = it.y + it.size - (H - 4)
        if over > 0:
            it.y = max(4, it.y - over)
    return items


def drawtext(item: TextItem, textfile: str) -> str:
    return ("drawtext="
            f"fontfile='{ffmpeg_filter_path(item.font)}':textfile='{textfile}':expansion=none:"
            f"fontcolor={item.color}:fontsize={item.size}:borderw={item.border}:bordercolor=black:"
            f"shadowx=2:shadowy=2:shadowcolor=black@0.6:x=(w-text_w)/2:y={item.y}")


def _zoom_x(frac: str, out_w: int) -> str:
    """Горизонталь обрезки: доля кадра → чётный x. Дроби в одинарных кавычках, запятые живы."""
    return f"'trunc(clip(({frac})*iw-{out_w}/2,0,iw-{out_w})/2)*2'"


# Затемнение хвоста, если исходник кончился раньше голоса. Короткий фейд, не залипший кадр.
HOLD_FADE_SEC = 0.8
# Оригинал на продлении затухает, а не обрывается в тишину посередине русской фразы.
SRC_FADE_SEC = 0.55


def _soft_hold(pad_video: float, out_dur: float) -> str:
    """Кадры после конца исходника: размытие и короткий фейд в чёрный, субтитры ещё впереди."""
    if pad_video <= 0.001:
        return ""
    tail_at = max(out_dur - pad_video, 0.0) if out_dur > 0 else 0.0
    fade_d = min(HOLD_FADE_SEC, pad_video)
    blur_at = max(0.0, tail_at - 0.12) if out_dur > 0 else 0.0
    return (
        f",tpad=stop_mode=clone:stop_duration={pad_video:.3f}"
        f",gblur=sigma=16:enable='gte(t,{blur_at:.3f})'"
        f",fade=t=out:st={tail_at:.3f}:d={fade_d:.3f}:color=black"
    )


def _fade_source(chain: str, pad_audio: float, out_dur: float) -> str:
    """Плавный уход оригинала в тишину к началу затемнения."""
    if pad_audio <= 0.001 or out_dur <= 0:
        return chain
    real_end = out_dur - pad_audio
    if real_end <= 0.05:
        return chain
    fade_d = min(SRC_FADE_SEC, real_end)
    st = max(0.0, real_end - fade_d)
    return chain + f",afade=t=out:st={st:.3f}:d={fade_d:.3f}"


def build_graph(geo: Geometry, *, fps: int, src_h: int, text_filters: list[str],
                ass_file: str | None, keep_expr: str | None, crop_x_expr: str | None,
                has_audio: bool, blur_sigma: float = 6.0, audio_in: str = "0:a",
                duck: str = "", voice_in: str = "", out_dur: float = 0.0,
                zoom_frac: str | None = None, pad_video: float = 0.0,
                pad_audio: float = 0.0) -> tuple[str, bool]:
    """Граф для -filter_complex. Возвращает (граф, есть ли [a]).

    voice_in — вход с готовой дорожкой озвучки (она уже во времени клипа, паузы не режем),
    duck — выражение громкости оригинала под озвучкой, audio_in — откуда оригинальный звук
    (0:a или музыка без голоса после demucs).
    pad_video — мягкий хвост, если закадр длиннее исходника: последний кадр размывается
    и уходит в чёрный. pad_audio — тишина оригинала на ту же длину, с фейдом перед ней,
    чтобы микс не обрезал голос и английский не щёлкал в ноль.
    """
    W, H = geo.W, geo.H
    pre = f"[0:v]fps={fps},setsar=1"
    if geo.mode == "smart":
        x = f"'{crop_x_expr}'" if crop_x_expr else f"(iw-{geo.crop_w})/2"
        pre += f",crop={geo.crop_w}:{src_h - (src_h % 2)}:{x}:0"
    if keep_expr:
        pre += f",select='{keep_expr}',setpts=N/({fps}*TB)"
    post = _soft_hold(pad_video, out_dur)
    post += "".join("," + f for f in text_filters)
    if ass_file:
        post += f",ass='{ass_file}'"

    if geo.mode == "blur":
        bw, bh = even(W / 4), even(H / 4)
        if geo.fg_w > W:
            x = _zoom_x(zoom_frac or "0.5", W)
            fg = (f"[fg0]scale={geo.fg_w}:{geo.video_h}:flags=lanczos,"
                  f"crop={W}:{geo.video_h}:{x}:0,setsar=1[fg]")
        else:
            fg = f"[fg0]scale={W}:{geo.video_h}:flags=lanczos,setsar=1[fg]"
        graph = (f"{pre},split=2[bg0][fg0];"
                 f"[bg0]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                 f"gblur=sigma={blur_sigma:g},scale={W}:{H},setsar=1[bg];"
                 f"{fg};"
                 f"[bg][fg]overlay=0:{geo.video_y}{post}[v]")
    elif geo.mode == "full" and geo.zoom > 1.001:
        zw, zh = even(W * geo.zoom), even(H * geo.zoom)
        x = _zoom_x(zoom_frac or "0.5", W)
        graph = (f"{pre},scale={zw}:{zh}:force_original_aspect_ratio=increase:flags=lanczos,"
                 f"crop={W}:{H}:{x}:(ih-{H})/2,setsar=1{post}[v]")
    else:
        graph = (f"{pre},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                 f"crop={W}:{H},setsar=1{post}[v]")
    if has_audio:
        chain = f"[{audio_in}]"
        if keep_expr:
            chain += f"aselect='{keep_expr}',asetpts=N/SR/TB,"
        chain += "aresample=48000"
        if duck:
            chain += f",volume='{duck}':eval=frame"
        chain = _fade_source(chain, pad_audio, out_dur)
        if pad_audio > 0.001:
            chain += f",apad=pad_dur={pad_audio:.3f}"
        if voice_in:
            mix = "longest" if pad_audio > 0.001 or pad_video > 0.001 else "first"
            graph += (f";{chain}[a0];[{voice_in}]aresample=48000[vo];"
                      f"[a0][vo]amix=inputs=2:duration={mix}:dropout_transition=0:normalize=0[a]")
        else:
            graph += f";{chain}[a]"
    elif voice_in:
        graph += f";[{voice_in}]aresample=48000,apad=whole_dur={out_dur:.3f}[a]"
        has_audio = True
    return graph, has_audio
