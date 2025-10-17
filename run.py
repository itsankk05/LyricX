import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame
import librosa
import vlc  # VLC backend for streaming playback
from yt_dlp import YoutubeDL  # resolve YouTube URLs and playlists

try:
    import syncedlyrics  # for fetching time-synced lyrics (LRC)
except Exception:
    syncedlyrics = None

# -----------------------------
# Config
# -----------------------------
LYRICS_FILE = "lyrics.txt"  # put your lyrics here (one block, as you pasted)
AUDIO_FILE = "something_just_like_this.mp3"  # your audio file
FPS = 60  # render FPS
FONT_NAME = "Consolas"  # monospaced font recommended
FONT_SIZE = 28
WINDOW_W = 1280
WINDOW_H = 720
MARGIN_X = 40
MARGIN_Y = 40
COL_GAP = 2  # px between chars horizontally
ROW_GAP = 6  # px between lines vertically
BG_COLOR = (0, 0, 0)
CHAR_COLOR = (30, 34, 40)  # near-invisible base so glow pops
GLOW_COLOR = (255, 255, 255)  # max glow color
WORD_GLOW_BOOST = 0.9  # stronger whole-word pulses on beats
LETTER_GLOW_NOISE = 0.45  # stronger per-letter noise intensity
LETTER_NOISE_PROB = 0.08  # per-frame noise probability per visible char
DECAY_PER_SEC = 1.4  # slower decay so glow is clearly visible
SHOW_NOISE = False  # if True, per-letter noise twinkles
SYNC_LEAD_SEC = 0.15  # draw slightly ahead of audio to counter lag
MIN_FONT_SIZE = 12  # minimum font size when fitting grid
MAX_FONT_SIZE = 96  # maximum font size when fitting grid
SEED = 42  # reproducibility


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class CharCell:
    ch: str
    line_idx: int
    char_idx: int
    rect: pygame.Rect
    intensity: float = 0.0  # 0..1 glow
    is_space: bool = False
    word_id: int = -1


def load_lyrics(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        block = f.read().strip("\n")
    # Normalize newlines; keep blank lines to preserve stanza breaks
    lines = block.splitlines()
    return lines


def tokenize_words(lines: List[str]) -> List[List[Tuple[str, Tuple[int, int]]]]:
    """
    Returns per-line list of (word, (start_char_index, end_char_index_inclusive))
    Spaces/punctuation stay separated only by char grid; here we identify words for beat pulses.
    """
    out = []
    for line in lines:
        words = []
        start = 0
        i = 0
        while i < len(line):
            if line[i].isspace():
                i += 1
                start = i
                continue
            # read until next space
            j = i
            while j < len(line) and not line[j].isspace():
                j += 1
            words.append((line[i:j], (i, j - 1)))
            i = j
            start = i
        out.append(words)
    return out


def tokenize_words_linear(lines: List[str]) -> List[str]:
    """
    Flatten lines into a sequence of words (split by whitespace), preserving order.
    Empty tokens are skipped.
    """
    words: List[str] = []
    for line in lines:
        for w in line.split():
            if w:
                words.append(w)
    return words


def compute_grid_positions(
    lines: List[str], font, max_width_px: int
) -> Tuple[List[List[CharCell]], int, int]:
    """
    Original per-line layout (retained for reference, not used by default).
    """
    # measure a typical character
    char_w, char_h = font.size("M")
    line_h = char_h + ROW_GAP
    char_w_eff = char_w + COL_GAP

    grid: List[List[CharCell]] = []
    y = MARGIN_Y
    word_maps = tokenize_words(lines)
    global_word_id = 0

    for li, line in enumerate(lines):
        x = MARGIN_X
        row: List[CharCell] = []

        # map char indices -> word id
        char_to_word = {}
        for word, (s, e) in word_maps[li]:
            for k in range(s, e + 1):
                char_to_word[k] = global_word_id
            global_word_id += 1

        for ci, ch in enumerate(line):
            rect = pygame.Rect(x, y, char_w, char_h)
            row.append(
                CharCell(
                    ch=ch,
                    line_idx=li,
                    char_idx=ci,
                    rect=rect,
                    intensity=0.0,
                    is_space=ch.isspace(),
                    word_id=char_to_word.get(ci, -1),
                )
            )
            x += char_w_eff

        grid.append(row)
        y += line_h

    return grid, char_w, line_h


def _tokenize_words_flat(text: str) -> Tuple[dict, int]:
    """
    Build a mapping from absolute char index to word_id for a single string.
    Returns (char_to_word, next_word_id).
    """
    char_to_word = {}
    word_id = -1
    i = 0
    next_word_id = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        word_id = next_word_id
        for k in range(i, j):
            char_to_word[k] = word_id
        next_word_id += 1
        i = j
    return char_to_word, next_word_id


def build_word_matrix_grid(
    lines: List[str], font, words_seq: List[str]
) -> Tuple[List[List[CharCell]], int, int]:
    """
    Lay out WHOLE WORDS across a 2D grid sized to the current window.
    Ensures every word from words_seq appears at least once (shuffled order).
    If extra capacity remains, fills with spaces.
    """
    # Use current window size instead of static constants
    surf = pygame.display.get_surface()
    win_w, win_h = surf.get_size() if surf else (WINDOW_W, WINDOW_H)

    char_w, char_h = font.size("M")
    line_h = char_h + ROW_GAP
    char_w_eff = char_w + COL_GAP

    cols = max(1, (win_w - 2 * MARGIN_X) // char_w_eff)
    rows = max(1, (win_h - 2 * MARGIN_Y) // line_h)

    order = list(range(len(words_seq)))
    random.shuffle(order)
    order_idx = 0

    grid: List[List[CharCell]] = []
    for r in range(rows):
        y = MARGIN_Y + r * line_h
        row: List[CharCell] = []
        c = 0
        while c < cols and order_idx < len(order):
            widx = order[order_idx]
            w = words_seq[widx]
            wlen = len(w)
            # If word does not fit in remaining columns, wrap to next row
            if wlen > (cols - c):
                break
            # Place the word
            for k in range(wlen):
                x = MARGIN_X + c * char_w_eff
                row.append(
                    CharCell(
                        ch=w[k],
                        line_idx=r,
                        char_idx=c,
                        rect=pygame.Rect(x, y, char_w, char_h),
                        intensity=0.0,
                        is_space=False,
                        word_id=widx,
                    )
                )
                c += 1
            # Add a space after each word if there is room
            if c < cols:
                x = MARGIN_X + c * char_w_eff
                row.append(
                    CharCell(
                        ch=" ",
                        line_idx=r,
                        char_idx=c,
                        rect=pygame.Rect(x, y, char_w, char_h),
                        intensity=0.0,
                        is_space=True,
                        word_id=-1,
                    )
                )
                c += 1
            order_idx += 1
        # Pad any remaining columns with spaces
        while c < cols:
            x = MARGIN_X + c * char_w_eff
            row.append(
                CharCell(
                    ch=" ",
                    line_idx=r,
                    char_idx=c,
                    rect=pygame.Rect(x, y, char_w, char_h),
                    intensity=0.0,
                    is_space=True,
                    word_id=-1,
                )
            )
            c += 1
        grid.append(row)

    return grid, char_w, line_h


def _parse_lrc(lrc_text: str) -> List[Tuple[float, str]]:
    """
    Parse LRC text into a sorted list of (timestamp_sec, line_text).
    Handles multiple timestamps per line.
    """
    import re

    ts_re = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
    entries: List[Tuple[float, str]] = []
    for raw in lrc_text.splitlines():
        times = [(int(m.group(1)), float(m.group(2))) for m in ts_re.finditer(raw)]
        if not times:
            continue
        # text with timestamps stripped
        text = ts_re.sub("", raw).strip()
        for mm, ss in times:
            t = mm * 60 + ss
            entries.append((t, text))
    entries.sort(key=lambda x: x[0])
    return entries


def build_word_schedule_from_lrc(
    audio_path: str, lrc_text: str, track_duration: float | None = None
) -> Tuple[List["WordSlot"], List[str]]:
    """
    Convert LRC into word-level schedule by dividing each line's duration across its words.
    Returns (schedule, words_seq) where words_seq is the flattened list of words in LRC order.
    """
    lrc_entries = _parse_lrc(lrc_text)
    # If duration unknown, we’ll infer line durations from next timestamps
    track_dur = track_duration if track_duration is not None else 0.0
    schedule: List[WordSlot] = []
    words_seq: List[str] = []
    if not lrc_entries:
        return schedule, words_seq

    # compute per-line start/end
    starts = [t for t, _ in lrc_entries]
    texts = [txt for _, txt in lrc_entries]
    ends = starts[1:] + [
        track_dur if track_dur and track_dur > 0 else (starts[-1] + 2.0)
    ]

    for start, end, line_text in zip(starts, ends, texts):
        dur = max(0.15, end - start)
        # tokenize words in this line
        ws = [w for w in line_text.split() if w]
        if not ws:
            continue
        step = dur / max(1, len(ws))
        for j, w in enumerate(ws):
            ws_start = start + j * step
            ws_end = start + (j + 1) * step
            idx = len(words_seq)
            schedule.append(WordSlot(start=ws_start, end=ws_end, word_idx=idx))
            words_seq.append(w)

    return schedule, words_seq


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


# -----------------------------
# Beat scheduling
# -----------------------------
@dataclass
class BeatTimeline:
    beats: np.ndarray  # times (sec)
    idx: int = 0


def analyze_beats(audio_path: str, sr: int = 44100) -> BeatTimeline:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=True)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return BeatTimeline(beats=beat_times, idx=0)


@dataclass
class WordSlot:
    start: float
    end: float
    word_idx: int


def build_word_schedule(
    audio_path: str, words_seq: List[str], sr: int = 44100, top_db: int = 35
) -> Tuple[List[WordSlot], float]:
    """
    Build a time schedule for lyric words using non-silent intervals of the audio.
    Words are distributed across active (non-silent) segments in order, so during
    gaps no word advances. Returns (schedule, duration_sec).
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = float(len(y)) / sr if len(y) else 0.0
    schedule: List[WordSlot] = []
    if not words_seq:
        return schedule, duration

    # Detect non-silent intervals (in samples), convert to seconds
    intervals = librosa.effects.split(y, top_db=top_db)
    if intervals.size == 0:
        intervals = np.array([[0, len(y)]], dtype=int)
    segs = [(s / sr, e / sr) for s, e in intervals]
    seg_durs = [max(0.0, e - s) for s, e in segs]
    total_active = sum(seg_durs)
    if total_active <= 0.0:
        segs = [(0.0, duration)]
        seg_durs = [duration]
        total_active = duration

    n_words = len(words_seq)
    # Initial proportional allocation
    alloc = [int(round(n_words * (d / total_active))) for d in seg_durs]
    diff = n_words - sum(alloc)
    # Fix rounding so sum equals n_words
    if diff != 0:
        # Distribute remainder by giving/taking from longest segments first
        order = sorted(range(len(segs)), key=lambda i: seg_durs[i], reverse=True)
        i = 0
        while diff != 0 and order:
            idx = order[i % len(order)]
            if diff > 0:
                alloc[idx] += 1
                diff -= 1
            else:
                if alloc[idx] > 0:
                    alloc[idx] -= 1
                    diff += 1
            i += 1

    # Build slots spaced evenly within each segment
    w = 0
    for (s, e), k in zip(segs, alloc):
        if k <= 0:
            continue
        seg_len = max(1e-6, e - s)
        step = seg_len / k
        for j in range(k):
            if w >= n_words:
                break
            ws = s + j * step
            we = s + (j + 1) * step
            schedule.append(WordSlot(start=ws, end=we, word_idx=w))
            w += 1

    # If we still have leftover words (due to alloc rounding), place them at the end of last seg
    while w < n_words and segs:
        s, e = segs[-1]
        t = e + (w - n_words) * 0.25
        schedule.append(WordSlot(start=max(0.0, t), end=max(0.0, t + 0.2), word_idx=w))
        w += 1

    # Ensure schedule sorted by time
    schedule.sort(key=lambda sl: sl.start)
    return schedule, duration


def build_uniform_schedule(total_dur: float, words_seq: List[str]) -> List[WordSlot]:
    """
    Build a simple uniform schedule across total_dur for the given words.
    Useful when streaming (no local audio file to analyze for silence).
    """
    schedule: List[WordSlot] = []
    if total_dur <= 0 or not words_seq:
        return schedule
    n = len(words_seq)
    step = max(0.15, total_dur / max(1, n))
    t = 0.0
    for i in range(n):
        start = t
        end = min(total_dur, t + step)
        schedule.append(WordSlot(start=start, end=end, word_idx=i))
        t += step
    return schedule


def _clean_title_for_query(title: str) -> str:
    """Clean common noise from a YouTube title for better lyric search."""
    t = title or ""
    lowers = [
        "(official video)",
        "(lyrics)",
        "[lyrics]",
        "(audio)",
        "(live)",
        "official",
        "lyrics",
        "video",
    ]
    tt = t
    for frag in lowers:
        tt = tt.replace(frag, " ")
        tt = tt.replace(frag.title(), " ")
        tt = tt.replace(frag.upper(), " ")
    # Drop bracketed annotations
    import re

    tt = re.sub(r"\([^)]*\)", " ", tt)
    tt = re.sub(r"\[[^\]]*\]", " ", tt)
    tt = re.sub(r"\s+", " ", tt).strip()
    return tt


# -----------------------------
# Main app
# -----------------------------
def main():
    if not os.path.exists(LYRICS_FILE):
        print(f"Missing {LYRICS_FILE}. Create it and paste your lyrics.")
        sys.exit(1)
    # Local audio file is optional now (streaming or user input may be used)
    if not os.path.exists(AUDIO_FILE):
        print(f"Note: {AUDIO_FILE} not found. Local file playback disabled.")

    random.seed(SEED)
    np.random.seed(SEED)

    lines = load_lyrics(LYRICS_FILE)
    words_seq: List[str] = []
    schedule: List[WordSlot] = []

    # Try to fetch synced lyrics via syncedlyrics
    lrc_text = None
    if syncedlyrics is not None:
        try:
            # Try using the audio file to infer metadata; if that fails, try a simple query from first non-empty line
            lrc_text = syncedlyrics.search(AUDIO_FILE)
            if not lrc_text and lines:
                # Use first 7 words as a query hint
                first_line = next((ln for ln in lines if ln.strip()), "")
                hint = " ".join(first_line.split()[:7])
                if hint:
                    lrc_text = syncedlyrics.search(hint)
        except Exception:
            lrc_text = None
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
    pygame.display.set_caption("Lyrics Matrix - Word Sync")
    clock = pygame.time.Clock()

    # Font + layout helpers
    def make_font(sz: int):
        try:
            return pygame.font.SysFont(FONT_NAME, sz)
        except Exception:
            return pygame.font.Font(None, sz)

    def rebuild_layout():
        nonlocal font, grid, char_w, line_h, glyph_cache
        win_w, win_h = screen.get_size()
        max_word_len = max((len(w) for w in words_seq), default=1)
        total_chars = max(
            1, sum(len(w) for w in words_seq) + max(0, len(words_seq) - 1)
        )
        # Find the largest font size that still fits all words on screen
        best_fs = None
        for fs in range(MIN_FONT_SIZE, MAX_FONT_SIZE + 1):
            f = make_font(fs)
            cw, ch = f.size("M")
            lh = ch + ROW_GAP
            cwe = cw + COL_GAP
            cols = max(1, (win_w - 2 * MARGIN_X) // cwe)
            rows = max(1, (win_h - 2 * MARGIN_Y) // lh)
            capacity = cols * rows
            if cols >= max_word_len and capacity >= total_chars:
                best_fs = (fs, f, cw, lh)
        if best_fs is None:
            # Nothing fits; fall back to minimum size
            fs = MIN_FONT_SIZE
            f = make_font(fs)
            cw, ch = f.size("M")
            lh = ch + ROW_GAP
            font = f
            char_w, line_h = cw, lh
        else:
            fs, f, cw, lh = best_fs
            font = f
            char_w, line_h = cw, lh
        # Build matrix covering all words once
        grid, char_w, line_h = build_word_matrix_grid(lines, font, words_seq)
        glyph_cache = {}

    # Prefer LRC-derived schedule and words when available; otherwise fallback
    if lrc_text:
        schedule, words_seq = build_word_schedule_from_lrc(AUDIO_FILE, lrc_text)
        if not words_seq:
            words_seq = tokenize_words_linear(lines)
            schedule, _ = build_word_schedule(AUDIO_FILE, words_seq)
    else:
        words_seq = tokenize_words_linear(lines)
        schedule, _ = build_word_schedule(AUDIO_FILE, words_seq)

    # Build a full-screen 2D word matrix from selected words sequence (fit all words)
    font = make_font(FONT_SIZE)
    grid: List[List[CharCell]]
    char_w = 0
    line_h = 0
    glyph_cache = {}
    rebuild_layout()

    sched_idx = 0
    is_borderless_fs = False

    # Pre-render surfaces for characters to speed up draw
    # ---- Input screen and streaming setup ----
    input_active = True
    user_input = ""
    font_ui = pygame.font.SysFont(FONT_NAME, 28)
    info_lines = [
        "Paste a YouTube URL or type a search query, then press Enter.",
        "Press F11 for borderless full-screen.",
    ]

    def draw_input_screen():
        screen.fill(BG_COLOR)
        y = MARGIN_Y
        for t in info_lines:
            surf = font_ui.render(t, True, (180, 190, 200))
            screen.blit(surf, (MARGIN_X, y))
            y += surf.get_height() + 8
        prompt = "> " + user_input
        surf = font_ui.render(prompt, True, (240, 240, 240))
        screen.blit(surf, (MARGIN_X, y + 8))
        pygame.display.flip()

    def draw_progress_screen(message: str, current: int, total: int):
        screen.fill(BG_COLOR)
        # Title
        title = "Preparing your songs…"
        s1 = font_ui.render(title, True, (200, 210, 220))
        screen.blit(s1, (MARGIN_X, MARGIN_Y))
        # Message
        s2 = font_ui.render(message, True, (180, 190, 200))
        screen.blit(s2, (MARGIN_X, MARGIN_Y + s1.get_height() + 10))
        # Progress bar
        bar_w = max(100, screen.get_width() - 2 * MARGIN_X)
        bar_h = 18
        x0 = MARGIN_X
        y0 = MARGIN_Y + s1.get_height() + s2.get_height() + 24
        pygame.draw.rect(screen, (50, 55, 60), (x0, y0, bar_w, bar_h))
        ratio = 0.0 if total <= 0 else max(0.0, min(1.0, float(current) / float(total)))
        fill_w = int(bar_w * ratio)
        if fill_w > 0:
            pygame.draw.rect(screen, (80, 140, 220), (x0, y0, fill_w, bar_h))
        # Percent text
        pct_txt = f"{int(ratio*100)}%"
        s3 = font_ui.render(pct_txt, True, (230, 235, 240))
        screen.blit(s3, (x0 + bar_w + 10, y0 - 2))
        pygame.display.flip()

    def resolve_playlist_or_search(q: str):
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": False,
            "nocheckcertificate": True,
            "format": "bestaudio/best",
        }
        entries = []
        with YoutubeDL(ydl_opts) as ydl:
            target = q if q.startswith("http") else f"ytsearch1:{q}"
            info = ydl.extract_info(target, download=False)
            iterable = (
                info.get("entries")
                if isinstance(info, dict) and info.get("entries")
                else [info]
            )
            for e in iterable:
                if isinstance(e, dict) and (e.get("webpage_url") or e.get("url")):
                    ei = ydl.extract_info(
                        e.get("url") or e.get("webpage_url"), download=False
                    )
                else:
                    ei = e
                url = None
                title = ei.get("title") if isinstance(ei, dict) else None
                duration = ei.get("duration") if isinstance(ei, dict) else None
                artist = ei.get("artist") if isinstance(ei, dict) else None
                track = ei.get("track") if isinstance(ei, dict) else None
                channel = ei.get("channel") if isinstance(ei, dict) else None
                uploader = ei.get("uploader") if isinstance(ei, dict) else None
                if isinstance(ei, dict) and ei.get("url") and not ei.get("formats"):
                    url = ei["url"]
                else:
                    fmts = (ei.get("formats") if isinstance(ei, dict) else None) or []
                    audio_fmts = [
                        f
                        for f in fmts
                        if f.get("acodec") and (f.get("vcodec") in ("none", None))
                    ]
                    best = (
                        audio_fmts[-1] if audio_fmts else (fmts[-1] if fmts else None)
                    )
                    if best:
                        url = best.get("url")
                if url:
                    entries.append(
                        {
                            "title": title,
                            "url": url,
                            "duration": duration,
                            "artist": (artist if artist else None),
                            "track": (track if track else None),
                            "channel": (channel if channel else None),
                            "uploader": (uploader if uploader else None),
                        }
                    )
        return entries

    def build_schedule_for_entry(entry: dict, bias_query: str | None = None):
        nonlocal words_seq, schedule
        title = (entry.get("title") or "") if isinstance(entry, dict) else ""
        duration = entry.get("duration") if isinstance(entry, dict) else None
        track_dur = float(duration) if duration else None
        lrc_text = None
        if syncedlyrics is not None:
            # Prefer the per-entry bias (original user segment) if present
            if bias_query is None and isinstance(entry, dict):
                bias_query = entry.get("bias")
            artist = (entry.get("artist") or "") if isinstance(entry, dict) else ""
            track = (entry.get("track") or "") if isinstance(entry, dict) else ""
            channel = (entry.get("channel") or "") if isinstance(entry, dict) else ""
            uploader = (entry.get("uploader") or "") if isinstance(entry, dict) else ""
            queries = []
            if artist and track:
                queries.append(f"{artist} - {track} lyrics")
                queries.append(f"{artist} {track} lyrics")
            if artist and title:
                queries.append(f"{artist} {title} lyrics")
            if channel and track:
                queries.append(f"{channel} {track} lyrics")
            if uploader and title:
                queries.append(f"{uploader} {title} lyrics")
            if bias_query and not bias_query.strip().lower().startswith("http"):
                queries.append(bias_query.strip())
            cleaned = _clean_title_for_query(title)
            if cleaned:
                queries.append(cleaned)
                queries.append(f"{cleaned} lyrics")
            for q in queries:
                try:
                    lrc_text = syncedlyrics.search(q)
                except Exception:
                    lrc_text = None
                if lrc_text:
                    break
        if lrc_text:
            schedule, words_seq = build_word_schedule_from_lrc(
                "", lrc_text, track_duration=track_dur
            )
            if not words_seq:
                words_seq = tokenize_words_linear(lines)
                schedule = build_uniform_schedule(track_dur or 0.0, words_seq)
        else:
            words_seq = tokenize_words_linear(lines)
            schedule = build_uniform_schedule(track_dur or 0.0, words_seq)

    def start_entry(index: int, user_bias: str | None = None):
        nonlocal current_index, using_vlc, sched_idx
        if not playlist:
            return
        current_index = index % len(playlist)
        entry = playlist[current_index]
        # Build schedule and word list for this track
        build_schedule_for_entry(entry, user_bias)
        # Reset schedule index for new track
        sched_idx = 0
        rebuild_layout()
        # Start playback
        url = entry.get("url")
        if isinstance(url, str) and url.startswith("http"):
            using_vlc = True
            try:
                player.play(url)
            except Exception:
                using_vlc = False
        else:
            using_vlc = False
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(url)
            pygame.mixer.music.play()

    class VLCPlayer:
        def __init__(self):
            self.instance = vlc.Instance()
            self.player = self.instance.media_player_new()

        def play(self, stream_url: str):
            media = self.instance.media_new(stream_url)
            self.player.set_media(media)
            self.player.play()

        def get_time_sec(self) -> float:
            tms = self.player.get_time()
            return max(0.0, (tms or 0) / 1000.0)

        def is_playing(self) -> bool:
            try:
                return bool(self.player.is_playing())
            except Exception:
                return False

    playlist = []
    current_index = 0
    player = VLCPlayer()
    using_vlc = False
    # Track last SPACE press time for double-press detection
    last_space_ts = 0.0
    glyph_cache = {}

    def get_glyph(ch: str, color: Tuple[int, int, int]):
        key = (ch, color)
        surf = glyph_cache.get(key)
        if surf is None:
            surf = font.render(ch, True, color)
            glyph_cache[key] = surf
        return surf

    # Do not auto-start the hardcoded local song; wait for user action instead
    start_time = time.perf_counter()
    word_progress = 0  # advances with beats, indexes into words_seq

    # For rough line â€œprogressionâ€: advance a line on strong beats
    current_line = 0
    last_beat_time = -999.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # Input mode: show search prompt and wait for Enter
        if input_active:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_RETURN:
                        # Allow multiple comma-separated queries/URLs
                        segments = [
                            p.strip() for p in user_input.split(",") if p.strip()
                        ]
                        total = len(segments)
                        combined: List[dict] = []
                        for idx, seg in enumerate(segments, start=1):
                            # Draw minimal progress before resolving this segment
                            preview = (seg[:60] + "…") if len(seg) > 60 else seg
                            draw_progress_screen(
                                f"Resolving {idx}/{total}: {preview}", idx - 1, total
                            )
                            pygame.event.pump()
                            res = resolve_playlist_or_search(seg)
                            if res:
                                for _e in res:
                                    _e["bias"] = seg
                                combined.append(_e)
                            # Update bar after each segment
                            draw_progress_screen(f"Resolved {idx}/{total}", idx, total)
                            pygame.event.pump()
                        if combined:
                            playlist = combined
                            start_entry(0)
                            input_active = False
                            sched_idx = 0
                        else:
                            # No results; keep prompting
                            pass
                    elif ev.key == pygame.K_BACKSPACE:
                        user_input = user_input[:-1]
                    elif ev.key == pygame.K_F11:
                        if is_borderless_fs:
                            screen = pygame.display.set_mode(
                                (WINDOW_W, WINDOW_H), pygame.RESIZABLE
                            )
                            is_borderless_fs = False
                        else:
                            # Prefer real fullscreen; fallback to borderless sized to desktop
                            try:
                                screen = pygame.display.set_mode(
                                    (0, 0), pygame.FULLSCREEN
                                )
                                info = pygame.display.Info()
                                sw, sh = screen.get_size()
                                if sw != info.current_w or sh != info.current_h:
                                    raise Exception("fullscreen size mismatch")
                            except Exception:
                                info = pygame.display.Info()
                                screen = pygame.display.set_mode(
                                    (info.current_w, info.current_h),
                                    pygame.NOFRAME | pygame.SCALED,
                                )
                            is_borderless_fs = True
                    else:
                        if ev.unicode:
                            user_input += ev.unicode
            draw_input_screen()
            continue

        # Use audio clock for precise sync and apply small lead to counter render lag
        if using_vlc:
            t_audio = player.get_time_sec()
        else:
            t_audio = max(0.0, pygame.mixer.music.get_pos() / 1000.0)
        t_now = t_audio + SYNC_LEAD_SEC

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                # Toggle borderless fullscreen
                if is_borderless_fs:
                    screen = pygame.display.set_mode(
                        (WINDOW_W, WINDOW_H), pygame.RESIZABLE
                    )
                    is_borderless_fs = False
                else:
                    # Prefer real fullscreen; fallback to borderless sized to desktop
                    try:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        info = pygame.display.Info()
                        sw, sh = screen.get_size()
                        if sw != info.current_w or sh != info.current_h:
                            raise Exception("fullscreen size mismatch")
                    except Exception:
                        info = pygame.display.Info()
                        screen = pygame.display.set_mode(
                            (info.current_w, info.current_h),
                            pygame.NOFRAME | pygame.SCALED,
                        )
                    is_borderless_fs = True
                rebuild_layout()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # Space: pause/unpause; double-space: next
                if using_vlc:
                    player.player.pause()
                else:
                    # toggle mixer pause
                    if pygame.mixer.get_init():
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()
                # Double-press: treat if within 350ms
                now = time.perf_counter()
                if (now - last_space_ts) <= 0.35:
                    # Next track request: advance playlist if available
                    if playlist and len(playlist) > 1:
                        start_entry(current_index + 1)
                last_space_ts = now
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # Enter: open lightweight search overlay; keep audio playing
                input_active = True
                user_input = ""
            elif event.type == pygame.VIDEORESIZE:
                # Window resized; rebuild layout to continue fitting all words
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                rebuild_layout()

        # Advance schedule index based on time (respects singing gaps)
        if schedule:
            while sched_idx + 1 < len(schedule) and t_now >= schedule[sched_idx].end:
                sched_idx += 1

        # Random per-letter noise (always on, faint)
        for row in grid:
            for cell in row:
                if cell.is_space:
                    # allow space to be dim background
                    cell.intensity = clamp01(cell.intensity - DECAY_PER_SEC * dt * 0.5)
                    continue
                # base noise â€œtwinkleâ€
                if SHOW_NOISE and (np.random.rand() < LETTER_NOISE_PROB):
                    cell.intensity = max(
                        cell.intensity,
                        LETTER_GLOW_NOISE * (0.8 + 0.4 * np.random.rand()),
                    )

                # decay
                cell.intensity = clamp01(cell.intensity - DECAY_PER_SEC * dt)

        # Highlight only the scheduled lyric word at current time
        if schedule and 0 <= sched_idx < len(schedule):
            slot = schedule[sched_idx]
            if slot.start <= t_now <= slot.end:
                target_idx = slot.word_idx
                for row in grid:
                    for cell in row:
                        if not cell.is_space and cell.word_id == target_idx:
                            cell.intensity = clamp01(cell.intensity + WORD_GLOW_BOOST)

        # No auxiliary ripple; keep background steady/dim for stronger contrast

        # Draw
        screen.fill(BG_COLOR)
        for row in grid:
            for cell in row:
                # Mix color by intensity
                color = mix(CHAR_COLOR, GLOW_COLOR, cell.intensity)
                # render only visible area
                if cell.ch == " ":
                    continue
                glyph = get_glyph(cell.ch, color)
                screen.blit(glyph, (cell.rect.x, cell.rect.y))
        # Removed underline to avoid distraction

        pygame.display.flip()

        # Auto-advance when track finishes
        if playlist and len(playlist) > 1:
            if using_vlc:
                if (t_audio > 2.0) and (not player.is_playing()):
                    start_entry(current_index + 1)
            else:
                if (
                    pygame.mixer.get_init()
                    and (t_now > 2.0)
                    and (not pygame.mixer.music.get_busy())
                ):
                    start_entry(current_index + 1)

    pygame.quit()


if __name__ == "__main__":
    main()
