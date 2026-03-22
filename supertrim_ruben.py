#!/usr/bin/env python3
"""
Ruben — super trimmer
──────────────────────
Takes the already-trimmed episodes from ruben_trimmed/ and strips the
intro (summary + theme) and outro, leaving only the raw story content.

The trim is EXCLUSIVE — content starts immediately after the intro jingle
ends, and stops immediately when the outro jingle begins.

Requirements:
    pip install numpy scipy tqdm
    ffmpeg on PATH (https://ffmpeg.org/download.html)

Usage:
    python supertrim_ruben.py                          # all episodes
    python supertrim_ruben.py 2002-01-02 2002-02-01    # date range
    python supertrim_ruben.py 2002-03-15               # single day
    python supertrim_ruben.py 2002-01-02 2002-02-01 --rerun

Reference clips (place alongside this script):
    supertrim_intro.mp3     — the inner intro jingle (content starts after this)
    supertrim_outro.mp3     — the outro jingle (content ends OUTRO_TAIL_SECONDS after detection)

Input dir  : ./ruben_trimmed/
Output dir : ./ruben_content/
Review log : ./supertrim_review.txt
"""

import re
import sys
import subprocess
import argparse
import logging
from datetime import date
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR   = Path("ruben_trimmed")
OUTPUT_DIR  = Path("ruben_content")
REVIEW_FILE = Path("supertrim_review.txt")

INTRO_FILE        = Path("supertrim_intro.mp3")
OUTRO_FILE        = Path("supertrim_outro.mp3")

SAMPLE_RATE           = 8000
MATCH_THRESHOLD_INTRO = 0.25
MATCH_THRESHOLD_OUTRO = 0.25

# How many seconds to keep after the detected outro start point.
# Increase this if the outro is being cut off too early.
OUTRO_TAIL_SECONDS    = 3.5

# Search windows — the intro is always near the start, outro near the end
SEARCH_WINDOW_INTRO   = 60.0   # 1 minute from the start
SEARCH_WINDOW_OUTRO   = 60.0   # 1 minute from the end

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    found_intro  : bool
    found_outro  : bool
    start_sec    : float   # content starts here (after intro)
    end_sec      : float   # content ends here (before outro)
    score_intro  : float
    score_outro  : float


# ── Audio helpers ─────────────────────────────────────────────────────────────

def ffmpeg_decode(path: Path, sr: int = SAMPLE_RATE,
                  offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(offset)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"ffmpeg failed on {path.name}:\n"
                           f"{result.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def get_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", str(path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return float(result.stdout.decode().strip())
    except ValueError:
        return 0.0


def cross_correlate(episode: np.ndarray, jingle: np.ndarray) -> tuple[float, float]:
    """FFT-based cross-correlation. Returns (position_sec, normalised_score)."""
    def norm(x):
        x = x - x.mean()
        std = x.std()
        return x / std if std > 1e-6 else x

    corr = fftconvolve(norm(episode), norm(jingle)[::-1], mode="valid")
    score = float(np.max(np.abs(corr))) / float(len(jingle))
    peak  = int(np.argmax(np.abs(corr)))
    return peak / SAMPLE_RATE, score


def find_boundaries(episode_path: Path,
                    intro: np.ndarray,
                    outro: np.ndarray) -> MatchResult:
    """
    Locate the inner intro and outro jingles in an episode.
    Content starts AFTER the intro jingle, ends OUTRO_TAIL_SECONDS after
    the outro jingle is detected (to avoid cutting off too early).
    """
    tqdm.write(f"    {episode_path.name} …", end="\r")

    total_dur = get_duration(episode_path)

    intro_window = min(SEARCH_WINDOW_INTRO, total_dur)
    ep_intro = ffmpeg_decode(episode_path, duration=intro_window)

    outro_window = min(SEARCH_WINDOW_OUTRO, total_dur)
    outro_offset = max(0.0, total_dur - outro_window)
    ep_outro = ffmpeg_decode(episode_path, offset=outro_offset, duration=outro_window)

    intro_pos, intro_score = cross_correlate(ep_intro, intro)
    outro_pos_rel, outro_score = cross_correlate(ep_outro, outro)
    outro_pos = outro_offset + outro_pos_rel

    found_intro = intro_score >= MATCH_THRESHOLD_INTRO
    found_outro = outro_score >= MATCH_THRESHOLD_OUTRO

    intro_dur = len(intro) / SAMPLE_RATE

    # Content starts right after the intro jingle ends
    content_start = (intro_pos + intro_dur) if found_intro else 0.0
    # Content ends OUTRO_TAIL_SECONDS after the outro is detected
    content_end   = min(outro_pos + OUTRO_TAIL_SECONDS, total_dur) if found_outro else total_dur

    # Sanity check
    if found_intro and found_outro and content_start >= content_end:
        tqdm.write(f"  ⚠ {episode_path.name}: start ({content_start:.1f}s) ≥ "
                   f"end ({content_end:.1f}s) — flagging")
        found_intro = found_outro = False

    return MatchResult(
        found_intro=found_intro,
        found_outro=found_outro,
        start_sec=content_start,
        end_sec=content_end,
        score_intro=intro_score,
        score_outro=outro_score,
    )


def trim_and_save(src: Path, result: MatchResult, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = result.end_sec - result.start_sec
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", str(result.start_sec),
        "-t",  str(duration),
        "-i",  str(src),
        "-acodec", "libmp3lame",
        "-q:a", "2",
        str(dest),
    ]
    r = subprocess.run(cmd, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode(errors='replace')}")


# ── File helpers ──────────────────────────────────────────────────────────────

def parse_filename(path: Path):
    """Extract (date_str, episode_num) from trimmed episode filenames."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})_.*_episode_(\d+)\.mp3", path.name)
    if m:
        return m.group(1), int(m.group(2))
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Super-trim Ruben episodes to raw story content only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python supertrim_ruben.py
  python supertrim_ruben.py 2002-01-02 2002-02-01
  python supertrim_ruben.py 2002-03-15
  python supertrim_ruben.py 2002-01-02 2002-02-01 --rerun
        """
    )
    parser.add_argument("start_date", nargs="?", type=date.fromisoformat)
    parser.add_argument("end_date",   nargs="?", type=date.fromisoformat)
    parser.add_argument("--rerun", action="store_true",
                        help="Re-process even if output already exists.")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.start_date and args.end_date:
        date_from = min(args.start_date, args.end_date)
        date_to   = max(args.start_date, args.end_date)
        log.info(f"Date range : {date_from} → {date_to}")
    elif args.start_date:
        date_from = date_to = args.start_date
        log.info(f"Single day : {date_from}")
    else:
        date_from = date_to = None
        log.info("Date range : all episodes")

    def in_range(date_str: str) -> bool:
        if date_from is None:
            return True
        try:
            d = date.fromisoformat(date_str)
            return date_from <= d <= date_to
        except ValueError:
            return False

    # Validate
    if not INTRO_FILE.exists():
        log.error(f"✗ Not found: {INTRO_FILE}")
        sys.exit(1)

    if not OUTRO_FILE.exists():
        log.error(f"✗ Not found: {OUTRO_FILE}")
        sys.exit(1)

    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run([tool, "-version"], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            log.error(f"✗ '{tool}' not found — https://ffmpeg.org/download.html")
            sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load reference clips
    log.info("Loading reference clips …")
    intro_ref  = ffmpeg_decode(INTRO_FILE)
    outro_ref  = ffmpeg_decode(OUTRO_FILE)
    log.info(f"  Intro : {len(intro_ref)/SAMPLE_RATE:.2f}s  ({INTRO_FILE.name})")
    log.info(f"  Outro : {len(outro_ref)/SAMPLE_RATE:.2f}s  ({OUTRO_FILE.name})")
    log.info(f"  Outro tail padding : {OUTRO_TAIL_SECONDS}s")
    log.info("")

    # Collect episodes
    all_episodes = sorted(INPUT_DIR.glob("*.mp3"))
    episodes = []
    for f in all_episodes:
        parsed = parse_filename(f)
        if parsed:
            date_str, ep_num = parsed
            if in_range(date_str):
                episodes.append((ep_num, date_str, f))
    episodes.sort()

    if not episodes:
        log.error(f"✗ No episodes found in {INPUT_DIR}/")
        sys.exit(1)

    # Build episode number map for the review log (global, not just filtered)
    all_parsed = [(parse_filename(f), f) for f in all_episodes]
    all_nums   = sorted(p[1] for p, _ in all_parsed if p)

    log.info(f"Found {len(episodes)} episodes to process.\n")

    flagged   = []
    done      = 0
    skipped   = 0
    failed    = 0

    for ep_num, date_str, src in tqdm(episodes, desc="Super-trimming", unit="ep"):
        dest_name = src.name.replace("_trimmed_", "_content_")
        dest      = OUTPUT_DIR / dest_name

        if dest.exists() and not args.rerun:
            skipped += 1
            continue

        result = find_boundaries(src, intro_ref, outro_ref)

        tqdm.write(
            f"  ep {ep_num:03d} {date_str}  "
            f"intro={'✓' if result.found_intro else '✗'} ({result.score_intro:.3f})  "
            f"outro={'✓' if result.found_outro else '✗'} ({result.score_outro:.3f})  "
            f"{result.end_sec - result.start_sec:.1f}s content"
        )

        if result.found_intro and result.found_outro:
            try:
                trim_and_save(src, result, dest)
                done += 1
            except Exception as e:
                tqdm.write(f"  ✗ Save failed: {e}")
                failed += 1
        else:
            flagged.append({
                "ep_num":   ep_num,
                "date_str": date_str,
                "file":     str(src),
                "result":   result,
            })
            tqdm.write(f"  ⚑ FLAGGED: episode {ep_num:03d} {date_str}")

    # ── Review log ────────────────────────────────────────────────────────────
    with open(REVIEW_FILE, "w", encoding="utf-8") as f:
        f.write(f"Super-trim manual review — {len(flagged)} episode(s)\n")
        f.write("=" * 60 + "\n\n")
        for entry in flagged:
            r = entry["result"]
            intro_str = f"{'✓' if r.found_intro else '✗'} {r.score_intro:.3f} {r.start_sec:.2f}s"
            outro_str = f"{'✓' if r.found_outro else '✗'} {r.score_outro:.3f} {r.end_sec:.2f}s"
            f.write(f"Episode : {entry['ep_num']:03d} — {entry['date_str']}\n")
            f.write(f"File    : {entry['file']}  "
                    f"(Intro: {intro_str}  Outro: {outro_str})\n\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Processed : {len(episodes)}
  Done      : {done}
  Skipped   : {skipped}  (already existed — use --rerun to redo)
  Flagged   : {len(flagged)}  → see {REVIEW_FILE}
  Failed    : {failed}
  Output    : {OUTPUT_DIR.resolve()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thresholds — intro: {MATCH_THRESHOLD_INTRO}  outro: {MATCH_THRESHOLD_OUTRO}
Adjust in the script if too many episodes are flagged or mis-trimmed.
""")


if __name__ == "__main__":
    main()
