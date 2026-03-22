#!/usr/bin/env python3
"""
Ruben episode trimmer
─────────────────────
Scans all downloaded episodes, finds the intro and outro jingles using
cross-correlation, and trims each episode to just the segment content.

Prefers the 06:50 broadcast (04:50 UTC winter / 05:50 UTC summer).
Falls back to 17:40 (15:40 UTC winter / 16:40 UTC summer).
If neither broadcast for a day has both jingles, the day is flagged.

Requirements:
    pip install numpy scipy tqdm
    ffmpeg must be installed and on PATH (https://ffmpeg.org/download.html)

Usage:
    1. Put your reference clips alongside this script:
         jingle_start.mp3   (3–5 sec clip of the intro jingle)
         jingle_end.mp3     (3–5 sec clip of the outro jingle)
    2. Run:
         python trim_ruben.py

    Input dir  : ./ruben_episodes/       (downloaded files)
    Output dir : ./ruben_trimmed/        (trimmed files, same filenames)
    Review log : ./manual_review.txt     (days with no clean broadcast)

Tuning:
    MATCH_THRESHOLD_START – score to accept intro jingle match (default 0.4)
    MATCH_THRESHOLD_END   – score to accept outro jingle match (default 0.2,
                            looser because the end jingle is often short/talked over)
    JINGLE_END_OFFSET     – seconds after detected jingle start to trim to.
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

INPUT_DIR   = Path("ruben_episodes")
OUTPUT_DIR  = Path("ruben_trimmed")
REVIEW_FILE = Path("manual_review.txt")

JINGLE_START_PATTERN  = "jingle_start_*.mp3"
# End jingle variants: jingle_end_1.mp3, jingle_end_2.mp3, etc.
# The best-scoring match across all variants is used.
JINGLE_END_PATTERN = "jingle_end_*.mp3"

SAMPLE_RATE          = 8000   # Hz — low rate is fine for jingle detection, much faster
MATCH_THRESHOLD_START = 0.25   # stricter — start jingle is clean and consistent
MATCH_THRESHOLD_END   = 0.25   # looser  — end jingle is short and sometimes talked over
JINGLE_END_OFFSET     = 0.3   # seconds of fade-out margin after jingle detection point

# UTC hour prefixes for each broadcast slot.
# Two values each to cover winter time and summer time (DST).
PREF_UTC_HOURS     = ["04", "05"]   # 06:50 local → 04:50 UTC (winter) / 05:50 UTC (summer)
FALLBACK_UTC_HOURS = ["15", "16"]   # 17:40 local → 15:40 UTC (winter) / 16:40 UTC (summer)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    found_start : bool
    found_end   : bool
    start_sec   : float   # content starts here
    end_sec     : float   # content ends here
    score_start : float
    score_end   : float


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _ffmpeg_decode(path: Path, sr: int,
                   offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    """
    Decode audio via ffmpeg to raw PCM, bypassing libmpg123.
    Handles corrupt MP3 frames gracefully (ffmpeg skips bad frames).
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(offset),
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-i", str(path),
        "-f", "f32le",          # 32-bit float, little-endian PCM
        "-ac", "1",             # mono
        "-ar", str(sr),         # resample to target rate
        "pipe:1",               # stream to stdout
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(
            f"ffmpeg failed on {path.name}:\n{result.stderr.decode(errors='replace')}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32)


def load_mono(path: Path, sr: int = SAMPLE_RATE,
              offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    return _ffmpeg_decode(path, sr, offset=offset, duration=duration)


def load_stereo_native(path: Path) -> tuple[np.ndarray, int]:
    """Load full file at native sample rate for trimming (preserves quality)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "csv=p=0",
        str(path),
    ]
    probe = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    parts = probe.stdout.decode().strip().split(",")
    native_sr   = int(parts[0]) if parts[0].isdigit() else 44100
    channels    = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1

    cmd2 = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-ac", str(channels),
        "-ar", str(native_sr),
        "pipe:1",
    ]
    result = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = np.frombuffer(result.stdout, dtype=np.float32)
    if channels > 1:
        raw = raw.reshape(-1, channels)
    return raw, native_sr


def get_duration(path: Path) -> float:
    """Fast duration check using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return float(result.stdout.decode().strip())
    except ValueError:
        return 0.0


def cross_correlate(episode: np.ndarray, jingle: np.ndarray) -> tuple[float, float]:
    """
    Slide jingle over episode using FFT-based cross-correlation — O(n log n).
    Returns (position_in_seconds, normalised_score 0–1).
    """
    def norm(x):
        x = x - x.mean()
        std = x.std()
        return x / std if std > 1e-6 else x

    ep = norm(episode)
    ji = norm(jingle)

    # FFT convolution: flip jingle to turn convolution into correlation
    correlation = fftconvolve(ep, ji[::-1], mode="valid")

    max_possible = float(len(ji))
    score = float(np.max(np.abs(correlation))) / max_possible

    peak_sample = int(np.argmax(np.abs(correlation)))
    position_sec = peak_sample / SAMPLE_RATE

    return position_sec, score


# How many seconds to search at each end of the file for the jingles.
SEARCH_WINDOW_START = 240.0   # 4 minutes — intro jingle can appear late
SEARCH_WINDOW_END   = 180.0    # 3 minute  — outro jingle is always near the end


def find_jingles(episode_path: Path,
                 jingle_starts: list[np.ndarray],
                 jingle_ends: list[np.ndarray]) -> MatchResult:
    """
    Find intro and outro jingle positions in an episode file.
    Tests all end jingle variants and picks the one with the highest score.
    Only loads the first and last SEARCH_WINDOW seconds to keep it fast.
    """
    tqdm.write(f"    loading {episode_path.name} …", end="\r")

    total_dur = get_duration(episode_path)

    # Load only the first N seconds for start jingle
    start_window = min(SEARCH_WINDOW_START, total_dur)
    ep_start = load_mono(episode_path, duration=start_window)

    # Load only the last N seconds for end jingle
    end_window = min(SEARCH_WINDOW_END, total_dur)
    end_offset = max(0.0, total_dur - end_window)
    ep_end = load_mono(episode_path, offset=end_offset, duration=end_window)

    # test every start jingle variant — keep the best score
    best_start_score = -1.0
    best_start_pos   = 0.0
    best_start_dur   = 0.0
    for js in jingle_starts[1:]:
        pos_rel, score = cross_correlate(ep_start, js)
        if score > best_start_score:
            best_start_score = score
            best_start_pos   = pos_rel
            best_start_dur   = len(js) / SAMPLE_RATE
            
    start_score = best_start_score
    start_pos   = best_start_pos

    # Test every end jingle variant — keep the best score
    best_end_score = -1.0
    best_end_pos   = 0.0
    best_end_dur   = 0.0
    for je in jingle_ends:
        pos_rel, score = cross_correlate(ep_end, je)
        if score > best_end_score:
            best_end_score = score
            best_end_pos   = end_offset + pos_rel
            best_end_dur   = len(je) / SAMPLE_RATE

    end_score = best_end_score
    end_pos   = best_end_pos

    duration = total_dur

    found_start = start_score >= MATCH_THRESHOLD_START
    found_end   = end_score   >= MATCH_THRESHOLD_END

    # Trim starts at the intro jingle (not after it)
    content_start = start_pos if found_start else 0.0

    # Trim ends after the outro jingle finishes
    content_end = (end_pos + best_end_dur + JINGLE_END_OFFSET) if found_end else duration

    # Sanity check — start must be before end
    if found_start and found_end and content_start >= content_end:
        log.warning(f"  ⚠ Start ({content_start:.1f}s) ≥ end ({content_end:.1f}s) — "
                    f"treating as no match for {episode_path.name}")
        found_start = found_end = False

    return MatchResult(
        found_start=found_start,
        found_end=found_end,
        start_sec=content_start,
        end_sec=content_end,
        score_start=start_score,
        score_end=end_score,
    )


def trim_and_save(episode_path: Path, result: MatchResult, out_path: Path):
    """Trim via ffmpeg directly — fast, no full decode into memory needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = result.end_sec - result.start_sec
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", str(result.start_sec),
        "-t",  str(duration),
        "-i",  str(episode_path),
        "-acodec", "libmp3lame",
        "-q:a", "2",            # VBR ~190 kbps — good quality
        str(out_path),
    ]
    r = subprocess.run(cmd, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg trim failed:\n{r.stderr.decode(errors='replace')}"
        )


# ── File grouping ─────────────────────────────────────────────────────────────

def utc_hour_to_slot(hour: str) -> str | None:
    """Map a UTC hour string to a canonical slot name ('pref' or 'fallback')."""
    if hour in PREF_UTC_HOURS:
        return "pref"
    if hour in FALLBACK_UTC_HOURS:
        return "fallback"
    return None


def parse_filename(path: Path) -> tuple[str, str] | None:
    """
    Extract (date, utc_hour) from filenames like:
        2002-01-02_0450_0001_Ruben.mp3   →  ("2002-01-02", "04")
        2002-01-02_1540_0002_Ruben.mp3   →  ("2002-01-02", "15")
        2002-06-01_0550_0001_Ruben.mp3   →  ("2002-06-01", "05")  ← summer time
    """
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})\d{2}_", path.name)
    if m:
        return m.group(1), m.group(2)
    return None


def group_by_day(input_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Returns {date: {"pref": Path, "fallback": Path}} for all episode files.
    Handles both winter (04/15) and summer (05/16) UTC offsets transparently.
    """
    days: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in sorted(input_dir.glob("*.mp3")):
        parsed = parse_filename(f)
        if parsed:
            date_str, hour = parsed
            slot = utc_hour_to_slot(hour)
            if slot:
                days[date_str][slot] = f
    return days


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trim Ruben episodes to jingle boundaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trim_ruben.py                          # process all days
  python trim_ruben.py 2002-01-02 2002-02-01    # process a date range (inclusive)
  python trim_ruben.py 2002-03-15               # process a single day
  python trim_ruben.py --rerun 2002-01-02 2002-02-01  # rerun even if output exists
        """
    )
    parser.add_argument(
        "start_date", nargs="?", type=date.fromisoformat,
        help="First day to process (YYYY-MM-DD). Omit to process all days."
    )
    parser.add_argument(
        "end_date", nargs="?", type=date.fromisoformat,
        help="Last day to process (YYYY-MM-DD, inclusive). Omit to process only start_date."
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-process days even if a trimmed output already exists."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build date filter
    if args.start_date and args.end_date:
        date_from, date_to = args.start_date, args.end_date
        if date_from > date_to:
            date_from, date_to = date_to, date_from   # swap silently
        log.info(f"Date range : {date_from} → {date_to}")
    elif args.start_date:
        date_from = date_to = args.start_date
        log.info(f"Single day : {date_from}")
    else:
        date_from = date_to = None
        log.info("Date range : all days")

    def in_range(date_str: str) -> bool:
        if date_from is None:
            return True
        try:
            d = date.fromisoformat(date_str)
            return date_from <= d <= date_to
        except ValueError:
            return False

    # Validate inputs
    start_jingle_files = sorted(Path(".").glob(JINGLE_START_PATTERN))
    if not start_jingle_files:
        log.error(f"✗ No start jingle files found matching '{JINGLE_START_PATTERN}'")
        log.error("  Expected e.g.: jingle_start.mp3, jingle_start1.mp3, etc.")
        sys.exit(1)

    end_jingle_files = sorted(Path(".").glob(JINGLE_END_PATTERN))
    if not end_jingle_files:
        log.error(f"✗ No end jingle files found matching '{JINGLE_END_PATTERN}'")
        log.error("  Expected e.g.: jingle_end.mp3, jingle_end2.mp3, jingle_end_short.mp3 …")
        sys.exit(1)

    # Check ffmpeg is available
    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run([tool, "-version"], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            log.error(f"✗ '{tool}' not found. Install ffmpeg and make sure it's on your PATH.")
            log.error("  Download: https://ffmpeg.org/download.html")
            sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    log.info("Loading jingle references …")
    jingle_starts = [load_mono(f) for f in start_jingle_files]
    for f, js in zip(start_jingle_files, jingle_starts):
        log.info(f"  Start jingle : {len(js)/SAMPLE_RATE:.2f}s  ({f.name})")
        
    jingle_ends = [load_mono(f) for f in end_jingle_files]
    for f, je in zip(end_jingle_files, jingle_ends):
        log.info(f"  End jingle   : {len(je)/SAMPLE_RATE:.2f}s  ({f.name})")
    log.info("")

    days = group_by_day(INPUT_DIR)
    all_days = sorted(days.keys())
    filtered_days = [d for d in all_days if in_range(d)]

    # Episode numbers are assigned by position across ALL days, not just the
    # filtered range — so numbers stay stable no matter what range you run.
    episode_number = {d: i for i, d in enumerate(all_days, start=1)}

    log.info(f"Found {len(all_days)} total days, {len(filtered_days)} in range.\n")

    flagged_days = []
    trimmed_count = 0
    skipped_count = 0

    for date_str in tqdm(filtered_days, desc="Processing days", unit="day"):
        broadcasts = days[date_str]
        ep_num = episode_number[date_str]

        out_path = OUTPUT_DIR / f"{date_str}_Ruben_trimmed_episode_{ep_num:03d}.mp3"
        if out_path.exists() and not args.rerun:
            skipped_count += 1
            continue

        # Try preferred broadcast first, then fallback
        slots_to_try = []
        if "pref" in broadcasts:
            slots_to_try.append(("pref", broadcasts["pref"]))
        if "fallback" in broadcasts:
            slots_to_try.append(("fallback", broadcasts["fallback"]))

        best_result  = None
        best_source  = None
        per_file_results = []

        for slot, path in slots_to_try:
            result = find_jingles(path, jingle_starts, jingle_ends)
            per_file_results.append(result)
            tqdm.write(
                f"  {date_str} [{slot}]  "
                f"start={'✓' if result.found_start else '✗'} ({result.score_start:.3f})  "
                f"end={'✓' if result.found_end else '✗'} ({result.score_end:.3f})  "
                f"← {path.name}"
            )

            if result.found_start and result.found_end:
                best_result = result
                best_source = path
                break

            if best_result is None:
                best_result = result
                best_source = path
            else:
                prev_count = best_result.found_start + best_result.found_end
                this_count = result.found_start + result.found_end
                if this_count > prev_count:
                    best_result = result
                    best_source = path

        if best_result and best_result.found_start and best_result.found_end:
            trim_and_save(best_source, best_result, out_path)
            trimmed_count += 1
        else:
            flagged_days.append({
                "date":             date_str,
                "files":            [str(p) for _, p in slots_to_try],
                "per_file_results": per_file_results,
                "result":           best_result,
            })
            tqdm.write(f"  ⚑ FLAGGED: {date_str} — no broadcast had both jingles\n")

    # ── Write manual review log ───────────────────────────────────────────────
    with open(REVIEW_FILE, "w", encoding="utf-8") as f:
        f.write(f"Manual review required — {len(flagged_days)} day(s)\n")
        f.write("=" * 60 + "\n\n")
        for entry in flagged_days:
            ep_num = episode_number[entry['date']]
            suggested = f"{entry['date']}_Ruben_trimmed_episode_{ep_num:03d}.mp3"
            f.write(f"Episode : {ep_num:03d} — {entry['date']}\n")
            f.write(f"Save as : {OUTPUT_DIR / suggested}\n")
            for fp, per_file in zip(entry["files"], entry["per_file_results"]):
                r = per_file
                start_str = f"Start: {r.score_start:.3f} {r.start_sec:.2f}s"
                end_str   = f"End: {r.score_end:.3f} {r.end_sec:.2f}s"
                f.write(f"File : {fp} ({start_str}  {end_str})\n")
            f.write("\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Days in range  : {len(filtered_days)}
  Trimmed        : {trimmed_count}
  Skipped        : {skipped_count}  (already existed — use --rerun to redo)
  Flagged        : {len(flagged_days)}  → see {REVIEW_FILE}
  Output dir     : {OUTPUT_DIR.resolve()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tip: if you get too many false matches or missed matches, adjust
MATCH_THRESHOLD in the script (currently START: {MATCH_THRESHOLD_START} END: {MATCH_THRESHOLD_END}).
  Lower  → more permissive (catches weak jingles, more false positives)
  Higher → stricter (fewer false positives, more flagged days)
""")


if __name__ == "__main__":
    main()
