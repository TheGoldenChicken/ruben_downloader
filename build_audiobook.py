#!/usr/bin/env python3
"""
Ruben — audiobook builder
──────────────────────────
Combines all trimmed episodes into a single M4B audiobook file,
with each episode as a named chapter.

M4B is the standard audiobook format supported by Apple Books,
VLC, and most podcast/audiobook players.

Requirements:
    ffmpeg on PATH (https://ffmpeg.org/download.html)

Usage:
    python build_audiobook.py

    # Custom output filename
    python build_audiobook.py --output "Ruben_audiobook.m4b"

    # Only include a date range
    python build_audiobook.py 2002-01-02 2002-06-01

    # Preview chapter list without building
    python build_audiobook.py --dry-run

Input dir  : ./ruben_trimmed/
Output     : ./Ruben_drengen_der_kan_tale_med_ting.m4b
"""

import re
import sys
import argparse
import subprocess
import tempfile
import logging
from datetime import date
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR_TRIMMED  = Path("ruben_trimmed")
INPUT_DIR_CONTENT  = Path("ruben_content")
DEFAULT_OUTPUT     = Path("Ruben_drengen_der_kan_tale_med_ting.m4b")

# ID3 / audiobook metadata
TITLE           = "Ruben — drengen der kan tale med ting"
ARTIST          = "DR P3"
ALBUM           = "Ruben — drengen der kan tale med ting"
GENRE           = "Audiobook"
LANGUAGE        = "dan"

# Audio encoding for the output file
# aac at 64k is standard for spoken-word audiobooks — small file, excellent quality
AUDIO_CODEC     = "aac"
AUDIO_BITRATE   = "64k"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_filename(path: Path):
    """
    Extract (date_str, episode_num) from filenames like:
        2002-01-02_Ruben_trimmed_episode_001.mp3
    Returns None if the filename doesn't match.
    """
    m = re.match(r"(\d{4}-\d{2}-\d{2})_.*_episode_(\d+)\.mp3", path.name)
    if m:
        return m.group(1), int(m.group(2))
    return None


def get_duration(path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
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


def ms(seconds: float) -> int:
    """Convert seconds to milliseconds (integer), used for chapter timestamps."""
    return int(seconds * 1000)


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS for display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an M4B audiobook from trimmed Ruben episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_audiobook.py                               # trimmed episodes
  python build_audiobook.py --content                     # super-trimmed (story only)
  python build_audiobook.py --content --output Ruben.m4b  # custom filename
  python build_audiobook.py 2002-01-02 2002-06-01         # date range
  python build_audiobook.py --dry-run                     # preview chapters only
        """
    )
    parser.add_argument(
        "start_date", nargs="?", type=date.fromisoformat,
        help="First episode date to include (YYYY-MM-DD)."
    )
    parser.add_argument(
        "end_date", nargs="?", type=date.fromisoformat,
        help="Last episode date to include (YYYY-MM-DD, inclusive)."
    )
    parser.add_argument(
        "--content", action="store_true",
        help="Use super-trimmed episodes from ruben_content/ instead of ruben_trimmed/."
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output filename (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print chapter list and total duration without building the file."
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Date filter
    if args.start_date and args.end_date:
        date_from = min(args.start_date, args.end_date)
        date_to   = max(args.start_date, args.end_date)
    elif args.start_date:
        date_from = date_to = args.start_date
    else:
        date_from = date_to = None

    def in_range(date_str: str) -> bool:
        if date_from is None:
            return True
        try:
            d = date.fromisoformat(date_str)
            return date_from <= d <= date_to
        except ValueError:
            return False

    # Resolve source directory and filename pattern based on mode
    if args.content:
        input_dir    = INPUT_DIR_CONTENT
        file_pattern = "*_content_episode_*.mp3"
        mode_label   = "super-trimmed (story only)"
    else:
        input_dir    = INPUT_DIR_TRIMMED
        file_pattern = "*_trimmed_episode_*.mp3"
        mode_label   = "trimmed (with intro/outro jingles)"

    log.info(f"Mode : {mode_label}")
    log.info(f"From : {input_dir}/\n")

    # Check ffmpeg
    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run([tool, "-version"], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            print(f"✗ '{tool}' not found. Install ffmpeg: https://ffmpeg.org/download.html")
            sys.exit(1)

    # Collect and filter episode files
    all_files = sorted(input_dir.glob(file_pattern))
    episodes = []
    for f in all_files:
        parsed = parse_filename(f)
        if parsed:
            date_str, ep_num = parsed
            if in_range(date_str):
                episodes.append((ep_num, date_str, f))

    episodes.sort()   # sort by episode number

    if not episodes:
        print(f"✗ No episode files found in {input_dir}/")
        print(f"  Expected filenames like: 2002-01-02_Ruben_{'content' if args.content else 'trimmed'}_episode_001.mp3")
        sys.exit(1)

    print(f"Found {len(episodes)} episodes to include.\n")

    # Measure durations and build chapter list
    print("Measuring episode durations …")
    chapters = []
    cursor   = 0.0

    for ep_num, date_str, path in episodes:
        dur = get_duration(path)
        if dur <= 0:
            print(f"  ⚠ Skipping {path.name} — could not read duration")
            continue
        # Format chapter title: "Episode 001 — 2002-01-02"
        title = f"Episode {ep_num:03d} — {date_str}"
        chapters.append({
            "title":      title,
            "path":       path,
            "start_ms":   ms(cursor),
            "end_ms":     ms(cursor + dur),
            "start_sec":  cursor,
            "duration":   dur,
        })
        cursor += dur

    total_duration = cursor
    print(f"\nTotal duration : {format_duration(total_duration)}")
    print(f"Chapters       : {len(chapters)}\n")

    # Dry run — just print chapter list
    if args.dry_run:
        print(f"{'#':>5}  {'Start':>8}  {'Duration':>9}  Title")
        print(f"{'─'*5}  {'─'*8}  {'─'*9}  {'─'*40}")
        for i, ch in enumerate(chapters, 1):
            print(
                f"{i:>5}  "
                f"{format_duration(ch['start_sec']):>8}  "
                f"{format_duration(ch['duration']):>9}  "
                f"{ch['title']}"
            )
        print(f"\n(Dry run — no file written. Remove --dry-run to build.)")
        return

    # ── Build the audiobook ───────────────────────────────────────────────────

    output_path = args.output
    print(f"Building audiobook → {output_path}")
    print("(This may take several minutes for a large collection)\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── 1. Write ffmpeg concat list ───────────────────────────────────────
        concat_file = tmp / "concat.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for ch in chapters:
                # ffmpeg concat demuxer requires escaped paths
                escaped = str(ch["path"].resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        # ── 2. Write FFMETADATA chapter file ──────────────────────────────────
        meta_file = tmp / "metadata.txt"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            f.write(f"title={TITLE}\n")
            f.write(f"artist={ARTIST}\n")
            f.write(f"album={ALBUM}\n")
            f.write(f"genre={GENRE}\n")
            f.write(f"language={LANGUAGE}\n")
            f.write("\n")
            for ch in chapters:
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={ch['start_ms']}\n")
                f.write(f"END={ch['end_ms']}\n")
                f.write(f"title={ch['title']}\n")
                f.write("\n")

        # ── 3. Concatenate + encode to M4B in one ffmpeg pass ─────────────────
        cmd = [
            "ffmpeg", "-v", "error", "-stats", "-y",
            "-f",      "concat",
            "-safe",   "0",
            "-i",      str(concat_file),
            "-i",      str(meta_file),
            "-map_metadata", "1",        # apply chapter/tag metadata
            "-map",    "0:a",            # audio from concat
            "-c:a",    AUDIO_CODEC,
            "-b:a",    AUDIO_BITRATE,
            "-movflags", "+faststart",   # place moov atom at start (better streaming)
            str(output_path),
        ]

        print("Running ffmpeg …")
        result = subprocess.run(cmd, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"\n✗ ffmpeg failed:\n{result.stderr.decode(errors='replace')}")
            sys.exit(1)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Done!
  Episodes  : {len(chapters)}
  Duration  : {format_duration(total_duration)}
  File size : {size_mb:.1f} MB
  Output    : {output_path.resolve()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Open in Apple Books, VLC, or any M4B-compatible player.
Each episode is a named chapter you can skip between directly.
""")


if __name__ == "__main__":
    main()
