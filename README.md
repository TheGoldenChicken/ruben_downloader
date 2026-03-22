# Ruben — drengen der kan tale med ting
### DR Arkivet bulk downloader, trimmer & audiobook builder

Tools for downloading all episodes of the radio series *Ruben — drengen der kan tale med ting* from the Danish Royal Library's DR Arkivet, automatically trimming each episode to its jingle boundaries, and compiling everything into an M4B audiobook.

---

## Overview

The series was broadcast twice daily on DR P3 over roughly two years:
- **Morning** at 06:50 local time (04:50 UTC winter / 05:50 UTC summer)
- **Afternoon** at 17:40 local time (15:40 UTC winter / 16:40 UTC summer)

Each broadcast is a raw radio snippet — the actual episode segment starts and ends at varying points within the recording. These tools handle the full pipeline: downloading every broadcast, trimming to the broadcast jingles, optionally stripping the summary and intro to leave only the story, and packaging everything as a chaptered audiobook.

---

## Requirements

### Python packages
```
pip install requests librosa numpy scipy tqdm
```

### ffmpeg
Required for audio decoding and trimming. Must be on your system PATH.

- **Windows:** `winget install ffmpeg` or download from https://ffmpeg.org/download.html
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

---

## File structure

Place all scripts and reference clips in the same folder:

```
your-folder/
├── download_ruben.py
├── trim_ruben.py
├── supertrim_ruben.py
├── build_audiobook.py
├── diagnose_jingle.py
│
├── jingle_start.mp3          ← broadcast intro jingle
├── jingle_end_1.mp3          ← broadcast outro jingle variant 1
├── jingle_end_2.mp3          ← broadcast outro jingle variant 2 (add as many as needed)
│
├── supertrim_intro.mp3       ← inner story intro jingle (content starts after this)
├── supertrim_outro.mp3       ← inner story outro jingle (content ends here)
│
├── ruben_episodes/           ← raw downloaded broadcasts  (created by downloader)
├── ruben_trimmed/            ← broadcast-trimmed episodes (created by trim_ruben.py)
├── ruben_content/            ← story-only episodes        (created by supertrim_ruben.py)
│
├── manual_review.txt         ← days flagged by trim_ruben.py
└── supertrim_review.txt      ← episodes flagged by supertrim_ruben.py
```

---

## Step 1 — Download episodes

### Get your session token

The KB search API requires an authentication cookie from your browser session.

1. Open https://www.kb.dk/find-materiale/dr-arkivet in your browser
2. Open **Developer Tools** → **Network** tab (F12)
3. Search for something on the page to trigger an API call
4. Find a request to `ds-api/bff/v1/proxy/` and click it
5. Under **Request Headers**, find the `Cookie:` header
6. Copy the value of the `Authorization=` cookie (the long base64 string)

### Configure and run

Open `download_ruben.py` and paste your token as `AUTH_TOKEN`:

```python
AUTH_TOKEN = "paste-your-token-here"
```

Then run:

```bash
python download_ruben.py
```

Episodes are saved to `ruben_episodes/` with filenames like:
```
2002-01-02_0550_0001_Ruben.mp3
2002-01-02_1640_0002_Ruben.mp3
```

The filename encodes the broadcast date and UTC time, which the trimmer uses to pair morning and afternoon broadcasts for the same day.

### Token expiry

The auth cookie expires after a while. If you see a `401` or `403` error mid-download, grab a fresh token from your browser and replace `AUTH_TOKEN` in the script. Re-running is safe — already-downloaded files are skipped automatically.

---

## Step 2 — Prepare broadcast jingle reference clips

You need short audio clips of the broadcast intro and outro jingles to use as fingerprints. Extract these once from any clean episode using Audacity or any audio editor.

### Broadcast intro jingle — `jingle_start.mp3`
A clean clip of the broadcast intro jingle, roughly 5–8 seconds. Trim away any silence before the jingle starts.

### Broadcast outro jingle — `jingle_end_1.mp3`, `jingle_end_2.mp3`, …
The outro is often short and sometimes partially talked over, so it is worth preparing several variants:
- The full outro jingle
- Just the first half
- Just the last few seconds
- Any other clearly distinct portion

Name them `jingle_end_1.mp3`, `jingle_end_2.mp3`, etc. The trimmer tests all variants and uses whichever scores highest.

---

## Step 3 — Trim episodes to broadcast boundaries

This step trims each raw broadcast snippet so it starts at the broadcast intro jingle and ends at the broadcast outro jingle, including both jingles in the output.

```bash
# Process all downloaded episodes
python trim_ruben.py

# Process a date range (inclusive)
python trim_ruben.py 2002-01-02 2002-02-01

# Process a single day
python trim_ruben.py 2002-03-15

# Re-process a range even if output already exists
python trim_ruben.py 2002-01-02 2002-02-01 --rerun
```

### What the trimmer does

For each day:
1. Tries the **morning broadcast** (06:50) first
2. Runs FFT-based cross-correlation to locate the intro and outro jingles
3. If both found → trims and saves, including both jingles
4. If not → tries the **afternoon broadcast** (17:40) as a fallback
5. If neither has both jingles → flags the day for manual review

Trimmed files are saved to `ruben_trimmed/` with episode numbers:
```
2002-01-02_Ruben_trimmed_episode_001.mp3
2002-01-03_Ruben_trimmed_episode_002.mp3
2002-01-07_Ruben_trimmed_episode_004.mp3   ← gap immediately visible
```

Episode numbers are based on position across all days and stay stable regardless of which date range you run.

### Manual review — `manual_review.txt`

Days where neither broadcast had both jingles are written to `manual_review.txt`:

```
Episode : 003 — 2002-01-04
Save as : ruben_trimmed\2002-01-04_Ruben_trimmed_episode_003.mp3
File : ruben_episodes/2002-01-04_0550_0005_Ruben.mp3 (Start: 0.776 62.02s  End: 0.123 482.02s)
File : ruben_episodes/2002-01-04_1640_0006_Ruben.mp3 (Start: 0.336 14.02s  End: 0.152 412.02s)
```

Each file line shows the correlation score (0–1) and timestamp in seconds for the best-matching position of each jingle. To fix a flagged episode manually:
1. Open the source file in Audacity — use the timestamps as a starting point
2. Trim to the correct boundaries
3. Export using the **Save as** filename — copy/paste it directly from the log

### Tuning

| Setting | Default | Description |
|---|---|---|
| `MATCH_THRESHOLD_START` | `0.25` | Minimum score to accept an intro jingle match |
| `MATCH_THRESHOLD_END` | `0.15` | Minimum score to accept an outro jingle match |
| `SEARCH_WINDOW_START` | `360.0` | Seconds from start of file to search for intro |
| `SEARCH_WINDOW_END` | `180.0` | Seconds from end of file to search for outro |
| `JINGLE_END_OFFSET` | `0.3` | Extra seconds kept after the detected outro end point |

Lower thresholds if too many episodes are flagged. Raise them if trim points are wrong.

---

## Step 4 — Super-trim to story content only *(optional)*

Each trimmed episode still contains a spoken summary and broadcast intro before the actual story begins. This step strips everything before the inner story intro jingle and everything after the inner story outro jingle, leaving only the raw narrative.

### Prepare reference clips

- **`supertrim_intro.mp3`** — the inner story intro jingle. Content starts immediately after this ends.
- **`supertrim_outro.mp3`** — the inner story outro jingle. Content ends here (plus a configurable tail).

```bash
# Process all trimmed episodes
python supertrim_ruben.py

# Process a date range
python supertrim_ruben.py 2002-01-02 2002-02-01

# Single day
python supertrim_ruben.py 2002-03-15

# Re-process even if output already exists
python supertrim_ruben.py 2002-01-02 2002-02-01 --rerun
```

Output goes to `ruben_content/`:
```
ruben_content/2002-01-02_Ruben_content_episode_001.mp3
```

Both versions are preserved side by side — `ruben_trimmed/` keeps the full broadcast cut, `ruben_content/` has the story only.

### Tuning

| Setting | Default | Description |
|---|---|---|
| `MATCH_THRESHOLD_INTRO` | `0.25` | Minimum score to accept the story intro match |
| `MATCH_THRESHOLD_OUTRO` | `0.15` | Minimum score to accept the story outro match |
| `SEARCH_WINDOW_INTRO` | `300.0` | Seconds from start of file to search for intro |
| `SEARCH_WINDOW_OUTRO` | `120.0` | Seconds from end of file to search for outro |
| `OUTRO_TAIL_SECONDS` | `3.0` | Seconds kept after detected outro start — increase if outro is cut off too early |

### Manual review — `supertrim_review.txt`

Flagged episodes follow the same score + timestamp format as `manual_review.txt`.

---

## Step 5 — Build audiobook

Combines all episodes into a single M4B audiobook file with each episode as a named chapter. M4B is natively supported by Apple Books, VLC, Overcast, Pocketcasts, and most audiobook players.

```bash
# Build from broadcast-trimmed episodes (ruben_trimmed/)
python build_audiobook.py

# Build from story-only episodes (ruben_content/)
python build_audiobook.py --content

# Custom output filename
python build_audiobook.py --content --output Ruben_komplet.m4b

# Limit to a date range
python build_audiobook.py 2002-01-02 2002-06-01

# Preview chapter list and total duration without building
python build_audiobook.py --dry-run
python build_audiobook.py --content --dry-run
```

Each chapter is named `Episode 001 — 2002-01-02`. For ~430 episodes of ~6 minutes each, expect roughly **1.2–1.5 GB** for the complete series.

---

## Diagnostic tool — `diagnose_jingle.py`

If the trimmer is not finding a jingle (scores near zero or unexpectedly low), use this tool to visualise the full correlation curve for a single episode.

```bash
# Check where the broadcast outro jingle appears
python diagnose_jingle.py ruben_episodes/2002-01-03_0550_0003_Ruben.mp3 jingle_end_1.mp3

# Check the broadcast intro jingle
python diagnose_jingle.py ruben_episodes/2002-01-03_0550_0003_Ruben.mp3 jingle_start.mp3

# Also works on trimmed episodes for the supertrim reference clips
python diagnose_jingle.py ruben_trimmed/2002-01-03_Ruben_trimmed_episode_002.mp3 supertrim_outro.mp3
```

Prints the top 5 peaks with scores and timestamps, and saves a `correlation_plot.png` showing the score across the full episode. Requires `matplotlib`:

```bash
pip install matplotlib
```

### Interpreting results

| Pattern | Meaning |
|---|---|
| Flat line near zero everywhere | The jingle clip has a problem — re-export from Audacity with no silence padding |
| Clear peak but below threshold | Jingle is present — lower the relevant threshold in the script |
| Peak above threshold but at wrong time | Search window too short — increase `SEARCH_WINDOW_START` or `SEARCH_WINDOW_END` |
| Good peak on one variant, not others | Keep the best-scoring variant, remove the rest |

---

## Troubleshooting

**`401` / `403` on download** — Auth token has expired. Get a fresh one from your browser's Network tab and update `AUTH_TOKEN` in `download_ruben.py`.

**Trimmer or supertrimmer is very slow** — Make sure ffmpeg is on your PATH. Audio is decoded via ffmpeg at 8000 Hz for speed; falling back to libmpg123 is much slower and can hang on corrupt frames.

**Corrupt MP3 error during trimming** — ffmpeg skips bad frames and continues. If a file is too corrupt to process it will appear in the review log.

**All end jingle scores are near zero** — The reference clip may have silence at the start or be at a very different volume to the episodes. Re-export from Audacity as a plain MP3 starting exactly where the jingle begins, with no padding.

**Outro is cut off too early in supertrim** — Increase `OUTRO_TAIL_SECONDS` at the top of `supertrim_ruben.py`.
