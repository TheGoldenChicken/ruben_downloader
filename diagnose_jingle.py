#!/usr/bin/env python3
"""
Jingle correlation diagnostic
──────────────────────────────
Shows where (if anywhere) the jingle appears in a given episode,
by plotting the full correlation curve and printing the top 5 peaks.

Usage:
    python diagnose_jingle.py <episode.mp3> <jingle.mp3>

Examples:
    python diagnose_jingle.py ruben_episodes/2002-01-03_0550_0003_Ruben.mp3 jingle_end.mp3
    python diagnose_jingle.py ruben_episodes/2002-01-03_0550_0003_Ruben.mp3 jingle_start.mp3

Requirements:
    pip install numpy scipy matplotlib
    ffmpeg on PATH
"""

import sys
import subprocess
import numpy as np
from scipy.signal import fftconvolve

SAMPLE_RATE = 8000


def ffmpeg_decode(path: str, offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(offset)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return np.frombuffer(r.stdout, dtype=np.float32)


def norm(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    std = x.std()
    return x / std if std > 1e-6 else x


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    episode_path = sys.argv[1]
    jingle_path  = sys.argv[2]

    print(f"Episode : {episode_path}")
    print(f"Jingle  : {jingle_path}")
    print(f"Decoding at {SAMPLE_RATE} Hz …")

    episode = ffmpeg_decode(episode_path)
    jingle  = ffmpeg_decode(jingle_path)

    ep_dur  = len(episode) / SAMPLE_RATE
    ji_dur  = len(jingle)  / SAMPLE_RATE
    print(f"Episode duration : {ep_dur:.1f}s")
    print(f"Jingle duration  : {ji_dur:.1f}s\n")

    if len(jingle) == 0:
        print("ERROR: jingle decoded to zero samples — check the file.")
        sys.exit(1)
    if len(episode) < len(jingle):
        print("ERROR: episode is shorter than the jingle.")
        sys.exit(1)

    # Full correlation across entire episode
    corr = fftconvolve(norm(episode), norm(jingle)[::-1], mode="valid")
    corr_norm = np.abs(corr) / len(jingle)

    # Time axis
    times = np.arange(len(corr_norm)) / SAMPLE_RATE

    # Top 5 peaks
    top_idx = np.argsort(corr_norm)[-5:][::-1]
    print("Top 5 correlation peaks:")
    print(f"  {'Time':>8}   {'Score':>7}")
    print(f"  {'────────':>8}   {'───────':>7}")
    for i in top_idx:
        print(f"  {times[i]:>7.2f}s   {corr_norm[i]:>7.4f}")

    overall_max = corr_norm.max()
    print(f"\nOverall max score : {overall_max:.4f}")
    print(f"Score at t=0      : {corr_norm[0]:.4f}")

    if overall_max < 0.1:
        print("\n⚠ Very low scores everywhere — the jingle clip likely has a problem.")
        print("  Try re-exporting a clean 3–5 second section with no silence padding.")
    elif overall_max >= 0.4:
        best_t = times[np.argmax(corr_norm)]
        print(f"\n✓ Good match found at {best_t:.2f}s (score {overall_max:.4f})")
        print(f"  Content would start/end around {best_t + ji_dur:.2f}s")
    else:
        print("\n~ Weak match — may work with a lower MATCH_THRESHOLD, or try a better clip.")

    # Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 4))
        plt.plot(times, corr_norm, linewidth=0.8, color="steelblue")
        plt.axhline(0.4, color="green",  linestyle="--", linewidth=1, label="threshold 0.4")
        plt.axhline(0.3, color="orange", linestyle="--", linewidth=1, label="threshold 0.3")
        plt.xlabel("Time in episode (seconds)")
        plt.ylabel("Normalised correlation score")
        plt.title(f"Jingle match: {jingle_path}  vs  {episode_path}")
        plt.legend()
        plt.tight_layout()
        out = "correlation_plot.png"
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
        plt.show()
    except ImportError:
        print("\n(Install matplotlib to get a plot: pip install matplotlib)")


if __name__ == "__main__":
    main()
