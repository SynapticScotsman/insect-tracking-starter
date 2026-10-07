"""Choose an event recording; this tracks every insect in it and measures each
one's wingbeat, then prints what it found.

It asks the questions that run.py takes as flags, runs exactly the same code
(run.run), and reads the results back for you:

    python try_my_data.py                    # file dialog, then questions
    python try_my_data.py recording.raw      # skip the dialog
    python try_my_data.py --no-dialog        # type or paste the path instead

Every question shows its default in [brackets]; press Enter to take it.
Run verify.py once after installing: it checks the install on synthetic
insects whose wingbeats are known. This script checks nothing; it measures.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import run as runner

HERE = Path(__file__).resolve().parent
FORMATS = (".raw", ".kai")      # Prophesee .raw, Kairos .raw.kai
ANIMALS = {
    "moth": "moths and other slow flyers: 12 ms detector, wingbeats 18-80 Hz",
    "bee": "bees: 5 ms detector, wingbeats 120-320 Hz",
    "wide": "unknown flyers: 8 ms detector, wingbeats 15-500 Hz",
}


def ask(question, default, cast=str, ok=lambda v: True, why=""):
    """One question, re-asked until the answer parses and passes `ok`."""
    while True:
        text = input("{} [{}]: ".format(question, default)).strip()
        if not text:
            return default
        try:
            value = cast(text)
        except ValueError:
            print("  could not read {!r}".format(text))
            continue
        if ok(value):
            return value
        print("  " + why)


def choose_file(no_dialog: bool) -> Path:
    """A file dialog when one can be shown, otherwise a typed path."""
    if not no_dialog:
        try:
            # tkinter ships with most Python installs. The hidden root window
            # is needed to host the dialog; topmost keeps the dialog from
            # opening behind the terminal.
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            chosen = filedialog.askopenfilename(
                title="Choose an event recording",
                filetypes=[("Raw event recordings (.raw, .raw.kai)", " ".join("*" + e for e in FORMATS)),
                           ("All files", "*.*")])
            root.destroy()
            if chosen:
                return Path(chosen)
            print("No file chosen in the dialog.")
        except Exception as exc:          # no tkinter, or no display to show it on
            print("(No file dialog available here: {})".format(exc))
    # Quotes are stripped because "Copy as path" in Windows Explorer adds them.
    while True:
        text = input("Path to the event recording: ").strip().strip('"').strip("'")
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_file():
            return path
        print("  no file at {}".format(path))


def summarise(out: Path, preset: str) -> None:
    """Read the outputs back and print the wingbeat of every tracked insect."""
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if not summary["windows"]:
        print("\nNo window held the 1,000 events needed to process it. "
              "Try a different start or a longer duration.")
        return
    tracks = pd.read_csv(out / "tracks.csv")
    strokes = pd.read_csv(out / "strokes.csv").dropna(subset=["stroke_rate_hz"])
    wb = tracks[tracks.verdict == "wingbeat"].copy()
    print("\n" + "=" * 72)
    print("{} tracks in {} window(s); {} with a measured wingbeat ({} preset)".format(
        len(tracks), len(summary["windows"]), len(wb), preset))
    print("Tracks are not animals: one insect can be several tracks.")
    if not len(wb):
        print("No track held a wingbeat long and clearly enough to measure.")
        return
    print("Wingbeat across tracks, p10 / p50 / p90: {:.1f} / {:.1f} / {:.1f} Hz".format(
        *np.percentile(wb.f0_wingbeat_hz, [10, 50, 90])))
    # Per track, the median of its per-stroke rates and its number of
    # intervals. Track IDs restart in each window, so the key is the pair
    # (window_start_s, track_id).
    per = strokes.groupby(["window_start_s", "track_id"]).stroke_rate_hz
    wb = wb.join(per.median().rename("stroke_median_hz"), on=["window_start_s", "track_id"])
    wb = wb.join(per.size().rename("intervals"), on=["window_start_s", "track_id"])
    wb = wb.sort_values("duration_ms", ascending=False)
    show = wb.head(15)
    print("\nLongest {} of {} wingbeat tracks:".format(len(show), len(wb)))
    print("  {:>9} {:>6} {:>10} {:>12} {:>10} {:>14} {:>9}".format(
        "window s", "track", "length ms", "wingbeat Hz", "from", "per-stroke Hz", "strokes"))
    # A track with no measured stroke shows "-"; strokes = intervals + 1.
    for _, r in show.iterrows():
        print("  {:>9.3f} {:>6d} {:>10.0f} {:>12.1f} {:>10} {:>14} {:>9}".format(
            r.window_start_s, int(r.track_id), r.duration_ms, r.f0_wingbeat_hz, r.fundamental_from,
            "{:.1f}".format(r.stroke_median_hz) if np.isfinite(r.stroke_median_hz) else "-",
            int(r.intervals) + 1 if np.isfinite(r.intervals) else 0))
    print("\nwingbeat Hz: the track's average. per-stroke Hz: median of 1 / each "
          "stroke interval (strokes.csv has every stroke).")


def open_file(path: Path) -> None:
    """Open a file in the system's default viewer (Windows, macOS or Linux)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))              # noqa: S606 (the user's own output)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print("  could not open it here ({}); it is at {}".format(exc, path))


def main() -> int:
    """Ask for the file and settings, run, then summarise the outputs."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("input", nargs="?", type=Path, help="event recording; omit to be asked")
    p.add_argument("--no-dialog", action="store_true", help="type the path instead of a dialog")
    p.add_argument("--out", type=Path, help="output folder (default: outputs/NAME_PRESET_STARTs)")
    p.add_argument("--no-open", action="store_true", help="do not offer to open the figure")
    a = p.parse_args()
    # Answers can be piped in instead of typed. PowerShell puts a UTF-8
    # byte-order mark in front of piped text, which would spoil the first
    # answer; reading piped input as utf-8-sig drops it. Typing at a
    # terminal is unaffected.
    if not sys.stdin.isatty():
        sys.stdin.reconfigure(encoding="utf-8-sig", errors="replace")
    try:
        print(__doc__.split("\n\n")[0] + "\n")
        path = a.input if a.input else choose_file(a.no_dialog)
        if not path.is_file():
            p.exit(2, "error: no file at {}\n".format(path))
        if path.suffix.lower() not in FORMATS:
            p.exit(2, "error: {} is not a .raw or .raw.kai event recording\n".format(path.name))
        print("File: {}".format(path))
        for name, text in ANIMALS.items():
            print("  {:<5} {}".format(name, text))
        preset = ask("Which settings", "moth", str.lower, lambda v: v in runner.PRESETS,
                     "answer one of: " + ", ".join(sorted(runner.PRESETS)))
        width = height = None
        start = ask("Start, seconds after the first event", 0.0, float, lambda v: v >= 0,
                    "must be 0 or more")
        duration = ask("Seconds to process (a busy 4 s takes 1-2 minutes)", 4.0, float,
                       lambda v: v > 0, "must be more than 0")
        out = a.out or HERE / "outputs" / "{}_{}_{:g}s".format(path.stem, preset, start)
        args = argparse.Namespace(input=path, preset=preset, out=out, start=start,
                                  duration=duration, window=4.0, roi=None, width=width,
                                  height=height, encoding=None, figures=True, seed=7)
        print("\nRunning the {} settings on {:g}-{:g} s. Outputs go to {}\n".format(
            preset, start, start + duration, out))
        runner.run(args)
    except (EOFError, KeyboardInterrupt):
        print("\nStopped.")
        return 1
    except (OSError, ValueError, ImportError) as exc:
        p.exit(2, "error: {}\n".format(exc))
    out = out.resolve()
    summarise(out, preset)
    figures = sorted(out.glob("census_*.png"))
    print("\nWritten to {}:".format(out))
    for name in ["tracks.csv", "strokes.csv", "detections.csv", "summary.json"] + [f.name for f in figures]:
        print("  " + name)
    if figures and not a.no_open:
        try:
            if ask("Open the first figure now? (y/n)", "y", str.lower, lambda v: v in ("y", "n"),
                   "answer y or n") == "y":
                open_file(figures[0])
        except EOFError:
            pass
    # Optional video. Answers run out when they are piped in, which ends here.
    try:
        make_video(out, preset, a.no_open)
    except EOFError:
        pass
    return 0


def make_video(out: Path, preset: str, no_open: bool) -> None:
    """Offer a short slow-motion clip of the run (make_clip.py)."""
    if ask("Make a short video clip? (y/n)", "n", str.lower, lambda v: v in ("y", "n"),
           "answer y or n") != "y":
        return
    import make_clip
    # Bees beat about 230 times a second, so they need a slower, shorter clip.
    fast = preset == "bee"
    follow = ask("Track ID to follow (Enter for the whole view)", "", str,
                 lambda v: v == "" or v.isdigit(), "answer a track number or press Enter")
    start = ask("Start, seconds after the run's start", 0.0, float, lambda v: v >= 0,
                "must be 0 or more")
    seconds = ask("Seconds of recording to show", 0.5 if fast else 2.0, float, lambda v: v > 0,
                  "must be more than 0")
    slow = ask("Slow-motion factor", 40.0 if fast else 10.0, float, lambda v: v >= 1,
               "must be 1 or more")
    try:
        clip = make_clip.make_clip(out, start=start, seconds=seconds, slow=slow,
                                   follow=int(follow) if follow else None)
    except ValueError as exc:
        print("  no clip: {}".format(exc))
        return
    if not no_open:
        open_file(clip)


if __name__ == "__main__":
    sys.exit(main())
