"""Track insects in an event recording and measure each one's wingbeat.

This is the chain that produced the Bogong moth results and the one the bee
work used, copied from the Ecology repository by analysis/export_starter.py
(see PROVENANCE.json), not a re-implementation:

  events -> noise filter (background activity, 1 ms) -> detect_blobs ->
  link_tracks -> per track: wingbeat line (analyse_periodicity), octave
  settled by where the animal's events repeat (pose_autocorr), and the
  time of every wing stroke (stroke_times).

The recording is processed in back-to-back windows (default 4 s, as the
whole-recording sweep did). A track that crosses a window edge is cut there
and appears in both windows, so track counts are not animal counts.

Start reading at run(), then census.run_census.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np

import census
from census import CensusConfig
from event_input import open_recording, select_packets
from insect_evs.events import EventStream

# The settings each result was produced with. Only physical scales differ:
# the detector window is about one wing stroke of the animal, and the
# wingbeat band brackets its wingbeat.
PRESETS = {
    # Bogong moths at a lamp (recording_2026-10-02_21-38-40, reel 5):
    # recording_census.py --window-ms 12 --hop-ms 4 --coast-ms 100, other
    # settings at their defaults (wide 15-500 Hz line search, 18-80 Hz
    # wingbeat candidates).
    "moth": CensusConfig(window_us=12_000, hop_us=4_000, coast_us=100_000),
    # Honey bees: the frozen detection recipe (5 ms / 2.5 ms windows, 10 ms
    # coast) and the explicit 120-320 Hz honey-bee band. The bee results in
    # the repository also used a calibrated front end (Conv1 or the frozen
    # threshold map) that needs per-site calibration; it is not included.
    "bee": CensusConfig(window_us=5_000, hop_us=2_500, coast_us=10_000,
                        fmin_hz=120.0, fmax_hz=320.0, wingbeat_band=(120.0, 320.0)),
}

# Leading columns of tracks.csv, in reading order; every other census column
# follows, so nothing the census computes is hidden from the output.
TRACK_FIELDS = ["window_start_s", "track_id", "t_start_s", "t_end_s", "duration_ms",
                "verdict", "f0_wingbeat_hz", "fundamental_from", "f0_signed_hz", "snr_signed_db",
                "f0_yin_hz", "n_detections", "n_events_raw", "max_side_px", "median_side_px",
                "speed_px_s"]


def synthetic_events(duration_s, seed, width, height, f0s=(38.0, 52.0)):
    """Two flying patches that beat their 'wings' at known rates, plus noise.

    Each patch fires ON events on one side and OFF events on the other,
    swapping sides every half stroke, with the event rate following |cos| of
    the stroke phase: the spatial alternation is what the per-track octave
    test reads, and the rate gives the wingbeat line. No recorded data.
    """
    rng = np.random.default_rng(seed)
    dt_us = 100
    n = int(duration_s * 1e6 / dt_us)
    tg = np.arange(n) * dt_us
    rows = []
    for k, f0 in enumerate(f0s):
        phase = 2 * np.pi * f0 * tg / 1e6 + k
        lam = 25_000.0 * np.abs(np.cos(phase)) / np.mean(np.abs(np.cos(phase)))
        counts = rng.poisson(lam * dt_us / 1e6)
        idx = np.repeat(np.arange(n), counts)
        t = tg[idx] + rng.integers(0, dt_us, len(idx))
        frac = t / (duration_s * 1e6)
        cx = width * (0.15 + 0.7 * frac)
        cy = height * (0.3 + 0.4 * k) + 6 * np.sin(2 * np.pi * 1.5 * frac)
        on = np.cos(phase[idx]) >= 0
        side = np.where(on, -1, 1)
        x = np.clip(np.round(cx + side * rng.uniform(1, 6, len(idx))), 0, width - 1)
        y = np.clip(np.round(cy + rng.normal(0, 2.5, len(idx))), 0, height - 1)
        rows.append(np.column_stack([x, y, t, np.where(on, 1, -1)]))
    m = rng.poisson(2_000 * duration_s)
    rows.append(np.column_stack([rng.integers(0, width, m), rng.integers(0, height, m),
                                 rng.integers(0, int(duration_s * 1e6), m), rng.choice([-1, 1], m)]))
    ev = np.concatenate(rows).astype(np.int64)
    return ev[np.argsort(ev[:, 2], kind="stable")]


def windows_of(selected, start_us, end_us, window_us):
    """Group selected packets into back-to-back windows [w, w + window_us)."""
    buf, w0 = [], start_us
    for packet in selected:
        while len(packet):
            cut = np.searchsorted(packet[:, 2], w0 + window_us)
            if cut:
                buf.append(packet[:cut])
            if cut == len(packet):
                break
            yield w0, np.concatenate(buf) if buf else np.zeros((0, 4), np.int64)
            buf, w0 = [], w0 + window_us
            packet = packet[cut:]
    if buf:
        yield w0, np.concatenate(buf)


def run(args):
    cfg = replace(PRESETS[args.preset])
    if args.input:
        metadata, packets = open_recording(args.input, args.width, args.height, args.encoding)
    else:
        metadata = {"format": "synthetic", "width": 320, "height": 240}
        f0s = (38.0, 52.0) if args.preset == "moth" else (180.0, 230.0)
        packets = [synthetic_events(args.duration, args.seed, 320, 240, f0s)]
        metadata["true_wingbeats_hz"] = list(f0s)
    width, height = metadata["width"], metadata["height"]
    if args.roi:
        x0, y0, x1, y1 = args.roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROI must be x0 y0 x1 y1 inside the sensor; upper bounds are exclusive")
    status = {}
    selected = select_packets(packets, metadata, args.start, args.duration, args.roi, status)
    first = next(selected, None)
    if first is None:
        raise ValueError("no events in the selected interval/ROI; change --start, --duration or --roi")
    lower, upper = status["requested_start_us"], status["requested_end_us"]

    def chained():
        yield first
        yield from selected

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary = dict(preset=args.preset, config=asdict(cfg), input=str(args.input or "synthetic"),
                   input_metadata=metadata, selection=dict(status, roi=args.roi),
                   window_s=args.window, windows=[],
                   interpretation="Counts and frequencies only; no accuracy was measured. "
                                  "Tracks are not animals: one animal can be several tracks.")
    with open(out / "tracks.csv", "w", newline="", encoding="utf-8") as ft, \
            open(out / "detections.csv", "w", newline="", encoding="utf-8") as fd, \
            open(out / "strokes.csv", "w", newline="", encoding="utf-8") as fs:
        wt = None
        wd = csv.writer(fd)
        wd.writerow(["window_start_s", "track_id", "t_us", "x", "y", "size_px", "n_events"])
        ws = csv.writer(fs)
        ws.writerow(["window_start_s", "track_id", "stroke_t_us", "interval_us", "stroke_rate_hz",
                     "rate_median5_hz"])
        for w0, arr in windows_of(chained(), lower, upper, int(args.window * 1e6)):
            if len(arr) < 1_000:
                continue
            ev = EventStream(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], width, height)
            print("window {:.3f}-{:.3f} s, {:,} events".format(w0 / 1e6, (w0 + args.window * 1e6) / 1e6, len(ev)),
                  flush=True)
            c = census.run_census(ev, cfg, raw_path=str(args.input or ""), verbose=True)
            ws0 = round(w0 / 1e6, 6)
            if wt is None:
                rest = [k for k in c.table.columns if k not in TRACK_FIELDS]
                wt = csv.DictWriter(ft, fieldnames=TRACK_FIELDS + rest)
                wt.writeheader()
            for _, r in c.table.iterrows():
                wt.writerow(dict(r.to_dict(), window_start_s=ws0))
            for trk in c.tracks:
                for d in trk.detections:
                    wd.writerow([ws0, trk.track_id, d.t_us, round(d.x, 2), round(d.y, 2), d.size_px, d.n_events])
            n_strokes = 0
            for _, r in c.table[c.table.verdict == "wingbeat"].iterrows():
                trk = c.track(int(r.track_id))
                st = census.stroke_times(trk.extract_events(c.ev, pad_px=c.config.pad_px), trk,
                                         float(r.f0_wingbeat_hz))
                # rate_median5_hz: the median rate of the last five strokes, the
                # number the research clips show. It ignores a stroke counted
                # twice or missed (gross errors on real moths 10.7% -> 3.3%,
                # bees 7.1% -> 0.9%) and blurs swings faster than about 4 per
                # second on moths. See census.running_stroke_rate.
                med = census.running_stroke_rate(st, 5)
                for k in range(len(st)):
                    iv = st[k] - st[k - 1] if k else None
                    ws.writerow([ws0, int(r.track_id), int(st[k]), int(iv) if iv else "",
                                 round(1e6 / iv, 2) if iv else "", round(float(med[k - 1]), 2) if k else ""])
                    n_strokes += 1
            if args.figures:
                census.census_figure(c, str(out / "census_{:09.3f}s.png".format(w0 / 1e6)),
                                     title="{} ({} preset), ".format(Path(summary["input"]).name, args.preset))
            wb = c.table[c.table.verdict == "wingbeat"]
            summary["windows"].append(dict(
                start_s=ws0, events=len(ev), tracks=len(c.table), wingbeat_tracks=len(wb),
                wingbeat_p10_p50_p90_hz=[round(float(v), 2) for v in np.percentile(wb.f0_wingbeat_hz, [10, 50, 90])]
                if len(wb) else [], strokes=n_strokes, lamp_hz=c.scene_lines.get("lamp_hz")))
    summary["processing_seconds"] = round(time.perf_counter() - started, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    nt = sum(w["tracks"] for w in summary["windows"])
    nw = sum(w["wingbeat_tracks"] for w in summary["windows"])
    print("{} windows, {} tracks, {} with a wingbeat. Outputs in {}".format(len(summary["windows"]), nt, nw, out))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path, help=".raw/.dat/.npz/.csv/.h5 events; omit for a synthetic demo")
    p.add_argument("--preset", choices=sorted(PRESETS), default="moth",
                   help="moth: 12 ms detector, 18-80 Hz; bee: 5 ms detector, 120-320 Hz")
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--start", type=float, default=0.0, help="seconds after the recording's first event")
    p.add_argument("--duration", type=float, default=4.0, help="seconds to process")
    p.add_argument("--window", type=float, default=4.0, help="processing window, seconds")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--encoding", choices=["evt2", "evt21", "evt3", "dat"])
    p.add_argument("--no-figures", dest="figures", action="store_false",
                   help="skip the per-window census figure")
    p.add_argument("--seed", type=int, default=7, help="synthetic demo only")
    a = p.parse_args()
    try:
        if a.duration <= 0 or a.window <= 0 or a.start < 0:
            raise ValueError("--duration and --window must be positive, --start non-negative")
        if not a.input and a.start:
            raise ValueError("--start applies to recording input")
        return run(a)
    except (OSError, ValueError, ImportError) as exc:
        p.exit(2, "error: {}\n".format(exc))


if __name__ == "__main__":
    sys.exit(main())
