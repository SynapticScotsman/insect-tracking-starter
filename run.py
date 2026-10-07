"""Track insects in an event recording and measure each one's wingbeat.

The processing chain:

  events -> noise filter (an event needs a neighbour within 1 px and 1 ms)
  -> detect_blobs (insect-sized blobs in short time windows)
  -> link_tracks (blobs joined into one track per insect)
  -> per track: the wingbeat frequency (analyse_periodicity), a check of
     which multiple of the stroke rate is the true wingbeat (pose_autocorr),
     and the time of every wing stroke (stroke_times).

The recording is processed in back-to-back windows (default 4 s). A track
that crosses a window edge is cut there and appears in both windows, so
track counts are not animal counts.

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

# Settings per kind of insect. Only physical scales differ: the detector
# window is about one wing stroke of the animal, the coast is how long a
# track may go unseen before it ends, and the band brackets the wingbeat.
PRESETS = {
    # Moths and other large, slow flyers (tested on Bogong moths at a lamp):
    # 12 ms windows every 4 ms, 100 ms coast, wingbeats 18-80 Hz.
    "moth": CensusConfig(window_us=12_000, hop_us=4_000, coast_us=100_000),
    # Honey bees: 5 ms windows every 2.5 ms, 10 ms coast, wingbeats 120-320 Hz.
    "bee": CensusConfig(window_us=5_000, hop_us=2_500, coast_us=10_000,
                        fmin_hz=120.0, fmax_hz=320.0, wingbeat_band=(120.0, 320.0)),
    # Unknown flyers: wingbeats anywhere in 15-500 Hz, an 8 ms detector
    # between the moth and bee settings, 50 ms coast. Checked only on the
    # synthetic insects in verify.py. Once the wingbeats of a recording are
    # known, narrow it with --band and --window-ms.
    "wide": CensusConfig(window_us=8_000, hop_us=4_000, coast_us=50_000,
                         fmin_hz=15.0, fmax_hz=500.0, wingbeat_band=(15.0, 500.0)),
}

# Leading columns of tracks.csv, in reading order; every other column the
# census computes follows them.
TRACK_FIELDS = ["window_start_s", "track_id", "t_start_s", "t_end_s", "duration_ms",
                "verdict", "f0_wingbeat_hz", "fundamental_from", "f0_signed_hz", "snr_signed_db",
                "f0_yin_hz", "n_detections", "n_events_raw", "max_side_px", "median_side_px",
                "speed_px_s"]


def synthetic_events(duration_s, seed, width, height, f0s=(38.0, 52.0)):
    """Two flying patches that beat their 'wings' at known rates, plus noise.

    Each patch fires ON events on one side and OFF events on the other,
    swapping sides every half stroke, with the event rate following |cos| of
    the stroke phase. The side-to-side swap is what the check of which
    multiple is the true wingbeat reads; the rate gives the wingbeat
    frequency. Background noise is scattered over the whole frame.
    """
    # Events are drawn on a 100 us grid: Poisson counts per step, then a
    # random offset inside the step.
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


def configure(args):
    """The preset, with any of --band, --window-ms and --max-size laid over it.
    The detector step (hop) stays half the window."""
    cfg = replace(PRESETS[args.preset])
    if getattr(args, "band", None):
        lo, hi = args.band
        if not 0 < lo < hi:
            raise ValueError("--band needs 0 < LO < HI")
        cfg = replace(cfg, wingbeat_band=(lo, hi), fmin_hz=min(cfg.fmin_hz, lo),
                      fmax_hz=max(cfg.fmax_hz, hi))
    if getattr(args, "window_ms", None):
        if args.window_ms <= 0:
            raise ValueError("--window-ms must be positive")
        w = int(round(args.window_ms * 1e3))
        cfg = replace(cfg, window_us=w, hop_us=w // 2)
    if getattr(args, "max_size", None):
        cfg = replace(cfg, max_size_px=int(args.max_size))
    return cfg


def run(args):
    """Process the selected time range window by window and write the outputs."""
    cfg = configure(args)
    # Input: a recording, or (with no --input) synthetic insects whose
    # wingbeats are known, which is what verify.py checks against.
    if args.input:
        metadata, packets = open_recording(args.input, args.width, args.height, args.encoding)
    else:
        metadata = {"format": "synthetic", "width": 320, "height": 240}
        f0s = (38.0, 52.0) if args.preset == "moth" else (180.0, 230.0) if args.preset == "bee" \
            else (38.0, 230.0)
        packets = [synthetic_events(args.duration, args.seed, 320, 240, f0s)]
        metadata["true_wingbeats_hz"] = list(f0s)
    width, height = metadata["width"], metadata["height"]
    if args.roi:
        x0, y0, x1, y1 = args.roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROI must be x0 y0 x1 y1 inside the sensor; upper bounds are exclusive")
    # select_packets is lazy: nothing is decoded until a packet is asked for.
    # It fills `status` with the absolute start and end times (microseconds,
    # the recording's own clock) once it has seen the first event.
    status = {}
    selected = select_packets(packets, metadata, args.start, args.duration, args.roi, status)
    first = next(selected, None)
    if first is None:
        raise ValueError("no events in the selected interval/ROI; change --start, --duration or --roi")
    lower, upper = status["requested_start_us"], status["requested_end_us"]

    # The packet already pulled out to test for an empty selection goes back
    # in front of the rest.
    def chained():
        yield first
        yield from selected

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary = dict(preset=args.preset, config=asdict(cfg), input=str(args.input or "synthetic"),
                   input_metadata=metadata, selection=dict(status, roi=args.roi),
                   duration_s=args.duration, seed=args.seed, encoding=args.encoding,
                   window_s=args.window, windows=[],
                   interpretation="Counts and frequencies only; no accuracy was measured. "
                                  "Tracks are not animals: one animal can be several tracks.")
    # The three tables are written as each window finishes, so a long run
    # keeps everything done so far if it is stopped.
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
            # A nearly empty window (the tail of the range) has nothing to track.
            if len(arr) < 1_000:
                continue
            ev = EventStream(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], width, height)
            print("window {:.3f}-{:.3f} s, {:,} events".format(w0 / 1e6, (w0 + args.window * 1e6) / 1e6, len(ev)),
                  flush=True)
            # The whole algorithm: noise filter, detection, tracking and the
            # wingbeat of every track (census.run_census).
            c = census.run_census(ev, cfg, raw_path=str(args.input or ""), verbose=True)
            # Track IDs restart in every window, so every row carries the
            # window's start time; (window_start_s, track_id) names a track.
            ws0 = round(w0 / 1e6, 6)
            if wt is None:
                # The column list is only known once the first census exists:
                # the named columns first, then everything else it computed.
                rest = [k for k in c.table.columns if k not in TRACK_FIELDS]
                wt = csv.DictWriter(ft, fieldnames=TRACK_FIELDS + rest)
                wt.writeheader()
            for _, r in c.table.iterrows():
                wt.writerow(dict(r.to_dict(), window_start_s=ws0))
            for trk in c.tracks:
                for d in trk.detections:
                    wd.writerow([ws0, trk.track_id, d.t_us, round(d.x, 2), round(d.y, 2), d.size_px, d.n_events])
            # Wing strokes only for tracks with a measured wingbeat: the stroke
            # finder is tuned around that wingbeat.
            n_strokes = 0
            for _, r in c.table[c.table.verdict == "wingbeat"].iterrows():
                trk = c.track(int(r.track_id))
                st = census.stroke_times(trk.extract_events(c.ev, pad_px=c.config.pad_px), trk,
                                         float(r.f0_wingbeat_hz))
                # rate_median5_hz: the median rate of the last five strokes. It
                # ignores a single stroke counted twice or missed, and smooths
                # over wingbeat changes faster than about four per second on
                # moths. See census.running_stroke_rate and the README.
                med = census.running_stroke_rate(st, 5)
                # The first stroke has no interval before it, so its rate
                # columns are left empty.
                for k in range(len(st)):
                    iv = st[k] - st[k - 1] if k else None
                    ws.writerow([ws0, int(r.track_id), int(st[k]), int(iv) if iv else "",
                                 round(1e6 / iv, 2) if iv else "", round(float(med[k - 1]), 2) if k else ""])
                    n_strokes += 1
            if args.figures:
                census.census_figure(c, str(out / "census_{:09.3f}s.png".format(w0 / 1e6)),
                                     title="{} ({} preset), ".format(Path(summary["input"]).name, args.preset))
            # One line per window in summary.json: counts, the spread of
            # wingbeats, and the scene's strongest line near 100 Hz.
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


def track(input=None, preset="moth", out="out", start=0.0, duration=4.0, window=4.0,
          band=None, window_ms=None, max_size=None, roi=None, width=None, height=None,
          encoding=None, figures=True, seed=7):
    """Track every insect in a recording and measure its wingbeat, in one call.

        import run
        folder = run.track("recording.raw", preset="moth", start=10, duration=8)

    The arguments are the command-line options of the same name; `input=None`
    runs the synthetic demo. Writes tracks.csv, strokes.csv, detections.csv,
    summary.json and one figure per window, and returns the output folder.
    make_clip.make_clip(folder) then makes a video from it.
    """
    if duration <= 0 or window <= 0 or start < 0:
        raise ValueError("duration and window must be positive, start non-negative")
    if preset not in PRESETS:
        raise ValueError("preset must be one of " + ", ".join(sorted(PRESETS)))
    args = argparse.Namespace(
        input=Path(input) if input else None, preset=preset, out=Path(out), start=start,
        duration=duration, window=window, band=band, window_ms=window_ms, max_size=max_size,
        roi=roi, width=width, height=height, encoding=encoding, figures=figures, seed=seed)
    run(args)
    return args.out.resolve()


def main():
    """Command line: check the options, run, and turn input errors into a
    one-line message with exit status 2 instead of a traceback."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path, help=".raw/.dat/.npz/.csv/.h5 events; omit for a synthetic demo")
    p.add_argument("--preset", choices=sorted(PRESETS), default="moth",
                   help="moth: 12 ms detector, 18-80 Hz; bee: 5 ms detector, 120-320 Hz; "
                        "wide: 8 ms detector, 15-500 Hz, for unknown flyers")
    p.add_argument("--band", nargs=2, type=float, metavar=("LO", "HI"),
                   help="override the wingbeat range, Hz")
    p.add_argument("--window-ms", type=float, help="override the detector window, ms (about one wing stroke)")
    p.add_argument("--max-size", type=int, help="override the largest blob kept, px (default 90)")
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
