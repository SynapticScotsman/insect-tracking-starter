"""Make a short slow-motion video of the tracked insects from a finished run.

    python make_clip.py outputs/mybees_bee_0s                    # bees: first 0.5 s, 1/40 speed
    python make_clip.py outputs/mybees_bee_0s --start 1.2
    python make_clip.py outputs/mybees_bee_0s --follow 134       # zoom on track 134
    python make_clip.py outputs/mymoths_moth_0s                  # moths: first 2 s, 1/10 speed

What each frame shows:
- events from the last few milliseconds: blue ON (brighter), orange OFF
  (darker), cyan for events no track is near, which is where insects were
  missed;
- a circle on every track alive at that moment, with its ID (#14) and a short
  trail of where it has just been. Magenta: a track with a measured wingbeat,
  labelled with the median rate of its last five wing strokes. Grey: tracked,
  but no wingbeat measured.

It reads the run's own outputs (summary.json, tracks.csv, detections.csv,
strokes.csv) and the recording named in summary.json, so tracking is not run
again: make as many clips of one run as you like.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from event_input import open_recording, select_packets

# Colours in OpenCV's blue-green-red order.
ON_C, OFF_C, MISSED_C = (255, 155, 59), (66, 140, 255), (255, 224, 53)
WING_C, NOLINE_C, TEXT_C = (166, 61, 255), (160, 151, 138), (255, 255, 255)
NEAR_PX = 25          # an event this close to a live track counts as covered
TRAIL_US = 60_000     # length of the trail drawn behind each track


def load_run(run_dir: Path):
    """The run's settings and its three tables."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    tracks = pd.read_csv(run_dir / "tracks.csv")
    dets = pd.read_csv(run_dir / "detections.csv")
    strokes = pd.read_csv(run_dir / "strokes.csv")
    return summary, tracks, dets, strokes


def track_paths(tracks, dets, strokes):
    """Per track: its smoothed path on a 1 ms grid, verdict, size and stroke
    labels. Track IDs restart in every processing window, so a track is named
    by (window_start_s, track_id)."""
    out = {}
    for key, d in dets.groupby(["window_start_s", "track_id"]):
        d = d.sort_values("t_us")
        t = np.arange(d.t_us.iloc[0], d.t_us.iloc[-1] + 1, 1_000)
        # Detection centres jump with the wing when a detector window is
        # shorter than a stroke, so the path is averaged over 25 ms.
        x = uniform_filter1d(np.interp(t, d.t_us, d.x), 25, mode="nearest")
        y = uniform_filter1d(np.interp(t, d.t_us, d.y), 25, mode="nearest")
        out[key] = dict(t=t, x=x, y=y)
    for _, r in tracks.iterrows():
        key = (r.window_start_s, r.track_id)
        if key in out:
            out[key].update(wing=r.verdict == "wingbeat", f0=r.f0_wingbeat_hz,
                            radius=max(10.0, 0.6 * float(r.median_side_px)))
    s = strokes.dropna(subset=["rate_median5_hz"])
    for key, g in s.groupby(["window_start_s", "track_id"]):
        if key in out:
            out[key].update(st=g.stroke_t_us.to_numpy(), hz=g.rate_median5_hz.to_numpy())
    return out


def clip_events(summary, t0_us, t1_us):
    """The recording's events in [t0_us, t1_us) as an (n, 4) array x, y, t, p."""
    meta = summary["input_metadata"]
    if summary["input"] == "synthetic":
        import run
        f0s = tuple(meta["true_wingbeats_hz"])
        ev = run.synthetic_events(summary["duration_s"], summary["seed"], meta["width"],
                                  meta["height"], f0s)
        packets = [ev]
    else:
        _, packets = open_recording(summary["input"], meta["width"], meta["height"],
                                    summary.get("encoding"))
    # select_packets counts its start from the recording's first event.
    origin = summary["selection"]["recording_origin_us"]
    status = {}
    sel = list(select_packets(packets, meta, (t0_us - origin) / 1e6, (t1_us - t0_us) / 1e6,
                              None, status))
    return np.concatenate(sel) if sel else np.zeros((0, 4), np.int64)


def open_writer(path: Path, size, fps):
    """H.264 MP4 through imageio-ffmpeg, which plays everywhere; OpenCV's own
    MP4 writer as a fallback when it is not installed."""
    w, h = size
    try:
        import imageio_ffmpeg
        gen = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264",
                                          pix_fmt_in="bgr24", output_params=["-crf", "20"],
                                          macro_block_size=1)
        gen.send(None)
        return lambda frame: gen.send(np.ascontiguousarray(frame)), gen.close
    except ImportError:
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        return vw.write, vw.release


def make_clip(run_dir, start=None, seconds=None, slow=None, follow=None, zoom_px=150,
              fps=25, out=None) -> Path:
    """Write an MP4 of a finished run and return its path.

    start    seconds after the run's start (default 0, or the followed track's start)
    seconds  how much recording time the clip covers
    slow     slow-motion factor: 10 plays 1 s of recording in 10 s. Bees beat
             about 230 times a second and need about 40; moths about 10.
             By default, and for `seconds`, the run's preset decides: a bee
             run gets 0.5 s at 1/40, any other 2 s at 1/10, both 20 s of video.
    follow   a track ID to keep centred and zoomed in on, zoom_px pixels above and
             below it. With several processing windows, the first window that has
             that ID is used.
    """
    run_dir = Path(run_dir)
    summary, tracks, dets, strokes = load_run(run_dir)
    bee = summary.get("preset") == "bee"
    slow = slow or (40.0 if bee else 10.0)
    seconds = seconds or (0.5 if bee else 2.0)
    paths = track_paths(tracks, dets, strokes)
    W, H = summary["input_metadata"]["width"], summary["input_metadata"]["height"]
    run0 = summary["selection"]["requested_start_us"]
    fkey = None
    if follow is not None:
        keys = sorted(k for k in paths if int(k[1]) == int(follow))
        if not keys:
            raise ValueError("no track {} in this run".format(follow))
        fkey = keys[0]
    if start is None:
        t0 = int(paths[fkey]["t"][0]) if fkey else run0
    else:
        t0 = run0 + int(round(start * 1e6))
    t1 = t0 + int(round(seconds * 1e6))
    if fkey and start is None:
        t1 = min(t1, int(paths[fkey]["t"][-1]))
    # One frame shows `step` of recording time; events stay on screen for 1.5
    # steps (at least 1 ms) so a wing is visible as a stroke, not single dots.
    step = 1e6 / (fps * slow)
    accum = max(1.5 * step, 1_000.0)
    ev = clip_events(summary, int(t0 - accum), t1)
    if not len(ev):
        raise ValueError("no events between {:.3f} and {:.3f} s".format(t0 / 1e6, t1 / 1e6))
    x, y, t, p = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]

    # Output size: the sensor size, made even and at least 720 px tall, so a
    # small sensor is enlarged; a followed track gets a 16:9 1280 x 720 view.
    scale = max(1, int(np.ceil(720 / H)))
    size = (1280, 720) if fkey else ((W * scale) // 2 * 2, (H * scale) // 2 * 2)
    out = Path(out) if out else run_dir / "clip_{:.3f}s{}.mp4".format(
        (t0 - run0) / 1e6, "_track{}".format(follow) if fkey else "")
    write, close = open_writer(out, size, fps)
    n_frames = 0
    for tc in np.arange(t0 + accum, t1, step):
        s, e = np.searchsorted(t, [tc - accum, tc])
        # Tracks alive at this instant, and where each one is now.
        live = [(k, v) for k, v in paths.items() if v["t"][0] <= tc <= v["t"][-1]]
        centres = np.array([[np.interp(tc, v["t"], v["x"]), np.interp(tc, v["t"], v["y"])]
                            for _, v in live]) if live else np.zeros((0, 2))
        # Event layer: count events per pixel by colour, brighter where more fired.
        xs, ys, ps = x[s:e], y[s:e], p[s:e]
        covered = np.zeros(e - s, bool)
        for cx, cy in centres:
            covered |= (xs - cx) ** 2 + (ys - cy) ** 2 <= NEAR_PX ** 2
        img = np.zeros((H, W, 3), np.float32)
        for sel, col in ((covered & (ps > 0), ON_C), (covered & (ps < 0), OFF_C), (~covered, MISSED_C)):
            cnt = np.bincount(ys[sel] * W + xs[sel], minlength=W * H).reshape(H, W)
            img += np.clip(cnt / 2.0, 0, 1)[..., None] * np.array(col, np.float32) * 0.85
        frame = np.clip(img, 0, 255).astype(np.uint8)
        frame = cv2.resize(frame, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        # Track layer, drawn in the enlarged image so lines and text stay sharp.
        n_wing = 0
        for (key, v), (cx, cy) in zip(live, centres):
            col = WING_C if v.get("wing") else NOLINE_C
            m = (v["t"] >= tc - TRAIL_US) & (v["t"] <= tc)
            trail = np.stack([v["x"][m], v["y"][m]], 1) * scale
            if len(trail) > 1:
                cv2.polylines(frame, [trail.astype(np.int32)], False, col, 2, cv2.LINE_AA)
            c = (int(cx * scale), int(cy * scale))
            r = int(v.get("radius", 10.0) * scale)
            cv2.circle(frame, c, r, col, 2, cv2.LINE_AA)
            cv2.putText(frame, "#{}".format(int(key[1])), (c[0] + r, c[1] + r + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_C, 1, cv2.LINE_AA)
            if v.get("wing"):
                n_wing += 1
                hz = v.get("f0")
                if "st" in v:
                    k = int(np.searchsorted(v["st"], tc, side="right")) - 1
                    if k >= 0:
                        hz = v["hz"][k]
                cv2.putText(frame, "{:.0f} Hz".format(hz), (c[0] + r, c[1] - r),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        if fkey:
            v = paths[fkey]
            fx = np.interp(tc, v["t"], v["x"]) * scale
            fy = np.interp(tc, v["t"], v["y"]) * scale
            hh = zoom_px * scale
            hw = hh * 16 // 9
            pad = cv2.copyMakeBorder(frame, hh, hh, hw, hw, cv2.BORDER_CONSTANT, value=0)
            crop = pad[int(fy):int(fy) + 2 * hh, int(fx):int(fx) + 2 * hw]
            frame = cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
        else:
            frame = frame[:size[1], :size[0]]
        title = "t = {:.3f} s   1/{:g} speed   {} tracked, {} with a wingbeat   " \
                "number = median rate of the last 5 strokes".format(tc / 1e6, slow, len(live), n_wing)
        cv2.rectangle(frame, (0, 0), (size[0], 26), (0, 0, 0), -1)
        cv2.putText(frame, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_C, 1, cv2.LINE_AA)
        write(frame)
        n_frames += 1
    close()
    print("{} frames, {:.1f} s of video -> {}".format(n_frames, n_frames / fps, out))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path, help="output folder of run.py or try_my_data.py")
    p.add_argument("--start", type=float, help="seconds after the run's start")
    p.add_argument("--seconds", type=float,
                   help="recording time covered (default: 0.5 for a bee run, 2 otherwise)")
    p.add_argument("--slow", type=float,
                   help="slow-motion factor (default: 40 for a bee run, 10 otherwise)")
    p.add_argument("--follow", type=int, help="track ID to zoom in on")
    p.add_argument("--zoom-px", type=int, default=150, help="half-height of the followed view, px")
    p.add_argument("--out", type=Path, help="output .mp4 (default: in the run folder)")
    a = p.parse_args()
    try:
        make_clip(a.run_dir, a.start, a.seconds, a.slow, a.follow, a.zoom_px, out=a.out)
    except (OSError, ValueError) as exc:
        p.exit(2, "error: {}\n".format(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
