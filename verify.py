"""Check an install, offline, on generated events.

Run after installing requirements-raw.txt and h5py. The checks cover data
handling and whether known wingbeats come back from synthetic insects; they
do not measure accuracy on real recordings. All files go to a temporary
folder that is deleted afterwards. Prints the list of checks passed; any
failure stops with an AssertionError that says what was wrong.
"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from event_input import canonical, open_recording, select_packets
from run import synthetic_events

ROOT = Path(__file__).resolve().parent


def command(*args, error=None):
    """Run run.py as a separate process, as a user would. With `error`, the
    run must fail with exit status 2 and that text in its message."""
    result = subprocess.run([sys.executable, str(ROOT / "run.py"), *map(str, args), "--no-figures"],
                            capture_output=True, text=True)
    if error:
        assert result.returncode == 2, result.stdout + result.stderr
        assert error in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def check_wingbeats(out, true_hz, tol, stroke_tol):
    """Score a synthetic run against its known wingbeats.

    Every track's wingbeat must be within `tol` (a fraction) of one of the
    true values, and each true value must be found. Per track, the median
    per-stroke rate is compared the same way; the median track must be within
    `stroke_tol`. Returns the number of wingbeat tracks and the rounded values.
    """
    tracks = pd.read_csv(out / "tracks.csv")
    wb = tracks[tracks.verdict == "wingbeat"]
    assert len(wb), "no wingbeat tracks"
    # Relative error of each track against the nearest true wingbeat.
    err = np.min(np.abs(wb.f0_wingbeat_hz.to_numpy()[:, None] - np.asarray(true_hz)[None, :])
                 / np.asarray(true_hz)[None, :], axis=1)
    assert np.all(err <= tol), "wingbeats {} not within {:.0%} of {}".format(
        sorted(wb.f0_wingbeat_hz.round(1)), tol, true_hz)
    for f in true_hz:
        assert np.any(np.abs(wb.f0_wingbeat_hz - f) <= tol * f), "{} Hz not found".format(f)
    # dropna removes each track's first stroke, which has no interval before it.
    strokes = pd.read_csv(out / "strokes.csv").dropna()
    # Both per-stroke columns: 1 / each interval, and the 5-stroke median.
    for col in ("stroke_rate_hz", "rate_median5_hz"):
        med = strokes.groupby("track_id")[col].median()
        serr = np.min(np.abs(med.to_numpy()[:, None] - np.asarray(true_hz)[None, :])
                      / np.asarray(true_hz)[None, :], axis=1)
        assert np.median(serr) <= stroke_tol, "{} medians {} far from {}".format(
            col, sorted(med.round(1)), true_hz)
    return len(wb), sorted(set(np.round(wb.f0_wingbeat_hz).astype(int)))


def main():
    """Run the five groups of checks in turn and print what passed."""
    checks = []
    with tempfile.TemporaryDirectory(prefix="insect-starter-check-") as work:
        work = Path(work)
        # 1. Known wingbeats come back, for every preset. Typical results:
        # every track's wingbeat within 0.1% (moth) and 0.3% (bee) of the
        # truth, and the median track's per-stroke rate within 1%. The
        # tolerances (1% and 3%) leave a margin of about three times.
        command("--preset", "moth", "--duration", 2, "--window", 2, "--out", work / "moth")
        n, hz = check_wingbeats(work / "moth", (38.0, 52.0), 0.01, 0.03)
        checks.append("moth preset: synthetic 38 and 52 Hz wings recovered within 1% "
                      "({} tracks: {} Hz)".format(n, hz))
        command("--preset", "bee", "--duration", 2, "--window", 2, "--out", work / "bee")
        n, hz = check_wingbeats(work / "bee", (180.0, 230.0), 0.01, 0.03)
        checks.append("bee preset: synthetic 180 and 230 Hz wings recovered within 1% "
                      "({} tracks: {} Hz)".format(n, hz))
        # The wide preset gets one slow and one fast wing in the same scene.
        command("--preset", "wide", "--duration", 2, "--window", 2, "--out", work / "wide")
        n, hz = check_wingbeats(work / "wide", (38.0, 230.0), 0.01, 0.03)
        checks.append("wide preset: synthetic 38 and 230 Hz wings recovered within 1% "
                      "({} tracks: {} Hz)".format(n, hz))
        baseline = pd.read_csv(work / "moth" / "tracks.csv")

        # 2. Every input format gives the same tracks as the in-memory demo:
        # the same events are written as NPZ (signed and 0/1 polarity), CSV,
        # HDF5 (at the root and under /events) and EVT3 RAW, then read back.
        ev = synthetic_events(2.0, 7, 320, 240, (38.0, 52.0))
        x, y, t, p = ev.T
        np.savez(work / "ev.npz", x=x, y=y, t=t, p=p, width=320, height=240)
        np.savez(work / "ev01.npz", x=x, y=y, t=t, p=(p > 0).astype(np.uint8), width=320, height=240)
        with open(work / "ev.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "t_us", "polarity"])
            w.writerows(ev.tolist())
        inputs = [(work / "ev.npz", []), (work / "ev01.npz", []),
                  (work / "ev.csv", ["--width", 320, "--height", 240])]
        try:
            import h5py
            for group_name in ("root", "events"):
                with h5py.File(work / (group_name + ".h5"), "w") as h:
                    g = h if group_name == "root" else h.create_group("events")
                    for name, v in (("x", x), ("y", y), ("t", t), ("p", p)):
                        g.create_dataset(name, data=v)
                    g.attrs["width"], g.attrs["height"] = 320, 240
                inputs.append((work / (group_name + ".h5"), []))
        except ImportError:
            checks.append("HDF5 SKIPPED: pip install h5py")
        try:
            import faery
            rec = np.zeros(len(t), dtype=faery.EVENTS_DTYPE)
            rec["t"], rec["x"], rec["y"], rec["on"] = t, x, y, p > 0
            faery.events_stream_from_array(rec, dimensions=(320, 240)).to_file(
                str(work / "ev.raw"), version="evt3", zero_t0=False)
            inputs.append((work / "ev.raw", []))
        except ImportError:
            checks.append("RAW SKIPPED: install requirements-raw.txt")
        for i, (path, flags) in enumerate(inputs):
            out = work / ("fmt{}".format(i))
            command("--input", path, "--preset", "moth", "--duration", 2, "--window", 2, *flags, "--out", out)
            got = pd.read_csv(out / "tracks.csv")
            pd.testing.assert_frame_equal(got, baseline, check_exact=False, rtol=1e-12)
        checks.append("NPZ, 0/1-polarity NPZ, CSV, HDF5 and EVT3 RAW give identical tracks: " +
                      ", ".join(path.suffix for path, _ in inputs))

        # 3. Decoders hand over fields of packed records (faery: 13-byte
        # t/x/y/on rows, so most x and y values are misaligned in memory).
        # canonical() must accept them, and boolean polarity, unchanged.
        packed = np.zeros(len(t), dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")])
        packed["t"], packed["x"], packed["y"], packed["on"] = t, x, y, p > 0
        assert not packed["x"].flags["ALIGNED"]
        rows = canonical({"x": packed["x"], "y": packed["y"], "t": packed["t"], "p": packed["on"]})
        assert np.array_equal(rows[:, 3], np.where(p > 0, 1, -1))
        checks.append("packed, misaligned decoder fields and boolean polarity accepted")

        # 4. Bad input is refused with a clear message: packets out of time
        # order, an unsorted CSV, a CSV with no sensor size, a video file, and
        # a start time after the last event.
        status = {}
        try:
            list(select_packets([ev[:100], ev[:2]], {"width": 320, "height": 240}, 0, 10, None, status))
        except ValueError as exc:
            assert "decreased" in str(exc)
        else:
            raise AssertionError("out-of-order packets accepted")
        (work / "bad.csv").write_text("x,y,t,p\n40,40,20,1\n40,40,10,-1\n")
        command("--input", work / "bad.csv", "--width", 320, "--height", 240, "--out", work / "b1",
                error="nondecreasing")
        command("--input", work / "ev.csv", "--out", work / "b2", error="sensor width missing")
        command("--input", "ordinary.mp4", "--out", work / "b3", error="ordinary video")
        command("--input", work / "ev.npz", "--start", 10, "--out", work / "b4", error="no events")
        checks.append("out-of-order packets, unsorted CSV, missing dimensions, video and empty selection rejected")

        # 5. The interactive runner, with its answers piped in: settings, start,
        # seconds. It must give the same tracks as run.py. The answers start
        # with the UTF-8 byte-order mark that PowerShell puts in front of piped
        # text, so that case is covered too.
        typed = subprocess.run([sys.executable, str(ROOT / "try_my_data.py"), str(work / "ev.npz"),
                                "--out", str(work / "typed"), "--no-open"],
                               input=b"\xef\xbb\xbfmoth\n0\n2\n", capture_output=True)
        out_text = typed.stdout.decode(errors="replace") + typed.stderr.decode(errors="replace")
        assert typed.returncode == 0, out_text
        assert "with a measured wingbeat" in out_text, out_text
        pd.testing.assert_frame_equal(pd.read_csv(work / "typed" / "tracks.csv"), baseline,
                                      check_exact=False, rtol=1e-12)
        checks.append("try_my_data.py with typed answers gives the same tracks as run.py")
    print(json.dumps({"checks_passed": checks}, indent=2))


if __name__ == "__main__":
    main()
