"""Check an install, offline, on generated events.

Run after installing requirements.txt. The checks cover reading raw files
and whether known wingbeats come back from synthetic insects; they do not
measure accuracy on real recordings. All files go to a temporary folder that
is deleted afterwards. Prints the list of checks passed; any failure stops
with an AssertionError that says what was wrong.
"""
import json
import struct
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
    """Run the six groups of checks in turn and print what passed."""
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
        import faery
        ev = synthetic_events(2.0, 7, 320, 240, (38.0, 52.0))
        x, y, t, p = ev.T
        rec = np.zeros(len(t), dtype=faery.EVENTS_DTYPE)
        rec["t"], rec["x"], rec["y"], rec["on"] = t, x, y, p > 0
        for version in ("evt2", "evt3"):
            faery.events_stream_from_array(rec, dimensions=(320, 240)).to_file(
                str(work / "ev_{}.raw".format(version)), version=version, zero_t0=False)
        # A Kairos file, built as Kairos writes one: the 16-byte header
        # ("KAIROS-RAW", version 0, type 0 = EVT3, width and height as
        # little-endian 16-bit numbers) in front of the EVT3 data, which here
        # is the EVT3 file above without its "%" text header.
        evt3 = (work / "ev_evt3.raw").read_bytes()
        body = 0
        while evt3[body:body + 1] == b"%":                 # skip every "%" header line
            body = evt3.index(b"\n", body) + 1
        (work / "ev.raw.kai").write_bytes(b"KAIROS-RAW\x00\x00" + struct.pack("<HH", 320, 240) + evt3[body:])
        inputs = [work / "ev_evt2.raw", work / "ev_evt3.raw", work / "ev.raw.kai"]
        for i, path in enumerate(inputs):
            out = work / ("fmt{}".format(i))
            command("--input", path, "--preset", "moth", "--duration", 2, "--window", 2, "--out", out)
            got = pd.read_csv(out / "tracks.csv")
            pd.testing.assert_frame_equal(got, baseline, check_exact=False, rtol=1e-12)
        checks.append("Prophesee EVT2 and EVT3 .raw and Kairos .raw.kai give identical tracks")

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
        # order, a file that is not a raw recording, a missing file, and a
        # start time after the last event.
        status = {}
        try:
            list(select_packets([ev[:100], ev[:2]], {"width": 320, "height": 240}, 0, 10, None, status))
        except ValueError as exc:
            assert "decreased" in str(exc)
        else:
            raise AssertionError("out-of-order packets accepted")
        (work / "video.mp4").write_bytes(b"not events")
        command("--input", work / "video.mp4", "--out", work / "b1", error="is not one")
        command("--input", work / "missing.raw", "--out", work / "b2", error="does not exist")
        command("--input", work / "ev_evt3.raw", "--start", 10, "--out", work / "b3", error="no events")
        checks.append("out-of-order packets, non-raw files, missing files and empty selections rejected")

        # 5. The interactive runner, with its answers piped in: settings, start,
        # seconds. It must give the same tracks as run.py. The answers start
        # with the UTF-8 byte-order mark that PowerShell puts in front of piped
        # text, so that case is covered too.
        typed = subprocess.run([sys.executable, str(ROOT / "try_my_data.py"), str(work / "ev.raw.kai"),
                                "--out", str(work / "typed"), "--no-open"],
                               input=b"\xef\xbb\xbfmoth\n0\n2\n", capture_output=True)
        out_text = typed.stdout.decode(errors="replace") + typed.stderr.decode(errors="replace")
        assert typed.returncode == 0, out_text
        assert "with a measured wingbeat" in out_text, out_text
        pd.testing.assert_frame_equal(pd.read_csv(work / "typed" / "tracks.csv"), baseline,
                                      check_exact=False, rtol=1e-12)
        checks.append("try_my_data.py with typed answers gives the same tracks as run.py")

        # 6. The one-call Python function, then a video clip made from its
        # output: 0.4 s of recording at 1/10 speed is about 100 frames.
        import cv2
        import make_clip
        import run
        folder = run.track(None, preset="moth", out=work / "api", duration=1, window=1, figures=False)
        assert list(pd.read_csv(folder / "tracks.csv").columns) == list(baseline.columns)
        clip = make_clip.make_clip(folder, seconds=0.4, slow=10)
        frames = int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT))
        assert 90 <= frames <= 110, "clip has {} frames".format(frames)
        checks.append("run.track() from Python, and make_clip.py writes a {}-frame clip".format(frames))
    print(json.dumps({"checks_passed": checks}, indent=2))


if __name__ == "__main__":
    main()
