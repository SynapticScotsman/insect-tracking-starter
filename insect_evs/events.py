"""Event stream container and I/O.

Convention used everywhere in this package:
    x, y   pixel coordinates, origin top-left, int32
    t      timestamp in MICROSECONDS, int64, monotonically non-decreasing
    p      polarity, int8, +1 for ON (brightness increase), -1 for OFF

Timestamps are microseconds because that is what every commercial sensor
(Prophesee EVT2/EVT3, iniVation AEDAT4) reports natively. Converting to
seconds early loses precision in float32 after ~10 s of recording.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class EventStream:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.int32)
        self.y = np.ascontiguousarray(self.y, dtype=np.int32)
        self.t = np.ascontiguousarray(self.t, dtype=np.int64)
        self.p = np.ascontiguousarray(self.p, dtype=np.int8)
        n = len(self.t)
        if not (len(self.x) == len(self.y) == len(self.p) == n):
            raise ValueError("event arrays have mismatched lengths")

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def __repr__(self) -> str:
        return "EventStream(n={:,}, {}x{}, {:.3f} s)".format(
            len(self), self.width, self.height, self.duration_s
        )

    @property
    def duration_s(self) -> float:
        if len(self) == 0:
            return 0.0
        return float(self.t[-1] - self.t[0]) / 1e6

    @property
    def rate_hz(self) -> float:
        return len(self) / self.duration_s if self.duration_s > 0 else 0.0

    def sorted_by_time(self) -> "EventStream":
        if len(self) == 0 or np.all(np.diff(self.t) >= 0):
            return self
        order = np.argsort(self.t, kind="stable")
        return self.select(order)

    def select(self, idx: np.ndarray) -> "EventStream":
        """Index with a boolean mask or integer array; returns a new stream."""
        return EventStream(
            self.x[idx], self.y[idx], self.t[idx], self.p[idx], self.width, self.height
        )

    def time_slice(self, t0_us: int, t1_us: int) -> "EventStream":
        """Half-open slice [t0, t1). Assumes time-sorted (uses searchsorted)."""
        i0, i1 = np.searchsorted(self.t, [int(t0_us), int(t1_us)])
        return self.select(slice(i0, i1))

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> "EventStream":
        """Half-open spatial crop. Coordinates are NOT re-based to the crop."""
        m = (self.x >= x0) & (self.x < x1) & (self.y >= y0) & (self.y < y1)
        return self.select(m)

    def count_image(self, signed: bool = False) -> np.ndarray:
        """Accumulate events into an image. Fast path via bincount."""
        flat = self.y.astype(np.int64) * self.width + self.x.astype(np.int64)
        w = self.p.astype(np.float64) if signed else None
        img = np.bincount(flat, weights=w, minlength=self.width * self.height)
        return img.reshape(self.height, self.width)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path, x=self.x, y=self.y, t=self.t, p=self.p,
            width=self.width, height=self.height,
        )

    @staticmethod
    def concatenate(streams) -> "EventStream":
        streams = [s for s in streams if len(s) > 0]
        if not streams:
            raise ValueError("nothing to concatenate")
        w, h = streams[0].width, streams[0].height
        out = EventStream(
            np.concatenate([s.x for s in streams]),
            np.concatenate([s.y for s in streams]),
            np.concatenate([s.t for s in streams]),
            np.concatenate([s.p for s in streams]),
            w, h,
        )
        return out.sorted_by_time()


def load_events(
    path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    encoding: Optional[str] = None,
) -> EventStream:
    """Load events, dispatching on file extension.

    Supported: .npz (this package's own format), .h5/.hdf5, .raw/.dat
    (Prophesee, via expelliarmus), .aedat4 (via dv_processing), .csv.

    `encoding` overrides the sniffed Prophesee format (evt2, evt21, evt3, dat).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path)
        return EventStream(d["x"], d["y"], d["t"], d["p"], int(d["width"]), int(d["height"]))

    # HDF5 stays with h5py: faery reports "unsupported file extension" on
    # Prophesee HDF5, so routing it there would break the main Münster path.
    if ext in (".h5", ".hdf5"):
        return _load_hdf5(path, width, height)

    # Everything faery handles goes to faery first, so one decoder covers .raw,
    # .dat, .es, .aedat4, .csv and .npy. The older readers remain as a fallback
    # when faery is absent or rejects a file.
    if ext in FAERY_EXTENSIONS:
        try:
            return load_events_faery(path)
        except ImportError:
            pass
        except Exception as exc:
            warnings.warn(
                "faery could not read {} ({}); falling back to the built-in reader"
                .format(os.path.basename(path), exc),
                RuntimeWarning, stacklevel=2,
            )

    if ext in (".raw", ".dat"):
        return _load_prophesee(path, encoding=encoding)
    if ext == ".aedat4":
        return _load_aedat4(path)
    if ext in (".csv", ".txt"):
        return _load_csv(path, width, height)
    raise ValueError("unrecognised event file extension: {}".format(ext))


def _infer_size(x: np.ndarray, y: np.ndarray, width, height) -> Tuple[int, int]:
    if width is None:
        width = int(x.max()) + 1 if len(x) else 1
    if height is None:
        height = int(y.max()) + 1 if len(y) else 1
    return int(width), int(height)


def _load_hdf5(path, width, height) -> EventStream:
    import h5py

    try:
        # Prophesee HDF5 stores events with the ECF codec, which h5py cannot
        # decode on its own. Importing hdf5plugin registers it globally; without
        # it the read fails with an opaque filter error.
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass

    with h5py.File(path, "r") as f:
        # Two common layouts: /events/{x,y,t,p} (this package, Prophesee HDF5)
        # and a single compound dataset /events.
        grp = f["events"] if "events" in f else f
        if isinstance(grp, h5py.Dataset):
            arr = grp[:]
            x, y, t, p = arr["x"], arr["y"], arr["t"], arr["p"]
        else:
            x, y, t = grp["x"][:], grp["y"][:], grp["t"][:]
            p = grp["p"][:] if "p" in grp else np.ones(len(x), np.int8)
        width = width or f.attrs.get("width")
        height = height or f.attrs.get("height")
    p = np.where(np.asarray(p) > 0, 1, -1).astype(np.int8)
    width, height = _infer_size(x, y, width, height)
    return EventStream(x, y, t, p, width, height).sorted_by_time()


#: Extensions faery decodes. Deliberately excludes .h5: faery raises
#: "unsupported file extension" on Prophesee HDF5, so h5py remains the only path
#: for it. Verified against a Münster file, not taken from the documentation.
FAERY_EXTENSIONS = (".raw", ".dat", ".es", ".aedat4", ".aedat", ".csv", ".npy")

#: Timestamps beyond this are microseconds since the Unix epoch rather than since
#: the start of the recording. AEDAT4 does this. Left uncorrected it survives
#: int64 but destroys any float32 path, because 24 bits of mantissa cannot hold
#: 1.7e18 to sub-second precision, so an FFT would silently return nonsense.
EPOCH_THRESHOLD_US = 10 ** 15


def _events_from_ics_dtype(
    t, x, y, on, width=None, height=None
) -> Tuple["EventStream", int]:
    """Convert the ICNS-stack layout to an EventStream. Returns (stream, epoch_offset).

    The whole ICNS stack (faery, aedat, event_stream, neuromorphic-drivers) shares
    `[("t","<u8"),("x","<u2"),("y","<u2"),("on","?")]`. Two conversions here are
    load-bearing:

    * Polarity is a numpy bool. `on.astype(np.int8)` would give 0/1, which makes
      every OFF event *neutral* rather than negative and silently halves the
      contrast of every signed accumulation -- i.e. it would quietly gut the
      ON-minus-OFF signal the whole wingbeat measurement depends on.
    * `<u8` becomes int64 by `astype`, never `view`.
    """
    t = np.asarray(t).astype(np.int64)
    offset = 0
    if t.size and t[0] > EPOCH_THRESHOLD_US:
        offset = int(t[0])
        t = t - offset
    p = np.where(np.asarray(on), 1, -1).astype(np.int8)
    x = np.asarray(x).astype(np.int32)
    y = np.asarray(y).astype(np.int32)
    width, height = _infer_size(x, y, width, height)
    return EventStream(x, y, t, p, width, height), offset


def load_events_faery(
    path: str,
    t0_us: Optional[int] = None,
    t1_us: Optional[int] = None,
) -> EventStream:
    """Read any faery-supported format into an EventStream.

    Covers Prophesee `.raw`/`.dat`, `.es`, `.aedat4`, `.csv` and `.npy` through one
    decoder. Cross-checked against this package's own HDF5 reader on a recording
    Münster ships in both encodings: 10,073 events over a two-second window agreed
    byte for byte on timestamp, x, y and polarity, and the y-flip hypothesis
    matched zero events. So faery's origin, polarity and timestamp conventions are
    confirmed identical to ours, rather than assumed.
    """
    import faery

    stream = faery.events_stream_from_file(path)
    dims = None
    try:
        dims = stream.dimensions()
    except Exception:
        pass

    xs, ys, ts, ps = [], [], [], []
    for packet in stream:
        if len(packet) == 0:
            continue
        if t1_us is not None and packet["t"][0] > t1_us:
            break
        if t0_us is not None and packet["t"][-1] < t0_us:
            continue
        m = np.ones(len(packet), bool)
        if t0_us is not None:
            m &= packet["t"] >= t0_us
        if t1_us is not None:
            m &= packet["t"] < t1_us
        if not m.any():
            continue
        xs.append(packet["x"][m]); ys.append(packet["y"][m])
        ts.append(packet["t"][m]); ps.append(packet["on"][m])

    if not ts:
        w, h = dims if dims else (1, 1)
        z = np.zeros(0)
        return EventStream(z, z, z, z, w, h)

    ev, offset = _events_from_ics_dtype(
        np.concatenate(ts), np.concatenate(xs), np.concatenate(ys),
        np.concatenate(ps),
        width=dims[0] if dims else None, height=dims[1] if dims else None,
    )
    if offset:
        warnings.warn(
            "timestamps looked like microseconds since the Unix epoch; subtracted "
            "an offset of {} us so the recording starts near zero".format(offset),
            RuntimeWarning, stacklevel=2,
        )
    return ev


def capture_live(
    duration_s: float = 2.0,
    configuration=None,
    serial: Optional[str] = None,
    max_events: int = 20_000_000,
) -> EventStream:
    """Capture from a live event camera via neuromorphic-drivers.

    Supports EVK4, EVK3 HD, SilkyEvCam HD, DVXplorer and DAVIS 346 with no
    Metavision or libcaer dependency. This is the one thing the file readers
    cannot do, and it is why the library is worth having despite not reading
    files at all.

    Dropped packets are reported rather than ignored. The driver exposes overflow
    indices because its ring buffer discards data under load, and a silent gap in
    the middle of a wingbeat sequence would look exactly like an insect pausing.
    """
    import neuromorphic_drivers as nd

    xs, ys, ts, ps = [], [], [], []
    overflows = 0
    t_end = None
    dims = None

    with nd.open(configuration=configuration, serial=serial) as device:
        try:
            props = device.properties()
            dims = (props.width, props.height)
        except Exception:
            pass
        for status, packet in device:
            events = getattr(packet, "polarity_events", None)
            if events is None or len(events) == 0:
                continue
            if getattr(status, "overflow_indices", None):
                overflows += len(status.overflow_indices)
            xs.append(events["x"]); ys.append(events["y"])
            ts.append(events["t"]); ps.append(events["on"])
            if t_end is None:
                t_end = int(events["t"][0]) + int(duration_s * 1e6)
            if int(events["t"][-1]) >= t_end:
                break
            if sum(len(a) for a in ts) >= max_events:
                warnings.warn("hit max_events; capture truncated", RuntimeWarning)
                break

    if overflows:
        warnings.warn(
            "{} buffer overflow(s) during capture: events were dropped, so gaps in "
            "the stream are the driver's, not the animal's".format(overflows),
            RuntimeWarning, stacklevel=2,
        )
    if not ts:
        z = np.zeros(0)
        w, h = dims if dims else (1, 1)
        return EventStream(z, z, z, z, w, h)

    ev, _ = _events_from_ics_dtype(
        np.concatenate(ts), np.concatenate(xs), np.concatenate(ys),
        np.concatenate(ps),
        width=dims[0] if dims else None, height=dims[1] if dims else None,
    )
    return ev


def sniff_prophesee_encoding(path: str) -> str:
    """Read the encoding out of a Prophesee file header.

    Prophesee `.raw` files begin with '%'-prefixed ASCII header lines carrying
    the format, e.g. `% evt 2.0` or `% format EVT3;height=720;width=1280`.
    Guessing instead of reading is not safe: the Münster ictrap recordings are
    EVT2 and decoding them as EVT3 yields garbage rather than an error, because
    both are valid bit-streams that happen to disagree about what the bits mean.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        head = f.read(4096)
    text = head.decode("ascii", errors="ignore")

    if re.search(r"evt[\s_]*2\.1|EVT21", text, re.IGNORECASE):
        return "evt21"
    if re.search(r"evt[\s_]*3(\.0)?|EVT3", text, re.IGNORECASE):
        return "evt3"
    if re.search(r"evt[\s_]*2(\.0)?|EVT2", text, re.IGNORECASE):
        return "evt2"
    # No usable header. .dat carries its own two-byte type/size preamble.
    return "dat" if ext == ".dat" else "evt3"


def _prophesee_stream(arr, width=None, height=None) -> EventStream:
    p = np.where(arr["p"] > 0, 1, -1).astype(np.int8)
    width, height = _infer_size(arr["x"], arr["y"], width, height)
    return EventStream(arr["x"], arr["y"], arr["t"], p, width, height)


def _load_prophesee(path, encoding: Optional[str] = None) -> EventStream:
    from expelliarmus import Wizard

    wiz = Wizard(encoding=encoding or sniff_prophesee_encoding(path), fpath=path)
    return _prophesee_stream(wiz.read())


def load_events_window(
    path: str,
    t0_us: int,
    t1_us: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
    encoding: Optional[str] = None,
) -> EventStream:
    """Load only [t0_us, t1_us) from a recording.

    HD recordings at ~1 Mev/s do not fit comfortably in memory, and every
    analysis here works on windows anyway. For Prophesee files this decodes
    incrementally and stops once past `t1_us`; for other formats it falls back to
    a full read and a slice, which is correct but not cheap.
    """
    t0_us, t1_us = int(t0_us), int(t1_us)
    ext = os.path.splitext(path)[1].lower()

    if ext in FAERY_EXTENSIONS:
        try:
            return load_events_faery(path, t0_us, t1_us)
        except ImportError:
            pass
        except Exception:
            pass

    if ext not in (".raw", ".dat"):
        return load_events(path, width=width, height=height).time_slice(t0_us, t1_us)

    from expelliarmus import Wizard

    wiz = Wizard(encoding=encoding or sniff_prophesee_encoding(path), fpath=path)
    step_us = max(int(min(t1_us - t0_us, 1_000_000)), 1)
    wiz.set_time_window(step_us)

    chunks = []
    elapsed = 0
    for chunk in wiz.read_time_window():
        if chunk is None or len(chunk) == 0:
            elapsed += step_us
            if elapsed >= t1_us:
                break
            continue
        if chunk["t"][-1] < t0_us:
            elapsed = int(chunk["t"][-1])
            continue
        chunks.append(chunk)
        if chunk["t"][-1] >= t1_us:
            break

    if not chunks:
        return EventStream(
            np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
            width or 1, height or 1,
        )
    arr = np.concatenate(chunks)
    return _prophesee_stream(arr, width, height).time_slice(t0_us, t1_us)


def _load_aedat4(path) -> EventStream:
    try:
        import dv_processing as dv
    except ImportError as exc:  # pragma: no cover - depends on optional SDK
        raise ImportError(
            "reading .aedat4 needs dv_processing: pip install dv-processing"
        ) from exc
    reader = dv.io.MonoCameraRecording(path)
    res = reader.getEventResolution()
    xs, ys, ts, ps = [], [], [], []
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None:
            continue
        a = batch.numpy()
        xs.append(a["x"]); ys.append(a["y"]); ts.append(a["timestamp"]); ps.append(a["polarity"])
    if not xs:
        return EventStream(np.array([]), np.array([]), np.array([]), np.array([]), res[0], res[1])
    p = np.where(np.concatenate(ps) > 0, 1, -1).astype(np.int8)
    return EventStream(
        np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), p, res[0], res[1]
    ).sorted_by_time()


def _load_csv(path, width, height) -> EventStream:
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "t" not in cols and "timestamp" in cols:
        cols["t"] = cols["timestamp"]
    if "p" not in cols and "polarity" in cols:
        cols["p"] = cols["polarity"]
    missing = {"x", "y", "t"} - set(cols)
    if missing:
        raise ValueError("CSV is missing column(s): {}".format(sorted(missing)))
    x = df[cols["x"]].to_numpy()
    y = df[cols["y"]].to_numpy()
    t = df[cols["t"]].to_numpy()
    p = df[cols["p"]].to_numpy() if "p" in cols else np.ones(len(x))
    p = np.where(p > 0, 1, -1).astype(np.int8)
    width, height = _infer_size(x, y, width, height)
    return EventStream(x, y, t, p, width, height).sorted_by_time()
