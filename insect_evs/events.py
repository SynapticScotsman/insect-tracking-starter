"""Event stream container and I/O.

Convention used everywhere in this package:
    x, y   pixel coordinates, origin top-left, int32
    t      timestamp in microseconds (us), int64, non-decreasing
    p      polarity, int8, +1 for ON (brightness increase), -1 for OFF

Timestamps are kept in microseconds because that is what Prophesee and
iniVation sensors report. Converting to seconds early loses precision: a
float32 cannot hold microsecond resolution past about 16 s of recording.

Loaders here return an EventStream in this convention whatever the file
stored: polarity is mapped to +1/-1 and the events are sorted by time.
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
    """Events from one sensor as four parallel arrays, plus the sensor size in
    pixels. The arrays are cast to the dtypes in the module docstring on
    construction. Methods that index return a new stream; none modify this one.
    """
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
        """Last timestamp minus first, in seconds. Assumes time order."""
        if len(self) == 0:
            return 0.0
        return float(self.t[-1] - self.t[0]) / 1e6

    @property
    def rate_hz(self) -> float:
        """Mean event rate over the whole stream, events per second."""
        return len(self) / self.duration_s if self.duration_s > 0 else 0.0

    def sorted_by_time(self) -> "EventStream":
        """This stream in time order. Returns the same object when it is
        already sorted. The sort is stable, so equal timestamps keep their
        order."""
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
        """Events with t0_us <= t < t1_us. The stream must be time-sorted; on
        an unsorted stream the result is wrong and no error is raised."""
        i0, i1 = np.searchsorted(self.t, [int(t0_us), int(t1_us)])
        return self.select(slice(i0, i1))

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> "EventStream":
        """Events with x0 <= x < x1 and y0 <= y < y1. Coordinates and sensor
        size stay those of the full sensor; they are not shifted to the crop."""
        m = (self.x >= x0) & (self.x < x1) & (self.y >= y0) & (self.y < y1)
        return self.select(m)

    def count_image(self, signed: bool = False) -> np.ndarray:
        """Events per pixel as a (height, width) float array.

        With signed=True each event adds its polarity, so the image is ON
        minus OFF. Otherwise every event adds 1.
        """
        flat = self.y.astype(np.int64) * self.width + self.x.astype(np.int64)
        w = self.p.astype(np.float64) if signed else None
        img = np.bincount(flat, weights=w, minlength=self.width * self.height)
        return img.reshape(self.height, self.width)

    def save(self, path: str) -> None:
        """Write a compressed .npz that load_events reads back exactly.
        Creates the parent folder if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path, x=self.x, y=self.y, t=self.t, p=self.p,
            width=self.width, height=self.height,
        )

    @staticmethod
    def concatenate(streams) -> "EventStream":
        """Join several streams from one sensor into a single time-sorted stream."""
        # Empty streams are dropped first, so an empty stream at the front
        # cannot set the sensor size.
        streams = [s for s in streams if len(s) > 0]
        if not streams:
            raise ValueError("nothing to concatenate")
        # The sensor size comes from the first stream. The others are not
        # checked against it, so the caller must pass streams from one sensor.
        w, h = streams[0].width, streams[0].height
        out = EventStream(
            np.concatenate([s.x for s in streams]),
            np.concatenate([s.y for s in streams]),
            np.concatenate([s.t for s in streams]),
            np.concatenate([s.p for s in streams]),
            w, h,
        )
        # The inputs may overlap in time or arrive in any order, so the joined
        # stream is re-sorted. sorted_by_time uses a stable sort, so events with
        # equal timestamps keep the order of the input list.
        return out.sorted_by_time()


def load_events(
    path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    encoding: Optional[str] = None,
) -> EventStream:
    """Load a whole recording, choosing the reader by file extension.

    Supported: .npz (this package's own format, see EventStream.save),
    .h5/.hdf5 (h5py), and through the faery library .raw, .dat, .es, .aedat4,
    .aedat, .csv and .npy. When faery is not installed or fails on a file,
    .raw/.dat fall back to expelliarmus, .aedat4 to dv_processing, and
    .csv/.txt to pandas.

    `width` and `height` set the sensor size where the file does not carry it.
    `encoding` overrides the Prophesee format read from the file header
    (evt2, evt21, evt3, dat); it applies only to the expelliarmus fallback.
    For a long recording, load_events_window reads only the part you need.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path)
        return EventStream(d["x"], d["y"], d["t"], d["p"], int(d["width"]), int(d["height"]))

    # HDF5 goes to h5py, because faery does not read Prophesee HDF5 files.
    if ext in (".h5", ".hdf5"):
        return _load_hdf5(path, width, height)

    # Every format faery reads goes to faery first, so one decoder covers
    # .raw, .dat, .es, .aedat4, .csv and .npy. The format-specific readers
    # below are the fallback when faery is absent or rejects a file.
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
    """Sensor size in pixels. A missing width or height is taken as the
    largest coordinate plus one, which is too small when the edge pixels
    never fired."""
    if width is None:
        width = int(x.max()) + 1 if len(x) else 1
    if height is None:
        height = int(y.max()) + 1 if len(y) else 1
    return int(width), int(height)


def _load_hdf5(path, width, height) -> EventStream:
    """Read a whole .h5/.hdf5 event file with h5py. `width` and `height`, when
    given, override any size stored in the file."""
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
            # A file with no polarity field loads as all ON events.
            p = grp["p"][:] if "p" in grp else np.ones(len(x), np.int8)
        # Sensor size: the caller's value first, then the file's root
        # attributes. If both are missing, _infer_size below uses the largest
        # coordinate plus one, which undercounts when the edge pixels are silent.
        width = width or f.attrs.get("width")
        height = height or f.attrs.get("height")
    # Map any stored polarity (0/1, bool or -1/+1) to this package's +1/-1.
    # Leaving 0 for OFF would make OFF events count as zero in signed sums.
    p = np.where(np.asarray(p) > 0, 1, -1).astype(np.int8)
    width, height = _infer_size(x, y, width, height)
    # Timestamps are kept as stored. They must already be in microseconds.
    return EventStream(x, y, t, p, width, height).sorted_by_time()


#: Extensions sent to faery. .h5 is left out because faery rejects Prophesee
#: HDF5 files with "unsupported file extension"; h5py reads those instead.
FAERY_EXTENSIONS = (".raw", ".dat", ".es", ".aedat4", ".aedat", ".csv", ".npy")

#: A first timestamp above this (about 31 years in us) is taken to count from
#: the Unix epoch rather than from the start of the recording, as AEDAT4 files
#: do. Such values fit in int64, but any float32 step, such as an FFT, cannot
#: resolve them to better than minutes and gives meaningless output. The faery
#: loader subtracts the first timestamp when it sees one.
EPOCH_THRESHOLD_US = 10 ** 15


def _events_from_ics_dtype(
    t, x, y, on, width=None, height=None
) -> Tuple["EventStream", int]:
    """Convert arrays in the layout used by faery and neuromorphic-drivers to
    an EventStream. Returns (stream, epoch_offset_us).

    Those libraries store events as `[("t","<u8"),("x","<u2"),("y","<u2"),
    ("on","?")]`. Two conversions here matter:

    * Polarity arrives as a bool. Casting it straight to int8 would give 0/1,
      so every OFF event would count as zero instead of -1 in signed sums.
      The wingbeat measurement uses the ON minus OFF rate, so OFF must be -1.
    * Unsigned 64-bit time is converted to int64 by value (`astype`), not by
      reinterpreting the bytes (`view`).

    If the first timestamp is above EPOCH_THRESHOLD_US it is subtracted from
    every timestamp, and returned as epoch_offset_us; otherwise the offset is 0.
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
    """Read any format the faery library supports into an EventStream.

    Covers Prophesee `.raw`/`.dat`, `.es`, `.aedat4`, `.csv` and `.npy` with one
    decoder. faery's pixel origin, polarity and timestamp conventions match
    this package's. On a test recording stored both as Prophesee HDF5 and as
    .raw, this reader and the HDF5 reader agreed on every event's time, x, y
    and polarity.

    With `t0_us` and `t1_us`, only events with t0_us <= t < t1_us are kept, and
    reading stops at the first packet past t1_us. Timestamps that count from
    the Unix epoch are shifted to start near zero, with a warning.
    """
    import faery

    stream = faery.events_stream_from_file(path)
    # Sensor size, when the format records it. Otherwise it is inferred from
    # the coordinates.
    dims = None
    try:
        dims = stream.dimensions()
    except Exception:
        pass

    # faery yields packets in time order. Skip packets that end before t0_us,
    # stop at the first that starts after t1_us, and trim the packets that
    # straddle either bound.
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
    """Record `duration_s` seconds from a live event camera through the
    neuromorphic-drivers library. Not used by the rest of this package.

    The library supports the Prophesee EVK4 and EVK3 HD, the SilkyEvCam HD,
    and the iniVation DVXplorer and DAVIS 346, without the vendors' own SDKs.
    `serial` picks one camera when several are connected. Capture stops early,
    with a warning, after `max_events` events.

    Dropped data is reported with a warning. The driver's buffer discards
    packets when the computer cannot keep up, and a gap in the middle of a
    wingbeat sequence would otherwise look like the insect pausing.
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
    The format must be read, not guessed: decoding an EVT2 file as EVT3 gives
    wrong events rather than an error, because both are valid bit streams that
    assign different meanings to the same bits. Returns "evt2", "evt21", "evt3"
    or "dat". With no recognisable header, a .dat file is "dat" and anything
    else is assumed to be "evt3".
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
    """EventStream from an expelliarmus record array, with polarity mapped to
    +1/-1. Event order is kept as read."""
    p = np.where(arr["p"] > 0, 1, -1).astype(np.int8)
    width, height = _infer_size(arr["x"], arr["y"], width, height)
    return EventStream(arr["x"], arr["y"], arr["t"], p, width, height)


def _load_prophesee(path, encoding: Optional[str] = None) -> EventStream:
    """Read a whole Prophesee .raw/.dat file with expelliarmus. load_events
    reaches this only when faery is absent or fails on the file. The sensor
    size is inferred from the coordinates."""
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
    """Load only the events with t0_us <= t < t1_us from a recording.

    An HD sensor can produce a million events per second, so a long recording
    may not fit in memory. Formats faery reads are decoded packet by packet
    and reading stops once past `t1_us`. If faery is unavailable, Prophesee
    .raw/.dat files are decoded in chunks with expelliarmus, also stopping
    past `t1_us`. Any other format is read whole and then sliced, which gives
    the same result but uses as much memory as the full recording.
    """
    t0_us, t1_us = int(t0_us), int(t1_us)
    ext = os.path.splitext(path)[1].lower()

    # faery first. Any failure falls through to the readers below, without a
    # warning.
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

    # Decode in chunks of the requested span, at most 1 s each. Chunks that
    # end before t0_us are discarded; reading stops at the first chunk that
    # reaches t1_us, and the joined chunks are trimmed to the exact bounds.
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
    """Read a whole iniVation .aedat4 file with dv_processing. load_events
    reaches this only when faery is absent or fails on the file."""
    try:
        import dv_processing as dv
    except ImportError as exc:  # pragma: no cover - depends on optional SDK
        raise ImportError(
            "reading .aedat4 needs dv_processing: pip install dv-processing"
        ) from exc
    reader = dv.io.MonoCameraRecording(path)
    # Sensor size from the recording itself, as (width, height).
    res = reader.getEventResolution()
    # Read batch by batch until the reader reports the end of the file.
    # A None batch is skipped and the loop asks again.
    xs, ys, ts, ps = [], [], [], []
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None:
            continue
        a = batch.numpy()
        xs.append(a["x"]); ys.append(a["y"]); ts.append(a["timestamp"]); ps.append(a["polarity"])
    if not xs:
        return EventStream(np.array([]), np.array([]), np.array([]), np.array([]), res[0], res[1])
    # Map the stored polarity to this package's +1/-1.
    p = np.where(np.concatenate(ps) > 0, 1, -1).astype(np.int8)
    # Timestamps are kept as stored. Unlike the faery path, no Unix-epoch
    # offset is subtracted here; see EPOCH_THRESHOLD_US.
    return EventStream(
        np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), p, res[0], res[1]
    ).sorted_by_time()


def _load_csv(path, width, height) -> EventStream:
    """Read a .csv or .txt event table with a header row. Needs x, y and t
    columns; p is optional. The sensor size is inferred when not given."""
    import pandas as pd

    df = pd.read_csv(path)
    # Column names are matched case-insensitively, and "timestamp" and
    # "polarity" are accepted as names for t and p.
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
    # t must already be in microseconds. EventStream casts it to int64, so a
    # column in seconds would lose everything below one second.
    t = df[cols["t"]].to_numpy()
    # No polarity column: every event loads as ON.
    p = df[cols["p"]].to_numpy() if "p" in cols else np.ones(len(x))
    # Map 0/1 or -1/+1 polarity to this package's +1/-1.
    p = np.where(p > 0, 1, -1).astype(np.int8)
    width, height = _infer_size(x, y, width, height)
    return EventStream(x, y, t, p, width, height).sorted_by_time()
