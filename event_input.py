"""Decode event files into x, y, timestamp-us, polarity packets.

RAW is a container name, not a universal format. This reader supports
Prophesee/Metavision EVT2, EVT2.1 and EVT3 (and Prophesee DAT) through faery,
the decoder Ecology's loader tries first. It reads bounded packets rather than
loading the recording.

Why not expelliarmus: on an IDS uEye EVS (Sony IMX636) EVT3 recording it
returns the right events in the right order with the right x, y and polarity,
but its clock runs at half speed. Over the file's first 1,179,230 events it
spans 2.065 s where faery spans 1.032 s, the difference growing in exact
multiples of 4,096 us (the EVT3 coarse-time step), and the scene's lamp
flicker reads 49.99 Hz instead of 99.96 Hz (Australian mains is 50 Hz, so a
lamp flickers at 100). Every speed, duration and frequency downstream would be
halved. faery agrees event for event with an independent decoder (eventcv) on
three Prophesee EVK4 EVT3 files, and with Ecology's HDF5 reader on Muenster data.
"""
from pathlib import Path
import csv
import re

import numpy as np


ALIASES = {"x": ("x",), "y": ("y",), "t": ("t_us", "t", "timestamp"),
           "p": ("polarity", "p", "on")}


def columns(names):
    """Resolve supported column names without guessing their units."""
    lookup = {name.lower(): name for name in names}
    result = {}
    for key, alternatives in ALIASES.items():
        for alternative in alternatives:
            if alternative in lookup:
                result[key] = lookup[alternative]
                break
        else:
            raise ValueError("missing {} field; accepted names: {}".format(key, alternatives))
    return result


def canonical(arrays):
    """Validate before casting, so fractional coordinates are never truncated."""
    checked = []
    for name in ("x", "y", "t", "p"):
        # A contiguous, aligned copy first. Decoders hand over fields of packed
        # records (faery: 13-byte t/x/y/on rows, so most x and y values sit at
        # odd addresses). On numpy 2.0.2, comparing such a misaligned uint16
        # field with a Python integer outside its range (`x >= 2**63`, or even
        # `x >= 70000`) crashes the interpreter with an access violation;
        # the same comparison on a contiguous copy is fine.
        value = np.ascontiguousarray(arrays[name])
        if value.ndim != 1 or value.dtype.kind not in "biuf":
            raise ValueError("{} must be a one-dimensional numeric array".format(name))
        if value.dtype.kind == "b":
            value = value.astype(np.int64)      # boolean polarity: True = ON
        if not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
            raise ValueError("{} must contain finite integers".format(name))
        # Range check without comparing an array against an out-of-range
        # Python integer (see above): only uint64 and float can exceed int64.
        if value.size and value.dtype.kind == "u" and value.dtype.itemsize == 8 \
                and value.max() > np.iinfo(np.int64).max:
            raise ValueError("{} exceeds signed 64-bit range".format(name))
        if value.size and value.dtype.kind == "f" and np.abs(value).max() >= 2.0 ** 63:
            raise ValueError("{} exceeds signed 64-bit range".format(name))
        checked.append(value.astype(np.int64))
    if len({len(value) for value in checked}) != 1:
        raise ValueError("event arrays must have equal lengths")
    x, y, t, p = checked
    if np.any(x < 0) or np.any(y < 0) or np.any(t < 0):
        raise ValueError("coordinates and microsecond timestamps must be non-negative")
    if len(t) > 1 and np.any(t[1:] < t[:-1]):
        raise ValueError("event timestamps must be nondecreasing; input is not sorted")
    if not np.isin(p, [-1, 0, 1]).all():
        raise ValueError("polarity must use -1/+1 or 0/1")
    # Both common sensor conventions map to the tracker's signed convention.
    p = np.where(p > 0, 1, -1)
    return np.column_stack((x, y, t, p))


def header_metadata(path):
    """Read only the initial ASCII header, never decode payload bytes as text."""
    lines = []
    with path.open("rb") as handle:
        for _ in range(100):
            line = handle.readline(4096)
            if not line.startswith(b"%"):
                break
            lines.append(line.decode("ascii", errors="replace"))
            if line.strip() == b"% end":
                break
    text = "".join(lines)
    encoding = None
    for pattern, name in [(r"EVT21|evt\s*2\.1", "evt21"),
                          (r"EVT3|evt\s*3(?:\.0)?", "evt3"),
                          (r"EVT2|evt\s*2(?:\.0)?", "evt2")]:
        if re.search(pattern, text, re.I):
            encoding = name
            break
    dimensions = {}
    for key in ("width", "height"):
        found = re.search(r"\b" + key + r"[=\s]+(\d+)", text, re.I)
        if found:
            dimensions[key] = int(found.group(1))
    geometry = re.search(r"geometry\s+(\d+)x(\d+)", text, re.I)
    if geometry:
        dimensions.setdefault("width", int(geometry.group(1)))
        dimensions.setdefault("height", int(geometry.group(2)))
    return {"encoding": encoding, **dimensions}


def _csv_packets(path, size=8192):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = columns(reader.fieldnames or [])
        batch = []
        for line, row in enumerate(reader, 2):
            try:
                batch.append([int(row[names[key]]) for key in ("x", "y", "t", "p")])
            except (TypeError, ValueError) as exc:
                raise ValueError("CSV line {} needs integer values".format(line)) from exc
            if len(batch) == size:
                array = np.asarray(batch, dtype=np.int64)
                yield canonical(dict(zip(("x", "y", "t", "p"), array.T)))
                batch = []
        if batch:
            yield canonical(dict(zip(("x", "y", "t", "p"), np.asarray(batch).T)))


def _numpy_packets(path):
    with np.load(path, allow_pickle=False) as data:
        names = columns(data.files)
        arrays = {key: data[name] for key, name in names.items()}
        # Compressed NPZ arrays decompress in memory. RAW, CSV and HDF5 stream.
        packet = canonical(arrays)
        for start in range(0, len(packet), 8192):
            yield packet[start:start + 8192]


def _h5_group(handle):
    return handle["events"] if "events" in handle else handle


def _h5_packets(path):
    import h5py
    with h5py.File(path, "r") as handle:
        group = _h5_group(handle)
        names = columns(group.keys())
        lengths = {len(group[name]) for name in names.values()}
        if len(lengths) != 1:
            raise ValueError("HDF5 event arrays must have equal lengths")
        for start in range(0, lengths.pop(), 8192):
            yield canonical({key: group[name][start:start + 8192] for key, name in names.items()})


FAERY_VERSION = {"evt2": "evt2", "evt21": "evt2.1", "evt3": "evt3"}


def _raw_packets(path, encoding, width, height):
    """faery packets carry t (microseconds), x, y and a boolean `on`.

    faery reads the encoding and sensor size from the file header. The
    fallbacks below only matter for a headerless file; open_recording has
    already made sure width and height are known, because faery would
    otherwise assume 1280 x 720 without saying so."""
    import faery
    stream = faery.events_stream_from_file(
        str(path), dimensions_fallback=(int(width), int(height)),
        version_fallback=FAERY_VERSION.get(encoding))
    for packet in stream:
        if len(packet):
            yield canonical({"x": packet["x"], "y": packet["y"],
                             "t": packet["t"], "p": packet["on"]})


def open_recording(path, width=None, height=None, encoding=None):
    """Return sensor metadata and a lazy packet iterator. Never infer sensor
    dimensions from active pixels: a quiet border is still part of the sensor.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in (".raw", ".dat", ".csv", ".npz", ".h5", ".hdf5"):
        raise ValueError("supported event inputs: RAW, DAT, CSV, NPZ, H5/HDF5; ordinary video is not supported")
    if not path.is_file():
        raise ValueError("input file does not exist: {}".format(path))
    metadata = {"format": ext[1:]}
    if ext in (".raw", ".dat"):
        try:
            import faery  # noqa: F401
        except ImportError as exc:
            raise ValueError("RAW/DAT needs: python -m pip install -r requirements-raw.txt") from exc
        metadata.update(header_metadata(path))
        encoding = encoding or ("dat" if ext == ".dat" else metadata.get("encoding"))
        if not encoding:
            raise ValueError("RAW header does not identify its encoding; pass --encoding evt2, evt21 or evt3")
        metadata["encoding"] = encoding
        for key, override in (("width", width), ("height", height)):
            if override is not None:
                metadata[key] = override
            if key not in metadata or metadata[key] < 1:
                raise ValueError("sensor {} missing; pass --{} in pixels".format(key, key))
        packets = _raw_packets(path, encoding, metadata["width"], metadata["height"])
    elif ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in ("width", "height"):
                if key in data:
                    metadata[key] = int(data[key])
        packets = _numpy_packets(path)
    elif ext in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError as exc:
            raise ValueError("HDF5 needs: python -m pip install h5py") from exc
        with h5py.File(path, "r") as handle:
            group = _h5_group(handle)
            if not hasattr(group, "keys"):
                raise ValueError("HDF5 needs separate x,y,t,p arrays at root or in /events")
            columns(group.keys())
            for key in ("width", "height"):
                if key in group.attrs or key in handle.attrs:
                    metadata[key] = int(group.attrs.get(key, handle.attrs.get(key)))
        packets = _h5_packets(path)
    else:
        packets = _csv_packets(path)
    for key, override in (("width", width), ("height", height)):
        if override is not None:
            metadata[key] = override
        if key not in metadata or metadata[key] < 1:
            raise ValueError("sensor {} missing; pass --{} in pixels".format(key, key))
    return metadata, packets


def select_packets(packets, metadata, start_s, duration_s, roi, status):
    """Crop by time and optionally region, preserving input order, as arrays.

    --start is relative to the first decoded event. Output timestamps keep the
    recording's original microsecond clock. Yields (n, 4) int64 arrays of
    x, y, t, p. Whole packets, not single events: a 4 s window of a busy
    1280x720 recording is about 5 million events, and a Python tuple per event
    costs minutes before any processing starts.
    """
    origin = previous = None
    for packet in packets:
        if not len(packet):
            continue
        if previous is not None and packet[0, 2] < previous:
            raise ValueError("event timestamps decreased across packets")
        previous = int(packet[-1, 2])
        if np.any(packet[:, 0] >= metadata["width"]) or np.any(packet[:, 1] >= metadata["height"]):
            raise ValueError("event coordinates exceed sensor dimensions")
        if origin is None:
            origin = int(packet[0, 2])
            lower = origin + round(start_s * 1e6)
            upper = lower + round(duration_s * 1e6)
            status.update(recording_origin_us=origin, requested_start_us=lower, requested_end_us=upper)
        if packet[0, 2] >= upper:
            break
        mask = (packet[:, 2] >= lower) & (packet[:, 2] < upper)
        if roi:
            x0, y0, x1, y1 = roi
            mask &= (packet[:, 0] >= x0) & (packet[:, 0] < x1) & (packet[:, 1] >= y0) & (packet[:, 1] < y1)
        if mask.any():
            yield packet[mask]
        if packet[-1, 2] >= upper:
            break
