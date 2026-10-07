"""Read a raw event recording as packets of x, y, timestamp (us), polarity.

Two kinds of raw file are accepted, told apart by their first bytes:

- Prophesee .raw, written by Prophesee/Metavision software and by cameras
  built on Prophesee sensors, such as the IDS uEye EVS. The events use one of
  three encodings, EVT2, EVT2.1 or EVT3, named in a text header.
- Kairos .raw.kai, written by the Kairos recorder from a Prophesee EVK4. A
  16-byte binary header ("KAIROS-RAW", version, type, width, height) is
  followed by plain EVT3 data.

The faery library decodes the events. The file is read in bounded packets,
so a long recording never has to fit in memory.

Use faery, not the expelliarmus decoder. On EVT3 files from an IDS uEye EVS
camera (Sony IMX636 sensor), expelliarmus returns the right events with
timestamps running at half speed: a lamp flickering at 100 Hz reads 50 Hz,
and every speed, duration and frequency computed later is halved.
"""
from pathlib import Path
import re
import shutil
import struct
import tempfile

import numpy as np

KAIROS_SIGNATURE = b"KAIROS-RAW"
KAIROS_HEADER_BYTES = 16      # signature (10), version (1), type (1), width (2), height (2)


def canonical(arrays):
    """Check one packet's x, y, t, p arrays and stack them as an (n, 4) int64
    array with polarity -1 (OFF) / +1 (ON). Values are validated before any
    cast, so a fractional coordinate is refused rather than truncated."""
    checked = []
    for name in ("x", "y", "t", "p"):
        # A contiguous, aligned copy first. faery hands over fields of packed
        # 13-byte records, so most x and y values sit at odd memory addresses.
        # On numpy 2.0.2, comparing such a field with a Python integer outside
        # its range crashes the interpreter; a contiguous copy is safe.
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
    """Encoding and sensor size from a .raw file's header.

    The header is a run of ASCII lines starting with '%' before the binary
    events. Only those lines are read, so event bytes are never decoded as
    text. Returns {"encoding": "evt2" | "evt21" | "evt3" | None} plus
    "width" and "height" when the header states them."""
    lines = []
    with path.open("rb") as handle:
        # At most 100 header lines of at most 4,096 bytes, stopping at the
        # first line that is not a header line or at "% end".
        for _ in range(100):
            line = handle.readline(4096)
            if not line.startswith(b"%"):
                break
            lines.append(line.decode("ascii", errors="replace"))
            if line.strip() == b"% end":
                break
    text = "".join(lines)
    encoding = None
    # EVT2.1 is tested before EVT2, because "EVT2" also matches "EVT21".
    for pattern, name in [(r"EVT21|evt\s*2\.1", "evt21"),
                          (r"EVT3|evt\s*3(?:\.0)?", "evt3"),
                          (r"EVT2|evt\s*2(?:\.0)?", "evt2")]:
        if re.search(pattern, text, re.I):
            encoding = name
            break
    # Size appears either as "width 1280" / "height=720" or as
    # "geometry 1280x720", depending on the recording software.
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


FAERY_VERSION = {"evt2": "evt2", "evt21": "evt2.1", "evt3": "evt3"}


def _raw_packets(path, encoding, width, height):
    """faery packets carry t (microseconds), x, y and a boolean `on`.

    faery reads the encoding and sensor size from the file header. The
    fallbacks below only matter for a file without a header; open_recording
    has already made sure width and height are known, because faery would
    otherwise assume 1280 x 720 without saying so."""
    import faery
    stream = faery.events_stream_from_file(
        str(path), dimensions_fallback=(int(width), int(height)),
        version_fallback=FAERY_VERSION.get(encoding))
    for packet in stream:
        if len(packet):
            yield canonical({"x": packet["x"], "y": packet["y"],
                             "t": packet["t"], "p": packet["on"]})


def kairos_metadata(path):
    """Encoding and sensor size from a Kairos .raw.kai header, or None when
    the file does not start with the Kairos signature."""
    with path.open("rb") as handle:
        head = handle.read(KAIROS_HEADER_BYTES)
    if not head.startswith(KAIROS_SIGNATURE) or len(head) < KAIROS_HEADER_BYTES:
        return None
    version, kind = head[10], head[11]
    if kind != 0:
        raise ValueError("Kairos file type {} is not EVT3, the only type this reader knows".format(kind))
    width, height = struct.unpack("<HH", head[12:16])
    return {"format": "kairos", "kairos_version": version, "encoding": "evt3",
            "width": width, "height": height}


EVT3_HEADER = b"% evt 3.0\n% end\n"    # the smallest Prophesee header faery accepts


def _kairos_packets(path, width, height):
    """Events of a Kairos file. faery reads only whole files and does not know
    the Kairos header, so the header is swapped for a minimal Prophesee EVT3
    header in a temporary copy, which faery decodes and which is deleted
    afterwards. The EVT3 data starts with a time word, so it decodes
    correctly with no earlier decoder state."""
    import faery
    tmp = Path(tempfile.mkdtemp(prefix="kairos-"))
    stream = None
    try:
        payload = tmp / "events.raw"
        with path.open("rb") as src, payload.open("wb") as dst:
            dst.write(EVT3_HEADER)
            src.seek(KAIROS_HEADER_BYTES)
            shutil.copyfileobj(src, dst, 16 << 20)
        stream = faery.events_stream_from_file(
            str(payload), dimensions_fallback=(int(width), int(height)), version_fallback="evt3")
        for packet in stream:
            if len(packet):
                yield canonical({"x": packet["x"], "y": packet["y"],
                                 "t": packet["t"], "p": packet["on"]})
    finally:
        # faery holds the copy open until its stream is released; Windows
        # refuses to delete an open file.
        del stream
        shutil.rmtree(tmp, ignore_errors=True)


def open_recording(path, width=None, height=None, encoding=None):
    """Sensor metadata and a lazy packet iterator for a raw recording,
    Prophesee .raw or Kairos .raw.kai.

    The sensor size comes from the header, or from width and height when the
    header lacks it. It is never guessed from where events landed: a quiet
    border is still part of the sensor."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("input file does not exist: {}".format(path))
    try:
        import faery  # noqa: F401
    except ImportError as exc:
        raise ValueError("raw files need faery: python -m pip install -r requirements.txt") from exc
    kairos = kairos_metadata(path)
    if kairos:
        return kairos, _kairos_packets(path, kairos["width"], kairos["height"])
    if path.suffix.lower() != ".raw":
        raise ValueError("this package reads Prophesee .raw and Kairos .raw.kai event recordings; "
                         "{} is not one".format(path.name))
    metadata = {"format": "raw"}
    metadata.update(header_metadata(path))
    encoding = encoding or metadata.get("encoding")
    if not encoding:
        raise ValueError("the .raw header does not name its encoding; pass --encoding evt2, evt21 or evt3")
    metadata["encoding"] = encoding
    for key, override in (("width", width), ("height", height)):
        if override is not None:
            metadata[key] = override
        if key not in metadata or metadata[key] < 1:
            raise ValueError("sensor {} missing from the header; pass --{} in pixels".format(key, key))
    return metadata, _raw_packets(path, encoding, metadata["width"], metadata["height"])


def select_packets(packets, metadata, start_s, duration_s, roi, status):
    """Crop by time and optionally region, preserving input order, as arrays.

    start_s is relative to the first decoded event. Output timestamps keep the
    recording's original microsecond clock. Yields (n, 4) int64 arrays of
    x, y, t, p. Whole packets, not single events: a 4 s window of a busy
    1280x720 recording is about 5 million events, and a Python tuple per event
    costs minutes before any processing starts. Fills `status` with the
    recording's first event time and the absolute start and end selected.
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
