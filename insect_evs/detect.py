"""Blob detection in short time windows.

Deliberately permissive. This stage exists to propose candidates with high
recall, not to decide what is an insect: a swaying leaf cluster produces a
perfectly good blob and will be proposed here. Rejection happens later, on the
periodicity of the candidate's event stream, because that is the axis on which
insects and vegetation actually separate. Tuning this stage to suppress clutter
spatially is the classic mistake; it costs you the small, fast, low-contrast
targets first.

READ THIS BEFORE CHOOSING PARAMETERS. Instrumentation against 609 reference bee
tracks on the AEMOT Bee Swarm sequence found several traps here, all now either
fixed or documented in place:

  THE SENSITIVITY FLOOR IS `pixel_threshold`, NOT `min_events`. `cv2.GaussianBlur`
  normalises, so a single event deposits only 0.1592 at its own pixel at
  blur_sigma=1.0 -- pixel_threshold=0.5 therefore demands 3.1 COINCIDENT EVENTS on
  one pixel before any mask pixel exists at all. Use `coincidence=` to say that in
  events rather than guessing at density units.

  TWO THINGS WERE CALLED `min_events`. One rejected the whole frame, one rejected a
  blob. They are now `min_events_frame` and `min_events_blob`; `min_events` remains
  as an alias for the blob gate, which is what every caller meant.

  SAME-FRAME DUPLICATES ARE THE DOMINANT TRACKING DEFECT. Measured, 49% of track
  breaks are two output tracks alive on one bee at the same instant: 1,060 of
  19,959 bee-frames carry more than one detection for a single bee, median
  separation 26.6 px, the second blob holding 45% of the first's events. One insect
  torn in two. `nms_dist_px` now suppresses this at source and is ON by default.

  THE WINDOW GRID IS ANCHORED ON EACH STREAM'S OWN FIRST TIMESTAMP, so two calls on
  two different streams do NOT share a `t_us` grid. Matching detections between
  calls by equal `t_us` silently returns zero matches. Pass `grid_origin_us` to tie
  several calls to one grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .events import EventStream

try:
    import insect_evs_rust as _rust_detect
except ImportError:  # pragma: no cover
    _rust_detect = None


def rust_available() -> bool:
    return _rust_detect is not None


@dataclass
class Detection:
    t_us: int              # window centre
    t0_us: int
    t1_us: int
    x: float               # centroid, pixels
    y: float
    bbox: tuple            # x0, y0, x1, y1 (half-open)
    n_events: int
    area_px: int
    polarity_balance: float  # (ON - OFF) / total within the blob

    @property
    def size_px(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return float(max(x1 - x0, y1 - y0))


def merge_detections(
    detections: List["Detection"],
    time_tol_us: float = 5_000.0,
    dist_tol_px: float = 12.0,
) -> List["Detection"]:
    """Collapse detections of the same target arriving from different scales.

    Keeps the one with the most events among any group that coincides in space
    and time. Preferring the richest rather than the earliest matters because
    the whole point of the longer windows is that they see targets the short
    ones cannot, so the short-window version is often the impoverished one.

    Bucketed by time, because the obvious implementation compares every survivor
    against every candidate and is O(n^2): on a full recording that cost 767 s
    for a call whose detection work was 20 s. Two detections can only clash if
    they are within `time_tol_us`, so only neighbouring time buckets need
    checking, and the result is bit-identical.
    """
    if not detections:
        return []
    order = sorted(range(len(detections)),
                   key=lambda i: (-detections[i].n_events, detections[i].t_us))

    # Bucket width is the clash window itself, so a candidate can only collide
    # with survivors in its own bucket or the two beside it.
    width = max(int(time_tol_us), 1)
    buckets: dict = {}
    kept: List[Detection] = []
    for i in order:
        d = detections[i]
        b = int(d.t_us) // width
        clash = False
        for nb in (b - 1, b, b + 1):
            for k in buckets.get(nb, ()):
                if abs(k.t_us - d.t_us) <= time_tol_us and \
                        np.hypot(k.x - d.x, k.y - d.y) <= dist_tol_px:
                    clash = True
                    break
            if clash:
                break
        if not clash:
            kept.append(d)
            buckets.setdefault(b, []).append(d)
    kept.sort(key=lambda d: d.t_us)
    return kept


def detect_blobs_multiscale(
    ev: EventStream,
    windows_us: Sequence[int] = (5_000, 10_000, 20_000, 40_000, 80_000),
    scale_hop: float = 0.5,
    merge_time_us: float = 5_000.0,
    merge_dist_px: float = 12.0,
    grid_origin_us: Optional[int] = None,
    **kwargs,
) -> List["Detection"]:
    """Detect at a bank of accumulation windows and merge the results.

    A single window length is a single arbitrary choice, and the diagnostic says
    it is the wrong one for most targets. Measured over 3,566 annotated keyframe
    boxes, detection recall depends far more on how many events land inside the
    box than on any threshold: below 20 events per 10 ms window recall is
    literally zero, and above 250 it exceeds 0.96. The detector is not
    threshold-limited, it is starved. Widening the window from 5 ms to 80 ms
    lifted recall from 0.30 to 0.66 on one scene and 0.16 to 0.66 on another,
    while dropping the pixel threshold from 2.5 to 0.4 bought only about 0.13.

    So integrate at several scales and keep whatever any of them finds. Bright
    fast targets are caught by the short windows without smearing; faint slow
    ones need the long ones.

    The hop is proportional to the window, which is what keeps this affordable:
    each doubling of the window halves the number of steps, so the whole bank
    costs roughly twice the finest scale alone rather than five times it.

    `min_events` is scaled with the window too, since a fixed floor would make
    the long windows trivially easy to trigger and flood the merge with clutter.
    """
    if len(ev) == 0:
        return []
    base_window = kwargs.pop("window_us", 5_000)
    base_min_events = kwargs.pop("min_events", None)
    if base_min_events is None:
        base_min_events = kwargs.pop("min_events_blob", 10)
    else:
        kwargs.pop("min_events_blob", None)
    kwargs.pop("hop_us", None)

    out: List[Detection] = []
    for w in windows_us:
        scale = w / float(base_window)
        out.extend(detect_blobs(
            ev,
            window_us=int(w),
            hop_us=max(int(w * scale_hop), 1),
            # The BLOB gate scales with the window; the FRAME gate must not, or
            # the long scales silently switch themselves off. They used to be one
            # parameter, and the 80 ms scale inherited a frame gate of 160.
            min_events_blob=max(int(round(base_min_events * scale)), 8),
            # Forward the caller's origin, or None. None lets every scale take
            # its own ABSOLUTE anchor ((t0 // hop) * hop); the default hops
            # nest (2.5/5/10/20/40 ms), so all scales land phase-0 on the
            # shared timebase and detections still match across windows by
            # timestamp. This used to hard-code int(ev.t[0]) -- anchoring on
            # the first SURVIVING event, the exact survivor-dependent-phase
            # defect the absolute anchor was built to remove (same arm +68
            # aligned / -17 misaligned) -- and, because it was hard-coded,
            # passing grid_origin_us raised TypeError, so the documented
            # remedy was unreachable. Found by the 2026-08-10 review.
            grid_origin_us=grid_origin_us,
            **kwargs,
        ))
    # The merge radius has to cover the coarsest scale's own blobs: at 80 ms a
    # fast target legitimately spans tens of pixels, and a fixed 12 px cannot
    # merge its coarse and fine detections at all.
    dist = max(merge_dist_px, 0.3 * max(windows_us) / 1000.0 * 1.2 + 12.0)
    return merge_detections(out, merge_time_us, dist)


def _label_components(mask: np.ndarray):
    """Connected components, with a small pure-numpy fallback if cv2 is absent."""
    if cv2 is not None:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        return n, labels, stats, centroids
    from scipy import ndimage

    labels, n_obj = ndimage.label(mask, structure=np.ones((3, 3)))
    stats = np.zeros((n_obj + 1, 5), np.int64)
    centroids = np.zeros((n_obj + 1, 2), np.float64)
    for i in range(1, n_obj + 1):
        ys, xs = np.nonzero(labels == i)
        stats[i] = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, len(xs)]
        centroids[i] = [xs.mean(), ys.mean()]
    return n_obj + 1, labels, stats, centroids


def psf_peak(blur_sigma: float) -> float:
    """Blurred density a SINGLE event deposits at its own pixel.

    The number `pixel_threshold` has to be read against. 0.1592 at sigma 1.0,
    0.1105 at 1.2. Without it, "blurred event count per pixel" reads as though
    pixel_threshold=0.5 meant half an event; it means 3.1 of them.
    """
    if blur_sigma <= 0:
        return 1.0
    if cv2 is not None:
        d = np.zeros((41, 41), np.float32)
        d[20, 20] = 1.0
        return float(cv2.GaussianBlur(d, (0, 0), blur_sigma)[20, 20])
    return float(1.0 / (2.0 * np.pi * blur_sigma ** 2))


def detect_blobs(
    ev: EventStream,
    window_us: int = 5_000,
    hop_us: int = 2_500,
    blur_sigma: float = 1.0,
    pixel_threshold: Optional[float] = None,
    coincidence: float = 3.0,
    min_events: Optional[int] = None,
    min_events_blob: int = 10,
    min_events_frame: int = 0,
    min_area_px: int = 3,
    max_area_px: Optional[int] = None,
    max_size_px: Optional[int] = None,
    nms_dist_px: float = 25.0,
    grid_origin_us: Optional[int] = None,
    backend: str = "auto",
) -> List[Detection]:
    """Sliding-window connected-component detection.

    backend
        ``"auto"`` uses the Rust accelerator when installed, else Python/OpenCV.
        ``"rust"`` requires ``insect_evs_rust``. ``"python"`` forces the reference
        path (use this when checking bit-level agreement).

    window_us
        Long enough to accumulate a readable blob, short enough that the target
        does not smear. At 150 px/s a 10 ms window smears 1.5 px, which is fine;
        at 1000 px/s you want 2-3 ms.
    coincidence
        HOW MANY EVENTS MUST LAND ON ONE PIXEL before it can join a blob. This is
        the detector's real sensitivity floor and the parameter to reach for. It
        is converted to a density internally as `coincidence * psf_peak(sigma)`,
        so it means the same thing at any blur width -- which raw
        `pixel_threshold` does not. `coincidence=1` admits a lone event.
    pixel_threshold
        The raw density, if you would rather set it directly. Overrides
        `coincidence` when given. NOTE it is NOT "blurred event count per pixel":
        cv2.GaussianBlur normalises, so one event deposits only
        `psf_peak(blur_sigma)` = 0.159 at sigma 1.0. The old default of 1.5 with
        sigma 1.2 demanded 13.6 coincident events and found 99 tracks where the
        labels hold 609.
    min_events_blob
        Events a BLOB must contain. This is what almost every caller means, and
        `min_events` is kept as an alias for it.
        Two caveats, both measured: it is compared against a count that has
        ALREADY been decimated, because an event joins a blob only if its own
        pixel cleared the threshold (20% of events in a window are dropped before
        this test); and of the components it rejects, 68% are entirely target.
    min_events_frame
        Events the WHOLE FRAME must contain before the window is considered at
        all. Defaults to 0 -- off. It was previously fused with the blob gate
        under one name, where it never fired on whole-scene calls but erased 70%
        of targets when the same value was passed for a single target's events.
    max_size_px
        Reject blobs whose larger bbox side exceeds this. Scale it with the
        window: at 5-20 ms it never fires, but at 80 ms a fixed 90 px discards
        43% of all target events, because a long window legitimately smears a
        fast target across tens of pixels. `None` scales it automatically.
    nms_dist_px
        Suppress same-instant duplicate detections within this distance, keeping
        the richest. Two blobs on one animal are the single largest cause of
        track fragmentation -- 49% of all track breaks -- so this is ON.

        PASS 0 WHEN BUILDING A GROUND-TRUTH REFERENCE SET. Suppression cannot
        tell one animal reported twice from two animals passing within 25 px, so
        on a labelled-target-only stream it merges genuine neighbours: measured,
        the AEMOT reference set comes back as 1,381 tracks with the default
        against 1,505 with `nms_dist_px=0`. A denominator built with it is
        silently 8% too small, and every recall computed against it too high.
    grid_origin_us
        Anchor the window grid here instead of at this stream's first event, so
        that several calls on different streams share one `t_us` grid and their
        detections can be matched by timestamp.
    """
    if len(ev) == 0:
        return []
    if min_events is not None:
        min_events_blob = min_events

    # backend="auto" prefers Rust ONLY where the two paths are verified
    # bit-identical: the no-blur path. Two measured divergences keep the
    # blurred and negative-time cases on the reference path (2026-08-10
    # review, both reproduced):
    #   * stamp_gaussian applies reflect-101 to the DESTINATION while
    #     scattering -- the transpose of OpenCV's gather -- so blurred density
    #     within ~4 px of the frame border is wrong by up to 2x and the Rust
    #     path emits detections the reference path does not. The 0.07%
    #     detection mismatch previously recorded as float noise is this bug.
    #   * the wrapper encodes grid_origin_us=None as -1, so every negative
    #     origin is silently discarded, and the Rust anchor uses truncating
    #     division where Python floors -- both put the backends on different
    #     window grids for rebased (negative-time) streams.
    # backend="rust" remains an explicit override for people who accept this.
    _neg_time = bool(len(ev)) and (int(ev.t[0]) < 0
                                   or (grid_origin_us is not None
                                       and int(grid_origin_us) < 0))
    _rust_safe = (not blur_sigma) and not _neg_time
    use_rust = backend == "rust" or (backend == "auto" and _rust_detect is not None
                                     and _rust_safe)
    if backend == "rust" and _rust_detect is None:
        raise ImportError(
            "backend='rust' requires insect_evs_rust. "
            "Build with: python -m maturin build --release -i PYTHON "
            "-m crates/insect_evs_rust/pyproject.toml"
        )
    if use_rust:
        return _detect_blobs_rust(
            ev,
            window_us=window_us,
            hop_us=hop_us,
            blur_sigma=blur_sigma,
            pixel_threshold=pixel_threshold,
            coincidence=coincidence,
            min_events_blob=min_events_blob,
            min_events_frame=min_events_frame,
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            max_size_px=max_size_px,
            nms_dist_px=nms_dist_px,
            grid_origin_us=grid_origin_us,
        )
    return _detect_blobs_python(
        ev,
        window_us=window_us,
        hop_us=hop_us,
        blur_sigma=blur_sigma,
        pixel_threshold=pixel_threshold,
        coincidence=coincidence,
        min_events_blob=min_events_blob,
        min_events_frame=min_events_frame,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        max_size_px=max_size_px,
        nms_dist_px=nms_dist_px,
        grid_origin_us=grid_origin_us,
    )


def _detect_blobs_rust(
    ev: EventStream,
    window_us: int,
    hop_us: int,
    blur_sigma: float,
    pixel_threshold: Optional[float],
    coincidence: float,
    min_events_blob: int,
    min_events_frame: int,
    min_area_px: int,
    max_area_px: Optional[int],
    max_size_px: Optional[int],
    nms_dist_px: float,
    grid_origin_us: Optional[int],
) -> List[Detection]:
    ev = ev.sorted_by_time()
    cols = _rust_detect.detect_blobs_arrays(
        ev.x, ev.y, ev.t, ev.p, ev.width, ev.height,
        window_us=window_us,
        hop_us=hop_us,
        blur_sigma=blur_sigma,
        pixel_threshold=pixel_threshold,
        coincidence=coincidence,
        min_events_blob=min_events_blob,
        min_events_frame=min_events_frame,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        max_size_px=max_size_px,
        nms_dist_px=nms_dist_px,
        grid_origin_us=grid_origin_us,
    )
    (t_us, t0_us, t1_us, xs, ys,
     x0, y0, x1, y1, n_events, area_px, polarity_balance) = cols
    out: List[Detection] = []
    for i in range(len(t_us)):
        out.append(
            Detection(
                t_us=int(t_us[i]), t0_us=int(t0_us[i]), t1_us=int(t1_us[i]),
                x=float(xs[i]), y=float(ys[i]),
                bbox=(int(x0[i]), int(y0[i]), int(x1[i]), int(y1[i])),
                n_events=int(n_events[i]), area_px=int(area_px[i]),
                polarity_balance=float(polarity_balance[i]),
            )
        )
    return out


def _detect_blobs_python(
    ev: EventStream,
    window_us: int,
    hop_us: int,
    blur_sigma: float,
    pixel_threshold: Optional[float],
    coincidence: float,
    min_events_blob: int,
    min_events_frame: int,
    min_area_px: int,
    max_area_px: Optional[int],
    max_size_px: Optional[int],
    nms_dist_px: float,
    grid_origin_us: Optional[int],
) -> List[Detection]:
    if pixel_threshold is None:
        pixel_threshold = coincidence * psf_peak(blur_sigma)

    ev = ev.sorted_by_time()
    t_end = int(ev.t[-1])
    if grid_origin_us is None:
        # Anchor on an ABSOLUTE grid, not on this stream's first event.
        #
        # Anchoring on ev.t[0] makes the window phase depend on which events
        # survived whatever filter ran upstream, so changing a Conv1 threshold
        # silently moves the grid. Measured, that is not a rounding effect: the
        # same detector arm scored +68 bees when it happened to share a phase
        # with the grid the reference set was built on and -17 when it did not,
        # and absolute MT moves by 286 of 1,505 across phases with no change to
        # the data. Rounding down to a multiple of hop_us makes any two calls on
        # the same timebase share a grid whatever they were fed.
        t_start = (int(ev.t[0]) // hop_us) * hop_us
    else:
        t_start = int(grid_origin_us)
        if t_start > int(ev.t[0]):
            t_start -= ((t_start - int(ev.t[0])) // hop_us + 1) * hop_us
    if max_area_px is None:
        max_area_px = int(0.05 * ev.width * ev.height)
    if max_size_px is None:
        # A target moving at 1000 px/s smears 1 px per ms, so the cap has to grow
        # with the window or it starts rejecting the very targets it was meant to
        # let through.
        max_size_px = int(min(0.35 * min(ev.width, ev.height),
                              40 + window_us / 1000.0 * 1.2))

    detections: List[Detection] = []
    for t0 in range(t_start, max(t_end - window_us + 1, t_start + 1), hop_us):
        t1 = t0 + window_us
        win = ev.time_slice(t0, t1)
        if len(win) < max(min_events_frame, 1):
            continue

        img = win.count_image().astype(np.float32)
        if blur_sigma > 0:
            if cv2 is not None:
                img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
            else:  # pragma: no cover
                from scipy.ndimage import gaussian_filter

                img = gaussian_filter(img, blur_sigma)
        mask = img >= pixel_threshold
        if not mask.any():
            continue

        n, labels, stats, centroids = _label_components(mask)
        if n <= 1:
            continue

        ev_labels = labels[win.y, win.x]
        for i in range(1, n):
            x0, y0, w, h, area = stats[i][:5]
            if area < min_area_px or area > max_area_px:
                continue
            if max(w, h) > max_size_px:
                continue
            sel = ev_labels == i
            n_ev = int(sel.sum())
            if n_ev < min_events_blob:
                continue
            pol = win.p[sel]
            balance = float(pol.sum()) / n_ev
            detections.append(
                Detection(
                    t_us=(t0 + t1) // 2, t0_us=t0, t1_us=t1,
                    x=float(win.x[sel].mean()), y=float(win.y[sel].mean()),
                    bbox=(int(x0), int(y0), int(x0 + w), int(y0 + h)),
                    n_events=n_ev, area_px=int(area), polarity_balance=balance,
                )
            )
    if nms_dist_px > 0:
        # Same-instant only: time_tol_us=0 compares detections sharing a window
        # centre, so this suppresses one animal reported twice without ever
        # merging a target with its own past.
        detections = merge_detections(detections, time_tol_us=0,
                                      dist_tol_px=nms_dist_px)
    return detections
