"""Blob detection in short time windows.

The detector slides a window of `window_us` microseconds along the stream,
starting a new window every `hop_us`. In each window it counts events per
pixel, blurs the count image, thresholds it, and cuts the mask into connected
components. Each component that passes the size and event-count gates becomes
one Detection.

This stage is permissive on purpose. Its job is to propose candidates with high
recall, not to decide what is an insect. A swaying leaf cluster makes a good
blob and will be proposed here. Rejection happens later, on the periodicity of
each track's events, because that is where insects and vegetation differ.
Tightening this stage to suppress clutter by shape or size loses the small,
fast, low-contrast insects first.

Four things to know before choosing parameters:

  The sensitivity floor is set by `pixel_threshold`, not by the event-count
  gates. The Gaussian blur is normalised, so one event adds only 0.159 to its
  own pixel at blur_sigma=1.0. A pixel_threshold of 0.5 therefore needs about
  3 events on the same pixel before that pixel joins the mask. The
  `coincidence` argument states the threshold in events directly.

  `min_events_frame` skips a whole window with fewer events.
  `min_events_blob` drops a single blob with fewer events. `min_events` is an
  older name for `min_events_blob`, kept so existing calls still work.

  One insect can appear as two blobs in the same window, a few tens of pixels
  apart. Each blob then seeds its own track. `nms_dist_px` keeps only the
  richer of two blobs closer than that distance in the same window. It is on
  by default (25 px). It cannot tell one animal reported twice from two animals
  close together, so set it to 0 when every nearby animal must be counted.

  Window start times lie on an absolute grid: multiples of `hop_us` from time
  zero. Two calls with the same `hop_us` share a grid, whatever events each one
  was given, so their detections can be matched by `t_us`. Pass
  `grid_origin_us` to set the grid origin explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# OpenCV is optional. Without it the blur and the component labelling fall
# back to scipy, which is slower.
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .events import EventStream

# Optional compiled extension, not shipped with this package. Everything works
# without it.
try:
    import insect_evs_rust as _rust_detect
except ImportError:  # pragma: no cover
    _rust_detect = None


def rust_available() -> bool:
    """True when the optional compiled extension insect_evs_rust is installed.
    The package runs without it."""
    return _rust_detect is not None


@dataclass
class Detection:
    """One blob found in one time window."""
    t_us: int              # window centre, us
    t0_us: int             # window start, us
    t1_us: int             # window end, us (exclusive)
    x: float               # centroid, pixels: mean of the blob's event coordinates
    y: float
    bbox: tuple            # x0, y0, x1, y1 in pixels (half-open)
    n_events: int          # events that landed on the blob's pixels
    area_px: int           # mask pixels in the blob
    polarity_balance: float  # (ON - OFF) / total within the blob

    @property
    def size_px(self) -> float:
        """Larger side of the bounding box, in pixels."""
        x0, y0, x1, y1 = self.bbox
        return float(max(x1 - x0, y1 - y0))


def merge_detections(
    detections: List["Detection"],
    time_tol_us: float = 5_000.0,
    dist_tol_px: float = 12.0,
) -> List["Detection"]:
    """Collapse detections that coincide in space and time into one.

    Two detections clash when their window centres are within `time_tol_us`
    microseconds and their centroids within `dist_tol_px` pixels. Of any group
    that clashes, the one with the most events is kept. Used two ways: to merge
    the same target seen at several window lengths, and, with time_tol_us=0, to
    suppress two blobs on one target in the same window.

    The richest detection wins, not the earliest, because a longer window often
    sees a faint target that a shorter window catches only partly.

    Detections are grouped into time buckets one `time_tol_us` wide, so each is
    compared only with the buckets beside it rather than with every other
    detection. The result is the same as the all-pairs comparison.
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
    """Run detect_blobs at several window lengths and merge the results.

    Whether a target is detected depends mostly on how many of its events land
    in one window. A faint or distant insect may give too few events in 5 ms to
    form a blob, and enough in 40 or 80 ms. A bright, fast insect is best seen
    in a short window, where it does not smear. So this runs every window length
    in `windows_us` and keeps whatever any of them finds; merge_detections then
    removes the copies of one target found at several lengths.

    The hop is `scale_hop` times each window. Doubling the window halves the
    number of steps, so the default five lengths cost about twice the shortest
    one alone, not five times.

    The blob event gate (`min_events_blob`, or its older name `min_events`) is
    given for `window_us` and scaled in proportion to each window, with a floor
    of 8. A fixed gate would make the long windows far easier to trigger and
    flood the merge with clutter. Other keyword arguments go to detect_blobs
    unchanged; any `hop_us` passed is ignored.
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
            # Only the blob gate scales with the window. A whole-frame gate
            # passed through kwargs is left as given: scaling it would make the
            # long windows skip frames that the short ones keep.
            min_events_blob=max(int(round(base_min_events * scale)), 8),
            # None gives each window length its own absolute grid,
            # (t0 // hop) * hop. The default hops (2.5, 5, 10, 20, 40 ms) are
            # multiples of each other, so every grid starts on a multiple of
            # the shortest hop and all lengths stay in phase.
            grid_origin_us=grid_origin_us,
            **kwargs,
        ))
    # The merge radius must cover the longest window's blobs. At 80 ms a fast
    # target smears across tens of pixels, so its centroid can sit well over
    # 12 px from the short-window detection of the same insect. The radius grows
    # by about 0.36 px per ms of the longest window, above a 12 px base.
    dist = max(merge_dist_px, 0.3 * max(windows_us) / 1000.0 * 1.2 + 12.0)
    return merge_detections(out, merge_time_us, dist)


def _label_components(mask: np.ndarray):
    """8-connected components of a boolean mask.

    Returns (n, labels, stats, centroids) in OpenCV's layout: label 0 is the
    background, and each stats row is x, y, width, height, area in pixels.
    Uses OpenCV when installed, else scipy with the same output layout.
    """
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
    """Blurred density that one event leaves at its own pixel.

    `pixel_threshold` should be read in units of this value: 0.159 at sigma 1.0
    and 0.111 at sigma 1.2. So pixel_threshold=0.5 at sigma 1.0 does not mean
    half an event. It means about 3.1 events on the same pixel. With no blur
    one event counts 1.0.
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

    Returns a list of Detection, in window order. All times are in
    microseconds (us) and all sizes in pixels (px).

    window_us
        Length of each window. Long enough to collect a readable blob, short
        enough that the target does not smear. At 150 px/s a 10 ms window
        smears 1.5 px, which is fine; at 1000 px/s use 2 to 3 ms.
    hop_us
        Time between the starts of successive windows. A hop of half the
        window means each event falls in two windows.
    blur_sigma
        Width in px of the Gaussian blur applied to the count image, so that
        events on neighbouring pixels add up. 0 turns the blur off.
    coincidence
        How many events must land on one pixel before it can join a blob. This
        sets the detector's sensitivity. It is converted to a density as
        `coincidence * psf_peak(blur_sigma)`, so it means the same thing at any
        blur width. `coincidence=1` admits a pixel with a single event.
    pixel_threshold
        The density threshold, if you would rather set it directly. Overrides
        `coincidence` when given. It is not an event count: the blur is
        normalised, so one event adds only `psf_peak(blur_sigma)` to its own
        pixel, 0.159 at sigma 1.0. A threshold of 1.5 at sigma 1.2 needs about
        14 events on one pixel and misses most insects.
    min_events_blob
        A blob with fewer events is dropped. `min_events` is an older name for
        the same thing, kept so existing calls still work; when given it
        overrides `min_events_blob`. Only events on pixels that cleared the
        threshold are counted, so this count is already lower than the number
        of events the insect produced in the window.
    min_events_frame
        A window with fewer events in the whole frame is skipped. Default 0,
        which only skips empty windows.
    min_area_px, max_area_px
        A blob whose mask area in pixels falls outside this range is dropped.
        `max_area_px=None` means 5% of the sensor's pixels.
    max_size_px
        A blob whose larger bounding-box side exceeds this is dropped. The limit
        must grow with the window, because a long window smears a fast target
        across tens of pixels. `None` sets it to 40 px plus 1.2 px per ms of
        window, capped at 35% of the sensor's shorter side.
    nms_dist_px
        Within one window, of two blobs closer than this, keep only the one with
        more events. One insect often appears as two blobs, and each would
        start its own track. Default 25 px.

        Pass 0 when every animal must be counted, for example when building a
        reference set from labelled events. Suppression cannot tell one animal
        reported twice from two animals within 25 px of each other, so it merges
        genuine neighbours and the count comes out too low.
    grid_origin_us
        Start the window grid here. By default windows start on multiples of
        `hop_us` from time zero, so calls on different streams that share a
        timebase and a hop already share a grid. Give an origin to force a
        particular phase. If it lies after the first event, the grid is moved
        back by whole hops so the first event still falls in a window.
    backend
        "auto" uses the compiled insect_evs_rust extension when it is installed
        and the call has no blur and no negative times, else the Python/OpenCV
        path. "rust" forces the extension. "python" forces the reference path.
    """
    if len(ev) == 0:
        return []
    if min_events is not None:
        min_events_blob = min_events

    # backend="auto" uses the Rust extension only where it gives the same
    # detections as the Python path: no blur, and no negative times. Outside
    # that it differs in two ways. With blur, its density within about 4 px of
    # the frame border differs from OpenCV's, so it can report extra blobs
    # there. With negative times or a negative grid origin, its window grid
    # differs from Python's. backend="rust" still forces it.
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
    """detect_blobs on the optional compiled extension insect_evs_rust. Same
    arguments and output as _detect_blobs_python. detect_blobs picks it only
    where the two give the same detections; see the backend note there."""
    # Sort by time before handing the raw arrays to the extension, as the
    # Python path does before slicing windows.
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
    # The extension returns one array per Detection field, in this order, with
    # one entry per detection. Rebuild the Detection objects the rest of the
    # package expects, casting to plain Python ints and floats.
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
    """detect_blobs on the reference Python/OpenCV path. See detect_blobs for
    the meaning of every argument."""
    # Turn "events coincident on one pixel" into the blurred density the mask
    # is thresholded at. An explicit pixel_threshold skips this.
    if pixel_threshold is None:
        pixel_threshold = coincidence * psf_peak(blur_sigma)

    # time_slice below uses searchsorted, which needs time order.
    ev = ev.sorted_by_time()
    t_end = int(ev.t[-1])
    if grid_origin_us is None:
        # Start on a multiple of hop_us from time zero, not at this stream's
        # first event. If the grid followed the first event, its phase would
        # depend on which events an upstream filter happened to keep, and
        # changing a filter setting would shift every window. Window phase
        # alone changes which blobs form, and with them the tracking results.
        # With an absolute grid, any two calls on the same timebase and hop
        # share windows, whatever events they were given.
        t_start = (int(ev.t[0]) // hop_us) * hop_us
    else:
        t_start = int(grid_origin_us)
        # An origin after the first event would leave the earliest events in
        # no window. Step back by whole hops, which keeps the grid phase, until
        # the first window starts at or before the first event.
        if t_start > int(ev.t[0]):
            t_start -= ((t_start - int(ev.t[0])) // hop_us + 1) * hop_us
    # Default area cap: 5% of the sensor's pixels. A component larger than
    # that is rejected below.
    if max_area_px is None:
        max_area_px = int(0.05 * ev.width * ev.height)
    if max_size_px is None:
        # A target moving at 1000 px/s smears 1 px per ms, so the cap grows
        # with the window: 40 px plus 1.2 px per ms, and never more than 35% of
        # the sensor's shorter side. A fixed cap would reject fast insects in
        # long windows.
        max_size_px = int(min(0.35 * min(ev.width, ev.height),
                              40 + window_us / 1000.0 * 1.2))

    detections: List[Detection] = []
    # Windows of window_us microseconds start every hop_us. The last window
    # ends at or before t_end; the max() guarantees at least one window when
    # the stream is shorter than a window.
    for t0 in range(t_start, max(t_end - window_us + 1, t_start + 1), hop_us):
        t1 = t0 + window_us
        win = ev.time_slice(t0, t1)
        # Skip empty windows, and windows under the optional whole-frame gate.
        if len(win) < max(min_events_frame, 1):
            continue

        # Events per pixel in this window, both polarities, then a Gaussian
        # blur so neighbouring events add up. Pixels at or above the threshold
        # form the mask that components are cut from.
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

        # 8-connected components of the mask. Label 0 is the background, so
        # n <= 1 means no component.
        n, labels, stats, centroids = _label_components(mask)
        if n <= 1:
            continue

        # Component label at each event's pixel. An event on a pixel below the
        # threshold gets label 0 and joins no blob.
        ev_labels = labels[win.y, win.x]
        for i in range(1, n):
            # Gate each component on mask area in pixels, then on its larger
            # bounding-box side in pixels.
            x0, y0, w, h, area = stats[i][:5]
            if area < min_area_px or area > max_area_px:
                continue
            if max(w, h) > max_size_px:
                continue
            # Then on the number of events that landed on the component's pixels.
            sel = ev_labels == i
            n_ev = int(sel.sum())
            if n_ev < min_events_blob:
                continue
            # Polarities are +1/-1, so the mean is (ON - OFF) / total.
            pol = win.p[sel]
            balance = float(pol.sum()) / n_ev
            # The centroid is the mean of the blob's event coordinates, not the
            # pixel centroid from the labeller, so busy pixels weigh more. The
            # bbox is half-open and t_us is the window centre.
            detections.append(
                Detection(
                    t_us=(t0 + t1) // 2, t0_us=t0, t1_us=t1,
                    x=float(win.x[sel].mean()), y=float(win.y[sel].mean()),
                    bbox=(int(x0), int(y0), int(x0 + w), int(y0 + h)),
                    n_events=n_ev, area_px=int(area), polarity_balance=balance,
                )
            )
    if nms_dist_px > 0:
        # Same window only: time_tol_us=0 compares detections that share a
        # window centre. This removes one animal reported twice in one window,
        # and never merges a target with its own detection from an earlier one.
        detections = merge_detections(detections, time_tol_us=0,
                                      dist_tol_px=nms_dist_px)
    return detections
