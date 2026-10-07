"""Low-level event denoising.

These filters remove sensor artefacts: isolated background-activity events,
hot pixels that fire far more often than the rest, and rapid repeat events
from one pixel. They do not remove moving vegetation, which is real motion and
is rejected later by the periodicity measurement in `periodicity.py`. They
still matter, because background activity raises the noise floor of every
spectrum computed downstream.

`denoise` runs the standard sequence: hot pixels, then background activity,
then an optional refractory period.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from .events import EventStream

# numba compiles the per-event loop in _baf_kernel. Without it, njit below is
# a do-nothing decorator, and background_activity_mask switches to the
# vectorised voxel filter, because the plain Python loop would be very slow.
try:  # pragma: no cover - availability depends on the environment
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap if not args else args[0]


@njit(cache=True)
def _baf_kernel(x, y, t, width, height, dt_us, radius):
    """Background-activity filter after Delbruck: keep an event only if a
    neighbouring pixel fired within the last `dt_us` microseconds.

    Real edges move across neighbouring pixels in quick succession, while
    sensor noise events are isolated in space and time. Returns a boolean
    keep-mask in input order. The input must be time-sorted.
    """
    # Time of the most recent event at each pixel, in microseconds, either polarity.
    # It starts far in the past (-2^40 us) so no pixel counts as active before
    # its first event.
    last = np.full((height, width), np.int64(-1) << 40, dtype=np.int64)
    keep = np.zeros(x.shape[0], dtype=np.bool_)
    # One pass in event order. The test reads `last`, so the events must be
    # time-sorted for "recent" to mean "earlier and within dt_us".
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        ti = t[i]
        # The neighbourhood is a (2 * radius + 1) px square, clipped at the
        # sensor edge.
        x0 = xi - radius
        if x0 < 0:
            x0 = 0
        x1 = xi + radius
        if x1 > width - 1:
            x1 = width - 1
        y0 = yi - radius
        if y0 < 0:
            y0 = 0
        y1 = yi + radius
        if y1 > height - 1:
            y1 = height - 1
        # Keep the event if any OTHER pixel in the square fired within the last
        # dt_us microseconds. The event's own pixel is skipped, so a pixel that
        # fires repeatedly on its own cannot support itself.
        found = False
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if xx == xi and yy == yi:
                    continue
                if ti - last[yy, xx] <= dt_us:
                    found = True
                    break
            if found:
                break
        keep[i] = found
        # The pixel's timestamp is updated whether or not the event was kept,
        # so a rejected event can still support a later neighbour.
        last[yi, xi] = ti
    return keep


"""
Every filter below comes in two forms: a `*_mask` function returning a boolean
keep-array over the input events, and a wrapper that applies it. `denoise`
chains the masks to track where each surviving event sat in the input, so
per-event data such as ground-truth labels can follow the events exactly.
A new filter should provide a mask form too.
"""


def background_activity_mask(
    ev: EventStream, dt_us: int = 5_000, radius: int = 1
) -> np.ndarray:
    """Boolean keep-mask: True where another pixel within `radius` px fired in
    the previous `dt_us` microseconds. The mask form of
    `background_activity_filter`.

    `ev` must already be time-sorted. The mask is not sorted here, because it
    has to line up with the caller's event order. Without numba this falls
    back to `voxel_density_mask`, with a warning.
    """
    if len(ev) == 0:
        return np.zeros(0, bool)
    if not _HAVE_NUMBA:
        warnings.warn(
            "numba is not installed; falling back to the voxel-density filter, "
            "which approximates the BAF more coarsely. pip install numba",
            RuntimeWarning,
            stacklevel=2,
        )
        return voxel_density_mask(ev, dt_us=dt_us, radius=radius)
    return _baf_kernel(
        ev.x.astype(np.int64), ev.y.astype(np.int64), ev.t,
        int(ev.width), int(ev.height), int(dt_us), int(radius),
    )


def background_activity_filter(
    ev: EventStream, dt_us: int = 5_000, radius: int = 1
) -> EventStream:
    """Drop events with no neighbouring pixel active in the last `dt_us`.

    `radius` is the neighbourhood half-width in pixels: 1 means the 8
    surrounding pixels. `dt_us` is the parameter that matters. Too long and
    more noise survives, because some neighbour will have fired by chance.
    Too short and wing events are deleted with the noise, because the wing tip
    crosses a given pixel only once per stroke. Keep it at or above one
    wingbeat period (5 ms at 200 Hz) unless you have checked how many wing
    events survive.
    """
    if len(ev) == 0:
        return ev
    ev = ev.sorted_by_time()
    return ev.select(background_activity_mask(ev, dt_us=dt_us, radius=radius))


def voxel_density_mask(
    ev: EventStream, dt_us: int = 5_000, radius: int = 1, min_count: int = 2
) -> np.ndarray:
    """Boolean keep-mask: True where the event's space-time voxel holds at
    least `min_count` events. The mask form of `voxel_density_filter`, and
    the fallback `background_activity_mask` uses when numba is missing."""
    if len(ev) == 0:
        return np.zeros(0, bool)
    # Voxels are (2 * radius + 1) px square, matching the BAF neighbourhood
    # width, and dt_us microseconds long, counted from the first event.
    cell = 2 * radius + 1
    gx = ev.x // cell
    gy = ev.y // cell
    gt = (ev.t - ev.t.min()) // max(dt_us, 1)
    nx = int(gx.max()) + 1
    ny = int(gy.max()) + 1
    # Pack (time bin, row, column) into one integer per event so a single
    # np.unique counts every voxel's population at once.
    key = (gt * ny + gy) * nx + gx
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    # Unlike the BAF, the event counts itself and polarity is ignored, so
    # min_count=2 means "one other event shares the voxel".
    return counts[inv] >= min_count


def voxel_density_filter(
    ev: EventStream, dt_us: int = 5_000, radius: int = 1, min_count: int = 2
) -> EventStream:
    """Keep events in space-time voxels that hold at least `min_count` events.

    A vectorised stand-in for the background-activity filter. It is coarser:
    two events on either side of a voxel boundary do not count as neighbours.
    It needs no compiler and runs in a few numpy passes.
    """
    if len(ev) == 0:
        return ev
    return ev.select(voxel_density_mask(ev, dt_us, radius, min_count))


def hot_pixel_mask(
    ev: EventStream, max_rate_hz: Optional[float] = None, percentile: float = 99.9
) -> Tuple[np.ndarray, np.ndarray]:
    """Find hot pixels and mask out their events.

    Returns (keep_mask, hot_pixel_coords), where coords are (y, x) pairs. A
    pixel is hot when its event count exceeds `max_rate_hz` times the stream
    duration, or, if no rate is given, the `percentile` of counts over the
    pixels that fired at all. The percentile default of 99.9 always removes
    roughly the top 0.1% of active pixels, whether or not any is faulty.
    """
    if len(ev) == 0:
        return np.zeros(0, bool), np.zeros((0, 2), np.int64)
    counts = ev.count_image()
    if max_rate_hz is not None:
        thresh = max_rate_hz * max(ev.duration_s, 1e-9)
    else:
        active = counts[counts > 0]
        thresh = np.percentile(active, percentile) if active.size else np.inf
    hot = counts > thresh
    if not hot.any():
        return np.ones(len(ev), bool), np.zeros((0, 2), np.int64)
    return ~hot[ev.y, ev.x], np.argwhere(hot)


def hot_pixel_filter(
    ev: EventStream, max_rate_hz: Optional[float] = None, percentile: float = 99.9
) -> Tuple[EventStream, np.ndarray]:
    """Remove the events of pixels that fire far more than the rest of the
    sensor. See `hot_pixel_mask` for the rule.

    Returns the filtered stream and the (y, x) coordinates of the removed
    pixels, so the same list can be applied to other files from one session.
    """
    if len(ev) == 0:
        return ev, np.zeros((0, 2), np.int64)
    keep, hot = hot_pixel_mask(ev, max_rate_hz, percentile)
    return ev.select(keep), hot


def refractory_mask(ev: EventStream, dt_us: int = 1_000) -> np.ndarray:
    """Boolean keep-mask: drop an event that follows the previous event at the
    same pixel and polarity by less than `dt_us` microseconds. The mask form
    of `refractory_filter`."""
    if len(ev) == 0:
        return np.zeros(0, bool)
    # One integer per (pixel, polarity) channel: pixel index times 2, plus 1 for ON.
    key = (ev.y.astype(np.int64) * ev.width + ev.x) * 2 + (ev.p > 0)
    # Sort by channel, then by time within a channel, so each channel's events
    # sit next to each other in time order. Vectorised, no per-pixel loop.
    order = np.lexsort((ev.t, key))
    k_sorted = key[order]
    t_sorted = ev.t[order]
    # `same` marks an event whose predecessor in sorted order is the same channel.
    same = np.empty(len(ev), bool)
    same[0] = False
    same[1:] = k_sorted[1:] == k_sorted[:-1]
    # Gap in microseconds to that predecessor. Only meaningful where `same` is True.
    gap = np.empty(len(ev), np.int64)
    gap[0] = np.iinfo(np.int64).max
    gap[1:] = t_sorted[1:] - t_sorted[:-1]
    # Keep the first event of each channel and any event at least dt_us after
    # the previous one. The gap is measured to the previous event whether or not
    # that one was kept, so a burst spaced under dt_us keeps only its first event.
    keep_sorted = (~same) | (gap >= dt_us)
    # Scatter back to the input order so the mask lines up with `ev`.
    keep = np.zeros(len(ev), bool)
    keep[order] = keep_sorted
    return keep


def refractory_filter(ev: EventStream, dt_us: int = 1_000) -> EventStream:
    """Enforce a minimum interval between events at the same pixel and polarity.

    Useful against a pixel firing repeatedly on one high-contrast edge. Set
    `dt_us` well below the wingbeat period, or it removes the modulation you
    want to measure: at 250 Hz the period is 4000 us, so 1000 us is already a
    quarter cycle.
    """
    if len(ev) == 0:
        return ev
    ev = ev.sorted_by_time()
    return ev.select(refractory_mask(ev, dt_us=dt_us))


def denoise(
    ev: EventStream,
    hot_percentile: float = 99.9,
    baf_dt_us: int = 5_000,
    baf_radius: int = 1,
    refractory_us: int = 0,
    return_index: bool = False,
):
    """Standard cleaning: hot pixels, then background activity, then an
    optional refractory period.

    `hot_percentile` goes to `hot_pixel_mask`. `baf_dt_us` and `baf_radius`
    go to the background-activity filter. `refractory_us=0` skips the
    refractory step.

    With `return_index=True` this also returns an integer array giving each
    surviving event's position in the time-sorted input. Carry any per-event
    data, such as ground-truth labels, through with that index. Matching
    events back afterwards by their (t, x, y) values is error-prone, because
    several events can share a timestamp and pixel.
    """
    ev = ev.sorted_by_time()
    # Position of each surviving event in the sorted input, narrowed by each
    # filter's keep-mask in turn.
    idx = np.arange(len(ev), dtype=np.int64)

    keep, _ = hot_pixel_mask(ev, percentile=hot_percentile)
    ev = ev.select(keep)
    idx = idx[keep]

    keep = background_activity_mask(ev, dt_us=baf_dt_us, radius=baf_radius)
    ev = ev.select(keep)
    idx = idx[keep]

    if refractory_us > 0:
        keep = refractory_mask(ev, dt_us=refractory_us)
        ev = ev.select(keep)
        idx = idx[keep]

    return (ev, idx) if return_index else ev
