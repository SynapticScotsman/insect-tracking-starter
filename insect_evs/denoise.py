"""Low-level event denoising.

These filters remove *sensor* artefacts: uncorrelated background activity, stuck
pixels, and per-pixel bursts. They do not remove vegetation clutter, which is
genuine motion and needs the periodicity gate in `periodicity.py`. Running these
first matters anyway, because background activity otherwise inflates the noise
floor of every periodogram downstream.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from .events import EventStream

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
    """Delbruck background-activity filter: keep events with a recent neighbour."""
    last = np.full((height, width), np.int64(-1) << 40, dtype=np.int64)
    keep = np.zeros(x.shape[0], dtype=np.bool_)
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        ti = t[i]
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
        last[yi, xi] = ti
    return keep


"""
Every filter here comes in two forms: a `*_mask` returning a boolean array over
the input events, and a thin wrapper applying it. The mask form exists so that
`denoise` can compose an exact index map back to the original stream, which is
what lets per-event ground-truth labels follow the events through filtering
without any matching heuristic. Do not add a filter that only returns a stream.
"""


def background_activity_mask(
    ev: EventStream, dt_us: int = 5_000, radius: int = 1
) -> np.ndarray:
    """Boolean keep-mask: does this event have a spatial neighbour active recently?

    Assumes `ev` is time-sorted; the caller is responsible for that, because
    sorting here would invalidate the mask's correspondence to the input order.
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
    """Drop events with no spatial neighbour active in the last `dt_us`.

    `dt_us` is the one parameter that matters. Too long and vegetation clutter
    survives intact; too short and the wing tip, which crosses a pixel once per
    stroke, gets deleted along with the noise. Keep it well above the wingbeat
    period (5 ms at 200 Hz is one full cycle) unless you have checked the effect
    on wing event yield.
    """
    if len(ev) == 0:
        return ev
    ev = ev.sorted_by_time()
    return ev.select(background_activity_mask(ev, dt_us=dt_us, radius=radius))


def voxel_density_mask(
    ev: EventStream, dt_us: int = 5_000, radius: int = 1, min_count: int = 2
) -> np.ndarray:
    if len(ev) == 0:
        return np.zeros(0, bool)
    cell = 2 * radius + 1
    gx = ev.x // cell
    gy = ev.y // cell
    gt = (ev.t - ev.t.min()) // max(dt_us, 1)
    nx = int(gx.max()) + 1
    ny = int(gy.max()) + 1
    key = (gt * ny + gy) * nx + gx
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inv] >= min_count


def voxel_density_filter(
    ev: EventStream, dt_us: int = 5_000, radius: int = 1, min_count: int = 2
) -> EventStream:
    """Vectorised stand-in for the BAF: keep events in populated space-time voxels.

    Coarser than the true neighbourhood test (it cannot see across voxel
    boundaries) but needs no compiler and runs in a few numpy passes.
    """
    if len(ev) == 0:
        return ev
    return ev.select(voxel_density_mask(ev, dt_us, radius, min_count))


def hot_pixel_mask(
    ev: EventStream, max_rate_hz: Optional[float] = None, percentile: float = 99.9
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (keep_mask, hot_pixel_coords) where coords are (y, x) pairs."""
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
    """Remove pixels firing far above the array's own distribution.

    Returns the filtered stream and the (y, x) coordinates that were removed, so
    the mask can be reused across a recording session rather than re-estimated
    per file.
    """
    if len(ev) == 0:
        return ev, np.zeros((0, 2), np.int64)
    keep, hot = hot_pixel_mask(ev, max_rate_hz, percentile)
    return ev.select(keep), hot


def refractory_mask(ev: EventStream, dt_us: int = 1_000) -> np.ndarray:
    if len(ev) == 0:
        return np.zeros(0, bool)
    key = (ev.y.astype(np.int64) * ev.width + ev.x) * 2 + (ev.p > 0)
    order = np.lexsort((ev.t, key))
    k_sorted = key[order]
    t_sorted = ev.t[order]
    same = np.empty(len(ev), bool)
    same[0] = False
    same[1:] = k_sorted[1:] == k_sorted[:-1]
    gap = np.empty(len(ev), np.int64)
    gap[0] = np.iinfo(np.int64).max
    gap[1:] = t_sorted[1:] - t_sorted[:-1]
    keep_sorted = (~same) | (gap >= dt_us)
    keep = np.zeros(len(ev), bool)
    keep[order] = keep_sorted
    return keep


def refractory_filter(ev: EventStream, dt_us: int = 1_000) -> EventStream:
    """Enforce a minimum interval between events at the same pixel and polarity.

    Useful against ringing on high-contrast edges. Set `dt_us` well below the
    wingbeat period, or you will decimate the very modulation you want to
    measure: at 250 Hz the period is 4000 us, so 1000 us is already 1/4 cycle.
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
    """Standard front end: hot pixels, then background activity, then optional refractory.

    With `return_index=True` also returns an integer array mapping each surviving
    event back to its position in the *time-sorted* input. Any per-event side
    data (ground-truth labels, annotation ids) should be carried through with
    that index rather than re-matched afterwards: re-matching on a packed
    (t, x, y) key silently mislabels events once the sensor is wider than the
    bits allotted to x, which is exactly what happened at 1280x720.
    """
    ev = ev.sorted_by_time()
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
