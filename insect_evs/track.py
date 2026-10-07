"""Linking per-window detections into tracks, and pulling each track's events
back out.

`link_tracks` joins detections from successive windows into tracks, one per
insect where linking succeeds. `Track.event_mask` then selects the raw events
that fell inside a track's detection boxes, which is what the wingbeat
measurement reads.

For a wingbeat measurement a track must be long enough to hold several wing
cycles: ten cycles at 100 Hz take 100 ms. Shorter tracks still link, but
their frequency estimates are loose. The periodicity gate sets its own
minimum cycle count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detect import Detection, merge_detections  # noqa: F401  (re-export)
from .events import EventStream


@dataclass
class Track:
    """One linked track: its detections in time order, plus a free-form
    `meta` dict for anything a caller wants to attach."""
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def t_start_us(self) -> int:
        """Start of the first detection's window, us."""
        return self.detections[0].t0_us

    @property
    def t_end_us(self) -> int:
        """End of the last detection's window, us."""
        return self.detections[-1].t1_us

    @property
    def duration_us(self) -> int:
        """Span from the first window's start to the last window's end.

        This includes one full window length: a track with a single detection
        already lasts `window_us`. So a `min_duration_us` equal to the window
        length rejects nothing, and only `min_detections` has an effect. Use
        `span_us` for the time actually covered by detections.
        """
        return self.t_end_us - self.t_start_us

    @property
    def span_us(self) -> int:
        """First to last detection centre, us. Zero for a single detection."""
        return int(self.detections[-1].t_us - self.detections[0].t_us)

    @property
    def duration_s(self) -> float:
        """duration_us in seconds."""
        return self.duration_us / 1e6

    @property
    def n_events(self) -> int:
        """Sum of the detections' event counts. Windows overlap, so an event
        can be counted more than once; event_mask counts each event once."""
        return int(sum(d.n_events for d in self.detections))

    def positions(self) -> np.ndarray:
        """(N, 3) array of t_us, x, y."""
        return np.array([[d.t_us, d.x, d.y] for d in self.detections], float)

    def mean_speed_px_s(self) -> float:
        """Mean of the speeds between successive detections, px/s. 0 for a
        track with fewer than two detections."""
        pos = self.positions()
        if len(pos) < 2:
            return 0.0
        dt = np.diff(pos[:, 0]) / 1e6
        d = np.hypot(np.diff(pos[:, 1]), np.diff(pos[:, 2]))
        good = dt > 0
        return float(np.mean(d[good] / dt[good])) if good.any() else 0.0

    def mean_size_px(self) -> float:
        """Mean of the detections' larger bounding-box side, px."""
        return float(np.mean([d.size_px for d in self.detections]))

    def event_mask(self, ev: EventStream, pad_px: int = 3) -> np.ndarray:
        """Boolean mask over `ev` selecting this track's events.

        An event is selected if it falls inside any detection's bounding box,
        grown by `pad_px` pixels on each side, during that detection's window.
        `ev` must be time-sorted.

        Windows overlap, so concatenating per-window slices would include some
        events twice. Those duplicates would add a false spectral line at the
        hop rate. A mask selects each event at most once.
        """
        mask = np.zeros(len(ev), bool)
        for d in self.detections:
            i0, i1 = np.searchsorted(ev.t, [d.t0_us, d.t1_us])
            if i1 <= i0:
                continue
            x0, y0, x1, y1 = d.bbox
            sx = ev.x[i0:i1]
            sy = ev.y[i0:i1]
            inside = (
                (sx >= x0 - pad_px) & (sx < x1 + pad_px)
                & (sy >= y0 - pad_px) & (sy < y1 + pad_px)
            )
            mask[i0:i1] |= inside
        return mask

    def extract_events(self, ev: EventStream, pad_px: int = 3) -> EventStream:
        """This track's events as a new stream; see event_mask."""
        return ev.select(self.event_mask(ev, pad_px))


def link_tracks(
    detections: List[Detection],
    gate_px: float = 45.0,
    coast_us: int = 10_000,
    max_missed_frames: Optional[int] = None,
    min_detections: int = 2,
    min_duration_us: int = 5_000,
) -> List[Track]:
    """Greedy nearest-neighbour linking with constant-velocity prediction.

    Detections are processed one window time at a time. Each active track
    predicts where its target is now from its last position and velocity.
    Every (track, detection) pair closer than `gate_px` to the prediction is a
    candidate, and pairs are accepted shortest distance first, each track and
    each detection used at most once. A detection left over starts a new track.

    Greedy matching is enough here: an optimal one-to-one assignment (the
    Hungarian method) tracks no more insects on test recordings. Most tracking
    errors come from the detector rather than the linker, so if tracks break
    up, tune the detector first.

    gate_px
        Largest distance in px between a track's predicted position and a
        detection it may claim.
    coast_us
        How long a track may go without a detection before it is retired, in
        microseconds. It must cover the longest gap you expect in an insect's
        detections, such as a brief pause or a faint stretch.
    max_missed_frames
        Older way to set coast_us, as a number of detector hops. When given, it
        overrides coast_us: the hop is taken as the median gap between
        successive detection times, and coast_us = max_missed_frames * hop.
        Prefer coast_us, which means the same thing in any scene.
    min_detections, min_duration_us
        Tracks with fewer detections, or a shorter `Track.duration_us`, are
        dropped from the output. duration_us includes one full window, so
        `min_duration_us` only has an effect when it is longer than the window.
    """
    if not detections:
        return []

    # Group detections by window centre. Each distinct time is one linking step.
    by_time: Dict[int, List[Detection]] = {}
    for d in detections:
        by_time.setdefault(d.t_us, []).append(d)
    frame_times = sorted(by_time)

    if max_missed_frames is not None:
        # Convert a count of hops to microseconds. The hop is estimated as the
        # median gap between window times that hold a detection; with a single
        # window time it defaults to 2,500 us. A track then survives a gap of
        # at most max_missed_frames hops, so max_missed_frames=0 links nothing.
        gaps = np.diff(frame_times) if len(frame_times) > 1 else np.array([2_500])
        hop = float(np.median(gaps)) if gaps.size else 2_500.0
        coast_us = int(round(max_missed_frames * hop))

    tracks: List[Track] = []
    active: List[Tuple[Track, int, np.ndarray]] = []  # track, last seen t_us, velocity px/us
    next_id = 0

    for fi, ft in enumerate(frame_times):
        # Retire tracks unseen for longer than coast_us before matching at this
        # time. Retiring them after matching would let a stale track claim the
        # next detection however long its gap had been.
        active = [a for a in active if ft - a[1] <= coast_us]
        dets = list(by_time[ft])
        assigned = set()

        # Predict each track's position at this time from its last detection
        # and velocity in px/us, and list every detection within gate_px of
        # the prediction. Then accept pairs shortest distance first.
        candidates = []
        for ai, (trk, last_t, vel) in enumerate(active):
            last = trk.detections[-1]
            dt = ft - last.t_us
            px = last.x + vel[0] * dt
            py = last.y + vel[1] * dt
            for di, d in enumerate(dets):
                dist = float(np.hypot(d.x - px, d.y - py))
                if dist <= gate_px:
                    candidates.append((dist, ai, di))
        candidates.sort()

        used_active = set()
        for dist, ai, di in candidates:
            if ai in used_active or di in assigned:
                continue
            trk, _, vel = active[ai]
            last = trk.detections[-1]
            d = dets[di]
            dt = max(d.t_us - last.t_us, 1)
            # Velocity is smoothed: half the previous estimate plus half the
            # latest step. A new track starts at zero velocity, so its first
            # prediction is its last position.
            new_vel = 0.5 * vel + 0.5 * np.array([(d.x - last.x) / dt, (d.y - last.y) / dt])
            trk.detections.append(d)
            active[ai] = (trk, ft, new_vel)
            used_active.add(ai)
            assigned.add(di)

        # Every detection no track claimed starts a new track.
        for di, d in enumerate(dets):
            if di in assigned:
                continue
            trk = Track(track_id=next_id, detections=[d])
            next_id += 1
            active.append((trk, ft, np.zeros(2)))
            tracks.append(trk)

    return [
        t for t in tracks
        if len(t.detections) >= min_detections and t.duration_us >= min_duration_us
    ]


def link_tracks_pda(
    detections: List[Detection],
    width: int = 1280,
    height: int = 720,
    params=None,
    min_detections: int = 5,
    min_duration_us: int = 100_000,
    confirmed_only: bool = True,
) -> List[Track]:
    """Link detections with an asynchronous probabilistic data association
    (PDA) filter, as an alternative to `link_tracks`.

    Not usable in this package as shipped: it imports a `pda` module that is
    not included, so calling it raises ImportError. It is kept so that code
    written against the full library still imports. Returns the same `Track`
    objects as `link_tracks`.
    """
    from .pda import CONFIRMED, AsyncPdaTracker, FiestaParams

    if not detections:
        return []

    order = sorted(range(len(detections)), key=lambda i: detections[i].t_us)
    tracker = AsyncPdaTracker(width, height, params or FiestaParams())

    # Pass all detections from one window to the filter together. Fed one at a
    # time, each would count as a miss for every other track, and tracks whose
    # target is still visible would be retired.
    batch_t, batch_z, batch_i = None, [], []
    for i in order:
        d = detections[i]
        if batch_t is not None and d.t_us != batch_t:
            tracker.step_batch(batch_t, batch_z, detection_indices=batch_i)
            batch_z, batch_i = [], []
        batch_t = d.t_us
        batch_z.append((d.x, d.y))
        batch_i.append(i)
    if batch_t is not None and batch_z:
        tracker.step_batch(batch_t, batch_z, detection_indices=batch_i)

    out: List[Track] = []
    for pt in tracker.finish():
        if confirmed_only and pt.status != CONFIRMED and pt.n_associations < tracker.params.a_confirm:
            continue
        dets = [detections[i] for i in pt.detection_indices]
        if len(dets) < min_detections:
            continue
        dets.sort(key=lambda d: d.t_us)
        trk = Track(track_id=pt.track_id, detections=dets)
        if trk.duration_us < min_duration_us:
            continue
        trk.meta.update({
            "tracker": "pda",
            "status": pt.status,
            "n_associations": pt.n_associations,
            "n_missed": pt.n_missed,
            "score": pt.score,
            "params": tracker.params.to_dict(),
        })
        out.append(trk)
    return out


def track_from_events(ev: EventStream, **kwargs) -> List[Track]:
    """Run detect_blobs and then link_tracks on one stream.

    Each keyword argument goes to detect_blobs if detect_blobs has a parameter
    of that name, and to link_tracks otherwise. The split is read from
    detect_blobs' own signature, so every detector parameter can be passed.
    """
    import inspect

    from .detect import detect_blobs

    det_keys = set(inspect.signature(detect_blobs).parameters) - {"ev"}
    det_kwargs = {k: v for k, v in kwargs.items() if k in det_keys}
    link_kwargs = {k: v for k, v in kwargs.items() if k not in det_keys}
    return link_tracks(detect_blobs(ev, **det_kwargs), **link_kwargs)
