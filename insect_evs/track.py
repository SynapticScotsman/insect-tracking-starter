"""Linking per-window detections into tracks, and pulling each track's events back out.

A track is only useful here if it is long enough to carry a spectrum. Ten
wingbeat cycles at 100 Hz is 100 ms, so `min_duration_us` defaults to that:
shorter tracks cannot support a frequency estimate tight enough to separate one
individual from another, whatever the rest of the pipeline does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detect import Detection, merge_detections  # noqa: F401  (re-export)
from .events import EventStream


@dataclass
class Track:
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def t_start_us(self) -> int:
        return self.detections[0].t0_us

    @property
    def t_end_us(self) -> int:
        return self.detections[-1].t1_us

    @property
    def duration_us(self) -> int:
        """Span from the first window's start to the last window's end.

        NOTE this is inflated by one `window_us`: a track with a single detection
        already "lasts" the whole window. A `min_duration_us` equal to the window
        length is therefore vacuous and only `min_detections` binds -- which is
        how a reference set of 609 tracks was once reported as 526. Use
        `span_us` when you mean the time actually covered by detections.
        """
        return self.t_end_us - self.t_start_us

    @property
    def span_us(self) -> int:
        """First to last detection CENTRE -- zero for a single detection."""
        return int(self.detections[-1].t_us - self.detections[0].t_us)

    @property
    def duration_s(self) -> float:
        return self.duration_us / 1e6

    @property
    def n_events(self) -> int:
        return int(sum(d.n_events for d in self.detections))

    def positions(self) -> np.ndarray:
        """(N, 3) array of t_us, x, y."""
        return np.array([[d.t_us, d.x, d.y] for d in self.detections], float)

    def mean_speed_px_s(self) -> float:
        pos = self.positions()
        if len(pos) < 2:
            return 0.0
        dt = np.diff(pos[:, 0]) / 1e6
        d = np.hypot(np.diff(pos[:, 1]), np.diff(pos[:, 2]))
        good = dt > 0
        return float(np.mean(d[good] / dt[good])) if good.any() else 0.0

    def mean_size_px(self) -> float:
        return float(np.mean([d.size_px for d in self.detections]))

    def event_mask(self, ev: EventStream, pad_px: int = 3) -> np.ndarray:
        """Boolean mask over `ev` selecting this track's events.

        Windows overlap, so a mask (rather than concatenating per-window slices)
        is what keeps each event counted exactly once. Double-counted events
        would put a spurious line at the hop rate into every periodogram.
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

    Greedy is the right call, though not for the reason previously given here.
    The old docstring claimed the detections that compete for a link are usually
    clutter blobs. Measured: 797 of 798 frames contain a contested detection and
    there are 1,456 thefts of a target's own detection -- but only 13% are by a
    track on a DIFFERENT object. The other 87% is one animal reported twice by
    the detector, which `detect_blobs(nms_dist_px=...)` now suppresses at source.
    Hungarian assignment was measured against greedy and bought +0 tracked
    targets, so the conclusion survives its broken justification.

    Nothing else in here is worth replacing either: an oracle that assigns every
    detection to its true target and never errs reaches 443 of 609 reference
    bees against greedy's 438. The linker costs 5; the detector costs 166. Tune
    the detector.

    coast_us
        How long a track may go unmatched before it is retired, IN MICROSECONDS.
        This replaces `max_missed_frames`, which counted entries in the list of
        instants where some detection existed anywhere in frame -- so its meaning
        moved with scene density. On a full 1280x720 sensor every 2.5 ms slot is
        occupied and 4 frames meant exactly 10 ms; on a 160x90 crop the same
        setting meant 10 ms median but 112 ms at worst, and 402 ms at 80x45. A
        40x spread in one number. Passing `max_missed_frames` still works and is
        interpreted as that many hops of the detector's own grid.
    """
    if not detections:
        return []

    by_time: Dict[int, List[Detection]] = {}
    for d in detections:
        by_time.setdefault(d.t_us, []).append(d)
    frame_times = sorted(by_time)

    if max_missed_frames is not None:
        # Backwards compatibility, with a MEASURED caveat (2026-08-10 review,
        # reproduced against the be291ab original): this shim is one hop
        # TIGHTER than the historical code. be291ab pruned AFTER matching, so
        # max_missed_frames=N allowed a re-match at an effective gap of
        # (N+1)*hop; this code prunes before matching with coast = N*hop, so
        # mmf=N here behaves like the old mmf=N-1, and mmf=0 links nothing
        # where the old code still linked consecutive frames. The conversion is
        # NOT changed to (N+1)*hop, deliberately: the frozen protocol pins
        # mmf=4 == coast_us=10_000 under CURRENT semantics
        # (analysis/verify_refactor.py), wf6_stack.py reproduces the published
        # row with mmf=4 as it stands, and moving every stale caller by one hop
        # would silently shift all of that. New code should pass coast_us.
        gaps = np.diff(frame_times) if len(frame_times) > 1 else np.array([2_500])
        hop = float(np.median(gaps)) if gaps.size else 2_500.0
        coast_us = int(round(max_missed_frames * hop))

    tracks: List[Track] = []
    active: List[Tuple[Track, int, np.ndarray]] = []  # track, last seen t_us, velocity px/us
    next_id = 0

    for fi, ft in enumerate(frame_times):
        # Expire by wall-clock time before matching at this timestamp. Doing it
        # afterwards lets a stale track claim the first later detection no matter
        # how long the silent gap was; inserting an unrelated occupied timestamp
        # would then change the result by retiring that same track sooner.
        active = [a for a in active if ft - a[1] <= coast_us]
        dets = list(by_time[ft])
        assigned = set()

        # Prediction, then greedy match by ascending distance.
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
            new_vel = 0.5 * vel + 0.5 * np.array([(d.x - last.x) / dt, (d.y - last.y) / dt])
            trk.detections.append(d)
            active[ai] = (trk, ft, new_vel)
            used_active.add(ai)
            assigned.add(di)

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
    """Link detections with the FIESTA asynchronous PDA filter.

    A drop-in alternative to `link_tracks`, returning the same `Track` objects so
    nothing downstream changes. Greedy linking stays the default and the
    baseline: it is cheaper, and on sparse scenes the two often agree, so the
    PDA tracker has to earn its place by measurably reducing fragmentation.

    The filter carries a covariance, so it can coast through the gaps where an
    insect turns, is briefly occluded, or stops emitting events. That is the
    property greedy linking lacks and the reason long trajectories fragment.
    """
    from .pda import CONFIRMED, AsyncPdaTracker, FiestaParams

    if not detections:
        return []

    order = sorted(range(len(detections)), key=lambda i: detections[i].t_us)
    tracker = AsyncPdaTracker(width, height, params or FiestaParams())

    # Group by timestamp: a windowed detector emits several detections at the
    # same instant, and charging every track a miss for each of them kills
    # tracks whose own target is still visible.
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
    """Convenience: detect then link, with detection kwargs passed through.

    The split is derived from detect_blobs' own signature rather than a
    hand-maintained set: the old literal set was six parameters stale
    (min_events_blob, min_events_frame, nms_dist_px, grid_origin_us,
    coincidence, backend), so the frozen reference recipe raised TypeError
    through this wrapper, and the only way to make it run was to drop
    nms_dist_px=0 and silently inherit 25 px NMS -- the documented
    8%-too-small-denominator mistake. Found by the 2026-08-10 review.
    """
    import inspect

    from .detect import detect_blobs

    det_keys = set(inspect.signature(detect_blobs).parameters) - {"ev"}
    det_kwargs = {k: v for k, v in kwargs.items() if k in det_keys}
    link_kwargs = {k: v for k, v in kwargs.items() if k not in det_keys}
    return link_tracks(detect_blobs(ev, **det_kwargs), **link_kwargs)
