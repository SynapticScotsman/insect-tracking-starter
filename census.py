"""The census: find every flying insect in one window of events and measure its wingbeat.

Start with run_census. It runs these steps in order:

1. Background-activity filter (BAF). An event survives only if a neighbouring
   pixel fired within baf_dt_us. This removes isolated sensor noise.
2. Blob detection on the surviving events, in short time windows.
3. Tracking: detections are linked into tracks, one per insect passage.
4. Wingbeat candidates per track, from the event-rate spectrum and from YIN,
   a time-domain pitch estimator.
5. The octave test (pose_autocorr, pick_fundamental). The rate spectrum often
   peaks at 2 or 3 times the true wingbeat. The test checks which candidate
   period the track's spatial event pattern actually repeats at.
6. A lamp test per track (lamp_lock). Mains-powered lights flicker at 100 Hz
   and modulate the events of every insect they light.
7. A verdict per track: "wingbeat" or "no line".

stroke_times and running_stroke_rate then time individual wing strokes, and
census_figure draws a one-page summary.

Times are in microseconds (us) unless a name says otherwise. Positions are in
sensor pixels (px). Frequencies are in Hz.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from insect_evs.gate import GateConfig, PeriodicityGate
from insect_evs.denoise import background_activity_mask
from insect_evs.detect import detect_blobs, detect_blobs_multiscale
from insect_evs.events import EventStream
from insect_evs.periodicity import (
    _detrend_highpass, _fine_peak, _line_snr_db, _psd,
    half_cycle_similarity, rate_signals,
)
from insect_evs.track import link_tracks

# Figure colours: background, text, dim text, grid lines.
BG, INK, DIM, GRID = "#05070A", "#FFFFFF", "#8A97A0", "#1C242C"


# Figure colours: insects with a wingbeat, the lamp, a reference, and tracks
# without a measured wingbeat.
TARGET, CLUTTER, REF, DROPPED = "#FF3DA6", "#FFB000", "#7CE0C0", "#39424A"


#: Verdict names and the colour each is drawn in.
#: "wingbeat" = the track has a clear spectral line, lasts enough wing strokes
#:              to count, and the octave test settled its fundamental.
#: "no line"  = no wingbeat was settled. Most such tracks are too short.
#: There is no "lamp" verdict, on purpose. A flickering lamp modulates every
#: insect it lights, so any insect whose own wing signal is too weak to
#: measure shows the lamp as its strongest rhythm. That track is still an
#: insect, just one without a measured wingbeat, which is what "no line" says.
#: The `lamp_dominant` column records that the lamp line is strongest without
#: claiming what the object is.
VERDICT_COLOUR = {"wingbeat": TARGET, "no line": DROPPED}


# Bin width of the event-rate signals, 200 us, which samples at 5 kHz. This is
# also the default of periodicity.rate_signals.
BIN_US = 200


# Detector hop, 2.5 ms: the step between consecutive detection windows.
HOP_US = 2_500


@dataclass
class CensusConfig:
    """Settings for run_census.

    wingbeat_band defaults to moths near 40 Hz. The detector window defaults
    to bees near 230 Hz. Read the notes on window_us and wingbeat_band before
    running on a different insect.
    """
    # Frequency range searched by the periodicity features, Hz.
    fmin_hz: float = 15.0
    fmax_hz: float = 500.0
    baf_dt_us: int = 1_000          # BAF coincidence window, us; 0 disables the filter
    coast_us: int = 50_000          # a track may go this long without a detection
    gate_px: float = 45.0           # max jump from a track's prediction to a detection
    max_size_px: int = 90           # largest blob side, px; check `size_px` percentiles
    min_snr_db: float = 10.0        # a spectral line must stand this far over the floor
    min_cycles: float = 10.0        # strokes to span: 43 ms at 233 Hz, 250 ms at 40 Hz
    pad_px: int = 3                 # detection boxes grow by this when collecting events
    # Detector window. 5 ms with a 2.5 ms hop suits a 230 Hz bee: 5 ms is 1.15
    # strokes, so every blob holds a whole stroke and its centroid is the body.
    # For a 40 Hz moth 5 ms is a fifth of a stroke, so a blob can be one wing
    # and the centroid swings with that wing. Scale the window with the
    # insect's stroke period.
    window_us: int = 5_000
    hop_us: int = 2_500
    detector: str = "single"        # or "multiscale": a bank of 5-80 ms windows
    # Same-frame non-maximum suppression distance, px. 0 turns it off. It
    # cannot tell one animal detected twice from two animals passing close
    # together, so with it on, real neighbours can merge into one track.
    nms_px: float = 0.0
    # Wingbeat candidates are kept only inside this band, Hz. 18-80 Hz is set
    # for moths. It excludes a 100 Hz lamp line and the 15-17 Hz values YIN
    # returns when it locks onto a subharmonic. For honey bees use about
    # 120-320 Hz.
    wingbeat_band: tuple = (18.0, 80.0)
    # Octave test resolution: events are counted in 0.5 ms time bins, on a
    # 6 x 6 grid of cells around the insect, separately for ON and OFF.
    pose_bin_us: int = 500
    pose_cells: int = 6
    # Once the octave test has chosen the period, any other estimate within
    # this fraction of it measures the SAME period. The reported number then
    # comes from the most precise of them (see same_period_value). 0 reports
    # the chosen candidate as it is.
    same_period_tol: float = 0.10


@dataclass
class Census:
    """Everything one call of run_census produced, for one window of events."""
    raw_path: str                   # where the events came from; a label only
    t0_us: int                      # window start, us
    t1_us: int                      # window end, us, one past the last event
    config: CensusConfig
    ev: EventStream                 # the raw window, time-sorted
    keep: np.ndarray                # BAF keep mask over `ev`
    grid_origin_us: int             # start of the detector's window grid
    detections: list
    tracks: list
    table: "object"                 # pandas DataFrame, one row per track
    # Scene-wide line strengths from scene_lines, plus "lamp_hz".
    scene_lines: Dict[str, float] = field(default_factory=dict)
    # track_id -> (t_us, f_hz): the wingbeat over the track's life, for every
    # track with verdict "wingbeat" (see wingbeat_trace).
    traces: Dict[int, tuple] = field(default_factory=dict)

    @property
    def ev_dn(self) -> EventStream:
        """The events that survived the BAF."""
        return self.ev.select(self.keep)

    def track(self, track_id: int):
        """The track with this id. Raises KeyError if there is none."""
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        raise KeyError(track_id)

    def heroes(self, n: int = 5, min_events: int = 5_000, min_cycles: float = 30.0):
        """The n wingbeat tracks best suited to a close-up view of the wing.

        Ranked by the largest box side times the line SNR, with SNR capped at
        25 dB. The cap stops a very clean but tiny target outranking a large
        one whose wing can be resolved.

        min_cycles: a close-up that folds events over the wing stroke needs
        many strokes, so a short track is excluded however large it is. A
        127 ms track at 42 Hz spans only 5 strokes."""
        df = self.table
        cyc = df.f0_wingbeat_hz * df.duration_ms / 1e3
        cand = df[(df.verdict == "wingbeat") & (df.n_events_raw >= min_events)
                  & np.isfinite(df.f0_wingbeat_hz) & (cyc >= min_cycles)]
        cand = cand.assign(hero=cand.max_side_px * cand.snr_signed_db.clip(upper=25))
        return cand.sort_values("hero", ascending=False).head(n)


def scene_lines(ev: EventStream, t0: int, t1: int) -> Dict[str, float]:
    """Line SNR at 100, 200 and 300 Hz of the event rate summed over the whole scene.

    Returned in dB over the spectrum's median level, for the signed rate
    (ON minus OFF) and the unsigned rate (ON plus OFF).

    A lamp flickers on every animal in phase, so its line adds up across the
    scene. Independent wingbeats have different rates and phases and smear
    out. A strong line here therefore points to a flickering light in view.
    """
    fs = 1e6 / BIN_US
    _, u, s = rate_signals(ev, bin_us=BIN_US, t0_us=t0, t1_us=t1)
    out = {}
    for name, sig in (("signed", s), ("unsigned", u)):
        # Remove drift below 5 Hz, then take the spectrum at 0.5 Hz resolution.
        f, P, _ = _psd(_detrend_highpass(sig, fs, 5.0), fs, df_target_hz=0.5)
        band = (f >= 20) & (f <= 1000)
        for line in (100.0, 200.0, 300.0):
            out["{}_{:.0f}Hz_db".format(name, line)] = _line_snr_db(
                f, P, line, exclude_hz=1.5, band=band)
    return out


def lamp_frequency(ev: EventStream, t0: int, t1: int) -> float:
    """The scene's lamp frequency in Hz: the refined peak of the scene-summed
    signed rate between 95 and 105 Hz.

    It is measured rather than assumed to be 100.000 Hz because the mains
    frequency drifts, and a lamp test at the wrong frequency loses phase over
    a few seconds: 0.02 Hz off over 4 s is 0.08 of a cycle. Returns 100.0 if
    no peak is found."""
    fs = 1e6 / BIN_US
    _, _, s = rate_signals(ev, bin_us=BIN_US, t0_us=t0, t1_us=t1)
    f, _ = _fine_peak(_detrend_highpass(s, fs, 5.0), fs, 95.0, 105.0)
    return float(f) if np.isfinite(f) else 100.0


def track_path(trk, smooth_ms: float = 25.0, step_us: int = 1_000):
    """The track's centre on a 1 ms grid, smoothed by a moving average over `smooth_ms`.

    When the detector window is shorter than a wing stroke, detection centres
    swing with the wing. Averaging over 25 ms, one stroke at 40 Hz, removes
    that swing and leaves the body's path. Returns (t_us, x, y)."""
    from scipy.ndimage import uniform_filter1d

    pos = trk.positions()
    t = np.arange(pos[0, 0], pos[-1, 0] + 1, step_us)
    x = np.interp(t, pos[:, 0], pos[:, 1])
    y = np.interp(t, pos[:, 0], pos[:, 2])
    w = max(int(round(smooth_ms * 1e3 / step_us)), 1)
    return t, uniform_filter1d(x, w, mode="nearest"), uniform_filter1d(y, w, mode="nearest")


def pose_autocorr(sub: EventStream, trk, cfg: CensusConfig, slow_us: float) -> np.ndarray:
    """How strongly the spatial pattern of a track's events repeats at each time lag.

    Events are placed on a pose_cells x pose_cells grid that moves with the
    smoothed track centre, one grid for ON and one for OFF, and counted in
    time bins of pose_bin_us. Each time bin is then a pattern vector, and
    r[k] is the correlation between patterns k bins apart.

    Why a spatial pattern and not an event rate: on many tracks ON and OFF
    bursts arrive twice per wing stroke, in both polarities. Every rate-based
    count then reads double the wingbeat. Only the position of the bursts
    alternates between the two half strokes, so a pattern that sees where
    events land repeats at the true period. On such tracks r was about 0.28
    at the true period and 0.03 at half of it.

    Why the slow trend is removed: without it the static body shape is in
    every bin, r is high at all lags, and the choice between T and 3T is
    decided by noise. Subtracting a moving average over `slow_us` (8 x the
    longest candidate period) leaves only what changes within a stroke.

    Returns r[k] at lag k * pose_bin_us.
    """
    from insect_evs.descriptor import descriptor_autocorr
    from scipy.ndimage import uniform_filter1d

    G, bus = cfg.pose_cells, cfg.pose_bin_us
    # Grid side, px: the median detection size plus a 6 px margin, at least 10 px.
    side = max(float(np.median([d.size_px for d in trk.detections])), 4.0) + 6
    # Each event's cell on the grid centred on the smoothed track position.
    tp, xp, yp = track_path(trk)
    cx = np.interp(sub.t, tp, xp)
    cy = np.interp(sub.t, tp, yp)
    gx = np.clip(((sub.x - cx) / side + 0.5) * G, 0, G - 1e-6).astype(np.int64)
    gy = np.clip(((sub.y - cy) / side + 0.5) * G, 0, G - 1e-6).astype(np.int64)
    # D[time bin, polarity x cell]: event counts, lightly smoothed over 3 bins.
    b = ((sub.t - sub.t[0]) // bus).astype(np.int64)
    D = np.zeros((int(b.max()) + 1, 2 * G * G))
    np.add.at(D, (b, (sub.p > 0).astype(np.int64) * G * G + gy * G + gx), 1.0)
    D = uniform_filter1d(D, 3, axis=0, mode="nearest")
    # Subtract the slow trend (at least 5 bins wide) so only within-stroke change remains.
    w = max(int(round(slow_us / bus)), 5)
    D = D - uniform_filter1d(D, w, axis=0, mode="nearest")
    return descriptor_autocorr(D)


def pick_fundamental(r: np.ndarray, cands: List[float], bin_us: int, tie: float = 0.05,
                     return_tied: bool = False):
    """Choose the wingbeat among candidate frequencies using the pose autocorrelation r.

    Each candidate is scored by r at its period. The answer is the SHORTEST
    period whose score is within `tie` of the best score. A true period T
    also repeats at 2T and 3T, so a longer candidate can match T's score by
    noise. Preferring the shortest one inside the tie stops a multiple of
    the period winning that way. 0.05 is well below the typical gap between
    a true period and a wrong one, about 0.17 to 0.26.

    Returns (f_hz, r at that period, r at 1.37 x that period). The 1.37 x lag
    is not a multiple of any candidate, so it gives the background level of
    r. Lags past 60% of the series are not scored because too few pairs of
    bins remain. Returns NaNs if no candidate can be scored.

    return_tied=True also returns the frequencies of every candidate inside
    the tie, which same_period_value needs."""
    n = len(r)
    scored = []
    for f in cands:
        # Candidate period in pose bins.
        lag = 1e6 / f / bin_us
        if not np.isfinite(lag) or lag * 1.37 >= 0.6 * n:
            continue
        scored.append((f, float(np.interp(lag, np.arange(n), r)),
                       float(np.interp(lag * 1.37, np.arange(n), r))))
    if not scored:
        return (np.nan, np.nan, np.nan, []) if return_tied else (np.nan, np.nan, np.nan)
    top = max(s[1] for s in scored)
    near = [s for s in scored if s[1] >= top - tie]
    best = max(near, key=lambda s: s[0])     # highest frequency = shortest period
    return best + ([s[0] for s in near],) if return_tied else best


def same_period_value(f_pick: float, tied: List[float], f_signed: float, f_yin: float,
                      f_lamp: float, tol: float = 0.10) -> float:
    """The wingbeat number to report, once pick_fundamental has chosen the period.

    The signed-rate line and YIN often name the same period a few percent
    apart, and the pose test cannot separate them, so both are in the tie.
    pick_fundamental then returns the higher of the two. That is the max of
    two estimates of one number, so it is biased upward.

    This function fixes the number, not the period. Among the TIED
    candidates within `tol` of the pick, it reports the signed line (or its
    half or third) first, then YIN, then the pick itself. The signed line
    comes first because _fine_peak refines it over the whole track
    (8 x zero-padded spectrum, parabolic vertex). A candidate outside the tie
    is never used, because there the pose test did prefer one period.

    A signed line that sits on the lamp (within 1 Hz of 1, 2 or 3 x f_lamp)
    ranks last, because it measures the lamp. Its third, 33.3 Hz for a
    100 Hz lamp, is not a wingbeat even when an insect's period is close.

    tol = 0.10 is far below the 33% that separates neighbouring octave
    candidates (f/2 against f/3). tol = 0 returns the pick unchanged.

    On synthetic insects of known rate this rule cut the median error from
    0.33% to 0.02% on moths and from 0.17% to 0.00% on bees, and changed no
    octave choice.
    """
    if not (np.isfinite(f_pick) and tol > 0):
        return f_pick
    on_lamp = np.isfinite(f_signed) and any(abs(f_signed - m * f_lamp) <= 1.0 for m in (1, 2, 3))

    def rank(f):
        # Lower rank wins: 0 = signed line or its half or third, 1 = YIN,
        # 2 = anything else or a signed line on the lamp. Candidates are
        # rounded to 0.001 Hz in run_census, hence the 1e-3 match.
        if np.isfinite(f_signed) and any(abs(f - f_signed / k) < 1e-3 for k in (1, 2, 3)):
            return 2 if on_lamp else 0
        return 1 if (np.isfinite(f_yin) and abs(f - f_yin) < 1e-3) else 2

    # Tied candidates within tol of the pick, on a log scale; ties broken
    # toward the higher frequency.
    same = [f for f in tied if abs(np.log(f / f_pick)) < np.log1p(tol)] or [f_pick]
    return min(same, key=lambda f: (rank(f), -f))


def wingbeat_trace(sub: EventStream, f0_hz: float, smooth_ms: float = 50.0,
                   edge_ms: float = 50.0, lowpass_hz: float = 8.0):
    """The wingbeat over a track's life, as the slope of periodicity.phase_track.

    A moth's wingbeat is not one number: within 100 ms it can range over a
    quarter of its value. phase_track demodulates the events at the track's
    settled f0 and follows the rate within +/- lowpass_hz of it, so the
    octave choice is kept and not re-decided at every instant. The slope is
    averaged over `smooth_ms` (two strokes at 40 Hz). `edge_ms` is cut from
    each end, where the zero-phase filter has data on one side only.

    Limits: it follows slow drift well but cannot show stroke-to-stroke
    change. For a bee whose rate swings by +/-15%, the swing leaves the
    +/- 8 Hz band and the trace misses most of it. Use stroke_times and
    running_stroke_rate for those.

    Returns (t_us, f_hz) on a 1 ms grid, or None for a track too short to
    trim and still hold a smoothing window.
    """
    from scipy.ndimage import uniform_filter1d
    from insect_evs.periodicity import phase_track

    if len(sub) < 200 or not np.isfinite(f0_hz):
        return None
    t_us, cyc = phase_track(sub, f0_hz, lowpass_hz=lowpass_hz)
    if len(t_us) < 4:
        return None
    # Accumulated phase in cycles on a 1 ms grid; its time derivative is the
    # instantaneous frequency in Hz.
    g = np.arange(t_us[0], t_us[-1], 1_000.0)
    c = np.interp(g, t_us, cyc)
    f = np.gradient(c, g / 1e6)
    f = uniform_filter1d(f, max(int(smooth_ms), 1), mode="nearest")
    e = int(edge_ms)
    if len(g) <= 2 * e + int(smooth_ms):
        return None
    return g[e:-e], f[e:-e]


def stroke_times(sub: EventStream, trk, f0_hz: float, bin_us: int = 500,
                 subbin: bool = True, band: tuple = (0.5, 1.5),
                 min_gap: float = 0.75) -> np.ndarray:
    """Times (us) of individual wing strokes, found directly in the events.

    Method: count ON events per 0.5 ms in the track's boxes. A count does not
    depend on where the insect is, so its motion does not enter. Then remove
    the slow trend over two wing periods, band-pass to `band` x f0, and take
    peaks at least `min_gap` / f0 apart. Each peak is one stroke.

    Why the band stops at 1.5 x f0: with the upper edge at 2 x f0 the second
    harmonic sits at the band edge and puts a second peak into some strokes.
    On synthetic bees of known rate that counted 14% of strokes twice; with
    the edge at 1.5 x f0 it counted none twice, and the median per-stroke
    error fell from 2.05% to 1.50%.

    Why min_gap = 0.75: on real bee recordings it cut intervals reading over
    1.4 x or under 0.7 x the bee's own wingbeat from 11.7% to 7.1%. A honey
    bee varies by only about 1 Hz within itself, so those intervals are
    counting errors. On moths it has a cost: it turns strokes counted twice
    into skipped strokes (about 11% skipped).

    What it is NOT: checked against an independent stroke clock. A clock
    built from the event centroid disagrees with it stroke by stroke, so
    1 / interval is this clock's rate, not established truth. It exists
    because wingbeat_trace cannot show change from one stroke to the next.
    """
    import scipy.signal as sps
    from scipy.ndimage import uniform_filter1d

    if len(sub) < 200 or not np.isfinite(f0_hz):
        return np.zeros(0)
    t0 = int(sub.t[0])
    # ON-event count per bin_us bin.
    n = int((sub.t[-1] - t0) // bin_us) + 1
    b = ((sub.t - t0) // bin_us).astype(np.int64)
    on = np.bincount(b, weights=(sub.p > 0).astype(float), minlength=n)
    fs = 1e6 / bin_us
    # Remove the trend over two wing periods, then band-pass with a 2nd-order
    # Butterworth filter. The upper edge is capped below the Nyquist frequency.
    s = on - uniform_filter1d(on, max(int(2 * fs / f0_hz), 3), mode="nearest")
    sos = sps.butter(2, [band[0] * f0_hz / (fs / 2), min(band[1] * f0_hz / (fs / 2), 0.95)],
                     btype="band", output="sos")
    if n < 30:
        return np.zeros(0)
    s = sps.sosfiltfilt(sos, s)
    pk, _ = sps.find_peaks(s, distance=max(int(min_gap * fs / f0_hz), 1))
    pos = pk.astype(float)
    if subbin:
        # Place each peak between bins at the vertex of the parabola through
        # the peak bin and its two neighbours. Without this every stroke time
        # sits on a 0.5 ms bin centre, so a 230 Hz bee's 4.35 ms stroke reads
        # as 4.0 or 4.5 ms, which is 250 or 222 Hz. The band-passed signal is
        # smooth at the bin scale (a 40 Hz stroke spans 50 bins, a 230 Hz one
        # 8.7), so the vertex is a good estimate.
        inner = (pk > 0) & (pk < len(s) - 1)
        k = pk[inner]
        a, b, c = s[k - 1], s[k], s[k + 1]
        den = a - 2.0 * b + c
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
        pos[inner] += np.clip(d, -0.5, 0.5)
    # Bin index to time at the bin centre, us.
    return t0 + (pos + 0.5) * bin_us


def running_stroke_rate(st: np.ndarray, k: int = 5) -> np.ndarray:
    """A live wingbeat label: the median rate of the last k strokes, Hz.

    st holds stroke times in us, as from stroke_times. The result has one
    value per stroke from the second on, aligned with st[1:]. Each value is
    the median of 1 / interval over the last min(k, available) intervals,
    so the label at a stroke uses only strokes already seen. k = 1 gives
    1 / the last interval. A median of k ignores up to (k - 1) / 2 bad
    intervals (a stroke counted twice or skipped) and costs about k strokes
    of time resolution.

    Why k = 5:
    - On real recordings, labels more than 1.4 x or less than 0.7 x the
      track's own wingbeat fell from 10.7% to 3.3% on moths and from 7.1%
      to 0.9% on bees, against k = 1.
    - On synthetic insects whose rate swings by +/-15%, the label shows this
      share of the swing: moths 0.95 at 2 Hz, 0.70 at 4 Hz, 0.32 at 8 Hz,
      about 55-60 ms late; bees 0.98 at 5 Hz, 0.94 at 10 Hz, 0.79 at 20 Hz,
      11 ms late.
    - Errors that come in runs of several strokes are not removed: a median
      of 5 cannot outvote them.
    So on moths the label shows drift slower than about 4 Hz and hides faster
    swings. For moths, wingbeat_trace follows faster swings better (0.83 at
    4 Hz, no delay); for bees it does not, and this label is the better one."""
    st = np.asarray(st, float)
    if len(st) < 2:
        return np.zeros(0)
    rate = 1e6 / np.diff(st)
    return np.array([np.median(rate[max(0, i - k + 1):i + 1]) for i in range(len(rate))])


def lamp_lock(sub: EventStream, f_lamp: float):
    """How strongly a track's events flicker with the lamp.

    Polarity-weighted vector strength at the lamp frequency, computed in
    absolute time so every track shares one phase reference:
    R = |sum p exp(-2 pi i f t)| / sqrt(n). With independent events R^2 is
    exponentially distributed, so R > 2.63 means p < 0.001.

    Returns (R, phase_rad, depth), where depth = |sum| / n is the modulation
    depth. Returns NaNs for fewer than 50 events."""
    if len(sub) < 50:
        return np.nan, np.nan, np.nan
    z = np.sum(sub.p * np.exp(-2j * np.pi * f_lamp * (sub.t * 1e-6)))
    return float(abs(z) / np.sqrt(len(sub))), float(np.angle(z)), float(abs(z) / len(sub))


def _verdict(row, cfg: CensusConfig, f_lamp: float) -> str:
    """Return "wingbeat" if the track has a countable line, a settled fundamental
    that is not the lamp, and spans at least min_cycles strokes; else "no line"."""
    countable = (row["snr_signed_db"] >= cfg.min_snr_db
                 and np.isfinite(row["f0_signed_hz"]))
    fw = row["f0_wingbeat_hz"]
    # A settled fundamental AT the lamp frequency or its harmonics is the
    # lamp's pattern repeating, not a wing. Half the lamp frequency is not
    # excluded, because moths can beat near 50 Hz. This only matters when
    # wingbeat_band admits 100 Hz or more.
    if countable and np.isfinite(fw) and not any(abs(fw - k * f_lamp) <= 1.0 for k in (1, 2, 3)) \
            and fw * row["duration_ms"] / 1e3 >= cfg.min_cycles:
        return "wingbeat"
    return "no line"


def _lamp_dominant(row, f_lamp: float) -> bool:
    """The track's strongest signed line is the lamp and no wingbeat settled.
    A property of the track's events, not a claim about what the object is."""
    return bool(row["verdict"] != "wingbeat" and np.isfinite(row["f0_signed_hz"])
                and abs(row["f0_signed_hz"] - f_lamp) <= 1.0)


def run_census(ev: EventStream, cfg: Optional[CensusConfig] = None,
               raw_path: str = "", verbose: bool = True) -> Census:
    """Run the whole census on one window of events and return a Census.

    Steps: BAF, detection, tracking, periodicity features per track, the
    octave choice per track by pose_autocorr, the lamp lock, and a verdict
    per track. `raw_path` is only recorded in the result. `verbose` prints a
    summary after each stage.

    Main columns of the returned table (one row per track):
    f0_signed_hz    strongest line of the signed event rate. Can be 2 or 3 x
                    the wingbeat.
    f0_yin_hz       YIN estimate of the same rate.
    f0_wingbeat_hz  the wingbeat after the octave test. NaN when no period
                    settled.
    verdict         "wingbeat" or "no line" (see VERDICT_COLOUR).
    lamp_R          lock to the lamp line; above 2.63 is p < 0.001.
    """
    import pandas as pd

    cfg = cfg or CensusConfig()
    # The BAF mask and the time slicing below both assume time order.
    # t1 is one microsecond past the last event, so [t0, t1) holds every event.
    ev = ev.sorted_by_time()
    t0, t1 = int(ev.t[0]), int(ev.t[-1]) + 1
    # BAF keep-mask over the raw window: an event survives if a neighbouring
    # pixel fired within baf_dt_us microseconds. baf_dt_us = 0 keeps everything.
    keep = (background_activity_mask(ev, dt_us=cfg.baf_dt_us, radius=1)
            if cfg.baf_dt_us > 0 else np.ones(len(ev), bool))
    ev_dn = ev.select(keep)
    # Scene lines are measured on the raw window, not on the BAF survivors.
    lines = scene_lines(ev, t0, t1)
    if verbose:
        print("window {:.3f}-{:.3f} s: {:,} events ({:.2f} Mev/s), BAF keeps {:.1%}".format(
            t0 / 1e6, t1 / 1e6, len(ev), len(ev) / ((t1 - t0) / 1e6) / 1e6, keep.mean()))
        print("scene lines (dB over floor): " + ", ".join(
            "{} {:.1f}".format(k.replace("_db", ""), v) for k, v in lines.items()))

    # One window grid anchored at floor(t0 / hop), stored in the Census, so a
    # rerun of the same window lands on the same window phase. Detection runs
    # on the BAF survivors. A blob needs at least 10 events and 3 px.
    grid = (t0 // cfg.hop_us) * cfg.hop_us
    if cfg.detector == "multiscale":
        dets = detect_blobs_multiscale(ev_dn, grid_origin_us=grid, blur_sigma=1.0,
                                       nms_dist_px=cfg.nms_px)
    else:
        dets = detect_blobs(ev_dn, window_us=cfg.window_us, hop_us=cfg.hop_us,
                            blur_sigma=1.0, pixel_threshold=0.5, min_events_blob=10,
                            min_area_px=3, max_size_px=cfg.max_size_px,
                            nms_dist_px=cfg.nms_px, grid_origin_us=grid)
    # A track needs at least 2 detections and must last at least 5 ms.
    tracks = link_tracks(dets, gate_px=cfg.gate_px, coast_us=cfg.coast_us,
                         min_detections=2, min_duration_us=5_000)
    if verbose:
        # Many detections at the size cap means max_size_px is too small for
        # these insects.
        sz = np.array([d.size_px for d in dets]) if dets else np.zeros(1)
        print("{:,} detections (size px p50 {:.0f}, p90 {:.0f}, p99 {:.0f}; "
              "{:.1%} at the {} px cap) -> {:,} tracks".format(
                  len(dets), *np.percentile(sz, [50, 90, 99]),
                  float(np.mean(sz >= cfg.max_size_px)), cfg.max_size_px, len(tracks)))

    # The library's periodicity gate computes the per-track features: the
    # signed and unsigned spectral peaks, YIN, line SNR and others. Only those
    # features are used here; its accept/reject decision is not. Its two
    # rejection thresholds are switched off: phase locking (min_plv) rejects
    # most real insects, and the harmonic ratio cannot tell a wing from a
    # flickering light. The scene-coherence check is also off; lamp_lock
    # below tests for the lamp directly.
    gate = PeriodicityGate(GateConfig(fmin_hz=cfg.fmin_hz, fmax_hz=cfg.fmax_hz,
                                      min_plv=0.0, min_harmonic_ratio=0.0))
    results = gate.apply(tracks, ev_dn, pad_px=cfg.pad_px, bin_us=BIN_US,
                         check_scene_coherence=False)
    # The lamp line is measured once per window from the raw scene, then every
    # track is tested against that one frequency.
    f_lamp = lamp_frequency(ev, t0, t1)
    lo, hi = cfg.wingbeat_band
    rows = []
    traces = {}
    for r in results:
        trk, f = r.track, r.features
        # The track's BAF survivors: events in its detection boxes grown by
        # pad_px pixels. Only counted here; the gate already used them.
        m = trk.event_mask(ev_dn, pad_px=cfg.pad_px)
        # The gate's two frequency estimates, in Hz: the strongest signed-rate
        # line and YIN.
        f0s, fy = f.f0_signed_hz, f.f0_yin_hz
        sides = [d.size_px for d in trk.detections]
        # Wing-stroke statistics use the RAW events in the track's boxes. The
        # BAF is a coincidence filter, so it can remove events preferentially
        # at some phases of the stroke.
        m_raw = trk.event_mask(ev, pad_px=cfg.pad_px)
        sub_raw = ev.select(m_raw)
        # Wingbeat candidates: the signed peak, its half and third, and YIN,
        # kept inside wingbeat_band. The third matters under a lamp: a 33 Hz
        # moth puts its third harmonic on the 100 Hz lamp line, and the
        # signed peak can land there.
        cands = sorted({round(v, 3) for v in (f0s, f0s / 2, f0s / 3, fy)
                        if np.isfinite(v) and lo <= v <= hi})
        # No candidate in the band, or under 200 raw events, leaves the track
        # with no fundamental. Otherwise the pose autocorrelation scores each
        # candidate period. Its slow trend is removed over 8 x the longest
        # candidate period, in microseconds.
        f_best, r_best, r_off, tied = (np.nan, np.nan, np.nan, [])
        if cands and len(sub_raw) >= 200:
            r_pose = pose_autocorr(sub_raw, trk, cfg, slow_us=8e6 / min(cands))
            f_best, r_best, r_off, tied = pick_fundamental(r_pose, cands, cfg.pose_bin_us,
                                                           return_tied=True)
        # Settled = r at the chosen period beats both zero and r at the
        # off-period lag by more than 0.05.
        settled = np.isfinite(f_best) and r_best > max(r_off, 0.0) + 0.05
        if settled:
            # The period and `settled` stay the pose test's; only the number
            # reported for that period can change.
            f_best = same_period_value(f_best, tied, f0s, fy, f_lamp, cfg.same_period_tol)
            # The wingbeat over the track's life, stored in Census.traces.
            tr = wingbeat_trace(sub_raw, f_best)
            if tr is not None:
                traces[trk.track_id] = tr
        # Lock to the lamp line on the raw box events. R > 2.63 is p < 0.001.
        R, ph, depth = lamp_lock(sub_raw, f_lamp)
        # One table row per track. f0_wingbeat_hz is NaN unless the pose test
        # settled a period, and fundamental_from names the candidate the
        # reported number came from. halfcyc_r compares the two halves of a
        # cycle folded at half the signed peak; it is high when the signed
        # peak is a true wingbeat and low when it is double the wingbeat.
        rows.append(dict(
            track_id=trk.track_id,
            t_start_s=trk.t_start_us / 1e6, t_end_s=trk.t_end_us / 1e6,
            duration_ms=trk.duration_us / 1e3, n_detections=len(trk.detections),
            n_events_dn=int(m.sum()), n_events_raw=int(m_raw.sum()),
            max_side_px=float(np.max(sides)), median_side_px=float(np.median(sides)),
            speed_px_s=trk.mean_speed_px_s(),
            f0_signed_hz=f0s, snr_signed_db=f.peak_snr_signed_db,
            f0_yin_hz=fy, yin_conf=f.yin_confidence,
            f0_unsigned_hz=f.f0_unsigned_hz,
            f0_over_yin=(f0s / fy) if np.isfinite(fy) else np.nan,
            f0_wingbeat_hz=f_best if settled else np.nan,
            fundamental_from=("signed" if settled and abs(f_best - f0s) < 0.01 else
                              "signed/2" if settled and abs(f_best - f0s / 2) < 0.01 else
                              "signed/3" if settled and abs(f_best - f0s / 3) < 0.01 else
                              "yin" if settled else "none"),
            pose_r=r_best, pose_r_off=r_off,
            halfcyc_r=half_cycle_similarity(sub_raw, f0s / 2.0),
            lamp_R=R, lamp_phase=ph, lamp_depth=depth,
            plv=f.plv, harmonic_ratio=f.harmonic_ratio,
        ))
    # Verdict and lamp_dominant read whole rows, so they are added once the
    # table exists. The lamp frequency is stored with the scene lines, where
    # census_figure reads it.
    df = pd.DataFrame(rows)
    df["verdict"] = [_verdict(r, cfg, f_lamp) for _, r in df.iterrows()] if len(df) else []
    df["lamp_dominant"] = [_lamp_dominant(r, f_lamp) for _, r in df.iterrows()] if len(df) else []
    lines["lamp_hz"] = f_lamp
    if verbose and len(df):
        vc = df.verdict.value_counts()
        print("lamp line {:.3f} Hz; tracks by verdict: ".format(f_lamp)
              + ", ".join("{} {}".format(k, v) for k, v in vc.items()))
        wb = df[df.verdict == "wingbeat"]
        if len(wb):
            print("wingbeat (Hz): p10 {:.1f}, p50 {:.1f}, p90 {:.1f}; fundamental from {}".format(
                *np.percentile(wb.f0_wingbeat_hz, [10, 50, 90]),
                dict(wb.fundamental_from.value_counts())))
        print("locked to the lamp (R > 2.63, p < 0.001): {}/{} tracks; strongest line is the "
              "lamp with no wingbeat settled: {}".format(
                  int((df.lamp_R > 2.63).sum()), len(df), int(df.lamp_dominant.sum())))
        print("durations ms p25/p50/p75 {:.0f}/{:.0f}/{:.0f}; >= 250 ms: {}".format(
            *df.duration_ms.quantile([.25, .5, .75]), int((df.duration_ms >= 250).sum())))
    # A trace was made for every settled track, but the verdict also needs the
    # SNR gate and min_cycles, so some settled tracks end as "no line". Keep
    # traces only for tracks whose final verdict is wingbeat.
    traces = {k: v for k, v in traces.items()
              if k in set(df.track_id[df.verdict == "wingbeat"]) or not len(df)}
    return Census(raw_path, t0, t1, cfg, ev, keep, grid, dets, tracks, df, lines, traces)


def coverage(c: Census, radius_px: float = 25.0, win_us: int = 5_000,
             cell_px: int = 16, min_events: int = 15) -> Dict[str, float]:
    """How much of the raw event stream no track is near: a pointer to missed insects.

    An event counts as covered when some track's smoothed centre, at the
    event's time, lies within `radius_px`. The radius is fixed and the same
    for every setting. Detection boxes are deliberately not used: a longer
    detector window draws bigger boxes and would cover more events for that
    reason alone. Uncovered events are then binned into cells of win_us
    (5 ms) by cell_px x cell_px (16 px). A cell with at least min_events (15)
    of them counts as one possible missed insect at that moment.

    This is a rough guide for comparing settings, not a recall measurement.
    It is only meaningful on a clean scene such as a dark sky, where few
    events come from anything but insects. With moving foliage in view, the
    uncovered cells will include foliage.

    Returns covered_frac (fraction of events covered), uncovered_cells_per_s,
    and mask (the per-event covered mask).
    """
    ev = c.ev
    covered = np.zeros(len(ev), bool)
    for trk in c.tracks:
        # Events within the track's time span (ev is time-sorted), then the
        # distance from each to the smoothed track centre at its time.
        tp, xp, yp = track_path(trk)
        i0, i1 = np.searchsorted(ev.t, [tp[0], tp[-1] + 1])
        if i1 <= i0:
            continue
        t = ev.t[i0:i1]
        d2 = (ev.x[i0:i1] - np.interp(t, tp, xp)) ** 2 + (ev.y[i0:i1] - np.interp(t, tp, yp)) ** 2
        covered[i0:i1] |= d2 <= radius_px ** 2
    # One integer key per (time window, cell row, cell column) of each
    # uncovered event; count events per key.
    u = ~covered
    W = ev.width // cell_px + 1
    key = ((ev.t[u] - c.t0_us) // win_us) * (W * (ev.height // cell_px + 1)) \
        + (ev.y[u] // cell_px) * W + ev.x[u] // cell_px
    _, cnt = np.unique(key, return_counts=True)
    dur = (c.t1_us - c.t0_us) / 1e6
    return dict(covered_frac=float(covered.mean()),
                uncovered_cells_per_s=float((cnt >= min_events).sum() / dur),
                mask=covered)


def census_figure(c: Census, path: str, title: str = "") -> None:
    """Save a one-page summary figure of a Census to `path`.

    Top: every track on the sensor frame, coloured by verdict. Bottom left:
    the wingbeat population. Bottom right: the lamp test per track. The
    frame is the evidence; the two plots say what the colours mean.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = c.table
    W, H = c.ev.width, c.ev.height
    # Layout: the frame across the full top row, two plots side by side below.
    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.18, wspace=0.18)
    # Top panel: the frame. Raw events per pixel over the whole window, both
    # polarities, on a log scale. The grey scale saturates at 80% of the log
    # maximum, which brightens the rest of the frame.
    ax = fig.add_subplot(gs[0, :])
    cnt = np.bincount(c.ev.y.astype(np.int64) * W + c.ev.x.astype(np.int64),
                      minlength=W * H).reshape(H, W)
    ax.imshow(np.log1p(cnt), cmap="gray", interpolation="nearest", vmax=np.log1p(cnt).max() * 0.8)
    # Each track's path over the frame, in its verdict colour. "no line" tracks
    # are drawn first, thin and faint, so wingbeat tracks sit on top of them.
    order = {"no line": 0, "wingbeat": 1}
    for _, r in sorted(df.iterrows(), key=lambda kv: order[kv[1].verdict]):
        pos = c.track(int(r.track_id)).positions()
        col = VERDICT_COLOUR[r.verdict]
        ax.plot(pos[:, 1], pos[:, 2], color=col, lw=1.6 if r.verdict != "no line" else 0.6,
                alpha=0.95 if r.verdict != "no line" else 0.5)
        # Wingbeat tracks are labelled with their rate in Hz, 4 px right of
        # the track's last position.
        if r.verdict != "no line":
            lab = "{:.0f}".format(r.f0_wingbeat_hz) if np.isfinite(r.f0_wingbeat_hz) else "?"
            ax.text(pos[-1, 1] + 4, pos[-1, 2], lab, color=col, fontsize=9, va="center")
    # The y limits are reversed so row 0 is at the top, as in the sensor image.
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_facecolor(BG); ax.tick_params(colors=DIM)
    n = df.verdict.value_counts()
    f_lamp = c.scene_lines.get("lamp_hz", 100.0)
    ax.set_title("{}{:.2f}-{:.2f} s, {:,} events. {} tracks with a measured wingbeat "
                 "(magenta), {} without (grey).\nLabels: wingbeat in Hz, the fundamental "
                 "chosen per track by where its events repeat"
                 .format(title, c.t0_us / 1e6, c.t1_us / 1e6, len(c.ev), n.get("wingbeat", 0),
                         n.get("no line", 0)), color=INK, fontsize=11)

    # Bottom left: the wingbeat population, wingbeat tracks only, in 2.5 Hz
    # bins from fmin_hz up to at most 250 Hz. The grey outline is each track's
    # strongest signed-rate line; the filled bars are the rate after the
    # octave test. Where the two differ, the test chose a candidate other than
    # the signed peak. The dashed line is the measured lamp line.
    a2 = fig.add_subplot(gs[1, 0])
    bins = np.arange(c.config.fmin_hz, min(c.config.fmax_hz, 250) + 2.5, 2.5)
    wb = df[df.verdict == "wingbeat"]
    a2.hist(wb.f0_signed_hz, bins=bins, histtype="step", color=DIM, lw=1.2,
            label="strongest signed-rate line (n={})".format(len(wb)))
    a2.hist(wb.f0_wingbeat_hz, bins=bins, color=TARGET, alpha=0.85,
            label="wingbeat, after the per-track octave test (n={})".format(len(wb)))
    a2.axvline(f_lamp, color=CLUTTER, lw=1.2, ls="--")
    a2.text(f_lamp + 2, a2.get_ylim()[1] * 0.92, "lamp {:.2f} Hz".format(f_lamp),
            color=CLUTTER, fontsize=10)
    a2.set_xlim(c.config.fmin_hz, min(c.config.fmax_hz, 250))
    a2.set_xlabel("frequency (Hz)", color=DIM)
    a2.set_ylabel("tracks", color=DIM)
    a2.legend(facecolor=BG, edgecolor=GRID, labelcolor=INK)

    # Bottom right: the lamp test for every track. Track duration in ms, on a
    # log axis, against lamp_R, the vector strength at the lamp line. Points
    # above the dashed line at R = 2.63 are locked to the lamp at p < 0.001.
    a3 = fig.add_subplot(gs[1, 1])
    for v, lab in (("no line", "no wingbeat measured"), ("wingbeat", "wingbeat")):
        sel = df[df.verdict == v]
        a3.scatter(sel.duration_ms, sel.lamp_R, s=10 if v == "no line" else 22,
                   color=VERDICT_COLOUR[v] if v == "wingbeat" else DIM, alpha=0.7,
                   label="{} (n={})".format(lab, len(sel)), edgecolor="none")
    a3.axhline(2.63, color=CLUTTER, lw=0.8, ls="--")
    a3.set_xscale("log")
    a3.set_xlabel("track duration (ms)", color=DIM)
    a3.set_ylabel("lock to the lamp line: vector strength R (above dashed line: p < 0.001)",
                  color=DIM, fontsize=9)
    a3.set_title("{} of {} tracks flicker with the lamp".format(
        int((df.lamp_R > 2.63).sum()), len(df)), color=INK, fontsize=11)
    a3.legend(facecolor=BG, edgecolor=GRID, labelcolor=INK)
    # Dark styling for the two lower plots.
    for a in (a2, a3):
        a.set_facecolor(BG); a.tick_params(colors=DIM)
        for s in a.spines.values():
            s.set_color(GRID)
    fig.savefig(path, facecolor=BG, bbox_inches="tight", dpi=90)
    plt.close(fig)
