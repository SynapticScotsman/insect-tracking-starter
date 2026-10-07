"""The census chain used on the Bogong moth recording, lifted verbatim from
analysis/recording_census.py by analysis/export_starter.py. Read run_census
first. Edits to lifted text are listed in PROVENANCE.json."""
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

BG, INK, DIM, GRID = "#05070A", "#FFFFFF", "#8A97A0", "#1C242C"


TARGET, CLUTTER, REF, DROPPED = "#FF3DA6", "#FFB000", "#7CE0C0", "#39424A"


#: Verdict names, in the colours the figure and every downstream clip use.
#: "wingbeat" = a countable line whose fundamental the pose test settled.
#: "no line"  = no wingbeat settled: mostly tracks too short to count.
#: There is deliberately no "lamp" verdict. It existed twice and was wrong both
#: times on real light. First as a grid/phase test (18 of 31 moths mislabelled).
#: Then as "the strongest line is the lamp and no wingbeat settled": on the
#: 12 ms census that still put 4 tracks in it, and the two the user saw in the
#: clip were moths (track 149 flies 65 px in 284 ms with YIN 46.7 Hz and an
#: unsigned peak at 47.3 Hz; track 140 is a 72 ms, 164-event glimpse of one).
#: Because the lamp lights every moth (195 of 284 locked), ANY moth whose wing
#: signal is too weak to settle shows the lamp as its strongest rhythm. That
#: makes it a moth without a measured wingbeat, which is what "no line" says.
#: `lamp_dominant` records the fact without naming the object.
VERDICT_COLOUR = {"wingbeat": TARGET, "no line": DROPPED}


BIN_US = 200          # 5 kHz rate signal; periodicity.rate_signals' own default


HOP_US = 2_500        # frozen recipe


@dataclass
class CensusConfig:
    fmin_hz: float = 15.0
    fmax_hz: float = 500.0
    baf_dt_us: int = 1_000          # 0 disables the filter
    coast_us: int = 50_000
    gate_px: float = 45.0
    max_size_px: int = 90           # frozen recipe; check `size_px` percentiles
    min_snr_db: float = 10.0        # project-wide line gate (sp3_feast_select.py)
    min_cycles: float = 10.0        # 43 ms at 233 Hz: AEMOT's "countable"
    pad_px: int = 3
    # Detector. The frozen recipe is window 5 ms / hop 2.5 ms, sized for a
    # 230 Hz bee: 5 ms is 1.15 strokes, so every blob holds a whole stroke and
    # the centroid is the body. For a 40 Hz moth 5 ms is a fifth of a stroke,
    # so a blob can be one wing and the centroid swings with it (1.6-1.9 px rms
    # in the wing band on track 71). Scale the window by the period, not by eye.
    window_us: int = 5_000
    hop_us: int = 2_500
    detector: str = "single"        # or "multiscale" (detect.py's 5-80 ms bank)
    nms_px: float = 0.0             # 0 = frozen recipe; see CLAUDE.md on NMS
    # Wingbeat candidates are kept inside this band. 18-80 Hz is the moth band
    # read off this recording's population (arbitrated p10-p90 29-55 Hz) and
    # confirmed by eye as moths; it excludes the 100 Hz lamp line and the
    # 15-17 Hz values YIN reaches when it walks to a subharmonic.
    wingbeat_band: tuple = (18.0, 80.0)
    pose_bin_us: int = 500          # the verifier's detrended variant: 0.5 ms
    pose_cells: int = 6             # bins, 6 x 6 cells x 2 polarities
    # Once the pose test has chosen the period, an estimate within this
    # fraction of it is the SAME period, and the reported number comes from
    # the most precise one (same_period_value). 0 reproduces every census up
    # to commit a5fbd54, where the higher of the two estimates won.
    same_period_tol: float = 0.10


@dataclass
class Census:
    raw_path: str
    t0_us: int
    t1_us: int
    config: CensusConfig
    ev: EventStream                 # the raw window, time-sorted
    keep: np.ndarray                # BAF keep mask over `ev`
    grid_origin_us: int
    detections: list
    tracks: list
    table: "object"                 # pandas DataFrame, one row per track
    scene_lines: Dict[str, float] = field(default_factory=dict)
    # track_id -> (t_us, f_hz): the wingbeat over the track's life, for every
    # wingbeat track (see wingbeat_trace). Pickles made before this field load
    # without it; use getattr(c, "traces", {}).
    traces: Dict[int, tuple] = field(default_factory=dict)

    @property
    def ev_dn(self) -> EventStream:
        return self.ev.select(self.keep)

    def track(self, track_id: int):
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        raise KeyError(track_id)

    def heroes(self, n: int = 5, min_events: int = 5_000, min_cycles: float = 30.0):
        """Best wingbeat tracks to render, by hero_flap.py:96's own rule:
        largest box side times SNR capped at 25 dB. Capping SNR stops one very
        clean but tiny target outranking a big one whose wing is resolvable.

        min_cycles: the phase renderers fold over many wingbeats (hero_flap 11,
        phase_bee up to 90) and pick the densest stretch inside the track, so a
        127 ms track at 42 Hz (5 cycles) cannot feed them however large it is."""
        df = self.table
        cyc = df.f0_wingbeat_hz * df.duration_ms / 1e3
        cand = df[(df.verdict == "wingbeat") & (df.n_events_raw >= min_events)
                  & np.isfinite(df.f0_wingbeat_hz) & (cyc >= min_cycles)]
        cand = cand.assign(hero=cand.max_side_px * cand.snr_signed_db.clip(upper=25))
        return cand.sort_values("hero", ascending=False).head(n)


def scene_lines(ev: EventStream, t0: int, t1: int) -> Dict[str, float]:
    """Line SNR of the scene-summed signed and unsigned rate at 100/200/300 Hz.

    A lamp lights every animal in phase, so its line adds coherently across the
    scene, while independent wingbeats smear. Measured on the 8-18 s stretch of
    recording_2026-10-02_21-38-40: signed 100.1 Hz at 35.5 dB over the median.
    """
    fs = 1e6 / BIN_US
    _, u, s = rate_signals(ev, bin_us=BIN_US, t0_us=t0, t1_us=t1)
    out = {}
    for name, sig in (("signed", s), ("unsigned", u)):
        f, P, _ = _psd(_detrend_highpass(sig, fs, 5.0), fs, df_target_hz=0.5)
        band = (f >= 20) & (f <= 1000)
        for line in (100.0, 200.0, 300.0):
            out["{}_{:.0f}Hz_db".format(name, line)] = _line_snr_db(
                f, P, line, exclude_hz=1.5, band=band)
    return out


def lamp_frequency(ev: EventStream, t0: int, t1: int) -> float:
    """The scene's own lamp line, by fine peak of the scene-summed signed rate
    in 95-105 Hz. Measured, not assumed to be 100.000: the mains drifts, and a
    lamp test at the wrong frequency loses lock over a few seconds (0.02 Hz
    over 4 s is 0.08 cycle). 17.5-21.5 s: 100.015 Hz."""
    fs = 1e6 / BIN_US
    _, _, s = rate_signals(ev, bin_us=BIN_US, t0_us=t0, t1_us=t1)
    f, _ = _fine_peak(_detrend_highpass(s, fs, 5.0), fs, 95.0, 105.0)
    return float(f) if np.isfinite(f) else 100.0


def track_path(trk, smooth_ms: float = 25.0, step_us: int = 1_000):
    """Track centre on a 1 ms grid, averaged over `smooth_ms`.

    Detection centres swing with the wing when the detector window is shorter
    than a stroke (track 71: 3.5 px rms residual, 29-49% of it in the 30-55 Hz
    band). 25 ms is one stroke at 40 Hz. Returns (t_us, x, y)."""
    from scipy.ndimage import uniform_filter1d

    pos = trk.positions()
    t = np.arange(pos[0, 0], pos[-1, 0] + 1, step_us)
    x = np.interp(t, pos[:, 0], pos[:, 1])
    y = np.interp(t, pos[:, 0], pos[:, 2])
    w = max(int(round(smooth_ms * 1e3 / step_us)), 1)
    return t, uniform_filter1d(x, w, mode="nearest"), uniform_filter1d(y, w, mode="nearest")


def pose_autocorr(sub: EventStream, trk, cfg: CensusConfig, slow_us: float) -> np.ndarray:
    """evfilt's descriptor autocorrelation of a track-centred ON/OFF histogram,
    with the slow trend removed so a lag that is not a period reads near zero.

    Why a descriptor and not a rate: on this recording's doubled tracks the ON
    and OFF bursts come every T_s in BOTH polarities (median gap 1.03 T_s), so
    every rate-based count reads the doubled frequency; only the SPATIAL pattern
    of consecutive bursts alternates. A descriptor that sees where the events
    land comes back at the true period. Measured (17.5-21.5 s, this variant):
    doubled tracks r(T_s) +0.028, r(2T_s) +0.284; agree tracks r(T_s) +0.360,
    r(2T_s) +0.186; off-lag 1.37 T near 0 in both.

    Why the detrend: evfilt's raw usage (no detrend) keeps the static body
    shape in every lag, so r is high at ALL lags and the choice between T and 3T
    rides on noise; the census self-test caught it calling a 130.3 Hz insect
    43.4 Hz. Subtracting a moving average over `slow_us` (8 x the longest
    candidate period) leaves only what changes within the stroke.
    Returns r[k] at lag k * pose_bin_us.
    """
    from insect_evs.descriptor import descriptor_autocorr
    from scipy.ndimage import uniform_filter1d

    G, bus = cfg.pose_cells, cfg.pose_bin_us
    side = max(float(np.median([d.size_px for d in trk.detections])), 4.0) + 6
    tp, xp, yp = track_path(trk)
    cx = np.interp(sub.t, tp, xp)
    cy = np.interp(sub.t, tp, yp)
    gx = np.clip(((sub.x - cx) / side + 0.5) * G, 0, G - 1e-6).astype(np.int64)
    gy = np.clip(((sub.y - cy) / side + 0.5) * G, 0, G - 1e-6).astype(np.int64)
    b = ((sub.t - sub.t[0]) // bus).astype(np.int64)
    D = np.zeros((int(b.max()) + 1, 2 * G * G))
    np.add.at(D, (b, (sub.p > 0).astype(np.int64) * G * G + gy * G + gx), 1.0)
    D = uniform_filter1d(D, 3, axis=0, mode="nearest")
    w = max(int(round(slow_us / bus)), 5)
    D = D - uniform_filter1d(D, w, axis=0, mode="nearest")
    return descriptor_autocorr(D)


def pick_fundamental(r: np.ndarray, cands: List[float], bin_us: int, tie: float = 0.05,
                     return_tied: bool = False):
    """The SHORTEST candidate period whose pose repeat comes within `tie` of the
    strongest, with its r and the r at an off-period lag (1.37 x, a subharmonic
    of nothing). A true period T repeats at 2T and 3T too, so a longer
    candidate can match T's r by noise; preferring the shortest one inside the
    tie stops a multiple winning that way. 0.05 sits well under the measured
    separations (doubled r(2T) - r(T) +0.26, agree r(T) - r(2T) +0.17).
    Lags past 60% of the series are not read: too few pairs.

    return_tied=True appends the frequencies of every candidate inside the tie,
    which same_period_value needs to choose among same-period estimates."""
    n = len(r)
    scored = []
    for f in cands:
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
    """The wingbeat NUMBER, once pick_fundamental has chosen the period.

    "Shortest period inside the tie" is there to stop 2T or 3T beating T. It
    also runs when the signed line and YIN name the SAME period a few percent
    apart and the pose test cannot separate them (both inside the tie), and
    there it reports whichever reads higher: the max of two estimates of one
    number, biased upward by construction.

    So among the TIED candidates within `tol` of the pick, the number comes
    from the signed line (or its half or third), else YIN, else the pick. A
    candidate outside the tie is never used: there the pose test did prefer
    one period, and the first version of this rule, which ignored the tie,
    overrode it and moved numbers UP by as much as 8.9%. The signed line is
    preferred because _fine_peak refines it on the whole track (8x zero-padded,
    parabolic vertex); a signed line ON the lamp (within 1 Hz of 1, 2 or
    3 x f_lamp) is ranked last, because it measures the lamp and its third,
    33.34 Hz, is not a wingbeat even when a moth's period is near it. 10% is
    far below the 33% between octave candidates (f/2 against f/3).
    tol = 0 returns the pick: every census up to commit a5fbd54.

    Measured (analysis/census_precision.py, 2026-10-07). Synthetic insects,
    truth known, 8 scenes x 3 insects per rate: on moths the rule changed 36
    numbers and 35 moved nearer the truth (median error 0.33% -> 0.02%); on
    bees all 23 it changed moved nearer (0.17% -> 0.00%); no octave changed.
    Real windows, no truth: Bogong 17.5-21.5 s, 40 of 110 numbers moved, all
    down, median -1.8%, population median 44.54 -> 43.99 Hz; AEMOT standard
    window, 40 of 106 moved, all down, median -1.0%, 231.50 -> 227.76 Hz.
    """
    if not (np.isfinite(f_pick) and tol > 0):
        return f_pick
    on_lamp = np.isfinite(f_signed) and any(abs(f_signed - m * f_lamp) <= 1.0 for m in (1, 2, 3))

    def rank(f):
        # Candidates are rounded to 0.001 Hz in run_census, hence the 1e-3.
        if np.isfinite(f_signed) and any(abs(f - f_signed / k) < 1e-3 for k in (1, 2, 3)):
            return 2 if on_lamp else 0
        return 1 if (np.isfinite(f_yin) and abs(f - f_yin) < 1e-3) else 2

    same = [f for f in tied if abs(np.log(f / f_pick)) < np.log1p(tol)] or [f_pick]
    return min(same, key=lambda f: (rank(f), -f))


def wingbeat_trace(sub: EventStream, f0_hz: float, smooth_ms: float = 50.0,
                   edge_ms: float = 50.0, lowpass_hz: float = 8.0):
    """The wingbeat over a track's life: the slope of periodicity.phase_track.

    Why: a moth's wingbeat is not one number. Track 71/14 ran 35.8-45.5 Hz
    (p5-p95) within 100 ms, and the user saw the rate change in the clip while
    the label stayed fixed. phase_track demodulates at the track's settled f0
    and follows within +/- lowpass_hz of it, so the octave call is inherited,
    not re-decided per instant. The slope is averaged over `smooth_ms` (two
    strokes at 40 Hz) and `edge_ms` is cut from each end, where the zero-phase
    filter has data on one side only.

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
    """Times (us) of individual wing strokes, from raw events, without f(t).

    The ON-event clock of the wingbeat_trace check (scratchpad verify2,
    strokes.py, promoted): ON-event count per 0.5 ms in the track's boxes
    (a count, so the moth's motion does not enter), slow trend over two
    periods removed, band-passed `band` x f0, peaks at least `min_gap` / f0
    apart. With the first settings (0.5-2 x f0, 0.6) it found 0.91-1.07
    strokes per expected stroke on 8 moths, and of three clocks tried it
    reproduced best on split halves of the events (r 0.79).

    Why 0.5-1.5 x f0 and 0.75 (2026-10-07, analysis/census_precision.py,
    synthetic insects whose rate is known): with the upper edge at 2 f0 the
    second harmonic sits at the band edge and puts a second peak in some
    strokes. On synthetic bees 14.0% of intervals were a stroke counted twice;
    with the edge at 1.5 f0, 0.0%, and the median per-stroke error fell from
    2.05% to 1.50% (moths 0.94% -> 0.81%). The 0.75 gap barely moves the
    synthetic data (moths: double counts 0.5% -> 0.0%, skipped strokes 2.8% ->
    3.4%; bees unchanged); on the real bee window it cut intervals reading over 1.4x
    or under 0.7x the bee's own wingbeat from 11.7% to 7.1% (a honey bee
    varies about 1.2 Hz within itself, so those are clock errors). On the real
    moth window it turned double counts into skipped strokes (0.0% / 10.7%
    against 6.5% / 5.3% without it): a cost on moths, recorded, not hidden.
    band=(0.5, 2.0), min_gap=0.6, subbin=False reproduce every stroke time up
    to commit a5fbd54.

    What it is NOT: validated against an independent stroke clock. The
    event-centroid clock disagrees with it stroke by stroke (r -0.34 to +0.38
    over 8 moths), so 1 / interval is the ON clock's rate, not established
    truth. It exists because wingbeat_trace cannot show stroke-level change at
    all (gain 0.02 at 13 Hz modulation) and the user sees the rate change.
    """
    import scipy.signal as sps
    from scipy.ndimage import uniform_filter1d

    if len(sub) < 200 or not np.isfinite(f0_hz):
        return np.zeros(0)
    t0 = int(sub.t[0])
    n = int((sub.t[-1] - t0) // bin_us) + 1
    b = ((sub.t - t0) // bin_us).astype(np.int64)
    on = np.bincount(b, weights=(sub.p > 0).astype(float), minlength=n)
    fs = 1e6 / bin_us
    s = on - uniform_filter1d(on, max(int(2 * fs / f0_hz), 3), mode="nearest")
    sos = sps.butter(2, [band[0] * f0_hz / (fs / 2), min(band[1] * f0_hz / (fs / 2), 0.95)],
                     btype="band", output="sos")
    if n < 30:
        return np.zeros(0)
    s = sps.sosfiltfilt(sos, s)
    pk, _ = sps.find_peaks(s, distance=max(int(min_gap * fs / f0_hz), 1))
    pos = pk.astype(float)
    if subbin:
        # Vertex of the parabola through each peak bin and its neighbours.
        # Without it every stroke time sits on a 0.5 ms bin centre, so a
        # 230 Hz bee's 4.35 ms stroke read as 4.0 or 4.5 ms, 250 or 222 Hz.
        # The band-passed signal is smooth at the bin scale (a 40 Hz stroke
        # spans 50 bins, a 230 Hz one 8.7), so the vertex places the peak
        # between bins. subbin=False reproduces every result up to a5fbd54.
        inner = (pk > 0) & (pk < len(s) - 1)
        k = pk[inner]
        a, b, c = s[k - 1], s[k], s[k + 1]
        den = a - 2.0 * b + c
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
        pos[inner] += np.clip(d, -0.5, 0.5)
    return t0 + (pos + 0.5) * bin_us


def running_stroke_rate(st: np.ndarray, k: int = 5) -> np.ndarray:
    """The live wingbeat label: the median rate of the last k strokes.

    One value per stroke from the second on (aligned with st[1:]): the median
    of 1 / interval over the last min(k, available) intervals, so the label at
    a stroke uses only strokes already seen. k = 1 is the old label, 1 / the
    last interval. A median of k ignores up to (k - 1) / 2 bad intervals (a
    stroke counted twice or skipped) and costs about k strokes of time
    resolution.

    k = 5 is the label in the clips and the starter's rate_median5_hz, chosen
    by the user on these numbers (analysis/stroke_label.py, 2026-10-08):
    - Real windows, labels over 1.4x or under 0.7x the track's own wingbeat:
      moths 10.7% (k = 1) -> 3.3%, bees 7.1% -> 0.9%.
    - Synthetic insects swinging +-15%, share of the swing shown and delay:
      moths 0.95 at 2 Hz, 0.70 at 4 Hz, 0.32 at 8 Hz, ~55-60 ms late (k = 1:
      1.00 / 0.97 / 0.92, 12 ms); bees 0.98 at 5 Hz, 0.94 at 10 Hz, 0.79 at
      20 Hz, 11 ms late.
    - Synthetic steady moths: median error 0.68% -> 0.29%, but labels over
      10% off barely move (7.7% -> 7.1%): those errors come in runs that a
      median of 5 cannot outvote. On the real windows most gross errors are
      single intervals, since the median removes two thirds of them on moths
      and seven eighths on bees.
    So on moths the label shows drift slower than about 4 Hz and hides faster
    swings; the tracked phase slope (wingbeat_trace) follows moths better
    (0.83 at 4 Hz, no delay) but fails on bees (0.16 at 2 Hz), because their
    +-15% swing leaves its 8 Hz band."""
    st = np.asarray(st, float)
    if len(st) < 2:
        return np.zeros(0)
    rate = 1e6 / np.diff(st)
    return np.array([np.median(rate[max(0, i - k + 1):i + 1]) for i in range(len(rate))])


def lamp_lock(sub: EventStream, f_lamp: float):
    """Polarity-weighted vector strength of a track's events at the lamp line,
    in absolute time so every track shares one phase reference.

    R = |sum p exp(-2 pi i f t)| / sqrt(n); with independent events R^2 is
    exponential, so R > 2.63 is p < 0.001. Returns (R, phase_rad, depth) with
    depth = |sum| / n, the modulation depth."""
    if len(sub) < 50:
        return np.nan, np.nan, np.nan
    z = np.sum(sub.p * np.exp(-2j * np.pi * f_lamp * (sub.t * 1e-6)))
    return float(abs(z) / np.sqrt(len(sub))), float(np.angle(z)), float(abs(z) / len(sub))


def _verdict(row, cfg: CensusConfig, f_lamp: float) -> str:
    countable = (row["snr_signed_db"] >= cfg.min_snr_db
                 and np.isfinite(row["f0_signed_hz"]))
    fw = row["f0_wingbeat_hz"]
    # A settled fundamental AT the lamp line or its harmonics (not its half:
    # moths beat at 48-51 Hz here) is the lamp's pattern repeating, not a
    # wing. Only reachable when the band admits 100 Hz, as in the synthetic
    # self-test with a flickering source in view.
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
    import pandas as pd

    cfg = cfg or CensusConfig()
    ev = ev.sorted_by_time()
    t0, t1 = int(ev.t[0]), int(ev.t[-1]) + 1
    keep = (background_activity_mask(ev, dt_us=cfg.baf_dt_us, radius=1)
            if cfg.baf_dt_us > 0 else np.ones(len(ev), bool))
    ev_dn = ev.select(keep)
    lines = scene_lines(ev, t0, t1)
    if verbose:
        print("window {:.3f}-{:.3f} s: {:,} events ({:.2f} Mev/s), BAF keeps {:.1%}".format(
            t0 / 1e6, t1 / 1e6, len(ev), len(ev) / ((t1 - t0) / 1e6) / 1e6, keep.mean()))
        print("scene lines (dB over floor): " + ", ".join(
            "{} {:.1f}".format(k.replace("_db", ""), v) for k, v in lines.items()))

    grid = (t0 // cfg.hop_us) * cfg.hop_us
    if cfg.detector == "multiscale":
        dets = detect_blobs_multiscale(ev_dn, grid_origin_us=grid, blur_sigma=1.0,
                                       nms_dist_px=cfg.nms_px)
    else:
        dets = detect_blobs(ev_dn, window_us=cfg.window_us, hop_us=cfg.hop_us,
                            blur_sigma=1.0, pixel_threshold=0.5, min_events_blob=10,
                            min_area_px=3, max_size_px=cfg.max_size_px,
                            nms_dist_px=cfg.nms_px, grid_origin_us=grid)
    tracks = link_tracks(dets, gate_px=cfg.gate_px, coast_us=cfg.coast_us,
                         min_detections=2, min_duration_us=5_000)
    if verbose:
        sz = np.array([d.size_px for d in dets]) if dets else np.zeros(1)
        print("{:,} detections (size px p50 {:.0f}, p90 {:.0f}, p99 {:.0f}; "
              "{:.1%} at the {} px cap) -> {:,} tracks".format(
                  len(dets), *np.percentile(sz, [50, 90, 99]),
                  float(np.mean(sz >= cfg.max_size_px)), cfg.max_size_px, len(tracks)))

    # The library gate, with the two synthetic-only thresholds switched off
    # (min_plv=0.50 "rejects almost every genuine target", classify.py:50-57;
    # min_harmonic_ratio=0.12 sits between a bee's 0.097 and flicker's 0.146).
    # Only its features are used; `accepted` is not.
    gate = PeriodicityGate(GateConfig(fmin_hz=cfg.fmin_hz, fmax_hz=cfg.fmax_hz,
                                      min_plv=0.0, min_harmonic_ratio=0.0))
    results = gate.apply(tracks, ev_dn, pad_px=cfg.pad_px, bin_us=BIN_US,
                         check_scene_coherence=False)
    # scene_coherence is no longer computed. It was retired as a lamp test on
    # this recording: at a moth's 30-70 Hz a short track has 3-7 independent
    # windows and two unrelated signals score up to 0.6 (null p90 0.604 on
    # tracks under four windows), and at 100 Hz it read the same on wingbeat
    # tracks as at an 87.3 Hz control (0.219 vs 0.216) although the lamp is
    # demonstrably present in them. lamp_lock below replaces it.
    f_lamp = lamp_frequency(ev, t0, t1)
    lo, hi = cfg.wingbeat_band
    rows = []
    traces = {}
    for r in results:
        trk, f = r.track, r.features
        m = trk.event_mask(ev_dn, pad_px=cfg.pad_px)
        f0s, fy = f.f0_signed_hz, f.f0_yin_hz
        sides = [d.size_px for d in trk.detections]
        # Phase statistics on RAW box events: BAF is a coincidence filter and
        # may censor stroke phase as Conv1 does (wf8_taps.py).
        m_raw = trk.event_mask(ev, pad_px=cfg.pad_px)
        sub_raw = ev.select(m_raw)
        # Fundamental, per track. Candidates: the signed peak, its half and
        # third (a lamp-lit moth at 33 Hz puts its third harmonic on the 100 Hz
        # lamp line: 22 tracks sat there), and YIN; kept inside the moth band.
        cands = sorted({round(v, 3) for v in (f0s, f0s / 2, f0s / 3, fy)
                        if np.isfinite(v) and lo <= v <= hi})
        f_best, r_best, r_off, tied = (np.nan, np.nan, np.nan, [])
        if cands and len(sub_raw) >= 200:
            r_pose = pose_autocorr(sub_raw, trk, cfg, slow_us=8e6 / min(cands))
            f_best, r_best, r_off, tied = pick_fundamental(r_pose, cands, cfg.pose_bin_us,
                                                           return_tied=True)
        # Settled = a real repeat at the chosen period, above the off-period lag.
        settled = np.isfinite(f_best) and r_best > max(r_off, 0.0) + 0.05
        if settled:
            # The period and `settled` stay the pose test's; only the number
            # reported for that period can change.
            f_best = same_period_value(f_best, tied, f0s, fy, f_lamp, cfg.same_period_tol)
            tr = wingbeat_trace(sub_raw, f_best)
            if tr is not None:
                traces[trk.track_id] = tr
        R, ph, depth = lamp_lock(sub_raw, f_lamp)
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
    traces = {k: v for k, v in traces.items()
              if k in set(df.track_id[df.verdict == "wingbeat"]) or not len(df)}
    return Census(raw_path, t0, t1, cfg, ev, keep, grid, dets, tracks, df, lines, traces)


def coverage(c: Census, radius_px: float = 25.0, win_us: int = 5_000,
             cell_px: int = 16, min_events: int = 15) -> Dict[str, float]:
    """How much of the raw event stream no track is near: candidate missed moths.

    An event counts as covered when some track's smoothed centre, interpolated
    to the event's time, lies within `radius_px`. The same fixed radius for
    every tracker, NOT the detection boxes: a longer detector window draws
    bigger boxes and would cover more by construction (the box-coverage proxies
    in CLAUDE.md all inverted). Uncovered events are then binned into 5 ms x
    16 px cells; a cell with >= 15 of them is one candidate missed animal-slice.
    This is a proxy for narrowing a choice, and the scene is unusually clean
    (dark sky, BAF keeps 84%), which is the only reason it means anything here.
    Returns the covered fraction, uncovered cells per second, and the mask.
    """
    ev = c.ev
    covered = np.zeros(len(ev), bool)
    for trk in c.tracks:
        tp, xp, yp = track_path(trk)
        i0, i1 = np.searchsorted(ev.t, [tp[0], tp[-1] + 1])
        if i1 <= i0:
            continue
        t = ev.t[i0:i1]
        d2 = (ev.x[i0:i1] - np.interp(t, tp, xp)) ** 2 + (ev.y[i0:i1] - np.interp(t, tp, yp)) ** 2
        covered[i0:i1] |= d2 <= radius_px ** 2
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
    """Tracks on the frame by verdict, then the f0 population and the lamp test.

    Few large panels on purpose: the frame is the evidence, the two plots say
    what the colours mean.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = c.table
    W, H = c.ev.width, c.ev.height
    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.18, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    cnt = np.bincount(c.ev.y.astype(np.int64) * W + c.ev.x.astype(np.int64),
                      minlength=W * H).reshape(H, W)
    ax.imshow(np.log1p(cnt), cmap="gray", interpolation="nearest", vmax=np.log1p(cnt).max() * 0.8)
    order = {"no line": 0, "wingbeat": 1}
    for _, r in sorted(df.iterrows(), key=lambda kv: order[kv[1].verdict]):
        pos = c.track(int(r.track_id)).positions()
        col = VERDICT_COLOUR[r.verdict]
        ax.plot(pos[:, 1], pos[:, 2], color=col, lw=1.6 if r.verdict != "no line" else 0.6,
                alpha=0.95 if r.verdict != "no line" else 0.5)
        if r.verdict != "no line":
            lab = "{:.0f}".format(r.f0_wingbeat_hz) if np.isfinite(r.f0_wingbeat_hz) else "?"
            ax.text(pos[-1, 1] + 4, pos[-1, 2], lab, color=col, fontsize=9, va="center")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_facecolor(BG); ax.tick_params(colors=DIM)
    n = df.verdict.value_counts()
    f_lamp = c.scene_lines.get("lamp_hz", 100.0)
    ax.set_title("{}{:.2f}-{:.2f} s, {:,} events. {} tracks with a measured wingbeat "
                 "(magenta), {} without (grey).\nLabels: wingbeat in Hz, the fundamental "
                 "chosen per track by where its events repeat"
                 .format(title, c.t0_us / 1e6, c.t1_us / 1e6, len(c.ev), n.get("wingbeat", 0),
                         n.get("no line", 0)), color=INK, fontsize=11)

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
    for a in (a2, a3):
        a.set_facecolor(BG); a.tick_params(colors=DIM)
        for s in a.spines.values():
            s.set_color(GRID)
    fig.savefig(path, facecolor=BG, bbox_inches="tight", dpi=90)
    plt.close(fig)
