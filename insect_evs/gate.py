"""Wingbeat test for every track.

PeriodicityGate measures the periodicity features of each track's events
(see `periodicity.analyse_periodicity`) and checks them against the
thresholds in GateConfig. A track is accepted when it passes every test:
a clear spectral line, a steady phase, harmonics, a fundamental inside the
wingbeat band, enough cycles and events, little power below 20 Hz, and no
phase locking to the rest of the scene, which would point to a flickering
lamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .events import EventStream
from .periodicity import (PeriodicityFeatures, analyse_periodicity,
                          cross_phase_locking, rate_signals)
from .track import Track


@dataclass
class GateConfig:
    """Thresholds for the wingbeat test. The note under each field says what
    it tests.

    The defaults were set on synthetic scenes, in the gap between insect and
    clutter values with about a factor of two of margin on each side. Treat
    them as a starting point. Real recordings have lower signal-to-noise
    ratios, so check the thresholds against tracks you have labelled by eye
    from your own recordings before relying on them. `min_plv` in particular
    is too strict for real insects; see its note.
    """

    min_snr_db: float = 10.0
    """The spectral line at the fundamental must stand this many dB above the
    local noise floor. A close wingbeat gives 25 to 35 dB on synthetic scenes,
    and short clutter transients in the same band rarely exceed 11 dB. Lower
    it for distant or low-contrast insects, but below about 6 dB the frequency
    estimate is too loose to tell one individual from another."""

    min_plv: float = 0.50
    """Smallest phase-locking value (PLV): how steady the wing phase stays from
    cycle to cycle, from 0 to 1. Too strict for real insects at this value. A
    simulated wing is a perfect oscillator and scores 1.0, but real insects
    with strong spectral lines (21 to 31 dB) score around 0.2 to 0.3, so this
    default rejects almost all of them. Lower it for field data; census.py
    sets it to 0. `periodicity.phase_locking_value` explains why real animals
    score lower than a simulation."""

    min_harmonic_ratio: float = 0.12
    """Smallest share of the in-band power at the harmonics of the
    fundamental. A wing stroke is not a pure sine wave, so it always has
    harmonics. Clutter transients and single-tone interference generally do
    not."""

    max_scene_coherence: float = 0.60
    """Largest phase locking between the track and the rest of the scene at
    the track's own fundamental. This is the artificial-light test: mains
    flicker at 100 or 120 Hz lies inside the wingbeat band and passes every
    other test, so without this check a lit hedge looks like a swarm.
    `periodicity.cross_phase_locking` explains why the threshold cannot go
    much below 0.5 on short tracks."""

    max_low_freq_ratio: float = 0.85
    """Largest share of power below 20 Hz. A track with more is swaying
    vegetation moved by wind."""

    min_cycles: float = 20.0
    """Fewest wing cycles the track must span: 20 cycles is about 100 ms at
    200 Hz. On shorter tracks the frequency estimate and the phase-locking
    value both look better than they are, so short clutter fragments pass
    the other tests. This threshold removes them."""

    min_events: int = 200
    fmin_hz: float = 40.0
    fmax_hz: float = 400.0
    """`min_events` is the fewest events a track may have. `fmin_hz` and
    `fmax_hz` bound the band, in Hz, searched for the fundamental."""

    require_octave_agreement: bool = False
    """Strict mode: also require the unsigned spectral peak to sit at twice the
    fundamental. Fewer false accepts, but it loses insects seen edge-on to the
    stroke plane, where the ON/OFF difference that carries the fundamental
    largely cancels."""


@dataclass
class TrackResult:
    """The gate's verdict on one track: its features, whether it was
    accepted, the weakest-link score, and the names of the tests it failed.
    `truth_label` and `truth_purity` are for callers that score against
    labelled data; the gate leaves them unset."""
    track: Track
    features: PeriodicityFeatures
    accepted: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    truth_label: Optional[int] = None
    truth_purity: float = np.nan

    @property
    def track_id(self) -> int:
        return self.track.track_id


def _margins(f: PeriodicityFeatures, cfg: GateConfig) -> Dict[str, float]:
    """Distance past each threshold, in scaled units. Negative means the test
    failed.

    Each denominator is a rough size of one meaningful step in that quantity,
    for example 6 dB of line strength or 5 cycles. They set how the tests
    compare in the combined score. They are chosen by hand, not fitted. A
    track with no fundamental gets -inf for the band test.
    """
    return {
        "snr": (f.peak_snr_db - cfg.min_snr_db) / 6.0,
        "plv": (f.plv - cfg.min_plv) / 0.15,
        "harmonics": (f.harmonic_ratio - cfg.min_harmonic_ratio) / 0.05,
        "low_freq": (cfg.max_low_freq_ratio - f.low_freq_ratio) / 0.10,
        "scene_coherence": (cfg.max_scene_coherence - f.scene_coherence) / 0.10,
        "cycles": (f.n_cycles - cfg.min_cycles) / 5.0,
        "events": (f.n_events - cfg.min_events) / 200.0,
        "band": min(f.f0_hz - cfg.fmin_hz, cfg.fmax_hz - f.f0_hz) / 20.0
        if np.isfinite(f.f0_hz) else -np.inf,
    }


def gate_score(f: PeriodicityFeatures, cfg: GateConfig) -> Tuple[float, List[str]]:
    """Weakest-link score: the smallest margin across all tests. Returns
    (score, names of the failed tests).

    The minimum is used rather than a sum so that a large margin on one test
    cannot make up for failing another. A track that fails the wind test, for
    example, is rejected however strong its spectral line. Any test that
    fails with a margin of -inf makes the score -inf.
    """
    m = _margins(f, cfg)
    if cfg.require_octave_agreement and not f.octave_agreement:
        m["octave"] = -1.0
    finite = {k: v for k, v in m.items() if np.isfinite(v)}
    score = min(finite.values()) if finite else -np.inf
    if any(not np.isfinite(v) and v < 0 for v in m.values()):
        score = -np.inf
    reasons = [k for k, v in m.items() if v < 0]
    return float(score), reasons


class PeriodicityGate:
    """Accept or reject tracks on their wingbeat periodicity. Uses the
    default GateConfig unless one is given."""
    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def evaluate(self, features: PeriodicityFeatures) -> Tuple[bool, float, List[str]]:
        """(accepted, score, failed tests) for one track's features. Accepted
        means the weakest-link score is at least 0."""
        score, reasons = gate_score(features, self.config)
        return (score >= 0.0), score, reasons

    def apply(
        self,
        tracks: Sequence[Track],
        ev: EventStream,
        pad_px: int = 3,
        bin_us: int = 200,
        check_scene_coherence: bool = True,
    ) -> List[TrackResult]:
        """Measure periodicity features for every track and score each against
        the config. Returns one TrackResult per track, in input order.

        `ev` is the stream the tracks were built from. `bin_us` is the rate-signal
        bin in microseconds, and `pad_px` grows each detection box when picking
        a track's events. `check_scene_coherence=False` skips the lamp test and
        leaves `scene_coherence` at whatever analyse_periodicity set."""
        if len(ev) == 0 or not tracks:
            return []

        # Scene-wide signed rate, computed once. Each track's own contribution is
        # subtracted from it to form that track's background, which is exact
        # (rates are additive) and far cheaper than re-binning the complement of
        # every track mask.
        t0_us, t1_us = int(ev.t[0]), int(ev.t[-1]) + 1
        fs = 1e6 / bin_us
        scene_t, _, scene_signed = rate_signals(ev, bin_us=bin_us, t0_us=t0_us, t1_us=t1_us)

        results = []
        for trk in tracks:
            # The track's events: those inside any of its detection boxes,
            # grown by pad_px pixels, during that detection's time window.
            mask = trk.event_mask(ev, pad_px=pad_px)
            sub = ev.select(mask)
            # Spectral line, YIN, phase locking, harmonics and the rest, with
            # the fundamental searched only inside the config's band in Hz.
            feats = analyse_periodicity(
                sub, bin_us=bin_us, fmin_hz=self.config.fmin_hz, fmax_hz=self.config.fmax_hz
            )

            # Lamp test, only when there is an f0 to test at. The track's rate
            # is binned on the scene's own time base, so it lines up bin for bin
            # with scene_signed and the subtraction is exact.
            if check_scene_coherence and np.isfinite(feats.f0_hz):
                _, _, trk_full = rate_signals(sub, bin_us=bin_us, t0_us=t0_us, t1_us=t1_us)
                background = scene_signed - trk_full
                # Compare only the bins inside the track's lifetime. Outside it
                # the track's rate is zero and would carry no phase.
                i0 = max(int((trk.t_start_us - t0_us) // bin_us), 0)
                i1 = min(int((trk.t_end_us - t0_us) // bin_us) + 1, len(scene_signed))
                # At least 32 bins, 6.4 ms at the default 200 us bin. A shorter
                # track keeps the scene_coherence analyse_periodicity gave it.
                if i1 - i0 >= 32:
                    feats.scene_coherence = cross_phase_locking(
                        trk_full[i0:i1], background[i0:i1], fs, feats.f0_hz
                    )

            # Weakest-link score over every criterion; accepted when it is >= 0.
            # `reasons` names each criterion the track failed.
            accepted, score, reasons = self.evaluate(feats)
            results.append(TrackResult(trk, feats, accepted, score, reasons))
        return results
