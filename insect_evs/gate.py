"""PeriodicityGate and its config, lifted verbatim from
src/insect_evs/classify.py (export_starter.py). classify.py is not copied
whole: it imports the dataset-annotation package for label scoring."""
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
    """Default thresholds.

    These were chosen on synthetic scenes to sit in the *gap* between insect and
    clutter distributions with roughly a factor of two of margin on each side,
    not at the edge of either. They are a starting point and nothing more: real
    scenes will have lower SNR, and the honest workflow is to record labelled
    field tracks and call `fit_thresholds`. Check `bench` across seeds before
    trusting any change to these.
    """

    min_snr_db: float = 10.0
    """Line must stand this far above the local noise floor. A wingbeat at close
    range gives 25-35 dB on synthetic scenes; in-band clutter transients rarely
    exceed 11 dB. Lower this for distant or low-contrast targets, but note that
    below about 6 dB the f0 estimate is too loose for identity work anyway."""

    min_plv: float = 0.50
    """DOES NOT TRANSFER TO REAL DATA AT THIS VALUE. Measured on Münster
    recordings, real insect tracks carrying 21-31 dB spectral lines score PLV
    around 0.2-0.3, so this threshold rejects almost every genuine target. It was
    calibrated on synthetic scenes where the simulated insect is a perfect
    oscillator and scores 1.00. Re-fit before using outdoors, and see
    `periodicity.phase_locking_value` for why the measure behaves differently on
    wild animals than on a simulation."""

    min_harmonic_ratio: float = 0.12
    """A wing is a non-sinusoidal mechanical oscillator and always has harmonics.
    Clutter transients and single-tone interference generally do not."""

    max_scene_coherence: float = 0.60
    """Reject tracks phase-locked to the rest of the scene at their own f0. This
    is the artificial-light test: mains flicker at 100/120 Hz sits inside the
    wingbeat band and passes every other criterion here, so without this check a
    lit hedge is indistinguishable from a swarm. See
    `periodicity.cross_phase_locking` for why the threshold cannot be pushed much
    below 0.5 on short tracks."""

    max_low_freq_ratio: float = 0.85
    """Reject tracks whose power is overwhelmingly below 20 Hz: that is wind."""

    min_cycles: float = 20.0
    """Twenty cycles is roughly 100 ms at 200 Hz. Below that, both the frequency
    estimate and the phase-locking value become unreliable in the same direction
    (both look better than they are), so short clutter fragments are the dominant
    false positive. This is the threshold that removes them."""

    min_events: int = 200
    fmin_hz: float = 40.0
    fmax_hz: float = 400.0

    require_octave_agreement: bool = False
    """Strict mode: also demand that the unsigned peak sit at 2*f0. High precision,
    but it costs recall on tracks viewed along the stroke plane, where the ON/OFF
    asymmetry that carries the fundamental largely cancels."""


@dataclass
class TrackResult:
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
    """Normalised distance past each threshold. Negative means the test failed.

    The scales in the denominators set how these trade off against each other in
    the combined score; they are rough units of "one meaningful step" in each
    quantity, not fitted values.
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
    """Weakest-link score: the smallest margin across all criteria.

    Taking the minimum rather than a sum means a track cannot buy its way past a
    failed criterion with a large margin elsewhere, which is exactly the
    behaviour you want when one criterion encodes "this is not wind".
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
    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def evaluate(self, features: PeriodicityFeatures) -> Tuple[bool, float, List[str]]:
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
            mask = trk.event_mask(ev, pad_px=pad_px)
            sub = ev.select(mask)
            feats = analyse_periodicity(
                sub, bin_us=bin_us, fmin_hz=self.config.fmin_hz, fmax_hz=self.config.fmax_hz
            )

            if check_scene_coherence and np.isfinite(feats.f0_hz):
                _, _, trk_full = rate_signals(sub, bin_us=bin_us, t0_us=t0_us, t1_us=t1_us)
                background = scene_signed - trk_full
                i0 = max(int((trk.t_start_us - t0_us) // bin_us), 0)
                i1 = min(int((trk.t_end_us - t0_us) // bin_us) + 1, len(scene_signed))
                if i1 - i0 >= 32:
                    feats.scene_coherence = cross_phase_locking(
                        trk_full[i0:i1], background[i0:i1], fs, feats.f0_hz
                    )

            accepted, score, reasons = self.evaluate(feats)
            results.append(TrackResult(trk, feats, accepted, score, reasons))
        return results
