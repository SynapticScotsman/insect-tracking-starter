"""Wingbeat periodicity: the clutter filter and the front end for identity (O3.1).

The claim the whole approach rests on: in a meadow, an insect is close to the
only thing that modulates a pixel periodically at 100-250 Hz. Wind-driven
foliage is aperiodic and band-limited below roughly 20 Hz. So the separable axis
is not brightness, size, speed or event count, all of which overlap heavily; it
is the presence of a stable spectral line with harmonic structure and a phase
that stays locked over many cycles.

Three measurements, deliberately not one:

  peak_snr_db      Is there a line in the band, above the local noise floor?
                   On real Münster recordings this is the STRONGEST of the three
                   (AUC 0.79 against size-matched background patches).
  harmonic_ratio   Does it have harmonics? A wingbeat is a non-sinusoidal
                   mechanical oscillation and always does. Narrowband
                   interference usually does not. Second strongest on real data
                   (AUC 0.75).
  plv              Is the phase locked while the animal is beating its wings?
                   Weakest on real data (AUC 0.64), for reasons documented in
                   `phase_locking_value`: a wild insect's wingbeat is
                   intermittent over a multi-second track, so this measure is
                   far less decisive outdoors than on synthetic scenes, where it
                   looked perfect. Corroborating evidence, not a gate.

Reporting all three, rather than a single fused score, is what lets you say
which property failed when a track is rejected in the field. It is also what
made the disagreement above visible: a single fused score would have hidden the
fact that the criterion trusted most was the one that transferred worst.

ON/OFF and the factor of two. A wing sweeps fastest at mid-stroke, so the
unsigned event rate peaks twice per cycle (energy at 2*f0), while polarity
reverses once per cycle (signed rate carries f0). `analyse_periodicity` estimates
both and cross-checks them, because taking the unsigned peak at face value gives
you an octave error, and an octave error in f0 is fatal to identity work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

from .events import EventStream

#: Default search band. Covers Hymenoptera and Diptera (roughly 100-250 Hz) with
#: headroom for slower Lepidoptera at the bottom and small Diptera at the top.
#: Narrow it to your taxa if you know them: a narrower band is a stronger filter.
DEFAULT_FMIN_HZ = 40.0
DEFAULT_FMAX_HZ = 400.0

#: Above this frequency, vegetation has essentially no power. Energy below it is
#: the signature of wind, not wings.
CLUTTER_FMAX_HZ = 20.0


@dataclass
class PeriodicityFeatures:
    """Everything `analyse_periodicity` measures about one track."""

    f0_hz: float = np.nan
    peak_snr_db: float = -np.inf
    peak_snr_signed_db: float = -np.inf
    """SNR at f0_signed_hz specifically, not at the reconciled f0_hz. peak_snr_db
    is evaluated at whatever `_reconcile_fundamental` settled on, which can be a
    YIN estimate; a countable-track gate on that number inherits every YIN
    corruption of the fundamental. This field never moves regardless of how f0
    was reconciled, so a gate built on it stays at the frequency the census
    actually reports (2026-08 review finding #4)."""
    harmonic_ratio: float = 0.0
    plv: float = 0.0
    low_freq_ratio: float = 1.0     # power below CLUTTER_FMAX_HZ / total power
    spectral_flatness: float = 1.0  # 1.0 = white, 0 = pure tone
    f0_signed_hz: float = np.nan
    f0_unsigned_hz: float = np.nan
    f0_yin_hz: float = np.nan
    yin_confidence: float = 0.0
    octave_agreement: bool = False  # unsigned peak sits at ~2x the signed peak
    octave_corrected: bool = False
    """True when the spectral peak was an integer multiple of the YIN period and
    was overridden. Worth counting across a dataset: a high rate means the
    spectral estimator alone would have been folding waveforms at the wrong
    rate."""

    fundamental_source: str = "none"
    """One of: yin, spectral, spectral+yin, unsigned/2, none."""
    scene_coherence: float = 0.0
    """Phase locking between this track and the rest of the scene at f0. High
    means a common driver, i.e. artificial light rather than an animal. Only
    populated when the gate is given scene context; 0 otherwise."""

    n_events: int = 0
    duration_s: float = 0.0
    n_cycles: float = 0.0
    freq_resolution_hz: float = np.nan
    polarity_balance: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def rate_signals(
    ev: EventStream,
    bin_us: int = 200,
    t0_us: Optional[int] = None,
    t1_us: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a track's events into unsigned and signed rate signals.

    Returns (t_s, unsigned, signed) where t_s is the bin centre in seconds.

    `bin_us` sets the Nyquist limit: 200 us gives fs = 5 kHz, so harmonics up to
    2.5 kHz are representable, which is ample for a 250 Hz fundamental. Do not
    coarsen much beyond 500 us or you start aliasing the third harmonic of a fast
    flier back into the band you are searching.
    """
    if len(ev) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    t0 = int(ev.t[0]) if t0_us is None else int(t0_us)
    t1 = int(ev.t[-1]) + 1 if t1_us is None else int(t1_us)
    n_bins = max(int(np.ceil((t1 - t0) / bin_us)), 1)
    idx = np.clip((ev.t - t0) // bin_us, 0, n_bins - 1).astype(np.int64)
    unsigned = np.bincount(idx, minlength=n_bins).astype(np.float64)
    signed = np.bincount(idx, weights=ev.p.astype(np.float64), minlength=n_bins)
    t_s = (t0 + (np.arange(n_bins) + 0.5) * bin_us) / 1e6
    return t_s, unsigned, signed


def _detrend_highpass(sig: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Remove the slow envelope from translation, occlusion and range change.

    Without this, a track that simply gets brighter as the insect approaches has
    a huge DC-adjacent component that dominates the periodogram and depresses the
    apparent SNR of the real line.
    """
    sig = sig - sig.mean()
    nyq = fs / 2.0
    wn = min(max(cutoff_hz / nyq, 1e-4), 0.99)
    if len(sig) < 30:
        return sig
    sos = sps.butter(2, wn, btype="high", output="sos")
    pad = min(len(sig) - 1, 3 * 6)
    try:
        return sps.sosfiltfilt(sos, sig, padlen=pad)
    except ValueError:  # signal shorter than the filter needs
        return sig


def _psd(sig: np.ndarray, fs: float, df_target_hz: float = 8.0):
    """Welch PSD with a segment length chosen for the requested resolution."""
    nperseg = int(min(len(sig), max(64, 2 ** int(np.ceil(np.log2(fs / df_target_hz))))))
    nperseg = max(nperseg, 32)
    if nperseg > len(sig):
        nperseg = len(sig)
    freqs, psd = sps.welch(
        sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        window="hann", detrend=False, scaling="density",
    )
    return freqs, psd, fs / nperseg


def _fine_peak(sig: np.ndarray, fs: float, fmin: float, fmax: float, zero_pad: int = 8):
    """Zero-padded FFT peak with parabolic interpolation, for a precise f0.

    Welch is the right tool for a robust noise floor but its resolution is set by
    the segment length. Individual identification needs f0 to a fraction of a
    hertz, so the peak location is refined here on the full-length transform.
    """
    n = len(sig)
    if n < 16:
        return np.nan, np.nan
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * zero_pad)))
    spec = np.abs(np.fft.rfft(sig * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return np.nan, np.nan
    sub = np.where(band)[0]
    k = sub[int(np.argmax(spec[sub]))]
    if 0 < k < len(spec) - 1:
        a, b, c = np.log(spec[k - 1] + 1e-30), np.log(spec[k] + 1e-30), np.log(spec[k + 1] + 1e-30)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    return float(freqs[k] + delta * df), float(spec[k])


def _line_snr_db(freqs, psd, f_line, exclude_hz=None, band=None) -> float:
    """Power at a spectral line relative to the median floor around it."""
    if not np.isfinite(f_line):
        return -np.inf
    if band is None:
        band = (freqs > 0)
    if exclude_hz is None:
        exclude_hz = max(3.0, 0.05 * f_line)
    near = np.abs(freqs - f_line) <= exclude_hz
    if not near.any():
        return -np.inf
    peak = float(psd[near].max())
    # Floor: everything in band that is not this line or one of its harmonics.
    floor_mask = band.copy()
    for h in range(1, 6):
        floor_mask &= np.abs(freqs - h * f_line) > exclude_hz
    if floor_mask.sum() < 4:
        floor_mask = band & ~near
    if floor_mask.sum() < 4:
        return -np.inf
    floor = float(np.median(psd[floor_mask]))
    if floor <= 0:
        return np.inf
    return float(10.0 * np.log10(peak / floor))


def _plv_segment(sig: np.ndarray, fs: float, f0: float, cycles: float) -> float:
    """Resultant length of the demodulated phase over one segment."""
    n = len(sig)
    if n < 16:
        return np.nan
    t = np.arange(n) / fs
    z = sig.astype(np.float64) * np.exp(-2j * np.pi * f0 * t)
    win = int(max(3, round(cycles * fs / f0)))
    if win >= n:
        win = max(3, n // 3)
    zs = uniform_filter1d(z.real, win) + 1j * uniform_filter1d(z.imag, win)
    edge = win // 2
    if n - 2 * edge > 8:
        zs = zs[edge:n - edge]
    mag = np.abs(zs)
    good = mag > (1e-12 + 0.05 * np.median(mag))
    if good.sum() < 8:
        return np.nan
    return float(np.abs(np.mean(zs[good] / mag[good])))


def phase_locking_value(
    sig: np.ndarray,
    fs: float,
    f0: float,
    cycles: float = 3.0,
    window_cycles: Optional[float] = 25.0,
) -> float:
    """Consistency of the instantaneous phase at f0, measured over short windows.

    Complex demodulation: multiply by exp(-2*pi*i*f0*t), smooth over a few cycles,
    then take the resultant length of the unit phasors.

    MEASURED ON REAL DATA, AND IT CHANGED THE DESIGN. Computing this across a
    whole track, as this function originally did, returns near zero for real
    insects that carry an obvious 25 dB spectral line: on Münster tracks the
    whole-track value was 0.03 to 0.23 while the same tracks scored 21 to 31 dB
    peak SNR. The cause is not frequency drift in the gentle sense. Across a
    multi-second track a wild insect's wingbeat is intermittent: it lands, turns,
    is occluded, or leaves the annotation box, and the phase reference is lost at
    every interruption. Integrating through those gaps converts a real oscillator
    into apparent noise.

    So the phase is measured over `window_cycles` at a time and the median is
    returned, which asks the answerable question ("is the phase locked while the
    animal is beating its wings") rather than the unanswerable one ("has it been
    locked continuously for eighteen seconds").

    A CAVEAT THAT MUST NOT BE DROPPED. Short windows are biased upward: with k
    independent smoothing windows inside a segment, uncorrelated noise still
    returns roughly 1/sqrt(k), around 0.3 at 25 cycles. Real insect medians sit
    near that same value, so on Münster data this measure separates targets from
    background far less well than peak SNR (AUC 0.64 against 0.79) despite the
    earlier claim in this file that it was the strongest criterion. It is not.
    Treat it as corroborating evidence, not as a gate on its own, and re-fit its
    threshold per dataset with `classify.fit_thresholds`.

    Set `window_cycles=None` to recover the old whole-track behaviour.
    """
    if not np.isfinite(f0) or f0 <= 0 or len(sig) < 16:
        return 0.0

    n = len(sig)
    if window_cycles is None:
        v = _plv_segment(sig, fs, f0, cycles)
        return 0.0 if not np.isfinite(v) else v

    wlen = int(round(window_cycles * fs / f0))
    if wlen < 32 or wlen >= n:
        v = _plv_segment(sig, fs, f0, cycles)
        return 0.0 if not np.isfinite(v) else v

    vals = [
        _plv_segment(sig[s:s + wlen], fs, f0, cycles)
        for s in range(0, n - wlen + 1, wlen)
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else 0.0


def autocorrelation(sig: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Unbiased-ish autocorrelation via FFT, lag 0 upward, normalised to r[0] = 1."""
    x = np.asarray(sig, np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return np.zeros(1)
    if max_lag is None:
        max_lag = n // 2
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, nfft)
    r = np.fft.irfft(spec * np.conj(spec), nfft)[: max_lag + 1]
    return r / r[0] if r[0] > 0 else r


def estimate_period_yin(
    sig: np.ndarray,
    fs: float,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
    threshold: float = 0.25,
) -> Tuple[float, float]:
    """Fundamental frequency by the YIN cumulative-mean-normalised difference.

    Returns (f0_hz, confidence in 0..1).

    WHY THIS EXISTS ALONGSIDE THE SPECTRAL PEAK. A wingbeat is a strongly
    non-sinusoidal oscillation, so its energy is spread across f0, 2*f0, 3*f0.
    Picking the largest peak in a periodogram picks whichever harmonic happens to
    dominate, which for a wing is often 2*f0 because the stroke fires twice per
    cycle. On real Münster targets the spectral estimator returned 359 Hz for
    what are almost certainly ~180 Hz insects: a clean octave error, and one that
    is fatal downstream because folding at twice the true rate scrambles every
    waveform feature used for identity.

    Autocorrelation inverts the problem. Harmonics *reinforce* the fundamental
    lag rather than competing with it, so the true period shows up as the first
    strong peak. The remaining trap is that autocorrelation also peaks at 2T, 3T,
    which would give a sub-octave error in the other direction. YIN's answer, and
    the reason to use YIN rather than plain autocorrelation, is the cumulative
    mean normalisation plus an absolute threshold: take the *first* lag that dips
    below the threshold, not the deepest one. That systematically prefers T over
    its multiples.

    `threshold` is YIN's aperiodicity tolerance. Lower is stricter; 0.1 suits
    clean laboratory signals, 0.25 is more forgiving and appropriate for a wild
    insect whose beat is intermittent.
    """
    x = np.asarray(sig, np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 64 or fmin_hz <= 0:
        return np.nan, 0.0

    tau_min = max(int(np.floor(fs / fmax_hz)), 2)
    tau_max = min(int(np.ceil(fs / fmin_hz)), n // 2 - 1)
    if tau_max <= tau_min + 2:
        return np.nan, 0.0

    # d(tau) = sum_{i<W} (x[i] - x[i+tau])^2 = m(0) + m(tau) - 2*r_W(tau).
    #
    # r_W must be the correlation of the FIRST W samples against the shifted
    # signal, NOT the full-length FFT autocorrelation. Mixing the two makes d
    # negative (the full-length correlation sums far more terms than the windowed
    # energies), which clamps to zero over wide lag ranges and produces a d'
    # that is identically zero at both T and T/2 -- i.e. an estimator that picks
    # whichever comes first, for no reason at all.
    w = min(n // 2, n - tau_max - 1)
    if w < 32:
        return np.nan, 0.0
    seg_len = w + tau_max + 1
    nfft = int(2 ** np.ceil(np.log2(seg_len + w)))
    fa = np.fft.rfft(x[:w], nfft)
    fb = np.fft.rfft(x[:seg_len], nfft)
    r_w = np.fft.irfft(np.conj(fa) * fb, nfft)[: tau_max + 1]

    cumsq = np.concatenate([[0.0], np.cumsum(x * x)])
    m0 = cumsq[w] - cumsq[0]
    idx = np.arange(tau_max + 1)
    m_tau = cumsq[idx + w] - cumsq[idx]
    d = m0 + m_tau - 2.0 * r_w
    d[d < 0] = 0.0

    # Cumulative mean normalisation: d'(tau) = d(tau) / mean(d(1..tau)).
    d_prime = np.ones_like(d)
    running = np.cumsum(d[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        d_prime[1:] = d[1:] * np.arange(1, len(d)) / running
    d_prime[~np.isfinite(d_prime)] = 1.0

    band = d_prime[tau_min:tau_max + 1]
    tau = -1
    for i in range(1, len(band) - 1):
        if band[i] < threshold and band[i] <= band[i - 1] and band[i] <= band[i + 1]:
            tau = tau_min + i
            break
    if tau < 0:
        tau = tau_min + int(np.argmin(band))

    # Sub-harmonic check: is the true period an integer multiple of what we took?
    #
    # YIN's first-dip rule protects against picking 2T when the period is T, but
    # not against picking T/2 when a harmonic dominates the waveform. A wing with
    # a second harmonic 2.5x stronger than its fundamental repeats every T, yet
    # d'(T/2) can still dip under the threshold because only the weak fundamental
    # differs across that lag.
    #
    # The discriminator is not "which dip is deeper" but "is the longer lag
    # *materially* better". A signal genuinely periodic at tau is equally
    # periodic at 2*tau, so both dips are near zero and we must keep tau. A
    # signal whose real period is 2*tau leaves a residual at tau, so d'(2*tau)
    # is much smaller. Requiring a factor-of-two improvement separates the two.
    for k in (2, 3):
        k_tau = tau * k
        if k_tau > tau_max:
            break
        if d_prime[k_tau] < 0.5 * d_prime[tau] and d_prime[k_tau] < threshold:
            tau = k_tau

    # Parabolic refinement of the dip location.
    if 0 < tau < len(d_prime) - 1:
        a, b, c = d_prime[tau - 1], d_prime[tau], d_prime[tau + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        tau_ref = tau + float(np.clip(shift, -0.5, 0.5))
    else:
        tau_ref = float(tau)

    if tau_ref <= 0:
        return np.nan, 0.0
    f0 = fs / tau_ref
    if not (fmin_hz * 0.95 <= f0 <= fmax_hz * 1.05):
        return np.nan, 0.0
    return float(f0), float(np.clip(1.0 - d_prime[tau], 0.0, 1.0))


def cross_phase_locking(
    sig_a: np.ndarray, sig_b: np.ndarray, fs: float, f0: float, cycles: float = 3.0
) -> float:
    """Is signal A phase-locked to signal B at f0?

    This is the discriminator against artificial light. Mains-driven lamps
    modulate at 100/120 Hz, inside the wingbeat band, and every surface they
    illuminate flickers in lockstep because they share a supply. So a flickering
    leaf edge is phase-locked to the rest of the scene, while an insect is
    phase-locked to nothing: its wingbeat has no fixed relationship to anything
    else in the field of view.

    Both signals are complex-demodulated at f0 and the resultant length of their
    phase *difference* is returned. Near 1 means driven by a common source;
    near 0 means independent.

    Bias warning: with only n independent smoothing windows in the track, an
    uncorrelated pair still returns roughly 1/sqrt(n). A 100 ms track at 200 Hz
    gives about 6 windows, so expect a floor near 0.4 on short tracks and read
    the threshold accordingly. Longer tracks make this test sharper.

    Note on what is deliberately NOT used here: track speed. A static source is
    tempting to reject as "not flying", but a foraging insect handling a flower
    is also static, and that is precisely the case SP3's fallback plan depends on
    being able to measure. Rejecting on motion would delete the observations that
    matter most.
    """
    n = min(len(sig_a), len(sig_b))
    if not np.isfinite(f0) or f0 <= 0 or n < 32:
        return 0.0
    t = np.arange(n) / fs
    carrier = np.exp(-2j * np.pi * f0 * t)
    win = int(max(3, round(cycles * fs / f0)))
    if win >= n:
        win = max(3, n // 3)

    def demod(sig):
        z = sig[:n].astype(np.float64) * carrier
        return uniform_filter1d(z.real, win) + 1j * uniform_filter1d(z.imag, win)

    za, zb = demod(sig_a), demod(sig_b)
    edge = win // 2
    if n - 2 * edge > 8:
        za, zb = za[edge:n - edge], zb[edge:n - edge]
    mag = np.abs(za) * np.abs(zb)
    good = mag > (1e-12 + 0.05 * np.median(mag))
    if good.sum() < 8:
        return 0.0
    rel = za[good] * np.conj(zb[good])
    return float(np.abs(np.mean(rel / np.abs(rel))))


def spectral_flatness(psd: np.ndarray) -> float:
    """Geometric over arithmetic mean. 1.0 is white noise, 0 is a pure tone."""
    p = psd[psd > 0]
    if p.size < 4:
        return 1.0
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def _reconcile_fundamental(
    f_spectral: float, f_unsigned: float, f_yin: float, yin_conf: float,
    agree_tol: float = 0.06, min_conf: float = 0.15, override_min_conf: float = 0.5,
) -> Tuple[float, str, bool]:
    """Combine the spectral peak and the YIN period into one fundamental.

    They answer different questions and fail differently. The spectral peak is
    precise but picks whichever harmonic is loudest; YIN identifies the true
    period but resolves it only to a sample lag. So:

    * If they agree, keep the spectral value for its precision.
    * If the spectral peak is close to an integer multiple of the YIN period,
      that is an octave error and YIN wins. This is the case that matters.
    * If YIN has no confidence, fall back to the spectral peak, then to half the
      unsigned peak, which is the last resort the ON/OFF physics allows.

    A YIN candidate may only OVERRIDE a finite spectral peak (the two branches
    above the "no confidence" fallback) when yin_confidence >= override_min_conf.
    Below that, `min_conf` alone let a barely-confident YIN estimate silently
    replace a clean spectral line: verified spec=180 Hz beaten by yin=158 Hz at
    conf 0.20 (2026-08 review finding #4). When the spectral peak is nan this
    gate does not apply -- YIN at >= min_conf is still the best available
    estimate, same as before.
    """
    have_yin = np.isfinite(f_yin) and yin_conf >= min_conf
    have_spec = np.isfinite(f_spectral)

    if have_spec and have_yin:
        if abs(f_spectral - f_yin) <= agree_tol * f_yin:
            return f_spectral, "spectral+yin", False
        if yin_conf >= override_min_conf:
            for k in (2, 3, 4):
                if abs(f_spectral - k * f_yin) <= agree_tol * k * f_yin:
                    return f_yin, "yin", True
            # Disagreement that is not a clean multiple: trust the period
            # estimate, which degrades more gracefully on intermittent signals.
            return f_yin, "yin", False
        return f_spectral, "spectral", False

    if have_yin:
        return f_yin, "yin", False
    if have_spec:
        return f_spectral, "spectral", False
    if np.isfinite(f_unsigned):
        return f_unsigned / 2.0, "unsigned/2", False
    return np.nan, "none", False


def analyse_periodicity(
    ev: EventStream,
    bin_us: int = 200,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
    detrend_hz: Optional[float] = None,
) -> PeriodicityFeatures:
    """Full periodicity analysis of one track's events.

    Pass the events belonging to a single track (see `Track.extract_events`).
    Passing a whole scene will average every source together and measure nothing.
    """
    feats = PeriodicityFeatures()
    if len(ev) == 0:
        return feats

    t_s, unsigned, signed = rate_signals(ev, bin_us=bin_us)
    fs = 1e6 / bin_us
    feats.n_events = len(ev)
    feats.duration_s = float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0
    feats.polarity_balance = float(ev.p.sum()) / len(ev)

    if len(unsigned) < 32:
        return feats

    if detrend_hz is None:
        detrend_hz = max(fmin_hz * 0.5, 15.0)
    u = _detrend_highpass(unsigned, fs, detrend_hz)
    s = _detrend_highpass(signed, fs, detrend_hz)

    # Fundamental candidates from each signal.
    f_signed, _ = _fine_peak(s, fs, fmin_hz, fmax_hz)
    f_unsigned, _ = _fine_peak(u, fs, fmin_hz, min(2 * fmax_hz, fs / 2 * 0.9))
    feats.f0_signed_hz = f_signed
    feats.f0_unsigned_hz = f_unsigned

    # Cross-check: for a real wingbeat the unsigned peak sits at twice the signed
    # peak. Agreement promotes the signed estimate; disagreement is a warning the
    # caller can act on rather than a silent octave error.
    octave_ok = (
        np.isfinite(f_signed) and np.isfinite(f_unsigned)
        and abs(f_unsigned - 2 * f_signed) < 0.12 * max(f_signed, 1.0)
    )
    feats.octave_agreement = bool(octave_ok)

    # Independent period estimate that is immune to which harmonic dominates.
    f_yin, yin_conf = estimate_period_yin(s, fs, fmin_hz, fmax_hz)
    if not np.isfinite(f_yin):
        f_yin, yin_conf = estimate_period_yin(u, fs, fmin_hz, fmax_hz)
    feats.f0_yin_hz = f_yin
    feats.yin_confidence = yin_conf

    f0, source, corrected = _reconcile_fundamental(f_signed, f_unsigned, f_yin, yin_conf)
    if not np.isfinite(f0):
        return feats
    feats.fundamental_source = source
    feats.octave_corrected = corrected
    feats.f0_hz = float(f0)
    feats.n_cycles = float(f0 * feats.duration_s)

    # Spectrum for SNR and harmonics: use the unsigned rate, which carries the
    # most event mass, and evaluate lines at f0 and its multiples.
    freqs, psd, df = _psd(u, fs)
    feats.freq_resolution_hz = float(df)
    band = (freqs >= fmin_hz * 0.5) & (freqs <= min(fmax_hz * 3, fs / 2))

    freqs_s, psd_s, _ = _psd(s, fs)
    band_s = (freqs_s >= fmin_hz * 0.5) & (freqs_s <= min(fmax_hz * 3, fs / 2))
    snr_signed = _line_snr_db(freqs_s, psd_s, f0, band=band_s)
    snr_unsigned = _line_snr_db(freqs, psd, 2 * f0, band=band)
    feats.peak_snr_db = float(max(snr_signed, snr_unsigned))
    # At f0_signed_hz itself, not the reconciled f0: _line_snr_db is already
    # nan-safe (returns -inf when the line frequency is non-finite).
    feats.peak_snr_signed_db = _line_snr_db(freqs_s, psd_s, f_signed, band=band_s)

    # Harmonic support: how much of the in-band power sits at multiples of f0.
    harm_power = 0.0
    for h in (1, 2, 3, 4):
        fh = h * f0
        if fh >= freqs[-1]:
            break
        near = np.abs(freqs - fh) <= max(2 * df, 0.03 * fh)
        if near.any():
            harm_power += float(psd[near].max())
    total = float(psd[band].sum() * df) if band.any() else 0.0
    feats.harmonic_ratio = float(harm_power * df / total) if total > 0 else 0.0

    feats.plv = phase_locking_value(s, fs, f0)

    # Vegetation signature: fraction of power below the wind ceiling. Computed on
    # the *undetrended* signal, since the high pass has already removed it.
    freqs_raw, psd_raw, _ = _psd(unsigned - unsigned.mean(), fs)
    low = freqs_raw <= CLUTTER_FMAX_HZ
    tot_raw = float(psd_raw.sum())
    feats.low_freq_ratio = float(psd_raw[low].sum() / tot_raw) if tot_raw > 0 else 1.0

    feats.spectral_flatness = spectral_flatness(psd[band])
    return feats


def instantaneous_frequency(
    ev: EventStream,
    window_us: int = 200_000,
    step_us: int = 20_000,
    bin_us: int = 200,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
    min_confidence: float = 0.45,
    continuity_tol: float = 0.15,
    reference_hz: float = np.nan,
    reference_tol: float = 0.25,
    min_events: int = 2000,
):
    """Wingbeat frequency in a sliding window, with rejected points returned.

    Returns (times_s, freqs_hz, rejected_times_s, rejected_freqs_hz). Everything
    discarded comes back rather than vanishing, because a trace that has been
    quietly cleaned looks identical to one that never had a problem.

    Two filters, and they reject different things:

    `min_confidence` drops windows where YIN could not find a period at all --
    the animal was between wingbeats, occluded, or out of the box.

    `continuity_tol` catches something more interesting: a window that returns a
    confident but wrong period. Measured on a real Münster target, one window in
    149 came back at almost exactly half the true rate at high YIN confidence --
    a confident sub-harmonic, which is the one error mode the octave guard in
    `estimate_period_yin` does not catch, because locally the signal really does
    look periodic at twice the period. The rejection is physiological rather than
    cosmetic: a wingbeat is driven by a resonant thoracic system and cannot
    change by tens of percent between windows 20 ms apart, so a point far from
    its local neighbourhood is an estimator failure, not a measurement.

    `reference_hz` closes the gap the local filter cannot. A median filter only
    sees a neighbourhood, so a *run* of consecutive sub-harmonic windows is
    self-consistent and survives -- which is exactly what happened at the start
    of one track, where several windows in a row sat near half rate and formed
    their own perfectly smooth little plateau. Passing the track-level estimate,
    computed from every event rather than from 60 ms of them, gives an anchor
    that a run cannot fake. The tolerance is deliberately wide, since insects do
    change wingbeat with load and manoeuvre; at 25 % it still cleanly excludes a
    factor-of-two error without touching plausible variation.
    """
    if len(ev) == 0:
        z = np.zeros(0)
        return z, z, z, z

    fs = 1e6 / bin_us
    t0, t1 = int(ev.t[0]), int(ev.t[-1])
    times, freqs = [], []
    rej_t, rej_f = [], []

    for centre in range(t0 + window_us // 2, t1 - window_us // 2 + 1, step_us):
        w = ev.time_slice(centre - window_us // 2, centre + window_us // 2)
        if len(w) < min_events:
            continue
        n_bins = max(int(window_us / bin_us), 16)
        idx = np.clip((w.t - (centre - window_us // 2)) // bin_us, 0, n_bins - 1)
        sig = np.bincount(idx.astype(np.int64), weights=w.p.astype(np.float64),
                          minlength=n_bins)
        sig = _detrend_highpass(sig, fs, max(fmin_hz * 0.5, 15.0))
        f, conf = estimate_period_yin(sig, fs, fmin_hz, fmax_hz)
        if not np.isfinite(f):
            continue
        if conf < min_confidence:
            rej_t.append((centre - t0) / 1e6)
            rej_f.append(f)
            continue
        times.append((centre - t0) / 1e6)
        freqs.append(f)

    times, freqs = np.array(times), np.array(freqs)

    if np.isfinite(reference_hz) and reference_hz > 0 and reference_tol > 0 and len(freqs):
        keep = np.abs(freqs - reference_hz) <= reference_tol * reference_hz
        rej_t.extend(times[~keep].tolist())
        rej_f.extend(freqs[~keep].tolist())
        times, freqs = times[keep], freqs[keep]

    if len(freqs) >= 7 and continuity_tol > 0:
        from scipy.ndimage import median_filter

        local = median_filter(freqs, size=5, mode="nearest")
        keep = np.abs(freqs - local) <= continuity_tol * local
        rej_t.extend(times[~keep].tolist())
        rej_f.extend(freqs[~keep].tolist())
        times, freqs = times[keep], freqs[keep]

    order = np.argsort(rej_t) if rej_t else np.zeros(0, int)
    return times, freqs, np.array(rej_t)[order], np.array(rej_f)[order]


def phase_track(
    ev: EventStream,
    f0_hz: float,
    bin_us: int = 200,
    lowpass_hz: float = 8.0,
    detrend_hz: float = 10.0,
    t0_us: Optional[int] = None,
    t1_us: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wingbeat phase over time, in unwrapped cycles, by complex demodulation.

    Returns (t_us, cycles) on the rate-signal grid; read an event's phase with
    np.interp(ev.t, t_us, cycles) % 1. Phase 0 is arbitrary.

    WHY. Epoch folding at one f0 assumes the wingbeat holds still, and a moth's
    does not. On recording_2026-10-02_21-38-40 track 71, f(t) over 100 ms ran
    35.8/42.2/45.5 Hz (p5/p50/p95), so a 40-stroke fold at one f0 walked 1.05
    cycles out of phase inside its own window: the folded stroke averaged every
    wing position into every frame. Following the phase instead of assuming it
    is what phase-locked averaging in electrophysiology does.

    HOW. The signed rate (which carries f0; the unsigned need not, see
    half_cycle_similarity) is demodulated at f0 and low-passed at `lowpass_hz`,
    so the residual angle follows any frequency within f0 +/- lowpass_hz; that
    residual is added to the nominal 2*pi*f0*t. Zero-phase filtering, so no lag.

    MEASURED (same track, cross-validated: phase estimated from a random half of
    the events, stroke images built from the other half): stroke contrast at
    N=40 cycles 0.027 with one f0, 0.089 phase-tracked, -0.004 for an
    off-frequency control; phase tracking won in 13 of 14 windows with N=5-40.
    Split-half phase noise 0.01-0.04 cycle. 8 Hz is what was measured; it is not
    an optimum.
    """
    t0 = int(ev.t[0]) if t0_us is None else int(t0_us)
    t1 = int(ev.t[-1]) + 1 if t1_us is None else int(t1_us)
    fs = 1e6 / bin_us
    t_s, _u, s = rate_signals(ev, bin_us=bin_us, t0_us=t0, t1_us=t1)
    s = _detrend_highpass(s, fs, detrend_hz)
    tt = t_s - t0 / 1e6
    z = s * np.exp(-2j * np.pi * f0_hz * tt)
    b, a = sps.butter(3, lowpass_hz / (fs / 2))
    if len(z) > 3 * max(len(a), len(b)):
        z = sps.filtfilt(b, a, z.real) + 1j * sps.filtfilt(b, a, z.imag)
    resid = np.unwrap(np.angle(z)) / (2 * np.pi)
    return t_s * 1e6, tt * f0_hz + resid


def fold_waveform(
    t_s: np.ndarray, sig: np.ndarray, f0: float, n_phase_bins: int = 32,
    phase_cycles: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-fold a rate signal at f0 (epoch folding).

    Returns (phase_centres, mean_waveform, cycle_matrix) where cycle_matrix is
    (n_cycles, n_phase_bins) with NaN for empty bins. The mean waveform is the
    per-individual stroke shape that O3.3 tests; the cycle matrix is what lets
    you ask how repeatable that shape is within a single track, which is the
    lower bound on any between-individual claim.

    `phase_cycles`, if given, is each sample's phase in unwrapped cycles (from
    `phase_track`) and replaces the fixed clock (t - t0) * f0, so a wingbeat
    that wanders is folded on its own beat.
    """
    if not np.isfinite(f0) or f0 <= 0 or len(sig) < 8:
        z = np.zeros(n_phase_bins)
        return np.linspace(0, 1, n_phase_bins, endpoint=False), z, np.zeros((0, n_phase_bins))

    if phase_cycles is None:
        rel = (t_s - t_s[0]) * f0
    else:
        # Floor at the minimum, not the first sample: a tracked phase may dip
        # below its starting value, and a negative cycle index would wrap.
        pc = np.asarray(phase_cycles, np.float64)
        rel = pc - np.floor(pc.min())
    cycle = np.floor(rel).astype(np.int64)
    phase_bin = np.clip(((rel - cycle) * n_phase_bins).astype(np.int64), 0, n_phase_bins - 1)
    n_cycles = int(cycle.max()) + 1

    acc = np.zeros((n_cycles, n_phase_bins))
    cnt = np.zeros((n_cycles, n_phase_bins))
    np.add.at(acc, (cycle, phase_bin), sig)
    np.add.at(cnt, (cycle, phase_bin), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cycles = np.where(cnt > 0, acc / cnt, np.nan)

    with warnings_suppressed():
        mean_wave = np.nanmean(cycles, axis=0)
    mean_wave = np.nan_to_num(mean_wave)
    phase = (np.arange(n_phase_bins) + 0.5) / n_phase_bins
    return phase, mean_wave, cycles


def half_cycle_similarity(ev: EventStream, f_fold: float, n_phase_bins: int = 32) -> float:
    """Octave arbiter: fold the signed rate at f_fold and correlate its halves.

    Call it at HALF the periodogram frequency. One folded cycle then holds two
    candidate strokes. If the periodogram line is the true wingbeat, the two
    halves are the same stroke twice and correlate near the no-doubling value of
    a control group; if half of it is the true wingbeat, they are an up-stroke
    and a down-stroke, which fire different events, and correlate lower.

    The statistic has no absolute threshold. Read it against a control group
    whose estimators already agree (honey bees in analysis/visapp_arbitrate.py,
    where this was written and from which it is promoted unchanged) and against
    an off-period fold at f_fold * 1.37, which is a subharmonic of nothing and
    shows what folding alone produces. On VISAPP that arbitration settled the
    bumble-bee doubling: the periodogram read the second harmonic on 23 of 44.

    Use raw track events. Coincidence filters censor stroke phase (Conv1's
    discards sit 123 degrees away in the cycle, wf8_taps.py), and this is a
    phase statistic.
    """
    if not np.isfinite(f_fold) or f_fold <= 0:
        return np.nan
    t_s, _unsigned, signed = rate_signals(ev, bin_us=200)
    if len(signed) < 64:
        return np.nan
    _ph, wave, _cyc = fold_waveform(t_s, signed, f_fold, n_phase_bins=n_phase_bins)
    h = n_phase_bins // 2
    a1, a2 = wave[:h], wave[h:]
    if np.std(a1) < 1e-12 or np.std(a2) < 1e-12:
        return np.nan
    return float(np.corrcoef(a1, a2)[0, 1])


class warnings_suppressed:
    """Silence the all-NaN-slice warning from nanmean on sparse folds."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False


def lomb_scargle_peak(
    t_us: np.ndarray, fmin_hz: float = DEFAULT_FMIN_HZ, fmax_hz: float = DEFAULT_FMAX_HZ,
    n_freqs: int = 2000,
) -> Tuple[float, float]:
    """Periodicity straight from event timestamps, with no binning.

    For sparse tracks (a distant target, or a few hundred events) binning throws
    away most of the timing information the sensor gave you. Lomb-Scargle on the
    raw arrival times keeps it. Slower, so it is not the default path.

    Returns (f_peak_hz, normalised_power).
    """
    if len(t_us) < 16:
        return np.nan, 0.0
    t = (t_us - t_us[0]).astype(np.float64) / 1e6
    y = np.ones_like(t)
    y = y - y.mean()
    if np.allclose(y, 0):
        # Uniform weights carry no amplitude information; use the local inter-arrival
        # rate as the observable instead.
        dt = np.diff(t, prepend=t[0])
        y = 1.0 / np.maximum(dt, 1e-9)
        y = y - y.mean()
    freqs = np.linspace(fmin_hz, fmax_hz, n_freqs)
    power = sps.lombscargle(t, y, 2 * np.pi * freqs, normalize=True)
    k = int(np.argmax(power))
    return float(freqs[k]), float(power[k])
