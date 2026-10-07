"""Wingbeat periodicity of one insect track.

The method rests on one observation. In a field scene, a flying insect is close
to the only thing that modulates a pixel periodically at 100-250 Hz. Foliage
moved by wind is aperiodic and carries little power above about 20 Hz.
Brightness, size, speed and event count all overlap heavily between insects and
clutter. What separates them is a stable spectral line in the wingbeat band,
with harmonics, whose phase stays locked over many cycles.

The module reports three measurements of that, kept separate:

  peak_snr_db      Is there a line in the band, above the local noise floor?
                   The strongest of the three on field recordings: AUC 0.79
                   against size-matched background patches. AUC is the area
                   under the ROC curve; 0.5 is chance and 1.0 is perfect.
  harmonic_ratio   Does the line have harmonics? A wingbeat is a non-sinusoidal
                   mechanical oscillation and always does. Narrowband
                   interference usually does not. AUC 0.75.
  plv              Is the phase locked while the animal is beating its wings?
                   The weakest, AUC 0.64, because a wild insect's wingbeat is
                   intermittent over a track of several seconds. See
                   `phase_locking_value`. Treat it as supporting evidence, not
                   as a gate on its own.

Three numbers instead of one fused score means a rejected track can be traced
to the property that failed.

ON/OFF polarity and the factor of two. A wing sweeps fastest at mid-stroke, in
both directions, so the unsigned event rate peaks twice per wing cycle and
carries most of its energy at 2*f0. Polarity reverses once per cycle, so the
signed rate, ON count minus OFF count, carries f0 itself. Taking the unsigned
peak at face value gives an octave error: a frequency twice the true wingbeat.
An octave error makes any comparison between individuals meaningless, so
`analyse_periodicity` estimates both and cross-checks them.

The bundled pipeline calls `rate_signals`, `analyse_periodicity`,
`cross_phase_locking`, `phase_track` and `half_cycle_similarity`.
`instantaneous_frequency`, `autocorrelation` and `lomb_scargle_peak` are not
called by it and are kept as stand-alone tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

from .events import EventStream

#: Default search band in Hz. Covers bees, wasps and flies (roughly 100-250 Hz)
#: with room for slower moths and butterflies at the bottom and small flies at
#: the top. Narrow it to your taxa if you know them: a narrower band rejects
#: more clutter.
DEFAULT_FMIN_HZ = 40.0
DEFAULT_FMAX_HZ = 400.0

#: Vegetation moved by wind has almost no power above this frequency, in Hz.
#: Power below it is the signature of wind, not wings.
CLUTTER_FMAX_HZ = 20.0


@dataclass
class PeriodicityFeatures:
    """Everything `analyse_periodicity` measures about one track.

    Frequencies are in Hz, SNRs in dB, durations in seconds. Defaults are the
    values meaning "nothing measured": nan frequencies, -inf SNR, zero locking.
    """

    f0_hz: float = np.nan
    peak_snr_db: float = -np.inf
    peak_snr_signed_db: float = -np.inf
    """Line strength in dB at f0_signed_hz itself. peak_snr_db is measured at
    f0_hz, which may have come from YIN; this one always tests the spectral
    line, so a threshold on it judges the frequency the census reports."""
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
    """True when the spectral peak was a whole multiple of the YIN frequency
    and YIN's value replaced it in f0_hz."""

    fundamental_source: str = "none"
    """Which estimate f0_hz came from: yin, spectral, spectral+yin,
    unsigned/2, or none."""
    scene_coherence: float = 0.0
    """Phase locking between this track and the rest of the scene at f0. High
    means a shared driver such as a flickering lamp rather than an animal.
    This package does not compute it, so it stays 0."""

    n_events: int = 0
    duration_s: float = 0.0
    n_cycles: float = 0.0            # wing cycles in the track: f0_hz * duration_s
    freq_resolution_hz: float = np.nan
    polarity_balance: float = 0.0   # (ON - OFF) / all events, from -1 to +1

    def to_dict(self) -> Dict:
        return asdict(self)


def rate_signals(
    ev: EventStream,
    bin_us: int = 200,
    t0_us: Optional[int] = None,
    t1_us: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a track's events into unsigned and signed rate signals.

    Returns (t_s, unsigned, signed). t_s is each bin centre in seconds.
    unsigned is the event count per bin. signed is ON count minus OFF count
    per bin. t0_us and t1_us fix the time span in microseconds, so several
    tracks can share one grid; by default the span is the track's own.

    `bin_us` sets the sample rate and so the Nyquist limit. 200 us gives
    fs = 5 kHz, so frequencies up to 2.5 kHz are representable, ample for a
    250 Hz fundamental and its harmonics. Do not coarsen much beyond 500 us,
    or the third harmonic of a fast flier aliases back into the search band.
    """
    if len(ev) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    t0 = int(ev.t[0]) if t0_us is None else int(t0_us)
    t1 = int(ev.t[-1]) + 1 if t1_us is None else int(t1_us)
    n_bins = max(int(np.ceil((t1 - t0) / bin_us)), 1)
    # Events outside [t0, t1) are clipped into the first or last bin.
    idx = np.clip((ev.t - t0) // bin_us, 0, n_bins - 1).astype(np.int64)
    unsigned = np.bincount(idx, minlength=n_bins).astype(np.float64)
    signed = np.bincount(idx, weights=ev.p.astype(np.float64), minlength=n_bins)
    t_s = (t0 + (np.arange(n_bins) + 0.5) * bin_us) / 1e6
    return t_s, unsigned, signed


def _detrend_highpass(sig: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Remove the slow envelope caused by translation, occlusion and range change.

    Without this, a track that simply gets brighter as the insect approaches has
    a large component near 0 Hz that dominates the spectrum and lowers the
    apparent SNR of the real line. The filter is a 2nd-order Butterworth high
    pass at `cutoff_hz`, run forwards and backwards so it adds no phase lag.
    Signals shorter than 30 samples only have their mean removed.
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
    """Welch power spectral density at about `df_target_hz` resolution.

    The segment length is the power of two nearest fs / df_target_hz, at least
    32 samples and at most the signal length. Averaging half-overlapping
    segments gives a stable noise floor. Returns (freqs_hz, psd, resolution_hz).
    """
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
    """Strongest frequency in [fmin, fmax] Hz, located to a fraction of a bin.

    The Welch spectrum gives a stable noise floor, but its resolution is set by
    the segment length, about 8 Hz here. Comparing individuals needs f0 to a
    fraction of a hertz, so the peak is found on a Hann-windowed FFT of the
    whole signal, zero-padded `zero_pad` times, then refined by fitting a
    parabola to the log power of the peak bin and its two neighbours.

    Returns (f_peak_hz, power at the peak bin), or (nan, nan) if the signal is
    shorter than 16 samples or the band holds no bins.
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
    # Vertex of the parabola through three log-power points, as an offset in
    # bins from k, limited to half a bin either way.
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
    """Power at a spectral line relative to the median floor around it, in dB."""
    # No line to measure scores -inf, so it fails any SNR threshold.
    if not np.isfinite(f_line):
        return -np.inf
    # Default floor region: every positive frequency, which leaves out the DC bin.
    if band is None:
        band = (freqs > 0)
    # Half-width of the line in Hz: 3 Hz or 5% of the line, whichever is wider.
    # It sets both where the peak is looked for and what is kept out of the floor.
    if exclude_hz is None:
        exclude_hz = max(3.0, 0.05 * f_line)
    # The peak is the highest PSD bin within exclude_hz of the line, so a line
    # that falls between bins still finds its strongest bin.
    near = np.abs(freqs - f_line) <= exclude_hz
    if not near.any():
        return -np.inf
    peak = float(psd[near].max())
    # Floor: everything in band that is not this line or one of its harmonics.
    floor_mask = band.copy()
    # Harmonics 1 to 5 are removed, each with the same half-width.
    for h in range(1, 6):
        floor_mask &= np.abs(freqs - h * f_line) > exclude_hz
    # On a coarse frequency grid the harmonic cut can leave almost nothing.
    # Then fall back to removing the line alone, and give up below 4 bins.
    if floor_mask.sum() < 4:
        floor_mask = band & ~near
    if floor_mask.sum() < 4:
        return -np.inf
    # The median, not the mean, so a few other strong lines in the band move
    # the floor little.
    floor = float(np.median(psd[floor_mask]))
    if floor <= 0:
        return np.inf
    # PSD is power, so the ratio converts to dB with 10 log10.
    return float(10.0 * np.log10(peak / floor))


def _plv_segment(sig: np.ndarray, fs: float, f0: float, cycles: float) -> float:
    """Phase locking at f0 over one segment, from 0 (random) to 1 (locked)."""
    n = len(sig)
    if n < 16:
        return np.nan
    # Complex demodulation. Multiplying by exp(-2 pi i f0 t) moves the f0
    # component to 0 Hz, so its phase relative to a fixed f0 reference shows
    # up as the angle of z. Time t is in seconds from the segment start.
    t = np.arange(n) / fs
    z = sig.astype(np.float64) * np.exp(-2j * np.pi * f0 * t)
    # Smoothing window: `cycles` periods of f0, in samples, at least 3.
    # If that is longer than the segment, use a third of the segment instead.
    win = int(max(3, round(cycles * fs / f0)))
    if win >= n:
        win = max(3, n // 3)
    # A moving average of the real and imaginary parts is a low-pass filter.
    # It suppresses the image at twice f0 and leaves the slowly changing
    # amplitude and phase of the f0 component.
    zs = uniform_filter1d(z.real, win) + 1j * uniform_filter1d(z.imag, win)
    # Drop half a window at each end, where the average runs past the data.
    # Kept only if more than 8 samples remain.
    edge = win // 2
    if n - 2 * edge > 8:
        zs = zs[edge:n - edge]
    # Samples with almost no f0 amplitude, under 5% of the median, have no
    # meaningful phase and are left out.
    mag = np.abs(zs)
    good = mag > (1e-12 + 0.05 * np.median(mag))
    if good.sum() < 8:
        return np.nan
    # Mean of the unit phasors. 1 means the phase held constant over the
    # segment; near 0 means it wandered all the way round.
    return float(np.abs(np.mean(zs[good] / mag[good])))


def phase_locking_value(
    sig: np.ndarray,
    fs: float,
    f0: float,
    cycles: float = 3.0,
    window_cycles: Optional[float] = 25.0,
) -> float:
    """How steady the phase at f0 is, as the median over short windows.

    Returns a value from 0 (phase random) to 1 (phase fixed). Each window of
    `window_cycles` wing cycles is complex-demodulated at f0, smoothed over
    `cycles` cycles, and scored as the resultant length of its unit phasors.
    See `_plv_segment`.

    Why short windows. Over a track of several seconds a wild insect's
    wingbeat is intermittent: it lands, turns, is occluded, or leaves the
    tracked box, and the phase reference is lost at each interruption.
    Integrated over the whole track, insects with an obvious 20-30 dB spectral
    line score only 0.03 to 0.23. A window of 25 cycles asks whether the phase
    is locked while the animal is beating its wings, which is the question
    that can be answered.

    Bias. Short windows push the value up. With k independent smoothing
    windows in a segment, uncorrelated noise still scores about 1/sqrt(k),
    around 0.3 at 25 cycles, and real insects score near that same value. As a
    result this measure separates insects from background less well than
    peak SNR. Use it as supporting evidence, not as a gate on its own, and fit
    its threshold to your own data.

    `window_cycles=None` measures the whole signal as one segment.
    """
    if not np.isfinite(f0) or f0 <= 0 or len(sig) < 16:
        return 0.0

    n = len(sig)
    if window_cycles is None:
        v = _plv_segment(sig, fs, f0, cycles)
        return 0.0 if not np.isfinite(v) else v

    # Window length in samples. Too short to demodulate, or longer than the
    # signal: score the whole signal as one segment instead.
    wlen = int(round(window_cycles * fs / f0))
    if wlen < 32 or wlen >= n:
        v = _plv_segment(sig, fs, f0, cycles)
        return 0.0 if not np.isfinite(v) else v

    # Non-overlapping windows; a leftover tail shorter than wlen is ignored.
    vals = [
        _plv_segment(sig[s:s + wlen], fs, f0, cycles)
        for s in range(0, n - wlen + 1, wlen)
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else 0.0


def autocorrelation(sig: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Autocorrelation by FFT for lags 0 to max_lag samples, scaled so r[0] = 1.

    The mean is removed first. Each lag is a plain sum, not divided by its
    overlap length, so values shrink toward long lags. max_lag defaults to half
    the signal. Not called by the bundled pipeline.
    """
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
    """Fundamental frequency by YIN, from the waveform's period.

    Returns (f0_hz, confidence from 0 to 1), or (nan, 0.0) when no period in
    [fmin_hz, fmax_hz] is found or the signal is under 64 samples. YIN is the
    pitch estimator of de Cheveigne and Kawahara (2002).

    Why this exists alongside the spectral peak. A wingbeat is strongly
    non-sinusoidal, so its energy is spread over f0, 2*f0, 3*f0 and so on. The
    largest peak in a spectrum is whichever harmonic happens to dominate, and
    for a wing that is often 2*f0. A spectral estimate can therefore report
    about 360 Hz for an insect beating at 180 Hz. Folding the signal at twice
    the true rate then mixes the two half-strokes and ruins any stroke-shape
    feature.

    A period estimate avoids this. Harmonics repeat at the fundamental period
    too, so they reinforce it instead of competing with it. The remaining risk
    is that the signal also repeats at 2T and 3T, which would give an error of
    a factor of two in the other direction. YIN handles that with a cumulative
    mean normalisation and an absolute threshold: it takes the first lag whose
    normalised difference dips below the threshold, not the deepest one, and so
    prefers T over its multiples.

    `threshold` is YIN's tolerance for aperiodicity. Lower is stricter. 0.1
    suits clean laboratory signals; 0.25 suits a wild insect whose beat comes
    and goes.

    Warning: on large insects with a very strong second harmonic, compare this
    estimate with the spectral peak before quoting a frequency. The two can
    disagree by a factor of two, and YIN is usually the one to trust when the
    track spans many cycles.
    """
    x = np.asarray(sig, np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 64 or fmin_hz <= 0:
        return np.nan, 0.0

    # Search lags in samples: the shortest period is fs / fmax, the longest
    # fs / fmin, and the lag may not exceed half the signal.
    tau_min = max(int(np.floor(fs / fmax_hz)), 2)
    tau_max = min(int(np.ceil(fs / fmin_hz)), n // 2 - 1)
    if tau_max <= tau_min + 2:
        return np.nan, 0.0

    # Difference function over a window of W samples:
    #   d(tau) = sum_{i<W} (x[i] - x[i+tau])^2 = m(0) + m(tau) - 2*r_W(tau),
    # where m(tau) is the energy of the W samples starting at tau.
    #
    # r_W must be the correlation of the FIRST W samples against the shifted
    # signal, computed here by FFT. A full-length autocorrelation sums more
    # terms than the windowed energies, which drives d negative. After clamping
    # to zero, d' would then be zero at both T and T/2 and the estimator could
    # not tell them apart.
    w = min(n // 2, n - tau_max - 1)
    if w < 32:
        return np.nan, 0.0
    seg_len = w + tau_max + 1
    nfft = int(2 ** np.ceil(np.log2(seg_len + w)))
    fa = np.fft.rfft(x[:w], nfft)
    fb = np.fft.rfft(x[:seg_len], nfft)
    r_w = np.fft.irfft(np.conj(fa) * fb, nfft)[: tau_max + 1]

    # Windowed energies from a running sum of squares.
    cumsq = np.concatenate([[0.0], np.cumsum(x * x)])
    m0 = cumsq[w] - cumsq[0]
    idx = np.arange(tau_max + 1)
    m_tau = cumsq[idx + w] - cumsq[idx]
    d = m0 + m_tau - 2.0 * r_w
    d[d < 0] = 0.0

    # Cumulative mean normalisation: d'(tau) = d(tau) / mean(d(1..tau)).
    # d'(0) is defined as 1. Lags where the mean is zero are also set to 1.
    d_prime = np.ones_like(d)
    running = np.cumsum(d[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        d_prime[1:] = d[1:] * np.arange(1, len(d)) / running
    d_prime[~np.isfinite(d_prime)] = 1.0

    # Absolute threshold: the first local minimum of d' below `threshold`.
    # If none dips that low, take the overall minimum in the search range.
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
    # The test is not "which dip is deeper" but "is the longer lag clearly
    # better". A signal truly periodic at tau is equally periodic at 2*tau, so
    # both dips are near zero and tau is kept. A signal whose real period is
    # 2*tau leaves a residual at tau, so d'(2*tau) is much smaller. Requiring a
    # factor-of-two improvement separates the two cases.
    for k in (2, 3):
        k_tau = tau * k
        if k_tau > tau_max:
            break
        if d_prime[k_tau] < 0.5 * d_prime[tau] and d_prime[k_tau] < threshold:
            tau = k_tau

    # Parabolic refinement of the dip location, to a fraction of a sample.
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
    # Allow 5% outside the band for the refinement step, reject anything further.
    if not (fmin_hz * 0.95 <= f0 <= fmax_hz * 1.05):
        return np.nan, 0.0
    # Confidence is 1 - d'(tau): a perfect repeat gives d' = 0 and confidence 1.
    return float(f0), float(np.clip(1.0 - d_prime[tau], 0.0, 1.0))


def cross_phase_locking(
    sig_a: np.ndarray, sig_b: np.ndarray, fs: float, f0: float, cycles: float = 3.0
) -> float:
    """Is signal A phase-locked to signal B at f0? Returns 0 to 1.

    This is the test against artificial light. Lamps on mains power modulate
    at 100 or 120 Hz, inside the wingbeat band, and every surface they light
    flickers in step because they share a supply. A flickering leaf edge is
    therefore phase-locked to the rest of the scene. An insect is phase-locked
    to nothing: its wingbeat has no fixed relationship to anything else in view.

    Both signals are complex-demodulated at f0, smoothed over `cycles` cycles,
    and the resultant length of their phase difference is returned. Near 1
    means a common driver; near 0 means independent.

    Bias: with n independent smoothing windows in the track, an uncorrelated
    pair still scores about 1/sqrt(n). A 100 ms track at 200 Hz holds about 6
    windows, so expect a floor near 0.4 on short tracks and set the threshold
    with that in mind. Longer tracks make the test sharper.

    Track speed is deliberately not used. A static source is tempting to
    reject as "not flying", but an insect working a flower is also static, and
    rejecting on motion would remove exactly those observations.
    """
    n = min(len(sig_a), len(sig_b))
    if not np.isfinite(f0) or f0 <= 0 or n < 32:
        return 0.0
    t = np.arange(n) / fs
    carrier = np.exp(-2j * np.pi * f0 * t)
    win = int(max(3, round(cycles * fs / f0)))
    if win >= n:
        win = max(3, n // 3)

    # Shift the f0 component to 0 Hz and low-pass it, as in _plv_segment.
    def demod(sig):
        z = sig[:n].astype(np.float64) * carrier
        return uniform_filter1d(z.real, win) + 1j * uniform_filter1d(z.imag, win)

    za, zb = demod(sig_a), demod(sig_b)
    # Drop the edges where the moving average runs past the data.
    edge = win // 2
    if n - 2 * edge > 8:
        za, zb = za[edge:n - edge], zb[edge:n - edge]
    # Keep samples where both signals have some f0 amplitude.
    mag = np.abs(za) * np.abs(zb)
    good = mag > (1e-12 + 0.05 * np.median(mag))
    if good.sum() < 8:
        return 0.0
    # za * conj(zb) has the phase difference as its angle; average its unit
    # phasors and take the length.
    rel = za[good] * np.conj(zb[good])
    return float(np.abs(np.mean(rel / np.abs(rel))))


def spectral_flatness(psd: np.ndarray) -> float:
    """Geometric over arithmetic mean of a PSD. 1.0 is white noise, 0 a pure tone."""
    p = psd[psd > 0]
    if p.size < 4:
        return 1.0
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def _reconcile_fundamental(
    f_spectral: float, f_unsigned: float, f_yin: float, yin_conf: float,
    agree_tol: float = 0.06, min_conf: float = 0.15, override_min_conf: float = 0.5,
) -> Tuple[float, str, bool]:
    """Combine the spectral peak and the YIN period into one fundamental.

    Returns (f0_hz, source, octave_corrected). source names the estimator used.

    The two estimators fail differently. The spectral peak is precise but picks
    whichever harmonic is loudest. YIN finds the true period but resolves it
    only to about a sample lag. So:

    * If they agree within `agree_tol` (6%), keep the spectral value for its
      precision.
    * If the spectral peak is close to 2, 3 or 4 times the YIN frequency, the
      peak is a harmonic and YIN wins. This is the case that matters.
    * If YIN has no confidence, fall back to the spectral peak, then to half
      the unsigned peak, the last resort the ON/OFF physics allows.

    YIN may replace a finite spectral peak only when its confidence is at least
    `override_min_conf`. Below that, a barely confident YIN value could replace
    a clean spectral line, for example a 180 Hz line replaced by 158 Hz at
    confidence 0.20. When there is no spectral peak, YIN at `min_conf` or above
    is used as the best estimate available.
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
    Passing a whole scene averages every source together and measures nothing.

    bin_us is the rate-signal bin in microseconds (200 us = 5 kHz). fmin_hz and
    fmax_hz bound the search for the fundamental. detrend_hz is the high-pass
    cutoff that removes slow brightness changes; by default half of fmin_hz,
    and never below 15 Hz.

    Steps: bin the events into signed and unsigned rates, find the spectral
    peak of each, estimate the period with YIN, reconcile these into one f0,
    then measure line SNR, harmonic content, phase locking and low-frequency
    power at that f0. Tracks too short to analyse come back with the defaults
    of `PeriodicityFeatures`.
    """
    feats = PeriodicityFeatures()
    if len(ev) == 0:
        return feats

    t_s, unsigned, signed = rate_signals(ev, bin_us=bin_us)
    fs = 1e6 / bin_us
    feats.n_events = len(ev)
    feats.duration_s = float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0
    feats.polarity_balance = float(ev.p.sum()) / len(ev)

    # Under 32 bins (6.4 ms at 200 us) there is too little signal for a spectrum.
    if len(unsigned) < 32:
        return feats

    if detrend_hz is None:
        detrend_hz = max(fmin_hz * 0.5, 15.0)
    u = _detrend_highpass(unsigned, fs, detrend_hz)
    s = _detrend_highpass(signed, fs, detrend_hz)

    # Fundamental candidates from each signal. The unsigned rate carries 2*f0,
    # so its search extends to twice fmax, kept below 90% of Nyquist.
    f_signed, _ = _fine_peak(s, fs, fmin_hz, fmax_hz)
    f_unsigned, _ = _fine_peak(u, fs, fmin_hz, min(2 * fmax_hz, fs / 2 * 0.9))
    feats.f0_signed_hz = f_signed
    feats.f0_unsigned_hz = f_unsigned

    # Cross-check: for a real wingbeat the unsigned peak sits at twice the signed
    # peak, within 12%. The flag is reported so the caller can act on a
    # disagreement instead of meeting it later as an octave error.
    octave_ok = (
        np.isfinite(f_signed) and np.isfinite(f_unsigned)
        and abs(f_unsigned - 2 * f_signed) < 0.12 * max(f_signed, 1.0)
    )
    feats.octave_agreement = bool(octave_ok)

    # Independent period estimate that does not depend on which harmonic is
    # loudest. Signed rate first; the unsigned rate only if that finds nothing.
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

    # Spectra of both signals. The band for the noise floor and harmonics runs
    # from half of fmin to three times fmax, capped at Nyquist.
    freqs, psd, df = _psd(u, fs)
    feats.freq_resolution_hz = float(df)
    band = (freqs >= fmin_hz * 0.5) & (freqs <= min(fmax_hz * 3, fs / 2))

    freqs_s, psd_s, _ = _psd(s, fs)
    band_s = (freqs_s >= fmin_hz * 0.5) & (freqs_s <= min(fmax_hz * 3, fs / 2))
    # Line SNR: the signed rate at f0 and the unsigned rate at 2*f0, where each
    # carries its line. The stronger of the two is reported.
    snr_signed = _line_snr_db(freqs_s, psd_s, f0, band=band_s)
    snr_unsigned = _line_snr_db(freqs, psd, 2 * f0, band=band)
    feats.peak_snr_db = float(max(snr_signed, snr_unsigned))
    # The same measure at f0_signed_hz itself, not the reconciled f0.
    # _line_snr_db returns -inf when the line frequency is not finite.
    feats.peak_snr_signed_db = _line_snr_db(freqs_s, psd_s, f_signed, band=band_s)

    # Harmonic support: the fraction of in-band power that sits at f0 and its
    # first three multiples, on the unsigned spectrum. Each harmonic counts its
    # strongest bin within 2 bins or 3% of its frequency, whichever is wider.
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
    # the signal before the high pass, since the high pass removes exactly this.
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
    """Wingbeat frequency by YIN in a sliding window, with rejected points kept.

    Returns (times_s, freqs_hz, rejected_times_s, rejected_freqs_hz), times in
    seconds from the first event. Not called by the bundled pipeline.

    Windows of `window_us` step by `step_us`; windows with fewer than
    `min_events` events are skipped. Three filters reject points:
    `min_confidence` drops windows where YIN found no clear period.
    `reference_hz`, if given, drops points more than `reference_tol` (25%) from
    a track-level f0; this catches a run of half-rate windows that agree with
    each other. `continuity_tol` drops points more than 15% from the median of
    their 5 neighbours, since a wingbeat cannot change that much in 20 ms.
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

    Returns (t_us, cycles) on the rate-signal grid. Read an event's phase with
    np.interp(ev.t, t_us, cycles) % 1. Phase 0 is arbitrary. The slope of
    cycles against time is the wingbeat frequency at each moment.

    Why. Folding at one fixed f0 assumes the wingbeat frequency holds still,
    and it often does not. In one moth track the frequency over 100 ms ranged
    from 36 to 46 Hz, so a 40-stroke fold at a single f0 drifted about one full
    cycle out of phase within its own window and blurred every wing position
    together. Following the phase avoids this. Phase-locked averaging in
    electrophysiology works the same way.

    How. The signed rate, which carries f0, is demodulated at f0_hz and
    low-passed at `lowpass_hz`. The angle that remains follows any frequency
    within f0_hz +/- lowpass_hz, and is added to the nominal f0_hz * t. The
    filter runs forwards and backwards, so it adds no lag. On that moth track,
    with the phase taken from one random half of the events and the stroke
    image built from the other half, stroke contrast at 40 cycles rose from
    0.027 at a fixed f0 to 0.089 with phase tracking. lowpass_hz = 8 is the
    value that was tested, not a proven optimum.
    """
    t0 = int(ev.t[0]) if t0_us is None else int(t0_us)
    t1 = int(ev.t[-1]) + 1 if t1_us is None else int(t1_us)
    fs = 1e6 / bin_us
    t_s, _u, s = rate_signals(ev, bin_us=bin_us, t0_us=t0, t1_us=t1)
    s = _detrend_highpass(s, fs, detrend_hz)
    tt = t_s - t0 / 1e6
    z = s * np.exp(-2j * np.pi * f0_hz * tt)
    # 3rd-order Butterworth low pass, applied only when the signal is long
    # enough for filtfilt's edge padding.
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

    Each sample is assigned to a wing cycle and a phase bin within it.
    Returns (phase_centres, mean_waveform, cycle_matrix). phase_centres run
    from 0 to 1. cycle_matrix is (n_cycles, n_phase_bins) with NaN for empty
    bins. The mean waveform is the average stroke shape of this individual.
    The cycle matrix shows how repeatable that shape is within one track,
    which bounds how well any two individuals could be told apart.

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
    # Integer part is the cycle number, fractional part the phase within it.
    cycle = np.floor(rel).astype(np.int64)
    phase_bin = np.clip(((rel - cycle) * n_phase_bins).astype(np.int64), 0, n_phase_bins - 1)
    n_cycles = int(cycle.max()) + 1

    # Mean of the samples falling in each (cycle, phase bin) cell.
    acc = np.zeros((n_cycles, n_phase_bins))
    cnt = np.zeros((n_cycles, n_phase_bins))
    np.add.at(acc, (cycle, phase_bin), sig)
    np.add.at(cnt, (cycle, phase_bin), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cycles = np.where(cnt > 0, acc / cnt, np.nan)

    # Average over cycles, ignoring empty cells; a bin empty in every cycle is 0.
    with warnings_suppressed():
        mean_wave = np.nanmean(cycles, axis=0)
    mean_wave = np.nan_to_num(mean_wave)
    phase = (np.arange(n_phase_bins) + 0.5) / n_phase_bins
    return phase, mean_wave, cycles


def half_cycle_similarity(ev: EventStream, f_fold: float, n_phase_bins: int = 32) -> float:
    """Octave check: fold the signed rate at f_fold and correlate its two halves.

    Returns the Pearson correlation between the first and second half of the
    folded cycle, or nan if the track is too short or a half is flat.

    Call it at HALF the spectral peak frequency. One folded cycle then holds
    two candidate strokes. If the spectral peak is the true wingbeat, the two
    halves are the same stroke twice and correlate highly. If half the peak is
    the true wingbeat, the halves are an up-stroke and a down-stroke, which
    produce different events, and correlate lower.

    There is no absolute threshold. Read the value against a control group of
    tracks whose spectral and YIN estimates already agree, and against a fold
    at f_fold * 1.37, which is a subharmonic of nothing and shows what folding
    alone produces.

    Use the track's raw events, not denoised ones. A coincidence filter
    removes more events at some stroke phases than at others, and this is a
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
    """Strongest periodicity straight from event timestamps, with no binning.

    For sparse tracks, a distant target or a few hundred events, binning
    discards much of the timing information. Lomb-Scargle works on the raw
    arrival times. Slower than the binned path. Not called by the bundled
    pipeline.

    Returns (f_peak_hz, normalised_power), searched over `n_freqs` evenly
    spaced frequencies in [fmin_hz, fmax_hz].
    """
    if len(t_us) < 16:
        return np.nan, 0.0
    t = (t_us - t_us[0]).astype(np.float64) / 1e6
    y = np.ones_like(t)
    y = y - y.mean()
    if np.allclose(y, 0):
        # Equal weights are all zero once the mean is removed, so this branch
        # always runs: the observable is the instantaneous rate 1 / inter-arrival.
        dt = np.diff(t, prepend=t[0])
        y = 1.0 / np.maximum(dt, 1e-9)
        y = y - y.mean()
    freqs = np.linspace(fmin_hz, fmax_hz, n_freqs)
    power = sps.lombscargle(t, y, 2 * np.pi * freqs, normalize=True)
    k = int(np.argmax(power))
    return float(freqs[k]), float(power[k])
