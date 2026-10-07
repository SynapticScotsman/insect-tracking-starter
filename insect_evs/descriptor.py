"""How strongly a spatial pattern repeats after a lag.

Copied unchanged from the evfilt package (MIT licence,
Copyright (c) 2026 Paul Kirkland)."""
import numpy as np


def descriptor_autocorr(descriptors: np.ndarray) -> np.ndarray:
    r"""Correlation of a descriptor sequence with itself at each lag.

    `descriptors` is an (n_samples, n_features) array: one descriptor per time
    step, for example a histogram of ON and OFF events around a tracked insect.
    Returns an array of length n_samples, where r[k] is the mean correlation
    between descriptors k samples apart. r[0] is 1 for any sequence of nonzero
    descriptors. Returns an empty array if there are fewer than 2 samples.

    MATHS
        With D[j] the descriptor at sample j, scaled to unit L2 length,

            r[k] = mean_j < D[j], D[j+k] >

        Each lag averages over the m - k pairs that overlap, so long lags rest
        on few pairs and are noisier.

    WHY A DESCRIPTOR AND NOT A RATE. A scalar summary of a moving, rotating
    body, such as its event rate, silhouette area or spatial spread, repeats
    at the body's symmetry order. A cube looks the same every quarter turn, so
    all of those summaries peak there, and a period read from them is too
    short by that factor. A descriptor detailed enough to tell the faces apart
    only matches again when the pose does. On a rotating cube:

        quarter turn  r = 0.567
        half turn     r = 0.534
        full turn     r = 0.918

    The full-turn peak is not merely present, it dominates. So the first
    prominent peak marks the true period, without the harmonic ambiguity that
    affects a rate signal.
    """
    d = np.asarray(descriptors, np.float64)
    if d.ndim != 2 or d.shape[0] < 2:
        return np.zeros(0, np.float64)
    # Scale each descriptor to unit length; all-zero rows are left as zero.
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.where(n > 0, n, 1.0)
    m = d.shape[0]
    # Row-wise dot products at each lag, averaged over the overlapping pairs.
    return np.array([float((d[: m - k] * d[k:]).sum(1).mean())
                     for k in range(m)], np.float64)
