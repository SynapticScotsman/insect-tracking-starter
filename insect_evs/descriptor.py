"""descriptor_autocorr, lifted verbatim from evfilt.periodicity
(Code/evfilt, MIT licence, Copyright (c) 2026 Paul Kirkland)."""
import numpy as np


def descriptor_autocorr(descriptors: np.ndarray) -> np.ndarray:
    r"""Correlation of a descriptor sequence against itself at each lag.

    MATHS
        With D[j] the (L2-normalised) descriptor at sample j,

            r[k] = mean_j < D[j], D[j+k] >

    WHY A DESCRIPTOR AND NOT A RATE. This is the whole point of the module's
    trap 3. A rotating body's silhouette, event rate, spatial spread and every
    other scalar summary repeat at the SYMMETRY ORDER: a cube looks the same
    every quarter turn, so all of them peak there and a period read off any of
    them is short by a factor of `fold`. A descriptor rich enough to tell the
    faces apart does not come back until the pose does.

    Reported from fastnfeast (`docs/FACE_READOUT_METHOD.md` section 3), where
    five of six period estimates were wrong for exactly this reason:

        quarter turn  r = 0.567
        half turn     r = 0.534
        FULL turn     r = 0.918

    The margin is what makes it usable: the fundamental is not merely present,
    it dominates, so the same first-prominent-peak rule applies without the
    harmonic ambiguity that makes it necessary on a rate series.
    """
    d = np.asarray(descriptors, np.float64)
    if d.ndim != 2 or d.shape[0] < 2:
        return np.zeros(0, np.float64)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.where(n > 0, n, 1.0)
    m = d.shape[0]
    return np.array([float((d[: m - k] * d[k:]).sum(1).mean())
                     for k in range(m)], np.float64)
