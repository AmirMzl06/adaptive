"""Adaptive-epsilon estimators for ACORN (PGD budget from signal statistics).

This module is deliberately self-contained: it depends only on ``numpy`` (and
optionally ``scipy`` for the nearest-neighbour estimator). None of it imports
torch or CEBRA, so it can be unit-tested and plotted on a machine without a GPU
stack. The torch wiring that plugs these budgets into the real solver lives in
``torch_adapter.py``.

Every estimator answers the same question in a different way:

    "How large may we perturb each neuron's input before we leave the range of
     variation the signal shows on its own?"

That question is the honest, data-driven replacement for a hand-picked epsilon.
Each estimator returns a *per-neuron* budget ``eps`` of shape ``(N,)`` (or a
scalar) given neural activity of shape ``(T, N)`` = (time, neurons), scaled by a
single dimensionless coefficient ``coef``. The coefficient is the ONLY thing a
user sets, and it is meant to be fixed a priori and shared across datasets --
not tuned until the adversarial model wins.

Family relationship (why these are principled, not arbitrary):
    per_neuron_std is exactly the diagonal of the data covariance. The
    Mahalanobis / ZCA-whitening estimator (``whitening``) is the full-covariance
    generalisation: it also accounts for correlations between neurons. If the
    covariance is diagonal, whitening collapses back to per_neuron_std. So the
    whole set is one idea -- "spend budget proportional to how the data actually
    varies" -- seen at increasing levels of structure (global -> per-neuron ->
    full covariance).
"""

from __future__ import annotations

import numpy as np

# Consistency constant making a MAD a std-equivalent scale for Gaussian data.
_MAD_TO_STD = 1.4826


def _as_TN(neural: np.ndarray) -> np.ndarray:
    """Validate and return neural activity as a (T, N) float array."""
    arr = np.asarray(neural, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"neural must be 2D (time, neurons); got shape {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError("neural needs at least 2 time steps.")
    return arr


# ----------------------------------------------------------------------
# Amplitude-based budgets: "perturb within the neuron's own spread"
# ----------------------------------------------------------------------
def eps_std(neural: np.ndarray, coef: float) -> np.ndarray:
    """eps_j = coef * std_j.

    PGD link: the plainest notion of "the signal's own scale". A neuron that
    swings a lot may be pushed a lot; a near-silent neuron barely at all. This
    is the diagonal of the covariance -- see ``whitening`` for the full version.
    """
    x = _as_TN(neural)
    return coef * x.std(axis=0)


def eps_global_std(neural: np.ndarray, coef: float) -> float:
    """eps = coef * std(all neurons) -- a single scalar budget for every input.

    PGD link: the classic constant-epsilon attack, but with the constant tied to
    the overall scale of the recording so it is at least dataset-agnostic.
    """
    x = _as_TN(neural)
    return float(coef * x.std())


def eps_mad(neural: np.ndarray, coef: float) -> np.ndarray:
    """eps_j = coef * 1.4826 * MAD_j (robust std).

    PGD link: same intent as ``eps_std`` but the budget is not inflated by a few
    large transients / artifacts. Good when the recording has occasional spikes.
    """
    x = _as_TN(neural)
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0)
    return coef * _MAD_TO_STD * mad


def eps_iqr(neural: np.ndarray, coef: float) -> np.ndarray:
    """eps_j = coef * IQR_j (75th - 25th percentile).

    PGD link: budget = the width the bulk (middle 50%) of the samples occupy.
    Robust, and directly interpretable as "stay inside where most data lives".
    """
    x = _as_TN(neural)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    return coef * (q75 - q25)


def eps_prange(neural: np.ndarray,
               coef: float,
               lo: float = 5.0,
               hi: float = 95.0) -> np.ndarray:
    """eps_j = coef * (P_hi - P_lo) -- robust dynamic range per neuron.

    PGD link: budget proportional to the neuron's usable dynamic range, using
    percentiles (default 5-95) instead of raw min/max so single outliers do not
    blow it up.
    """
    x = _as_TN(neural)
    p_hi, p_lo = np.percentile(x, [hi, lo], axis=0)
    return coef * (p_hi - p_lo)


# ----------------------------------------------------------------------
# Noise-based budget: "perturb only within the signal's intrinsic noise"
# ----------------------------------------------------------------------
def eps_diff_noise(neural: np.ndarray,
                   coef: float,
                   robust: bool = True) -> np.ndarray:
    """eps_j from the scale of temporal differences x[t]-x[t-1].

    This is the estimator with, arguably, the tightest link to the *meaning* of
    an adversarial budget. ``std`` and friends measure the full amplitude of a
    neuron's tuning; but a large-amplitude, slowly-varying tuning curve is real
    signal we should NOT be free to overwrite. The frame-to-frame difference
    isolates the fast, unstructured part -- the neuron's intrinsic noise floor.
    Setting the budget to that noise floor says: "the attacker may move the
    input only as much as the signal jitters on its own between adjacent
    samples", i.e. stay within the noise, never rewrite the signal.

    ``robust=True`` uses 1.4826 * MAD of the differences (a standard wavelet-
    style noise estimator); ``robust=False`` uses their std.
    """
    x = _as_TN(neural)
    d = np.diff(x, axis=0)
    if robust:
        med = np.median(d, axis=0, keepdims=True)
        scale = _MAD_TO_STD * np.median(np.abs(d - med), axis=0)
    else:
        scale = d.std(axis=0)
    # diff of two iid-ish samples inflates variance by ~sqrt(2); undo it so the
    # result is comparable to a per-sample std.
    return coef * scale / np.sqrt(2.0)


# ----------------------------------------------------------------------
# Geometry-based budgets: "perturb less than the distance to other states"
# ----------------------------------------------------------------------
def eps_nn_percentile(neural: np.ndarray,
                      coef: float,
                      q: float = 5.0,
                      window: int = 1,
                      time_gap: int | None = None,
                      max_points: int = 4000,
                      seed: int = 0) -> float:
    """Scalar L2 budget from the low percentile of nearest-neighbour distances.

    This is the user's original ``minL2`` idea, made robust. For a bank of
    population-state vectors (optionally length-``window`` snippets), it looks at
    how close in Euclidean distance distinct states get, and sets the budget to
    a low percentile of the nearest-neighbour distances (not the raw min, which
    is noise-sensitive).

    PGD link: if the perturbation is smaller than the distance to the nearest
    genuinely different state, the attack cannot turn one state into another
    real one -- it can only explore the empty space around a sample. So this ties
    the budget to the margin between real neural states.

    Two traps handled here:
    * Overlapping/adjacent time points are nearly identical, so ``min L2 ~= 0``
      and degenerates. We decorrelate by sub-sampling points at least
      ``time_gap`` apart (defaults to ``window``).
    * O(T^2) distances are avoided with a KD-tree when scipy is available.
    """
    x = _as_TN(neural)
    T, N = x.shape
    if time_gap is None:
        time_gap = max(window, 1)

    # Build (optionally windowed) state vectors, decorrelated in time.
    starts = np.arange(0, T - window + 1, max(time_gap, 1))
    if len(starts) > max_points:
        rng = np.random.default_rng(seed)
        starts = np.sort(rng.choice(starts, size=max_points, replace=False))
    if window == 1:
        pts = x[starts]
    else:
        pts = np.stack([x[s:s + window].reshape(-1) for s in starts], axis=0)
    if len(pts) < 2:
        raise ValueError("Not enough decorrelated points for NN estimate.")

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        # k=2: first neighbour is the point itself (distance 0).
        dists, _ = tree.query(pts, k=2)
        nn = dists[:, 1]
    except Exception:
        # Pure-numpy fallback (O(M^2) memory); fine for modest M.
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(d2.min(axis=1))

    return float(coef * np.percentile(nn, q))


def whitening(neural: np.ndarray,
              coef: float,
              ridge: float = 1e-3):
    """ZCA / Mahalanobis budget: full-covariance generalisation of per_neuron_std.

    Returns ``(transform, radius)`` where ``transform`` is an ``(N, N)`` matrix
    and ``radius`` a scalar such that the allowed perturbations are the
    ellipsoid ``|| transform @ delta ||_2 <= radius``.

    PGD link: per-neuron budgets treat neurons independently, but neural data is
    correlated -- some directions in neuron-space carry almost no variance. ZCA
    whitening makes the budget large along high-variance directions and small
    along low-variance ones, i.e. the attacker gets room exactly where the data
    naturally moves and none where it does not. If the covariance is diagonal
    this reduces to ``eps_std`` with ``eps_j = coef * std_j`` (the family link).

    ``ridge`` is a relative floor on eigenvalues to keep near-silent directions
    from producing an unbounded budget.
    """
    x = _as_TN(neural)
    xc = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(xc, rowvar=False)
    cov = np.atleast_2d(cov)
    evals, evecs = np.linalg.eigh(cov)
    floor = ridge * float(evals.max()) if evals.max() > 0 else ridge
    evals = np.clip(evals, floor, None)
    # transform = Cov^{-1/2} = V diag(1/sqrt(lambda)) V^T (symmetric ZCA form).
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return inv_sqrt, float(coef)


# Registry so the demo / adapter can iterate estimators by name.
ESTIMATORS = {
    "per_neuron_std": eps_std,
    "global_std": eps_global_std,
    "per_neuron_mad": eps_mad,
    "per_neuron_iqr": eps_iqr,
    "per_neuron_prange": eps_prange,
    "per_neuron_diff_noise": eps_diff_noise,
    "nn_percentile": eps_nn_percentile,
    "whitening": whitening,
}
