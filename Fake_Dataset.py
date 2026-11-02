# Paper = Time-series attribution maps with regularized contrastive learning Steffen Schneider1 Rodrigo Gonz´alez Laiz1 Anastasiia Filippova1 Markus Frey1 Mackenzie Weygandt Mathis1 1EPFL mackenzie.mathis@epfl.ch & steffen.schneider@helmholtz-munich.de
#Fig 5

# ### Lorenzo
### Lorenzo v3  --  stochastic Lorenz latents, single-variable calibration
from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ==============================================================================
# 1. CONFIG
# ==============================================================================

DRY_RUN = False                # <<< smoke test.  Set False for real runs.
STAGE   = "select"             # "select" | "evaluate" | "both"
                               # deliberately NOT "both": run 1 produced ZERO
                               # included seeds, so confirm stage 1 is healthy
                               # before spending 10 seeds x 20k iters on it.

CEBRA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "CEBRA-original")

# --- architecture -------------------------------------------------------------
ARCH_VARIANT = "euclidean_tanh"

_ARCH_TABLE = {
    "euclidean_tanh":   dict(model="offset1-model-mse", normalize=False,
                             criterion="euclidean", out_extra=0, pinv_drop=0,
                             tanh_scale=1.0),
    "euclidean_linear": dict(model="offset1-model-mse", normalize=False,
                             criterion="euclidean", out_extra=0, pinv_drop=0,
                             tanh_scale=None),
    "cosine_sphere":    dict(model="offset1-model",     normalize=True,
                             criterion="cosine",        out_extra=1, pinv_drop=1,
                             tanh_scale=None),
}
_ARCH      = _ARCH_TABLE[ARCH_VARIANT]
MODEL_NAME = _ARCH["model"]
CRITERION  = _ARCH["criterion"]
OUT_EXTRA  = _ARCH["out_extra"]
PINV_DROP  = _ARCH["pinv_drop"]
TANH_SCALE = _ARCH["tanh_scale"]
NUM_UNITS  = 128

# --- data generating process (Figure 5 / Appendix B.1) ------------------------
D_LATENT_LIST   = [6]
N1, N2          = 25, 25
D_OBS           = N1 + N2
TARGET_PRE_SD   = 0.80
SIGMA_OBS_SWEEP = [0.00]
TIME_OFFSET     = 1
STANDARDIZE     = False

# --- LATENT PROCESS ----------------------------------------------------------
# WHY THIS EXISTS, and why "lorenz_det" is a trap.
#
# CEBRA's identifiability argument needs p(z_{t+1} | z_t) to be a genuine
# conditional distribution -- for the Euclidean criterion, isotropic Gaussian
# around the anchor.  A DETERMINISTIC chaotic ODE has p(z_{t+1}|z_t) = delta,
# and then InfoNCE has a free degenerate optimum: ANY encoder that is constant
# along the flow puts the positive exactly on the anchor.  Such an encoder
# carries no information about z at all.  That is not a theoretical worry --
# it is what the first Lorenz run actually did: R2 collapsed to 0.42 (gate is
# 0.95, so EVERY seed was excluded), tau pinned at the MIN_TEMPERATURE floor of
# 0.05 in every attacked arm, and held-out InfoNCE sat 2.3 nats ABOVE the
# training value, i.e. the model memorised the trajectory.
#
# "lorenz_sde" fixes this by integrating the Langevin form
#     dz = f_lorenz(z) dt + sigma_d sqrt(dt) dW
# with Euler-Maruyama.  The one-step conditional is then EXACTLY
# N(z + f(z) dt, sigma_d^2 dt I) -- isotropic Gaussian, as the theory wants --
# while the drift still sculpts the Lorenz attractor into the MARGINAL, which
# is the part that makes the attribution problem non-trivial.
#
# The tension is real and cannot be wished away: a noise-dominated conditional
# smears the attractor, a drift-dominated one reintroduces the degeneracy.
LATENT_PROCESS = "lorenz_sde"   # "brownian" | "lorenz_sde" | "lorenz_det"

BROWNIAN_SIGMA   = 0.10   # per-step SD in the box, "brownian" only
LATENT_STEP_SD   = 0.10   # reference value for reporting only; nothing is
                          # solved for it any more

# dt is now FIXED, not solved for.  The previous version solved for dt AND
# sigma_d simultaneously against two targets that turned out to be mutually
# unreachable for Lorenz-63, and the iteration ran sigma_d up until explicit
# Euler blew up.  Only sigma_d is searched now, dt is a plain knob.
LORENZ_DT              = 0.02
LORENZ_DRIFT_CAP       = 5.0    # max drift DISPLACEMENT per step, raw units
LORENZ_TARGET_DRIFT_R2 = 0.08   # what the noise level is bisected to achieve
LORENZ_SIGMA_D_LO      = 0.5    # bisection bracket
LORENZ_SIGMA_D_HI      = 400.0
LORENZ_BISECT_ITERS    = 14
LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA = 10.0, 28.0, 8.0 / 3.0
LORENZ_BURN_IN   = 2000
LORENZ_CAL_STEPS = 4000
STEP_SD_LO, STEP_SD_HI = 0.02, 0.30   # sane range for the per-step displacement

# --- optimisation -------------------------------------------------------------
LEARNING_RATE    = 3e-4
MIN_TEMPERATURE  = 0.05
INIT_TEMPERATURE = 1.0
NCE_FLOOR        = -1e4

# --- the regularizer (paper Eq. 10 / 15) -------------------------------------
LAMBDA_MAX     = 0.10
JREG_ESTIMATOR = "proj"
JREG_NPROJ     = 1
JREG_SUBBATCH  = 512
JREG_AT        = "clean"
JREG_ON        = "both"

# --- the attack --------------------------------------------------------------
ADV_ALPHA_RULE = "fork"
ADV_ALPHA_RAW  = 0.01
ADV_ALPHA_REL  = 0.20
ADV_STEPS      = 10
ADV_CLIP_RANGE = False
ADV_CACHE_POS_NEG = True

_EPS_LINF   = [0.0125, 0.025, 0.05, 0.10, 0.20]
ADV_EPS_GRID = {"linf": _EPS_LINF,
                "l2":   [round(e * math.sqrt(D_OBS), 4) for e in _EPS_LINF]}
ATTACK_NORMS = ["linf", "l2"]

NCE_TOL_ABS = 0.05
EVAL_NCE_BATCHES = 20

SELECTED_MANUAL = {"linf": None, "l2": None}

# --- attribution -------------------------------------------------------------
ATTR_METHOD_CANDIDATES = ["neuron-gradient", "neuron-gradient-batched",
                          "jacobian-based", "jacobian-based-batched"]
ATTR_JF_KEYS    = ["neuron-gradient", "jf", "jacobian"]
ATTR_JFINV_KEYS = ["neuron-gradient-inv-svd", "neuron-gradient-inv-lsq",
                   "jf-inv-svd", "jf-inv-lsq", "jf-inv",
                   "inverted-neuron-gradient"]
ATTR_NUM_BATCHES = 32

# --- quality gates -----------------------------------------------------------
R2_MIN              = 0.95
R2_PARITY_MAX       = 0.05
SURROGATE_AUROC_MIN = 0.995
JREG_SELFTEST_RTOL  = 0.15
ADV_CACHE_RTOL      = 1e-5

# The cheap pre-flight gates.  These cost milliseconds and would have caught the
# failed deterministic-Lorenz run before a single optimizer step.
DRIFT_R2_MAX  = 0.15   # linear R2 of predicting z[t+off]-z[t] from z[t].
                       # brownian -> 0.00 ; lorenz_sde (calibrated) -> ~0.08 ;
                       # lorenz_det -> >0.5, i.e. the increment is a function of
                       # the state and the conditional is not a distribution.
NCE_GAP_MAX   = 0.50   # eval_nce - nce_clean_train.  A large positive gap means
                       # the encoder memorised the trajectory rather than
                       # learning the latent geometry.  Warned, not fatal.

# --- reporting ---------------------------------------------------------------
PRIMARY_METRIC = "auroc_global"
N_BOOTSTRAP    = 1000
RESULT_TAG     = f"{ARCH_VARIANT}_{LATENT_PROCESS}"
SELECT_SEED0   = 90_001
EVAL_SEED0     = 1_234
SEED_STRIDE    = 977

# --- scale presets -----------------------------------------------------------
_SCALES = {
    "dry":         dict(n_seeds=2,  T=8_000,   iters=300,    batch=256,
                        map_pts=1500, attr_pts=1500),
    "pilot":       dict(n_seeds=3,  T=40_000,  iters=4_000,  batch=1024,
                        map_pts=4000, attr_pts=4000),
    "select":      dict(n_seeds=3,  T=40_000,  iters=8_000,  batch=2048,
                        map_pts=4000, attr_pts=4000),
    "paper_final": dict(n_seeds=10, T=100_000, iters=20_000, batch=5_000,
                        map_pts=8000, attr_pts=8000),
}
SELECT_SCALE   = "select"
EVALUATE_SCALE = "paper_final"

N_SEEDS = T_SAMPLES = MAX_ITER = BATCH_SIZE = MAP_POINTS = ATTR_POINTS = 0
LAMBDA_WARMUP = LAMBDA_RAMP = 1


def set_scale(name: str):
    global N_SEEDS, T_SAMPLES, MAX_ITER, BATCH_SIZE, MAP_POINTS, ATTR_POINTS
    global LAMBDA_WARMUP, LAMBDA_RAMP, SCALE_NAME
    s = _SCALES[name]
    N_SEEDS, T_SAMPLES, MAX_ITER = s["n_seeds"], s["T"], s["iters"]
    BATCH_SIZE, MAP_POINTS, ATTR_POINTS = s["batch"], s["map_pts"], s["attr_pts"]
    LAMBDA_WARMUP = max(1, int(round(MAX_ITER * 2500 / 20000)))
    LAMBDA_RAMP   = max(1, int(round(MAX_ITER * 2500 / 20000)))
    SCALE_NAME = name
    print(f"[scale] {name}: seeds={N_SEEDS} T={T_SAMPLES} iters={MAX_ITER} "
          f"batch={BATCH_SIZE} lambda warmup/ramp={LAMBDA_WARMUP}/{LAMBDA_RAMP}")


if DRY_RUN:
    SELECT_SCALE = EVALUATE_SCALE = "dry"
    ATTACK_NORMS = ["linf", "l2"]
    _EPS_LINF = [0.05, 0.20]
    ADV_EPS_GRID = {"linf": _EPS_LINF,
                    "l2": [round(e * math.sqrt(D_OBS), 4) for e in _EPS_LINF]}
    R2_MIN, R2_PARITY_MAX = 0.0, 1e9
    print("[config] DRY_RUN=True -> scales forced to 'dry', eps grid trimmed, "
          "R2 gates DISABLED.  auROC/auPRC from this run are MEANINGLESS: it "
          "is an error smoke test, not a result.")

# ==============================================================================
# 2. ENVIRONMENT
# ==============================================================================

import torch
import torch.nn as nn

try:
    from scipy import stats as _sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    warnings.warn("scipy missing: Wilcoxon replaced by a sign test.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cebra():
    import sys
    from pathlib import Path

    # IMPORTANT: clean/original CEBRA, NOT the advisor's fork -- the attack is
    # reimplemented in this file, so the fork would apply it twice.
    cebra_dir = Path(__file__).resolve().parent / "CEBRA-original"

    for module_name in list(sys.modules):
        if module_name == "cebra" or module_name.startswith("cebra."):
            del sys.modules[module_name]

    sys.path.insert(0, str(cebra_dir))

    try:
        import cebra
        import cebra.models
        import cebra.models.criterions as _crit
    except Exception as exc:
        raise SystemExit(
            f"cannot import ORIGINAL cebra from {cebra_dir}. error: {exc}"
        )

    print("CEBRA loaded from:", cebra.__file__)

    return cebra, _crit


CEBRA, CRIT = _load_cebra()


def adv_alpha(eps: float) -> float:
    if ADV_ALPHA_RULE == "madry":
        return 2.5 * eps / max(ADV_STEPS, 1)
    if ADV_ALPHA_RULE == "relative":
        return ADV_ALPHA_REL * eps
    return ADV_ALPHA_RAW


def _banner():
    print("=" * 78)
    print(f" STAGE={STAGE}  arch={ARCH_VARIANT}  model={MODEL_NAME} "
          f"tanh_scale={TANH_SCALE}")
    print(f" latent process = {LATENT_PROCESS}"
          + (f"  (dt={LORENZ_DT}, target drift_r2={LORENZ_TARGET_DRIFT_R2})"
             if LATENT_PROCESS.startswith("lorenz") else
             f"  (sigma={BROWNIAN_SIGMA})"))
    print(f" device={DEVICE}  standardize={STANDARDIZE}  d_list={D_LATENT_LIST}")
    print(f" lambda_max={LAMBDA_MAX}  jreg={JREG_ESTIMATOR}/{JREG_NPROJ} "
          f"sub={JREG_SUBBATCH} at={JREG_AT} on={JREG_ON}")
    print(f" attack: steps={ADV_STEPS} alpha_rule={ADV_ALPHA_RULE} "
          f"clip={ADV_CLIP_RANGE} cache_pos_neg={ADV_CACHE_POS_NEG}")
    for n in ATTACK_NORMS:
        print(f"   eps grid [{n:>4}]: {ADV_EPS_GRID[n]}")
    print(f" gates: R2>={R2_MIN} parity<={R2_PARITY_MAX} "
          f"drift_r2<={DRIFT_R2_MAX} surrogate_auROC>={SURROGATE_AUROC_MIN}")
    print(f" selection: largest eps with held-out clean InfoNCE within "
          f"{NCE_TOL_ABS} nats of baseline  (NEVER auROC, INCLUDED seeds only)")
    print(f" select seeds base={SELECT_SEED0}   eval seeds base={EVAL_SEED0} "
          f"(disjoint)")
    print("=" * 78)


# ==============================================================================
# 3. DATA GENERATING PROCESS
# ==============================================================================

def brownian_motion_box(T: int, d: int, sigma: float, rng) -> np.ndarray:
    """z_1 ~ U[-1,1]^d ; z_t ~ N_[-1,1](z_{t-1}, sigma^2 I), TRUNCATED (clipped).

    The paper clips to the box rather than reflecting off it; clipping leaves
    probability mass on the faces of the cube, reflecting does not."""
    z = np.empty((T, d), dtype=np.float64)
    z[0] = rng.uniform(-1.0, 1.0, size=d)
    for t in range(1, T):
        z[t] = np.clip(z[t - 1] + sigma * rng.normal(size=d), -1.0, 1.0)
    return z


def _lorenz_deriv(state: np.ndarray) -> np.ndarray:
    """f(z) for a stack of Lorenz-63 states, [..., 3] -> [..., 3]."""
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    return np.stack([LORENZ_SIGMA * (y - x),
                     x * (LORENZ_RHO - z) - y,
                     x * y - LORENZ_BETA * z], axis=-1)


def _integrate_lorenz(n_chains: int, n_steps: int, dt: float, sigma_d: float,
                      rng, burn_in: int) -> np.ndarray:
    """Euler-Maruyama for dz = f(z) dt + sigma_d sqrt(dt) dW, CAPPED drift step.

    Why the cap exists.  The Lorenz drift is quadratic (x*z, x*y), so it is not
    globally Lipschitz.  Additive noise occasionally pushes the state off the
    attractor; |f| then grows like the square of the excursion, the explicit
    Euler increment f*dt overshoots further out, and the iteration reaches inf
    within a few hundred steps.  That is a failure of explicit Euler, not of the
    SDE.  Capping the drift DISPLACEMENT (never the state) at LORENZ_DRIFT_CAP
    raw units is a no-op in the normal regime -- on the attractor |f|*dt is
    about 2.6 at dt=0.02, well under the cap -- and acts purely as a brake
    during a runaway, where the Lorenz drift points inward anyway, so the
    process stays recurrent instead of diverging.

    sigma_d = 0 recovers the deterministic system (the 'lorenz_det' trap)."""
    state = (np.array([1.0, 1.0, 1.0])[None]
             + rng.normal(scale=2.0, size=(n_chains, 3)))
    out = np.empty((n_steps, n_chains, 3), dtype=np.float64)
    sq = math.sqrt(dt)
    for t in range(burn_in + n_steps):
        step = _lorenz_deriv(state) * dt
        mag = np.linalg.norm(step, axis=-1, keepdims=True)
        step = step * np.minimum(1.0, LORENZ_DRIFT_CAP / np.maximum(mag, 1e-12))
        state = state + step + sigma_d * sq * rng.normal(size=state.shape)
        if t >= burn_in:
            out[t - burn_in] = state
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Lorenz SDE non-finite despite the drift cap; "
                           "lower LORENZ_DT.")
    return out


def _probe_lorenz(sigma_d: float, cal_seed: int = 20260913) -> Tuple[float, float]:
    """(drift_r2, normalised per-step displacement) at this noise level.

    A FIXED seed, so the bisection below sees a smooth deterministic function of
    sigma_d (common random numbers) rather than a noisy one."""
    rng = np.random.default_rng(cal_seed)
    traj = _integrate_lorenz(2, LORENZ_CAL_STEPS, LORENZ_DT, sigma_d, rng,
                             LORENZ_BURN_IN)
    z = traj[:, 0, :]
    z = z - z.mean(0, keepdims=True)
    scale = float(np.max(np.abs(z)))
    step = float(np.sqrt(np.mean((z[TIME_OFFSET:] - z[:-TIME_OFFSET]) ** 2)))
    return drift_predictability(z, TIME_OFFSET), step / max(scale, 1e-12)


_LORENZ_CAL: Optional[Tuple[float, float]] = None


def calibrate_lorenz() -> Tuple[float, float]:
    """Bisect sigma_d until drift_r2 hits LORENZ_TARGET_DRIFT_R2.

    Searching ONE variable against the quantity the gate actually reads -- not
    two variables against two analytic proxies -- is what makes this terminate.
    drift_r2 is monotonically decreasing in sigma_d (more noise, less of the
    increment is explained by the state), so bisection is well posed, and the
    bracket is hard-bounded so it cannot run away.

    Calibrated ONCE for the whole run with a fixed RNG, never per seed: if every
    seed used its own noise level the arms would be compared across different
    data-generating processes, which is a confound, not a replicate."""
    global _LORENZ_CAL
    if _LORENZ_CAL is not None:
        return _LORENZ_CAL
    tgt = LORENZ_TARGET_DRIFT_R2
    lo, hi = LORENZ_SIGMA_D_LO, LORENZ_SIGMA_D_HI
    r_lo, _ = _probe_lorenz(lo)
    r_hi, _ = _probe_lorenz(hi)
    print(f"[dgp] calibrating lorenz_sde: dt={LORENZ_DT:g} (fixed), "
          f"target drift_r2={tgt:g}")
    print(f"  [dgp] bracket: sigma_d={lo:g} -> drift_r2={r_lo:.4f} ; "
          f"sigma_d={hi:g} -> drift_r2={r_hi:.4f}")
    if r_lo <= tgt:
        sigma_d = lo
        print("  [dgp] even the smallest noise already meets the target.")
    elif r_hi > tgt:
        raise RuntimeError(
            f"drift_r2 is still {r_hi:.3f} at sigma_d={hi:g}: this Lorenz "
            f"configuration cannot be made noise-dominated at dt={LORENZ_DT:g}. "
            f"Raise LORENZ_SIGMA_D_HI, lower LORENZ_DT, or use "
            f"LATENT_PROCESS='brownian'.")
    else:
        for _ in range(LORENZ_BISECT_ITERS):
            mid = math.sqrt(lo * hi)          # geometric: sigma_d spans decades
            r, _ = _probe_lorenz(mid)
            if r > tgt:
                lo = mid                       # too deterministic, need more noise
            else:
                hi = mid
        sigma_d = hi                           # the side that MEETS the target
    r, s = _probe_lorenz(sigma_d)
    print(f"[dgp] calibrated: sigma_d={sigma_d:.3f}  drift_r2={r:.4f}  "
          f"step_sd={s:.4f}")
    if not (STEP_SD_LO <= s <= STEP_SD_HI):
        print(f"  !! [dgp] per-step displacement {s:.4f} is outside "
              f"[{STEP_SD_LO}, {STEP_SD_HI}].  It scales like sqrt(dt), so set "
              f"LORENZ_DT = {LORENZ_DT * (LATENT_STEP_SD / max(s, 1e-9)) ** 2:.5f} "
              f"and rerun the calibration.")
    if sigma_d > 60.0:
        print(f"  !! [dgp] sigma_d={sigma_d:.1f} is large relative to the "
              f"attractor (~20 raw units): the butterfly is mostly washed out "
              f"and this is closer to a drift-biased random walk than to Lorenz "
              f"dynamics.  Fine for the benchmark, but do not oversell it as "
              f"'chaotic latents' in the paper.")
    _LORENZ_CAL = (LORENZ_DT, sigma_d)
    return _LORENZ_CAL


def lorenz_latents(T: int, d: int, rng, stochastic: bool) -> np.ndarray:
    """ceil(d/3) independent Lorenz-63 chains -> [T, d] in the [-1,1] box.

    Independent chains (different random ICs) give genuinely independent latent
    blocks, and at d=6 the two chains line up exactly with the Figure-5 split
    d1 = d//2 + d%2 = 3, so z1 is chain 1 and z2 is chain 2.

    Centering before the max-abs rescale is not cosmetic: the Lorenz z
    coordinate is strictly positive, so without it latent columns 2 and 5 would
    carry a DC offset and never visit half the box, while mix.gain -- calibrated
    on a global std that already contains that offset -- would silently shrink
    the fluctuating part of every pre-activation below TARGET_PRE_SD."""
    n_chains = int(np.ceil(d / 3))
    if d % 3 != 0:
        warnings.warn(
            f"d={d} is not a multiple of 3: the last chain is truncated "
            f"mid-state, so some retained latent coordinates are driven by a "
            f"Lorenz coordinate that is NOT part of z.  Prefer d in {{3,6,9}}.")
    if stochastic:
        dt, sigma_d = calibrate_lorenz()
    else:
        dt, sigma_d = LORENZ_DT, 0.0
    traj = _integrate_lorenz(n_chains, T, dt, sigma_d, rng, LORENZ_BURN_IN)
    z = traj.reshape(T, n_chains * 3)[:, :d]
    z = z - z.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(z), axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    z = np.clip(z / scale, -1.0, 1.0)
    return z.astype(np.float64)


def make_latents(T: int, d: int, rng) -> np.ndarray:
    if LATENT_PROCESS == "brownian":
        return brownian_motion_box(T, d, BROWNIAN_SIGMA, rng)
    if LATENT_PROCESS == "lorenz_sde":
        return lorenz_latents(T, d, rng, stochastic=True)
    if LATENT_PROCESS == "lorenz_det":
        warnings.warn(
            "LATENT_PROCESS='lorenz_det': the transition is deterministic, so "
            "p(z'|z) is a delta and InfoNCE has a degenerate optimum (any "
            "encoder constant along the flow).  This is the configuration that "
            "produced R2=0.42 and excluded every seed.  Kept only for the "
            "record.")
        return lorenz_latents(T, d, rng, stochastic=False)
    raise ValueError(f"unknown LATENT_PROCESS {LATENT_PROCESS!r}")


def drift_predictability(z: np.ndarray, offset: int,
                         max_pts: int = 20000) -> float:
    """Linear R2 of predicting the increment z[t+offset]-z[t] from z[t].

    THE cheap pre-flight test.  For a valid conditional the increment is noise
    independent of the state, so this is ~0 (Brownian: exactly 0).  For a
    deterministic flow the increment IS a function of the state, so this is
    large and InfoNCE can be solved by a flow-invariant encoder that discards
    z entirely."""
    a = z[:-offset]
    delta = z[offset:] - a
    if len(a) > max_pts:
        sel = np.linspace(0, len(a) - 1, max_pts).astype(int)
        a, delta = a[sel], delta[sel]
    ac = a - a.mean(0, keepdims=True)
    dc = delta - delta.mean(0, keepdims=True)
    G = ac.T @ ac
    G = G + 1e-8 * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    W = np.linalg.solve(G, ac.T @ dc)
    res = dc - ac @ W
    return float(1.0 - (res ** 2).sum() / ((dc ** 2).sum() + 1e-30))


def describe_latents(z: np.ndarray) -> dict:
    """Characterise the DGP and ABORT before training if it cannot support the
    identifiability argument.  Run 1 burned 40 minutes to discover this from
    R2; it costs milliseconds here."""
    delta = z[TIME_OFFSET:] - z[:-TIME_OFFSET]
    step_sd = float(np.sqrt(np.mean(delta ** 2)))
    r2 = drift_predictability(z, TIME_OFFSET)
    zc = z - z.mean(0, keepdims=True)
    ac1 = float(np.mean(np.sum(zc[TIME_OFFSET:] * zc[:-TIME_OFFSET], 0) /
                        (np.sum(zc ** 2, 0) + 1e-30)))
    print(f"    [dgp] step_sd={step_sd:.4f} (ref {LATENT_STEP_SD:.4f})  "
          f"drift_r2={r2:.4f} (max {DRIFT_R2_MAX})  lag{TIME_OFFSET}_ac="
          f"{ac1:.4f}  |z|max={np.abs(z).max():.3f}")
    if r2 > DRIFT_R2_MAX:
        raise RuntimeError(
            f"drift_r2={r2:.3f} > {DRIFT_R2_MAX}: the one-step increment is "
            f"largely a deterministic function of the state, so p(z'|z) is not "
            f"a proper conditional and InfoNCE admits a flow-invariant "
            f"degenerate optimum that carries no information about z.  This is "
            f"exactly what made every seed fail the R2 gate.  Fix: lower "
            f"LORENZ_TARGET_DRIFT_R2, or LATENT_PROCESS='brownian'.")
    return dict(step_sd=step_sd, drift_r2=r2, lag_ac=ac1)


def orthonormal_columns(rows: int, cols: int, rng) -> np.ndarray:
    """Dense [rows, cols] with orthonormal columns (reduced QR).

    Orthonormality keeps the mixing well conditioned so the generator Jacobian
    has no near-zero entries that would blur the ground truth; the Gaussian
    seed makes it dense almost surely, so every declared edge in `gt` is real."""
    assert rows >= cols
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


@dataclass
class Mixer:
    """x1 = tanh(g1 W1 z[:d1]) (N1 neurons) ; x2 = tanh(g2 W2 z) (N2 neurons)."""
    W1: np.ndarray
    W2: np.ndarray
    d1: int
    gain1: float = 1.0
    gain2: float = 1.0

    def pre(self, z):
        return (self.gain1 * z[:, :self.d1] @ self.W1.T,
                self.gain2 * z @ self.W2.T)

    def __call__(self, z):
        p1, p2 = self.pre(z)
        return np.concatenate([np.tanh(p1), np.tanh(p2)], axis=1)

    def generator_jacobian(self, z) -> np.ndarray:
        """dx/dz, [S, C, D].  Unique -- this is what `gt` is derived from."""
        S, d = z.shape
        p1, p2 = self.pre(z)
        s1 = 1.0 - np.tanh(p1) ** 2
        s2 = 1.0 - np.tanh(p2) ** 2
        n1 = self.W1.shape[0]
        J = np.zeros((S, n1 + self.W2.shape[0], d))
        J[:, :n1, :self.d1] = s1[:, :, None] * (self.gain1 * self.W1)[None]
        J[:, n1:, :]        = s2[:, :, None] * (self.gain2 * self.W2)[None]
        return J


def generate_dataset(d: int, sigma_obs: float, data_seed: int) -> dict:
    rng = np.random.default_rng(data_seed)
    d1 = d // 2 + d % 2
    z = make_latents(T_SAMPLES, d, rng)
    W1 = orthonormal_columns(N1, d1, rng)
    W2 = orthonormal_columns(N2, d, rng)
    mix = Mixer(W1=W1, W2=W2, d1=d1)
    sub = z[rng.choice(T_SAMPLES, size=min(5000, T_SAMPLES), replace=False)]
    mix.gain1 = TARGET_PRE_SD / max(float(np.std(sub[:, :d1] @ W1.T)), 1e-12)
    mix.gain2 = TARGET_PRE_SD / max(float(np.std(sub @ W2.T)), 1e-12)
    x = mix(z)
    if sigma_obs > 0:
        x = x + sigma_obs * rng.normal(size=x.shape)
    return dict(z=z, x=x, mix=mix, d=d, d1=d1, d2=d - d1, seed=data_seed)


def ground_truth(d: int, d1: int) -> np.ndarray:
    """gt[C, D] = 1 iff neuron i is generated from latent j.  Support of dx/dz."""
    gt = np.zeros((D_OBS, d), dtype=np.int8)
    gt[:N1, :d1] = 1        # x1 sees z1 only
    gt[N1:, :] = 1          # x2 sees z1 and z2
    return gt


def split_data(data: dict) -> dict:
    """Disjoint train / map-fit / attribution segments."""
    T = T_SAMPLES
    assert MAP_POINTS + ATTR_POINTS < T // 2, "held-out segments too large"
    train_end = T - (MAP_POINTS + ATTR_POINTS) - TIME_OFFSET - 1
    idx_train = np.arange(0, train_end)
    idx_map   = np.arange(train_end, train_end + MAP_POINTS)
    idx_attr  = np.arange(train_end + MAP_POINTS,
                          train_end + MAP_POINTS + ATTR_POINTS)
    if STANDARDIZE:
        mu = data["x"][idx_train].mean(0, keepdims=True)
        sd = data["x"][idx_train].std(0, keepdims=True) + 1e-8
    else:
        mu = np.zeros((1, D_OBS))
        sd = np.ones((1, D_OBS))
    xs = (data["x"] - mu) / sd    # diagonal rescale: zeros of dx/dz unchanged
    return dict(xs=xs, mu=mu, sd=sd, idx_train=idx_train,
                idx_map=idx_map, idx_attr=idx_attr)


# ==============================================================================
# 4. METRIC MACHINERY
# ==============================================================================

def squeeze_to_3d(a: np.ndarray, name: str) -> np.ndarray:
    """Collapse trailing singleton axes ONLY.  Never average over a real axis:
    a signed mean over a lag axis cancels opposite-sign entries and turns a
    genuine edge into a zero."""
    a = np.asarray(a)
    while a.ndim > 3:
        if a.shape[-1] == 1:
            a = a[..., 0]
        else:
            raise ValueError(f"{name}: non-singleton extra axis {a.shape}")
    if a.ndim != 3:
        raise ValueError(f"{name}: expected 3 dims, got {a.shape}")
    return a


def canonicalize_jf(jf: np.ndarray, out_dim: int, name: str) -> np.ndarray:
    """-> [S, O, C]."""
    jf = squeeze_to_3d(jf, name)
    if jf.shape[1] == out_dim and jf.shape[2] == D_OBS:
        return jf
    if jf.shape[2] == out_dim and jf.shape[1] == D_OBS:
        return np.swapaxes(jf, 1, 2)
    raise ValueError(f"{name}: cannot orient {jf.shape} to [S,{out_dim},{D_OBS}]")


def truncated_pinv(jf: np.ndarray, drop: int) -> np.ndarray:
    """Batched pinv of J_f with rank truncation.  [S,O,C] -> [S,C,O]."""
    U, S, Vh = np.linalg.svd(jf, full_matrices=False)
    S = S.copy()
    if drop > 0:
        S[:, -drop:] = 0.0
    inv = np.zeros_like(S)
    good = S > np.maximum(1e-10 * S[:, :1], 1e-30)
    inv[good] = 1.0 / S[good]
    return np.einsum("ski,sk,sok->sio", Vh, inv, U)      # V diag(1/s) U^T


def fit_linear_maps(f: np.ndarray, z: np.ndarray, ridge: float = 1e-6):
    """A_z2e [O,D] with f ~ A z ; B_e2z [O,D] with z ~ f B ; R2 per latent dim."""
    f = np.asarray(f, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    fc = f - f.mean(0, keepdims=True)
    zc = z - z.mean(0, keepdims=True)
    O, D = f.shape[1], z.shape[1]
    G = zc.T @ zc
    G = G + ridge * np.trace(G) / max(D, 1) * np.eye(D)
    A_z2e = np.linalg.solve(G, zc.T @ fc).T
    H = fc.T @ fc
    H = H + ridge * np.trace(H) / max(O, 1) * np.eye(O)
    B_e2z = np.linalg.solve(H, fc.T @ zc)
    ss_res = ((zc - fc @ B_e2z) ** 2).sum(0)
    r2 = 1.0 - ss_res / ((zc ** 2).sum(0) + 1e-30)
    return A_z2e, B_e2z, r2


def align_to_latents(Q: np.ndarray, A_z2e: np.ndarray) -> np.ndarray:
    """R[s,i,d] = dx_i/dz_d = sum_o Q[s,i,o] A_z2e[o,d].  [S,C,O] -> [S,C,D].
    Exactly invariant to f -> M f: Q -> Q M^{-1}, A -> M A."""
    return np.einsum("sio,od->sid", Q, A_z2e)


def aggregate_map(R: np.ndarray) -> np.ndarray:
    """[S,C,D] -> [C,D].  ABSOLUTE value BEFORE the mean: a signed average over
    timepoints cancels a real edge whose sign flips along the trajectory."""
    return np.abs(R).mean(axis=0)


def score_variants(M: np.ndarray) -> Dict[str, np.ndarray]:
    return {"global": (M - M.mean()) / (M.std() + 1e-30),
            "colnorm": (M - M.mean(0, keepdims=True)) /
                       (M.std(0, keepdims=True) + 1e-30)}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if y.sum() == 0:
        return float("nan")
    y = y[np.argsort(-s, kind="mergesort")]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def binary_scores(M: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    out = {}
    for key, s in score_variants(M).items():
        out[f"auroc_{key}"] = _auc(s, gt)
        out[f"auprc_{key}"] = _auprc(s, gt)
    return out


# ==============================================================================
# 5. VALIDATION OF THE METRIC ITSELF
# ==============================================================================

def validate_alignment_metric(data: dict, sp: dict, gt: np.ndarray, seed: int):
    """End-to-end test of the PRIMARY scoring path on a known-perfect encoder.

    Failure means the METRIC is broken, not the model, so the run aborts rather
    than emitting numbers nobody can interpret."""
    rng = np.random.default_rng(seed)
    d = data["d"]
    O = d + OUT_EXTRA
    z = data["z"][sp["idx_attr"]]
    Jg = data["mix"].generator_jacobian(z) / sp["sd"][0][None, :, None]
    Jzx = np.linalg.pinv(Jg)                        # [S, d, C] = dz/dx
    A = rng.normal(size=(O, d))                     # the linear indeterminacy
    if _ARCH["normalize"]:
        u = z @ A.T
        nrm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        f = u / nrm
        I = np.eye(O)
        dfdz = (I[None] - f[:, :, None] * f[:, None, :]) @ A[None] / nrm[:, :, None]
    else:
        f = z @ A.T
        dfdz = np.broadcast_to(A[None], (len(z), O, d))
    jf = dfdz @ Jzx                                 # [S, O, C] = df/dx
    A_z2e, _, r2 = fit_linear_maps(f, z)
    Q = truncated_pinv(jf, PINV_DROP)
    res = binary_scores(aggregate_map(align_to_latents(Q, A_z2e)), gt)
    print(f"    [validate] surrogate auROC={res['auroc_global']:.4f} "
          f"auPRC={res['auprc_global']:.4f} R2min={r2.min():.4f}")
    if not (res["auroc_global"] >= SURROGATE_AUROC_MIN):
        raise RuntimeError(
            f"alignment metric failed its own sanity check "
            f"(auROC={res['auroc_global']:.4f} < {SURROGATE_AUROC_MIN}). "
            f"Do not trust any model comparison until this passes.")
    return res


# ==============================================================================
# 6. MODEL, HEAD, CRITERION, SAMPLER
# ==============================================================================

class ScaledTanh(nn.Module):
    """The paper's output layer: "one layer + scaled tanh" (Appendix B.1)."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return self.scale * torch.tanh(x / self.scale)

    def extra_repr(self):
        return f"scale={self.scale}"


def attach_scaled_tanh(model: nn.Module, scale: float) -> nn.Module:
    """Append the scaled-tanh head IN PLACE, after the model's own Squeeze."""
    head = ScaledTanh(scale)
    for attr in ("net", "module", "layers"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Sequential):
            sub.add_module("scaled_tanh", head)
            return model
    if isinstance(model, nn.Sequential):
        model.add_module("scaled_tanh", head)
        return model
    raise RuntimeError(
        f"cannot append the scaled-tanh head to {type(model).__name__}: no "
        f"nn.Sequential container found.  Run print(model), attach it "
        f"manually, or set ARCH_VARIANT='cosine_sphere'.")


def flat(e: torch.Tensor) -> torch.Tensor:
    return e if e.dim() == 2 else e.reshape(e.shape[0], -1)


_ONCE = {"head": False, "jreg": False, "cache": False, "attr": False,
         "arch": False}


def verify_head(model):
    """Prove the head is actually in the forward path."""
    if _ONCE["head"]:
        return
    with torch.no_grad():
        y = flat(model(torch.randn(8, D_OBS, 1, device=DEVICE) * 50.0))
    m = float(y.abs().max())
    print(f"    [validate] |f|_inf at 50x input scale = {m:.4f} "
          f"(bound = {TANH_SCALE})")
    if TANH_SCALE is not None and m > TANH_SCALE * 1.01:
        raise RuntimeError("scaled-tanh head is NOT in the forward path")
    _ONCE["head"] = True


def build_model(d: int) -> Tuple[nn.Module, nn.Module]:
    model = CEBRA.models.init(MODEL_NAME, D_OBS, NUM_UNITS,
                              d + OUT_EXTRA).to(DEVICE)
    if len(model.get_offset()) != 1:
        raise RuntimeError(
            f"{MODEL_NAME} has receptive field {len(model.get_offset())}; this "
            f"benchmark assumes an instantaneous (offset-1) encoder so that "
            f"J_f is a plain [O, C] matrix per timepoint.")
    got = bool(getattr(model, "normalize", None))
    if got != _ARCH["normalize"]:
        raise RuntimeError(
            f"{MODEL_NAME}.normalize={got} but ARCH_VARIANT={ARCH_VARIANT} "
            f"expects {_ARCH['normalize']}.")
    if TANH_SCALE is not None:
        attach_scaled_tanh(model, TANH_SCALE)
    if not _ONCE["arch"]:
        print(model)
        _ONCE["arch"] = True
    verify_head(model)
    if CRITERION == "cosine":
        crit = CRIT.LearnableCosineInfoNCE(temperature=INIT_TEMPERATURE,
                                           min_temperature=MIN_TEMPERATURE)
    else:
        crit = CRIT.LearnableEuclideanInfoNCE(temperature=INIT_TEMPERATURE,
                                              min_temperature=MIN_TEMPERATURE)
    return model, crit.to(DEVICE)


class TimeSampler:
    """CEBRA's `time` conditional.  Returns 3D [B, C, 1] tensors."""

    def __init__(self, X: torch.Tensor, idx: np.ndarray, gen):
        self.X = X
        self.base = int(idx[0])
        self.hi = len(idx) - TIME_OFFSET - 1
        self.gen = gen

    def __call__(self, batch: int):
        i = torch.randint(0, self.hi, (batch,), generator=self.gen) + self.base
        j = torch.randint(0, self.hi, (batch,), generator=self.gen) + self.base
        i = i.to(self.X.device)
        j = j.to(self.X.device)
        return (self.X[i].unsqueeze(-1),
                self.X[i + TIME_OFFSET].unsqueeze(-1),
                self.X[j].unsqueeze(-1))


# ==============================================================================
# 7. THE JACOBIAN-FROBENIUS REGULARIZER  (paper Eq. 10 / 15)
# ==============================================================================

def jacobian_frobenius_sq(model: nn.Module, x: torch.Tensor,
                          estimator: str = "proj", n_proj: int = 1
                          ) -> torch.Tensor:
    """E_x ||J_f(x)||_F^2, differentiable w.r.t. the model parameters."""
    x = x.detach().requires_grad_(True)
    out = flat(model(x))
    O = out.shape[1]
    total = 0.0
    if estimator == "exact":
        for o in range(O):
            g, = torch.autograd.grad(out[:, o].sum(), x,
                                     create_graph=True, retain_graph=True)
            total = total + (g ** 2).flatten(1).sum(1)
        return total.mean()
    for _ in range(n_proj):
        v = torch.randn(out.shape, device=out.device, dtype=out.dtype)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        g, = torch.autograd.grad((out * v).sum(), x,
                                 create_graph=True, retain_graph=True)
        total = total + (g ** 2).flatten(1).sum(1)
    return (O / n_proj) * total.mean()


def selftest_jreg(model: nn.Module, x: torch.Tensor):
    if _ONCE["jreg"]:
        return
    xs = x[:64]
    exact = float(jacobian_frobenius_sq(model, xs, "exact"))
    est = float(np.mean([float(jacobian_frobenius_sq(model, xs, "proj", 8))
                         for _ in range(24)]))
    rel = abs(est - exact) / max(exact, 1e-12)
    print(f"    [validate] ||J||_F^2 exact={exact:.5f} proj={est:.5f} "
          f"rel.err={rel:.3f}")
    if rel > JREG_SELFTEST_RTOL:
        raise RuntimeError(
            f"Hoffman projection estimator biased by {rel:.1%} -- refusing to "
            f"run a regularized arm on an untrustworthy penalty.")
    _ONCE["jreg"] = True


def lambda_at(it: int, lam_max: float) -> float:
    """0 for LAMBDA_WARMUP iterations, then a linear ramp over LAMBDA_RAMP."""
    if lam_max <= 0.0:
        return 0.0
    if it < LAMBDA_WARMUP:
        return 0.0
    return lam_max * min(1.0, max(0.0, (it - LAMBDA_WARMUP) / float(LAMBDA_RAMP)))


# ==============================================================================
# 8. THE ATTACK -- literal transcription of cebra/solver/base.py::Solver.step
# ==============================================================================

def _l2_normalize(t: torch.Tensor, eps: float = 1e-12):
    flat_ = t.reshape(t.size(0), -1)
    norm = flat_.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
    return t / norm.view(-1, *([1] * (t.dim() - 1)))


def _rand_radius_like(t: torch.Tensor):
    return torch.rand([t.size(0)] + [1] * (t.dim() - 1), device=t.device)


def _proj_l2_ball(adv: torch.Tensor, orig: torch.Tensor, epsilon: float):
    delta = adv - orig
    flat_ = delta.reshape(delta.size(0), -1)
    norm = flat_.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factor = torch.where(norm > epsilon, norm / epsilon, torch.ones_like(norm))
    delta = delta / factor.view(-1, *([1] * (delta.dim() - 1)))
    return orig + delta


def eps_ball_init(x_ref: torch.Tensor, norm: str, eps: float,
                  lo: float, hi: float) -> torch.Tensor:
    """PGD's random start; also the noise_2x control arm on its own."""
    if norm == "linf":
        x = x_ref + torch.empty_like(x_ref).uniform_(-eps, eps)
    elif norm == "l2":
        noise = _l2_normalize(torch.randn_like(x_ref))
        x = x_ref + noise * _rand_radius_like(x_ref) * eps
    else:
        raise ValueError(f"unknown attack norm {norm!r}")
    x = x.clone().detach()
    if ADV_CLIP_RANGE:
        x = x.clamp(lo, hi)
    return x


def pgd_attack(model, crit, x_ref, x_pos, x_neg, norm, eps, lo, hi):
    """Maximise the InfoNCE loss w.r.t. the REFERENCE inputs only."""
    alpha = adv_alpha(eps)
    x_adv = eps_ball_init(x_ref, norm, eps, lo, hi).requires_grad_(True)

    e_pos = e_neg = None
    if ADV_CACHE_POS_NEG:
        with torch.no_grad():
            e_pos = flat(model(x_pos)).detach()
            e_neg = flat(model(x_neg)).detach()

    for _ in range(ADV_STEPS):
        with torch.enable_grad():
            ep = e_pos if e_pos is not None else flat(model(x_pos))
            en = e_neg if e_neg is not None else flat(model(x_neg))
            loss, _, _ = crit(flat(model(x_adv)), ep, en)
        grad_x, = torch.autograd.grad(loss, x_adv, retain_graph=False,
                                      create_graph=False)
        with torch.no_grad():
            if norm == "linf":
                x_adv = x_adv + alpha * grad_x.sign()
                x_adv = torch.max(torch.min(x_adv, x_ref + eps), x_ref - eps)
            else:
                x_adv = x_adv + alpha * _l2_normalize(grad_x)
                x_adv = _proj_l2_ball(x_adv, x_ref, eps)
            if ADV_CLIP_RANGE:
                x_adv = x_adv.clamp(lo, hi)
        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()


def selftest_adv_cache(model, crit, x_ref, x_pos, x_neg, norm, eps, lo, hi):
    """Prove ADV_CACHE_POS_NEG changes nothing but the runtime."""
    global ADV_CACHE_POS_NEG
    if _ONCE["cache"]:
        return
    keep = ADV_CACHE_POS_NEG
    cpu_state = torch.random.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)

    def _run(flag):
        global ADV_CACHE_POS_NEG
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        ADV_CACHE_POS_NEG = flag
        return pgd_attack(model, crit, x_ref, x_pos, x_neg, norm, eps, lo, hi)

    a = _run(False)
    b = _run(True)
    ADV_CACHE_POS_NEG = keep
    dev = float((a - b).abs().max())
    print(f"    [validate] adv cache equivalence: max|dx_adv| = {dev:.3e} "
          f"(tol {ADV_CACHE_RTOL:.0e}, using cache={keep})")
    if dev > ADV_CACHE_RTOL:
        raise RuntimeError(
            f"caching positive/negative changed the attack by {dev:.3e}. "
            f"Set ADV_CACHE_POS_NEG=False and take the 2.4x slowdown.")
    _ONCE["cache"] = True


# ==============================================================================
# 9. TRAINING
# ==============================================================================

def _update(model, crit, opt, x_ref, x_pos, x_neg, lam, x_jreg, gen):
    """One optimizer.step(), optionally carrying the lambda ||J||_F^2 penalty."""
    opt.zero_grad(set_to_none=True)
    loss, align, uniform = crit(flat(model(x_ref)), flat(model(x_pos)),
                                flat(model(x_neg)))
    total = loss
    jval = 0.0
    if lam > 0.0:
        xj = x_jreg
        if JREG_SUBBATCH and JREG_SUBBATCH < xj.shape[0]:
            sel = torch.randint(0, xj.shape[0], (JREG_SUBBATCH,),
                                generator=gen).to(xj.device)
            xj = xj[sel]
        jf2 = jacobian_frobenius_sq(model, xj, JREG_ESTIMATOR, JREG_NPROJ)
        total = loss + lam * jf2
        jval = float(jf2)
    total.backward()
    opt.step()
    return float(loss), jval


TWO_UPDATE = ("clean_double", "fork_double", "noise_double")


def expected_steps(scheme: str) -> int:
    return MAX_ITER * (2 if scheme in TWO_UPDATE else 1)


@torch.no_grad()
def eval_clean_nce(model, crit, sampler, n_batches: int, batch: int) -> float:
    """Held-out clean InfoNCE.  This -- and only this -- drives epsilon
    selection, mirroring the paper's rule for lambda."""
    vals = []
    for _ in range(n_batches):
        xr, xp, xn = sampler(batch)
        loss, _, _ = crit(flat(model(xr)), flat(model(xp)), flat(model(xn)))
        vals.append(float(loss))
    return float(np.mean(vals))


def train_arm(arm: str, cfg: dict, data: dict, sp: dict, seed: int,
              verbose_every: int = 0):
    scheme, lam_max = cfg["scheme"], cfg["lam"]
    norm, eps = cfg.get("norm"), cfg.get("eps")
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)          # CPU generator: portable

    model, crit = build_model(data["d"])
    opt = torch.optim.Adam(list(model.parameters()) + list(crit.parameters()),
                           lr=LEARNING_RATE)
    X = torch.from_numpy(sp["xs"].astype(np.float32)).to(DEVICE)
    sampler = TimeSampler(X, sp["idx_train"], gen)
    held = TimeSampler(X, sp["idx_map"], torch.Generator().manual_seed(seed + 7))
    lo = float(X[sp["idx_train"]].min())
    hi = float(X[sp["idx_train"]].max())

    if lam_max > 0:
        selftest_jreg(model, X[sp["idx_train"][:64]].unsqueeze(-1))
    if scheme == "fork_double":
        xr, xp, xn = sampler(min(256, BATCH_SIZE))
        selftest_adv_cache(model, crit, xr, xp, xn, norm, eps, lo, hi)

    def lam_for(which: str, it: int) -> float:
        if JREG_ON == "clean_only" and which != "clean":
            return 0.0
        if JREG_ON == "adv_only" and which != "adv":
            return 0.0
        return lambda_at(it, lam_max)

    n_steps, hist, t0 = 0, [], time.time()
    for it in range(MAX_ITER):
        x_ref, x_pos, x_neg = sampler(BATCH_SIZE)
        nce_c = nce_a = float("nan")
        j_c = j_a = 0.0

        # ---- update 1: the fork's UNCONDITIONAL clean update -----------------
        if scheme != "adv_single":
            nce_c, j_c = _update(model, crit, opt, x_ref, x_pos, x_neg,
                                 lam_for("clean", it), x_ref, gen)
            n_steps += 1

        # ---- update 2 -------------------------------------------------------
        if scheme in ("fork_double", "adv_single"):
            # NOTE: the attack is built AFTER the clean step, i.e. with the
            # already-updated parameters -- exactly as in Solver.step.
            x2 = pgd_attack(model, crit, x_ref, x_pos, x_neg, norm, eps, lo, hi)
            xj = x_ref if JREG_AT == "clean" else x2
            nce_a, j_a = _update(model, crit, opt, x2, x_pos, x_neg,
                                 lam_for("adv", it), xj, gen)
            n_steps += 1
        elif scheme == "noise_double":
            x2 = eps_ball_init(x_ref, norm, eps, lo, hi)
            nce_a, j_a = _update(model, crit, opt, x2, x_pos, x_neg,
                                 lam_for("adv", it), x_ref, gen)
            n_steps += 1
        elif scheme == "clean_double":
            nce_a, j_a = _update(model, crit, opt, x_ref, x_pos, x_neg,
                                 lam_for("adv", it), x_ref, gen)
            n_steps += 1

        bad = [v for v in (nce_c, nce_a)
               if not math.isnan(v) and (not math.isfinite(v) or v < NCE_FLOOR)]
        if bad:
            raise RuntimeError(
                f"{arm}: InfoNCE diverged at it={it} (loss={bad[0]:.4e} < "
                f"{NCE_FLOOR:.0e}, tau={float(crit.temperature):.4f}).  An "
                f"unbounded head makes Euclidean InfoNCE unbounded below, and a "
                f"DETERMINISTIC latent transition makes it degenerate even with "
                f"a bounded head.  Check LATENT_PROCESS and ARCH_VARIANT.")

        hist.append((nce_c, nce_a, max(j_c, j_a), lambda_at(it, lam_max)))
        if verbose_every and (it % verbose_every == 0 or it == MAX_ITER - 1):
            print(f"      [{arm:16s}] it={it:6d} nce={nce_c:8.4f} "
                  f"nce2={nce_a:8.4f} |J|^2={max(j_c, j_a):9.4f} "
                  f"lam={lambda_at(it, lam_max):.3f} "
                  f"tau={float(crit.temperature):.4f}")

    exp = expected_steps(scheme)
    assert n_steps == exp, f"{arm}: {n_steps} steps, expected {exp}"
    model.eval()
    tail = hist[-50:]
    nce_clean_train = float(np.nanmean([h[0] for h in tail]))
    ev = eval_clean_nce(model, crit, held, EVAL_NCE_BATCHES,
                        min(BATCH_SIZE, 2048))
    tau = float(crit.temperature)
    # Collapse diagnostics.  tau resting ON the floor means the critic wanted an
    # even sharper similarity -- the signature of a degenerate optimum, which is
    # what the deterministic-Lorenz run showed in EVERY attacked arm.  A large
    # positive held-out minus training gap means trajectory memorisation.
    tau_pinned = tau <= MIN_TEMPERATURE * 1.001
    gap = ev - nce_clean_train
    if tau_pinned:
        print(f"      !! [{arm}] tau pinned at the MIN_TEMPERATURE floor "
              f"({MIN_TEMPERATURE}): the critic is saturated, treat this arm's "
              f"representation as suspect.")
    if gap > NCE_GAP_MAX:
        print(f"      !! [{arm}] held-out InfoNCE exceeds training by "
              f"{gap:+.3f} nats (>{NCE_GAP_MAX}): the encoder is memorising the "
              f"trajectory rather than learning the latent geometry.")
    return dict(
        model=model, crit=crit, n_steps=n_steps, seconds=time.time() - t0,
        nce_clean_train=nce_clean_train,
        nce_second_train=float(np.nanmean([h[1] for h in tail])),
        eval_nce=ev, nce_gap=gap, tau_pinned=bool(tau_pinned),
        final_jfro=float(np.mean([h[2] for h in tail])),
        temperature=tau)


@torch.no_grad()
def embed(model, xs: np.ndarray, idx: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = []
    for s in range(0, len(idx), chunk):
        xb = torch.from_numpy(xs[idx[s:s + chunk]].astype(np.float32)).to(DEVICE)
        out.append(flat(model(xb.unsqueeze(-1))).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


# ==============================================================================
# 10. ATTRIBUTION (library-provided Jacobian; the pinv is taken here)
# ==============================================================================

def _pick(dct: dict, keys: Sequence[str]):
    for k in keys:
        for kk in dct:
            if str(kk).lower().replace("_", "-") == k.lower().replace("_", "-"):
                return dct[kk]
    return None


def _to_np(v):
    if v is None:
        return None
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def library_jacobian(model, xs, idx, out_dim):
    """Call cebra.attribution for J_f (the paper's "neuron gradient")."""
    import cebra.attribution
    x = torch.from_numpy(xs[idx].astype(np.float32)).to(DEVICE).unsqueeze(-1)
    if not _ONCE["attr"]:
        try:
            print("    [attr] registry options:", cebra.attribution.get_options())
        except Exception as exc:
            print("    [attr] get_options() unavailable:", exc)
    errors = []
    for name in ATTR_METHOD_CANDIDATES:
        for kw in (dict(model=model, input_data=x, output_dimension=out_dim,
                        num_batches=ATTR_NUM_BATCHES),
                   dict(model=model, input_data=x, output_dimension=out_dim),
                   dict(model=model, input_data=x)):
            try:
                res = cebra.attribution.init(name, **kw).compute_attribution_map()
            except Exception as exc:
                errors.append(f"{name}{list(kw)}: {type(exc).__name__}: {exc}")
                continue
            jf = jfinv = None
            if isinstance(res, dict):
                if not _ONCE["attr"]:
                    print(f"    [attr] '{name}' -> keys: "
                          f"{sorted(map(str, res.keys()))}")
                jf = _pick(res, ATTR_JF_KEYS)
                jfinv = _pick(res, ATTR_JFINV_KEYS)
            elif isinstance(res, (tuple, list)) and len(res) == 2:
                jf, jfinv = res
            else:
                jf = res
            jf = _to_np(jf)
            if jf is None:
                errors.append(f"{name}: no J_f in {type(res)}")
                continue
            _ONCE["attr"] = True
            return canonicalize_jf(jf, out_dim, f"jf[{name}]"), _to_np(jfinv)
    raise RuntimeError("cebra.attribution could not be called. Tried:\n  " +
                       "\n  ".join(errors))


# ==============================================================================
# 11. PER-SEED RUNNER
# ==============================================================================

def run_seed(arms: Dict[str, dict], d: int, sigma_obs: float,
             seed: int, tag: str) -> List[dict]:
    print(f"\n--- [{tag}] d={d} sigma_obs={sigma_obs} seed={seed} ---")
    data = generate_dataset(d, sigma_obs, seed)
    gt = ground_truth(d, data["d1"])
    sp = split_data(data)
    print(f"    gt density={gt.mean():.3f}  d1={data['d1']} d2={data['d2']}")
    dgp = describe_latents(data["z"])          # aborts before any training
    validate_alignment_metric(data, sp, gt, seed)

    rows = []
    for arm, cfg in arms.items():
        tr = train_arm(arm, cfg, data, sp, seed,
                       verbose_every=max(1, MAX_ITER // 4))
        f_map = embed(tr["model"], sp["xs"], sp["idx_map"])
        A_z2e, _, r2 = fit_linear_maps(f_map, data["z"][sp["idx_map"]])
        jf, jfinv_lib = library_jacobian(tr["model"], sp["xs"], sp["idx_attr"],
                                          d + OUT_EXTRA)
        jf = jf.astype(np.float64)
        Q = truncated_pinv(jf, PINV_DROP)
        M_inv = aggregate_map(align_to_latents(Q, A_z2e))
        res = binary_scores(M_inv, gt)
        # the un-inverted "neuron gradient", for the paper's own comparison
        res_fwd = {f"{k}_jf": v for k, v in
                   binary_scores(aggregate_map(np.swapaxes(jf, 1, 2)), gt).items()}
        xcheck = float("nan")
        if jfinv_lib is not None:
            try:
                Ql = np.swapaxes(canonicalize_jf(jfinv_lib, d + OUT_EXTRA,
                                                 "jfinv_lib"), 1, 2)
                Ml = aggregate_map(align_to_latents(Ql.astype(np.float64), A_z2e))
                xcheck = float(np.corrcoef(Ml.ravel(), M_inv.ravel())[0, 1])
            except Exception as exc:
                print(f"    [attr] library jf-inv cross-check skipped: {exc}")
        sing = np.linalg.svd(jf, compute_uv=False)
        rows.append(dict(
            stage=tag, arm=arm, scheme=cfg["scheme"], lam=cfg["lam"],
            norm=cfg.get("norm") or "", eps=cfg.get("eps") if cfg.get("eps")
            is not None else "", seed=seed, d=d, sigma_obs=sigma_obs,
            scale=SCALE_NAME, latent_process=LATENT_PROCESS,
            drift_r2=dgp["drift_r2"], step_sd=dgp["step_sd"],
            r2_mean=float(r2.mean()), r2_min=float(r2.min()),
            eval_nce=tr["eval_nce"], nce_clean=tr["nce_clean_train"],
            nce_second=tr["nce_second_train"], nce_gap=tr["nce_gap"],
            tau_pinned=tr["tau_pinned"],
            final_jfro=tr["final_jfro"],
            temperature=tr["temperature"], seconds=tr["seconds"],
            n_steps=tr["n_steps"],
            sv_ratio=float(np.median(sing[:, -1] / (sing[:, 0] + 1e-30))),
            libinv_corr=xcheck, **res, **res_fwd))
        r = rows[-1]
        print(f"    [{arm:16s}] auROC={r['auroc_global']:.4f} "
              f"auPRC={r['auprc_global']:.4f} "
              f"(jf-only={r['auroc_global_jf']:.4f}) "
              f"R2={r['r2_mean']:.4f} evalNCE={r['eval_nce']:+.4f} "
              f"gap={r['nce_gap']:+.3f} tau={r['temperature']:.3f} "
              f"steps={r['n_steps']} {r['seconds']:.0f}s")

    r2s = [r["r2_mean"] for r in rows]
    ok = min(r2s) >= R2_MIN and (max(r2s) - min(r2s)) <= R2_PARITY_MAX
    if not ok:
        print(f"    !! seed EXCLUDED: R2 range [{min(r2s):.3f}, {max(r2s):.3f}] "
              f"violates R2_MIN={R2_MIN} / parity={R2_PARITY_MAX}.  Comparing "
              f"attribution across arms of unequal representation quality "
              f"measures representation quality, not attribution.")
        if min(r2s) < R2_MIN:
            print(f"       If EVERY arm (including clean `cebra`) is below the "
                  f"floor, the fault is the DGP, not the arms: the encoder "
                  f"cannot linearly recover z at all.  Check drift_r2="
                  f"{dgp['drift_r2']:.3f}, tau, and nce_gap above.")
    for r in rows:
        r["included"] = bool(ok)
    return rows


# ==============================================================================
# 11b. CSV IO -- append, resume, and NEVER duplicate
# ==============================================================================

_STR_COLS = ("stage", "arm", "scheme", "norm", "scale", "latent_process")
_INT_COLS = ("seed", "d", "n_steps")


def _coerce(k: str, v):
    if k in ("included", "tau_pinned"):
        return str(v).strip().lower() in ("true", "1", "yes")
    if k in _STR_COLS:
        return v
    if v == "" or v is None:
        return v
    try:
        return int(v) if k in _INT_COLS else float(v)
    except (TypeError, ValueError):
        return v


def load_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    import csv as _csv
    with open(path, newline="") as fh:
        return [{k: _coerce(k, v) for k, v in row.items()}
                for row in _csv.DictReader(fh)]


def completed_seeds(rows: List[dict], n_arms: int) -> set:
    """Seeds already fully written.

    Per-SEED granularity because the R2 parity gate is computed across all arms
    of one seed.  Without this a crash at seed 8 of 10 followed by a rerun
    appends DUPLICATE rows, which double-count in arm_table and misalign the
    paired arrays in comparisons."""
    seen: Dict[tuple, int] = {}
    for r in rows:
        key = (r["seed"], r["d"], r["sigma_obs"])
        seen[key] = seen.get(key, 0) + 1
    return {k for k, v in seen.items() if v >= n_arms}


def append_rows(rows: List[dict], path: str):
    """Flush after every seed.  Writing only at the end would discard every
    completed seed if the run dies at seed 8 of 10."""
    import csv as _csv
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def run_all_seeds(arms: Dict[str, dict], csv: str, seed0: int, tag: str,
                  sigmas: Sequence[float]) -> List[dict]:
    """Shared seed loop for both stages, with resume."""
    prior = load_rows(csv)
    done = completed_seeds(prior, len(arms))
    if done:
        print(f"[resume] {csv}: {len(done)} seed(s) already complete, skipping "
              f"them and reusing their rows for the statistics.")
    rows: List[dict] = list(prior)
    for d in D_LATENT_LIST:
        for sigma_obs in sigmas:
            for i in range(N_SEEDS):
                seed = seed0 + SEED_STRIDE * i
                if (seed, d, sigma_obs) in done:
                    continue
                r = run_seed(arms, d, sigma_obs, seed, tag)
                append_rows(r, csv)
                rows.extend(r)
    return rows


# ==============================================================================
# 12. STATISTICS
# ==============================================================================

def paired_stats(a, b, rng) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    dif = a[m] - b[m]
    n = len(dif)
    out = dict(n=n, mean_diff=float(dif.mean()) if n else float("nan"),
               n_wins=int((dif > 0).sum()),
               diffs=" ".join(f"{v:+.4f}" for v in dif))
    if n < 2:
        out.update(p=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                   test="n/a")
        return out
    if HAVE_SCIPY and np.any(dif != 0):
        out["p"], out["test"] = float(_sps.wilcoxon(dif).pvalue), "wilcoxon"
    else:
        nz, k = int((dif != 0).sum()), int((dif > 0).sum())
        out["p"] = float(_sps.binomtest(k, nz, 0.5).pvalue) if (
            HAVE_SCIPY and nz) else float("nan")
        out["test"] = "sign"
    boot = np.array([rng.choice(dif, size=n, replace=True).mean()
                     for _ in range(N_BOOTSTRAP)])
    out["ci_lo"], out["ci_hi"] = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
    return out


def arm_table(rows: List[dict], arms: Sequence[str], only_included=True):
    inc = [r for r in rows if r["included"] or not only_included]
    print(f"\n{'arm':<17}{'steps':>6}{'auROC':>18}{'auPRC':>18}"
          f"{'jf-only':>9}{'R2':>8}{'evalNCE':>10}{'gap':>8}"
          f"{'|J|^2':>9}{'sv':>9}{'s':>7}")
    for arm in arms:
        sel = [r for r in inc if r["arm"] == arm]
        if not sel:
            print(f"{arm:<17}{'--':>6}")
            continue
        g = lambda k: np.array([r[k] for r in sel], dtype=float)
        sd = lambda t: t.std(ddof=1) if len(t) > 1 else 0.0
        v, p = g(PRIMARY_METRIC), g("auprc_global")
        print(f"{arm:<17}{sel[0]['n_steps']:>6}"
              f"{v.mean():>10.4f}+-{sd(v):.4f}{p.mean():>10.4f}+-{sd(p):.4f}"
              f"{g('auroc_global_jf').mean():>9.4f}{g('r2_mean').mean():>8.4f}"
              f"{g('eval_nce').mean():>+10.4f}{g('nce_gap').mean():>+8.3f}"
              f"{g('final_jfro').mean():>9.3f}"
              f"{g('sv_ratio').mean():>9.2e}{g('seconds').mean():>7.0f}")


def comparisons(rows, pairs, label=""):
    rng = np.random.default_rng(0)
    inc = [r for r in rows if r["included"]]

    def get(arm):
        return np.array([r[PRIMARY_METRIC] for r in
                         sorted([x for x in inc if x["arm"] == arm],
                                key=lambda z: (z["d"], z["sigma_obs"], z["seed"]))])

    print(f"\npaired comparisons on {PRIMARY_METRIC} {label}")
    print(f"{'comparison':<34}{'diff':>9}{'95% CI':>22}{'p':>8}{'wins':>7}"
          f"   per-seed")
    for x, y, tag in pairs:
        if not (any(r["arm"] == x for r in inc) and
                any(r["arm"] == y for r in inc)):
            continue
        s = paired_stats(get(x), get(y), rng)
        print(f"{x + ' - ' + y:<34}{s['mean_diff']:>+9.4f}"
              f"  [{s['ci_lo']:>+7.4f},{s['ci_hi']:>+7.4f}]{s['p']:>8.3f}"
              f"{str(s['n_wins']) + '/' + str(s['n']):>7}   {s['diffs']}"
              f"{'   <-- ' + tag if tag else ''}")


# ==============================================================================
# 13. STAGE 1 -- EPSILON SELECTION (reads the LOSS, never auROC)
# ==============================================================================

def stage_select() -> Dict[str, Optional[float]]:
    set_scale(SELECT_SCALE)
    csv = f"select_{RESULT_TAG}.csv"
    arms = {"cebra": dict(scheme="clean_single", lam=0.0)}
    for norm in ATTACK_NORMS:
        for eps in ADV_EPS_GRID[norm]:
            arms[f"acorn_{norm}_{eps:g}"] = dict(scheme="fork_double", lam=0.0,
                                                 norm=norm, eps=eps)
    print(f"\n{'#' * 78}\n# STAGE 1: epsilon selection -- {len(arms)} arms x "
          f"{N_SEEDS} seeds\n{'#' * 78}")

    rows = run_all_seeds(arms, csv, SELECT_SEED0, "select", SIGMA_OBS_SWEEP[:1])

    # ONLY included seeds may inform the choice.  The previous version averaged
    # over every row regardless of `included`, so a run in which the R2 gate
    # excluded 100% of seeds still printed a confident selection table built
    # entirely from void data.
    inc = [r for r in rows if r["included"]]
    n_inc, n_tot = len(inc) // len(arms), len(rows) // len(arms)
    print(f"\n[stage 1] included seeds: {n_inc} of {n_tot}")
    if n_inc == 0:
        raise RuntimeError(
            "stage 1 produced ZERO included seeds -- every seed failed the R2 "
            "gate, so there is no data to select epsilon from.  Do NOT read the "
            "auROC column: it is noise.  Check, in this order: (1) drift_r2 in "
            "the CSV -- if it is large the latent transition is deterministic "
            "and InfoNCE is degenerate; (2) temperature -- pinned at "
            f"{MIN_TEMPERATURE} means the critic saturated; (3) nce_gap -- a "
            "large positive value means trajectory memorisation.")

    def mean_of(arm, key):
        v = [r[key] for r in inc if r["arm"] == arm]
        return float(np.mean(v)) if v else float("nan")

    ref = mean_of("cebra", "eval_nce")
    print(f"\n{'=' * 78}\nSELECTION  (criterion: held-out clean InfoNCE within "
          f"{NCE_TOL_ABS} nats of baseline {ref:+.4f})")
    print("auROC is printed for transparency but is NOT used to select.")
    print(f"{'=' * 78}")
    print(f"{'arm':<20}{'eps':>9}{'evalNCE':>10}{'delta':>9}{'ok':>5}"
          f"{'auROC':>9}{'R2':>8}{'sv':>10}")
    chosen: Dict[str, Optional[float]] = {}
    for norm in ATTACK_NORMS:
        feasible = []
        for eps in ADV_EPS_GRID[norm]:
            arm = f"acorn_{norm}_{eps:g}"
            nce = mean_of(arm, "eval_nce")
            delta = nce - ref
            ok = np.isfinite(delta) and delta <= NCE_TOL_ABS
            if ok:
                feasible.append(eps)
            print(f"{arm:<20}{eps:>9.4f}{nce:>+10.4f}{delta:>+9.4f}"
                  f"{'yes' if ok else 'no':>5}"
                  f"{mean_of(arm, 'auroc_global'):>9.4f}"
                  f"{mean_of(arm, 'r2_mean'):>8.4f}"
                  f"{mean_of(arm, 'sv_ratio'):>10.2e}")
        chosen[norm] = max(feasible) if feasible else None
        grid = ADV_EPS_GRID[norm]
        if chosen[norm] is None:
            print(f"  !! [{norm}] no epsilon keeps InfoNCE at its minimum. "
                  f"Every value on the grid degrades the representation, so "
                  f"there is no defensible choice -- extend the grid DOWNWARD "
                  f"before falling back to a default.")
        else:
            print(f"  -> [{norm}] selected eps = {chosen[norm]:g} "
                  f"(largest feasible)")
            # BOTH boundaries matter.  In the first Lorenz run l2 selected
            # 1.4142, the top of its grid, with no warning at all -- meaning the
            # feasible region was never bracketed from above and the "largest
            # feasible epsilon" is an artifact of where the grid happened to stop.
            if chosen[norm] >= max(grid) - 1e-12:
                print(f"  !! [{norm}] the selected epsilon is the LARGEST value "
                      f"on the grid, so the feasible region was not bracketed "
                      f"from above.  Extend the grid UPWARD and re-select "
                      f"before treating {chosen[norm]:g} as 'the largest "
                      f"epsilon the loss tolerates'.")
            if chosen[norm] <= min(grid) + 1e-12 and len(feasible) == 1:
                print(f"  !! [{norm}] only the SMALLEST grid value is feasible; "
                      f"extend the grid DOWNWARD.")
    print(f"wrote {csv}")
    return chosen


# ==============================================================================
# 14. STAGE 2 -- EVALUATION with epsilon frozen
# ==============================================================================

def stage_evaluate(chosen: Dict[str, Optional[float]]):
    set_scale(EVALUATE_SCALE)
    csv = f"evaluate_{RESULT_TAG}.csv"

    arms = {
        "cebra":     dict(scheme="clean_single", lam=0.0),
        "cebra_2x":  dict(scheme="clean_double", lam=0.0),
        "xcebra":    dict(scheme="clean_single", lam=LAMBDA_MAX),
        "xcebra_2x": dict(scheme="clean_double", lam=LAMBDA_MAX),
    }
    acorn_names = {}
    for norm in ATTACK_NORMS:
        eps = chosen.get(norm)
        if eps is None:
            print(f"[warn] no selected epsilon for {norm}; falling back to the "
                  f"fork default 0.05 (linf) / 0.354 (l2).  Say so in the "
                  f"paper: this value was NOT tuned.")
            eps = 0.05 if norm == "linf" else round(0.05 * math.sqrt(D_OBS), 4)
        name = f"acorn_{norm}"
        arms[name] = dict(scheme="fork_double", lam=0.0, norm=norm, eps=eps)
        acorn_names[norm] = name
    # perturbation-magnitude control, matched to the linf attack
    n0 = ATTACK_NORMS[0]
    arms["noise_2x"] = dict(scheme="noise_double", lam=0.0, norm=n0,
                            eps=arms[acorn_names[n0]]["eps"])
    # do the two mechanisms stack?
    arms["acorn_xreg"] = dict(scheme="fork_double", lam=LAMBDA_MAX, norm=n0,
                              eps=arms[acorn_names[n0]]["eps"])

    print(f"\n{'#' * 78}\n# STAGE 2: evaluation -- {len(arms)} arms x "
          f"{N_SEEDS} seeds, epsilon FROZEN")
    for k, v in arms.items():
        print(f"#   {k:<16} scheme={v['scheme']:<13} lam={v['lam']:<5} "
              f"norm={v.get('norm') or '-':<5} eps={v.get('eps')}")
    print(f"{'#' * 78}")

    rows = run_all_seeds(arms, csv, EVAL_SEED0, "evaluate", SIGMA_OBS_SWEEP)

    n_arms = max(1, len(arms))
    inc = [r for r in rows if r["included"]]
    print("\n" + "=" * 78)
    print(f" RESULTS   included seeds: {len(inc) // n_arms} of "
          f"{len(rows) // n_arms}")
    print(f" latent process: {LATENT_PROCESS}")
    print(f" primary metric: {PRIMARY_METRIC}")
    print(f" primary comparison: {acorn_names[n0]} vs cebra_2x "
          f"(compute matched, epsilon selected on disjoint seeds by loss only)")
    print("=" * 78)
    if not inc:
        print(" !! ZERO included seeds -- the table below is empty by design. "
              "Nothing here is reportable.")
    arm_table(rows, list(arms))

    best = acorn_names[n0]
    pairs = [(best, "cebra_2x", "PRIMARY"),
             (best, "noise_2x", "adversarial direction vs matched jitter"),
             ("cebra_2x", "cebra", "size of the free-second-update confound"),
             ("xcebra", "cebra", "the paper's own claim, on your data"),
             (best, "xcebra", "implicit vs explicit regularization"),
             (best, "cebra", "the 'as-published' number"),
             ("acorn_xreg", best, "do the two mechanisms stack?")]
    if len(ATTACK_NORMS) > 1:
        other = acorn_names[ATTACK_NORMS[1]]
        pairs.insert(1, (best, other, "linf vs l2"))
        pairs.append((other, "cebra_2x", "l2 vs compute-matched control"))
    comparisons(rows, pairs)

    print("""
How to read this
  cebra_2x - cebra    the free second optimizer.step() in Solver.step, on its
                      own.  In the first pilot this was +0.058 of the +0.065
                      apparent acorn-over-cebra gap -- 89% of it.  Subtract
                      this before crediting anything to the attack.
  acorn - noise_2x    same update count AND same perturbation magnitude, only
                      the direction differs.  If this is ~0, PGD is buying
                      nothing that random jitter would not.
  xcebra - cebra      the paper's own claim.  If this is not positive,
                      something upstream is wrong and the rest is void.
  acorn - cebra_2x    PRIMARY.  Positive => the attack itself buys identifiable
                      attribution.  First-order linf adversarial training
                      penalises eps*||grad_x L||_1, the dual-norm analogue of
                      the paper's Frobenius penalty, so this is the
                      mechanistically expected direction.
  acorn - xcebra      is the implicit penalty competitive with the explicit
                      one, at ~5x the cost?
  acorn_xreg - acorn  do the two mechanisms stack, or are they redundant?

Only the PRIMARY line supports a confirmatory claim; the rest are uncorrected
for multiplicity.  Epsilon was chosen on SELECT seeds by held-out InfoNCE
alone, so the auROC reported here is not inflated by that choice.""")
    print(f"\nwrote {csv}")


# ==============================================================================
# 15. MAIN
# ==============================================================================

def main():
    _banner()
    chosen = dict(SELECTED_MANUAL)
    if STAGE in ("select", "both"):
        chosen = stage_select()
    if STAGE in ("evaluate", "both"):
        stage_evaluate(chosen)


if __name__ == "__main__":
    main()


# from __future__ import annotations

# import math
# import os
# import time
# import warnings
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Sequence, Tuple

# import numpy as np

# # ==============================================================================
# # 1. CONFIG
# # ==============================================================================

# DRY_RUN = False                # <<< smoke test.  Set False for real runs.
# STAGE   = "both"               # "select" | "evaluate" | "both"

# # where the ORIGINAL (unmodified) CEBRA lives.  We deliberately use clean CEBRA
# # for the building blocks -- models, criterions, attribution -- because the
# # attack is reimplemented in this file as a literal transcription of the
# # advisor's Solver.step.  Importing the advisor's fork instead would apply the
# # attack twice.  So "advisor's adversarial solver present: False" is EXPECTED
# # and correct here.
# CEBRA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                          "CEBRA-original")

# # --- architecture -------------------------------------------------------------
# ARCH_VARIANT = "euclidean_tanh"

# _ARCH_TABLE = {
#     # PRIMARY: the paper's own head.  Bounded -> Euclidean InfoNCE is well
#     # posed even at lambda = 0, and the box [-s,s]^O can be filled by a LINEAR
#     # map of z, so linear identifiability survives.  Theory-matched: Brownian
#     # positives are conditionally Gaussian, so Euclidean similarity is the
#     # identifiable choice (Zimmermann et al. 2021).
#     "euclidean_tanh":   dict(model="offset1-model-mse", normalize=False,
#                              criterion="euclidean", out_extra=0, pinv_drop=0,
#                              tanh_scale=1.0),
#     # unbounded head: DIVERGES whenever lambda = 0.  Kept only to document it.
#     "euclidean_linear": dict(model="offset1-model-mse", normalize=False,
#                              criterion="euclidean", out_extra=0, pinv_drop=0,
#                              tanh_scale=None),
#     # bounded by construction (||f|| = 1); the production ACORN setting.
#     # J_f = (1/|u|)(I - f f^T) A is rank deficient by exactly one, so
#     # out_extra=1 and pinv_drop=1 are mandatory: keeping the null direction
#     # lets the pinv divide by a numerically zero singular value.
#     "cosine_sphere":    dict(model="offset1-model",     normalize=True,
#                              criterion="cosine",        out_extra=1, pinv_drop=1,
#                              tanh_scale=None),
# }
# _ARCH      = _ARCH_TABLE[ARCH_VARIANT]
# MODEL_NAME = _ARCH["model"]
# CRITERION  = _ARCH["criterion"]
# OUT_EXTRA  = _ARCH["out_extra"]
# PINV_DROP  = _ARCH["pinv_drop"]
# TANH_SCALE = _ARCH["tanh_scale"]
# NUM_UNITS  = 128

# # --- data generating process (Figure 5 / Appendix B.1) ------------------------
# D_LATENT_LIST   = [6]
# N1, N2          = 25, 25
# D_OBS           = N1 + N2
# BROWNIAN_SIGMA  = 0.10
# TARGET_PRE_SD   = 0.80
# SIGMA_OBS_SWEEP = [0.00]
# TIME_OFFSET     = 1
# STANDARDIZE     = False        # tanh output already lives in [-1,1]; leaving
#                                # this off keeps epsilon in the same units the
#                                # advisor's runs use on raw data

# # --- optimisation -------------------------------------------------------------
# LEARNING_RATE    = 3e-4
# MIN_TEMPERATURE  = 0.05
# INIT_TEMPERATURE = 1.0
# NCE_FLOOR        = -1e4        # divergence guard; see the header note

# # --- the regularizer (paper Eq. 10 / 15) -------------------------------------
# LAMBDA_MAX     = 0.10
# JREG_ESTIMATOR = "proj"        # "proj" (Hoffman et al. 2019) | "exact"
# JREG_NPROJ     = 1
# JREG_SUBBATCH  = 512
# JREG_AT        = "clean"
# JREG_ON        = "both"

# # --- the attack --------------------------------------------------------------
# ADV_ALPHA_RULE = "fork"        # "fork" -> alpha = ADV_ALPHA_RAW (solver default)
#                                # "madry" -> 2.5*eps/steps
# ADV_ALPHA_RAW  = 0.01          # solver default `adv_alpha`
# ADV_ALPHA_REL  = 0.20          # used when ADV_ALPHA_RULE == "relative":
#                                #   alpha = ADV_ALPHA_REL * eps  (keeps the step
#                                #   proportional across a wide eps sweep, which
#                                #   a fixed 0.01 does not)
# ADV_STEPS      = 10            # solver default `adv_steps`
# ADV_CLIP_RANGE = False         # the fork applies NO clamp to a valid data range
# ADV_CACHE_POS_NEG = True       # re-encoding positive/negative every inner step
#                                # is what _inference does, but parameters are
#                                # frozen during the attack so caching is
#                                # mathematically identical -- and ~2.4x faster.
#                                # selftest_adv_cache() proves it numerically.

# # epsilon sweep.  An linf ball of radius e has l2 radius up to e*sqrt(C), so
# # the l2 grid is scaled by sqrt(D_OBS); using the same numbers for both norms
# # would pit linf against an l2 attack ~7x weaker.
# _EPS_LINF   = [0.0125, 0.025, 0.05, 0.10, 0.20]
# ADV_EPS_GRID = {"linf": _EPS_LINF,
#                 "l2":   [round(e * math.sqrt(D_OBS), 4) for e in _EPS_LINF]}
# ATTACK_NORMS = ["linf", "l2"]

# # selection rule, transplanted from the paper's lambda protocol to epsilon
# NCE_TOL_ABS = 0.05             # "still allows InfoNCE to stay at its minimum":
#                                # held-out clean InfoNCE within 0.05 nats of the
#                                # clean baseline
# EVAL_NCE_BATCHES = 20          # held-out batches used to measure it

# # if STAGE == "evaluate", epsilon must already be known.  None -> fall back to
# # the fork defaults and say so loudly.
# SELECTED_MANUAL = {"linf": None, "l2": None}

# # --- attribution -------------------------------------------------------------
# # Key names confirmed from a real run of this fork:
# #   ['neuron-gradient', 'neuron-gradient-convabs',
# #    'neuron-gradient-convabs-inv-lsq', 'neuron-gradient-convabs-inv-svd',
# #    'neuron-gradient-inv-lsq', 'neuron-gradient-inv-svd',
# #    'time_inversion_lsq', 'time_inversion_svd']
# ATTR_METHOD_CANDIDATES = ["neuron-gradient", "neuron-gradient-batched",
#                           "jacobian-based", "jacobian-based-batched"]
# ATTR_JF_KEYS    = ["neuron-gradient", "jf", "jacobian"]
# ATTR_JFINV_KEYS = ["neuron-gradient-inv-svd", "neuron-gradient-inv-lsq",
#                    "jf-inv-svd", "jf-inv-lsq", "jf-inv",
#                    "inverted-neuron-gradient"]
# ATTR_NUM_BATCHES = 32

# # --- quality gates -----------------------------------------------------------
# R2_MIN              = 0.95     # the paper's confounder control
# R2_PARITY_MAX       = 0.05     # max spread of R2 across arms within a seed
# SURROGATE_AUROC_MIN = 0.995
# JREG_SELFTEST_RTOL  = 0.15
# ADV_CACHE_RTOL      = 1e-5

# # --- reporting ---------------------------------------------------------------
# PRIMARY_METRIC = "auroc_global"
# N_BOOTSTRAP    = 1000
# RESULT_TAG     = f"{ARCH_VARIANT}_Lorenzo"
# SELECT_SEED0   = 90_001        # disjoint from EVAL_SEED0 by construction
# EVAL_SEED0     = 1_234
# SEED_STRIDE    = 977

# # --- scale presets -----------------------------------------------------------
# _SCALES = {
#     "dry":         dict(n_seeds=2,  T=8_000,   iters=300,    batch=256,
#                         map_pts=1500, attr_pts=1500),
#     "pilot":       dict(n_seeds=3,  T=40_000,  iters=4_000,  batch=1024,
#                         map_pts=4000, attr_pts=4000),
#     "select":      dict(n_seeds=3,  T=40_000,  iters=8_000,  batch=2048,
#                         map_pts=4000, attr_pts=4000),
#     "paper_final": dict(n_seeds=10, T=100_000, iters=20_000, batch=5_000,
#                         map_pts=8000, attr_pts=8000),
# }
# SELECT_SCALE   = "select"
# EVALUATE_SCALE = "paper_final"

# # globals filled in by set_scale()
# N_SEEDS = T_SAMPLES = MAX_ITER = BATCH_SIZE = MAP_POINTS = ATTR_POINTS = 0
# LAMBDA_WARMUP = LAMBDA_RAMP = 1


# def set_scale(name: str):
#     """Install a scale preset into the module globals."""
#     global N_SEEDS, T_SAMPLES, MAX_ITER, BATCH_SIZE, MAP_POINTS, ATTR_POINTS
#     global LAMBDA_WARMUP, LAMBDA_RAMP, SCALE_NAME
#     s = _SCALES[name]
#     N_SEEDS, T_SAMPLES, MAX_ITER = s["n_seeds"], s["T"], s["iters"]
#     BATCH_SIZE, MAP_POINTS, ATTR_POINTS = s["batch"], s["map_pts"], s["attr_pts"]
#     # the paper's 2500/2500-of-20000 shape, held at any run length
#     LAMBDA_WARMUP = max(1, int(round(MAX_ITER * 2500 / 20000)))
#     LAMBDA_RAMP   = max(1, int(round(MAX_ITER * 2500 / 20000)))
#     SCALE_NAME = name
#     print(f"[scale] {name}: seeds={N_SEEDS} T={T_SAMPLES} iters={MAX_ITER} "
#           f"batch={BATCH_SIZE} lambda warmup/ramp={LAMBDA_WARMUP}/{LAMBDA_RAMP}")


# if DRY_RUN:
#     SELECT_SCALE = EVALUATE_SCALE = "dry"
#     ATTACK_NORMS = ["linf", "l2"]
#     _EPS_LINF = [0.05, 0.20]
#     ADV_EPS_GRID = {"linf": _EPS_LINF,
#                     "l2": [round(e * math.sqrt(D_OBS), 4) for e in _EPS_LINF]}
#     # smoke test only: 300 iterations cannot reach R2 >= 0.95, so the gate
#     # would exclude every seed and silently skip the whole statistics path.
#     R2_MIN, R2_PARITY_MAX = 0.0, 1e9
#     print("[config] DRY_RUN=True -> scales forced to 'dry', eps grid trimmed, "
#           "R2 gates DISABLED.  auROC/auPRC from this run are MEANINGLESS: it "
#           "is an error smoke test, not a result.")

# # ==============================================================================
# # 2. ENVIRONMENT
# # ==============================================================================

# import torch
# import torch.nn as nn

# try:
#     from scipy import stats as _sps
#     HAVE_SCIPY = True
# except Exception:
#     HAVE_SCIPY = False
#     warnings.warn("scipy missing: Wilcoxon replaced by a sign test.")

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # def _load_cebra():
# #     """Import the ORIGINAL CEBRA from CEBRA_DIR, shadowing anything already
# #     imported or pip-installed.

# #     Deliberate: the attack lives in THIS file as a transcription of the
# #     advisor's Solver.step, so importing the advisor's fork would apply it
# #     twice.  The fork is still the authoritative spec -- just not the runtime."""
# #     import sys
# #     from pathlib import Path

# #     cebra_dir = Path(CEBRA_DIR).resolve()
# #     if not (cebra_dir / "cebra").is_dir():
# #         raise SystemExit(
# #             f"{cebra_dir} does not contain a `cebra/` package directory.\n"
# #             f"Set CEBRA_DIR to the repo ROOT (the folder holding cebra/, "
# #             f"setup.py, pyproject.toml).")
# #     for module_name in list(sys.modules):
# #         if module_name == "cebra" or module_name.startswith("cebra."):
# #             del sys.modules[module_name]
# #     sys.path.insert(0, str(cebra_dir))
# #     try:
# #         import cebra
# #         import cebra.models
# #         import cebra.models.criterions as _crit
# #     except Exception as exc:
# #         raise SystemExit(f"cannot import ORIGINAL CEBRA from {cebra_dir}. "
# #                          f"error: {exc}")

# #     print(f"[env] cebra {getattr(cebra, '__version__', '?')} <- {cebra.__file__}")

# #     # upstream CEBRA has no `attribution` subpackage -- only the xCEBRA line
# #     # does.  Fail here rather than 20 minutes into training.
# #     try:
# #         import cebra.attribution                                    # noqa: F401
# #     except Exception as exc:
# #         raise SystemExit(
# #             f"the imported cebra has no `attribution` subpackage ({exc}).\n"
# #             f"  imported from: {cebra.__file__}\n"
# #             f"Point CEBRA_DIR at a checkout that contains cebra/attribution/.")

# #     try:
# #         import inspect
# #         import cebra.solver.base as _sb
# #         src = inspect.getsource(_sb)
# #         adv = ("attack_norm" in src) and ("training_mode" in src)
# #         print(f"[env] adversarial solver in THIS checkout: {adv}"
# #               f"{'  (expected False -- the attack is reimplemented here)' if not adv else ''}")
# #     except Exception as exc:
# #         print(f"[env] could not inspect cebra.solver.base: {exc}")

# #     return cebra, _crit


# # CEBRA, CRIT = _load_cebra()

# def _load_cebra():
#     import sys
#     from pathlib import Path

#     # IMPORTANT: clean/original CEBRA, NOT modified CEBRA-main
#     cebra_dir = Path(__file__).resolve().parent / "CEBRA-original"

#     # remove any cebra that may already be loaded
#     for module_name in list(sys.modules):
#         if module_name == "cebra" or module_name.startswith("cebra."):
#             del sys.modules[module_name]

#     # force loading the ORIGINAL version
#     sys.path.insert(0, str(cebra_dir))

#     try:
#         import cebra
#         import cebra.models
#         import cebra.models.criterions as _crit
#     except Exception as exc:
#         raise SystemExit(
#             f"cannot import ORIGINAL cebra from {cebra_dir}. error: {exc}"
#         )

#     print("CEBRA loaded from:", cebra.__file__)

#     return cebra, _crit


# CEBRA, CRIT = _load_cebra()


# def adv_alpha(eps: float) -> float:
#     if ADV_ALPHA_RULE == "madry":
#         return 2.5 * eps / max(ADV_STEPS, 1)
#     if ADV_ALPHA_RULE == "relative":
#         return ADV_ALPHA_REL * eps
#     return ADV_ALPHA_RAW


# def _banner():
#     print("=" * 78)
#     print(f" STAGE={STAGE}  arch={ARCH_VARIANT}  model={MODEL_NAME} "
#           f"tanh_scale={TANH_SCALE}")
#     print(f" device={DEVICE}  standardize={STANDARDIZE}  d_list={D_LATENT_LIST}")
#     print(f" lambda_max={LAMBDA_MAX}  jreg={JREG_ESTIMATOR}/{JREG_NPROJ} "
#           f"sub={JREG_SUBBATCH} at={JREG_AT} on={JREG_ON}")
#     print(f" attack: steps={ADV_STEPS} alpha_rule={ADV_ALPHA_RULE} "
#           f"clip={ADV_CLIP_RANGE} cache_pos_neg={ADV_CACHE_POS_NEG}")
#     for n in ATTACK_NORMS:
#         print(f"   eps grid [{n:>4}]: {ADV_EPS_GRID[n]}")
#     print(f" selection: largest eps with held-out clean InfoNCE within "
#           f"{NCE_TOL_ABS} nats of baseline  (NEVER auROC)")
#     print(f" select seeds base={SELECT_SEED0}   eval seeds base={EVAL_SEED0} "
#           f"(disjoint)")
#     print("=" * 78)


# # ==============================================================================
# # 3. DATA GENERATING PROCESS
# # ==============================================================================

# def brownian_motion_box(T: int, d: int, sigma: float, rng) -> np.ndarray:
#     """z_1 ~ U[-1,1]^d ; z_t ~ N_[-1,1](z_{t-1}, sigma^2 I), TRUNCATED (clipped).

#     The paper clips to the box rather than reflecting off it; clipping leaves
#     probability mass on the faces of the cube, reflecting does not."""
#     z = np.empty((T, d), dtype=np.float64)
#     z[0] = rng.uniform(-1.0, 1.0, size=d)
#     for t in range(1, T):
#         z[t] = np.clip(z[t - 1] + sigma * rng.normal(size=d), -1.0, 1.0)
#     return z


# def orthonormal_columns(rows: int, cols: int, rng) -> np.ndarray:
#     """Dense [rows, cols] with orthonormal columns (reduced QR).

#     Orthonormality keeps the mixing well conditioned so the generator Jacobian
#     has no near-zero entries that would blur the ground truth; the Gaussian
#     seed makes it dense almost surely, so every declared edge in `gt` is real."""
#     assert rows >= cols
#     q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
#     return q[:, :cols]


# @dataclass
# class Mixer:
#     """x1 = tanh(g1 W1 z[:d1]) (N1 neurons) ; x2 = tanh(g2 W2 z) (N2 neurons).

#     Figure 5: "z1 is connected to both x1 and x2, while z2 is connected only to
#     x2 [...] g1 takes 3 (d1) latent variables as input and outputs 25 neurons
#     (n1), whereas g2 takes 6 (d1+d2) latent variables as input and outputs 25
#     neurons (n2)." """
#     W1: np.ndarray
#     W2: np.ndarray
#     d1: int
#     gain1: float = 1.0
#     gain2: float = 1.0

#     def pre(self, z):
#         return (self.gain1 * z[:, :self.d1] @ self.W1.T,
#                 self.gain2 * z @ self.W2.T)

#     def __call__(self, z):
#         p1, p2 = self.pre(z)
#         return np.concatenate([np.tanh(p1), np.tanh(p2)], axis=1)

#     def generator_jacobian(self, z) -> np.ndarray:
#         """dx/dz, [S, C, D].  Unique -- this is what `gt` is derived from."""
#         S, d = z.shape
#         p1, p2 = self.pre(z)
#         s1 = 1.0 - np.tanh(p1) ** 2
#         s2 = 1.0 - np.tanh(p2) ** 2
#         n1 = self.W1.shape[0]
#         J = np.zeros((S, n1 + self.W2.shape[0], d))
#         J[:, :n1, :self.d1] = s1[:, :, None] * (self.gain1 * self.W1)[None]
#         J[:, n1:, :]        = s2[:, :, None] * (self.gain2 * self.W2)[None]
#         return J


# def lorenz_latents(T: int, d: int, sigma: float, rng,
#                    dt: float = 0.01, burn_in: int = 2000,
#                    lorenz_sigma: float = 10.0, lorenz_rho: float = 28.0,
#                    lorenz_beta: float = 8.0 / 3.0) -> np.ndarray:
#     """z_t via one or more independent Lorenz-63 chaotic systems (3 state
#     variables each) instead of a Brownian walk. Concatenates ceil(d/3)
#     independent chains (different random initial conditions -> genuinely
#     different trajectories, since Lorenz is extremely sensitive to ICs),
#     truncates to exactly d columns, discards a burn-in transient so the
#     trajectory starts on the attractor, then rescales+clips each column to
#     [-1, 1] -- same box contract as brownian_motion_box.

#     `sigma` (the caller's diffusion-scale argument) has no equivalent in a
#     deterministic system; it is repurposed here as the spread of the random
#     initial conditions across chains.
#     """
#     def _deriv(state):
#         x, y, z = state
#         return np.array([lorenz_sigma * (y - x),
#                          x * (lorenz_rho - z) - y,
#                          x * y - lorenz_beta * z])

#     n_chains = int(np.ceil(d / 3))
#     total_steps = burn_in + T
#     chains = []
#     for _ in range(n_chains):
#         state = np.array([1.0, 1.0, 1.0]) + rng.normal(scale=1.0 + sigma, size=3)
#         traj = np.empty((total_steps, 3), dtype=np.float64)
#         for t in range(total_steps):
#             traj[t] = state
#             k1 = _deriv(state)
#             k2 = _deriv(state + 0.5 * dt * k1)
#             k3 = _deriv(state + 0.5 * dt * k2)
#             k4 = _deriv(state + dt * k3)
#             state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
#         chains.append(traj[burn_in:])

#     z = np.concatenate(chains, axis=1)[:, :d]
#     scale = np.max(np.abs(z), axis=0, keepdims=True)
#     scale[scale < 1e-12] = 1.0
#     z = np.clip(z / scale, -1.0, 1.0)
#     return z.astype(np.float64)

# def generate_dataset(d: int, sigma_obs: float, data_seed: int) -> dict:
#     rng = np.random.default_rng(data_seed)
#     d1 = d // 2 + d % 2
#     z = lorenz_latents(T_SAMPLES, d, BROWNIAN_SIGMA, rng)
#     W1 = orthonormal_columns(N1, d1, rng)
#     W2 = orthonormal_columns(N2, d, rng)
#     mix = Mixer(W1=W1, W2=W2, d1=d1)
#     sub = z[rng.choice(T_SAMPLES, size=min(5000, T_SAMPLES), replace=False)]
#     mix.gain1 = TARGET_PRE_SD / max(float(np.std(sub[:, :d1] @ W1.T)), 1e-12)
#     mix.gain2 = TARGET_PRE_SD / max(float(np.std(sub @ W2.T)), 1e-12)
#     x = mix(z)
#     if sigma_obs > 0:
#         x = x + sigma_obs * rng.normal(size=x.shape)
#     return dict(z=z, x=x, mix=mix, d=d, d1=d1, d2=d - d1, seed=data_seed)


# def ground_truth(d: int, d1: int) -> np.ndarray:
#     """gt[C, D] = 1 iff neuron i is generated from latent j.  Support of dx/dz."""
#     gt = np.zeros((D_OBS, d), dtype=np.int8)
#     gt[:N1, :d1] = 1        # x1 sees z1 only
#     gt[N1:, :] = 1          # x2 sees z1 and z2
#     return gt


# def split_data(data: dict) -> dict:
#     """Disjoint train / map-fit / attribution segments."""
#     T = T_SAMPLES
#     assert MAP_POINTS + ATTR_POINTS < T // 2, "held-out segments too large"
#     train_end = T - (MAP_POINTS + ATTR_POINTS) - TIME_OFFSET - 1
#     idx_train = np.arange(0, train_end)
#     idx_map   = np.arange(train_end, train_end + MAP_POINTS)
#     idx_attr  = np.arange(train_end + MAP_POINTS,
#                           train_end + MAP_POINTS + ATTR_POINTS)
#     if STANDARDIZE:
#         mu = data["x"][idx_train].mean(0, keepdims=True)
#         sd = data["x"][idx_train].std(0, keepdims=True) + 1e-8
#     else:
#         mu = np.zeros((1, D_OBS))
#         sd = np.ones((1, D_OBS))
#     xs = (data["x"] - mu) / sd    # diagonal rescale: zeros of dx/dz unchanged
#     return dict(xs=xs, mu=mu, sd=sd, idx_train=idx_train,
#                 idx_map=idx_map, idx_attr=idx_attr)


# # ==============================================================================
# # 4. METRIC MACHINERY
# # ==============================================================================

# def squeeze_to_3d(a: np.ndarray, name: str) -> np.ndarray:
#     """Collapse trailing singleton axes ONLY.  Never average over a real axis:
#     a signed mean over a lag axis cancels opposite-sign entries and turns a
#     genuine edge into a zero."""
#     a = np.asarray(a)
#     while a.ndim > 3:
#         if a.shape[-1] == 1:
#             a = a[..., 0]
#         else:
#             raise ValueError(f"{name}: non-singleton extra axis {a.shape}")
#     if a.ndim != 3:
#         raise ValueError(f"{name}: expected 3 dims, got {a.shape}")
#     return a


# def canonicalize_jf(jf: np.ndarray, out_dim: int, name: str) -> np.ndarray:
#     """-> [S, O, C]."""
#     jf = squeeze_to_3d(jf, name)
#     if jf.shape[1] == out_dim and jf.shape[2] == D_OBS:
#         return jf
#     if jf.shape[2] == out_dim and jf.shape[1] == D_OBS:
#         return np.swapaxes(jf, 1, 2)
#     raise ValueError(f"{name}: cannot orient {jf.shape} to [S,{out_dim},{D_OBS}]")


# def truncated_pinv(jf: np.ndarray, drop: int) -> np.ndarray:
#     """Batched pinv of J_f with rank truncation.  [S,O,C] -> [S,C,O]."""
#     U, S, Vh = np.linalg.svd(jf, full_matrices=False)
#     S = S.copy()
#     if drop > 0:
#         S[:, -drop:] = 0.0
#     inv = np.zeros_like(S)
#     good = S > np.maximum(1e-10 * S[:, :1], 1e-30)
#     inv[good] = 1.0 / S[good]
#     return np.einsum("ski,sk,sok->sio", Vh, inv, U)      # V diag(1/s) U^T


# def fit_linear_maps(f: np.ndarray, z: np.ndarray, ridge: float = 1e-6):
#     """A_z2e [O,D] with f ~ A z ; B_e2z [O,D] with z ~ f B ; R2 per latent dim.

#     A_z2e is what the alignment needs (df/dz).  B_e2z gives the paper's "R2 for
#     predicting the auxiliary variable from the feature space"."""
#     f = np.asarray(f, dtype=np.float64)
#     z = np.asarray(z, dtype=np.float64)
#     fc = f - f.mean(0, keepdims=True)
#     zc = z - z.mean(0, keepdims=True)
#     O, D = f.shape[1], z.shape[1]
#     G = zc.T @ zc
#     G = G + ridge * np.trace(G) / max(D, 1) * np.eye(D)
#     A_z2e = np.linalg.solve(G, zc.T @ fc).T
#     H = fc.T @ fc
#     H = H + ridge * np.trace(H) / max(O, 1) * np.eye(O)
#     B_e2z = np.linalg.solve(H, fc.T @ zc)
#     ss_res = ((zc - fc @ B_e2z) ** 2).sum(0)
#     r2 = 1.0 - ss_res / ((zc ** 2).sum(0) + 1e-30)
#     return A_z2e, B_e2z, r2


# def align_to_latents(Q: np.ndarray, A_z2e: np.ndarray) -> np.ndarray:
#     """R[s,i,d] = dx_i/dz_d = sum_o Q[s,i,o] A_z2e[o,d].  [S,C,O] -> [S,C,D].
#     Exactly invariant to f -> M f: Q -> Q M^{-1}, A -> M A."""
#     return np.einsum("sio,od->sid", Q, A_z2e)


# def aggregate_map(R: np.ndarray) -> np.ndarray:
#     """[S,C,D] -> [C,D].  ABSOLUTE value BEFORE the mean: a signed average over
#     timepoints cancels a real edge whose sign flips along the trajectory."""
#     return np.abs(R).mean(axis=0)


# def score_variants(M: np.ndarray) -> Dict[str, np.ndarray]:
#     """Global z-score (primary; monotone, hence auROC-equivalent to raw) plus a
#     per-latent-column normalisation, which is NOT auROC-equivalent."""
#     return {"global": (M - M.mean()) / (M.std() + 1e-30),
#             "colnorm": (M - M.mean(0, keepdims=True)) /
#                        (M.std(0, keepdims=True) + 1e-30)}


# def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
#     s = np.asarray(scores, dtype=np.float64).ravel()
#     y = np.asarray(labels).ravel().astype(bool)
#     n_pos, n_neg = int(y.sum()), int((~y).sum())
#     if n_pos == 0 or n_neg == 0:
#         return float("nan")
#     order = np.argsort(s, kind="mergesort")
#     ranks = np.empty_like(s)
#     ss = s[order]
#     i = 0
#     while i < len(s):
#         j = i
#         while j + 1 < len(s) and ss[j + 1] == ss[i]:
#             j += 1
#         ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
#         i = j + 1
#     return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
#     s = np.asarray(scores, dtype=np.float64).ravel()
#     y = np.asarray(labels).ravel().astype(bool)
#     if y.sum() == 0:
#         return float("nan")
#     y = y[np.argsort(-s, kind="mergesort")]
#     tp = np.cumsum(y)
#     prec = tp / np.arange(1, len(y) + 1)
#     rec = tp / y.sum()
#     return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


# def binary_scores(M: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
#     out = {}
#     for key, s in score_variants(M).items():
#         out[f"auroc_{key}"] = _auc(s, gt)
#         out[f"auprc_{key}"] = _auprc(s, gt)
#     return out


# # ==============================================================================
# # 5. VALIDATION OF THE METRIC ITSELF
# # ==============================================================================

# def validate_alignment_metric(data: dict, sp: dict, gt: np.ndarray, seed: int):
#     """End-to-end test of the PRIMARY scoring path on a known-perfect encoder.

#     Composes the true inverse mixing with a random linear reparametrisation
#     plus whatever head ARCH_VARIANT prescribes.  If the pipeline is correct,
#     auROC must be ~1.  Failure means the METRIC is broken, not the model, so
#     the run aborts rather than emitting numbers nobody can interpret."""
#     rng = np.random.default_rng(seed)
#     d = data["d"]
#     O = d + OUT_EXTRA
#     z = data["z"][sp["idx_attr"]]
#     Jg = data["mix"].generator_jacobian(z) / sp["sd"][0][None, :, None]
#     Jzx = np.linalg.pinv(Jg)                        # [S, d, C] = dz/dx
#     A = rng.normal(size=(O, d))                     # the linear indeterminacy
#     if _ARCH["normalize"]:
#         u = z @ A.T
#         nrm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
#         f = u / nrm
#         I = np.eye(O)
#         dfdz = (I[None] - f[:, :, None] * f[:, None, :]) @ A[None] / nrm[:, :, None]
#     else:
#         f = z @ A.T
#         dfdz = np.broadcast_to(A[None], (len(z), O, d))
#     jf = dfdz @ Jzx                                 # [S, O, C] = df/dx
#     A_z2e, _, r2 = fit_linear_maps(f, z)
#     Q = truncated_pinv(jf, PINV_DROP)
#     res = binary_scores(aggregate_map(align_to_latents(Q, A_z2e)), gt)
#     print(f"    [validate] surrogate auROC={res['auroc_global']:.4f} "
#           f"auPRC={res['auprc_global']:.4f} R2min={r2.min():.4f}")
#     if not (res["auroc_global"] >= SURROGATE_AUROC_MIN):
#         raise RuntimeError(
#             f"alignment metric failed its own sanity check "
#             f"(auROC={res['auroc_global']:.4f} < {SURROGATE_AUROC_MIN}). "
#             f"Do not trust any model comparison until this passes.")
#     return res


# # ==============================================================================
# # 6. MODEL, HEAD, CRITERION, SAMPLER
# # ==============================================================================

# class ScaledTanh(nn.Module):
#     """The paper's output layer: "one layer + scaled tanh" (Appendix B.1).

#     s*tanh(y/s) is the identity near the origin and saturates at +-s, so it
#     bounds ||f||_inf <= s without rescaling gradients in the linear regime.
#     The bound is what makes Euclidean InfoNCE well posed -- see the header."""

#     def __init__(self, scale: float):
#         super().__init__()
#         self.scale = float(scale)

#     def forward(self, x):
#         return self.scale * torch.tanh(x / self.scale)

#     def extra_repr(self):
#         return f"scale={self.scale}"


# def attach_scaled_tanh(model: nn.Module, scale: float) -> nn.Module:
#     """Append the scaled-tanh head IN PLACE, after the model's own Squeeze.

#     Appending into the existing container -- rather than wrapping the model in
#     a new nn.Module -- keeps the object's class, .get_offset() and .normalize
#     intact, so cebra.attribution keeps working exactly as it already does."""
#     head = ScaledTanh(scale)
#     for attr in ("net", "module", "layers"):
#         sub = getattr(model, attr, None)
#         if isinstance(sub, nn.Sequential):
#             sub.add_module("scaled_tanh", head)
#             return model
#     if isinstance(model, nn.Sequential):
#         model.add_module("scaled_tanh", head)
#         return model
#     raise RuntimeError(
#         f"cannot append the scaled-tanh head to {type(model).__name__}: no "
#         f"nn.Sequential container found.  Run print(model), attach it "
#         f"manually, or set ARCH_VARIANT='cosine_sphere'.")


# def flat(e: torch.Tensor) -> torch.Tensor:
#     return e if e.dim() == 2 else e.reshape(e.shape[0], -1)


# _ONCE = {"head": False, "jreg": False, "cache": False, "attr": False,
#          "arch": False}


# def verify_head(model):
#     """Prove the head is actually in the forward path.  Appending to the wrong
#     container fails SILENTLY and reintroduces the divergence."""
#     if _ONCE["head"]:
#         return
#     with torch.no_grad():
#         y = flat(model(torch.randn(8, D_OBS, 1, device=DEVICE) * 50.0))
#     m = float(y.abs().max())
#     print(f"    [validate] |f|_inf at 50x input scale = {m:.4f} "
#           f"(bound = {TANH_SCALE})")
#     if TANH_SCALE is not None and m > TANH_SCALE * 1.01:
#         raise RuntimeError("scaled-tanh head is NOT in the forward path")
#     _ONCE["head"] = True


# def build_model(d: int) -> Tuple[nn.Module, nn.Module]:
#     model = CEBRA.models.init(MODEL_NAME, D_OBS, NUM_UNITS,
#                               d + OUT_EXTRA).to(DEVICE)
#     if len(model.get_offset()) != 1:
#         raise RuntimeError(
#             f"{MODEL_NAME} has receptive field {len(model.get_offset())}; this "
#             f"benchmark assumes an instantaneous (offset-1) encoder so that "
#             f"J_f is a plain [O, C] matrix per timepoint.")
#     got = bool(getattr(model, "normalize", None))
#     if got != _ARCH["normalize"]:
#         raise RuntimeError(
#             f"{MODEL_NAME}.normalize={got} but ARCH_VARIANT={ARCH_VARIANT} "
#             f"expects {_ARCH['normalize']}.  Before deleting this check, run "
#             f"print(model): with normalize=True and pinv_drop=0 the "
#             f"attribution map is destroyed SILENTLY.")
#     if TANH_SCALE is not None:
#         attach_scaled_tanh(model, TANH_SCALE)
#     if not _ONCE["arch"]:
#         print(model)
#         _ONCE["arch"] = True
#     verify_head(model)
#     if CRITERION == "cosine":
#         crit = CRIT.LearnableCosineInfoNCE(temperature=INIT_TEMPERATURE,
#                                            min_temperature=MIN_TEMPERATURE)
#     else:
#         crit = CRIT.LearnableEuclideanInfoNCE(temperature=INIT_TEMPERATURE,
#                                               min_temperature=MIN_TEMPERATURE)
#     return model, crit.to(DEVICE)


# class TimeSampler:
#     """CEBRA's `time` conditional.  Returns 3D [B, C, 1] tensors, the same shape
#     cebra.data loaders hand to Solver.step, so the attack below operates on
#     exactly the tensor layout the fork perturbs."""

#     def __init__(self, X: torch.Tensor, idx: np.ndarray, gen):
#         self.X = X
#         self.base = int(idx[0])
#         self.hi = len(idx) - TIME_OFFSET - 1
#         self.gen = gen

#     def __call__(self, batch: int):
#         i = torch.randint(0, self.hi, (batch,), generator=self.gen) + self.base
#         j = torch.randint(0, self.hi, (batch,), generator=self.gen) + self.base
#         i = i.to(self.X.device)
#         j = j.to(self.X.device)
#         return (self.X[i].unsqueeze(-1),
#                 self.X[i + TIME_OFFSET].unsqueeze(-1),
#                 self.X[j].unsqueeze(-1))


# # ==============================================================================
# # 7. THE JACOBIAN-FROBENIUS REGULARIZER  (paper Eq. 10 / 15)
# # ==============================================================================

# def jacobian_frobenius_sq(model: nn.Module, x: torch.Tensor,
#                           estimator: str = "proj", n_proj: int = 1
#                           ) -> torch.Tensor:
#     """E_x ||J_f(x)||_F^2, differentiable w.r.t. the model parameters.

#     exact: O backward passes, sum_o ||d f_o / d x||^2.
#     proj : Hoffman et al. (2019).  For v uniform on the unit sphere in R^O,
#            E_v ||v^T J||^2 = tr(J J^T)/O = ||J||_F^2 / O, so
#            (O / n_proj) * sum_mu ||d (v_mu . f) / d x||^2 is unbiased.  One
#            projection replaces O backward passes -- this is what makes the
#            penalty affordable at batch 5000."""
#     x = x.detach().requires_grad_(True)
#     out = flat(model(x))
#     O = out.shape[1]
#     total = 0.0
#     if estimator == "exact":
#         for o in range(O):
#             g, = torch.autograd.grad(out[:, o].sum(), x,
#                                      create_graph=True, retain_graph=True)
#             total = total + (g ** 2).flatten(1).sum(1)
#         return total.mean()
#     for _ in range(n_proj):
#         v = torch.randn(out.shape, device=out.device, dtype=out.dtype)
#         v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
#         g, = torch.autograd.grad((out * v).sum(), x,
#                                  create_graph=True, retain_graph=True)
#         total = total + (g ** 2).flatten(1).sum(1)
#     return (O / n_proj) * total.mean()


# def selftest_jreg(model: nn.Module, x: torch.Tensor):
#     """The projection estimator is only useful if it is unbiased for THIS
#     model.  Compare against the exact value on a small batch."""
#     if _ONCE["jreg"]:
#         return
#     xs = x[:64]
#     exact = float(jacobian_frobenius_sq(model, xs, "exact"))
#     est = float(np.mean([float(jacobian_frobenius_sq(model, xs, "proj", 8))
#                          for _ in range(24)]))
#     rel = abs(est - exact) / max(exact, 1e-12)
#     print(f"    [validate] ||J||_F^2 exact={exact:.5f} proj={est:.5f} "
#           f"rel.err={rel:.3f}")
#     if rel > JREG_SELFTEST_RTOL:
#         raise RuntimeError(
#             f"Hoffman projection estimator biased by {rel:.1%} -- refusing to "
#             f"run a regularized arm on an untrustworthy penalty.")
#     _ONCE["jreg"] = True


# def lambda_at(it: int, lam_max: float) -> float:
#     """0 for LAMBDA_WARMUP iterations, then a linear ramp over LAMBDA_RAMP.

#     "The first 2,500 training steps minimize the InfoNCE [...] loss with
#      lambda = 0; we then ramp up lambda to its maximum value over the following
#      2,500 steps, and continue to train until 20,000 total steps."

#     The counter advances per ITERATION, not per optimizer step, so lambda
#     reaches its maximum at the same point in the run for single- and
#     double-update arms."""
#     if lam_max <= 0.0:
#         return 0.0
#     if it < LAMBDA_WARMUP:
#         return 0.0
#     return lam_max * min(1.0, max(0.0, (it - LAMBDA_WARMUP) / float(LAMBDA_RAMP)))


# # ==============================================================================
# # 8. THE ATTACK -- literal transcription of cebra/solver/base.py::Solver.step
# # ==============================================================================
# # The three helpers below are copied verbatim from the fork so the l2 branch is
# # bit-for-bit the same geometry.

# def _l2_normalize(t: torch.Tensor, eps: float = 1e-12):
#     """Per-sample L2 normalisation (zero vectors stay zero)."""
#     flat_ = t.reshape(t.size(0), -1)
#     norm = flat_.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
#     return t / norm.view(-1, *([1] * (t.dim() - 1)))


# def _rand_radius_like(t: torch.Tensor):
#     """U(0,1) radius shaped like t but broadcastable (B,1,1,...)."""
#     return torch.rand([t.size(0)] + [1] * (t.dim() - 1), device=t.device)


# def _proj_l2_ball(adv: torch.Tensor, orig: torch.Tensor, epsilon: float):
#     """Project adv back to the closed L2 ball of radius eps around orig."""
#     delta = adv - orig
#     flat_ = delta.reshape(delta.size(0), -1)
#     norm = flat_.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
#     factor = torch.where(norm > epsilon, norm / epsilon, torch.ones_like(norm))
#     delta = delta / factor.view(-1, *([1] * (delta.dim() - 1)))
#     return orig + delta


# def eps_ball_init(x_ref: torch.Tensor, norm: str, eps: float,
#                   lo: float, hi: float) -> torch.Tensor:
#     """PGD's random start.  linf: reference + U(-eps, +eps) over the whole
#     eps-cube.  l2: reference + _l2_normalize(randn) * U(0,1) * eps.

#     Also used on its own by the noise_2x control arm -- that arm is exactly
#     this initialization with zero ascent steps, which is what isolates "the
#     adversarial DIRECTION matters" from "any jitter of size eps regularizes"."""
#     if norm == "linf":
#         x = x_ref + torch.empty_like(x_ref).uniform_(-eps, eps)
#     elif norm == "l2":
#         noise = _l2_normalize(torch.randn_like(x_ref))
#         x = x_ref + noise * _rand_radius_like(x_ref) * eps
#     else:
#         raise ValueError(f"unknown attack norm {norm!r}")
#     x = x.clone().detach()
#     if ADV_CLIP_RANGE:
#         x = x.clamp(lo, hi)
#     return x


# def pgd_attack(model, crit, x_ref, x_pos, x_neg, norm, eps, lo, hi):
#     """Maximise the InfoNCE loss w.r.t. the REFERENCE inputs only.

#     Faithful to Solver.step:
#       * only batch.reference is perturbed; positive and negative stay clean
#       * linf step: x_adv += alpha * grad.sign(), then
#                    torch.max(torch.min(x_adv, ref+eps), ref-eps)
#       * l2   step: x_adv += alpha * _l2_normalize(grad), then _proj_l2_ball
#       * the gradient comes from torch.autograd.grad(loss, x_adv), so model
#         parameters are never touched by the attack
#       * NO clamp to a valid data range -- the fork applies none
#       * ADV_CACHE_POS_NEG=False re-encodes positive/negative at every inner
#         step because _inference(adv_batch) re-runs the whole batch.  True
#         caches them, which is mathematically identical (parameters are frozen
#         during the attack) and ~2.4x faster; selftest_adv_cache proves it.
#     """
#     alpha = adv_alpha(eps)
#     x_adv = eps_ball_init(x_ref, norm, eps, lo, hi).requires_grad_(True)

#     e_pos = e_neg = None
#     if ADV_CACHE_POS_NEG:
#         with torch.no_grad():
#             e_pos = flat(model(x_pos)).detach()
#             e_neg = flat(model(x_neg)).detach()

#     for _ in range(ADV_STEPS):
#         with torch.enable_grad():
#             ep = e_pos if e_pos is not None else flat(model(x_pos))
#             en = e_neg if e_neg is not None else flat(model(x_neg))
#             loss, _, _ = crit(flat(model(x_adv)), ep, en)
#         grad_x, = torch.autograd.grad(loss, x_adv, retain_graph=False,
#                                       create_graph=False)
#         with torch.no_grad():
#             if norm == "linf":
#                 x_adv = x_adv + alpha * grad_x.sign()
#                 x_adv = torch.max(torch.min(x_adv, x_ref + eps), x_ref - eps)
#             else:
#                 x_adv = x_adv + alpha * _l2_normalize(grad_x)
#                 x_adv = _proj_l2_ball(x_adv, x_ref, eps)
#             if ADV_CLIP_RANGE:
#                 x_adv = x_adv.clamp(lo, hi)
#         x_adv = x_adv.detach().requires_grad_(True)

#     return x_adv.detach()


# def selftest_adv_cache(model, crit, x_ref, x_pos, x_neg, norm, eps, lo, hi):
#     """Prove ADV_CACHE_POS_NEG changes nothing but the runtime.

#     Same RNG state on both paths, so the random start is identical and the two
#     x_adv must agree to floating-point tolerance."""
#     global ADV_CACHE_POS_NEG
#     if _ONCE["cache"]:
#         return
#     keep = ADV_CACHE_POS_NEG
#     cpu_state = torch.random.get_rng_state()
#     cuda_state = (torch.cuda.get_rng_state_all()
#                   if torch.cuda.is_available() else None)

#     def _run(flag):
#         global ADV_CACHE_POS_NEG
#         torch.random.set_rng_state(cpu_state)
#         if cuda_state is not None:
#             torch.cuda.set_rng_state_all(cuda_state)
#         ADV_CACHE_POS_NEG = flag
#         return pgd_attack(model, crit, x_ref, x_pos, x_neg, norm, eps, lo, hi)

#     a = _run(False)
#     b = _run(True)
#     ADV_CACHE_POS_NEG = keep
#     dev = float((a - b).abs().max())
#     print(f"    [validate] adv cache equivalence: max|dx_adv| = {dev:.3e} "
#           f"(tol {ADV_CACHE_RTOL:.0e}, using cache={keep})")
#     if dev > ADV_CACHE_RTOL:
#         raise RuntimeError(
#             f"caching positive/negative changed the attack by {dev:.3e}. "
#             f"Set ADV_CACHE_POS_NEG=False and take the 2.4x slowdown.")
#     _ONCE["cache"] = True


# # ==============================================================================
# # 9. TRAINING
# # ==============================================================================

# def _update(model, crit, opt, x_ref, x_pos, x_neg, lam, x_jreg, gen):
#     """One optimizer.step(), optionally carrying the lambda ||J||_F^2 penalty."""
#     opt.zero_grad(set_to_none=True)
#     loss, align, uniform = crit(flat(model(x_ref)), flat(model(x_pos)),
#                                 flat(model(x_neg)))
#     total = loss
#     jval = 0.0
#     if lam > 0.0:
#         xj = x_jreg
#         if JREG_SUBBATCH and JREG_SUBBATCH < xj.shape[0]:
#             sel = torch.randint(0, xj.shape[0], (JREG_SUBBATCH,),
#                                 generator=gen).to(xj.device)
#             xj = xj[sel]
#         jf2 = jacobian_frobenius_sq(model, xj, JREG_ESTIMATOR, JREG_NPROJ)
#         total = loss + lam * jf2
#         jval = float(jf2)
#     total.backward()
#     opt.step()
#     return float(loss), jval


# TWO_UPDATE = ("clean_double", "fork_double", "noise_double")


# def expected_steps(scheme: str) -> int:
#     return MAX_ITER * (2 if scheme in TWO_UPDATE else 1)


# @torch.no_grad()
# def eval_clean_nce(model, crit, sampler, n_batches: int, batch: int) -> float:
#     """Held-out clean InfoNCE.  This -- and only this -- drives epsilon
#     selection, mirroring the paper's rule for lambda.  Measuring it on held-out
#     data rather than on a running training average keeps the criterion honest
#     for arms that train on perturbed inputs."""
#     vals = []
#     for _ in range(n_batches):
#         xr, xp, xn = sampler(batch)
#         loss, _, _ = crit(flat(model(xr)), flat(model(xp)), flat(model(xn)))
#         vals.append(float(loss))
#     return float(np.mean(vals))


# def train_arm(arm: str, cfg: dict, data: dict, sp: dict, seed: int,
#               verbose_every: int = 0):
#     scheme, lam_max = cfg["scheme"], cfg["lam"]
#     norm, eps = cfg.get("norm"), cfg.get("eps")
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     gen = torch.Generator().manual_seed(seed)          # CPU generator: portable

#     model, crit = build_model(data["d"])
#     opt = torch.optim.Adam(list(model.parameters()) + list(crit.parameters()),
#                            lr=LEARNING_RATE)
#     X = torch.from_numpy(sp["xs"].astype(np.float32)).to(DEVICE)
#     sampler = TimeSampler(X, sp["idx_train"], gen)
#     held = TimeSampler(X, sp["idx_map"], torch.Generator().manual_seed(seed + 7))
#     lo = float(X[sp["idx_train"]].min())
#     hi = float(X[sp["idx_train"]].max())

#     if lam_max > 0:
#         selftest_jreg(model, X[sp["idx_train"][:64]].unsqueeze(-1))
#     if scheme == "fork_double":
#         xr, xp, xn = sampler(min(256, BATCH_SIZE))
#         selftest_adv_cache(model, crit, xr, xp, xn, norm, eps, lo, hi)

#     def lam_for(which: str, it: int) -> float:
#         if JREG_ON == "clean_only" and which != "clean":
#             return 0.0
#         if JREG_ON == "adv_only" and which != "adv":
#             return 0.0
#         return lambda_at(it, lam_max)

#     n_steps, hist, t0 = 0, [], time.time()
#     for it in range(MAX_ITER):
#         x_ref, x_pos, x_neg = sampler(BATCH_SIZE)
#         nce_c = nce_a = float("nan")
#         j_c = j_a = 0.0

#         # ---- update 1: the fork's UNCONDITIONAL clean update -----------------
#         if scheme != "adv_single":
#             nce_c, j_c = _update(model, crit, opt, x_ref, x_pos, x_neg,
#                                  lam_for("clean", it), x_ref, gen)
#             n_steps += 1

#         # ---- update 2 -------------------------------------------------------
#         if scheme in ("fork_double", "adv_single"):
#             # NOTE: the attack is built AFTER the clean step, i.e. with the
#             # already-updated parameters -- exactly as in Solver.step, where
#             # the adversarial branch follows self.optimizer.step().
#             x2 = pgd_attack(model, crit, x_ref, x_pos, x_neg, norm, eps, lo, hi)
#             xj = x_ref if JREG_AT == "clean" else x2
#             nce_a, j_a = _update(model, crit, opt, x2, x_pos, x_neg,
#                                  lam_for("adv", it), xj, gen)
#             n_steps += 1
#         elif scheme == "noise_double":
#             x2 = eps_ball_init(x_ref, norm, eps, lo, hi)
#             nce_a, j_a = _update(model, crit, opt, x2, x_pos, x_neg,
#                                  lam_for("adv", it), x_ref, gen)
#             n_steps += 1
#         elif scheme == "clean_double":
#             nce_a, j_a = _update(model, crit, opt, x_ref, x_pos, x_neg,
#                                  lam_for("adv", it), x_ref, gen)
#             n_steps += 1

#         bad = [v for v in (nce_c, nce_a)
#                if not math.isnan(v) and (not math.isfinite(v) or v < NCE_FLOOR)]
#         if bad:
#             raise RuntimeError(
#                 f"{arm}: InfoNCE diverged at it={it} (loss={bad[0]:.4e} < "
#                 f"{NCE_FLOOR:.0e}, tau={float(crit.temperature):.4f}).  An "
#                 f"unbounded head makes Euclidean InfoNCE unbounded below.  "
#                 f"Fixes: ARCH_VARIANT='euclidean_tanh' (the paper's head), or "
#                 f"'cosine_sphere', or lambda > 0 (the Frobenius penalty grows "
#                 f"like s^2 and restores boundedness -- which is why only the "
#                 f"lambda = 0 arms blow up).")

#         hist.append((nce_c, nce_a, max(j_c, j_a), lambda_at(it, lam_max)))
#         if verbose_every and (it % verbose_every == 0 or it == MAX_ITER - 1):
#             print(f"      [{arm:16s}] it={it:6d} nce={nce_c:8.4f} "
#                   f"nce2={nce_a:8.4f} |J|^2={max(j_c, j_a):9.4f} "
#                   f"lam={lambda_at(it, lam_max):.3f} "
#                   f"tau={float(crit.temperature):.4f}")

#     exp = expected_steps(scheme)
#     assert n_steps == exp, f"{arm}: {n_steps} steps, expected {exp}"
#     model.eval()
#     tail = hist[-50:]
#     return dict(
#         model=model, crit=crit, n_steps=n_steps, seconds=time.time() - t0,
#         nce_clean_train=float(np.nanmean([h[0] for h in tail])),
#         nce_second_train=float(np.nanmean([h[1] for h in tail])),
#         eval_nce=eval_clean_nce(model, crit, held, EVAL_NCE_BATCHES,
#                                 min(BATCH_SIZE, 2048)),
#         final_jfro=float(np.mean([h[2] for h in tail])),
#         temperature=float(crit.temperature))


# @torch.no_grad()
# def embed(model, xs: np.ndarray, idx: np.ndarray, chunk: int = 4096) -> np.ndarray:
#     out = []
#     for s in range(0, len(idx), chunk):
#         xb = torch.from_numpy(xs[idx[s:s + chunk]].astype(np.float32)).to(DEVICE)
#         out.append(flat(model(xb.unsqueeze(-1))).cpu().numpy())
#     return np.concatenate(out, 0).astype(np.float64)


# # ==============================================================================
# # 10. ATTRIBUTION (library-provided Jacobian; the pinv is taken here)
# # ==============================================================================

# def _pick(dct: dict, keys: Sequence[str]):
#     for k in keys:
#         for kk in dct:
#             if str(kk).lower().replace("_", "-") == k.lower().replace("_", "-"):
#                 return dct[kk]
#     return None


# def _to_np(v):
#     if v is None:
#         return None
#     return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


# def library_jacobian(model, xs, idx, out_dim):
#     """Call cebra.attribution for J_f (the paper's "neuron gradient").

#     Only J_f is needed; the "inverted neuron gradient" is formed here with
#     truncated_pinv so the rank truncation stays explicit -- with normalize=True
#     an untruncated pinv amplifies the structurally null direction.  The
#     library's own inverse is fetched too, purely as a cross-check."""
#     import cebra.attribution
#     x = torch.from_numpy(xs[idx].astype(np.float32)).to(DEVICE).unsqueeze(-1)
#     if not _ONCE["attr"]:
#         try:
#             print("    [attr] registry options:", cebra.attribution.get_options())
#         except Exception as exc:
#             print("    [attr] get_options() unavailable:", exc)
#     errors = []
#     for name in ATTR_METHOD_CANDIDATES:
#         for kw in (dict(model=model, input_data=x, output_dimension=out_dim,
#                         num_batches=ATTR_NUM_BATCHES),
#                    dict(model=model, input_data=x, output_dimension=out_dim),
#                    dict(model=model, input_data=x)):
#             try:
#                 res = cebra.attribution.init(name, **kw).compute_attribution_map()
#             except Exception as exc:
#                 errors.append(f"{name}{list(kw)}: {type(exc).__name__}: {exc}")
#                 continue
#             jf = jfinv = None
#             if isinstance(res, dict):
#                 if not _ONCE["attr"]:
#                     print(f"    [attr] '{name}' -> keys: "
#                           f"{sorted(map(str, res.keys()))}")
#                 jf = _pick(res, ATTR_JF_KEYS)
#                 jfinv = _pick(res, ATTR_JFINV_KEYS)
#             elif isinstance(res, (tuple, list)) and len(res) == 2:
#                 jf, jfinv = res
#             else:
#                 jf = res
#             jf = _to_np(jf)
#             if jf is None:
#                 errors.append(f"{name}: no J_f in {type(res)}")
#                 continue
#             _ONCE["attr"] = True
#             return canonicalize_jf(jf, out_dim, f"jf[{name}]"), _to_np(jfinv)
#     raise RuntimeError("cebra.attribution could not be called. Tried:\n  " +
#                        "\n  ".join(errors))


# # ==============================================================================
# # 11. PER-SEED RUNNER
# # ==============================================================================

# def run_seed(arms: Dict[str, dict], d: int, sigma_obs: float,
#              seed: int, tag: str) -> List[dict]:
#     print(f"\n--- [{tag}] d={d} sigma_obs={sigma_obs} seed={seed} ---")
#     data = generate_dataset(d, sigma_obs, seed)
#     gt = ground_truth(d, data["d1"])
#     sp = split_data(data)
#     print(f"    gt density={gt.mean():.3f}  d1={data['d1']} d2={data['d2']}")
#     validate_alignment_metric(data, sp, gt, seed)

#     rows = []
#     for arm, cfg in arms.items():
#         tr = train_arm(arm, cfg, data, sp, seed,
#                        verbose_every=max(1, MAX_ITER // 4))
#         f_map = embed(tr["model"], sp["xs"], sp["idx_map"])
#         A_z2e, _, r2 = fit_linear_maps(f_map, data["z"][sp["idx_map"]])
#         jf, jfinv_lib = library_jacobian(tr["model"], sp["xs"], sp["idx_attr"],
#                                           d + OUT_EXTRA)
#         jf = jf.astype(np.float64)
#         Q = truncated_pinv(jf, PINV_DROP)
#         M_inv = aggregate_map(align_to_latents(Q, A_z2e))
#         res = binary_scores(M_inv, gt)
#         # the un-inverted "neuron gradient", for the paper's own comparison
#         res_fwd = {f"{k}_jf": v for k, v in
#                    binary_scores(aggregate_map(np.swapaxes(jf, 1, 2)), gt).items()}
#         xcheck = float("nan")
#         if jfinv_lib is not None:
#             try:
#                 Ql = np.swapaxes(canonicalize_jf(jfinv_lib, d + OUT_EXTRA,
#                                                  "jfinv_lib"), 1, 2)
#                 Ml = aggregate_map(align_to_latents(Ql.astype(np.float64), A_z2e))
#                 xcheck = float(np.corrcoef(Ml.ravel(), M_inv.ravel())[0, 1])
#             except Exception as exc:
#                 print(f"    [attr] library jf-inv cross-check skipped: {exc}")
#         sing = np.linalg.svd(jf, compute_uv=False)
#         rows.append(dict(
#             stage=tag, arm=arm, scheme=cfg["scheme"], lam=cfg["lam"],
#             norm=cfg.get("norm") or "", eps=cfg.get("eps") if cfg.get("eps")
#             is not None else "", seed=seed, d=d, sigma_obs=sigma_obs,
#             scale=SCALE_NAME,
#             r2_mean=float(r2.mean()), r2_min=float(r2.min()),
#             eval_nce=tr["eval_nce"], nce_clean=tr["nce_clean_train"],
#             nce_second=tr["nce_second_train"], final_jfro=tr["final_jfro"],
#             temperature=tr["temperature"], seconds=tr["seconds"],
#             n_steps=tr["n_steps"],
#             sv_ratio=float(np.median(sing[:, -1] / (sing[:, 0] + 1e-30))),
#             libinv_corr=xcheck, **res, **res_fwd))
#         r = rows[-1]
#         print(f"    [{arm:16s}] auROC={r['auroc_global']:.4f} "
#               f"auPRC={r['auprc_global']:.4f} "
#               f"(jf-only={r['auroc_global_jf']:.4f}) "
#               f"R2={r['r2_mean']:.4f} evalNCE={r['eval_nce']:+.4f} "
#               f"tau={r['temperature']:.3f} steps={r['n_steps']} "
#               f"{r['seconds']:.0f}s")

#     r2s = [r["r2_mean"] for r in rows]
#     ok = min(r2s) >= R2_MIN and (max(r2s) - min(r2s)) <= R2_PARITY_MAX
#     if not ok:
#         print(f"    !! seed EXCLUDED: R2 range [{min(r2s):.3f}, {max(r2s):.3f}] "
#               f"violates R2_MIN={R2_MIN} / parity={R2_PARITY_MAX}.  Comparing "
#               f"attribution across arms of unequal representation quality "
#               f"measures representation quality, not attribution.")
#     for r in rows:
#         r["included"] = bool(ok)
#     return rows


# def append_rows(rows: List[dict], path: str):
#     """Flush after every seed.  Writing only at the end would discard every
#     completed seed if the run dies at seed 8 of 10."""
#     import csv as _csv
#     if not rows:
#         return
#     new = not os.path.exists(path)
#     with open(path, "a", newline="") as fh:
#         w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
#         if new:
#             w.writeheader()
#         w.writerows(rows)


# # ==============================================================================
# # 12. STATISTICS
# # ==============================================================================

# def paired_stats(a, b, rng) -> dict:
#     a, b = np.asarray(a, float), np.asarray(b, float)
#     n = min(len(a), len(b))
#     a, b = a[:n], b[:n]
#     m = np.isfinite(a) & np.isfinite(b)
#     dif = a[m] - b[m]
#     n = len(dif)
#     out = dict(n=n, mean_diff=float(dif.mean()) if n else float("nan"),
#                n_wins=int((dif > 0).sum()),
#                diffs=" ".join(f"{v:+.4f}" for v in dif))
#     if n < 2:
#         out.update(p=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
#                    test="n/a")
#         return out
#     if HAVE_SCIPY and np.any(dif != 0):
#         out["p"], out["test"] = float(_sps.wilcoxon(dif).pvalue), "wilcoxon"
#     else:
#         nz, k = int((dif != 0).sum()), int((dif > 0).sum())
#         out["p"] = float(_sps.binomtest(k, nz, 0.5).pvalue) if (
#             HAVE_SCIPY and nz) else float("nan")
#         out["test"] = "sign"
#     boot = np.array([rng.choice(dif, size=n, replace=True).mean()
#                      for _ in range(N_BOOTSTRAP)])
#     out["ci_lo"], out["ci_hi"] = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
#     return out


# def arm_table(rows: List[dict], arms: Sequence[str], only_included=True):
#     inc = [r for r in rows if r["included"] or not only_included]
#     print(f"\n{'arm':<17}{'steps':>6}{'auROC':>18}{'auPRC':>18}"
#           f"{'jf-only':>9}{'R2':>8}{'evalNCE':>10}{'|J|^2':>9}{'sv':>9}{'s':>7}")
#     for arm in arms:
#         sel = [r for r in inc if r["arm"] == arm]
#         if not sel:
#             print(f"{arm:<17}{'--':>6}")
#             continue
#         g = lambda k: np.array([r[k] for r in sel], dtype=float)
#         sd = lambda t: t.std(ddof=1) if len(t) > 1 else 0.0
#         v, p = g(PRIMARY_METRIC), g("auprc_global")
#         print(f"{arm:<17}{sel[0]['n_steps']:>6}"
#               f"{v.mean():>10.4f}+-{sd(v):.4f}{p.mean():>10.4f}+-{sd(p):.4f}"
#               f"{g('auroc_global_jf').mean():>9.4f}{g('r2_mean').mean():>8.4f}"
#               f"{g('eval_nce').mean():>+10.4f}{g('final_jfro').mean():>9.3f}"
#               f"{g('sv_ratio').mean():>9.2e}{g('seconds').mean():>7.0f}")


# def comparisons(rows, pairs, label=""):
#     rng = np.random.default_rng(0)
#     inc = [r for r in rows if r["included"]]

#     def get(arm):
#         return np.array([r[PRIMARY_METRIC] for r in
#                          sorted([x for x in inc if x["arm"] == arm],
#                                 key=lambda z: (z["d"], z["sigma_obs"], z["seed"]))])

#     print(f"\npaired comparisons on {PRIMARY_METRIC} {label}")
#     print(f"{'comparison':<34}{'diff':>9}{'95% CI':>22}{'p':>8}{'wins':>7}"
#           f"   per-seed")
#     for x, y, tag in pairs:
#         if not (any(r["arm"] == x for r in inc) and
#                 any(r["arm"] == y for r in inc)):
#             continue
#         s = paired_stats(get(x), get(y), rng)
#         print(f"{x + ' - ' + y:<34}{s['mean_diff']:>+9.4f}"
#               f"  [{s['ci_lo']:>+7.4f},{s['ci_hi']:>+7.4f}]{s['p']:>8.3f}"
#               f"{str(s['n_wins']) + '/' + str(s['n']):>7}   {s['diffs']}"
#               f"{'   <-- ' + tag if tag else ''}")


# # ==============================================================================
# # 13. STAGE 1 -- EPSILON SELECTION (reads the LOSS, never auROC)
# # ==============================================================================

# def stage_select() -> Dict[str, Optional[float]]:
#     set_scale(SELECT_SCALE)
#     csv = f"select_{RESULT_TAG}.csv"
#     arms = {"cebra": dict(scheme="clean_single", lam=0.0)}
#     for norm in ATTACK_NORMS:
#         for eps in ADV_EPS_GRID[norm]:
#             arms[f"acorn_{norm}_{eps:g}"] = dict(scheme="fork_double", lam=0.0,
#                                                  norm=norm, eps=eps)
#     print(f"\n{'#' * 78}\n# STAGE 1: epsilon selection -- {len(arms)} arms x "
#           f"{N_SEEDS} seeds\n{'#' * 78}")

#     rows: List[dict] = []
#     for d in D_LATENT_LIST:
#         for i in range(N_SEEDS):
#             r = run_seed(arms, d, SIGMA_OBS_SWEEP[0],
#                          SELECT_SEED0 + SEED_STRIDE * i, "select")
#             append_rows(r, csv)
#             rows.extend(r)

#     def mean_of(arm, key):
#         v = [r[key] for r in rows if r["arm"] == arm]
#         return float(np.mean(v)) if v else float("nan")

#     ref = mean_of("cebra", "eval_nce")
#     print(f"\n{'=' * 78}\nSELECTION  (criterion: held-out clean InfoNCE within "
#           f"{NCE_TOL_ABS} nats of baseline {ref:+.4f})")
#     print("auROC is printed for transparency but is NOT used to select.")
#     print(f"{'=' * 78}")
#     print(f"{'arm':<20}{'eps':>9}{'evalNCE':>10}{'delta':>9}{'ok':>5}"
#           f"{'auROC':>9}{'R2':>8}{'sv':>10}")
#     chosen: Dict[str, Optional[float]] = {}
#     for norm in ATTACK_NORMS:
#         feasible = []
#         for eps in ADV_EPS_GRID[norm]:
#             arm = f"acorn_{norm}_{eps:g}"
#             nce = mean_of(arm, "eval_nce")
#             delta = nce - ref
#             ok = np.isfinite(delta) and delta <= NCE_TOL_ABS
#             if ok:
#                 feasible.append(eps)
#             print(f"{arm:<20}{eps:>9.4f}{nce:>+10.4f}{delta:>+9.4f}"
#                   f"{'yes' if ok else 'no':>5}"
#                   f"{mean_of(arm, 'auroc_global'):>9.4f}"
#                   f"{mean_of(arm, 'r2_mean'):>8.4f}"
#                   f"{mean_of(arm, 'sv_ratio'):>10.2e}")
#         chosen[norm] = max(feasible) if feasible else None
#         if chosen[norm] is None:
#             print(f"  !! [{norm}] no epsilon keeps InfoNCE at its minimum. "
#                   f"Every value on the grid degrades the representation, so "
#                   f"there is no defensible choice -- extend the grid DOWNWARD "
#                   f"before falling back to a default.")
#         else:
#             print(f"  -> [{norm}] selected eps = {chosen[norm]:g} "
#                   f"(largest feasible)")
#     print(f"wrote {csv}")
#     return chosen


# # ==============================================================================
# # 14. STAGE 2 -- EVALUATION with epsilon frozen
# # ==============================================================================

# def stage_evaluate(chosen: Dict[str, Optional[float]]):
#     set_scale(EVALUATE_SCALE)
#     csv = f"evaluate_{RESULT_TAG}.csv"

#     arms = {
#         "cebra":     dict(scheme="clean_single", lam=0.0),
#         "cebra_2x":  dict(scheme="clean_double", lam=0.0),
#         "xcebra":    dict(scheme="clean_single", lam=LAMBDA_MAX),
#         "xcebra_2x": dict(scheme="clean_double", lam=LAMBDA_MAX),
#     }
#     acorn_names = {}
#     for norm in ATTACK_NORMS:
#         eps = chosen.get(norm)
#         if eps is None:
#             print(f"[warn] no selected epsilon for {norm}; falling back to the "
#                   f"fork default 0.05 (linf) / 0.354 (l2).  Say so in the "
#                   f"paper: this value was NOT tuned.")
#             eps = 0.05 if norm == "linf" else round(0.05 * math.sqrt(D_OBS), 4)
#         name = f"acorn_{norm}"
#         arms[name] = dict(scheme="fork_double", lam=0.0, norm=norm, eps=eps)
#         acorn_names[norm] = name
#     # perturbation-magnitude control, matched to the linf attack
#     n0 = ATTACK_NORMS[0]
#     arms["noise_2x"] = dict(scheme="noise_double", lam=0.0, norm=n0,
#                             eps=arms[acorn_names[n0]]["eps"])
#     # do the two mechanisms stack?
#     arms["acorn_xreg"] = dict(scheme="fork_double", lam=LAMBDA_MAX, norm=n0,
#                               eps=arms[acorn_names[n0]]["eps"])

#     print(f"\n{'#' * 78}\n# STAGE 2: evaluation -- {len(arms)} arms x "
#           f"{N_SEEDS} seeds, epsilon FROZEN")
#     for k, v in arms.items():
#         print(f"#   {k:<16} scheme={v['scheme']:<13} lam={v['lam']:<5} "
#               f"norm={v.get('norm') or '-':<5} eps={v.get('eps')}")
#     print(f"{'#' * 78}")

#     rows: List[dict] = []
#     for d in D_LATENT_LIST:
#         for sigma_obs in SIGMA_OBS_SWEEP:
#             for i in range(N_SEEDS):
#                 r = run_seed(arms, d, sigma_obs,
#                              EVAL_SEED0 + SEED_STRIDE * i, "evaluate")
#                 append_rows(r, csv)
#                 rows.extend(r)

#     n_arms = max(1, len(arms))
#     inc = [r for r in rows if r["included"]]
#     print("\n" + "=" * 78)
#     print(f" RESULTS   included seeds: {len(inc) // n_arms} of "
#           f"{len(rows) // n_arms}")
#     print(f" primary metric: {PRIMARY_METRIC}")
#     print(f" primary comparison: {acorn_names[n0]} vs cebra_2x "
#           f"(compute matched, epsilon selected on disjoint seeds by loss only)")
#     print("=" * 78)
#     arm_table(rows, list(arms))

#     best = acorn_names[n0]
#     pairs = [(best, "cebra_2x", "PRIMARY"),
#              (best, "noise_2x", "adversarial direction vs matched jitter"),
#              ("cebra_2x", "cebra", "size of the free-second-update confound"),
#              ("xcebra", "cebra", "the paper's own claim, on your data"),
#              (best, "xcebra", "implicit vs explicit regularization"),
#              (best, "cebra", "the 'as-published' number"),
#              ("acorn_xreg", best, "do the two mechanisms stack?")]
#     if len(ATTACK_NORMS) > 1:
#         other = acorn_names[ATTACK_NORMS[1]]
#         pairs.insert(1, (best, other, "linf vs l2"))
#         pairs.append((other, "cebra_2x", "l2 vs compute-matched control"))
#     comparisons(rows, pairs)

#     print("""
# How to read this
#   cebra_2x - cebra    the free second optimizer.step() in Solver.step, on its
#                       own.  In the first pilot this was +0.058 of the +0.065
#                       apparent acorn-over-cebra gap -- 89% of it.  Subtract
#                       this before crediting anything to the attack.
#   acorn - noise_2x    same update count AND same perturbation magnitude, only
#                       the direction differs.  If this is ~0, PGD is buying
#                       nothing that random jitter would not.
#   xcebra - cebra      the paper's own claim.  If this is not positive,
#                       something upstream is wrong and the rest is void.
#   acorn - cebra_2x    PRIMARY.  Positive => the attack itself buys identifiable
#                       attribution.  First-order linf adversarial training
#                       penalises eps*||grad_x L||_1, the dual-norm analogue of
#                       the paper's Frobenius penalty, so this is the
#                       mechanistically expected direction.
#   acorn - xcebra      is the implicit penalty competitive with the explicit
#                       one, at ~5x the cost?
#   acorn_xreg - acorn  do the two mechanisms stack, or are they redundant?

# Only the PRIMARY line supports a confirmatory claim; the rest are uncorrected
# for multiplicity.  Epsilon was chosen on SELECT seeds by held-out InfoNCE
# alone, so the auROC reported here is not inflated by that choice.""")
#     print(f"\nwrote {csv}")


# # ==============================================================================
# # 15. MAIN
# # ==============================================================================

# def main():
#     _banner()
#     chosen = dict(SELECTED_MANUAL)
#     if STAGE in ("select", "both"):
#         chosen = stage_select()
#     if STAGE in ("evaluate", "both"):
#         stage_evaluate(chosen)


# if __name__ == "__main__":
#     main()
