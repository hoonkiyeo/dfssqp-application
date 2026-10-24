"""Stochastic random search: Chayti et al. (2025), Algorithm 1 and Section 4.

https://arxiv.org/abs/2510.15610v2
One shared training seed for two probes. Application choices: normalize the
selected candidate to enforce unit norm, and choose the positive candidate on ties.
"""

import numpy as np

import config


def initialize():
    """
    The same fixed unit-norm initial direction as DF-SSQP.
    """

    v = np.random.default_rng(config.initialization_seed).normal(size=config.dimension)
    return {"k": 0, "w": v / np.linalg.norm(v)}


def step(state, outer_seed, eta, loss):
    """
    One iteration -> (next_state, trace); loss(w, seed) returns a scalar MSE.
    eta is fixed by the caller. Indexed random streams allow replay from k and w;
    the same outer_seed and k share the direction and training seed across eta.
    """

    k, w = state["k"], np.asarray(state["w"], dtype=np.float64)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive.")

    # 1. A unit-sphere direction and an independent training seed.
    streams = np.random.SeedSequence([outer_seed, k, 20261023]).spawn(2)
    v = np.random.default_rng(streams[0]).normal(size=len(w))
    s = v / np.linalg.norm(v)
    seed = int(np.random.default_rng(streams[1]).integers(2**32, np.iinfo(np.int64).max))

    # 2. Evaluate both raw candidates using the same training randomness.
    w_plus, w_minus = w + eta * s, w - eta * s
    L_plus, L_minus = float(loss(w_plus, seed)), float(loss(w_minus, seed))
    if not np.isfinite([L_plus, L_minus]).all():
        raise FloatingPointError("Both candidate losses must be finite.")

    # 3. Choose the lower loss (positive candidate on ties), then normalize.
    selected = "+" if L_plus <= L_minus else "-"
    chosen = w_plus if selected == "+" else w_minus
    norm = np.linalg.norm(chosen)
    if not np.isfinite(norm) or norm == 0:
        raise FloatingPointError("Cannot normalize the selected candidate.")
    next_state = {"k": k + 1, "w": chosen / norm}
    trace = {"k": k, "eta": eta, "s": s, "seed": seed,
             "w_plus": w_plus, "w_minus": w_minus,
             "L_plus": L_plus, "L_minus": L_minus, "selected": selected}
    return next_state, trace
