"""
MiSTP: Boucherouite et al. (AAAI 2024), Algorithm 1.

https://doi.org/10.1609/aaai.v38i18.30016
Application: one shared training seed for three fresh fits; project the selected
point onto the unit sphere. Exact ties prefer current, then positive, then negative.
"""

import numpy as np

import config


def initialize():
    """
    The same fixed unit-norm initial direction as DF-SSQP.
    """

    v = np.random.default_rng(config.initialization_seed).normal(size=config.dimension)
    return {"k": 0, "w": v / np.linalg.norm(v)}


def step(state, outer_seed, alpha, loss):
    """
    One projected MiSTP iteration -> (next_state, trace).
    """

    k, w = state["k"], np.asarray(state["w"], dtype=np.float64)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")

    # 1. Draw a unit direction and a training seed, shared across alpha choices.
    streams = np.random.SeedSequence([outer_seed, k, 20261023]).spawn(2)
    u = np.random.default_rng(streams[0]).normal(size=len(w))
    s = u / np.linalg.norm(u)
    seed = int(np.random.default_rng(streams[1]).integers(2**32, np.iinfo(np.int64).max))

    # 2. Evaluate current, positive and negative candidates with the same seed.
    w_plus, w_minus = w + alpha * s, w - alpha * s
    L_current = float(loss(w, seed))
    L_plus = float(loss(w_plus, seed))
    L_minus = float(loss(w_minus, seed))
    if not np.isfinite([L_current, L_plus, L_minus]).all():
        raise FloatingPointError("All three candidate losses must be finite.")

    # 3. Select the smallest loss; argmin keeps the first candidate on exact ties.
    selected = int(np.argmin([L_current, L_plus, L_minus]))
    chosen = (w, w_plus, w_minus)[selected]

    # 4. Project the selected point; the next iteration uses this normalized w.
    norm = np.linalg.norm(chosen)
    if not np.isfinite(norm) or norm == 0:
        raise FloatingPointError("Cannot normalize the selected candidate.")
    next_state = {"k": k + 1, "w": chosen / norm}
    trace = {"k": k, "alpha": alpha, "s": s, "seed": seed,
             "w_plus": w_plus, "w_minus": w_minus,
             "L_current": L_current, "L_plus": L_plus, "L_minus": L_minus,
             "selected": ("current", "+", "-")[selected]}
    return next_state, trace
