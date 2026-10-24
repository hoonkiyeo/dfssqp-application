"""
ZO-RSGD: Li, Balasubramanian, and Ma, Algorithm 2.
https://arxiv.org/abs/2003.11238

Application: unit sphere, normalization retraction, m=1.
"""

import numpy as np

import config

def initialize():
    """
    The same fixed initial direction as DF-SSQP for the comparison.
    """

    v = np.random.default_rng(config.initialization_seed).normal(
        size=config.dimension
    )
    return {"k": 0, "w": v / np.linalg.norm(v)}

def step(state, outer_seed, eta, mu, loss):
    """
    One iteration -> (next_state, trace).
    """

    k, w = state["k"], state["w"]

    # DF-SSQP seed map: separate direction and training streams.
    streams = np.random.SeedSequence(
        [outer_seed, k, 20260910]
    ).spawn(2)
    rng = np.random.default_rng(streams[0])
    seed = int(
        np.random.default_rng(streams[1]).integers(
            2**32, np.iinfo(np.int64).max
        )
    )

    # Step1. Draw a Gaussian vector and project it onto the tangent space
    v = rng.normal(size=len(w))
    u = v - (w @ v) * w

    # Step2. Retract (project) the probe and evaluate both points with the same seed.
    w_probe = w + mu * u
    w_probe = w_probe / np.linalg.norm(w_probe)

    L_current = float(loss(w, seed))
    L_probe = float(loss(w_probe, seed))

    # Step3. Estimate the Riemannian gradient.
    g_hat = ((L_probe - L_current) / mu) * u

    # Step4. Take the gradient step and retract (project) onto the unit sphere.
    w_next = w - eta * g_hat
    w_next = w_next / np.linalg.norm(w_next)

    next_state = {"k": k + 1, "w": w_next}
    trace = {
        "k": k, "seed": seed, "eta": eta, "mu": mu,
        "u": u, "w_probe": w_probe,
        "L_current": L_current, "L_probe": L_probe,
        "g_hat": g_hat,
    }
    return next_state, trace