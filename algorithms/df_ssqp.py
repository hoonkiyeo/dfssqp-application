"""First-order DF-SSQP: Na (2025), Algorithm 1, arXiv:2510.22458v1.

Application: c(w)=w.T@w-1; B=I; AI uses 2w.T, ZI estimates the Jacobian.
Spectral clipping and numerical-zero continuation retain the existing application
choices. The latter is a finite-precision convention, not a new paper rule.
"""

import numpy as np

import config


def initialize():
    """One normalized Gaussian direction and the configured initial state."""
    v = np.random.default_rng(config.initialization_seed).normal(size=config.dimension)
    return {
        "k": 0, "w": v / np.linalg.norm(v), "lam": np.array([config.lambda_initial]),
        "g_bar": np.full(config.dimension, config.gradient_average_initial),
        "G_bar": np.full((1, config.dimension), config.jacobian_average_initial),
        "tau": config.tau_initial, "nu": config.nu_initial,
    }


@np.errstate(over="raise", invalid="raise", divide="raise", under="ignore")
def step(state, outer_seed, mode, loss):
    """One iteration -> (next_state, trace); loss(w, seed) returns a scalar MSE.

    State is a plain dictionary; lam has one entry and G_bar has one row.
    Neither input state nor the global RNG is changed. No early stop or projection.
    This implementation selects the lower endpoint, with the agreed psi=0.
    """
    k, w, lam = state["k"], state["w"], state["lam"]
    n = len(w)
    if mode not in ("AI", "ZI") or config.psi != 0:
        raise ValueError("Use AI or ZI with the agreed psi=0 step rule.")
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer.")
    if (w.shape != (n,) or lam.shape != (1,)
            or state["g_bar"].shape != (n,) or state["G_bar"].shape != (1, n)):
        raise ValueError("Require vector w,g_bar, one multiplier and a one-row Jacobian.")
    if not all(np.isfinite(state[key]).all() for key in ("w", "lam", "g_bar", "G_bar", "tau", "nu")):
        raise ValueError("State must be finite.")
    if (min(state["tau"], state["nu"], config.L_f, config.L_c) <= 0
            or not (0 < config.sigma < 1 and 0 < config.epsilon < 1)):
        raise ValueError("Use positive tau,nu,L_f,L_c and sigma,epsilon in (0,1).")
    if not 0 < config.jacobian_lower <= config.jacobian_upper < np.inf:
        raise ValueError("Require finite positive ordered Jacobian bounds.")

    # 1. Schedules and indexed random streams (the original real-data seed map).
    alpha = config.iota1 / np.power(np.float64(k + 1), np.float64(config.p1))
    beta = config.iota2 / np.power(np.float64(k + 1), np.float64(config.p2))
    b = config.iota3 / np.power(np.float64(k + 1), np.float64(config.p3))
    if not np.isfinite([alpha, beta, b]).all() or min(alpha, beta, b) <= 0:
        raise ValueError("Schedules must be finite and positive.")
    streams = np.random.SeedSequence([outer_seed, k, 20260910]).spawn(2)
    bits = np.random.default_rng(streams[0]).integers(0, 2, size=n, dtype=np.int8)
    delta = 2.0 * bits.astype(np.float64) - 1.0
    seed = int(np.random.default_rng(streams[1]).integers(2**32, np.iinfo(np.int64).max))

    # 2. Two unprojected probes; fresh fits share their training seed.
    w_plus, w_minus = w + b * delta, w - b * delta
    w_plus.flags.writeable = w_minus.flags.writeable = False
    L_plus, L_minus = np.float64(loss(w_plus, seed)), np.float64(loss(w_minus, seed))
    if not np.isfinite([L_plus, L_minus]).all():
        raise FloatingPointError("Both probe losses must be finite.")

    # 3. Objective estimate and moving average.
    g_hat = ((L_plus - L_minus) / (2.0 * b)) * np.reciprocal(delta)
    g_bar = (1.0 - beta) * state["g_bar"] + beta * g_hat

    # 4. Constraint Jacobian: exact in AI, estimated and averaged in ZI.
    c = np.array([float(w @ w) - 1.0])
    if mode == "AI":
        G_hat = 2.0 * w[None, :]
        G_bar = G_hat
    else:
        c_plus = np.array([float(w_plus @ w_plus) - 1.0])
        c_minus = np.array([float(w_minus @ w_minus) - 1.0])
        G_hat = ((c_plus - c_minus) / (2.0 * b))[:, None] * np.reciprocal(delta)[None, :]
        G_bar = (1.0 - beta) * state["G_bar"] + beta * G_hat

    # 5. Clip singular values; leave an in-range Jacobian exactly unchanged.
    lower, upper = config.jacobian_lower, config.jacobian_upper
    if not np.any(G_bar):
        singular, G = np.zeros(1), np.zeros_like(G_bar)
        G[0, 0] = lower
        clipped = True
    else:
        U, singular, Vt = np.linalg.svd(G_bar, full_matrices=False)
        bounded = np.clip(singular, lower, upper)
        clipped = not np.array_equal(singular, bounded)
        G = (U * bounded[None, :]) @ Vt if clipped else G_bar.copy()

    # 6. B=I: solve the full SQP system for primal d and dual increment eta.
    K = np.block([[np.eye(n), G.T], [G, np.zeros((1, 1))]])
    lagrangian_gradient = g_bar + G.T @ lam
    rhs = -np.concatenate((lagrangian_gradient, c))
    solution = np.linalg.solve(K, rhs)
    d, eta = solution[:n], solution[n:]
    checks = _roundoff_checks(w, lam, g_bar, G, c, d, eta, K, rhs)
    zero = checks["zero_step"]
    if not np.isfinite(solution).all():
        raise FloatingPointError("SQP solve returned nonfinite values.")

    # 7. Merit parameter and predicted local-model reduction.
    g_dot_d, d_squared = float(g_bar @ d), float(d @ d)
    denominator, c_norm = g_dot_d + d_squared, float(np.linalg.norm(c))
    tau_trial = (np.inf if denominator <= 0 or checks["zero_denominator"]
                 else (1.0 - config.sigma) * c_norm / denominator)
    tau = state["tau"] if state["tau"] <= tau_trial else (1.0 - config.epsilon) * tau_trial
    reduction = -tau * (g_dot_d + 0.5 * d_squared) + c_norm
    lower_reduction = 0.5 * tau * d_squared + config.sigma * c_norm
    tolerance = 128.0 * np.finfo(float).eps * max(1.0, abs(reduction), abs(lower_reduction))
    if checks["zero_denominator"]:
        tolerance += tau * checks["denominator_error_bound"]
    if (not np.isfinite([tau, reduction, tolerance]).all() or tau <= 0
            or (reduction <= 0 and not zero) or reduction - lower_reduction < -tolerance):
        raise FloatingPointError("Invalid merit parameter or predicted reduction.")

    # 8. A certified zero step retains nu; it still completes the ordinary update.
    if not zero and d_squared == 0:
        raise FloatingPointError("Nonzero primal direction has zero squared norm.")
    nu_trial = np.inf if zero else reduction / d_squared
    nu = state["nu"] if state["nu"] <= nu_trial else (1.0 - config.epsilon) * nu_trial
    if not np.isfinite(nu) or nu <= 0 or (not zero and not np.isfinite(nu_trial)):
        raise FloatingPointError("Invalid ratio parameter.")

    # 9. Actual step and raw primal/dual update; carry the unclipped averages.
    s = nu * alpha / (tau * config.L_f + config.L_c)
    w_next, lam_next = w + s * d, lam + s * eta
    if (not np.isfinite(s) or s <= 0
            or not np.isfinite(w_next).all() or not np.isfinite(lam_next).all()):
        raise FloatingPointError("Invalid actual step or next iterate.")
    next_state = {"k": k + 1, "w": w_next, "lam": lam_next,
                  "g_bar": g_bar, "G_bar": G_bar, "tau": tau, "nu": nu}
    trace = {
        "k": k, "alpha": alpha, "beta": beta, "b": b, "delta": delta, "seed": seed,
        "w_plus": w_plus, "w_minus": w_minus, "L_plus": L_plus, "L_minus": L_minus,
        "g_hat": g_hat, "g_bar": g_bar, "G_hat": G_hat, "G_bar": G_bar,
        "G": G, "singular_values": singular, "clipped": clipped, "c": c,
        "K": K, "rhs": rhs, "d": d, "eta": eta, "g_dot_d": g_dot_d,
        "d_squared": d_squared, "tau_denominator": denominator, "tau_trial": tau_trial,
        "tau": tau, "reduction": reduction, "nu_trial": nu_trial, "nu": nu, "s": s,
        "roundoff": checks,
    }
    return next_state, trace


# Roundoff checks below only classify numerical zeros and validate finite quantities.
# They preserve the original roundoff certificates without record/configuration classes.
def _gamma(count):
    unit = np.finfo(np.float64).eps / 2.0
    return count * unit / (1.0 - count * unit)


def _backward_error(K, solution, rhs):
    residual = float(np.linalg.norm(K @ solution - rhs, ord=np.inf))
    scale = float(np.linalg.norm(K, ord=np.inf) * np.linalg.norm(solution, ord=np.inf)
                  + np.linalg.norm(rhs, ord=np.inf))
    return residual, residual / scale if scale else (0.0 if residual == 0 else np.inf)


def _roundoff_checks(w, lam, g, G, c, d, eta, K, rhs):
    """Recognize cancellation from KKT/constraint error bounds, not a tuned threshold."""
    n = len(w)
    c_error = _gamma(n + 2) * (abs(float(w @ w)) + 1.0)
    feasible = bool(np.all(np.abs(c) <= c_error))
    lagrangian_gradient = g + G.T @ lam
    residual, backward = _backward_error(K, np.concatenate((d, eta)), rhs)
    zero_eta = np.linalg.lstsq(G.T, -lagrangian_gradient, rcond=None)[0]
    _, zero_backward = _backward_error(K, np.concatenate((np.zeros(n), zero_eta)), rhs)
    floor = _gamma(K.shape[0] + 2)
    stable = backward <= floor
    zero = feasible and stable and zero_backward <= floor

    # Bound cancellation in g.T@d + d.T@d using the two KKT identities.
    stationarity = d + G.T @ eta + lagrangian_gradient
    feasibility = G @ d + c
    stationarity_error = _gamma(G.shape[0] + 2) * (
        np.abs(d) + np.abs(G).T @ np.abs(eta) + np.abs(lagrangian_gradient))
    feasibility_error = _gamma(n + 1) * (np.abs(G) @ np.abs(d) + np.abs(c))
    stationarity_upper = float(np.linalg.norm(np.abs(stationarity) + stationarity_error))
    feasibility_upper = float(np.linalg.norm(np.abs(feasibility) + feasibility_error))
    g_dot_d, d_squared = float(g @ d), float(d @ d)
    unit = np.finfo(np.float64).eps / 2.0
    dot_error = _gamma(n) * (float(np.abs(g) @ np.abs(d)) + float(np.abs(d) @ np.abs(d)))
    dot_error += unit * (abs(g_dot_d) + abs(d_squared)) / (1.0 - unit)
    bound = (float(np.linalg.norm(d)) * stationarity_upper
             + float(np.linalg.norm(lam + eta)) * feasibility_upper
             + float(np.abs(lam + eta) @ np.abs(c)) + dot_error)
    if not np.isfinite([c_error, residual, backward, zero_backward, bound]).all():
        raise FloatingPointError("Nonfinite roundoff certificate.")
    return {"zero_step": bool(zero), "solve_stable": bool(stable),
            "zero_denominator": bool(feasible and stable and abs(g_dot_d + d_squared) <= bound),
            "constraint_error_bound": c_error, "kkt_residual_inf": residual,
            "kkt_backward_error": backward, "zero_backward_error": zero_backward,
            "roundoff_floor": floor, "denominator_error_bound": bound}
