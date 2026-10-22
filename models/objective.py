"""One fresh fit at fixed w and seed -> validation MSE L(w, seed).

Adam: Kingma and Ba (2014), https://arxiv.org/abs/1412.6980.
The finite training budget and train-only scaling are this application's settings.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

import config
from models.single_index import SingleIndexModel


def evaluate(w, seed, training, validation, *, details=False):
    """Use z/u/y dictionaries. Return MSE, or fit details when explicitly requested.

    Each call starts a fresh model and optimizer. No normalization/projection of w.
    Returned predictions/MSE use original response units; parameters use training's
    standardized coordinates. The caller sets Torch threads once before repeated fits.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1 or not np.isfinite(w).all():
        raise ValueError("w must be a finite vector.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not config.standardize or config.minibatch_policy != "reshuffle":
        raise ValueError("This implementation uses train-only scaling and reshuffled batches.")
    training, validation = (
        {key: np.asarray(data[key], dtype=np.float64) for key in ("z", "u", "y")}
        for data in (training, validation)
    )
    for data in (training, validation):
        z, u, y = (data[key] for key in ("z", "u", "y"))
        if z.shape != (len(y), len(w)) or u.shape != y.shape or y.ndim != 1 or len(y) == 0:
            raise ValueError("Require z=(rows, len(w)) and nonempty vectors u,y.")
        if not all(np.isfinite(a).all() for a in (z, u, y)):
            raise ValueError("Training and validation data must be finite.")
    if config.training_steps < 1 or not 1 <= config.batch_size <= len(training["y"]):
        raise ValueError("Use positive training steps and a batch size within training rows.")

    # 1. Compute the scalar index; fit all scaling statistics on TRAINING only.
    train_values = (training["z"] @ w, training["u"], training["y"])
    valid_values = (validation["z"] @ w, validation["u"], validation["y"])
    stats, train_tensors, valid_tensors = {}, [], []
    for name, train, valid in zip(("index", "control", "response"), train_values, valid_values):
        mean = float(np.mean(train))
        sd = float(np.std(train, ddof=config.standardization_ddof))
        if not np.isfinite([mean, sd]).all():
            raise FloatingPointError("Nonfinite training standardization statistics.")
        scale = sd if sd > 0 else 1.0
        stats.update({name + "_mean": mean, name + "_sd": sd, name + "_scale": scale})
        train_tensors.append(torch.tensor((train - mean) / scale, dtype=torch.float64, device="cpu"))
        valid_tensors.append(torch.tensor((valid - mean) / scale, dtype=torch.float64, device="cpu"))
    tx, tu, ty = train_tensors
    vx, vu, _ = valid_tensors

    # 2. Separate initialization and minibatch streams; preserve the original seed map.
    seeds = []
    for stream in ("init", "batch"):
        digest = hashlib.sha256(f"dfssqp_inner_v1:{int(seed)}:{stream}".encode()).digest()
        seeds.append(int.from_bytes(digest[:8], "little") % (2**63))
    initialization_seed, minibatch_seed = seeds
    model = SingleIndexModel(initialization_seed)
    batch_rng = torch.Generator(device="cpu").manual_seed(minibatch_seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, betas=config.adam_betas,
        eps=config.adam_epsilon, weight_decay=0, amsgrad=False, foreach=False, fused=False,
    )
    if details:
        initial = {key: p.detach().numpy().copy() for key, p in model.named_parameters()}
        with torch.no_grad():
            initial_mse = stats["response_scale"]**2 * float(((model(tx, tu) - ty)**2).mean())
    batches = [] if details else None

    # 3. Fit beta and theta jointly for K Adam steps, including each short final batch.
    order = torch.empty(0, dtype=torch.int64, device="cpu")
    offset = 0
    with torch.enable_grad():
        for step in range(config.training_steps):
            if offset >= len(order):
                order = torch.randperm(len(ty), generator=batch_rng, device="cpu")
                offset = 0
            indices = order[offset:offset + config.batch_size]
            offset += config.batch_size
            if details:
                batches.append(indices.numpy().copy())
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tx[indices], tu[indices]) - ty[indices])**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at Adam step {step}.")
            loss.backward()
            optimizer.step()

    # 4. Return to original response units, then average ALL validation squared errors.
    model.eval()
    with torch.no_grad():
        my, sy = stats["response_mean"], stats["response_scale"]
        validation_predictions = my + sy * model(vx, vu).numpy()
        mse = float(np.mean((validation_predictions - validation["y"])**2))
        if not np.isfinite(mse):
            raise FloatingPointError("Nonfinite validation MSE.")
        if not details:
            return mse
        training_predictions = my + sy * model(tx, tu).numpy()
    return {
        "mse": mse,
        "training_mse": float(np.mean((training_predictions - training["y"])**2)),
        "initial_training_mse": initial_mse,
        "training_predictions": training_predictions,
        "validation_predictions": validation_predictions,
        "initial_parameters": initial,
        "parameters": {key: p.detach().numpy().copy() for key, p in model.named_parameters()},
        "batches": batches, "statistics": stats,
        "initialization_seed": initialization_seed, "minibatch_seed": minibatch_seed,
    }


if __name__ == "__main__":
    # One example fit at w0; this is not an outer optimization experiment.
    torch.set_num_threads(config.torch_threads)
    folder = Path(__file__).resolve().parents[1] / "data/processed"
    with np.load(folder / "training.npz", allow_pickle=False) as file:
        training = {key: file[key] for key in ("z", "u", "y")}
    with np.load(folder / "validation.npz", allow_pickle=False) as file:
        validation = {key: file[key] for key in ("z", "u", "y")}
    v = np.random.default_rng(config.initialization_seed).normal(size=config.dimension)
    mse = evaluate(v / np.linalg.norm(v), config.evaluation_seeds[0], training, validation)
    print(f"Validation MSE: {mse:.12f}")
