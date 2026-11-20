"""Auditable dense regression network: chronological validation and exact forward traces."""

from copy import deepcopy

import numpy as np
from sklearn.neural_network import MLPRegressor


def fit_network(preprocess, frame, names, weights, seed):
    # Validation is the last fifth of the development period, never the final test set.
    cut = int(len(frame) * 0.8)
    fit, val = frame.iloc[:cut], frame.iloc[cut:]
    x = preprocess.fit_transform(fit[names])
    v = preprocess.transform(val[names])
    model = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        learning_rate_init=0.003,
        alpha=0.01,
        batch_size=min(128, len(fit)),
        random_state=seed,
    )
    history, best, best_mae, stale = [], None, float("inf"), 0
    checkpoints = []
    for epoch in range(1, 401):
        model.partial_fit(
            x, fit.observed_price, sample_weight=np.asarray(weights)[:cut]
        )
        train_mae = float(np.abs(model.predict(x) - fit.observed_price).mean())
        val_mae = float(np.abs(model.predict(v) - val.observed_price).mean())
        history.append(
            {
                "epoch": epoch,
                "loss": float(model.loss_),
                "train_mae": train_mae,
                "validation_mae": val_mae,
            }
        )
        if epoch == 1 or epoch % 10 == 0:
            checkpoints.append(
                {
                    "epoch": epoch,
                    "weights": deepcopy(model.coefs_),
                    "biases": deepcopy(model.intercepts_),
                }
            )
        if val_mae < best_mae - 0.0001:
            best, best_mae, stale = deepcopy(model), val_mae, 0
            best_epoch = epoch
        else:
            stale += 1
        if stale >= 35:
            break
    checkpoints = [c for c in checkpoints if c["epoch"] != best_epoch]
    checkpoints.append(
        {
            "epoch": best_epoch,
            "weights": deepcopy(best.coefs_),
            "biases": deepcopy(best.intercepts_),
        }
    )
    best.replay_checkpoints_ = sorted(checkpoints, key=lambda c: c["epoch"])
    return best, {
        "architecture": [x.shape[1], 16, 8, 1],
        "activation": "ReLU",
        "optimizer": "Adam",
        "history": history,
        "best_epoch": best_epoch,
        "replay_epochs": [c["epoch"] for c in best.replay_checkpoints_],
        "fit_n": len(fit),
        "validation_n": len(val),
        "validation_start": str(val.timestamp.min()),
        "fit_end": str(fit.timestamp.max()),
        "selection": "Lowest chronological validation MAE; patience 35, maximum 400 epochs. Final test rows are never used for checkpoint selection.",
        "parameters": sum(
            w.size + b.size for w, b in zip(best.coefs_, best.intercepts_)
        ),
    }


def trace(pipe, row, checkpoint=None):
    model = pipe["model"]
    if not isinstance(model, MLPRegressor):
        return None
    a = pipe["preprocess"].transform(row)[0]
    layers = [
        {
            "name": "Standardized / encoded inputs",
            "values": a.tolist(),
            "labels": pipe["preprocess"].get_feature_names_out().tolist(),
        }
    ]
    weights = checkpoint["weights"] if checkpoint else model.coefs_
    biases = checkpoint["biases"] if checkpoint else model.intercepts_
    for i, (w, b) in enumerate(zip(weights, biases)):
        z = a @ w + b
        a = np.maximum(z, 0) if i < len(model.coefs_) - 1 else z
        layers.append(
            {
                "name": f"Hidden {i + 1} · ReLU" if i < 2 else "Price · CU",
                "values": a.tolist(),
                "preactivation": z.tolist(),
                "bias": b.tolist(),
            }
        )
    return {
        "layers": layers,
        "weights": [w.tolist() for w in weights],
        "prediction": float(a[0]),
        "description": "Exact forward pass from saved weights. Edge sign/color and neuron values are measured, not decorative activity.",
    }
