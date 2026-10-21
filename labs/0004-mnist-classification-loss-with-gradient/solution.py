# --- cell 1 ---
import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    # Your implementation here
    N, C = preds.shape
    one_hot_target = np.zeros((N,C), dtype = preds.dtype)
    one_hot_target[np.arange(N), target] = 1.0
    eps = 1e-15
    preds = np.clip(preds, eps, 1.0 - eps)
    per_sample_loss = -np.sum(one_hot_target*np.log(preds), axis = 1)
    loss = np.mean(per_sample_loss)
    grad = preds - one_hot_target
    return (loss, grad)