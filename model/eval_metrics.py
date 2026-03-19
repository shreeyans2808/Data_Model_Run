import torch
import torch.nn as nn
import numpy as np
from pysteps.verification.spatialscores import fss


def soft_csi_loss(pred, target, threshold=1.0, smooth=1e-4):
    """
    Differentiable soft CSI loss for training.
    Uses sigmoid to approximate hard thresholding.
    CSI = TP / (TP + FP + FN), loss = 1 - CSI
    """
    # soft binarization via sigmoid
    pred_bin   = torch.sigmoid((pred   - threshold) * 10)
    target_bin = (target >= threshold).float()

    TP = (pred_bin * target_bin).sum()
    FP = (pred_bin * (1 - target_bin)).sum()
    FN = ((1 - pred_bin) * target_bin).sum()

    csi  = (TP + smooth) / (TP + FP + FN + smooth)
    return 1 - csi  # minimize loss → maximize CSI


def hard_csi(pred, target, threshold=1.0, smooth=1e-4):
    """
    Hard CSI for validation logging (non-differentiable).
    """
    pred_bin   = (pred   >= threshold).float()
    target_bin = (target >= threshold).float()

    TP = (pred_bin * target_bin).sum()
    FP = (pred_bin * (1 - target_bin)).sum()
    FN = ((1 - pred_bin) * target_bin).sum()

    return (TP + smooth) / (TP + FP + FN + smooth)


def compute_fss(pred, actual, threshold=1.0, scale=5):
    """
    Compute Fractions Skill Score (FSS)

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Shape can be:
            (B, T, H, W)
            (T, H, W)
            (H, W)

    actual : same shape as pred

    threshold : rainfall threshold (mm/hr)
    scale : neighbourhood window size

    Returns
    -------
    float : FSS score
    """

    # --- convert to numpy ---
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()

    # --- handle batch ---
    if pred.ndim == 4:
        pred = pred.mean(axis=1)      # average over time → (B,H,W)
        actual = actual.mean(axis=1)

        scores = []
        for i in range(pred.shape[0]):
            scores.append(
                fss(pred[i], actual[i], thr=threshold, scale=scale)
            )
        return float(np.mean(scores))

    elif pred.ndim == 3:
        pred = pred.mean(axis=0)
        actual = actual.mean(axis=0)

        return float(
            fss(pred, actual, thr=threshold, scale=scale)
        )

    elif pred.ndim == 2:
        return float(
            fss(pred, actual, thr=threshold, scale=scale)
        )

    else:
        raise ValueError("Unsupported shape for FSS computation")


def exp_weighted_temporal_fss(
    pred,
    actual,
    threshold=1.0,
    scale=5,
):
    """
    Exponentially weighted temporal FSS.

    Metric =
        sum_t w(t) * FSS(t)
        -------------------
             sum_t w(t)

    where

        w(t) = exp(t / T)

    Later timesteps → higher importance.

    Parameters
    ----------
    pred : torch.Tensor or np.ndarray
        Shape (B, T, H, W)

    actual : same shape

    Returns
    -------
    float
    """

    # --- convert to numpy ---
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()

    B, T, H, W = pred.shape

    # --- exponential weights ---
    t_idx = np.arange(1, T + 1)
    weights = np.exp(t_idx / T)

    weight_sum = np.sum(weights)

    total_score = 0.0

    for t in range(T):

        batch_scores = []

        for b in range(B):
            score = fss(
                pred[b, t],
                actual[b, t],
                thr=threshold,
                scale=scale
            )
            batch_scores.append(score)

        mean_fss_t = np.mean(batch_scores)

        total_score += weights[t] * mean_fss_t

    return float(total_score / weight_sum)