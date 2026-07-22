"""Scoring helpers copied from evaluation_normal (evaluation/evaluation.py).

Do not add new formulas here — only the stock accumulate + metrics pattern:

    pred_error = normal_utils.compute_normal_error(pred_norm, gt_norm)
    total_normal_errors = pred_error[gt_norm_mask]   # (or cat)
    metrics = normal_utils.compute_normal_metrics(total_normal_errors)

Used by finetune in-loop val and score_finaldataset so paper numbers match.
"""

from __future__ import annotations

import torch

from evaluation.util.normal_utils import compute_normal_error, compute_normal_metrics


def accumulate_normal_errors(pred_norm, gt_norm, gt_norm_mask, total_normal_errors=None):
    """Exact accumulate block from evaluation_normal (generate_prediction path).

    Args match evaluation_normal:
      pred_norm, gt_norm: (B, 3, H, W)
      gt_norm_mask: boolean, broadcastable to pred_error (B, 1, H, W)
                   e.g. (B, 1, H, W) or (1, 1, H, W)
    """
    # evaluation/evaluation.py lines ~341-345
    pred_error = compute_normal_error(pred_norm, gt_norm)
    if total_normal_errors is None:
        total_normal_errors = pred_error[gt_norm_mask]
    else:
        total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_norm_mask]), dim=0)
    return total_normal_errors


def metrics_from_errors(total_normal_errors):
    """Exact finalize from evaluation_normal: compute_normal_metrics on pooled errors."""
    if total_normal_errors is None or total_normal_errors.numel() == 0:
        return None
    # evaluation/evaluation.py lines ~353-354
    return compute_normal_metrics(total_normal_errors)
