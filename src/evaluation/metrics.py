from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import recall_score, roc_auc_score


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    roc_auc: float | None
    recall: float


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))
