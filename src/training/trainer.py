from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import EvalMetrics, compute_accuracy, compute_roc_auc


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    running_loss = 0.0
    sample_count = 0

    all_labels: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob_fake: list[np.ndarray] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

        all_labels.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob_fake.append(probs[:, 1].cpu().numpy())

    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    y_score = np.concatenate(all_prob_fake) if all_prob_fake else np.array([], dtype=np.float32)

    avg_loss = running_loss / max(sample_count, 1)
    accuracy = compute_accuracy(y_true, y_pred)
    roc_auc = compute_roc_auc(y_true, y_score)

    return EvalMetrics(loss=avg_loss, accuracy=accuracy, roc_auc=roc_auc)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    selection_metric: str,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "selection_metric": selection_metric,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
