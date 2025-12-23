"""蒸馏训练循环（HSK 迁移版）

要点：
- 教师：使用全量序列训练；
- 学生：使用“最后两年”序列（last2_mask），并用 KL 散度对齐教师概率（温度 T）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from survival.training.losses import BinaryFocalLoss
from survival.utils.logging import setup_logging
from survival.utils.seed import seed_everything


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> None:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


def _smooth_labels(labels: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    return labels * (1.0 - 2 * p) + p


def _collect_probs(outputs: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(outputs.detach()).cpu().numpy().flatten()


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def _metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs > threshold).astype(int)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "bacc": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc": _safe_auc(labels, probs),
    }


def _binary_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.clamp(min=eps, max=1 - eps)
    q = q.clamp(min=eps, max=1 - eps)
    return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()


def train_teacher(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 5e-5,
    log_dir: str = "logs_hsk_distill",
    log_filename: str = "teacher.log",
    checkpoint_dir: str = "checkpoints_hsk_distill",
    max_train_steps: Optional[int] = None,
    max_val_steps: Optional[int] = None,
    max_test_steps: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    seed_everything()
    log_path, logger = setup_logging(log_dir, log_filename)
    logger.info("[HSK][Teacher] Logging to %s", log_path)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = BinaryFocalLoss(alpha=0.7, gamma=3)

    best_val_auc = 0.0
    best_state = None
    history: Dict[str, Dict[str, list]] = {
        split: {k: [] for k in ("loss", "acc", "auc", "f1", "recall", "precision")}
        for split in ("train", "val", "test")
    }

    for epoch in range(int(num_epochs)):
        model.train()
        train_probs, train_labels = [], []
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[HSK][Teacher] Epoch {epoch + 1}/{num_epochs}")
        for step_idx, batch in enumerate(pbar, start=1):
            _move_batch_to_device(batch, device)
            outputs = model(batch)
            labels = (1.0 - batch["is_open"]).float().view(-1, 1)
            loss = criterion(outputs, _smooth_labels(labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            train_probs.append(_collect_probs(outputs))
            train_labels.append(labels.detach().cpu().numpy().flatten())

            if max_train_steps and step_idx >= max_train_steps:
                break

        train_probs_arr = np.concatenate(train_probs) if train_probs else np.array([])
        train_labels_arr = np.concatenate(train_labels) if train_labels else np.array([])
        train_metrics = _metrics_from_probs(train_probs_arr, train_labels_arr)
        history["train"]["loss"].append(total_loss / max(1, len(train_probs)))
        for k in ("acc", "auc", "f1", "recall", "precision"):
            history["train"][k].append(train_metrics[k])
        logger.info(
            "[HSK][Teacher][Train] loss=%.4f auc=%.4f acc=%.4f f1=%.4f",
            history["train"]["loss"][-1],
            train_metrics["auc"],
            train_metrics["acc"],
            train_metrics["f1"],
        )

        def _eval(loader, split: str, max_steps: Optional[int]) -> Tuple[float, Dict[str, float]]:
            model.eval()
            probs, labels = [], []
            loss_sum = 0.0
            with torch.no_grad():
                for step_idx, batch in enumerate(loader, start=1):
                    _move_batch_to_device(batch, device)
                    outputs = model(batch)
                    y = (1.0 - batch["is_open"]).float().view(-1, 1)
                    loss = criterion(outputs, _smooth_labels(y))
                    loss_sum += float(loss.item())
                    probs.append(_collect_probs(outputs))
                    labels.append(y.detach().cpu().numpy().flatten())
                    if max_steps and step_idx >= max_steps:
                        break
            probs_arr = np.concatenate(probs) if probs else np.array([])
            labels_arr = np.concatenate(labels) if labels else np.array([])
            metrics = _metrics_from_probs(probs_arr, labels_arr)
            return loss_sum / max(1, len(probs)), metrics

        val_loss, val_metrics = _eval(val_loader, "val", max_val_steps)
        test_loss, test_metrics = _eval(test_loader, "test", max_test_steps)
        for split, loss_val, metrics in (("val", val_loss, val_metrics), ("test", test_loss, test_metrics)):
            history[split]["loss"].append(loss_val)
            for k in ("acc", "auc", "f1", "recall", "precision"):
                history[split][k].append(metrics[k])
            logger.info(
                "[HSK][Teacher][%s] loss=%.4f auc=%.4f acc=%.4f f1=%.4f",
                split,
                loss_val,
                metrics["auc"],
                metrics["acc"],
                metrics["f1"],
            )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {"state": model.state_dict(), "meta": {"epoch": epoch + 1, "auc": float(best_val_auc)}}
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(best_state, os.path.join(checkpoint_dir, "teacher_best.pt"))

    if best_state is not None:
        model.load_state_dict(best_state["state"])  # type: ignore[arg-type]
    return model, history


def train_student_distill(
    *,
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 5e-5,
    temperature: float = 2.0,
    lambda_kd: float = 0.7,
    lambda_sup: float = 1.0,
    log_dir: str = "logs_hsk_distill",
    log_filename: str = "student.log",
    checkpoint_dir: str = "checkpoints_hsk_distill",
    max_train_steps: Optional[int] = None,
    max_val_steps: Optional[int] = None,
    max_test_steps: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    seed_everything()
    log_path, logger = setup_logging(log_dir, log_filename)
    logger.info("[HSK][Student] Logging to %s", log_path)

    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    criterion_sup = BinaryFocalLoss(alpha=0.7, gamma=3)

    best_val_auc = 0.0
    best_state = None
    history: Dict[str, Dict[str, list]] = {
        split: {k: [] for k in ("loss", "acc", "auc", "f1", "recall", "precision")}
        for split in ("train", "val", "test")
    }
    T = float(temperature)

    def _eval(loader, split: str, max_steps: Optional[int]) -> Tuple[float, Dict[str, float]]:
        student.eval()
        probs, labels = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for step_idx, batch in enumerate(loader, start=1):
                _move_batch_to_device(batch, device)
                logits_student = student(batch)
                y = (1.0 - batch["is_open"]).float().view(-1, 1)
                loss = criterion_sup(logits_student, _smooth_labels(y))
                loss_sum += float(loss.item())
                probs.append(_collect_probs(logits_student))
                labels.append(y.detach().cpu().numpy().flatten())
                if max_steps and step_idx >= max_steps:
                    break
        probs_arr = np.concatenate(probs) if probs else np.array([])
        labels_arr = np.concatenate(labels) if labels else np.array([])
        metrics = _metrics_from_probs(probs_arr, labels_arr)
        return loss_sum / max(1, len(probs)), metrics

    for epoch in range(int(num_epochs)):
        student.train()
        train_probs, train_labels = [], []
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[HSK][Student] Epoch {epoch + 1}/{num_epochs}")
        for step_idx, batch in enumerate(pbar, start=1):
            _move_batch_to_device(batch, device)

            with torch.no_grad():
                logits_teacher = teacher(batch)
            logits_student = student(batch)

            labels = (1.0 - batch["is_open"]).float().view(-1, 1)
            loss_sup = criterion_sup(logits_student, _smooth_labels(labels))

            p_teacher = torch.sigmoid(logits_teacher / T)
            p_student = torch.sigmoid(logits_student / T)
            loss_kd = (T * T) * _binary_kl(p_teacher, p_student)

            loss = float(lambda_sup) * loss_sup + float(lambda_kd) * loss_kd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            train_probs.append(_collect_probs(logits_student))
            train_labels.append(labels.detach().cpu().numpy().flatten())

            if max_train_steps and step_idx >= max_train_steps:
                break

        train_probs_arr = np.concatenate(train_probs) if train_probs else np.array([])
        train_labels_arr = np.concatenate(train_labels) if train_labels else np.array([])
        train_metrics = _metrics_from_probs(train_probs_arr, train_labels_arr)
        history["train"]["loss"].append(total_loss / max(1, len(train_probs)))
        for k in ("acc", "auc", "f1", "recall", "precision"):
            history["train"][k].append(train_metrics[k])
        logger.info(
            "[HSK][Student][Train] loss=%.4f auc=%.4f acc=%.4f f1=%.4f",
            history["train"]["loss"][-1],
            train_metrics["auc"],
            train_metrics["acc"],
            train_metrics["f1"],
        )

        val_loss, val_metrics = _eval(val_loader, "val", max_val_steps)
        test_loss, test_metrics = _eval(test_loader, "test", max_test_steps)
        for split, loss_val, metrics in (("val", val_loss, val_metrics), ("test", test_loss, test_metrics)):
            history[split]["loss"].append(loss_val)
            for k in ("acc", "auc", "f1", "recall", "precision"):
                history[split][k].append(metrics[k])
            logger.info(
                "[HSK][Student][%s] loss=%.4f auc=%.4f acc=%.4f f1=%.4f",
                split,
                loss_val,
                metrics["auc"],
                metrics["acc"],
                metrics["f1"],
            )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {"state": student.state_dict(), "meta": {"epoch": epoch + 1, "auc": float(best_val_auc)}}
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(best_state, os.path.join(checkpoint_dir, "student_best.pt"))

    if best_state is not None:
        student.load_state_dict(best_state["state"])  # type: ignore[arg-type]
    return student, history

