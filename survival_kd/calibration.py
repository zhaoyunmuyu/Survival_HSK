from __future__ import annotations

"""
Logit 标定模块：
- 冻结学生模型，只在其输出 logits 上增加一个非常小的线性标定层；
- 通过监督损失 + “负样本拉回”边际损失，整体压低误报的死亡概率，并拉大学生模型对正负样本的区分度。

用法概览：
- 训练：在脚本中加载已训练好的学生模型，调用 train_logit_calibrator；
- 推理：加载标定层权重，对学生模型输出 logits 先过标定层再做 sigmoid。
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from survival_st_gcn.training.losses import BinaryFocalLoss
from survival_st_gcn.utils.logging import setup_logging


class LogitCalibrator(nn.Module):
    """
    非常小的标定层：z' = scale * z + bias
    - scale、bias 为两个可学习标量参数；
    - 不改变模型结构，只对输出 logits 做全局缩放和平移。
    """

    def __init__(self, init_scale: float = 1.0, init_bias: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.scale * logits + self.bias


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> None:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


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


def train_logit_calibrator(
    *,
    student: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 1e-2,
    margin: float = 0.05,
    lambda_margin: float = 1.0,
    log_dir: str = "logs_kd",
    log_filename: str = "calibrator.log",
    checkpoint_path: str = "checkpoints_kd/calibrator.pt",
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
) -> Tuple[LogitCalibrator, Dict[str, Any]]:
    """
    在冻结学生模型的前提下训练标定层：
    - 监督损失：对标定后的 logits 使用 BinaryFocalLoss，对齐 1 - is_open；
    - 负样本拉回：在每个 batch 中，使“关闭”餐厅的平均死亡概率至少比“存活”餐厅高 margin。
    """
    log_path, logger = setup_logging(log_dir, log_filename)
    logger.info("[KD][Calibrator] Logging to %s", log_path)

    student = student.to(device)
    student.eval()

    calibrator = LogitCalibrator().to(device)
    optimizer = optim.Adam(calibrator.parameters(), lr=lr, weight_decay=0.0)
    criterion_sup = BinaryFocalLoss(alpha=0.7, gamma=3)

    best_val_auc = 0.0
    best_state: Dict[str, Any] | None = None
    history: Dict[str, Dict[str, list]] = {
        split: {k: [] for k in ("loss", "acc", "auc", "f1", "recall", "precision")}
        for split in ("train", "val")
    }

    for epoch in range(num_epochs):
        calibrator.train()
        train_probs, train_labels = [], []
        total_loss = 0.0

        for step_idx, batch in enumerate(train_loader, start=1):
            _move_batch_to_device(batch, device)

            with torch.no_grad():
                logits_student = student(batch)  # [B, 1]

            labels = (1.0 - batch["is_open"]).float().view(-1, 1)  # 1 = 死亡, 0 = 存活
            logits_cal = calibrator(logits_student)

            # 监督损失：对齐标签（带轻微平滑）
            labels_smooth = labels * 0.8 + 0.1  # 等价于 p=0.1 的双向 label smoothing
            loss_sup = criterion_sup(logits_cal, labels_smooth)

            # 负样本拉回：正负样本平均概率之间保持 margin
            probs = torch.sigmoid(logits_cal).view(-1)
            labels_flat = labels.view(-1)
            pos_mask = labels_flat > 0.5
            neg_mask = ~pos_mask

            loss_margin = torch.tensor(0.0, device=device)
            if pos_mask.any() and neg_mask.any():
                pos_mean = probs[pos_mask].mean()
                neg_mean = probs[neg_mask].mean()
                loss_margin = torch.relu(margin + neg_mean - pos_mean)

            loss = loss_sup + lambda_margin * loss_margin

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            train_probs.append(_collect_probs(logits_cal))
            train_labels.append(labels.detach().cpu().numpy().flatten())

            if max_train_steps is not None and step_idx >= max_train_steps:
                break

        train_probs_arr = np.concatenate(train_probs) if train_probs else np.array([])
        train_labels_arr = np.concatenate(train_labels) if train_labels else np.array([])
        train_metrics = _metrics_from_probs(train_probs_arr, train_labels_arr)
        history["train"]["loss"].append(total_loss / max(1, len(train_probs)))
        for k in ("acc", "auc", "f1", "recall", "precision"):
            history["train"][k].append(train_metrics[k])
        logger.info(
            "[Calibrator][Train] epoch=%d loss=%.4f auc=%.4f acc=%.4f f1=%.4f",
            epoch + 1,
            history["train"]["loss"][-1],
            train_metrics["auc"],
            train_metrics["acc"],
            train_metrics["f1"],
        )

        # 验证：仅评估标定后的概率
        calibrator.eval()
        val_probs, val_labels = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for step_idx, batch in enumerate(val_loader, start=1):
                _move_batch_to_device(batch, device)
                logits_student = student(batch)
                labels = (1.0 - batch["is_open"]).float().view(-1, 1)
                logits_cal = calibrator(logits_student)

                labels_smooth = labels * 0.8 + 0.1
                loss_sup = criterion_sup(logits_cal, labels_smooth)

                probs = torch.sigmoid(logits_cal).view(-1)
                labels_flat = labels.view(-1)
                pos_mask = labels_flat > 0.5
                neg_mask = ~pos_mask
                loss_margin = torch.tensor(0.0, device=device)
                if pos_mask.any() and neg_mask.any():
                    pos_mean = probs[pos_mask].mean()
                    neg_mean = probs[neg_mask].mean()
                    loss_margin = torch.relu(margin + neg_mean - pos_mean)

                loss = loss_sup + lambda_margin * loss_margin
                val_loss_sum += float(loss.item())

                val_probs.append(_collect_probs(logits_cal))
                val_labels.append(labels.detach().cpu().numpy().flatten())

                if max_val_steps is not None and step_idx >= max_val_steps:
                    break

        val_probs_arr = np.concatenate(val_probs) if val_probs else np.array([])
        val_labels_arr = np.concatenate(val_labels) if val_labels else np.array([])
        val_metrics = _metrics_from_probs(val_probs_arr, val_labels_arr)

        history["val"]["loss"].append(val_loss_sum / max(1, len(val_probs)))
        for k in ("acc", "auc", "f1", "recall", "precision"):
            history["val"][k].append(val_metrics[k])
        logger.info(
            "[Calibrator][Val] epoch=%d loss=%.4f auc=%.4f acc=%.4f f1=%.4f",
            epoch + 1,
            history["val"]["loss"][-1],
            val_metrics["auc"],
            val_metrics["acc"],
            val_metrics["f1"],
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {
                "state": calibrator.state_dict(),
                "meta": {"epoch": epoch + 1, "auc": float(best_val_auc)},
            }
            import os

            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(best_state, checkpoint_path)
            logger.info("[Calibrator] Updated best checkpoint: %.4f -> %.4f", best_val_auc, val_metrics["auc"])

    if best_state is not None:
        calibrator.load_state_dict(best_state["state"])  # type: ignore[arg-type]

    return calibrator, history

