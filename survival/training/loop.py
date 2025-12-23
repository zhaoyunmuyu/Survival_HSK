"""训练/验证/测试循环（含日志、指标与最优模型保存）

要点：
- 标签平滑（0.1/0.8）+ FocalLoss；
- 在训练集上网格搜索决策阈值（balanced accuracy），并用于 val/test；
- 每个 epoch 打印与记录关键指标；
- 在验证 AUC 提升时输出 test 预测结果 CSV 并缓存最佳权重；
- 保存完整训练历史到 `training_history.npy`。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
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

from survival.models.survival import RestaurantSurvivalModel
from survival.training.losses import BinaryFocalLoss
from survival.utils.logging import setup_logging
from survival.utils.seed import seed_everything


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> None:
    """将 batch 中的张量递归移动到指定设备。"""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)


def _smooth_labels(labels: torch.Tensor) -> torch.Tensor:
    """标签平滑：缓解过拟合并提高鲁棒性。"""
    smoothed = labels * 0.8 + 0.1
    return smoothed


def _collect_probs(outputs: torch.Tensor) -> np.ndarray:
    """将 logits 转概率并搬运到 CPU/NumPy，便于后续评估。"""
    probs = torch.sigmoid(outputs.detach()).cpu().numpy().flatten()
    return probs


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def _grid_search_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """在给定概率和标签上网格搜索最佳阈值（balanced accuracy）。"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_accuracy = 0.0
    best_threshold = 0.5
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        current_accuracy = balanced_accuracy_score(labels, preds)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_threshold = threshold
    return best_threshold, best_accuracy


def train_model(
    num_epochs: int,
    total_rests: int,
    train_loader,
    val_loader,
    test_loader,
    yearly_graphs,
    device: torch.device,
    *,
    lr_schedule: Dict[int, float] | None = None,
    log_filename: str = "training.log",
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    max_train_steps: Optional[int] = None,
    max_val_steps: Optional[int] = None,
    max_test_steps: Optional[int] = None,
) -> Tuple[RestaurantSurvivalModel, Dict[str, Dict[str, list]]]:
    seed_everything()
    log_path, logger = setup_logging(log_dir, log_filename)
    logger.info("Logging to %s", log_path)

    survival_model = RestaurantSurvivalModel(total_rests=total_rests, hidden_dim=512).to(device)
    optimizer = optim.Adam(survival_model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = BinaryFocalLoss(alpha=0.7, gamma=3)

    best_val_auc = 0.0
    best_model_weights = None
    history = {
        "train": {"loss": [], "acc": [], "auc": [], "auc1": [], "recall": [], "precision": [], "f1": []},
        "val": {"loss": [], "acc": [], "auc": [], "auc1": [], "recall": [], "precision": [], "f1": []},
        "test": {"loss": [], "acc": [], "auc": [], "auc1": [], "recall": [], "precision": [], "f1": []},
    }

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        if lr_schedule and (epoch + 1) in lr_schedule:
            current_lr = lr_schedule[epoch + 1]
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            logger.info("Adjusted LR to %f", current_lr)

        survival_model.train()
        train_loss = 0.0
        train_probs = []
        train_labels = []
        start_time = time.time()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Train")
        for step_idx, batch in enumerate(train_pbar, start=1):
            _move_batch_to_device(batch, device)
            outputs = survival_model(batch, yearly_graphs, device)
            labels = (1.0 - batch["is_open"]).float().view(-1, 1)
            smoothed_labels = _smooth_labels(labels)
            loss = criterion(outputs, smoothed_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_probs.append(_collect_probs(outputs))
            train_labels.append(labels.cpu().numpy().flatten())
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if max_train_steps and step_idx >= max_train_steps:
                break

        train_loss /= len(train_loader.dataset)
        train_probs = np.concatenate(train_probs)
        train_labels_arr = np.concatenate(train_labels)
        best_threshold, best_accuracy = _grid_search_threshold(train_probs, train_labels_arr)
        train_preds = (train_probs > best_threshold).astype(int)
        train_acc = accuracy_score(train_labels_arr, train_preds)
        train_auc = _safe_auc(train_labels_arr, train_probs)
        train_auc1 = _safe_auc(train_labels_arr, train_preds)
        train_recall = recall_score(train_labels_arr, train_preds, zero_division=0)
        train_precision = precision_score(train_labels_arr, train_preds, zero_division=0)
        train_f1 = f1_score(train_labels_arr, train_preds, zero_division=0)

        history["train"]["loss"].append(train_loss)
        history["train"]["acc"].append(train_acc)
        history["train"]["auc"].append(train_auc)
        history["train"]["auc1"].append(train_auc1)
        history["train"]["recall"].append(train_recall)
        history["train"]["precision"].append(train_precision)
        history["train"]["f1"].append(train_f1)

        survival_model.eval()
        val_loss = 0.0
        val_probs = []
        val_labels = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Val")
            for step_idx, batch in enumerate(val_pbar, start=1):
                _move_batch_to_device(batch, device)
                outputs = survival_model(batch, yearly_graphs, device)
                labels = (1.0 - batch["is_open"]).float().view(-1, 1)
                smoothed_labels = _smooth_labels(labels)
                loss = criterion(outputs, smoothed_labels)
                val_loss += loss.item() * labels.size(0)
                val_probs.append(_collect_probs(outputs))
                val_labels.append(labels.cpu().numpy().flatten())
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if max_val_steps and step_idx >= max_val_steps:
                    break
        val_loss /= len(val_loader.dataset)
        val_probs = np.concatenate(val_probs)
        val_labels_arr = np.concatenate(val_labels)
        val_preds = (val_probs > best_threshold).astype(int)
        val_acc = accuracy_score(val_labels_arr, val_preds)
        val_auc = _safe_auc(val_labels_arr, val_probs)
        val_auc1 = _safe_auc(val_labels_arr, val_preds)
        val_recall = recall_score(val_labels_arr, val_preds, zero_division=0)
        val_precision = precision_score(val_labels_arr, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels_arr, val_preds, zero_division=0)

        history["val"]["loss"].append(val_loss)
        history["val"]["acc"].append(val_acc)
        history["val"]["auc"].append(val_auc)
        history["val"]["auc1"].append(val_auc1)
        history["val"]["recall"].append(val_recall)
        history["val"]["precision"].append(val_precision)
        history["val"]["f1"].append(val_f1)

        test_loss = 0.0
        test_probs = []
        test_labels = []
        all_test_ids = []
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Test")
            for step_idx, batch in enumerate(test_pbar, start=1):
                _move_batch_to_device(batch, device)
                outputs = survival_model(batch, yearly_graphs, device)
                labels = (1.0 - batch["is_open"]).float().view(-1, 1)
                smoothed_labels = _smooth_labels(labels)
                loss = criterion(outputs, smoothed_labels)
                test_loss += loss.item() * labels.size(0)
                test_probs.append(_collect_probs(outputs))
                test_labels.append(labels.cpu().numpy().flatten())
                all_test_ids.append(batch["restaurant_id"].cpu().numpy().flatten())
                test_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if max_test_steps and step_idx >= max_test_steps:
                    break
        test_loss /= len(test_loader.dataset)
        test_probs_arr = np.concatenate(test_probs)
        test_labels_arr = np.concatenate(test_labels)
        test_preds = (test_probs_arr > best_threshold).astype(int)
        test_acc = accuracy_score(test_labels_arr, test_preds)
        test_auc = _safe_auc(test_labels_arr, test_probs_arr)
        test_auc1 = _safe_auc(test_labels_arr, test_preds)
        test_recall = recall_score(test_labels_arr, test_preds, zero_division=0)
        test_precision = precision_score(test_labels_arr, test_preds, zero_division=0)
        test_f1 = f1_score(test_labels_arr, test_preds, zero_division=0)

        history["test"]["loss"].append(test_loss)
        history["test"]["acc"].append(test_acc)
        history["test"]["auc"].append(test_auc)
        history["test"]["auc1"].append(test_auc1)
        history["test"]["recall"].append(test_recall)
        history["test"]["precision"].append(test_precision)
        history["test"]["f1"].append(test_f1)

        epoch_time = time.time() - start_time
        logger.info(
            "Epoch %d/%d - time %.2fs - LR %.6f",
            epoch + 1,
            num_epochs,
            epoch_time,
            current_lr,
        )
        logger.info(
            "Train - loss %.4f acc %.4f auc %.4f auc1 %.4f recall %.4f precision %.4f f1 %.4f",
            train_loss,
            train_acc,
            train_auc,
            train_auc1,
            train_recall,
            train_precision,
            train_f1,
        )
        logger.info(
            "Val   - loss %.4f acc %.4f auc %.4f auc1 %.4f recall %.4f precision %.4f f1 %.4f",
            val_loss,
            val_acc,
            val_auc,
            val_auc1,
            val_recall,
            val_precision,
            val_f1,
        )
        logger.info(
            "Test  - loss %.4f acc %.4f auc %.4f auc1 %.4f recall %.4f precision %.4f f1 %.4f",
            test_loss,
            test_acc,
            test_auc,
            test_auc1,
            test_recall,
            test_precision,
            test_f1,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_weights = {"survival_model": survival_model.state_dict()}
            logger.info("New best model (Val AUC %.4f) saved", best_val_auc)
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "model_state": best_model_weights["survival_model"],
                    "meta": {
                        "epoch": epoch + 1,
                        "val_auc": float(best_val_auc),
                        "total_rests": int(total_rests),
                        "hidden_dim": 512,
                    },
                },
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
            df = pd.DataFrame(
                {
                    "probability": test_probs_arr,
                    "true_label": test_labels_arr,
                    "restaurant_id": np.concatenate(all_test_ids),
                }
            )
            df.to_csv(f"{epoch + 1}_best_test_results.csv", index=False)

    if best_model_weights:
        survival_model.load_state_dict(best_model_weights["survival_model"])
    # Always save final model at end
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "model_state": survival_model.state_dict(),
                "meta": {
                    "epoch": num_epochs,
                    "best_val_auc": float(best_val_auc),
                    "total_rests": int(total_rests),
                    "hidden_dim": 512,
                },
            },
            os.path.join(checkpoint_dir, "last_model.pt"),
        )
    except Exception as exc:
        logger.warning("Failed to save final model checkpoint: %s", exc)

    np.save("training_history.npy", history)
    logger.info("Training complete - best Val AUC %.4f", best_val_auc)
    return survival_model, history
