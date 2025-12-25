"""对指定 split（train/val/test）批量用 Teacher/Student 推理，并统计整体概率分布。

说明：
- 数据构造与训练一致：复用 `prepare_dataloaders_kd` 返回的 dataset；
- 概率口径：sigmoid(logits) = is_open=1（仍营业）的概率；
- Windows 推理时强制 num_workers=0，避免多进程共享内存问题。

示例：
  python -m survival_kd.scripts.predict_teacher_student_split ^
      --split train ^
      --teacher-checkpoint checkpoints_kd/teacher_best.pt ^
      --student-checkpoint checkpoints_kd/student_best.pt ^
      --hidden-dim 512 ^
      --batch-size 256 ^
      --out-csv preds_train_teacher_student.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from survival_kd.data.loaders import prepare_dataloaders_kd
from survival_kd.models.bilstm_student import BiLSTMStudent
from survival_kd.models.mamba_teacher import MambaTeacher


def _move_batch_to_device(batch: Dict[str, object], device: torch.device) -> None:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


def _sigmoid_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits.detach()).cpu().view(-1).numpy()


def _summary(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "p01": float(np.quantile(arr, 0.01)) if arr.size else float("nan"),
        "p05": float(np.quantile(arr, 0.05)) if arr.size else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
        "p95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "p99": float(np.quantile(arr, 0.99)) if arr.size else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch predict teacher/student probs on a KD split and summarize distributions.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--teacher-checkpoint", type=str, default="checkpoints_kd/teacher_best.pt")
    parser.add_argument("--student-checkpoint", type=str, default="checkpoints_kd/student_best.pt")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-csv", type=str, default="", help="若非空，导出逐店铺预测结果 CSV")
    parser.add_argument("--device", type=str, default="", help="可选：cpu/cuda；默认自动选择")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.exists(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")
    if not os.path.exists(args.student_checkpoint):
        raise FileNotFoundError(f"Student checkpoint not found: {args.student_checkpoint}")

    loaders = prepare_dataloaders_kd(batch_size=args.batch_size)
    ds = {"train": loaders["train_loader"].dataset, "val": loaders["val_loader"].dataset, "test": loaders["test_loader"].dataset}[args.split]

    inf_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    teacher = MambaTeacher(d_model=args.hidden_dim).to(device)
    t_state = torch.load(args.teacher_checkpoint, map_location="cpu")
    teacher.load_state_dict(t_state.get("state", t_state))
    teacher.eval()

    student = BiLSTMStudent(d_model=args.hidden_dim).to(device)
    s_state = torch.load(args.student_checkpoint, map_location="cpu")
    student.load_state_dict(s_state.get("state", s_state))
    student.eval()

    all_teacher: List[np.ndarray] = []
    all_student: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    all_n_hist: List[np.ndarray] = []

    for batch in inf_loader:
        _move_batch_to_device(batch, device)
        t_logits = teacher(batch)  # [B,1]
        s_logits = student(batch)  # [B,1]
        t_prob = _sigmoid_probs(t_logits)
        s_prob = _sigmoid_probs(s_logits)

        y = batch["is_open"].detach().cpu().view(-1).numpy().astype(np.float32)
        rid = batch["restaurant_id"].detach().cpu().view(-1).numpy().astype(np.int64)
        n_hist = batch["last2_mask"].detach().cpu().sum(dim=1).view(-1).numpy().astype(np.int64)

        all_teacher.append(t_prob)
        all_student.append(s_prob)
        all_labels.append(y)
        all_ids.append(rid)
        all_n_hist.append(n_hist)

    t_arr = np.concatenate(all_teacher) if all_teacher else np.array([])
    s_arr = np.concatenate(all_student) if all_student else np.array([])
    y_arr = np.concatenate(all_labels) if all_labels else np.array([])
    rid_arr = np.concatenate(all_ids) if all_ids else np.array([])
    n_hist_arr = np.concatenate(all_n_hist) if all_n_hist else np.array([])

    print(f"split={args.split} rows={int(rid_arr.size)} device={device}")
    print("teacher_open_prob summary:", _summary(t_arr))
    print("student_open_prob summary:", _summary(s_arr))
    print("label is_open summary:", {0: int((y_arr < 0.5).sum()), 1: int((y_arr >= 0.5).sum())})
    if n_hist_arr.size:
        print("n_history_reviews (last2_mask sum) summary:", _summary(n_hist_arr.astype(float)))

    if args.out_csv:
        out_df = pd.DataFrame(
            {
                "restaurant_id": rid_arr.astype(np.int64),
                "is_open": y_arr.astype(np.float32),
                "n_history_reviews": n_hist_arr.astype(np.int64),
                "teacher_open_prob": t_arr.astype(np.float32),
                "student_open_prob": s_arr.astype(np.float32),
            }
        )
        out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"saved_csv={args.out_csv} rows={len(out_df)}")


if __name__ == "__main__":
    main()

