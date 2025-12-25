"""蒸馏训练 CLI

用法示例：
- 教师阶段：
  python -m survival_kd.scripts.train_distill --stage teacher --epochs 10 --batch-size 64
- 学生阶段：
  python -m survival_kd.scripts.train_distill --stage student --teacher-checkpoint checkpoints_kd/teacher_best.pt --epochs 10

说明：
- 本脚本依赖 survival_kd.data.loaders.prepare_dataloaders_kd 构建数据；
- 不使用图结构数据；
- 注释为中文，便于快速理解与二次开发。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from survival_kd.data.loaders import prepare_dataloaders_kd
from survival_kd.models.mamba_teacher import MambaTeacher
from survival_kd.models.bilstm_student import BiLSTMStudent
from survival_kd.training.distill import train_teacher, train_student_distill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-Student Distillation Training (KD)")
    parser.add_argument("--stage", type=str, choices=["teacher", "student"], required=True, help="训练阶段：teacher 或 student")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--teacher-checkpoint", type=str, default="", help="学生阶段加载的教师权重路径")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-kd", type=float, default=0.7)
    parser.add_argument("--lambda-sup", type=float, default=1.0)
    parser.add_argument("--log-filename", type=str, default="kd_train.log")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_kd")
    parser.add_argument(
        "--id-log-dir",
        type=str,
        default="",
        help="若非空，将每个 epoch 实际见到的 restaurant_id 写入该目录（train/val/test 分开）",
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--max-test-steps", type=int, default=None)
    return parser.parse_args()


def _best_epoch_from_history(history: Dict[str, Dict[str, list]], metric: str = "auc") -> Optional[int]:
    vals = history.get("val", {}).get(metric, [])
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    if not np.isfinite(arr).any():
        return None
    return int(np.nanargmax(arr) + 1)  # 1-based


def _write_history_csv(history: Dict[str, Dict[str, list]], out_path: str) -> None:
    import csv

    splits = [k for k in ("train", "val", "test") if k in history]
    metrics = [k for k in ("loss", "auc", "acc", "bacc", "f1", "recall", "precision") if any(k in history[s] for s in splits)]

    n_epochs = 0
    for s in splits:
        for m in metrics:
            n_epochs = max(n_epochs, len(history[s].get(m, [])))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["epoch"] + [f"{s}_{m}" for s in splits for m in metrics]
        writer.writerow(header)
        for e in range(n_epochs):
            row: list[Any] = [e + 1]
            for s in splits:
                for m in metrics:
                    seq = history[s].get(m, [])
                    row.append(seq[e] if e < len(seq) else "")
            writer.writerow(row)


def _plot_history(history: Dict[str, Dict[str, list]], out_path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Plot] matplotlib not installed; skip metric curve plotting.")
        return

    best_epoch = _best_epoch_from_history(history, metric="auc")
    splits = [k for k in ("train", "val", "test") if k in history]
    metrics = [k for k in ("loss", "auc", "f1", "recall", "precision", "acc") if any(k in history[s] for s in splits)]
    if not metrics:
        print("[Plot] No metrics found in history; skip plotting.")
        return

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        for split in splits:
            ys = history.get(split, {}).get(metric, [])
            if not ys:
                continue
            ax.plot(np.arange(1, len(ys) + 1), ys, label=split)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1)
            if metric == "auc":
                best_val = history.get("val", {}).get("auc", [])[best_epoch - 1]
                ax.set_title(f"{metric} (best epoch={best_epoch}, val_auc={best_val:.4f})")
            else:
                ax.set_title(metric)
        else:
            ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[Plot] Saved metric curves to: {out_path}")


def _dump_split_ids(out_dir: str, split_name: str, loader) -> None:
    try:
        df = getattr(loader.dataset, "restaurant_data", None)
        if df is None or "restaurant_id" not in df.columns:
            return
        ids = sorted({int(x) for x in df["restaurant_id"].astype(str).tolist() if str(x).isdigit()})
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"split_{split_name}_all.txt")
        with open(path, "w", encoding="utf-8") as handle:
            for rid in ids:
                handle.write(f"{rid}\n")
    except Exception:
        return


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = prepare_dataloaders_kd(
        batch_size=args.batch_size,
        reference_year=2019,  # 标签年口径：训练时不使用 operation_latest_year
        use_macro_features=False,  # 按需求：训练不使用宏观特征
    )
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    id_log_dir = args.id_log_dir.strip() or None
    if id_log_dir:
        _dump_split_ids(id_log_dir, "train", train_loader)
        _dump_split_ids(id_log_dir, "val", val_loader)
        _dump_split_ids(id_log_dir, "test", test_loader)

    if args.stage == "teacher":
        teacher = MambaTeacher(d_model=args.hidden_dim)
        _, history = train_teacher(
            model=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            log_filename=args.log_filename,
            checkpoint_dir=args.checkpoint_dir,
            id_log_dir=id_log_dir,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            max_test_steps=args.max_test_steps,
        )
        _plot_history(history, os.path.join(args.checkpoint_dir, "teacher_metrics.png"), "KD Teacher metrics")
        _write_history_csv(history, os.path.join(args.checkpoint_dir, "teacher_metrics.csv"))
    else:
        if not args.teacher_checkpoint or not os.path.exists(args.teacher_checkpoint):
            raise FileNotFoundError("学生阶段需要提供有效的 --teacher-checkpoint 路径")
        teacher = MambaTeacher(d_model=args.hidden_dim)
        state = torch.load(args.teacher_checkpoint, map_location="cpu")
        teacher.load_state_dict(state.get("state", state))
        student = BiLSTMStudent(d_model=args.hidden_dim)
        _, history = train_student_distill(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            lambda_kd=args.lambda_kd,
            lambda_sup=args.lambda_sup,
            log_filename=args.log_filename,
            checkpoint_dir=args.checkpoint_dir,
            id_log_dir=id_log_dir,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            max_test_steps=args.max_test_steps,
        )
        _plot_history(history, os.path.join(args.checkpoint_dir, "student_metrics.png"), "KD Student metrics")
        _write_history_csv(history, os.path.join(args.checkpoint_dir, "student_metrics.csv"))


if __name__ == "__main__":
    main()
