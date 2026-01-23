"""训练教师模型（HSK distill 迁移版）。

运行示例：
  python -m survival_new_data.distill.scripts.train_teacher --epochs 10 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from survival_new_data.distill.data import prepare_dataloaders_distill
from survival_new_data.distill.models import MambaTeacher
from survival_new_data.distill.training import train_teacher
from survival_st_gcn.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train distill teacher model on survival_hsk data.")
    p.add_argument("--data-dir", type=str, default="", help="可选：survival_hsk/data 的目录路径（默认自动定位）")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--text-dim", type=int, default=768)
    p.add_argument("--img-dim", type=int, default=512)
    p.add_argument("--feature-dim", type=int, default=8)
    p.add_argument("--time-step", type=str, choices=["year", "half", "quarter"], default="year")
    p.add_argument("--max-offset-years", type=int, default=10, help="时间偏移 embedding 的最大跨度（年）")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_hsk_distill")
    p.add_argument("--log-dir", type=str, default="logs_hsk_distill")
    p.add_argument("--log-filename", type=str, default="teacher.log")
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--max-test-steps", type=int, default=None)
    p.add_argument("--limit-restaurants", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Configure logging before data loading so loader progress (including image embeddings) is visible.
    setup_logging(args.log_dir, args.log_filename)

    loaders = prepare_dataloaders_distill(
        batch_size=args.batch_size,
        data_dir=data_dir,
        limit_restaurants=args.limit_restaurants,
        text_dim=args.text_dim,
        img_dim=args.img_dim,
        feature_dim=args.feature_dim,
        time_step=args.time_step,
    )

    model = MambaTeacher(
        d_model=args.hidden_dim,
        text_dim=args.text_dim,
        img_dim=args.img_dim,
        feature_dim=args.feature_dim,
        time_step=args.time_step,
        max_offset_years=args.max_offset_years,
    )
    train_teacher(
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        test_loader=loaders["test_loader"],
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        log_dir=args.log_dir,
        log_filename=args.log_filename,
        checkpoint_dir=args.checkpoint_dir,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        max_test_steps=args.max_test_steps,
        configure_logging=False,
    )


if __name__ == "__main__":
    main()
