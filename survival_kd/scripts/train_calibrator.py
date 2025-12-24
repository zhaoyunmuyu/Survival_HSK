from __future__ import annotations

"""
训练学生模型输出层的标定层（不重新训练学生主体）：
- 冻结 BiLSTMStudent，仅训练一个非常小的 LogitCalibrator；
- 目标：整体压低误报的死亡概率，并拉大正负样本（死亡/存活）之间的概率差异。

示例：
  python -m survival_kd.scripts.train_calibrator ^
      --student-checkpoint checkpoints_kd/student_best.pt ^
      --hidden-dim 512 ^
      --epochs 5 ^
      --batch-size 128
"""

import argparse
import os

import torch

from survival_kd.calibration import train_logit_calibrator
from survival_kd.data.loaders import prepare_dataloaders_kd
from survival_kd.models.bilstm_student import BiLSTMStudent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small logit calibrator on top of the student model.")
    parser.add_argument(
        "--student-checkpoint",
        type=str,
        default="checkpoints_kd/student_best.pt",
        help="学生模型权重路径（已完成蒸馏训练的 checkpoint）",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="学生模型 d_model（需与训练时一致）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="标定训练的 batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="标定层训练轮数（通常较少即可）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="标定层学习率（仅 2 个参数，可以适当大一些）",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="负样本拉回的概率间隔 margin（正样本平均概率至少高于负样本 margin）",
    )
    parser.add_argument(
        "--lambda-margin",
        type=float,
        default=1.0,
        help="负样本拉回损失的权重系数",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_kd/calibrator.pt",
        help="标定层 checkpoint 保存路径",
    )
    parser.add_argument(
        "--limit-restaurants",
        type=int,
        default=None,
        help="可选：仅使用前 N 家餐厅做标定，加快调试速度",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="可选：限制每个 epoch 的训练步数，用于快速测试",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=None,
        help="可选：限制每个 epoch 的验证步数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.student_checkpoint):
        raise FileNotFoundError(f"Student checkpoint not found: {args.student_checkpoint}")

    loaders = prepare_dataloaders_kd(
        batch_size=args.batch_size,
        limit_restaurants=args.limit_restaurants,
        downsample_train_open=False,
    )
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]

    student = BiLSTMStudent(d_model=args.hidden_dim)
    state = torch.load(args.student_checkpoint, map_location="cpu")
    student.load_state_dict(state.get("state", state))

    _, _ = train_logit_calibrator(
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        lambda_margin=args.lambda_margin,
        checkpoint_path=args.checkpoint,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
    )


if __name__ == "__main__":
    main()
