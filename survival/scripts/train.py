"""训练入口脚本（CLI）

用法示例：
- 默认使用缓存与 GPU（若可用）：
  `python survival_predict_v3-Copy2.py --batch-size 64 --epochs 100`
- 关闭缓存以强制重新预处理：
  `python survival_predict_v3-Copy2.py --no-cache`
"""

from __future__ import annotations

import argparse
import torch

from survival.data.cache import get_dataloaders_with_cache
from survival.training.loop import train_model


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train the restaurant survival model")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset caching")
    parser.add_argument("--cache-dir", type=str, default="preprocessed_data_cache", help="Cache directory for preprocessed datasets")
    parser.add_argument("--log-filename", type=str, default="restaurant_survival_training.log", help="Log filename")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--max-train-steps", type=int, default=50, help="Max train steps per epoch for quick smoke tests (None for full)")
    parser.add_argument("--max-val-steps", type=int, default=20, help="Max validation steps per epoch (None for full)")
    parser.add_argument("--max-test-steps", type=int, default=20, help="Max test steps per epoch (None for full)")
    return parser.parse_args()


def main() -> None:
    """主流程：
    - 解析参数与设置设备；
    - 通过缓存加载/构建 DataLoader 与图数据；
    - 组装学习率调度并启动训练。
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = get_dataloaders_with_cache(batch_size=args.batch_size, cache_dir=args.cache_dir, use_cache=not args.no_cache)
    train_loader = data_loaders["train_loader"]
    val_loader = data_loaders["val_loader"]
    test_loader = data_loaders["test_loader"]
    yearly_graphs = data_loaders.get("yearly_graphs")
    if not yearly_graphs:
        raise RuntimeError("Yearly graphs not loaded. Ensure data files are accessible.")
    total_rests = yearly_graphs[0].total_rests

    lr_schedule = {1: 0.00005, 10: 0.00001, 15: 0.000005, 50: 0.000001}

    train_model(
        num_epochs=args.epochs,
        total_rests=total_rests,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        yearly_graphs=yearly_graphs,
        device=device,
        lr_schedule=lr_schedule,
        log_filename=args.log_filename,
        checkpoint_dir=args.checkpoint_dir,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        max_test_steps=args.max_test_steps,
    )


if __name__ == "__main__":
    main()
