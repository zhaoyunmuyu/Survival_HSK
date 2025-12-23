"""预览 distill DataLoader 的 batch 结构与形状。

运行：
  python -m survival_hsk.distill.scripts.preview_batch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from survival_new_data.distill.data import prepare_dataloaders_distill


def _print_tensor(name: str, value: torch.Tensor) -> None:
    print(f"- {name}: shape={tuple(value.shape)} dtype={value.dtype}")


def _print_sample(sample: Dict[str, torch.Tensor]) -> None:
    keys = sorted(sample.keys())
    print("sample keys:", keys)
    for k in keys:
        v = sample[k]
        if isinstance(v, torch.Tensor):
            _print_tensor(k, v)
        else:
            print(f"- {k}: {type(v)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview survival_hsk distill dataloader batches.")
    p.add_argument("--data-dir", type=str, default="", help="可选：survival_hsk/data 的目录路径（默认自动定位）")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit-restaurants", type=int, default=None)
    p.add_argument("--text-dim", type=int, default=768)
    p.add_argument("--img-dim", type=int, default=512)
    p.add_argument("--feature-dim", type=int, default=8)
    p.add_argument("--time-step", type=str, choices=["year", "half", "quarter"], default="year")
    p.add_argument("--last-window-years", type=int, default=2, help="学生模型 last-window 的年数（默认 2 年）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    loaders = prepare_dataloaders_distill(
        batch_size=args.batch_size,
        data_dir=data_dir,
        limit_restaurants=args.limit_restaurants,
        text_dim=args.text_dim,
        img_dim=args.img_dim,
        feature_dim=args.feature_dim,
        time_step=args.time_step,
        last_window_years=args.last_window_years,
    )
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    print("dataset sizes:")
    print("- train:", len(train_loader.dataset))
    print("- val:", len(val_loader.dataset))
    print("- test:", len(test_loader.dataset))

    sample = train_loader.dataset[0]
    _print_sample(sample)


if __name__ == "__main__":
    main()
