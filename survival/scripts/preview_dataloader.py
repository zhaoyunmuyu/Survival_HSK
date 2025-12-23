"""
创建 DataLoader 并查看一条样例。

- 使用 loaders.prepare_dataloaders（已实现 data/raw 优先与图数据兜底）。
- 打印 train/val/test 的长度信息与第一条样例的各字段形状。

运行：
  python -m survival.scripts.preview_dataloader
"""

from __future__ import annotations

from typing import Dict

import torch

from survival.data.loaders import prepare_dataloaders


def _print_sample(sample: Dict[str, torch.Tensor]) -> None:
    print("样例字段：")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"- {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"- {k}: {type(v)}")


def main() -> None:
    loaders = prepare_dataloaders(batch_size=4, limit_restaurants=None)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    yearly_graphs = loaders.get("yearly_graphs", [])

    print("DataLoader 基本信息：")
    print("- train len:", len(train_loader.dataset))
    print("- val len:", len(val_loader.dataset))
    print("- test len:", len(test_loader.dataset))
    print("- yearly_graphs:", len(yearly_graphs))

    sample = train_loader.dataset[0]
    _print_sample(sample)


if __name__ == "__main__":
    main()
