"""
创建蒸馏版 DataLoader 并查看一条样例
- 使用 survival_kd.data.loaders.prepare_dataloaders_kd（不加载图数据）
- 打印 train/val/test 样本数与第一条样例字段形状
运行：  python -m survival_kd.scripts.preview_kd_dataloader
"""

from __future__ import annotations

from typing import Dict

import os
import torch

from survival_kd.data.loaders import prepare_dataloaders_kd


def _print_sample(sample: Dict[str, torch.Tensor]) -> None:
    print("样例字段：")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"- {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"- {k}: {type(v)}")


def main() -> None:
    # 可选关闭 tqdm 进度条，避免输出过长
    os.environ.setdefault("TQDM_DISABLE", "1")

    loaders = prepare_dataloaders_kd(batch_size=4, limit_restaurants=None)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    print("DataLoader 基本信息：")
    print("- train len:", len(train_loader.dataset))
    print("- val len:", len(val_loader.dataset))
    print("- test len:", len(test_loader.dataset))

    sample = train_loader.dataset[0]
    _print_sample(sample)


if __name__ == "__main__":
    main()

