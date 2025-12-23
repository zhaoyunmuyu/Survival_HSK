"""
测试 DataLoader 缓存：先构建小数据集保存缓存，再从缓存加载验证。

流程：
1) 使用 prepare_dataloaders(limit_restaurants=500) 构建精简数据集；
2) 保存到自定义缓存目录 `preprocessed_data_cache_small/`；
3) 从缓存加载，并补充 yearly_graphs（含零图兜底）；
4) 打印基本信息与一条样例字段形状。

运行：
  python -m survival.scripts.test_cache
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from survival_st_gcn.data.cache import (
    load_preprocessed_loaders,
    save_preprocessed_data,
)
from survival_st_gcn.data.graphs import load_yearly_graphs
from survival_st_gcn.data.loaders import prepare_dataloaders
from survival_st_gcn.utils.seed import seed_worker
from survival_st_gcn.utils.paths import resolve_data_dir


def _default_loader_kwargs(batch_size: int) -> Dict[str, Any]:
    return {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": True,
        "num_workers": max(1, os.cpu_count() // 2),
        "pin_memory": False,
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }


def _peek_tensor(t: torch.Tensor, *, rows: int = 1, cols: int = 3) -> str:
    try:
        if t.dim() == 0:
            vals = [float(t.item())]
        elif t.dim() == 1:
            vals = t[:cols].detach().cpu().numpy().tolist()
        else:
            vals = t.reshape(t.shape[0], -1)[:rows, :cols].detach().cpu().numpy().tolist()
        return str(vals)
    except Exception:
        return "[]"


def _print_sample(ds_item: Dict[str, torch.Tensor]) -> None:
    for k, v in ds_item.items():
        if isinstance(v, torch.Tensor):
            preview = _peek_tensor(v)
            print(f"- {k}: shape={tuple(v.shape)} dtype={v.dtype} peek={preview}")
        else:
            print(f"- {k}: {type(v)}")


def main() -> None:
    cache_dir = "preprocessed_data_cache_small"

    print("[1/4] 构建精简数据并保存缓存 ...")
    t0 = time.perf_counter()
    loaders = prepare_dataloaders(batch_size=4, limit_restaurants=None)
    kwargs = {split: _default_loader_kwargs(4) for split in ("train", "val", "test")}
    save_preprocessed_data(
        cache_dir,
        train_dataset=loaders["train_loader"].dataset,
        val_dataset=loaders["val_loader"].dataset,
        test_dataset=loaders["test_loader"].dataset,
        loader_kwargs=kwargs,
    )
    build_time = time.perf_counter() - t0
    print(f"  已保存到 {cache_dir} (构建+保存耗时: {build_time:.2f}s)")

    print("[2/4] 从缓存加载 DataLoader ...")
    t1 = time.perf_counter()
    cached = load_preprocessed_loaders(cache_dir)
    train_loader: DataLoader = cached["train_loader"]
    val_loader: DataLoader = cached["val_loader"]
    test_loader: DataLoader = cached["test_loader"]
    load_time = time.perf_counter() - t1
    print("  sizes:")
    print("  - train:", len(train_loader.dataset))
    print("  - val:", len(val_loader.dataset))
    print("  - test:", len(test_loader.dataset))
    print(f"  缓存加载耗时: {load_time:.2f}s")

    # 组装 yearly_graphs（零图兜底）
    try:
        restaurant_df = train_loader.dataset.restaurant_data
        rest_ids_numeric = (
            restaurant_df["restaurant_id"].astype(str).apply(pd.to_numeric, errors="coerce").dropna().astype(int)
        )
        restaurant_ids = rest_ids_numeric.drop_duplicates().tolist()
    except Exception:
        restaurant_ids = []
    graph_dir = resolve_data_dir("graph_data/10_year_graphs")
    yearly_graphs = load_yearly_graphs(graph_dir, restaurant_ids=restaurant_ids)
    print("  yearly_graphs:", len(yearly_graphs))

    print("[3/4] 打印 train/val/test 第一条样例 ...")
    print("- train[0]")
    _print_sample(train_loader.dataset[0])
    print("- val[0]")
    _print_sample(val_loader.dataset[0])
    print("- test[0]")
    _print_sample(test_loader.dataset[0])

    print("[4/4] 小结：")
    print(f"  构建+保存: {build_time:.2f}s, 缓存加载: {load_time:.2f}s, 速度提升≈x{(build_time/max(load_time,1e-6)):.1f}")


if __name__ == "__main__":
    main()
