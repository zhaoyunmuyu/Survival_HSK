"""预处理结果的磁盘缓存

用途：
- 首次运行会完整预处理并将 Dataset 与 DataLoader 配置持久化；
- 后续运行可直接从磁盘快速加载，大幅缩短启动时间；
- 可按需通过 `use_cache=False` 强制走全量预处理流程。
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pandas as pd

from survival.data.graphs import load_yearly_graphs
from survival.data.loaders import prepare_dataloaders
from survival.utils.seed import seed_worker
from survival.utils.paths import resolve_data_dir


def _default_loader_kwargs(batch_size: int) -> Dict[str, Any]:
    """构造 DataLoader 的通用参数（不包含 generator/shuffle），便于重建。"""
    return {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": True,
        "num_workers": max(1, os.cpu_count() // 2),
        "pin_memory": False,
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }


def save_preprocessed_data(
    save_dir: str,
    *,
    train_dataset,
    val_dataset,
    test_dataset,
    loader_kwargs: Dict[str, Dict[str, Any]],
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_dataset, os.path.join(save_dir, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(save_dir, "val_dataset.pt"))
    torch.save(test_dataset, os.path.join(save_dir, "test_dataset.pt"))
    with open(os.path.join(save_dir, "loader_kwargs.pkl"), "wb") as handle:
        pickle.dump(loader_kwargs, handle)


def load_preprocessed_loaders(save_dir: str) -> Dict[str, DataLoader]:
    """从磁盘载入已保存的数据集并重建 DataLoader（参数来自持久化的 kwargs）。"""
    train_dataset = torch.load(os.path.join(save_dir, "train_dataset.pt"))
    val_dataset = torch.load(os.path.join(save_dir, "val_dataset.pt"))
    test_dataset = torch.load(os.path.join(save_dir, "test_dataset.pt"))
    with open(os.path.join(save_dir, "loader_kwargs.pkl"), "rb") as handle:
        loader_kwargs = pickle.load(handle)

    train_loader = DataLoader(train_dataset, **loader_kwargs["train"])
    val_loader = DataLoader(val_dataset, **loader_kwargs["val"])
    test_loader = DataLoader(test_dataset, **loader_kwargs["test"])
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def get_dataloaders_with_cache(
    batch_size: int,
    *,
    cache_dir: str = "preprocessed_data_cache",
    use_cache: bool = True,
) -> Dict[str, Any]:
    cache_ready = (
        use_cache
        and os.path.exists(cache_dir)
        and {"train_dataset.pt", "val_dataset.pt", "test_dataset.pt", "loader_kwargs.pkl"}.issubset(
            set(os.listdir(cache_dir))
        )
    )

    if cache_ready:
        data_loaders = load_preprocessed_loaders(cache_dir)
        # 从缓存数据集中提取餐厅ID，供图数据兜底使用
        try:
            restaurant_df = data_loaders["train_loader"].dataset.restaurant_data
            rest_ids_numeric = (
                restaurant_df["restaurant_id"].astype(str).apply(pd.to_numeric, errors="coerce").dropna().astype(int)
            )
            restaurant_ids = rest_ids_numeric.drop_duplicates().tolist()
        except Exception:
            restaurant_ids = []
        # 解析图目录路径：优先数据根（或同级 junLi），兼容旧路径
        graph_dir = resolve_data_dir("graph_data/10_year_graphs")
        data_loaders["yearly_graphs"] = load_yearly_graphs(graph_dir, restaurant_ids=restaurant_ids)
        return data_loaders

    data_loaders = prepare_dataloaders(batch_size=batch_size)
    loader_kwargs = {
        split: _default_loader_kwargs(batch_size)
        for split in ("train", "val", "test")
    }
    save_preprocessed_data(
        cache_dir,
        train_dataset=data_loaders["train_loader"].dataset,
        val_dataset=data_loaders["val_loader"].dataset,
        test_dataset=data_loaders["test_loader"].dataset,
        loader_kwargs=loader_kwargs,
    )
    return data_loaders
