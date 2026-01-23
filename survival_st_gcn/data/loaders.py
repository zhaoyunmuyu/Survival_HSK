"""DataLoader 构建入口

职责：
- 读取 parquet/json 数据；
- 按标签分层划分 train/val/test；训练集对正例做下采样（控制类间比例）；
- 基于评论/图片/文本向量预构建餐厅级缓存张量；
- 创建 `RestaurantDataset` 与 `DataLoader`；
- 同时返回年度图对象列表，供模型图分支使用。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pyarrow.parquet as pq

from survival_st_gcn.data.datasets import RestaurantDataset
from survival_st_gcn.data.graphs import load_yearly_graphs
from survival_st_gcn.data.macro import prepare_macro_data
from survival_st_gcn.data.reviews import build_restaurant_review_cache, prepare_review_dataframe
from survival_st_gcn.data.text_vectors import build_text_vector_map
from survival_st_gcn.utils.seed import build_generator, seed_everything, seed_worker
from survival_st_gcn.utils.paths import resolve_data_file, resolve_data_dir

LOGGER = logging.getLogger(__name__)


def _resolve_data_file(filename: str) -> str:
    """解析数据文件路径，优先环境/同级 junLi，其次兼容旧路径。"""
    path = Path(resolve_data_file(filename))
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Data file not found: {filename} (checked data root and fallbacks)")


def _resolve_dir(dirname: str) -> str:
    """解析目录路径，优先数据根（或同级 junLi），再兼容旧路径。"""
    return resolve_data_dir(dirname)


def _read_parquet_df(path: str) -> pd.DataFrame:
    """使用 PyArrow 稳定读取 parquet 并转为 pandas.DataFrame。

    说明：直接用 pandas.read_parquet 可能受本地 pyarrow/fastparquet 版本影响；
    统一采用 pyarrow 读取以提升兼容性。
    """
    table = pq.read_table(path)
    return table.to_pandas()


def prepare_dataloaders(batch_size: int = 32, limit_restaurants: int | None = None) -> Dict[str, Any]:
    """Create train/val/test dataloaders along with yearly graphs.

    读取路径优先从 data/raw，其次兼容工作目录。
    """
    seed_everything()
    LOGGER.info("Loading restaurant data ...")
    restaurant_df = _read_parquet_df(_resolve_data_file("restaurant_data.parquet"))
    restaurant_df["restaurant_id"] = restaurant_df["restaurant_id"].astype(str)
    if limit_restaurants is not None and len(restaurant_df) > limit_restaurants:
        restaurant_df = restaurant_df.head(limit_restaurants).copy()
    label_col = "is_open"

    # 分层划分：保留总体分布，同时固定随机种子
    train_df, temp_df = train_test_split(
        restaurant_df,
        test_size=0.2,
        random_state=42,
        stratify=restaurant_df[label_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[label_col],
    )

    train_pos = train_df[train_df[label_col] == 1]
    train_neg = train_df[train_df[label_col] == 0]
    num_pos = len(train_pos)
    num_neg = len(train_neg)
    LOGGER.info(
        "Train class balance before downsampling - pos: %d, neg: %d (ratio %.2f:1)",
        num_pos,
        num_neg,
        num_pos / max(num_neg, 1),
    )

    # 控制正例数量上限（1:1）
    max_pos_allowed = 1 * num_neg
    if num_pos > max_pos_allowed:
        train_pos = train_pos.sample(n=max_pos_allowed, random_state=42)
        train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=42)
        LOGGER.info(
            "Train class balance after downsampling - pos: %d, neg: %d (ratio %.2f:1)",
            len(train_pos),
            num_neg,
            len(train_pos) / max(num_neg, 1),
        )
    else:
        LOGGER.info("Downsampling skipped; already within target ratio")

    LOGGER.info("Loading review data ...")
    review_df = _read_parquet_df(_resolve_data_file("review_data.parquet"))
    train_ids = set(train_df["restaurant_id"].unique())
    val_ids = set(val_df["restaurant_id"].unique())
    test_ids = set(test_df["restaurant_id"].unique())
    required_ids = train_ids | val_ids | test_ids
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
    review_df = review_df[review_df["restaurant_id"].isin(required_ids)]
    review_df = prepare_review_dataframe(review_df)

    # Rule: exclude restaurants with <3 distinct review years (train/val/test all apply).
    year_counts = (
        pd.DataFrame({"restaurant_id": review_df["restaurant_id"].astype(str), "review_year": review_df["review_year"].astype(int)})
        .query("review_year > 0")
        .groupby("restaurant_id", sort=False)["review_year"]
        .nunique()
    )
    valid_restaurant_ids = set(year_counts[year_counts >= 3].index.astype(str))
    before = (len(train_df), len(val_df), len(test_df))
    train_df = train_df[train_df["restaurant_id"].isin(valid_restaurant_ids)].copy()
    val_df = val_df[val_df["restaurant_id"].isin(valid_restaurant_ids)].copy()
    test_df = test_df[test_df["restaurant_id"].isin(valid_restaurant_ids)].copy()
    review_df = review_df[review_df["restaurant_id"].isin(valid_restaurant_ids)].copy()
    after = (len(train_df), len(val_df), len(test_df))
    LOGGER.info(
        "Filter restaurants by review years >=3: train %d->%d, val %d->%d, test %d->%d",
        before[0],
        after[0],
        before[1],
        after[1],
        before[2],
        after[2],
    )

    LOGGER.info("Loading macro-economic data ...")
    with open(_resolve_data_file("normalized_macro_data.json"), "r", encoding="utf-8") as handle:
        macro_raw = json.load(handle)
    macro_data, macro_default = prepare_macro_data(macro_raw)

    LOGGER.info("Loading text vectors ...")
    text_feat_df = _read_parquet_df(_resolve_data_file("text_vectors.parquet"))
    # 兼容列名差异：有些数据文件使用 'element' 存放文本向量
    if "text_vector" not in text_feat_df.columns and "element" in text_feat_df.columns:
        text_feat_df = text_feat_df.rename(columns={"element": "text_vector"})
    text_vector_map = build_text_vector_map(text_feat_df)
    del text_feat_df

    LOGGER.info("Building restaurant review cache ...")
    img_dirs = tuple(
        _resolve_dir(d)
        for d in (
            "precomputed_img_features",
            "precomputed_img_features1",
            "precomputed_img_features2",
        )
    )
    restaurant_reviews_cache = build_restaurant_review_cache(
        review_df,
        text_vector_map,
        img_feat_dirs=img_dirs,
    )
    del review_df

    train_dataset = RestaurantDataset(train_df, restaurant_reviews_cache, macro_data, macro_default)
    val_dataset = RestaurantDataset(val_df, restaurant_reviews_cache, macro_data, macro_default)
    test_dataset = RestaurantDataset(test_df, restaurant_reviews_cache, macro_data, macro_default)

    num_workers = max(1, os.cpu_count() // 2)
    LOGGER.info("Creating DataLoaders (batch_size=%d, num_workers=%d)", batch_size, num_workers)

    # DataLoader 通用参数（shuffle=False 保持确定性，如需随机可在此开启）
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(
        train_dataset,
        generator=build_generator(0),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        generator=build_generator(1),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        generator=build_generator(2),
        **loader_kwargs,
    )

    # 解析图数据目录（默认位于仓库根目录下）。若缺失，则用零图兜底。
    # 从餐厅表推导兜底所需餐厅ID列表（需为 int）。
    rest_ids_numeric = pd.to_numeric(restaurant_df["restaurant_id"], errors="coerce").dropna().astype(int)
    restaurant_ids = rest_ids_numeric.drop_duplicates().tolist()
    yearly_graphs = load_yearly_graphs(
        _resolve_dir("graph_data/10_year_graphs"),
        restaurant_ids=restaurant_ids,
        feature_dim=21,
    )

    LOGGER.info("Data preparation complete")
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "yearly_graphs": yearly_graphs,
    }
