"""DataLoader 构建（蒸馏版，含图像特征）

职责：
- 读取 parquet/json 数据并构建 train/val/test；
- 训练集对正例下采样以控制比例（pos:neg<=1:1）；
- 预构建评论缓存（含年份），并创建 Dataset/DataLoader；
- 不加载图结构数据，减小 IO 与内存占用，但会使用预计算图片特征。"""

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

from survival_st_gcn.utils.seed import build_generator, seed_everything, seed_worker
from survival_kd.data.reviews import build_restaurant_review_cache, prepare_review_dataframe
from survival_kd.data.datasets import RestaurantDatasetKD
from survival_st_gcn.data.text_vectors import build_text_vector_map
from survival_st_gcn.data.macro import prepare_macro_data
from survival_st_gcn.utils.paths import resolve_data_file, resolve_data_dir

LOGGER = logging.getLogger(__name__)


def _resolve_data_file(filename: str) -> str:
    """解析数据文件路径（支持环境变量 / 同级 junLi）"""
    p = Path(resolve_data_file(filename))
    if p.exists():
        return str(p)
    raise FileNotFoundError(f"Data file not found: {filename}")


def _resolve_dir(dirname: str) -> str:
    """解析目录路径，复用主工程的数据根逻辑。"""
    return resolve_data_dir(dirname)


def _read_parquet_df(path: str) -> pd.DataFrame:
    """使用 PyArrow 稳定读取 parquet 并转换为 pandas.DataFrame。"""
    table = pq.read_table(path)
    return table.to_pandas()


def prepare_dataloaders_kd(
    batch_size: int = 64,
    limit_restaurants: int | None = None,
    *,
    downsample_train_open: bool = True,
    reference_year: int | None = None,
    use_macro_features: bool = True,
) -> Dict[str, Any]:
    """创建蒸馏版 train/val/test DataLoader（不带图结构，但包含图片特征）。

    注意：
    - 训练默认会对 `is_open==1`（仍营业）做下采样以控制类别比例；
    - 若你需要让输出 sigmoid 更接近“真实人群先验概率”（例如训练标定层），可将
      `downsample_train_open=False`。
    """
    seed_everything()
    LOGGER.info("[KD] Loading restaurant data ...")
    restaurant_df = _read_parquet_df(_resolve_data_file("restaurant_data.parquet"))
    restaurant_df["restaurant_id"] = restaurant_df["restaurant_id"].astype(str)
    if limit_restaurants is not None and len(restaurant_df) > limit_restaurants:
        restaurant_df = restaurant_df.head(limit_restaurants).copy()

    label_col = "is_open"
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

    if downsample_train_open:
        # 下采样控制 pos:neg<=1:1（这里 pos 指 is_open==1，即“仍营业”）
        train_pos = train_df[train_df[label_col] == 1]
        train_neg = train_df[train_df[label_col] == 0]
        max_pos_allowed = 1 * len(train_neg)
        if len(train_pos) > max_pos_allowed:
            train_pos = train_pos.sample(n=max_pos_allowed, random_state=42)
            train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=42)

    LOGGER.info("[KD] Loading review data ...")
    review_df = _read_parquet_df(_resolve_data_file("review_data.parquet"))
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
    train_ids = set(train_df["restaurant_id"].unique())
    val_ids = set(val_df["restaurant_id"].unique())
    test_ids = set(test_df["restaurant_id"].unique())
    required_ids = train_ids | val_ids | test_ids
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
        "[KD] Filter restaurants by review years >=3: train %d->%d, val %d->%d, test %d->%d",
        before[0],
        after[0],
        before[1],
        after[1],
        before[2],
        after[2],
    )

    if use_macro_features:
        LOGGER.info("[KD] Loading macro-economic data ...")
        with open(_resolve_data_file("normalized_macro_data.json"), "r", encoding="utf-8") as handle:
            macro_raw = json.load(handle)
        macro_data, macro_default = prepare_macro_data(macro_raw)
    else:
        LOGGER.info("[KD] Macro features disabled; using zero vectors")
        import torch

        macro_data = {}
        macro_default = torch.zeros(62, dtype=torch.float32)

    LOGGER.info("[KD] Loading text vectors ...")
    text_feat_df = _read_parquet_df(_resolve_data_file("text_vectors.parquet"))
    if "text_vector" not in text_feat_df.columns and "element" in text_feat_df.columns:
        text_feat_df = text_feat_df.rename(columns={"element": "text_vector"})
    text_vector_map = build_text_vector_map(text_feat_df)
    del text_feat_df

    LOGGER.info("[KD] Building restaurant review cache (with years & images) ...")
    # 对齐主工程：接入预计算图片特征目录，如缺失则自动回退为零向量
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

    train_dataset = RestaurantDatasetKD(
        train_df,
        restaurant_reviews_cache,
        macro_data,
        macro_default,
        reference_year=reference_year,
        use_macro_features=use_macro_features,
    )
    val_dataset = RestaurantDatasetKD(
        val_df,
        restaurant_reviews_cache,
        macro_data,
        macro_default,
        reference_year=reference_year,
        use_macro_features=use_macro_features,
    )
    test_dataset = RestaurantDatasetKD(
        test_df,
        restaurant_reviews_cache,
        macro_data,
        macro_default,
        reference_year=reference_year,
        use_macro_features=use_macro_features,
    )

    num_workers = max(1, os.cpu_count() // 2)
    LOGGER.info("[KD] Creating DataLoaders (batch_size=%d, num_workers=%d)", batch_size, num_workers)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,  # 训练更合适；验证/测试将覆盖为 False
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
        **{**loader_kwargs, "shuffle": False},
    )
    test_loader = DataLoader(
        test_dataset,
        generator=build_generator(2),
        **{**loader_kwargs, "shuffle": False},
    )

    LOGGER.info("[KD] Data preparation complete")
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
