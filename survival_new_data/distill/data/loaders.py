"""DataLoader 构建（HSK 蒸馏迁移版）

读取：
- survival_hsk/data/restaurant_base.parquet
- survival_hsk/data/reviews_clean.parquet
- survival_hsk/data/review_bert_emb.parquet

说明：
- 当前版本不接入宏观/图片特征（占位为全零），先确保模型能用新数据跑通；
- 如需宏观/图片，可在后续修改点中按文件落地形态补齐。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from survival_st_gcn.utils.seed import build_generator, seed_everything, seed_worker
from survival_new_data.distill.data.datasets import RestaurantDatasetDistill
from survival_new_data.distill.data.reviews import (
    build_bert_vector_map,
    build_img_vector_map,
    build_restaurant_review_cache,
    load_review_bert_emb_df,
    load_review_img_emb_df,
    load_reviews_clean_df,
    prepare_reviews_clean,
)
from survival_new_data.distill.data.io import resolve_data_path

LOGGER = logging.getLogger(__name__)


def _load_restaurant_base_df(*, data_dir: Optional[Path] = None) -> pd.DataFrame:
    path = resolve_data_path("restaurant_base.parquet", data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"restaurant_base.parquet not found at: {path}")
    return pd.read_parquet(path)


def prepare_dataloaders_distill(
    *,
    batch_size: int = 64,
    data_dir: Optional[Path] = None,
    limit_restaurants: int | None = None,
    text_dim: int = 768,
    img_dim: int = 512,
    feature_dim: int = 8,
    time_step: str = "year",
    last_window_years: int = 2,
) -> Dict[str, Any]:
    seed_everything()

    LOGGER.info("[HSK][Distill] Loading restaurant_base ...")
    restaurant_df = _load_restaurant_base_df(data_dir=data_dir)
    restaurant_df["restaurant_id"] = restaurant_df["restaurant_id"].astype(str)
    if limit_restaurants is not None and len(restaurant_df) > limit_restaurants:
        restaurant_df = restaurant_df.head(limit_restaurants).copy()

    label_col = "is_open"
    train_df, temp_df = train_test_split(
        restaurant_df,
        test_size=0.2,
        random_state=42,
        stratify=restaurant_df[label_col] if label_col in restaurant_df.columns else None,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[label_col] if label_col in temp_df.columns else None,
    )

    LOGGER.info("[HSK][Distill] Loading reviews_clean ...")
    review_df = load_reviews_clean_df(data_dir=data_dir)
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
    required_ids = set(train_df["restaurant_id"].unique()) | set(val_df["restaurant_id"].unique()) | set(test_df["restaurant_id"].unique())
    review_df = review_df[review_df["restaurant_id"].isin(required_ids)]
    review_df = prepare_reviews_clean(review_df, time_step=time_step)  # type: ignore[arg-type]

    LOGGER.info("[HSK][Distill] Loading review_bert_emb ...")
    bert_df = load_review_bert_emb_df(data_dir=data_dir, text_dim=text_dim)
    bert_map = build_bert_vector_map(bert_df, text_dim=text_dim)

    img_map = None
    try:
        LOGGER.info("[HSK][Distill] Loading review_img_emb (optional) ...")
        img_df = load_review_img_emb_df(data_dir=data_dir, img_dim=img_dim)
        img_map = build_img_vector_map(img_df, img_dim=img_dim)
        LOGGER.info("[HSK][Distill] Loaded image embeddings: %d", len(img_map))
    except FileNotFoundError:
        LOGGER.info("[HSK][Distill] review_img_emb.parquet not found; using zero image vectors")

    LOGGER.info("[HSK][Distill] Building restaurant review cache ...")
    review_cache = build_restaurant_review_cache(
        review_df,
        bert_map,
        img_map,
        max_reviews=128,
        text_dim=text_dim,
        img_dim=img_dim,
        feature_dim=feature_dim,
    )

    train_dataset = RestaurantDatasetDistill(
        train_df,
        review_cache,
        text_dim=text_dim,
        img_dim=img_dim,
        feature_dim=feature_dim,
        time_step=time_step,  # type: ignore[arg-type]
        last_window_years=last_window_years,
    )
    val_dataset = RestaurantDatasetDistill(
        val_df,
        review_cache,
        text_dim=text_dim,
        img_dim=img_dim,
        feature_dim=feature_dim,
        time_step=time_step,  # type: ignore[arg-type]
        last_window_years=last_window_years,
    )
    test_dataset = RestaurantDatasetDistill(
        test_df,
        review_cache,
        text_dim=text_dim,
        img_dim=img_dim,
        feature_dim=feature_dim,
        time_step=time_step,  # type: ignore[arg-type]
        last_window_years=last_window_years,
    )

    num_workers = max(1, os.cpu_count() // 2)
    LOGGER.info("[HSK][Distill] Creating DataLoaders (batch_size=%d, num_workers=%d)", batch_size, num_workers)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(train_dataset, generator=build_generator(0), **loader_kwargs)
    val_loader = DataLoader(val_dataset, generator=build_generator(1), **{**loader_kwargs, "shuffle": False})
    test_loader = DataLoader(test_dataset, generator=build_generator(2), **{**loader_kwargs, "shuffle": False})

    return {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader}
