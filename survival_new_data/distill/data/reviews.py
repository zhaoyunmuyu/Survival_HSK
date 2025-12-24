"""评论数据预处理与餐厅级缓存构建（对接 survival_hsk/data）

约定输入：
- reviews_clean.parquet: 每条评论粒度的清洗表（来源：survival_new_data/preprocess/build_reviews_clean.py）
- review_bert_emb.parquet: 每条评论对应的 BERT 向量（来源：survival_new_data/preprocess/build_review_bert_emb.py）

输出（缓存）字段：
- text: [L, text_dim]
- images: [L, img_dim]（若无图片特征，默认为全零）
- features: [L, feature_dim]
- years: [L]  每条评论的年份，缺失为 0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from survival_new_data.distill.data.io import resolve_data_path

LOGGER = logging.getLogger(__name__)

TimeStep = Literal["year", "half", "quarter"]


REQUIRED_REVIEW_COLUMNS = [
    "restaurant_id",
    "review_id",
    "review_date",
    "user_location",
    "review_helpful_vote",
    "review_taste",
    "review_environment",
    "review_service",
    "review_hygiene",
    "review_dishi",
]


def load_reviews_clean_df(*, data_dir: Optional[Path] = None) -> pd.DataFrame:
    path = resolve_data_path("reviews_clean.parquet", data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"reviews_clean.parquet not found at: {path}")
    return pd.read_parquet(path)


def load_review_bert_emb_df(*, data_dir: Optional[Path] = None, text_dim: int = 768) -> pd.DataFrame:
    """加载 review_bert_emb.parquet。

    说明：
    - 当前实现会将 bert_emb_0..bert_emb_{text_dim-1} 作为列读取；
    - 若你输出文件包含不同维度，请同步传入 text_dim。
    """
    path = resolve_data_path("review_bert_emb.parquet", data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"review_bert_emb.parquet not found at: {path}")

    cols = ["review_id"]
    cols.extend([f"bert_emb_{i}" for i in range(int(text_dim))])
    return pd.read_parquet(path, columns=cols)


def _time_index_from_datetime(dt: pd.Series, *, time_step: TimeStep) -> pd.Series:
    """将 datetime 转为离散时间索引（int），缺失返回 0。

    - year: year
    - half: year*2 + (half-1)  (H1=0, H2=1)
    - quarter: year*4 + (q-1)  (Q1=0..Q4=3)
    """
    if time_step == "year":
        idx = dt.dt.year
    elif time_step == "half":
        idx = dt.dt.year * 2 + ((dt.dt.month - 1) // 6)
    elif time_step == "quarter":
        idx = dt.dt.year * 4 + (dt.dt.quarter - 1)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported time_step: {time_step}")
    return idx.fillna(0).astype(np.int32)


def prepare_reviews_clean(review_df: pd.DataFrame, *, time_step: TimeStep = "year") -> pd.DataFrame:
    """规范评论表的数据类型并派生辅助列（含 review_year）。"""
    missing_cols = [col for col in REQUIRED_REVIEW_COLUMNS if col not in review_df.columns]
    if missing_cols:
        raise KeyError(f"Missing reviews_clean columns: {missing_cols}")

    df = review_df.copy()
    df["restaurant_id"] = df["restaurant_id"].astype(str)
    df["review_id"] = df["review_id"].astype(str)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    numeric_cols = [
        "review_helpful_vote",
        "review_taste",
        "review_environment",
        "review_service",
        "review_hygiene",
        "review_dishi",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

    df["user_location"] = df["user_location"].fillna("")
    codes, _ = pd.factorize(df["user_location"], sort=False)
    df["user_location_code"] = np.where(codes >= 0, codes + 1, 0).astype(np.int32)

    df["review_year"] = df["review_date"].dt.year.fillna(0).astype(np.int32)
    df["review_time_idx"] = _time_index_from_datetime(df["review_date"], time_step=time_step)
    return df


def build_bert_vector_map(bert_df: pd.DataFrame, *, text_dim: int = 768) -> Dict[str, np.ndarray]:
    """将 review_bert_emb DataFrame 转成 {review_id: vector} 映射。"""
    text_dim = int(text_dim)
    expected = ["review_id"] + [f"bert_emb_{i}" for i in range(text_dim)]
    missing = [c for c in expected if c not in bert_df.columns]
    if missing:
        raise KeyError(f"review_bert_emb missing columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    df = bert_df.copy()
    df["review_id"] = df["review_id"].astype(str)
    vectors = df[[f"bert_emb_{i}" for i in range(text_dim)]].to_numpy(dtype=np.float32, copy=False)
    ids = df["review_id"].to_numpy(dtype=object)
    return {str(rid): vectors[i] for i, rid in enumerate(ids)}


def build_restaurant_review_cache(
    reviews_clean_df: pd.DataFrame,
    bert_vector_map: Dict[str, np.ndarray],
    *,
    max_reviews: int = 128,
    text_dim: int = 768,
    img_dim: int = 512,
    feature_dim: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """以餐厅为单位构建评论序列缓存（含年份）。

    当前版本不接入图片特征：images 全零向量，占位以对齐模型输入。
    """
    text_dim = int(text_dim)
    img_dim = int(img_dim)
    feature_dim = int(feature_dim)

    text_zero = np.zeros(text_dim, dtype=np.float32)

    restaurant_reviews: Dict[str, Dict[str, torch.Tensor]] = {}
    missing_text = 0

    grouped = reviews_clean_df.groupby("restaurant_id", sort=False)
    total_groups = reviews_clean_df["restaurant_id"].nunique()

    for restaurant_id, group in grouped:
        if group.empty:
            continue
        if len(group) > max_reviews:
            group = group.sample(n=max_reviews, random_state=42)
        group = group.sort_values("review_date", kind="mergesort")

        text_vectors = np.zeros((max_reviews, text_dim), dtype=np.float32)
        image_vectors = np.zeros((max_reviews, img_dim), dtype=np.float32)
        feature_vectors = np.zeros((max_reviews, feature_dim), dtype=np.float32)
        time_vectors = np.zeros((max_reviews,), dtype=np.int32)

        min_date = group["review_date"].min()
        min_date = None if pd.isna(min_date) else min_date

        for idx, row in enumerate(group.itertuples(index=False)):
            review_id = str(getattr(row, "review_id"))
            vec = bert_vector_map.get(review_id)
            if vec is None:
                missing_text += 1
                vec = text_zero
            text_vectors[idx] = vec

            review_date = getattr(row, "review_date")
            if min_date is None or pd.isna(review_date):
                date_seg = 0.0
            else:
                months = (review_date.year - min_date.year) * 12 + (review_date.month - min_date.month)
                date_seg = float((months // 3) + 1)

            feat_row = feature_vectors[idx]
            feat_row[0] = date_seg
            feat_row[1] = float(getattr(row, "user_location_code", 0))
            feat_row[2] = float(getattr(row, "review_helpful_vote", 0.0))
            feat_row[3] = float(getattr(row, "review_taste", 0.0))
            feat_row[4] = float(getattr(row, "review_environment", 0.0))
            feat_row[5] = float(getattr(row, "review_service", 0.0))
            feat_row[6] = float(getattr(row, "review_hygiene", 0.0))
            feat_row[7] = float(getattr(row, "review_dishi", 0.0))

            try:
                time_vectors[idx] = int(getattr(row, "review_time_idx", 0))
            except Exception:
                time_vectors[idx] = 0

        restaurant_reviews[str(restaurant_id)] = {
            "text": torch.from_numpy(text_vectors),
            "images": torch.from_numpy(image_vectors),
            "features": torch.from_numpy(feature_vectors),
            "years": torch.from_numpy(time_vectors.astype(np.int64)),
        }

    LOGGER.info(
        "[HSK][Distill] Built review cache for %d restaurants (missing bert vec: %d)",
        len(restaurant_reviews),
        missing_text,
    )
    return restaurant_reviews
