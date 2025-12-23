"""评论数据预处理与餐厅级缓存构建（蒸馏专用版本）

与旧版差异：
- 在构建缓存时额外保留每条评论的发布年份（`review_years`），用于学生模型筛选“最后两年”；
- 其他字段与旧版保持一致（text/images/features），便于平滑迁移。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from survival.utils.constants import (
    IMAGE_VECTOR_DIM,
    MAX_REVIEW_PHOTOS,
    MAX_REVIEWS_PER_RESTAURANT,
    REVIEW_FEATURE_DIM,
    TEXT_VECTOR_DIM,
)

LOGGER = logging.getLogger(__name__)


REQUIRED_REVIEW_COLUMNS = [
    "restaurant_id",
    "review_id",
    "review_photo_id",
    "review_date",
    "user_location",
    "review_helpful_vote",
    "review_taste",
    "review_environment",
    "review_service",
    "review_hygiene",
    "review_dishi",
]


def prepare_review_dataframe(review_df: pd.DataFrame) -> pd.DataFrame:
    """规范评论表的数据类型并派生辅助列（含 `review_year`）。

    变更点：在旧版基础上补充 `review_year`，缺失记为 0。
    """
    missing_cols = [col for col in REQUIRED_REVIEW_COLUMNS if col not in review_df.columns]
    if missing_cols:
        raise KeyError(f"Missing review dataframe columns: {missing_cols}")

    review_df = review_df.copy()
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
    review_df["review_id"] = review_df["review_id"].astype(str)
    review_df["review_date"] = pd.to_datetime(review_df["review_date"], errors="coerce")

    numeric_cols = [
        "review_helpful_vote",
        "review_taste",
        "review_environment",
        "review_service",
        "review_hygiene",
        "review_dishi",
    ]
    for col in numeric_cols:
        review_df[col] = pd.to_numeric(review_df[col], errors="coerce").fillna(0.0).astype(np.float32)

    # 用户地域编码 → 稳定整数编码（0 表示未知）
    review_df["user_location"] = review_df["user_location"].fillna("")
    codes, _ = pd.factorize(review_df["user_location"], sort=False)
    review_df["user_location_code"] = np.where(codes >= 0, codes + 1, 0).astype(np.int32)

    # 新增：评论年份（缺失→0）
    review_df["review_year"] = review_df["review_date"].dt.year.fillna(0).astype(np.int32)
    return review_df


def build_restaurant_review_cache(
    review_df: pd.DataFrame,
    text_vector_map: Dict[str, np.ndarray],
    *,
    max_reviews: int = MAX_REVIEWS_PER_RESTAURANT,
    max_photos: int = MAX_REVIEW_PHOTOS,
    img_feat_dirs: Iterable[str] = (),
    text_dim: int = TEXT_VECTOR_DIM,
    img_dim: int = IMAGE_VECTOR_DIM,
    feature_dim: int = REVIEW_FEATURE_DIM,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """以餐厅为单位预计算并缓存评论序列（含年份）。

    输出字段：
    - text: [L, text_dim]
    - images: [L, img_dim]
    - features: [L, feature_dim]
    - years: [L]  每条评论的年份，缺失位置为 0
    """
    valid_img_dirs: Tuple[str, ...] = tuple(d for d in img_feat_dirs if d and os.path.isdir(d))
    if img_feat_dirs and not valid_img_dirs:
        LOGGER.warning("None of the provided image feature directories exist; using zero vectors")

    text_zero = np.zeros(text_dim, dtype=np.float32)
    image_zero = np.zeros(img_dim, dtype=np.float32)

    restaurant_reviews: Dict[str, Dict[str, torch.Tensor]] = {}
    image_feature_cache: Dict[str, np.ndarray] = {}
    missing_text_counter = 0
    missing_image_counter = 0

    def load_photo_feature(photo_id: str) -> Optional[np.ndarray]:
        """加载单张图片的特征（支持多目录聚合求均值）。"""
        if photo_id in image_feature_cache:
            return image_feature_cache[photo_id]
        loaded_feats = []
        for directory in valid_img_dirs:
            path = os.path.join(directory, f"{photo_id}.npy")
            try:
                feat = np.load(path, allow_pickle=False)
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)
                if feat.size != img_dim:
                    if feat.size > img_dim:
                        feat = feat[:img_dim]
                    else:
                        padded = np.zeros(img_dim, dtype=np.float32)
                        padded[: feat.size] = feat
                        feat = padded
                loaded_feats.append(feat)
            except FileNotFoundError:
                continue
            except Exception as exc:
                LOGGER.debug("Failed to load image feature %s: %s", path, exc)
        if not loaded_feats:
            return None
        stacked = np.mean(np.stack(loaded_feats, axis=0), axis=0).astype(np.float32, copy=False)
        image_feature_cache[photo_id] = stacked
        return stacked

    grouped = review_df.groupby("restaurant_id", sort=False)
    total_groups = review_df["restaurant_id"].nunique()
    for restaurant_id, group in tqdm(grouped, desc="prepare_reviews_kd", total=total_groups):
        if group.empty:
            continue
        # 限长 + 稳定时序
        if len(group) > max_reviews:
            group = group.sample(n=max_reviews, random_state=42)
        group = group.sort_values("review_date", kind="mergesort")

        text_vectors = np.zeros((max_reviews, text_dim), dtype=np.float32)
        image_vectors = np.zeros((max_reviews, img_dim), dtype=np.float32)
        feature_vectors = np.zeros((max_reviews, feature_dim), dtype=np.float32)
        year_vectors = np.zeros((max_reviews,), dtype=np.int32)

        min_date = group["review_date"].min()
        min_date = None if pd.isna(min_date) else min_date

        for idx, row in enumerate(group.itertuples(index=False)):
            review_id = row.review_id
            text_vec = text_vector_map.get(review_id)
            if text_vec is None:
                missing_text_counter += 1
                text_vec = text_zero
            text_vectors[idx] = text_vec

            img_vec = image_zero
            photo_ids = row.review_photo_id
            if isinstance(photo_ids, str) and photo_ids.strip():
                feats = []
                for pid in filter(None, (pid.strip() for pid in photo_ids.split(";"))):
                    feat = load_photo_feature(pid)
                    if feat is not None:
                        feats.append(feat)
                    if len(feats) >= max_photos:
                        break
                if feats:
                    img_vec = np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32, copy=False)
                else:
                    missing_image_counter += 1
            else:
                missing_image_counter += 1
            image_vectors[idx] = img_vec

            # 相对时间段（季度段）
            review_date = row.review_date
            if min_date is None or pd.isna(review_date):
                date_seg = 0.0
            else:
                months = (review_date.year - min_date.year) * 12 + (review_date.month - min_date.month)
                date_seg = float((months // 3) + 1)

            feat_row = feature_vectors[idx]
            feat_row[0] = date_seg
            feat_row[1] = float(row.user_location_code)
            feat_row[2] = float(row.review_helpful_vote)
            feat_row[3] = float(row.review_taste)
            feat_row[4] = float(row.review_environment)
            feat_row[5] = float(row.review_service)
            feat_row[6] = float(row.review_hygiene)
            feat_row[7] = float(row.review_dishi)

            # 新增：评论年份
            try:
                year_vectors[idx] = int(row.review_year)
            except Exception:
                year_vectors[idx] = 0

        restaurant_reviews[restaurant_id] = {
            "text": torch.from_numpy(text_vectors),
            "images": torch.from_numpy(image_vectors),
            "features": torch.from_numpy(feature_vectors),
            "years": torch.from_numpy(year_vectors.astype(np.int64)),  # 统一为 long，便于后续计算
        }

    LOGGER.info(
        "[KD] Built review cache for %d restaurants (missing text: %d, missing images: %d)",
        len(restaurant_reviews),
        missing_text_counter,
        missing_image_counter,
    )
    return restaurant_reviews

