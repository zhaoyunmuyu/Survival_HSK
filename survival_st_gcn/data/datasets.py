"""Dataset 定义：每条样本对应一家餐厅

输出字典字段说明：
- restaurant_id: 餐厅ID（long张量，形如 [1]），用于从图特征中索引；
- is_open: 标签（1=营业，0=关闭），训练中会转换为 1-值；
- review_text: 评论文本序列张量 [128, 300]；
- review_images: 评论图片序列张量 [128, 512]；
- review_features: 评论结构化特征序列 [128, 8]；
- macro_features: 对应区域/年份的宏观特征向量 [62]；
- region_encoding: 区域编码（1..18），模型内会转为 one-hot 后线性映射。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from survival_st_gcn.utils.constants import (
    MAX_REVIEWS_PER_RESTAURANT,
    REGION_MAPPING,
    REVIEW_FEATURE_DIM,
    TEXT_VECTOR_DIM,
    IMAGE_VECTOR_DIM,
)


class RestaurantDataset(Dataset):
    """餐厅级样本：拼装评论序列 + 区域 + 宏观特征。"""

    def __init__(
        self,
        restaurant_data: pd.DataFrame,
        restaurant_reviews: Dict[str, Dict[str, torch.Tensor]],
        macro_data: Dict[str, Dict[int, torch.Tensor]],
        macro_default: torch.Tensor,
    ) -> None:
        self.restaurant_data = restaurant_data.reset_index(drop=True)
        self.restaurant_reviews = restaurant_reviews
        self.macro_data = macro_data
        self.macro_default = macro_default
        self.default_reviews = {
            "text": torch.zeros((MAX_REVIEWS_PER_RESTAURANT, TEXT_VECTOR_DIM), dtype=torch.float32),
            "images": torch.zeros((MAX_REVIEWS_PER_RESTAURANT, IMAGE_VECTOR_DIM), dtype=torch.float32),
            "features": torch.zeros((MAX_REVIEWS_PER_RESTAURANT, REVIEW_FEATURE_DIM), dtype=torch.float32),
        }

    @staticmethod
    def _safe_long(value: Any) -> torch.Tensor:
        try:
            return torch.tensor([int(value)], dtype=torch.long)
        except (TypeError, ValueError):
            return torch.tensor([-1], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.restaurant_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.restaurant_data.iloc[idx]
        restaurant_id = str(row["restaurant_id"])
        reviews = self.restaurant_reviews.get(restaurant_id, self.default_reviews)

        op_year = row.get("operation_latest_year")
        if pd.notna(op_year):
            try:
                year_key = 2019  # Original logic uses fixed 2019 lookback
            except (TypeError, ValueError):
                year_key = None
        else:
            year_key = None

        region_key = row.get("region_code") if isinstance(row.get("region_code"), str) else None
        region_macro = self.macro_data.get(region_key, {}) if region_key else {}
        macro_features = region_macro.get(year_key, self.macro_default)

        region_encoding = REGION_MAPPING.get(region_key, 0)
        is_open_value = float(row["is_open"]) if pd.notna(row["is_open"]) else 0.0

        return {
            "restaurant_id": self._safe_long(restaurant_id),
            "is_open": torch.tensor([is_open_value], dtype=torch.float32),
            "review_text": reviews["text"],
            "review_images": reviews["images"],
            "review_features": reviews["features"],
            "macro_features": macro_features,
            "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32),
        }
