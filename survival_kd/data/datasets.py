"""Dataset（蒸馏版）：在旧版基础上增加年份与“最后两年”掩码。

输出字段：
- restaurant_id: [1] 长整型
- is_open: [1] 浮点标签（训练时再做 1-值/平滑等）
- review_text: [L, 300]
- review_images: [L, 512]
- review_features: [L, 8]
- review_years: [L]   每条评论的年份（0 表示未知/缺失）
- last2_mask: [L]     是否属于“最后两年”的布尔掩码
- macro_features: [62]
- region_encoding: [1] 区域编码（1..18），上游模型再做 one-hot
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


class RestaurantDatasetKD(Dataset):
    """餐厅级样本（蒸馏版）：增加年份与两年掩码。"""

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
            "years": torch.zeros((MAX_REVIEWS_PER_RESTAURANT,), dtype=torch.long),
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

        # 参考年优先使用基础表中的 operation_latest_year；缺失则用该餐厅评论中最大年份
        ref_year = None
        op_year = row.get("operation_latest_year")
        if pd.notna(op_year):
            try:
                ref_year = int(op_year)
            except (TypeError, ValueError):
                ref_year = None
        years_tensor: torch.Tensor = reviews.get("years", self.default_reviews["years"])  # [L]
        if ref_year is None:
            if years_tensor.numel() > 0:
                max_year = int(torch.max(years_tensor).item())
                ref_year = max_year if max_year > 0 else 0
            else:
                ref_year = 0

        # “最后两年”掩码：只使用 <= ref_year 的历史评论，避免未来年份误入导致泄露
        if years_tensor.numel() > 0:
            last2_mask = (years_tensor <= ref_year) & (years_tensor >= (ref_year - 1)) & (years_tensor > 0)
        else:
            last2_mask = torch.zeros_like(self.default_reviews["years"], dtype=torch.bool)

        region_key = row.get("region_code") if isinstance(row.get("region_code"), str) else None
        region_macro = self.macro_data.get(region_key, {}) if region_key else {}
        macro_features = region_macro.get(ref_year, self.macro_default)

        region_encoding = REGION_MAPPING.get(region_key, 0)
        is_open_value = float(row["is_open"]) if pd.notna(row["is_open"]) else 0.0

        return {
            "restaurant_id": self._safe_long(restaurant_id),
            "is_open": torch.tensor([is_open_value], dtype=torch.float32),
            "review_text": reviews["text"],
            "review_images": reviews["images"],
            "review_features": reviews["features"],
            "review_years": years_tensor.long(),
            "last2_mask": last2_mask.to(torch.bool),
            "macro_features": macro_features,
            "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32),
            "reference_year": torch.tensor([int(ref_year)], dtype=torch.long),
        }
