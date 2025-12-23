"""Dataset（HSK 蒸馏迁移版）：对接 survival_hsk/data。

输出字段与 survival_kd 保持一致：
- restaurant_id: [1] long
- is_open: [1] float
- review_text: [L, text_dim]
- review_images: [L, img_dim]
- review_features: [L, feature_dim]
- review_years: [L]
- last2_mask: [L]
- macro_features: [62]（当前默认全零，占位）
- region_encoding: [1]（优先使用 restaurant_region_num；缺失则为 0）
- reference_year: [1]
"""

from __future__ import annotations

from typing import Any, Dict, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset


class RestaurantDatasetDistill(Dataset):
    def __init__(
        self,
        restaurant_data: pd.DataFrame,
        restaurant_reviews: Dict[str, Dict[str, torch.Tensor]],
        *,
        text_dim: int = 768,
        img_dim: int = 512,
        feature_dim: int = 8,
        time_step: Literal["year", "half", "quarter"] = "year",
        last_window_years: int = 2,
    ) -> None:
        self.restaurant_data = restaurant_data.reset_index(drop=True)
        self.restaurant_reviews = restaurant_reviews
        self.time_step: Literal["year", "half", "quarter"] = time_step
        self.periods_per_year = {"year": 1, "half": 2, "quarter": 4}[self.time_step]
        self.last_window_periods = int(last_window_years) * self.periods_per_year
        self.default_reviews = {
            "text": torch.zeros((128, int(text_dim)), dtype=torch.float32),
            "images": torch.zeros((128, int(img_dim)), dtype=torch.float32),
            "features": torch.zeros((128, int(feature_dim)), dtype=torch.float32),
            "years": torch.zeros((128,), dtype=torch.long),
        }
        self.macro_default = torch.zeros((62,), dtype=torch.float32)

    @staticmethod
    def _safe_long(value: Any) -> torch.Tensor:
        try:
            return torch.tensor([int(value)], dtype=torch.long)
        except (TypeError, ValueError):
            return torch.tensor([-1], dtype=torch.long)

    @staticmethod
    def _safe_region_encoding(row: pd.Series) -> float:
        if "restaurant_region_num" not in row:
            return 0.0
        v = row.get("restaurant_region_num")
        try:
            num = float(v)
        except (TypeError, ValueError):
            return 0.0
        if num <= 0:
            return 0.0
        return float(min(num, 18.0))

    def _reference_time_idx(self, row: pd.Series, years_tensor: torch.Tensor) -> int:
        """计算 reference_time（离散索引），用于时间偏移嵌入与 last-window mask。"""
        op_latest = row.get("operation_latest")
        if pd.notna(op_latest):
            dt = pd.to_datetime(op_latest, errors="coerce")
            if pd.notna(dt):
                year = int(dt.year)
                if self.time_step == "year":
                    return year
                if self.time_step == "half":
                    half = 0 if int(dt.month) <= 6 else 1
                    return year * 2 + half
                quarter = int(((int(dt.month) - 1) // 3))  # 0..3
                return year * 4 + quarter

        op_year = row.get("operation_latest_year")
        if pd.notna(op_year):
            try:
                year = int(op_year)
                if self.time_step == "year":
                    return year
                # 年级别缺失具体月份时，用该年的最后一个 period 作为参考
                return year * self.periods_per_year + (self.periods_per_year - 1)
            except (TypeError, ValueError):
                pass

        if years_tensor.numel() > 0:
            max_t = int(torch.max(years_tensor).item())
            return max_t if max_t > 0 else 0
        return 0

    def __len__(self) -> int:
        return len(self.restaurant_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.restaurant_data.iloc[idx]
        restaurant_id = str(row["restaurant_id"])
        reviews = self.restaurant_reviews.get(restaurant_id, self.default_reviews)

        years_tensor: torch.Tensor = reviews.get("years", self.default_reviews["years"])
        ref_time = self._reference_time_idx(row, years_tensor)

        if years_tensor.numel() > 0 and self.last_window_periods > 0:
            keep_from = ref_time - (self.last_window_periods - 1)
            last2_mask = (years_tensor >= keep_from) & (years_tensor > 0)
        else:
            last2_mask = torch.zeros_like(self.default_reviews["years"], dtype=torch.bool)

        is_open_value = float(row["is_open"]) if pd.notna(row.get("is_open")) else 0.0
        region_encoding = self._safe_region_encoding(row)

        return {
            "restaurant_id": self._safe_long(restaurant_id),
            "is_open": torch.tensor([is_open_value], dtype=torch.float32),
            "review_text": reviews["text"],
            "review_images": reviews["images"],
            "review_features": reviews["features"],
            "review_years": years_tensor.long(),
            "last2_mask": last2_mask.to(torch.bool),
            "macro_features": self.macro_default,
            "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32),
            "reference_year": torch.tensor([int(ref_time)], dtype=torch.long),
        }
