"""文本向量解析与映射

功能：
- 兼容多种存储格式（list/tuple/str[JSON]/bytes/ndarray），
  将其安全解析为指定维度的 float32 向量；
- 构建 `review_id -> 向量` 的快速查表字典，加速后续 Dataset 访问；

鲁棒性：
- 长度过长则截断、过短则补零；
- 无法解析的数据会被跳过并计数，方便定位源数据问题。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from survival_st_gcn.utils.constants import TEXT_VECTOR_DIM

LOGGER = logging.getLogger(__name__)


def _coerce_vector(value: Any, expected_dim: int = TEXT_VECTOR_DIM) -> Optional[np.ndarray]:
    """将多种原始值安全转换为固定维度的 float32 向量。"""
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        try:
            arr = np.frombuffer(value, dtype=np.float32)
        except Exception:
            return None
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
            arr = np.asarray(parsed, dtype=np.float32).reshape(-1)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    else:
        return None

    if arr.size == expected_dim:
        return arr.astype(np.float32, copy=False)

    if arr.size > expected_dim:
        return arr[:expected_dim].astype(np.float32, copy=False)

    padded = np.zeros(expected_dim, dtype=np.float32)
    padded[: arr.size] = arr
    return padded


def build_text_vector_map(
    text_feat_df: pd.DataFrame,
    key_column: str = "review_id",
    vector_column: str = "text_vector",
    expected_dim: int = TEXT_VECTOR_DIM,
) -> Dict[str, np.ndarray]:
    """构建 `review_id -> 文本向量` 的映射，用于 Dataset 快速查询。"""
    required_cols = {key_column, vector_column}
    if not required_cols.issubset(text_feat_df.columns):
        missing = required_cols - set(text_feat_df.columns)
        raise KeyError(f"Missing columns in text feature dataframe: {missing}")

    text_map: Dict[str, np.ndarray] = {}
    invalid_count = 0
    for row in text_feat_df[[key_column, vector_column]].itertuples(index=False):
        review_id = str(getattr(row, key_column))
        if review_id in text_map:
            continue
        vec = _coerce_vector(getattr(row, vector_column), expected_dim)
        if vec is None:
            invalid_count += 1
            continue
        text_map[review_id] = vec

    LOGGER.info(
        "Built text vector map with %d valid entries (%d invalid rows discarded)",
        len(text_map),
        invalid_count,
    )
    return text_map
