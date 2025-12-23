"""宏观经济数据预处理

功能：
- 将形如 `{region: {year: values}}` 的嵌套字典解析为张量；
- 统一补零/截断到固定长度，返回每个区域/年份的向量；
- 同时返回一个默认向量（全零），以便缺失时回退。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def prepare_macro_data(
    macro_raw: Dict[str, Dict[str, Any]],
    vector_length: int = 62,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], torch.Tensor]:
    """将原始字典转换为便于快速查询的张量结构。"""
    macro_default = torch.zeros(vector_length, dtype=torch.float32)
    macro_prepared: Dict[str, Dict[int, torch.Tensor]] = {}

    for region, yearly in macro_raw.items():
        region_map: Dict[int, torch.Tensor] = {}
        for year_str, values in yearly.items():
            try:
                year = int(year_str)
            except (TypeError, ValueError):
                continue
            tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
            if tensor.numel() < vector_length:
                padded = macro_default.clone()
                padded[: tensor.numel()] = tensor
                tensor = padded
            elif tensor.numel() > vector_length:
                tensor = tensor[:vector_length]
            region_map[year] = tensor
        if region_map:
            macro_prepared[region] = region_map

    return macro_prepared, macro_default
