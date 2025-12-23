"""图数据加载

功能：
- 依据年份顺序批量读取以 `torch.save` 序列化的图对象；
- 默认目录名使用 ASCII，避免在不同系统编码下出现解析问题；
- 进度条/日志输出也使用 ASCII，防止源文件编码错误。
"""

from __future__ import annotations

import logging
import os
from typing import List, Sequence

import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class _SimpleGraph:
    """极简图对象，满足项目用到的属性接口。

    属性：
    - x: 节点特征张量 [N, F]
    - edge_index: 边索引 [2, E]（此处兜底为 0 条边）
    - edge_weight: 边权重 [E]
    - local_idx_to_real_id: {本地图索引 -> 全局餐厅ID}
    - global_id_to_idx: {全局餐厅ID -> 统一索引}
    - total_rests: N（餐厅总数）
    """

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        local_idx_to_real_id: dict[int, int],
        global_id_to_idx: dict[int, int],
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.local_idx_to_real_id = local_idx_to_real_id
        self.global_id_to_idx = global_id_to_idx
        self.total_rests = x.shape[0]


def _make_zero_graph(restaurant_ids: Sequence[int], feature_dim: int) -> _SimpleGraph:
    n = max(1, len(restaurant_ids))
    x = torch.zeros((n, feature_dim), dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_weight = torch.zeros((0,), dtype=torch.float32)
    if restaurant_ids:
        local_to_real = {i: int(rid) for i, rid in enumerate(restaurant_ids)}
        real_to_global = {int(rid): i for i, rid in enumerate(restaurant_ids)}
    else:
        local_to_real = {i: i for i in range(n)}
        real_to_global = {i: i for i in range(n)}
    return _SimpleGraph(x, edge_index, edge_weight, local_to_real, real_to_global)


def load_yearly_graphs(
    graph_dir: str = "graph_data/10_year_graphs",
    *,
    restaurant_ids: Sequence[int] | None = None,
    feature_dim: int = 21,
) -> List[object]:
    """加载年度图；若缺失对应文件，则用零特征图兜底。

    - graph_dir: 包含 graph_2010.pt ... graph_2019.pt 的目录
    - restaurant_ids: 兜底图使用的餐厅ID列表（用于对齐与索引映射）；
    - feature_dim: 节点特征维度（与模型输入对齐，默认 21）
    返回长度为 10 的列表。
    """
    yearly_graphs: List[object] = []
    years = list(range(2010, 2020))
    for year in tqdm(years, desc="load_graphs"):
        path = os.path.join(graph_dir, f"graph_{year}.pt")
        if os.path.exists(path):
            try:
                yearly_graphs.append(torch.load(path, map_location="cpu"))
                continue
            except Exception as exc:
                LOGGER.warning("Failed to load graph %s: %s; using zero graph fallback", path, exc)
        else:
            LOGGER.warning("Graph %s not found; using zero graph fallback", path)
        yearly_graphs.append(_make_zero_graph(restaurant_ids or [], feature_dim))
    return yearly_graphs
