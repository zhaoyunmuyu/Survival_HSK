"""时空图卷积（ST-GCN）并对齐年度图特征

流程概述：
- 对每个年份图应用两层 GCN 提取空间特征；
- 将不同年份出现的餐厅按“全局餐厅ID”对齐到统一索引（不存在的补 0）；
- 在时间维度（年序列）上应用多头注意力，建模时间依赖；
- 聚合末年特征并线性映射，用于后续与图文/宏观/区域等融合。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
try:
    from torch_geometric.nn import GCNConv  # type: ignore
    _HAS_TORCH_GEOMETRIC = True
except Exception:
    _HAS_TORCH_GEOMETRIC = False

    class GCNConv(nn.Module):
        """Fallback GCNConv implementation when torch_geometric is unavailable.

        This simplified layer ignores graph structure and applies a linear transform.
        It allows the training pipeline to run for smoke tests without installing
        torch_geometric. Replace with real GCNConv when the dependency is available.
        """

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.lin = nn.Linear(in_channels, out_channels)
            self.bias = nn.Parameter(torch.zeros(out_channels))

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: D401,E501
            return self.lin(x) + self.bias


class SpatioTemporalAttentionGCN(nn.Module):
    """Runs GCN per year and applies temporal multi-head attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        time_steps: int,
        total_rests: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.time_steps = time_steps
        self.total_rests = total_rests
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.w1 = nn.Parameter(torch.ones(10, 64))
        self.w2 = nn.Parameter(torch.ones(10, 64))
        self.w3 = nn.Parameter(torch.ones(64))
        self.w4 = nn.Parameter(torch.ones(64))
        self.restaurant_ids: Optional[List[int]] = None
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self._init_gcn_weights()

    def _init_gcn_weights(self) -> None:
        for gcn in [self.gcn1, self.gcn2]:
            nn.init.kaiming_uniform_(gcn.lin.weight, mode="fan_in", nonlinearity="relu")
            if gcn.bias is not None:
                nn.init.zeros_(gcn.bias)

    def forward(self, yearly_graphs: Sequence, device: torch.device) -> tuple[torch.Tensor, Sequence[int]]:
        aligned_time_embeddings = []
        mask_list = []

        if self.restaurant_ids is None:
            graph = yearly_graphs[0]
            self.restaurant_ids = [graph.local_idx_to_real_id[local_idx] for local_idx in range(graph.x.shape[0])]
            all_rest_ids = set(graph.global_id_to_idx.keys())
            current_rest_ids = set(self.restaurant_ids)
            missing_ids = all_rest_ids - current_rest_ids
            self.restaurant_ids += list(missing_ids)
            self.restaurant_ids.sort(key=lambda x: graph.global_id_to_idx[x])

        for t in range(self.time_steps):
            graph = yearly_graphs[t]
            x_real = graph.x.to(device)
            edge_index_real = graph.edge_index.to(device)
            edge_weight_real = graph.edge_weight.to(device)
            spatial_feat = self.gcn1(x_real, edge_index_real, edge_weight=edge_weight_real)
            spatial_feat = torch.relu(spatial_feat)
            spatial_feat = self.gcn2(spatial_feat, edge_index_real, edge_weight=edge_weight_real)
            spatial_feat = torch.relu(spatial_feat)

            aligned_feat = torch.zeros((self.total_rests, self.hidden_dim), device=spatial_feat.device)
            mask_t = torch.zeros(self.total_rests, dtype=torch.bool, device=spatial_feat.device)
            for local_idx in range(spatial_feat.shape[0]):
                real_id = graph.local_idx_to_real_id[local_idx]
                global_idx = graph.global_id_to_idx[real_id]
                aligned_feat[global_idx] = spatial_feat[local_idx]
                mask_t[global_idx] = True

            aligned_time_embeddings.append(aligned_feat)
            mask_list.append(mask_t)

        time_embeds = torch.stack(aligned_time_embeddings, dim=1)
        mask = torch.stack(mask_list, dim=1)
        key_padding_mask = ~mask
        attn_output, _ = self.attention(
            query=time_embeds,
            key=time_embeds,
            value=time_embeds,
            key_padding_mask=key_padding_mask,
        )
        time_feat = self.norm1(self.w1 * time_embeds + self.w2 * self.dropout1(attn_output))
        final_time_feat = time_feat[:, -1, :]
        output = self.fc(final_time_feat)
        output = self.norm2(self.w3 * final_time_feat + self.w4 * self.dropout2(output))
        return output, self.restaurant_ids
