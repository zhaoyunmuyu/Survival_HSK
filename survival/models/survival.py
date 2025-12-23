"""顶层模型：融合图特征、图文评论、区域与宏观信息

子模块：
- ST-GCN：年度图的空间+时间建模；
- TextImageModule：评论图文的多层注意力融合；
- Macro FactorAtt：将宏观特征融入样本嵌入；
- 区域 one-hot 编码 -> 线性映射；
- 最后使用 MLP 分类器输出 logits（使用 BCEWithLogits 或 FocalLoss）。
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival.models.attention import FactorAtt
from survival.models.stgcn import SpatioTemporalAttentionGCN
from survival.models.text_image import TextImageModule


class RestaurantSurvivalModel(nn.Module):
    """Combines graph, text/image, region, and macro features into a classifier."""

    def __init__(self, total_rests: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.st_gcn = SpatioTemporalAttentionGCN(
            input_dim=21,
            hidden_dim=64,
            output_dim=64,
            time_steps=10,
            total_rests=total_rests,
        )
        self.text_image_module = TextImageModule(
            text_input_dim=300,
            hidden_dim=hidden_dim,
            num_heads=4,
            ff_dim=1024,
        )
        fused_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
        )
        self.final_norm1 = nn.LayerNorm(512)
        self.norm = nn.LayerNorm(64)
        self.gcnlin = nn.Linear(64, 64)
        self.gcnlin2 = nn.Linear(64, 512)
        self.rest_id_to_idx: Dict[int, int] | None = None
        self.macro_factor_att = FactorAtt(feature_dim=hidden_dim)
        self.region_encoder = nn.Linear(18, 512)
        self.alpha = nn.Parameter(torch.ones(512))
        self.beta = nn.Parameter(torch.ones(512))
        self.gamma = nn.Parameter(torch.ones(512))
        self.w1 = nn.Parameter(torch.ones(512))
        self.w2 = nn.Parameter(torch.ones(512))
        self.w3 = nn.Parameter(torch.ones(64))
        self.w4 = nn.Parameter(torch.ones(64))
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 512)

    def forward(self, batch: Dict[str, torch.Tensor], yearly_graphs: Sequence, device: torch.device) -> torch.Tensor:
        batch_rest_ids = batch["restaurant_id"]
        text_features = batch["review_text"]
        image_features = batch["review_images"]
        review_features = batch["review_features"]
        macro_features = batch["macro_features"]
        region_encoding = batch["region_encoding"]

        img_text_feat = self.text_image_module(text_features, image_features, review_features)
        # 图分支：按年度图运行 ST-GCN，并取回餐厅ID列表用于索引对齐
        spatiotemporal_features, restaurant_ids = self.st_gcn(yearly_graphs, device)
        spatiotemporal_features = self.norm(
            self.w3 * self.gcnlin(spatiotemporal_features) + self.w4 * spatiotemporal_features
        )
        spatiotemporal_features = self.gcnlin2(spatiotemporal_features)

        if self.rest_id_to_idx is None:
            self.rest_id_to_idx = {rest_id: idx for idx, rest_id in enumerate(restaurant_ids)}

        batch_size = batch_rest_ids.size(0)
        graph_feat = torch.zeros(batch_size, 512, device=device)
        for i in range(batch_size):  # 将 batch 中的餐厅 ID 映射到图特征索引
            rest_id = batch_rest_ids[i].item()
            if rest_id in self.rest_id_to_idx:
                idx = self.rest_id_to_idx[rest_id]
                graph_feat[i] = spatiotemporal_features[idx]

        region_encoding = region_encoding.squeeze().long()
        region_idx = region_encoding - 1
        region_onehot = F.one_hot(region_idx, num_classes=18).float()  # 区域独热编码
        region_feat = self.region_encoder(region_onehot)

        # 融合图/图文/区域特征，再通过宏观因子注意力细化
        fused_features = self.l1(self.alpha * graph_feat + self.beta * img_text_feat + self.gamma * region_feat)
        fused_features1 = self.macro_factor_att(fused_features, macro_features, 0.45879)
        fused_features2 = self.l2(self.w1 * fused_features + self.w2 * fused_features1)
        fused_features2 = self.final_norm1(fused_features2)
        output = self.classifier(fused_features)
        return output
