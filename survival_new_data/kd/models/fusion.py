"""模态融合与 Token 构建

功能：
- 将文本/图片/结构化评论特征对齐到同一维度并融合为 token；
- 依据参考年与评论年份加入“时序偏移嵌入”，显式编码“新近性”；
- 输出 token 序列与 padding 掩码，供教师/学生模型消费。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival_st_gcn.utils.constants import TEXT_VECTOR_DIM, IMAGE_VECTOR_DIM, REVIEW_FEATURE_DIM


class TokenBuilder(nn.Module):
    """将评论三路特征融合为定长 token，并叠加时间嵌入。"""

    def __init__(self, d_model: int = 512, max_year_offset: int = 10) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_year_offset = max_year_offset

        # 三路投影到同一维度
        self.text_proj = nn.Linear(TEXT_VECTOR_DIM, d_model)
        self.img_proj = nn.Linear(IMAGE_VECTOR_DIM, d_model)
        self.struct_proj = nn.Linear(REVIEW_FEATURE_DIM, d_model)

        # 门控融合（逐维 gate）
        self.gate = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

        # 时间偏移嵌入（0..max_year_offset），0 代表“未知/超出范围”
        self.year_emb = nn.Embedding(max_year_offset + 1, d_model)

    def _build_padding_mask(self, text: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """基于文本/图片全零判断 padding 位置（True=padding）。"""
        text_zero = (text.abs().sum(dim=-1) == 0)
        img_zero = (images.abs().sum(dim=-1) == 0)
        return text_zero & img_zero

    def _year_offset(self, review_years: torch.Tensor, reference_year: torch.Tensor) -> torch.Tensor:
        """将年份差转换为 [0..max_year_offset] 的离散索引。

        约定：offset = clamp(ref_year - review_year, 0, max_year_offset)，
        年份缺失（0）统一映射到 0（即无时间偏移信号）。
        """
        ref = reference_year.view(-1, 1).to(review_years.dtype)
        years = review_years.clone()
        years = torch.where(years <= 0, torch.zeros_like(years), years)  # 缺失→0
        raw = ref - years
        raw = torch.clamp(raw, min=0, max=self.max_year_offset)
        return raw.long()

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入 batch 字典，输出 tokens[B,L,d] 与 padding_mask[B,L]。"""
        text = batch["review_text"]  # [B,L,300]
        images = batch["review_images"]  # [B,L,512]
        struct = batch["review_features"]  # [B,L,8]
        years = batch["review_years"]  # [B,L]
        ref_year = batch["reference_year"].view(-1)  # [B]

        text_d = self.text_proj(text)
        img_d = self.img_proj(images)
        struct_d = self.struct_proj(struct)

        gate_inp = torch.cat([text_d, img_d, struct_d], dim=-1)
        gate = torch.sigmoid(self.gate(gate_inp))
        fused = gate * text_d + (1.0 - gate) * img_d + struct_d

        # 时间偏移嵌入（未知/缺失 → 0）
        offset_idx = self._year_offset(years, ref_year)  # [B,L]
        time_embed = self.year_emb(offset_idx)
        tokens = self.norm(fused + time_embed)

        padding_mask = self._build_padding_mask(text, images)
        return tokens, padding_mask
