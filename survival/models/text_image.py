"""图文评论特征融合模块

流程概述：
- 文本：BiLSTM -> 线性投影 -> Transformer（自注意力）；
- 图片：线性投影 -> Transformer（自注意力）；
- 交互：多次交叉注意力（图像 Query VS 文本 KV，反向亦然）；
- 融合：FactorAtt 将交互后的序列与评论结构化特征耦合，最后展平映射到 512 维；
- 缺失处理：
  - 同时缺失文本/图片：输出为 0 向量（初始化已覆盖）；
  - 仅缺文本：仅走图片支路 + FactorAtt；
  - 仅缺图片：仅走文本支路 + FactorAtt；
  - 正常（两者均在）：全流程。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival.models.attention import CrossAttentionBlock, FactorAtt, TransformerBlock


class TextImageModule(nn.Module):
    """Pipeline handling text/image sequences with multiple attention hops."""

    def __init__(self, text_input_dim: int = 300, hidden_dim: int = 512, num_heads: int = 8, ff_dim: int = 1024) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=text_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.text_transformer = TransformerBlock(hidden_dim, num_heads, ff_dim)
        self.image_transformer = TransformerBlock(hidden_dim, num_heads, ff_dim)

        self.cross_attn1 = CrossAttentionBlock(hidden_dim, num_heads, ff_dim)
        self.cross_attn2 = CrossAttentionBlock(hidden_dim, num_heads, ff_dim)
        self.cross_attn3 = CrossAttentionBlock(hidden_dim, num_heads, ff_dim)

        self.multi_factor_att = FactorAtt(feature_dim=hidden_dim)
        self.text_factor_att = FactorAtt(feature_dim=hidden_dim)
        self.image_factor_att = FactorAtt(feature_dim=hidden_dim)
        self.lin1 = nn.Linear(hidden_dim * 128, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.w1 = nn.Parameter(torch.ones(hidden_dim))
        self.w2 = nn.Parameter(torch.ones(hidden_dim))
        self.w3 = nn.Parameter(torch.ones(hidden_dim))
        self.w4 = nn.Parameter(torch.ones(hidden_dim))
        self.w5 = nn.Parameter(torch.ones(hidden_dim))
        self.w6 = nn.Parameter(torch.ones(hidden_dim))
        self.w7 = nn.Parameter(torch.ones(hidden_dim))
        self.w8 = nn.Parameter(torch.ones(hidden_dim))
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        review_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = image_features.shape
        device = text_features.device

        # 位置掩码：和原实现一致，按序列维判断是否为“全零填充”位置
        text_mask = (text_features.sum(dim=-1) == 0)
        image_mask = (image_features.sum(dim=-1) == 0)
        text_full_mask = text_mask.all(dim=1)
        image_full_mask = image_mask.all(dim=1)

        both_full_mask = text_full_mask & image_full_mask
        only_text_full_mask = text_full_mask & ~image_full_mask
        only_image_full_mask = ~text_full_mask & image_full_mask
        normal_mask = ~text_full_mask & ~image_full_mask

        # 文本初始处理（BiLSTM + 投影），对“非文本全缺失”的样本生效
        text_feat_init = torch.zeros((batch_size, seq_len, hidden_dim), device=device)
        non_text_full_mask = ~text_full_mask
        if non_text_full_mask.any():
            text_feat_non_full = text_features[non_text_full_mask]
            lstm_out, _ = self.lstm(text_feat_non_full)
            text_feat_init[non_text_full_mask] = self.text_proj(lstm_out)

        # 图像初始处理（线性投影），对“非图片全缺失”的样本生效
        image_feat_init = torch.zeros_like(image_features, device=device)
        non_image_full_mask = ~image_full_mask
        if non_image_full_mask.any():
            image_feat_non_full = image_features[non_image_full_mask]
            image_feat_init[non_image_full_mask] = self.lin2(image_feat_non_full)

        # 输出初始化：默认 0（用于缺失时直接回退）
        fused_img_text = torch.zeros((batch_size, hidden_dim), device=device)

        if normal_mask.any():
            text_feat_norm = text_feat_init[normal_mask]
            image_feat_norm = image_feat_init[normal_mask]
            text_mask_norm = text_mask[normal_mask]
            image_mask_norm = image_mask[normal_mask]
            review_feat_norm = review_features[normal_mask]

            # 自注意力增强（文本/图像各自一套 Transformer）
            text_feat_norm = self.text_transformer(text_feat_norm, mask=text_mask_norm)
            image_feat_norm = self.image_transformer(image_feat_norm, mask=image_mask_norm)

            image_depth_features = self.cross_attn1(
                q=image_feat_norm,
                k=text_feat_norm,
                v=text_feat_norm,
                key_padding_mask=text_mask_norm,
            )
            image_interactive = self.cross_attn2(
                q=image_depth_features,
                k=text_feat_norm,
                v=text_feat_norm,
                key_padding_mask=text_mask_norm,
            )
            text_interactive = self.cross_attn3(
                q=text_feat_norm,
                k=image_depth_features,
                v=image_depth_features,
                key_padding_mask=image_mask_norm,
            )

            # 交互特征加权融合后，逐样本应用 FactorAtt 与评论结构化特征进行耦合
            fuse_inter_fea = self.l1(self.w1 * text_interactive + self.w2 * image_interactive)
            normal_factor_outputs = [
                self.multi_factor_att(fuse_inter_fea[i], review_feat_norm[i], 1)
                for i in range(fuse_inter_fea.shape[0])
            ]
            normal_factor_output = torch.stack(normal_factor_outputs, dim=0)
            normal_factor_output = self.norm1(self.w3 * normal_factor_output + self.w4 * fuse_inter_fea)
            normal_final = self.lin1(normal_factor_output.flatten(start_dim=1))
            fused_img_text[normal_mask] = normal_final

        if only_text_full_mask.any():
            image_feat_otf = image_feat_init[only_text_full_mask]
            image_mask_otf = image_mask[only_text_full_mask]
            review_feat_otf = review_features[only_text_full_mask]
            image_feat_otf_trans = self.image_transformer(image_feat_otf, mask=image_mask_otf)
            otf_factor_outputs = []
            for i in range(image_feat_otf_trans.shape[0]):
                fia = self.image_factor_att(image_feat_otf_trans[i], review_feat_otf[i], 1)
                otf_factor_outputs.append(fia)
            otf_factor_output = torch.stack(otf_factor_outputs, dim=0)
            otf_factor_output = self.norm2(self.w5 * otf_factor_output + self.w6 * image_feat_otf_trans)
            otf_final = self.lin1(otf_factor_output.flatten(start_dim=1))
            fused_img_text[only_text_full_mask] = otf_final

        if only_image_full_mask.any():
            text_feat_oif = text_feat_init[only_image_full_mask]
            text_mask_oif = text_mask[only_image_full_mask]
            review_feat_oif = review_features[only_image_full_mask]
            text_feat_oif_trans = self.text_transformer(text_feat_oif, mask=text_mask_oif)
            oif_factor_outputs = []
            for i in range(text_feat_oif_trans.shape[0]):
                fia = self.text_factor_att(text_feat_oif_trans[i], review_feat_oif[i], 1)
                oif_factor_outputs.append(fia)
            oif_factor_output = torch.stack(oif_factor_outputs, dim=0)
            oif_factor_output = self.norm3(self.w7 * oif_factor_output + self.w8 * text_feat_oif_trans)
            oif_final = self.lin1(oif_factor_output.flatten(start_dim=1))
            fused_img_text[only_image_full_mask] = oif_final

        fused_img_text = self.final_norm(fused_img_text)
        return fused_img_text
