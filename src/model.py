"""The custom VQA model: image encoder + text encoder + fusion + answer classifier.

Pretrained backbones are used only as components here and fine-tuned by the custom
training loop in `train.py` — no off-the-shelf VQA model is used end-to-end.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_image_encoder, build_text_encoder


class ConcatFusion(nn.Module):
    """Naive baseline fusion: concatenate pooled features, then MLP."""

    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        raise NotImplementedError("TODO: concat -> MLP")


class CrossAttentionFusion(nn.Module):
    """Cross-modal attention: question tokens attend over image regions (and vice versa)."""

    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        raise NotImplementedError("TODO: co-attention / cross-attention")


def build_fusion(cfg, image_dim: int, text_dim: int) -> nn.Module:
    if cfg.type == "concat":
        return ConcatFusion(image_dim, text_dim, cfg.hidden_dim)
    if cfg.type == "cross_attention":
        return CrossAttentionFusion(image_dim, text_dim, cfg.hidden_dim)
    raise ValueError(f"unknown fusion type: {cfg.type}")


class VQAModel(nn.Module):
    def __init__(self, cfg, num_answers: int):
        super().__init__()
        self.image_encoder = build_image_encoder(cfg.model.image_encoder)
        self.text_encoder = build_text_encoder(cfg.model.text_encoder)
        self.fusion = build_fusion(
            cfg.model.fusion,
            self.image_encoder.out_dim,
            self.text_encoder.out_dim,
        )
        self.classifier = nn.Linear(cfg.model.fusion.hidden_dim, num_answers)

    def forward(self, image: torch.Tensor, question) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(question)
        fused = self.fusion(img_feat, txt_feat)
        return self.classifier(fused)
