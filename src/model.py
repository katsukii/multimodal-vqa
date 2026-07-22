"""The custom VQA model: image encoder + text encoder + fusion + answer classifier.

Pretrained backbones are used only as components here and fine-tuned by the custom
training loop in `train.py` — no off-the-shelf VQA model is used end-to-end.

Encoders return ``(features, mask)`` with ``features`` shaped ``(B, T, D)``. Fusion
modules consume the two ``(features, mask)`` pairs and return a fused ``(B, hidden_dim)``
vector, which the classifier maps to answer logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_image_encoder, build_text_encoder


def masked_mean(feat: torch.Tensor, mask) -> torch.Tensor:
    """Mean-pool ``(B, T, D)`` over T, honouring a ``(B, T)`` validity mask if given."""
    if mask is None:
        return feat.mean(dim=1)
    m = mask.unsqueeze(-1).to(feat.dtype)          # (B, T, 1)
    summed = (feat * m).sum(dim=1)
    count = m.sum(dim=1).clamp_min(1e-6)
    return summed / count


class ConcatFusion(nn.Module):
    """Baseline fusion: mean-pool each modality, concatenate, then an MLP."""

    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image, text):
        img_feat, img_mask = image
        txt_feat, txt_mask = text
        x = torch.cat([masked_mean(img_feat, img_mask), masked_mean(txt_feat, txt_mask)], dim=1)
        return self.mlp(x)


class CrossAttentionFusion(nn.Module):
    """Self-designed cross-modal attention: question tokens attend over image regions.

    Both modalities are projected to a shared width; the question queries the image with
    multi-head attention (a residual keeps the raw question signal), and the attended
    question context is combined with a pooled image summary. This is the report centerpiece
    and the ablation counterpart to `ConcatFusion`.
    """

    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.img_proj = nn.Linear(image_dim, hidden_dim)
        self.txt_proj = nn.Linear(text_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, image, text):
        img_feat, img_mask = image
        txt_feat, txt_mask = text
        img_p = self.img_proj(img_feat)                    # (B, Ti, H)
        txt_p = self.txt_proj(txt_feat)                    # (B, Tq, H)

        key_padding = None if img_mask is None else (img_mask == 0)
        attended, _ = self.attn(txt_p, img_p, img_p, key_padding_mask=key_padding)
        txt_ctx = self.norm(txt_p + attended)              # residual over the question tokens

        txt_vec = masked_mean(txt_ctx, txt_mask)           # (B, H)
        img_vec = masked_mean(img_p, img_mask)             # (B, H)
        return self.out(torch.cat([txt_vec, img_vec], dim=1))


def build_fusion(cfg, image_dim: int, text_dim: int) -> nn.Module:
    if cfg.type == "concat":
        return ConcatFusion(image_dim, text_dim, cfg.hidden_dim)
    if cfg.type == "cross_attention":
        return CrossAttentionFusion(image_dim, text_dim, cfg.hidden_dim)
    raise ValueError(f"unknown fusion type: {cfg.type}")


class VQAModel(nn.Module):
    def __init__(self, cfg, num_answers: int, onehot_dim: int | None = None,
                 word_vocab_size: int | None = None):
        super().__init__()
        self.image_encoder = build_image_encoder(cfg.model.image_encoder)
        self.text_encoder = build_text_encoder(
            cfg.model.text_encoder, onehot_dim=onehot_dim, word_vocab_size=word_vocab_size,
        )
        self.fusion = build_fusion(
            cfg.model.fusion,
            self.image_encoder.out_dim,
            self.text_encoder.out_dim,
        )
        self.classifier = nn.Linear(cfg.model.fusion.hidden_dim, num_answers)

    def forward(self, image: torch.Tensor, question) -> torch.Tensor:
        img = self.image_encoder(image)     # (feat, mask)
        txt = self.text_encoder(question)   # (feat, mask)
        fused = self.fusion(img, txt)
        return self.classifier(fused)
