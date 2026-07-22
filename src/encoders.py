"""Image and text encoders.

Each encoder is a swappable component selected by config so that ablations
(scratch vs. pretrained, one-hot vs. BERT) are just config changes.

Unified interface: every encoder's ``forward`` returns ``(features, mask)`` where
``features`` is ``(B, T, D)`` (a token / region sequence) and ``mask`` is ``(B, T)`` with
1 for valid positions or ``None`` when every position is valid. This lets both the concat
and the cross-attention fusion consume any encoder. Each encoder also exposes ``.out_dim``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- image

class _TorchvisionBackbone(nn.Module):
    """Wrap a torchvision ResNet, returning its (B, HW, C) feature-map tokens."""

    def __init__(self, name: str, pretrained: bool, freeze: bool):
        super().__init__()
        import torchvision
        from torchvision.models import ResNet18_Weights, ResNet50_Weights

        if name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.resnet18(weights=weights)
            self.out_dim = 512
        elif name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = torchvision.models.resnet50(weights=weights)
            self.out_dim = 2048
        else:
            raise ValueError(f"unsupported torchvision backbone: {name}")

        # Drop the global pool + fc so we keep the spatial feature map.
        self.body = nn.Sequential(*list(net.children())[:-2])
        if freeze:
            for p in self.body.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor):
        feat = self.body(image)            # (B, C, H, W)
        b, c, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        return feat, None


class _TimmBackbone(nn.Module):
    """Wrap a timm model (ViT / ConvNeXt), returning (B, T, D) tokens."""

    def __init__(self, name: str, pretrained: bool, freeze: bool):
        super().__init__()
        import timm

        model_name = {
            "vit": "vit_base_patch16_224",
            "convnext": "convnext_tiny",
        }[name]
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_dim = self.model.num_features
        self.is_vit = name == "vit"
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor):
        feat = self.model.forward_features(image)
        if feat.dim() == 4:                # ConvNeXt: (B, C, H, W)
            feat = feat.flatten(2).transpose(1, 2)
        # ViT already returns (B, T, D)
        return feat, None


def build_image_encoder(cfg) -> nn.Module:
    """type: resnet18 | resnet50 | vit | convnext."""
    t = cfg.type
    pretrained = bool(cfg.get("pretrained", False))
    freeze = bool(cfg.get("freeze", False))
    if t in ("resnet18", "resnet50"):
        return _TorchvisionBackbone(t, pretrained, freeze)
    if t in ("vit", "convnext"):
        return _TimmBackbone(t, pretrained, freeze)
    raise ValueError(f"unknown image encoder type: {t}")


# --------------------------------------------------------------------------- text

class OneHotTextEncoder(nn.Module):
    """Baseline text encoder: project a multi-hot question vector to a dense feature."""

    def __init__(self, vocab_size: int, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(vocab_size, out_dim)
        self.out_dim = out_dim

    def forward(self, question: torch.Tensor):
        # question: (B, vocab_size) multi-hot -> (B, 1, out_dim), no padding
        return self.proj(question).unsqueeze(1), None


class GRUTextEncoder(nn.Module):
    """Word embeddings + (bi)GRU over token ids."""

    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden: int = 512,
                 bidirectional: bool = True, pad_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, question):
        ids, mask = question               # (B, L), (B, L)
        emb = self.embed(ids)
        out, _ = self.gru(emb)             # (B, L, out_dim)
        return out, mask


class BertTextEncoder(nn.Module):
    """Pretrained BERT used as a component; fine-tuned by our training loop."""

    def __init__(self, name: str = "bert-base-uncased", freeze: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(name)
        self.out_dim = self.bert.config.hidden_size
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, question):
        ids, mask = question               # (B, L), (B, L)
        out = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        return out, mask


def build_text_encoder(cfg, onehot_dim: int | None = None,
                       word_vocab_size: int | None = None) -> nn.Module:
    """type: onehot | gru | bert.

    `onehot_dim` (one-hot vocab size) is required for the onehot encoder;
    `word_vocab_size` is required for the GRU encoder.
    """
    t = cfg.type
    if t == "onehot":
        assert onehot_dim is not None, "onehot encoder needs onehot_dim"
        return OneHotTextEncoder(onehot_dim, out_dim=int(cfg.get("out_dim", 512)))
    if t == "gru":
        assert word_vocab_size is not None, "gru encoder needs word_vocab_size"
        return GRUTextEncoder(
            word_vocab_size,
            embed_dim=int(cfg.get("embed_dim", 300)),
            hidden=int(cfg.get("hidden", 512)),
            bidirectional=bool(cfg.get("bidirectional", True)),
        )
    if t == "bert":
        return BertTextEncoder(
            name=str(cfg.get("name", "bert-base-uncased")),
            freeze=bool(cfg.get("freeze", False)),
        )
    raise ValueError(f"unknown text encoder type: {t}")
