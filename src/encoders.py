"""Image and text encoders.

Each encoder is a swappable component selected by config so that ablations
(scratch vs. pretrained, one-hot vs. BERT) are just config changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_image_encoder(cfg) -> nn.Module:
    """Return an image encoder exposing `.out_dim` and producing token/feature maps.

    type: resnet18 | resnet50 | vit | convnext
    """
    raise NotImplementedError("TODO: build image encoder from cfg.type / cfg.pretrained")


def build_text_encoder(cfg) -> nn.Module:
    """Return a text encoder exposing `.out_dim`.

    type: onehot | gru | bert
    """
    raise NotImplementedError("TODO: build text encoder from cfg.type")
