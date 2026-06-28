"""Inference / submission entry point.

Usage:
    python -m src.predict --config configs/baseline.yaml --ckpt experiments/baseline/best.pth
"""
from __future__ import annotations

import argparse

from .config import load_config


def predict(cfg, ckpt: str, out: str) -> None:
    # TODO:
    #   1. build test dataloader + model, load `ckpt`
    #   2. run inference, map predicted indices back to answer strings
    #   3. write predictions in the submission format expected by Omnicampus
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="submission/predictions.json")
    args = parser.parse_args()
    predict(load_config(args.config), args.ckpt, args.out)


if __name__ == "__main__":
    main()
