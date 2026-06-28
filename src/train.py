"""Training entry point. Usage: python -m src.train --config configs/baseline.yaml"""
from __future__ import annotations

import argparse

from .config import load_config


def train(cfg) -> None:
    # TODO:
    #   1. build answer vocab + datasets/dataloaders (cfg.data)
    #   2. build VQAModel(cfg, num_answers) and move to device
    #   3. optimizer (cfg.train.optimizer), scheduler (cosine + warmup), AMP scaler
    #   4. loss: CrossEntropy (hard) or soft-target CE / KLDiv (soft_label)
    #   5. train loop with VQA-accuracy eval each epoch; save best to cfg.output_dir
    #   6. dump metrics.json into cfg.output_dir (committed; weights are git-ignored)
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
