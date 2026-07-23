"""Ensemble inference: average the softmax of several trained models into one submission.

Usage:
    python -m src.ensemble \
        --configs configs/vit_bert_attn.yaml configs/r50_bert_attn.yaml \
        --ckpts   vit_model.pt r50_model.pt \
        --out submission/submission.npy

No retraining: it reuses already-trained checkpoints (either a bare state_dict `model.pt`, or a
full `best.pt` with a `model_state` key). All members must share the same answer vocabulary
(built deterministically from the training split) and the same question encoding (all BERT here).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import PAD, UNK, VizWizVQA, build_image_transform, norm_stats_for
from .model import VQAModel
from .train import build_tokenizer, pick_device, to_device


def _load_state(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    return obj


def ensemble_predict(config_paths, ckpt_paths, out: str) -> None:
    assert len(config_paths) == len(ckpt_paths), "one ckpt per config"
    device = pick_device()

    # Answer vocabulary is deterministic from the training split -> shared across members.
    ref = load_config(config_paths[0])
    tokenizer = build_tokenizer(ref)
    train_ds = VizWizVQA(
        root=ref.data.root, split="train", transform=None, answer=True, text_mode="tokens",
        answer_vocab_size=ref.data.get("answer_vocab_size"), tokenizer=tokenizer,
        max_qlen=int(ref.data.get("max_qlen", 32)),
    )
    idx2answer, num_answers = train_ds.idx2answer, train_ds.num_answers
    banned = [i for i, a in idx2answer.items() if a in (UNK, PAD)]
    max_qlen = int(ref.data.get("max_qlen", 32))
    batch_size = int(ref.train.get("batch_size", 32))

    # Build each model with its own test loader (image normalization can differ per backbone).
    models, loaders = [], []
    for cfg_path, ckpt_path in zip(config_paths, ckpt_paths):
        cfg = load_config(cfg_path)
        mean, std = norm_stats_for(cfg.model.image_encoder.type)
        tf = build_image_transform(cfg.data.image_size,
                                   bool(cfg.model.image_encoder.get("pretrained", False)),
                                   train=False, mean=mean, std=std)
        test = VizWizVQA(root=cfg.data.root, split="valid", transform=tf, answer=False,
                         text_mode="tokens", tokenizer=tokenizer, max_qlen=max_qlen)
        model = VQAModel(cfg, num_answers=num_answers).to(device)
        model.load_state_dict(_load_state(ckpt_path, device))
        model.eval()
        models.append(model)
        loaders.append(DataLoader(test, batch_size=batch_size, shuffle=False,
                                  num_workers=int(cfg.data.get("num_workers", 2))))

    preds: list[str] = []
    with torch.no_grad():
        for batches in zip(*loaders):  # aligned: every loader is shuffle=False over valid.json
            total = None
            for model, (image, question) in zip(models, batches):
                image, question = image.to(device), to_device(question, device)
                probs = torch.softmax(model(image, question), dim=1)
                total = probs if total is None else total + probs
            if banned:
                total[:, banned] = -1.0  # never emit placeholder classes
            preds.extend(idx2answer[i] for i in total.argmax(1).cpu().tolist())

    out_dir = os.path.dirname(out) or "."
    os.makedirs(out_dir, exist_ok=True)
    submission = np.array(preds)
    np.save(out, submission)
    print(f"wrote {out}  shape={submission.shape} dtype={submission.dtype} "
          f"(ensemble of {len(models)} models)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--out", default="submission/submission.npy")
    args = parser.parse_args()
    ensemble_predict(args.configs, args.ckpts, args.out)


if __name__ == "__main__":
    main()
