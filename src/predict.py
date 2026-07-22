"""Inference / submission entry point.

Usage:
    python -m src.predict --config configs/baseline.yaml --ckpt experiments/baseline/best.pt

Writes `submission.npy` (a numpy array of answer *strings*, one per test sample, in
`valid.json` order) plus a `model.pt` weights file — the two artifacts the Omnicampus zip
needs alongside the notebook.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import VizWizVQA, build_image_transform
from .model import VQAModel
from .train import build_tokenizer, pick_device, to_device


def predict(cfg, ckpt_path: str, out: str) -> None:
    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    idx2answer = ckpt["idx2answer"]
    question2idx = ckpt["question2idx"]
    word2idx = ckpt["word2idx"]

    text_mode = "onehot" if cfg.model.text_encoder.type == "onehot" else "tokens"
    pretrained = bool(cfg.model.image_encoder.get("pretrained", False))
    tokenizer = build_tokenizer(cfg)
    tf = build_image_transform(cfg.data.image_size, pretrained, train=False)

    # Test split ("valid.json"); overwrite its vocab with the training vocab from the ckpt.
    test = VizWizVQA(
        root=cfg.data.root, split="valid", transform=tf, answer=False,
        text_mode=text_mode, tokenizer=tokenizer, max_qlen=int(cfg.data.get("max_qlen", 32)),
    )
    test.question2idx = question2idx
    test.word2idx = word2idx
    test.answer2idx = {v: k for k, v in idx2answer.items()}
    test.idx2answer = idx2answer

    model = VQAModel(
        cfg, num_answers=len(idx2answer),
        onehot_dim=len(question2idx) + 1 if text_mode == "onehot" else None,
        word_vocab_size=len(word2idx) if cfg.model.text_encoder.type == "gru" else None,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(test, batch_size=int(cfg.train.get("batch_size", 64)), shuffle=False,
                        num_workers=int(cfg.data.get("num_workers", 2)))

    preds: list[str] = []
    with torch.no_grad():
        for image, question in loader:
            image, question = image.to(device), to_device(question, device)
            logits = model(image, question)
            for idx in logits.argmax(1).cpu().tolist():
                preds.append(idx2answer[idx])

    out_dir = os.path.dirname(out) or "."
    os.makedirs(out_dir, exist_ok=True)
    submission = np.array(preds)
    np.save(out, submission)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print(f"wrote {out}  shape={submission.shape} dtype={submission.dtype}")
    print(f"wrote {os.path.join(out_dir, 'model.pt')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="submission/submission.npy")
    args = parser.parse_args()
    predict(load_config(args.config), args.ckpt, args.out)


if __name__ == "__main__":
    main()
