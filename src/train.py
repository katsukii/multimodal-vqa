"""Training entry point. Usage: python -m src.train --config configs/baseline.yaml

Trains the custom `VQAModel` on the distributed training data. A slice of the training
split is held out as a local validation set so we can estimate the VQA accuracy (the test
answers are not distributed) and avoid wasting Omnicampus submissions. The best checkpoint
stores `idx2answer`, which `predict.py` needs to turn logits back into answer strings.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import load_config
from .dataset import PAD, UNK, VizWizVQA, build_image_transform, soft_target_from_answers
from .textutils import process_text
from .metrics import vqa_accuracy_batch
from .model import VQAModel


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_device(question, device):
    """Move a question (one-hot tensor, or (ids, mask) tuple) to the device."""
    if isinstance(question, (tuple, list)):
        return tuple(q.to(device) for q in question)
    return question.to(device)


def build_tokenizer(cfg):
    """A HF tokenizer is only needed for the BERT text encoder."""
    if cfg.model.text_encoder.type == "bert":
        from transformers import AutoTokenizer

        name = str(cfg.model.text_encoder.get("name", "bert-base-uncased"))
        return AutoTokenizer.from_pretrained(name)
    return None


def make_scheduler(optimizer, cfg, steps_per_epoch: int):
    total = steps_per_epoch * int(cfg.train.epochs)
    warmup = int(cfg.train.get("warmup_steps", 0))
    if cfg.train.get("scheduler", "none") != "cosine":
        return None

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, banned=None) -> float:
    model.eval()
    total, n = 0.0, 0
    for image, question, answers, _ in loader:
        image, question = image.to(device), to_device(question, device)
        logits = model(image, question)
        if banned:
            # Mirror inference: placeholder classes are never emitted, so the reported val
            # accuracy reflects the real (string-matched) test behaviour, not an inflated proxy.
            logits[:, banned] = float("-inf")
        total += vqa_accuracy_batch(logits.argmax(1).cpu(), answers) * image.size(0)
        n += image.size(0)
    return total / max(1, n)


def train(cfg) -> None:
    device = pick_device()
    torch.manual_seed(int(cfg.train.get("seed", 42)))
    text_mode = "onehot" if cfg.model.text_encoder.type == "onehot" else "tokens"
    pretrained = bool(cfg.model.image_encoder.get("pretrained", False))
    soft_label = bool(cfg.train.get("soft_label", False))

    tokenizer = build_tokenizer(cfg)
    kw = dict(root=cfg.data.root, split="train", answer=True, text_mode=text_mode,
              answer_vocab_size=cfg.data.get("answer_vocab_size"), tokenizer=tokenizer,
              max_qlen=int(cfg.data.get("max_qlen", 32)))
    # Build the model first so the image transform can use the backbone's own normalization
    # (e.g. ViT expects (0.5,0.5,0.5), not ImageNet stats). Vocab is built from the JSON only.
    full = VizWizVQA(transform=None, **kw)
    model = VQAModel(
        cfg, num_answers=full.num_answers,
        onehot_dim=full.onehot_dim if text_mode == "onehot" else None,
        word_vocab_size=full.word_vocab_size if cfg.model.text_encoder.type == "gru" else None,
    ).to(device)
    mean, std = model.image_encoder.norm_mean, model.image_encoder.norm_std

    # Separate transforms: augment only training; validation uses the deterministic eval
    # transform so val metrics and best-checkpoint selection match real inference.
    train_tf = build_image_transform(cfg.data.image_size, pretrained, train=True,
                                     augment=bool(cfg.data.get("augment", False)), mean=mean, std=std)
    eval_tf = build_image_transform(cfg.data.image_size, pretrained, train=False, mean=mean, std=std)
    full.transform = train_tf
    full_eval = VizWizVQA(transform=eval_tf, **kw)
    full_eval.update_dict(full)  # guarantee identical vocabulary

    val_fraction = float(cfg.data.get("val_fraction", 0.1))
    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(int(cfg.train.get("seed", 42)))
    perm = torch.randperm(len(full), generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_set = Subset(full, train_idx)          # augmented
    val_set = Subset(full_eval, val_idx)         # deterministic

    num_workers = int(cfg.data.get("num_workers", 2))
    train_loader = DataLoader(train_set, batch_size=int(cfg.train.batch_size), shuffle=True,
                              num_workers=num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_set, batch_size=int(cfg.train.batch_size), shuffle=False,
                            num_workers=num_workers, pin_memory=(device == "cuda"))

    opt_name = cfg.train.get("optimizer", "adam")
    optim_cls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    # Differential learning rates (course fine-tuning practice, lectures 9/10): pretrained
    # encoders at a low LR, the freshly-initialized fusion + classifier head at a higher LR so
    # they actually learn instead of the model defaulting to the majority answer.
    base_lr = float(cfg.train.lr)
    head_lr = float(cfg.train.get("head_lr", base_lr))
    encoder_params = list(model.image_encoder.parameters()) + list(model.text_encoder.parameters())
    head_params = list(model.fusion.parameters()) + list(model.classifier.parameters())
    optimizer = optim_cls(
        [{"params": encoder_params, "lr": base_lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    grad_clip = float(cfg.train.get("grad_clip", 0.0))
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    use_amp = bool(cfg.train.get("amp", False)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    banned = [i for i, a in full.idx2answer.items() if a in (UNK, PAD)]

    # Optional class-balanced weighting: down-weight frequent answers (e.g. "unanswerable")
    # so the model stops defaulting to the majority class (office-hour #2 recommendation).
    class_weight = None
    if bool(cfg.train.get("class_balanced", False)):
        counts = torch.ones(full.num_answers)  # +1 smoothing
        for answers in full.df["answers"]:
            for a in answers:
                counts[full.answer2idx.get(process_text(a["answer"]), full.unk_answer_idx)] += 1
        alpha = float(cfg.train.get("cb_alpha", 0.5))
        w = (counts.sum() / counts) ** alpha
        class_weight = (w / w.mean()).clamp(0.2, 10.0).to(device)

    def compute_loss(logits, answers, mode_answer):
        mode_answer = mode_answer.to(device)
        if soft_label:
            target = soft_target_from_answers(answers.to(device), full.num_answers)
            per = -(torch.log_softmax(logits, dim=1) * target).sum(dim=1)
        else:
            per = F.cross_entropy(logits, mode_answer, reduction="none")
        if class_weight is not None:
            sw = class_weight[mode_answer]
            return (per * sw).sum() / sw.sum()
        return per.mean()

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_acc, history = -1.0, []

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        running = 0.0
        for image, question, answers, mode_answer in train_loader:
            image, question = image.to(device), to_device(question, device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(image, question)
                loss = compute_loss(logits, answers, mode_answer)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()

        val_acc = evaluate(model, val_loader, device, banned=banned)
        train_loss = running / max(1, len(train_loader))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_vqa_acc": val_acc})
        print(f"[{epoch + 1}/{cfg.train.epochs}] train_loss={train_loss:.4f} val_vqa_acc={val_acc:.4f}")

        ckpt = {
            "model_state": model.state_dict(),
            "idx2answer": full.idx2answer,
            "question2idx": full.question2idx,
            "word2idx": full.word2idx,
            "config": dict(cfg),
            "val_vqa_acc": val_acc,
            "epoch": epoch + 1,
        }
        torch.save(ckpt, os.path.join(cfg.output_dir, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(cfg.output_dir, "best.pt"))
            # Optional off-runtime backup (e.g. Google Drive) so a mid-run disconnect on a
            # long job doesn't lose the best checkpoint. Never let a backup failure crash training.
            backup_dir = cfg.get("backup_dir")
            if backup_dir:
                try:
                    os.makedirs(backup_dir, exist_ok=True)
                    torch.save(ckpt, os.path.join(backup_dir, f"{cfg.name}_best.pt"))
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] backup to {backup_dir} failed: {e}")

    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val_vqa_acc": best_acc, "history": history}, f, indent=2)
    print(f"done. best_val_vqa_acc={best_acc:.4f}  (>= 0.499 target for completion)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
