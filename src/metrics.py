"""VQA accuracy metric (VizWiz / VQA v2 convention).

acc(answer) = min(#humans that gave that answer / 3, 1), averaged over the 10
leave-one-out subsets. See https://visualqa.org/evaluation.html

Two entry points:
- `vqa_accuracy(pred, gt_answers)`: single sample, operates on normalized strings.
- `vqa_accuracy_batch(pred_idx, answer_idx)`: batched, operates on answer indices and
  matches the official baseline's `VQA_criterion` exactly (used inside the train loop).
"""
from __future__ import annotations

from typing import Sequence

from .textutils import process_text


def vqa_accuracy(pred: str, gt_answers: Sequence[str]) -> float:
    """VQA accuracy for one prediction against the 10 annotator answers (strings).

    Both `pred` and `gt_answers` are normalized with `process_text` before comparison,
    then the leave-one-out formula is averaged over the 10 held-out subsets.
    """
    pred_n = process_text(pred)
    gts = [process_text(a) for a in gt_answers]
    n = len(gts)

    total = 0.0
    for i in range(n):
        match = 0
        for j in range(n):
            if i == j:
                continue
            if pred_n == gts[j]:
                match += 1
        total += min(match / 3, 1)
    return total / n


def vqa_accuracy_batch(pred_idx, answer_idx) -> float:
    """Batched VQA accuracy over answer *indices* (matches official `VQA_criterion`).

    Parameters
    ----------
    pred_idx : Tensor (B,)
        Predicted answer index per sample (e.g. ``logits.argmax(1)``).
    answer_idx : Tensor (B, 10)
        The 10 annotator answers as vocabulary indices.

    Returns the mean VQA accuracy over the batch.
    """
    total = 0.0
    batch = len(pred_idx)
    for pred, answers in zip(pred_idx, answer_idx):
        pred = int(pred)
        answers = [int(a) for a in answers]
        acc = 0.0
        n = len(answers)
        for i in range(n):
            match = 0
            for j in range(n):
                if i == j:
                    continue
                if pred == answers[j]:
                    match += 1
            acc += min(match / 3, 1)
        total += acc / n
    return total / batch
