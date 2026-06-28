"""VQA accuracy metric (VizWiz / VQA v2 convention).

acc(answer) = min(#humans that gave that answer / 3, 1), averaged over the 10
leave-one-out subsets. See https://visualqa.org/evaluation.html
"""
from __future__ import annotations


def vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
    raise NotImplementedError("TODO: implement min(count/3, 1) averaged over annotators")
