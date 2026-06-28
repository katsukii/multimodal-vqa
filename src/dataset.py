"""VizWiz VQA dataset.

Yields (image_tensor, question, target). With soft labels, `target` is a distribution
over the answer vocabulary built from the 10 annotator answers; otherwise the most
frequent answer index (hard label).
"""
from __future__ import annotations

from torch.utils.data import Dataset


class VizWizVQA(Dataset):
    def __init__(self, root: str, split: str, answer_vocab, transform=None, soft_label: bool = False):
        self.root = root
        self.split = split
        self.answer_vocab = answer_vocab
        self.transform = transform
        self.soft_label = soft_label
        raise NotImplementedError("TODO: load annotations + image paths for split")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def build_answer_vocab(annotations, vocab_size: int):
    """Build the top-N answer vocabulary from training annotations."""
    raise NotImplementedError("TODO: count answers, keep top-N (+ 'unanswerable')")
