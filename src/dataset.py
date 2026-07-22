"""VizWiz VQA dataset.

Data layout (matches the official course distribution):
    {root}/train.json, {root}/valid.json   (pandas.read_json; columns: image, question, answers)
    {root}/train/<image>, {root}/valid/<image>

`train.json` rows have `answers` = list of 10 dicts ({"answer": str, ...}); `valid.json`
(the held-out test set) has no answers. The question/answer vocabularies are built on the
training split and copied to val/test via `update_dict`.

Question representation is selectable:
- text_mode="onehot": a fixed (vocab+1,) multi-hot vector (baseline behaviour).
- text_mode="tokens": a padded (max_qlen,) LongTensor of token ids plus an attention mask,
  for the GRU / BERT encoders. Uses an internal word vocab for GRU, or a HF tokenizer for BERT.
"""
from __future__ import annotations

from collections import Counter
from statistics import mode

import numpy as np
import pandas
import torch
from PIL import Image
from torch.utils.data import Dataset

from .textutils import process_text

PAD, UNK = "<pad>", "<unk>"


class VizWizVQA(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        answer: bool = True,
        text_mode: str = "onehot",
        answer_vocab_size: int | None = None,
        tokenizer=None,
        max_qlen: int = 32,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.answer = answer
        self.text_mode = text_mode
        self.answer_vocab_size = answer_vocab_size
        self.tokenizer = tokenizer  # HF tokenizer when text_mode == "tokens" and encoder is BERT
        self.max_qlen = max_qlen

        self.image_dir = f"{root}/{split}"
        self.df = pandas.read_json(f"{root}/{split}.json")

        # Vocabs (built here for the training split; copied for val/test via update_dict).
        self.question2idx: dict[str, int] = {}
        self.answer2idx: dict[str, int] = {}
        self.idx2answer: dict[int, str] = {}
        # GRU word vocab (0 = pad, last = unk); only used for text_mode="tokens" without a tokenizer.
        self.word2idx: dict[str, int] = {PAD: 0}

        self._build_question_vocab()
        if self.answer:
            self._build_answer_vocab()

    # -- vocab construction -------------------------------------------------
    def _build_question_vocab(self) -> None:
        for question in self.df["question"]:
            for word in process_text(question).split(" "):
                if word and word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
                if word and word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
        self.word2idx[UNK] = len(self.word2idx)

    def _build_answer_vocab(self) -> None:
        counter: Counter[str] = Counter()
        for answers in self.df["answers"]:
            for a in answers:
                counter[process_text(a["answer"])] += 1

        if self.answer_vocab_size and self.answer_vocab_size < len(counter):
            kept = [w for w, _ in counter.most_common(self.answer_vocab_size)]
        else:
            kept = list(counter.keys())

        self.answer2idx = {w: i for i, w in enumerate(kept)}
        # Reserve a trailing index for out-of-vocab answers (top-N case).
        if UNK not in self.answer2idx:
            self.answer2idx[UNK] = len(self.answer2idx)
        self.idx2answer = {i: w for w, i in self.answer2idx.items()}

    def update_dict(self, train_dataset: "VizWizVQA") -> None:
        """Copy the training-split vocabularies onto a val/test dataset."""
        self.question2idx = train_dataset.question2idx
        self.answer2idx = train_dataset.answer2idx
        self.idx2answer = train_dataset.idx2answer
        self.word2idx = train_dataset.word2idx

    # -- sizes --------------------------------------------------------------
    @property
    def onehot_dim(self) -> int:
        return len(self.question2idx) + 1  # +1 for the unknown-word slot

    @property
    def num_answers(self) -> int:
        return len(self.answer2idx)

    @property
    def word_vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def unk_answer_idx(self) -> int:
        return self.answer2idx[UNK]

    # -- encoding helpers ---------------------------------------------------
    def _encode_question(self, question: str):
        words = process_text(question).split(" ")
        if self.text_mode == "onehot":
            vec = np.zeros(self.onehot_dim, dtype=np.float32)
            for w in words:
                vec[self.question2idx.get(w, self.onehot_dim - 1)] = 1.0
            return torch.from_numpy(vec)
        if self.text_mode == "tokens":
            if self.tokenizer is not None:  # BERT
                enc = self.tokenizer(
                    question,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_qlen,
                    return_tensors="pt",
                )
                return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
            # GRU: internal word vocab
            unk = self.word2idx[UNK]
            ids = [self.word2idx.get(w, unk) for w in words if w][: self.max_qlen]
            mask = [1] * len(ids)
            pad_n = self.max_qlen - len(ids)
            ids = ids + [self.word2idx[PAD]] * pad_n
            mask = mask + [0] * pad_n
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
        raise ValueError(f"unknown text_mode: {self.text_mode}")

    def _encode_answers(self, answers):
        unk = self.unk_answer_idx
        idxs = [self.answer2idx.get(process_text(a["answer"]), unk) for a in answers]
        return idxs

    # -- Dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        question = self._encode_question(self.df["question"][idx])

        if not self.answer:
            return image, question

        answer_idxs = self._encode_answers(self.df["answers"][idx])
        mode_answer_idx = mode(answer_idxs)
        return image, question, torch.tensor(answer_idxs, dtype=torch.long), int(mode_answer_idx)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4815, 0.4578, 0.4082)
CLIP_STD = (0.2686, 0.2613, 0.2758)


def norm_stats_for(encoder_type: str):
    """Normalization mean/std matching the pretrained backbone."""
    return (CLIP_MEAN, CLIP_STD) if encoder_type == "clip" else (IMAGENET_MEAN, IMAGENET_STD)


def build_image_transform(image_size: int, pretrained: bool, train: bool, augment: bool = False,
                          mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Build the torchvision image transform.

    Pretrained backbones expect their own normalization (ImageNet, or CLIP stats for CLIP);
    the scratch baseline just scales to [0, 1]. Augmentation is crop + color jitter only — no
    horizontal flip, which can flip the meaning of spatially/text-dependent VQA questions
    (office-hour caution: avoid augmentations that produce unrealistic examples).
    """
    from torchvision import transforms

    ops = []
    if train and augment:
        ops += [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2),
        ]
    else:
        ops += [transforms.Resize((image_size, image_size))]
    ops += [transforms.ToTensor()]
    if pretrained:
        ops += [transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def soft_target_from_answers(answer_idx: torch.Tensor, num_answers: int) -> torch.Tensor:
    """Build soft label targets from the 10 annotator answer indices.

    Uses the VQA-style score min(count/3, 1) per answer, normalized to a distribution.
    `answer_idx` is (B, 10); returns (B, num_answers).
    """
    b = answer_idx.size(0)
    target = torch.zeros(b, num_answers, device=answer_idx.device)
    for i in range(b):
        counts = Counter(int(a) for a in answer_idx[i])
        for ans, c in counts.items():
            target[i, ans] = min(c / 3, 1.0)
    total = target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return target / total
