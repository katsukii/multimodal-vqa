"""Generate tiny synthetic VizWiz-shaped data for a local CPU smoke test.

Writes data/smoke/{train,valid}.json and matching images so the full train -> predict
pipeline can be exercised without the real 12GB dataset. Not used for real training.
"""
from __future__ import annotations

import json
import os
import random

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "smoke")
QUESTIONS = ["what is this", "what color is it", "how many are there", "is it food", "where is it"]
ANSWERS = ["yes", "no", "red", "blue", "two", "unanswerable", "water", "dog"]
random.seed(0)


def make_split(split: str, n: int, with_answers: bool) -> None:
    img_dir = os.path.join(ROOT, split)
    os.makedirs(img_dir, exist_ok=True)
    records = []
    for i in range(n):
        name = f"{i}.jpg"
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        Image.new("RGB", (80, 80), color).save(os.path.join(img_dir, name))
        rec = {"image": name, "question": random.choice(QUESTIONS)}
        if with_answers:
            rec["answers"] = [{"answer": random.choice(ANSWERS)} for _ in range(10)]
        records.append(rec)
    with open(os.path.join(ROOT, f"{split}.json"), "w", encoding="utf-8") as f:
        json.dump(records, f)


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    make_split("train", 40, with_answers=True)
    make_split("valid", 12, with_answers=False)
    print(f"wrote synthetic smoke data under {ROOT}")
