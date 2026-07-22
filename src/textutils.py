"""Text normalization shared by the dataset and the metric.

`process_text` is ported verbatim from the official course baseline so that our
answer vocabulary and accuracy computation match the grader's normalization.
"""
from __future__ import annotations

import re


def process_text(text: str) -> str:
    """Normalize a question or answer string (lowercase, digits, drop articles, ...).

    Ported from the DL Basic 2026 Spring VQA baseline notebook. Keeping this identical
    to the official version is important: the answer vocabulary and the VQA accuracy are
    both defined over these normalized strings.
    """
    # lowercase
    text = text.lower()

    # convert number words to digits
    num_word_to_digit = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # remove periods that are not decimal points
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # remove articles
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # normalize a few contractions
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't",
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # punctuation -> space
    text = re.sub(r"[^\w\s':]", " ", text)

    # comma spacing
    text = re.sub(r"\s+,", ",", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
