# Final artifacts

## Graded submission (Omnicampus, completion secured)

- **Model**: `configs/vit_bert_plain.yaml` — ViT-B/16 + BERT + cross-attention + soft label,
  single LR, correct ViT normalization (0.5, 0.5, 0.5). 5 epochs.
- **Omnicampus test VQA accuracy: 0.57843** (held-out val 0.5624) — well above the 0.499
  completion line. This was the last submission before the 2026-07-23 16:00 JST deadline.

## Best model produced (this directory) — NOT submitted (missed deadline)

- **Model**: `configs/vit_bert_plain_long.yaml` — same as above but trained 8 epochs (the
  5-epoch run's validation was still rising, i.e. under-converged).
- **Held-out val 0.5705** (best at epoch 7). Qualitatively the best run: unanswerable share
  0.694 (vs 0.73–0.74 earlier) and 182 unique answers (vs ~89), i.e. least majority-class
  collapse. Estimated test ~0.58–0.59, but it finished just after the submission window closed.
- Files here:
  - `submission.npy` — predictions (4969 answer strings, valid.json order).
  - `model.pt` — weights (924MB, git-ignored via `*.pt`).
  - `executed_notebook.ipynb` — the run that produced these, with the per-epoch val log.

## Score progression (held-out val unless a test score is given)

| config | val | test |
| ------ | --- | ---- |
| official baseline (scratch R18 + one-hot + concat) | — | 0.499 |
| resnet50_concat (pretrained R50 + one-hot + concat) | 0.5005 | — |
| r50_bert_attn (R50 + BERT + cross-attn + soft) | 0.5391 | — |
| vit_bert_attn (ViT + BERT + cross-attn + soft, ImageNet norm) | 0.5534* | 0.57396 |
| vit_bert_plain (+ correct ViT norm, 5 ep) — **submitted** | 0.5624 | **0.57843** |
| vit_bert_plain_long (8 ep) — not submitted | **0.5705** | ~0.58–0.59 (est.) |

*augmented-val (before the validation-transform fix); later rows use the deterministic eval transform.

Dead ends (verified, not re-tried): CLIP backbone (0.5232, under-converged), class-balanced loss
(regressed with soft labels), differential learning rates (regressed 3×).
