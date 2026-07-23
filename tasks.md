# Tasks — multimodal-vqa

Deadline: **2026-07-23 (Thu) 16:00 JST**. Target: VQA acc ~60% (completion line 49.9%).

## Status — COMPLETE (submitted 2026-07-23, deadline 16:00 JST)

- **Graded submission**: `configs/vit_bert_plain.yaml` — ViT-B/16 + BERT + cross-attention + soft,
  single LR, correct ViT normalization (0.5,0.5,0.5), 5 epochs. **Omnicampus test 0.57843**
  (val 0.5624) — clears the 0.499 completion line by ~8 pts. Both artifacts submitted (zip + PDF).
- **Best model produced (not submitted, missed deadline)**: `vit_bert_plain_long` (8 epochs),
  val **0.5705**, test 0.58955, unanswerable 0.694 / 182 unique (least majority collapse). Archived in
  `results/final/` (submission.npy, model.pt [git-ignored], executed_notebook.ipynb).
- Key late win: fixing ViT input normalization (was ImageNet, should be its own 0.5,0.5,0.5)
  lifted test 0.57396 → 0.57843 — from a Codex code review.
- Ablation (val): baseline 0.499(test) / R50+onehot+concat 0.5005 / R50+BERT+attn+soft 0.5391 /
  ViT+BERT+attn+soft 0.5534 / CLIP 0.5232 / +class-balanced 0.5317 (regressed).
- Confirmed dead ends (no re-try): CLIP backbone (underperformed ViT), class-balanced loss
  (regressed with soft labels — corroborated on course Slack).
- Post-deadline improvement candidates: OCR features (VizWiz has many in-image-text questions),
  open-vocab/generative head, ViT+r50 ensemble.
- Submission spec: `submission.npy` = np.array of answer **strings** (len 4969, `valid.json` order);
  data layout `data/{train,valid}.json` + `data/{train,valid}/`.

## Milestones

- [ ] **M0 — Baseline runs.** Reproduce the provided baseline end-to-end; record score & runtime.
- [ ] **M1 — Pretrained image encoder.** Swap scratch ResNet18 → pretrained ViT/ConvNeXt + fine-tune; add ImageNet norm.
- [ ] **M2 — Text encoder.** One-hot → embeddings + GRU/Transformer, or BERT (as component) + fine-tune.
- [ ] **M3 — Cross-modal attention fusion.** Replace concat with co-/cross-attention (report centerpiece).
- [ ] **M4 — Soft labels.** Use annotator answer distribution as soft targets.
- [ ] **M5 — Training recipe.** Aug + AdamW + cosine(warmup) + AMP + more epochs + top-N answer vocab.
- [ ] **M6 — Ablations.** concat vs. attention / scratch vs. pretrained / hard vs. soft → table.
- [ ] **M7 — Report PDF.** Novelty + ablation table (~1–2 pages, English OK).
- [ ] **M8 — Submit.** zip (preds + code + weights ≤4.5GB) + PDF to Omnicampus; submit early & iterate.
  - [ ] **Decide submission packaging**: bundle `src/` into the zip vs. a self-contained notebook.
        The submitted notebook must reproduce outputs top-to-bottom (TA audits high scorers), so a
        thin runner that clones a private repo won't reproduce for the grader. Pick one and make the
        zip standalone.
  - [ ] Keep the last upload ≥ baseline; save best weights/preds locally and re-submit the best
        before the deadline (only the LAST submission is graded).
  - [ ] Verify zip ≤4.5GB and `submission.npy` shape/format before uploading.

## Plumbing (src/) — DONE

- [x] `textutils.py` — `process_text` ported verbatim from the official baseline.
- [x] `dataset.py` — VizWiz loader, question/answer vocab (full or top-N + `<unk>`), onehot/token
      modes, ImageNet transform builder, soft-label targets.
- [x] `encoders.py` — image (resnet18/50 torchvision, vit/convnext timm) & text (onehot/gru/bert);
      unified `(features, mask)` `(B,T,D)` interface.
- [x] `model.py` — ConcatFusion + self-designed CrossAttentionFusion (question attends image regions).
- [x] `metrics.py` — VQA accuracy (string + index-batch, matches official `VQA_criterion`).
- [x] `train.py` — loop, AdamW/cosine(warmup)/AMP, held-out val VQA-acc, save best/last + `idx2answer`.
- [x] `predict.py` — inference + write `submission.npy` (answer strings) + `model.pt`.
- [x] Local CPU smoke test green: onehot+concat, gru+cross_attention, soft-label, metric consistency.

## Next actions (GPU)

- [ ] Download data (`data_download_VQA.ipynb`) -> `data.zip` (12GB) on Google Drive. **Run once.**
- [ ] Run 1 (safety net): `configs/resnet50_concat.yaml` -> verify val acc >= 0.499 -> submit early.
- [ ] Run 2 (improve): `configs/vit_bert_attn.yaml` -> target ~60%.
- [ ] Run 3 (excellence): cross-attention ablations (concat vs attn / scratch vs pretrained / hard vs soft).

## Experiment log

| date | config | val acc | notes |
| ---- | ------ | ------- | ----- |
| —    | —      | —       | —     |

## Notes / blockers

- (none yet)
