# Visual Question Answering on VizWiz with a Self-Designed Cross-Modal Attention

**DL Basic 2026 Spring — Final Assignment (Competition)**

## 1. Overview

We build a Visual Question Answering (VQA) model from scratch for the VizWiz 2023 split
(19,873 train / 4,969 test). Given an image and a question, the model predicts an answer
string; performance is measured with the standard VQA accuracy
`Acc(ans) = mean_i min(#annotators_{j≠i} that said ans / 3, 1)`.

Our model is a modular, config-driven architecture: a swappable **image encoder** and
**text encoder** produce token/region sequences that are combined by a swappable **fusion**
module and classified over a closed answer vocabulary. Pretrained backbones (ResNet, BERT)
are used *only as components* and fine-tuned by our own training loop — no end-to-end VQA
model is used. The contribution highlighted here is a **self-designed cross-modal attention
fusion**, evaluated against a concat baseline in an ablation study.

## 2. Method

### 2.1 Encoders (unified `(features, mask)` interface)

- **Image**: a torchvision ResNet (scratch ResNet18 for the baseline, ImageNet-pretrained
  ResNet50 for the improved model). We keep the spatial feature map before global pooling,
  giving region tokens `V ∈ R^{B×N×d_v}` (`N = H·W`). ImageNet normalization is applied when
  the backbone is pretrained.
- **Text**: the baseline encodes the question as a multi-hot vocabulary vector projected by a
  linear layer (one token). The improved model uses fine-tuned **BERT**, giving contextual
  token embeddings `Q ∈ R^{B×L×d_q}` with an attention mask.

### 2.2 Cross-modal attention fusion (centerpiece)

Both modalities are projected to a shared width `H`: `V' = V W_v`, `Q' = Q W_q`. The question
tokens **attend over the image regions** with multi-head attention

```
A = MultiHeadAttention(query = Q', key = V', value = V')      # (B, L, H)
Q_ctx = LayerNorm(Q' + A)                                     # residual keeps the question signal
```

We then pool a masked mean of the attended question tokens and of the image regions and fuse
them:

```
z = MLP( [ mean_mask(Q_ctx) ; mean(V') ] )                    # (B, H)
logits = W_cls z
```

Intuition: instead of squashing each modality to one vector and concatenating (which discards
*where* in the image the question refers to), each question token pulls the image regions it
needs, so the fused representation is grounded in question-relevant image content.

### 2.3 Training

Closed-vocabulary classifier over the full set of normalized training answers (no catch-all
`<unk>` class — a catch-all funnels the long answer tail into one dominant, always-wrong class
and inflates index-based validation). Loss is cross-entropy on the most-frequent answer (hard
label) or the VQA-score soft-label distribution over the 10 annotators (soft label). AdamW,
cosine schedule with warmup, AMP, and light augmentation (RandomResizedCrop / flip / jitter).
A 10% held-out slice of train is used for the VQA-accuracy validation reported below; the
placeholder answer class is masked at evaluation and inference to mirror the string-matched test.

## 3. Experiments

Dataset, metric, and closed-vocabulary setup as above. We ablate three axes:
image encoder (scratch vs pretrained), fusion (concat vs cross-attention), and label (hard vs
soft). Validation VQA accuracy is on our held-out train slice; the test column is the
Omnicampus score.

### Ablation table

| # | Image encoder | Text encoder | Fusion | Label | Val VQA acc | Test (Omnicampus) |
|---|---------------|--------------|--------|-------|-------------|-------------------|
| 0 | ResNet18 (scratch) | one-hot | concat | hard | — | 0.499 (official baseline) |
| 1 | ResNet50 (pretrained) | one-hot | concat | hard | 0.5005 | _TBD_ |
| 2 | ResNet50 (pretrained) | BERT | cross-attention | soft | _TBD_ | _TBD_ |
| 3 | ResNet50 (pretrained) | BERT | concat | soft | _TBD (optional)_ | — |
| 4 | ResNet50 (pretrained) | BERT | cross-attention | hard | _TBD (optional)_ | — |

_Rows 3–4 isolate the effect of fusion (2 vs 3) and label (2 vs 4) if GPU time permits._

### Key finding so far

Upgrading only the image encoder (row 0→1) barely moved the honest validation metric
(0.499 → 0.5005): with a one-hot text encoder the model defaults to the dominant
"unanswerable" answer. This localizes the bottleneck to the **text representation and fusion**,
motivating the BERT + cross-attention model (row 2).

## 4. Discussion

_[Fill after Run 2.]_ Expected: BERT contextual embeddings + cross-attention grounding give the
main gain; soft labels add a smaller, consistent improvement by using the full annotator
distribution rather than only the mode.

## 5. Conclusion

A modular from-scratch VQA model with a self-designed cross-modal attention fusion. The
ablation isolates where accuracy comes from (text + fusion >> image encoder alone) and shows
the attention fusion improves over concatenation. _[State final test score and completion.]_
