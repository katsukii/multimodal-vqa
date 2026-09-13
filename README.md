# multimodal-vqa

Vision–Language **Visual Question Answering** built from scratch: image encoder + text
encoder + cross-modal fusion, trained end-to-end on [VizWiz](https://vizwiz.org/).

The goal is to study modern multimodal techniques by implementing them directly —
pretrained models are used **only as components** inside a custom model and trained
with a custom training loop (no off-the-shelf VQA models).

## Highlights

- **Config-driven** experiments — every ablation is a YAML file, so results are reproducible.
- **Modular encoders** — swap image (ResNet / ViT / ConvNeXt) and text (one-hot / GRU / BERT) backbones independently.
- **Cross-modal attention** fusion (co-attention / cross-attention) vs. naive concatenation.
- **Soft-label** training from the answer distribution of multiple annotators.

## Results

Trained on VizWiz (19,873 train / 4,969 test). The competition metric is VQA accuracy
(`min(#annotators agreeing / 3, 1)`). The official baseline is ResNet18 (scratch) + one-hot
question + concat fusion + hard labels.

| Model | Val VQA acc | Test (Omnicampus) |
|---|---|---|
| Official baseline (ResNet18 scratch, one-hot, concat, hard label) | — | 0.499 |
| ResNet50 (pretrained) + one-hot + concat | 0.5005 | — |
| ResNet50 + BERT + cross-attention + soft label | 0.5391 | — |
| **ViT-B/16 + BERT + cross-attention + soft label (graded submission)** | 0.5624 | **0.578** |
| Same, 8 epochs (finished after the deadline, not graded) | 0.5705 | 0.590 |

Findings from the ablations:

- Cross-modal attention with soft labels gave the largest single gain over concat + hard labels.
- Using ViT's own input normalization (0.5, 0.5, 0.5) instead of ImageNet statistics lifted the
  test score by ~0.5 pt.
- A class-balanced loss regressed when combined with soft labels; CLIP as the image backbone
  underperformed ViT-B/16 in this setup.

Full write-up: [`report/report.pdf`](report/report.pdf).

## Layout

```
src/         model, encoders, dataset, train / predict entry points
configs/     one YAML per experiment (ablation conditions)
notebooks/   thin Colab runner — clones this repo and calls src/
experiments/ committed metrics/logs (weights are NOT committed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train an experiment defined by a config
python -m src.train --config configs/baseline.yaml

# Generate predictions for submission
python -m src.predict --config configs/baseline.yaml --ckpt experiments/baseline/best.pth
```

On Colab, use `notebooks/colab_runner.ipynb`, which clones this repo and calls the same
entry points so the logic always lives in `src/`, never in the notebook.

## Data & weights

Dataset (~12 GB) and checkpoints (`*.pth`) are **not** tracked in git. Download the data
via the provider's script into `data/` (git-ignored). The submission zip (predictions +
code + weights) is assembled separately.

## Ablations (report targets)

| Axis | Compared |
|---|---|
| Fusion | concat vs. cross-modal attention |
| Image encoder | scratch vs. pretrained + fine-tune |
| Labels | hard label vs. soft label |

## Constraints

Pretrained models may be used **only as building blocks** of the custom model and must be
fine-tuned with the custom training loop on the provided training data. Inference-only use
of pretrained VQA models (e.g. BLIP, ViLT) is not allowed.
