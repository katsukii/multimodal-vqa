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
