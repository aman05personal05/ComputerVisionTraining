# Week 2 — Image Classification & Transfer Learning

| Attribute | Detail |
|-----------|--------|
| **Pillar** | Vision Intelligence Core |
| **Goal** | Move from training-from-scratch to fine-tuning pretrained backbones, and use data augmentation to improve performance on a custom dataset. |
| **Effort** | 32–40 hours (full-time) |

---

## Day-by-Day Plan

### Day 1: Modern CNN Architectures
- ResNet residual connections — why depth works
- EfficientNet compound scaling
- Load and inspect pretrained models from `torchvision`
- Benchmark inference speed; visualise filters & feature maps

**Materials:**
- [torchvision — Models & Pretrained Weights](https://pytorch.org/vision/stable/models.html)
- [ResNet paper (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385)

**Files:** [`day1_modern_architectures.py`](day1_modern_architectures/day1_modern_architectures.py) · [`notes.md`](day1_modern_architectures/notes.md)

**Run:**
```bash
cd day1_modern_architectures
python day1_modern_architectures.py
```

---

### Day 2: Transfer Learning Mechanics
- Feature extraction (freeze backbone) vs full fine-tuning (unfreeze + low LR)
- Replace the classifier head for your own classes
- Two-LR strategy to prevent catastrophic forgetting
- Side-by-side comparison on CIFAR-10

**Materials:**
- [PyTorch — Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [CS231n — Transfer Learning Notes](https://cs231n.github.io/transfer-learning/)

**Files:** [`day2_transfer_learning.py`](day2_transfer_learning/day2_transfer_learning.py) · [`notes.md`](day2_transfer_learning/notes.md)

**Run:**
```bash
cd day2_transfer_learning

# Quick 5-epoch smoke test (no W&B)
python day2_transfer_learning.py --epochs 5 --no-wandb

# Full 20-epoch comparison with W&B
python day2_transfer_learning.py --epochs 20

# Try EfficientNet-B0
python day2_transfer_learning.py --backbone efficientnet_b0 --epochs 15
```

---

### Day 3: Custom Dataset Pipeline
- Pick a custom dataset (Intel Image Classification — 6 classes, ~25k images)
- Build the full data pipeline: `Dataset`, `DataLoader`, transforms
- Analyse class distribution, compute dataset mean/std
- Visualise sample batches (sanity check before training)
- Plug into Day 2's transfer-learning code and compare both strategies

**Materials:**
- [PyTorch — Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

**Files:** [`day3_custom_dataset.py`](day3_custom_dataset/day3_custom_dataset.py) · [`notes.md`](day3_custom_dataset/notes.md)

**Run:**
```bash
cd day3_custom_dataset

# Explore dataset only (no training)
python day3_custom_dataset.py --explore-only

# Full run: explore + train both strategies (10 epochs)
python day3_custom_dataset.py --epochs 10 --no-wandb

# Try a different backbone
python day3_custom_dataset.py --backbone efficientnet_b0 --epochs 10
```

---

### Day 4: Data Augmentation with Albumentations
- Switch from torchvision transforms to **Albumentations** (faster, more flexible, detection-ready)
- Establish a **no-augmentation baseline** first
- Build 5 incremental pipelines: baseline → spatial → color → advanced → full
- Run a **controlled ablation study** — same backbone, same LR, same epochs, same split
- Visualise augmented samples (sanity check!)
- Build the augmentation **ablation table**

**Materials:**
- [Albumentations — Documentation](https://albumentations.ai/docs/)
- [Albumentations — PyTorch examples](https://albumentations.ai/docs/examples/pytorch_classification/)

**Files:** [`day4_augmentation.py`](day4_augmentation/day4_augmentation.py) · [`Week2_Day4_Colab.ipynb`](day4_augmentation/Week2_Day4_Colab.ipynb) · [`notes.md`](day4_augmentation/notes.md)

**Run:**
```bash
cd day4_augmentation

# Explore augmentations visually (no training)
python day4_augmentation.py --explore-only

# Full ablation (5 experiments × 10 epochs each)
python day4_augmentation.py --epochs 10 --no-wandb

# Try with a different backbone
python day4_augmentation.py --backbone efficientnet_b0 --epochs 10
```

---

### Day 5: Ablation Study & Final Results _(coming next)_
- Compare: baseline CNN vs feature extraction vs fine-tuned
- Confusion matrix + per-class metrics

---

## Folder Structure

```
week2_classification/
├── README.md                               ← This file
├── requirements.txt                        ← Week 2 specific deps
│
├── day1_modern_architectures/
│   ├── day1_modern_architectures.py        ← Architecture explorer
│   ├── Week2_Day1_Colab.ipynb              ← Colab notebook (Google Drive)
│   └── notes.md
│
├── day2_transfer_learning/
│   ├── day2_transfer_learning.py           ← Feature extract vs fine-tune
│   ├── Week2_Day2_Colab.ipynb              ← Colab notebook (Google Drive)
│   └── notes.md
│
├── day3_custom_dataset/
│   ├── day3_custom_dataset.py              ← Custom dataset pipeline + training
│   ├── Week2_Day3_Colab.ipynb              ← Colab notebook (Google Drive)
│   └── notes.md
│
├── day4_augmentation/
│   ├── day4_augmentation.py                ← Augmentation ablation study
│   ├── Week2_Day4_Colab.ipynb              ← Colab notebook (Google Drive)
│   └── notes.md
│
├── day5_ablation/                          ← Coming: Day 5
│
├── models/                                 ← Saved checkpoints
│   └── .gitkeep
│
└── results/                                ← Plots, JSON results
    └── .gitkeep
```

---

## Deliverable (End of Week)

A fine-tuned ResNet/EfficientNet on a custom dataset, with:
- [ ] Ablation table showing augmentation effect (positive or negative)
- [ ] Confusion matrix + per-class accuracy
- [ ] Short results write-up explaining what moved the metric most

## Success Gate (Self-Check)

- [ ] Fine-tuned model **beats from-scratch baseline** by a clear margin
- [ ] Ablation table shows **measurable augmentation effect** with correct interpretation
- [ ] Two-LR setup demonstrated and explained in notes

> ⚠️ **Watch out:** A frozen backbone needs a **normal LR** on the new head; full fine-tuning needs a **much smaller LR** on the backbone — or you'll destroy the pretrained weights.

---

## Results (fill in after completing the week)

| Strategy | Backbone | Dataset | Val Acc | Test Acc |
|----------|----------|---------|---------|----------|
| Feature Extraction | ResNet-18 | CIFAR-10 | __% | __% |
| Full Fine-Tuning | ResNet-18 | CIFAR-10 | __% | __% |
| Fine-Tuned + Aug | ResNet-18 | Custom | __% | __% |
