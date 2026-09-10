# Deepfake Detection using Swin Transformer and Celeb-DF-v2

A deep learning system for classifying facial images as **real** or **deepfake** using a fine-tuned **Swin Transformer Tiny** model on the **Celeb-DF-v2** dataset. Achieves **99.65% test accuracy** and **99.79% F1-score**.

---

## Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 99.65% |
| **F1-Score** | 99.79% |
| **ROC-AUC** | 0.9988 |
| **Average Precision** | 0.9991 |

### Confusion Matrix

<p align="center">
  <img src="04_test_metrics.png" width="90%">
</p>

### Normalized Confusion Matrix

<p align="center">
  <img src="05_confusion_matrix_normalized.png" width="45%">
</p>

### Training Curves

<p align="center">
  <img src="03_training_curves.png" width="85%">
</p>

> Best model checkpoint at **epoch 17** (green dashed line).

---

## Dataset

**Celeb-DF-v2** — a large-scale deepfake detection dataset containing real and synthesized celebrity face videos.

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Original | 17,729 | 111,919 | 129,648 |
| Train (after 2× real oversampling) | 24,820 | 78,343 | 103,163 |
| Validation | — | — | 19,447 |
| Test | 2,660 | 16,788 | 19,448 |

<p align="center">
  <img src="01_class_distribution.png" width="85%">
</p>

### Sample Images (after normalization + augmentation)

<p align="center">
  <img src="02_sample_images.png" width="85%">
</p>

---

## Model

**Swin Transformer Tiny** (`swin_tiny_patch4_window7_224`), pretrained on ImageNet, fine-tuned for binary classification (real vs. fake).

### Architecture

- **Backbone:** Swin Transformer Tiny (patch size 4, window size 7, input 224×224)
- **Classification head:** fine-tuned for 2 classes
- **Input:** face crops resized to 224×224 pixels

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate (head) | 1×10⁻⁴ (with cosine decay) |
| Epochs | 23 (best at epoch 17) |
| Batch size | 32 |
| Label smoothing | ✓ |
| Data augmentation | ✓ (random flip, color jitter, normalization) |
| Class balancing | 2× oversampling of real class |
| Regularization | Dropout, weight decay |

---

## Predictions

### Correct Predictions

<p align="center">
  <img src="06_correct_predictions.png" width="80%">
</p>

### Incorrect Predictions

<p align="center">
  <img src="07_wrong_predictions.png" width="80%">
</p>

> Most errors involve low-quality frames (occlusions, extreme angles, non-face crops) where even human judgment is difficult.

---

## Project Structure

```
├── training-celeb-df.ipynb      # Full training and evaluation notebook
├── 01_class_distribution.png    # Dataset class distribution
├── 02_sample_images.png         # Sample real/fake images
├── 03_training_curves.png       # Loss, accuracy, F1, LR curves
├── 04_test_metrics.png          # Confusion matrix, ROC, PR curves
├── 05_confusion_matrix_normalized.png
├── 06_correct_predictions.png   # Correctly classified examples
├── 07_wrong_predictions.png     # Misclassified examples
└── README.md
```

---

## Quick Start

### Requirements

```bash
pip install torch torchvision timm scikit-learn matplotlib tqdm pillow
```

### Run

Open and run `training-celeb-df.ipynb` in Jupyter or Kaggle. The notebook handles:
1. Dataset loading and preprocessing
2. Train/val/test splitting with class balancing
3. Model fine-tuning with early stopping
4. Full evaluation with confusion matrix, ROC-AUC, and PR curves

> **Note:** The Celeb-DF-v2 dataset must be downloaded separately. See the [official page](https://github.com/yuezunli/celeb-deepfakeforensics) for access.

---

## Key Findings

- **Swin Transformer** effectively captures both local facial texture artifacts and global structural inconsistencies in deepfakes.
- **2× oversampling** of the minority (real) class, combined with label smoothing, significantly improved balanced performance.
- The model achieves near-perfect detection (**AUC = 0.9988**) with only 69 misclassifications out of 19,448 test images.
- Most errors occur on heavily occluded or low-quality frames rather than on plausible deepfakes, suggesting strong generalization on clean face crops.

---

## Citation

If you use this work, please cite:

```
@misc{saqib2025deepfake,
  author = {Arif kamal},
  title  = {Deepfake Detection using Swin Transformer and Celeb-DF-v2},
  year   = {2025},
  url    = {https://github.com/arifabdali910/deepfake-swin-celeb-df}
}
```

---

## License

This project is for academic and research purposes.
