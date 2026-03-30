# 🩺 Deep Learning Breast Cancer Detection

A deep learning system for automated breast cancer detection via microcalcification segmentation in mammographic images, using a modified U-Net architecture with a custom HN Adam optimiser.

> **Final Year Project** | University of London  
> **Author:** Sora Park  
> **Source code:** [github.com/we1c0me2s0rapark/UoL-FP](https://github.com/we1c0me2s0rapark/UoL-FP)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Methodology](#methodology)
- [Evaluation](#evaluation)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Breast cancer remains a critical global health concern, accounting for approximately 670,000 deaths in 2022. This project addresses key pain points in current clinical workflows (radiologist fatigue, diagnostic inconsistency, and inter-observer variability) by developing an automated deep learning pipeline for microcalcification detection in mammograms.

The system performs **pixel-wise semantic segmentation** of regions of interest (ROI) in mammographic images sourced from the CBIS-DDSM dataset, providing a decision-support tool for medical professionals.

---

## Key Results

| Metric | Score |
|---|---|
| **Dice Coefficient** | 98.13% |
| **Jaccard Index** | 96.61% |
| **Precision** | 98% |
| **Recall** | 98% |
| **Accuracy** | 99.97% |
| **Loss** | 0.010941 |

> Best performance achieved with: **Custom HN Adam** optimiser + **Combined BCE + Dice loss** + **10-fold cross-validation**, 100 epochs, batch size 8.

---

## Architecture

The model uses a **modified U-Net** that maintains consistent spatial dimensions between input and output (unlike the original U-Net), improving lesion localisation and boundary delineation.

Key architectural features:
- Rectangular feature maps for handling varied mammographic aspect ratios
- Two additional convolutional layers in the decoder for enhanced feature learning
- Skip connections to propagate fine-grained spatial detail from encoder to decoder

### Custom HN Adam Optimiser

A bespoke optimiser extending the standard HN Adam (a hybrid of Adam + AMSGrad) with three enhancements for training stability:

1. **L2 norm clipping** - mitigates exploding gradients
2. **Dynamic scaling factor** - smooths learning-rate adjustments
3. **Weight update capping** - prevents abrupt parameter shifts

### Loss Function

A combined **Binary Cross-Entropy (BCE) + Dice loss** function balances pixel-wise accuracy with region-level overlap, addressing class imbalance inherent in medical image segmentation.

---

## Dataset

**CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography)

- High-quality, pre-processed, and annotated mammographic images
- Categorised into malignant and calcification classes
- Includes full mammograms, cropped ROI images, and ROI mask images

> ⚠️ **Note on data quality:** The original CBIS-DDSM CSV files contain incorrect image path references. This project includes a correction pipeline (`Test_DataQuality.ipynb` → `UpdatePaths.ipynb`) that resolves mislabelled ROI mask and cropped images by inspecting pixel value distributions (mask images contain only values 0 and 255; cropped images contain a wider range).

---

## Project Structure

```
UoL-FP/
│
├── Test_DataQuality.ipynb     # Verifies CBIS-DDSM dataset integrity; identifies path errors
├── UpdatePaths.ipynb          # Corrects image path references in CSV files
├── Test_OriginalData.ipynb    # Baseline CNN pipeline verification
├── Model.ipynb                # Main notebook: preprocessing, filtering, and U-Net training
├── Result.ipynb               # Loads trained model; visualises history and metrics
│
└── Components/
    ├── definitions.py         # Shared project constants
    ├── image_utils.py         # Image processing utility functions
    ├── unet_models.py         # Modified U-Net architecture definition
    ├── model_optimisers.py    # Custom HN Adam optimiser
    └── segmentation_utils.py  # Combined loss function, Dice coefficient, Jaccard index
```

---

## Installation and Setup

### Requirements

- Python 3.10.18
- TensorFlow 2.19.0
- CUDA 12.5 (recommended for GPU acceleration)
- WSL2 / Ubuntu 24.04 (development environment)

### Setup

```bash
# Clone the repository
git clone https://github.com/we1c0me2s0rapark/UoL-FP.git
cd UoL-FP

# Install dependencies
pip install -r requirements.txt
```

> GPU recommended. Tested on NVIDIA GeForce RTX 3070 Ti Laptop GPU with CUDA 12.5.

---

## Usage

Run the notebooks in the following order:

```
1. Test_DataQuality.ipynb   →   Identify dataset path errors
2. UpdatePaths.ipynb        →   Fix CSV path references
3. Test_OriginalData.ipynb  →   (Optional) Verify baseline pipeline
4. Model.ipynb              →   Preprocess data and train the modified U-Net
5. Result.ipynb             →   Evaluate and visualise results
```

---

## Methodology

### Preprocessing and Filtering

- **Gaussian filter** applied first to suppress noise
- **Laplacian filter** then applied to enhance edges and boundary features
- A weighting factor scales the Laplacian contribution for controlled enhancement
- Filtered image is combined with the original to preserve structural context

### Training Strategy

- **10-fold cross-validation** (outperformed 5-fold in all metrics)
- **100 epochs per fold**, batch size of 8
- Model checkpointing via Keras `ModelCheckpoint` (saves best epoch per fold)

### Validation Phases

| Phase | Target |
|---|---|
| Phase 1 | Achieve >90% Dice and Jaccard, regardless of epoch count |
| Phase 2 | Achieve or improve these scores within ≤75 epochs |

Both targets were surpassed, with the final model exceeding 98% on both metrics.

---

## Evaluation

### Optimiser Comparison (validation set, 100 epochs, no k-fold)

| Optimiser | Dice | Jaccard |
|---|---|---|
| Standard Adam | 36.31% | 23.47% |
| Standard HN Adam | 44.86% | 30.40% |
| **Custom HN Adam** | **45.32%** | **30.50%** |

### Cross-Validation Comparison

| Strategy | Optimiser | Dice | Jaccard |
|---|---|---|---|
| 5-fold | Custom HN Adam | 96.37% | 93.30% |
| **10-fold** | **Custom HN Adam** | **98.13%** | **96.61%** |

### Test Set Analysis

| Category | % of Test Set | Description |
|---|---|---|
| ✅ Perfect match | ~88.16% | 100% similarity and pixel accuracy |
| 🟡 Near-perfect match | 5.26% | >=99.99%, with only minor discrepancies |
| 🔴 Critical warning | 6.58% | <99.99%, with empty predictions or extra regions |

---

## Future Work

Implementation of a **modified Double U-Net** is proposed as the next development step. This architecture stacks two U-Net models, with the first using a pre-trained VGG-Net encoder. Prior work (Deb and Jha, 2022) reported strong results on CVC-ClinicDB (Dice: 93.12%, Jaccard: 86.13%), and integration with this project's custom HN Adam and combined loss function shows strong potential for further improvement in segmentation robustness.

---

## References

Key references underpinning this work:

- Ronneberger et al. (2015) - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28)
- Reyad, Sarhan and Arafa (2023) - [A Modified Adam Algorithm for Deep Neural Network Optimization](https://doi.org/10.1007/s00521-023-08568-z)
- Hossain (2019) - [Microcalcification Segmentation Using Modified U-Net](https://doi.org/10.1016/j.jksuci.2019.10.014)
- WHO (2024) - [Breast Cancer Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/breast-cancer)
- TCIA - [CBIS-DDSM Dataset](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

Full reference list available in the [project report](./report.pdf).

---

*Built with TensorFlow · CBIS-DDSM · Python 3.10*
