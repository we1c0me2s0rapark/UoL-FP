# breast-cancer-detection

A deep learning system for automated breast cancer detection via mammographic image segmentation, using a modified U-Net architecture and a custom HN Adam optimiser.

---

## Overview

This project develops and validates a deep learning-assisted X-ray mammography solution for detecting microcalcifications - small calcium deposits in breast tissue that are key early indicators of breast cancer. The model automates segmentation of regions of interest (ROI) in mammographic images, aiming to reduce radiologist workload, minimise inter-observer variability, and improve diagnostic consistency.

The best-performing model achieved:

| Metric | Score |
|---|---|
| Dice Coefficient | 98.13% |
| Jaccard Index | 96.61% |
| Precision | 98% |
| Recall | 98% |
| Accuracy | 99.97% |
| Loss | 0.010941 |

---

## Dataset

The project uses the **CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography), available via [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).

CBIS-DDSM was chosen for its high-quality, pre-processed, and well-annotated mammographic images, categorised into malignant and calcification classes. The raw dataset contains path labelling errors in the original CSV files, which are corrected as part of this pipeline.

---

## Architecture

A **modified U-Net** is used for pixel-wise segmentation. Key characteristics:

- Maintains consistent input and output spatial dimensions
- Rectangular feature maps for flexible resolution handling
- Two additional convolutional layers in the decoder for enhanced feature learning
- Trained with a **combined Binary Cross-Entropy (BCE) + Dice loss** function to address class imbalance

Image pre-processing applies a **Gaussian filter** (noise reduction) followed by a **Laplacian filter** (edge enhancement) to improve boundary delineation.

---

## Optimiser

A **custom HN Adam optimiser** was developed, extending the standard Adam + AMSGrad with three enhancements:

1. **L2 norm clipping**: mitigates exploding gradients
2. **Weight update capping**: prevents abrupt parameter shifts
3. **Dynamic scaling factor**: smooths learning-rate adjustments

This custom implementation outperformed both standard Adam and standard HN Adam on key segmentation metrics.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 100 per fold |
| Batch size | 8 |
| Cross-validation | 10-fold |
| Optimiser | Custom HN Adam |
| Loss function | Combined (BCE + Dice) |
| Framework | TensorFlow 2.19.0 |
| Python | 3.10.18 |
| Environment | WSL2 (Ubuntu 24.04) |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop (CUDA 12.5) |

---

## Repository Structure

```
breast-cancer-detection/
│
├── Test_DataQuality.ipynb       # Verifies CBIS-DDSM dataset integrity and identifies path errors
├── UpdatePaths.ipynb            # Corrects image path references in CSV files
├── Test_OriginalData.ipynb      # Baseline CNN pipeline verification
├── Model.ipynb                  # Main notebook: preprocessing, filtering, and U-Net training
├── Result.ipynb                 # Loads trained model and visualises evaluation metrics
│
└── Components/
    ├── definitions.py           # Shared constants and project-wide definitions
    ├── image_utils.py           # Image processing utility functions
    ├── unet_models.py           # Modified U-Net architecture definition
    ├── model_optimisers.py      # Custom HN Adam optimiser
    └── segmentation_utils.py    # Combined loss function and evaluation metrics (Dice, Jaccard)
```

---

## Evaluation Metrics

Performance is evaluated using the following metrics, all derived from the confusion matrix:

- **Dice Coefficient**: spatial overlap between predicted and ground truth masks
- **Jaccard Index**: stricter overlap metric (intersection over union)
- **Precision**: minimises false positives
- **Recall (Sensitivity)**: minimises false negatives
- **Accuracy**: overall proportion of correctly classified pixels
- **Histogram Similarity**: compares pixel intensity distributions
- **Pixel-wise Accuracy**: direct pixel-level segmentation quality

---

## Results Summary

Training was evaluated across loss functions, optimisers, and cross-validation strategies:

**Loss function comparison** (Adam, 100 epochs, no k-fold):

| Loss Function | Dice Coefficient | Jaccard Index |
|---|---|---|
| Dice loss | 4.58% | 2.36% |
| BCE | 36.26% | 23.28% |
| Combined | 36.31% | 23.47% |

**Optimiser comparison** (combined loss, 100 epochs, no k-fold):

| Optimiser | Dice Coefficient | Jaccard Index |
|---|---|---|
| Standard Adam | 36.31% | 23.47% |
| Standard HN Adam | 44.86% | 30.40% |
| Custom HN Adam | 45.32% | 30.50% |

**Cross-validation comparison** (custom HN Adam, combined loss, 100 epochs):

| Strategy | Dice Coefficient | Jaccard Index |
|---|---|---|
| 5-fold | 96.37% | 93.30% |
| 10-fold | 98.13% | 96.61% |

---

## Future Work

Implementation of a **modified double U-Net** is proposed as the next step, stacking two U-Net models with a pre-trained VGG-Net encoder in the first model. This approach has demonstrated strong segmentation performance on medical imaging datasets and is expected to further improve accuracy and robustness when combined with this project's custom HN Adam optimiser and combined loss function.

---

## References

Key references underpinning this work:

- Ronneberger et al. (2015), [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28)
- Reyad, Sarhan & Arafa (2023), [A Modified Adam Algorithm for Deep Neural Network Optimization](https://doi.org/10.1007/s00521-023-08568-z)
- Hossain (2019), [Microcalcification Segmentation Using Modified U-Net](https://doi.org/10.1016/j.jksuci.2019.10.014)
- The Cancer Imaging Archive, [CBIS-DDSM Dataset](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

Full reference list available in the project report.

---

## Author

**Sora Park**